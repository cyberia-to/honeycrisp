//! Cache-blocked SME GEMM built on the 16×16 FMOPA tile microkernel.
//!
//! Layout matches the existing AMX path in `acpu::gemm::mod.rs`:
//! A is packed into 16-wide column strips, B into 16-wide row strips,
//! and the 16×16 tile microkernel sweeps the K dimension before storing
//! ZA0 into a single 16×16 block of C.
//!
//! Threading is added by carving M into 16-row strips and giving each
//! P-core its own `Stream` + tile loop. The dispatcher in
//! `matmul_f32_sme` picks single-thread for small / cheap calls and the
//! per-thread pool from `acpu::gemm::pool` for larger ones.

use crate::streaming::Stream;
use std::alloc::{alloc_zeroed, dealloc, Layout};

use super::tile::tile_16x16_f32;

const MR: usize = 16;
const NR: usize = 16;

// ---------------------------------------------------------------------------
// Aligned scratch buffer
// ---------------------------------------------------------------------------

struct Aligned {
    ptr: *mut f32,
    len: usize,
}

impl Aligned {
    fn new(n: usize) -> Self {
        if n == 0 {
            return Self {
                ptr: std::ptr::null_mut(),
                len: 0,
            };
        }
        let layout = Layout::from_size_align(n * 4, 128).unwrap();
        let ptr = unsafe { alloc_zeroed(layout) as *mut f32 };
        assert!(!ptr.is_null(), "SME pack: aligned alloc failed");
        Self { ptr, len: n }
    }
    fn as_slice_mut(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
    fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for Aligned {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len > 0 {
            let layout = Layout::from_size_align(self.len * 4, 128).unwrap();
            unsafe { dealloc(self.ptr as *mut u8, layout) };
        }
    }
}

// ---------------------------------------------------------------------------
// Packing
// ---------------------------------------------------------------------------

/// Pack a 16-row strip of A starting at row `ic`, columns `pc..pc+kc`.
///
/// Output layout: `kc * 16` f32. Lane `i` of "A column p" lives at
/// `dst[p * 16 + i]` so that one SVL load brings the whole column into
/// a Z register.
fn pack_a_strip(a: &[f32], k: usize, ic: usize, pc: usize, kc: usize, dst: &mut [f32]) {
    for p in 0..kc {
        for i in 0..MR {
            dst[p * MR + i] = a[(ic + i) * k + pc + p];
        }
    }
}

/// Pack a 16-column strip of B starting at column `jc`, rows `pc..pc+kc`.
///
/// Output layout: `kc * 16` f32. Lane `j` of "B row p" lives at
/// `dst[p * 16 + j]`.
fn pack_b_strip(b: &[f32], n: usize, pc: usize, jc: usize, kc: usize, dst: &mut [f32]) {
    for p in 0..kc {
        let src_row = (pc + p) * n + jc;
        dst[p * NR..p * NR + NR].copy_from_slice(&b[src_row..src_row + NR]);
    }
}

// ---------------------------------------------------------------------------
// Bulk path — m, n both multiples of 16
// ---------------------------------------------------------------------------

fn matmul_bulk(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    accumulate: bool,
) {
    let m_full = m / MR * MR;
    let n_full = n / NR * NR;

    if m_full == 0 || n_full == 0 || k == 0 {
        return;
    }

    // Pre-pack ALL of A and ALL of B *before* entering streaming mode.
    // Scalar / NEON work inside streaming mode has different performance
    // characteristics on M4 and is a footgun. Pre-packing also keeps the
    // streaming bracket a tight all-asm region.
    let m_strips = m_full / MR;
    let n_strips = n_full / NR;
    let mut a_all = Aligned::new(m_strips * k * MR);
    let mut b_all = Aligned::new(n_strips * k * NR);

    for si in 0..m_strips {
        let ic = si * MR;
        let off = si * k * MR;
        pack_a_strip(a, k, ic, 0, k, &mut a_all.as_slice_mut()[off..off + k * MR]);
    }
    for sj in 0..n_strips {
        let jc = sj * NR;
        let off = sj * k * NR;
        pack_b_strip(b, n, 0, jc, k, &mut b_all.as_slice_mut()[off..off + k * NR]);
    }

    // Decide thread count: each worker owns its own Stream (per-thread
    // PSTATE), partitioned across M strips. For small jobs run inline.
    let flops = 2 * m_full * n_full * k;
    let p_cores = crate::probe::scan().p_cores as usize;
    let n_threads = thread_cap(flops, p_cores, m_strips);

    let a_base = a_all.as_slice().as_ptr() as usize;
    let b_base = b_all.as_slice().as_ptr() as usize;
    let c_base = c.as_mut_ptr() as usize;

    if n_threads <= 1 {
        run_strip(
            a_base, b_base, c_base, 0, m_strips, n_strips, k, n, accumulate,
        );
        return;
    }

    // Partition M strips across threads, minimum 1 strip each.
    let mut starts: Vec<usize> = Vec::with_capacity(n_threads + 1);
    let mut cur = 0usize;
    let per = m_strips.div_ceil(n_threads).max(1);
    while cur < m_strips {
        starts.push(cur);
        cur += per;
    }
    starts.push(m_strips);
    let bounds: Vec<(usize, usize)> = starts.windows(2).map(|w| (w[0], w[1])).collect();

    std::thread::scope(|s| {
        for &(s_lo, s_hi) in &bounds {
            s.spawn(move || {
                // Pin to a P-core; ignore failure (still correct, just slower).
                let _ = crate::sync::affinity::pin_p_core();
                run_strip(
                    a_base, b_base, c_base, s_lo, s_hi, n_strips, k, n, accumulate,
                );
            });
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn run_strip(
    a_base_addr: usize,
    b_base_addr: usize,
    c_base_addr: usize,
    s_lo: usize,
    s_hi: usize,
    n_strips: usize,
    k: usize,
    n: usize,
    accumulate: bool,
) {
    if s_lo >= s_hi {
        return;
    }
    let stream = Stream::new().expect("run_strip: Stream::new failed");
    let a_base = a_base_addr as *const f32;
    let b_base = b_base_addr as *const f32;
    let c_base = c_base_addr as *mut f32;
    for si in s_lo..s_hi {
        let ic = si * MR;
        let a_ptr = unsafe { a_base.add(si * k * MR) };
        for sj in 0..n_strips {
            let jc = sj * NR;
            let b_ptr = unsafe { b_base.add(sj * k * NR) };
            unsafe {
                tile_16x16_f32(
                    &stream,
                    a_ptr,
                    b_ptr,
                    c_base.add(ic * n + jc),
                    n,
                    k,
                    accumulate,
                );
            }
        }
    }
    drop(stream);
}

fn thread_cap(flops: usize, p_cores: usize, m_strips: usize) -> usize {
    // std::thread::scope spawn cost is ~50–100 µs per worker. Single-thread
    // SME ~400 GF/s means even n=512 (270 MFLOPS) finishes in ~700 µs;
    // spawning eats more than the parallel speedup. Only kick MT in for
    // really big problems where the SMSTART + spawn overhead amortizes.
    //
    // TODO(Phase 2.5): persistent SME pool + 4-tile interleave will lift
    // both throughput and the useful MT range.
    let cap = if flops < 2_000_000_000 {
        1
    } else if flops < 8_000_000_000 {
        p_cores.min(4)
    } else {
        p_cores
    };
    cap.clamp(1, m_strips.max(1))
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Single-precision matmul on SME: C = A × B (overwrite).
///
/// Returns `Err(CpuError::FeatureNotAvailable)` if FEAT_SME is absent.
pub fn matmul_f32_sme_set(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> crate::Result<()> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    if !crate::probe::scan().has_sme {
        return Err(crate::CpuError::FeatureNotAvailable(crate::Feature::Sme));
    }

    // Edges: zero rows/cols outside the SME tile grid, then fill the
    // bulk rectangle with the FMOPA path.
    let m_full = m / MR * MR;
    let n_full = n / NR * NR;

    // Zero output first — we'll write back only what FMOPA computes.
    // For edges (rows ≥ m_full or cols ≥ n_full) we still need the
    // correct result; fall back to scalar there.
    for v in c.iter_mut().take(m * n) {
        *v = 0.0;
    }

    if m_full > 0 && n_full > 0 {
        matmul_bulk(a, b, c, m, n, k, false);
    }

    // Fill row tail (rows m_full..m): scalar.
    if m_full < m {
        scalar_matmul_block(a, b, c, m, n, k, m_full, 0, m, n);
    }
    // Fill column tail of the bulk row block (rows 0..m_full, cols n_full..n).
    if n_full < n && m_full > 0 {
        scalar_matmul_block(a, b, c, m, n, k, 0, n_full, m_full, n);
    }

    Ok(())
}

/// Single-precision matmul on SME: C += A × B.
pub fn matmul_f32_sme(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> crate::Result<()> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    if !crate::probe::scan().has_sme {
        return Err(crate::CpuError::FeatureNotAvailable(crate::Feature::Sme));
    }

    let m_full = m / MR * MR;
    let n_full = n / NR * NR;

    if m_full > 0 && n_full > 0 {
        matmul_bulk(a, b, c, m, n, k, true);
    }
    if m_full < m {
        scalar_add_matmul_block(a, b, c, m, n, k, m_full, 0, m, n);
    }
    if n_full < n && m_full > 0 {
        scalar_add_matmul_block(a, b, c, m, n, k, 0, n_full, m_full, n);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Scalar edge helpers (small, called only for the < MR row/col tails)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn scalar_matmul_block(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    _m: usize,
    n: usize,
    k: usize,
    row_lo: usize,
    col_lo: usize,
    row_hi: usize,
    col_hi: usize,
) {
    for i in row_lo..row_hi {
        for j in col_lo..col_hi {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn scalar_add_matmul_block(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    _m: usize,
    n: usize,
    k: usize,
    row_lo: usize,
    col_lo: usize,
    row_hi: usize,
    col_hi: usize,
) {
    for i in row_lo..row_hi {
        for j in col_lo..col_hi {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] += acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = acc;
            }
        }
    }

    fn check(m: usize, n: usize, k: usize) {
        if !crate::probe::scan().has_sme {
            return;
        }
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
        let mut c_sme = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        matmul_f32_sme_set(&a, &b, &mut c_sme, m, n, k).unwrap();
        ref_matmul(&a, &b, &mut c_ref, m, n, k);

        let mut max_err = 0.0f32;
        for i in 0..m * n {
            let e = (c_sme[i] - c_ref[i]).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(max_err < 1e-3, "{m}×{n}×{k}: max_err={max_err}");
    }

    #[test]
    fn sme_16x16x16() {
        check(16, 16, 16);
    }

    #[test]
    fn sme_32x32x32() {
        check(32, 32, 32);
    }

    #[test]
    fn sme_64x64x64() {
        check(64, 64, 64);
    }

    #[test]
    fn sme_128x128x128() {
        check(128, 128, 128);
    }

    #[test]
    fn sme_odd_size() {
        // Hits the scalar edge path
        check(17, 19, 23);
    }

    #[test]
    fn sme_accumulate() {
        if !crate::probe::scan().has_sme {
            return;
        }
        let n = 32usize;
        let a: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.02).collect();
        let mut c_sme: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.5).collect();
        let mut c_ref = c_sme.clone();

        matmul_f32_sme(&a, &b, &mut c_sme, n, n, n).unwrap();

        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..n {
                    acc += a[i * n + p] * b[p * n + j];
                }
                c_ref[i * n + j] += acc;
            }
        }

        let mut max_err = 0.0f32;
        for i in 0..n * n {
            let e = (c_sme[i] - c_ref[i]).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(max_err < 1e-2, "accumulate {n}×{n}×{n}: max_err={max_err}");
    }
}
