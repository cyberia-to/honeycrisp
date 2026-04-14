//! Auto-dispatch GEMM (General Matrix Multiply) routines.
//!
//! All functions compute C += A × B in row-major order:
//!   A is m×k, B is k×n, C is m×n.
//!
//! `matmul_f32` uses AMX 16×16 microkernel with GEBP cache blocking,
//! parallel across P-cores, falling back to scalar.

pub mod gemv_amx_asm;
pub mod gemv_kern;
pub mod matvec;
mod other;
mod pool;
mod small;

pub use other::{matmul_bf16, matmul_f16, matmul_i8};

use crate::matrix;

use std::cell::RefCell;

// ---------------------------------------------------------------------------
// GEBP blocking parameters (tuned for Apple Silicon L1/L2)
// ---------------------------------------------------------------------------

const MR: usize = 16;
const NR: usize = 16;
pub(super) const MC: usize = 64;

// ---------------------------------------------------------------------------
// Thread-local pack buffer cache — keep buffers warm across matmul_f32 calls
// ---------------------------------------------------------------------------

thread_local! {
    static PACK_CACHE: RefCell<PackCache> = const { RefCell::new(PackCache { a: None, b: None, a_src: 0, a_dims: (0,0,0) }) };
    /// Persistent AMX context — set once per thread, never cleared.
    /// Saves 40ns per matmul_f32 call (AMX_SET + AMX_CLR overhead).
    static AMX_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct PackCache {
    a: Option<AlignedBuf>,
    b: Option<AlignedBuf>,
    /// Cached A source pointer — skip repacking if unchanged.
    a_src: usize,
    a_dims: (usize, usize, usize), // (m, k, row_start)
}

/// Ensure AMX is active on this thread. First call does AMX_SET,
/// subsequent calls are a no-op (Cell::get = single load).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn ensure_amx() {
    AMX_ACTIVE.with(|active| {
        if !active.get() {
            unsafe { matrix::asm::amx_set() };
            active.set(true);
        }
    });
}

/// Get or grow a cached pack buffer. Returns a warm buffer (in L1/L2).
fn cached_buf(slot: &mut Option<AlignedBuf>, needed: usize) -> AlignedBuf {
    if let Some(buf) = slot.take() {
        if buf.len >= needed {
            return buf;
        }
    }
    AlignedBuf::new(needed)
}

// ---------------------------------------------------------------------------
// Aligned allocation
// ---------------------------------------------------------------------------

pub(super) struct AlignedBuf {
    pub(super) ptr: *mut f32,
    pub(super) len: usize,
}

impl AlignedBuf {
    fn new(n: usize) -> Self {
        if n == 0 {
            return Self {
                ptr: std::ptr::null_mut(),
                len: 0,
            };
        }
        let size = n * 4;
        let layout = std::alloc::Layout::from_size_align(size, 128).unwrap();
        // Small buffers: skip zero-fill. Packing writes before reading.
        // Large buffers: zero-fill to pre-fault mmap'd pages.
        let ptr = if size <= 128 * 1024 {
            unsafe { std::alloc::alloc(layout) as *mut f32 }
        } else {
            unsafe { std::alloc::alloc_zeroed(layout) as *mut f32 }
        };
        assert!(!ptr.is_null(), "aligned allocation failed");
        Self { ptr, len: n }
    }

    fn as_mut_slice(&mut self) -> &mut [f32] {
        if self.len == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    fn as_slice(&self) -> &[f32] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len > 0 {
            let layout = std::alloc::Layout::from_size_align(self.len * 4, 64).unwrap();
            unsafe { std::alloc::dealloc(self.ptr as *mut u8, layout) };
        }
    }
}

// ---------------------------------------------------------------------------
// matmul_f32 — f32 matmul
// ---------------------------------------------------------------------------

/// Single-precision matrix multiply: C = A × B (overwrite).
///
/// Row-major: A[m×k], B[k×n], C[m×n]. Previous C values are discarded.
/// Faster than `matmul_f32` — skips preloading C into AMX registers.
pub fn matmul_f32_set(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k, "a.len() must equal m*k");
    assert_eq!(b.len(), k * n, "b.len() must equal k*n");
    assert_eq!(c.len(), m * n, "c.len() must equal m*n");

    #[cfg(target_arch = "aarch64")]
    {
        // Dedicated matvec for m ≤ 4 (single-token inference hot path)
        if m <= 4 && n >= 32 && k >= 32 {
            for row in 0..m {
                let a_row = &a[row * k..(row + 1) * k];
                let c_row = &mut c[row * n..(row + 1) * n];
                matvec::matvec_f32_set(a_row, b, c_row, n, k);
            }
            return;
        }

        let flops = 2 * m * n * k;
        if m < 2 * MR || n < 2 * NR {
            c.fill(0.0);
            matmul_f32_neon(a, b, c, m, n, k);
            return;
        }
        let p_cores = crate::probe::scan().p_cores as usize;
        let max_threads = if m >= MR { m / MR } else { 1 };
        let n_threads = thread_cap(flops, p_cores, max_threads);
        if n_threads > 1 && flops > 20_000_000 {
            matmul_f32_parallel(a, b, c, m, n, k, n_threads);
        } else if n * k <= 131072 {
            small::matmul_f32_amx_direct(a, b, c, m, n, k, true);
        } else {
            matmul_f32_amx_single(a, b, c, m, n, k);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        c.fill(0.0);
        matmul_f32_scalar(a, b, c, m, n, k);
    }
}

/// Single-precision matrix multiply: C += A × B.
///
/// Row-major: A[m×k], B[k×n], C[m×n].
/// Parallelizes across M dimension using P-core threads when beneficial.
pub fn matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k, "a.len() must equal m*k");
    assert_eq!(b.len(), k * n, "b.len() must equal k*n");
    assert_eq!(c.len(), m * n, "c.len() must equal m*n");

    #[cfg(target_arch = "aarch64")]
    {
        let flops = 2 * m * n * k;
        // NEON 4×16 microkernel beats AMX for sizes < 32 (packing overhead
        // dominates). AMX pair32 wins at 32+ with full tile utilization.
        if m < 2 * MR || n < 2 * NR {
            matmul_f32_neon(a, b, c, m, n, k);
            return;
        }
        // Direct AMX (no B packing) when B fits in L2.
        let p_cores = crate::probe::scan().p_cores as usize;
        let max_threads = if m >= MR { m / MR } else { 1 };
        let n_threads = thread_cap(flops, p_cores, max_threads);
        if n_threads > 1 && flops > 20_000_000 {
            matmul_f32_parallel(a, b, c, m, n, k, n_threads);
        } else if n * k <= 131072 {
            small::matmul_f32_amx_direct(a, b, c, m, n, k, false);
        } else {
            matmul_f32_amx_single(a, b, c, m, n, k);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        matmul_f32_scalar(a, b, c, m, n, k);
    }
}

/// Choose thread count based on FLOP count and available P-cores.
#[cfg(target_arch = "aarch64")]
fn thread_cap(flops: usize, p_cores: usize, max_threads: usize) -> usize {
    let cap = if flops < 20_000_000 {
        1
    } else if flops < 100_000_000 {
        2
    } else if flops < 500_000_000 {
        p_cores.min(4)
    } else {
        p_cores
    };
    cap.max(1).min(max_threads)
}

/// Parallel matmul_f32: dispatches M-strips to thread pool.
/// Small B (≤1MB): workers use direct AMX on raw B.
/// Large B (>1MB): main thread pre-packs B once, workers read shared packed B.
#[cfg(target_arch = "aarch64")]
fn matmul_f32_parallel(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    n_threads: usize,
) {
    let pool = pool::get_pool(n_threads);
    let base_rows = (m / n_threads / MR) * MR;
    let rows_per_thread = if base_rows == 0 { MR } else { base_rows };

    let a_addr = a.as_ptr() as usize;
    let c_addr = c.as_mut_ptr() as usize;

    // Direct AMX when B fits in L2 per-core (~4MB on M1 P-core).
    // Each worker gets its own B read from L2; avoids main-thread packing stall.
    let b_bytes = n * k * 4;
    if b_bytes <= 4 * 1024 * 1024 {
        let b_addr = b.as_ptr() as usize;
        let mut jobs: Vec<Box<dyn FnOnce() + Send>> = Vec::with_capacity(n_threads);
        let mut m_start = 0usize;
        while m_start < m {
            let m_this = if m - m_start <= rows_per_thread + MR {
                m - m_start
            } else {
                rows_per_thread
            };
            let a_off = m_start * k;
            let c_off = m_start * n;
            let m_local = m_this;
            jobs.push(Box::new(move || {
                let a_slice = unsafe {
                    std::slice::from_raw_parts((a_addr + a_off * 4) as *const f32, m_local * k)
                };
                let c_chunk = unsafe {
                    std::slice::from_raw_parts_mut((c_addr + c_off * 4) as *mut f32, m_local * n)
                };
                let b_slice = unsafe { std::slice::from_raw_parts(b_addr as *const f32, k * n) };
                small::matmul_f32_amx_direct(a_slice, b_slice, c_chunk, m_local, n, k, true);
            }));
            m_start += m_this;
        }
        while jobs.len() < n_threads {
            jobs.push(Box::new(|| {}));
        }
        pool.run(jobs);
        return;
    }

    // Large B: pre-pack B on main thread, workers share read-only packed B.
    let kc_max = k.min(512);
    let nc_max = n.min(512);
    let b_block = kc_max * nc_max.div_ceil(32) * 32;
    let k_blocks = k.div_ceil(kc_max);
    let n_blocks = n.div_ceil(nc_max);

    // Main thread packs all of B into contiguous 32-wide strips.
    let mut b_all = AlignedBuf::new(k_blocks * n_blocks * b_block);
    {
        let mut pc = 0;
        for pi in 0..k_blocks {
            let kc = (k - pc).min(kc_max);
            let mut jc = 0;
            for ji in 0..n_blocks {
                let nc = (n - jc).min(nc_max);
                let off = (pi * n_blocks + ji) * b_block;
                pack_b_32(b, n, pc, jc, kc, nc, &mut b_all.as_mut_slice()[off..]);
                jc += nc;
            }
            pc += kc;
        }
    }

    let b_addr = b_all.as_slice().as_ptr() as usize;
    let b_total = k_blocks * n_blocks * b_block;

    let mut jobs: Vec<Box<dyn FnOnce() + Send>> = Vec::with_capacity(n_threads);
    let mut m_start = 0usize;
    while m_start < m {
        let m_this = if m - m_start <= rows_per_thread + MR {
            m - m_start
        } else {
            rows_per_thread
        };
        let a_off = m_start * k;
        let c_off = m_start * n;
        let m_local = m_this;
        jobs.push(Box::new(move || {
            let a_slice = unsafe {
                std::slice::from_raw_parts((a_addr + a_off * 4) as *const f32, m_local * k)
            };
            let c_chunk = unsafe {
                std::slice::from_raw_parts_mut((c_addr + c_off * 4) as *mut f32, m_local * n)
            };
            let b_packed = unsafe { std::slice::from_raw_parts(b_addr as *const f32, b_total) };

            gebp_worker_cached_a(
                a_slice, b_packed, c_chunk, m_local, n, k, kc_max, nc_max, k_blocks, n_blocks,
                b_block,
            );
        }));
        m_start += m_this;
    }
    while jobs.len() < n_threads {
        jobs.push(Box::new(|| {}));
    }
    pool.run(jobs);
    // b_all lives until pool.run completes, then drops here.
}

/// GEBP worker with A-cache: packs A interleaved for full K (all MC blocks upfront),
/// then iterates pc -> ic -> jc (classic GEBP order for B-cache reuse).
/// A is packed once and reused across all KC blocks; on repeated calls with same A,
/// packing is skipped entirely.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn gebp_worker_cached_a(
    a: &[f32],
    b_packed: &[f32],
    c: &mut [f32],
    m_local: usize,
    n: usize,
    k: usize,
    kc_max: usize,
    nc_max: usize,
    k_blocks: usize,
    n_blocks: usize,
    b_block: usize,
) {
    use crate::matrix::tile;

    ensure_amx();

    // MC blocking: cap so each MC block's A pack fits in L2.
    let mc_max = if m_local > 128 { 256 } else { m_local };

    // Precompute per-MC-block layout.
    struct McBlock {
        ic: usize,
        mc: usize,
        n_pairs: usize,
        has_odd: bool,
        rem_rows: usize,
        a_base: usize,
        pair_end: usize,
        odd_end: usize,
    }
    let mut blocks: Vec<McBlock> = Vec::new();
    let mut a_total = 0usize;
    {
        let mut ic = 0usize;
        while ic < m_local {
            let mc = (m_local - ic).min(mc_max);
            let n_full = mc / MR;
            let n_pairs = n_full / 2;
            let has_odd = n_full % 2 == 1;
            let rem_rows = mc % MR;
            let pair_end = a_total + n_pairs * k * 32;
            let odd_end = pair_end + if has_odd { k * MR } else { 0 };
            let block_end = odd_end + if rem_rows > 0 { k * MR } else { 0 };
            blocks.push(McBlock {
                ic,
                mc,
                n_pairs,
                has_odd,
                rem_rows,
                a_base: a_total,
                pair_end,
                odd_end,
            });
            a_total = block_end;
            ic += mc;
        }
    }

    // If full-K A pack exceeds 32MB, fall back to per-KC-block GEBP.
    // Below 32MB: A-cache benefit outweighs the larger allocation on repeated calls.
    if a_total * 4 > 32 * 1024 * 1024 {
        gebp_worker_fallback(
            a, b_packed, c, m_local, n, k, kc_max, nc_max, k_blocks, n_blocks, b_block, mc_max,
        );
        return;
    }

    let a_src_id = a.as_ptr() as usize;
    let a_dims = (m_local, k, 0usize);

    let (mut a_pack, skip_pack) = PACK_CACHE.with(|cache_cell| {
        let mut cache = cache_cell.borrow_mut();
        let hit = cache.a_src == a_src_id && cache.a_dims == a_dims && cache.a.is_some();
        let buf = cached_buf(&mut cache.a, a_total);
        let skip = hit && buf.len >= a_total;
        (buf, skip)
    });

    // Pack all A upfront (or skip if cached).
    if !skip_pack {
        let a_buf = a_pack.as_mut_slice();
        for blk in &blocks {
            for pair in 0..blk.n_pairs {
                let off = blk.a_base + pair * k * 32;
                small::pack_a_interleaved_neon(a, k, blk.ic + pair * 2 * MR, k, &mut a_buf[off..]);
            }
            if blk.has_odd {
                let row_start = blk.ic + blk.n_pairs * 2 * MR;
                pack_a_strip_neon(a, k, row_start, 0, k, &mut a_buf[blk.pair_end..]);
            }
            if blk.rem_rows > 0 {
                let off = blk.odd_end;
                let row_start = blk.ic + (blk.mc / MR) * MR;
                for p in 0..k {
                    for i in 0..MR {
                        a_buf[off + p * MR + i] = 0.0;
                    }
                }
                for i in 0..blk.rem_rows {
                    let a_row = (row_start + i) * k;
                    for p in 0..k {
                        a_buf[off + p * MR + i] = a[a_row + p];
                    }
                }
            }
        }
    }

    // Loop order: pc -> ic -> jc (classic GEBP: B stays warm across MC blocks).
    unsafe {
        let a_buf = a_pack.as_slice();
        let mut pc = 0usize;
        for pi in 0..k_blocks {
            let kc = (k - pc).min(kc_max);
            let is_first = pc == 0;

            for blk in &blocks {
                let mut jc = 0usize;
                for ji in 0..n_blocks {
                    let nc = (n - jc).min(nc_max);
                    let b_base = b_packed.as_ptr().add((pi * n_blocks + ji) * b_block);
                    let n_b32 = nc / 32;
                    let b_rem = nc % 32;

                    // --- Pairs: 32x32 kernels ---
                    for pair in 0..blk.n_pairs {
                        let ir = blk.ic + pair * 2 * MR;
                        let ap =
                            a_buf.as_ptr().add(blk.a_base + pair * k * 32 + pc * 32) as *const u8;

                        for js in 0..n_b32 {
                            let bp = b_base.add(js * kc * 32) as *const u8;
                            let c0 = c.as_mut_ptr().add(ir * n + jc + js * 32);
                            if is_first {
                                crate::matrix::kern::kern_32x32_first(ap, bp, kc, 128);
                            } else {
                                tile::preload_c(c0, n, 0);
                                tile::preload_c(c0.add(NR), n, 1);
                                tile::preload_c(c0.add(MR * n), n, 2);
                                tile::preload_c(c0.add(MR * n + NR), n, 3);
                                tile::microkernel_32x32_acc_nopairx(ap, bp, kc, 128);
                            }
                            tile::store_c(c0, n, 0);
                            tile::store_c(c0.add(NR), n, 1);
                            tile::store_c(c0.add(MR * n), n, 2);
                            tile::store_c(c0.add(MR * n + NR), n, 3);
                        }

                        if b_rem > 0 {
                            let b_off = (pi * n_blocks + ji) * b_block + n_b32 * kc * 32;
                            for s in 0..2usize {
                                edge_kernel(
                                    a_buf,
                                    b_packed,
                                    c,
                                    n,
                                    ir + s * MR,
                                    jc + n_b32 * 32,
                                    blk.a_base + pair * k * 32 + pc * 32,
                                    32,
                                    s * MR,
                                    b_off,
                                    32,
                                    MR,
                                    b_rem,
                                    kc,
                                );
                            }
                        }
                    }

                    // --- Odd strip: 16x16 kernels ---
                    if blk.has_odd {
                        let ir = blk.ic + blk.n_pairs * 2 * MR;
                        let a_odd_off = blk.pair_end + pc * MR;
                        let ap = a_buf.as_ptr().add(a_odd_off) as *const u8;

                        for js in 0..n_b32 {
                            let bp = b_base.add(js * kc * 32) as *const u8;
                            let c0 = c.as_mut_ptr().add(ir * n + jc + js * 32);
                            if is_first {
                                tile::microkernel_16x16_first(ap, bp, kc, 128);
                            } else {
                                tile::preload_c(c0, n, 0);
                                tile::microkernel_16x16_acc(ap, bp, kc, 128);
                            }
                            tile::store_c(c0, n, 0);
                            if is_first {
                                tile::microkernel_16x16_first(ap, bp.add(64), kc, 128);
                            } else {
                                tile::preload_c(c0.add(NR), n, 0);
                                tile::microkernel_16x16_acc(ap, bp.add(64), kc, 128);
                            }
                            tile::store_c(c0.add(NR), n, 0);
                        }

                        if b_rem > 0 {
                            let b_off = (pi * n_blocks + ji) * b_block + n_b32 * kc * 32;
                            edge_kernel(
                                a_buf,
                                b_packed,
                                c,
                                n,
                                ir,
                                jc + n_b32 * 32,
                                a_odd_off,
                                MR,
                                0,
                                b_off,
                                32,
                                MR,
                                b_rem,
                                kc,
                            );
                        }
                    }

                    // --- Remainder strip (< MR rows): edge kernel ---
                    if blk.rem_rows > 0 {
                        let ir = blk.ic + (blk.mc / MR) * MR;
                        let a_rem_off = blk.odd_end + pc * MR;

                        for js in 0..n_b32 {
                            let b_off = (pi * n_blocks + ji) * b_block + js * kc * 32;
                            edge_kernel(
                                a_buf,
                                b_packed,
                                c,
                                n,
                                ir,
                                jc + js * 32,
                                a_rem_off,
                                MR,
                                0,
                                b_off,
                                32,
                                blk.rem_rows,
                                16,
                                kc,
                            );
                            edge_kernel(
                                a_buf,
                                b_packed,
                                c,
                                n,
                                ir,
                                jc + js * 32 + NR,
                                a_rem_off,
                                MR,
                                0,
                                b_off + NR,
                                32,
                                blk.rem_rows,
                                16,
                                kc,
                            );
                        }

                        if b_rem > 0 {
                            let b_off = (pi * n_blocks + ji) * b_block + n_b32 * kc * 32;
                            edge_kernel(
                                a_buf,
                                b_packed,
                                c,
                                n,
                                ir,
                                jc + n_b32 * 32,
                                a_rem_off,
                                MR,
                                0,
                                b_off,
                                32,
                                blk.rem_rows,
                                b_rem,
                                kc,
                            );
                        }
                    }

                    jc += nc;
                }
            }
            pc += kc;
        }
    }

    PACK_CACHE.with(|cache_cell| {
        let mut cache = cache_cell.borrow_mut();
        cache.a_src = a_src_id;
        cache.a_dims = a_dims;
        cache.a = Some(a_pack);
    });
}

/// Fallback GEBP worker: per-KC-block A packing (original algorithm).
/// Used when full-K A pack would exceed L2 cache.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn gebp_worker_fallback(
    a: &[f32],
    b_packed: &[f32],
    c: &mut [f32],
    m_local: usize,
    n: usize,
    k: usize,
    kc_max: usize,
    nc_max: usize,
    k_blocks: usize,
    n_blocks: usize,
    b_block: usize,
    mc_max: usize,
) {
    ensure_amx();

    let mc = m_local.min(mc_max);
    let a_need = mc.div_ceil(MR) * MR * kc_max;
    let mut a_pack = PACK_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        cached_buf(&mut cache.a, a_need)
    });
    let mut pc = 0;
    for pi in 0..k_blocks {
        let kc = (k - pc).min(kc_max);
        let mut ic = 0;
        while ic < m_local {
            let mc_cur = (m_local - ic).min(mc);
            pack_a_gebp(a, k, ic, pc, mc_cur, kc, a_pack.as_mut_slice());
            let mut jc = 0;
            for ji in 0..n_blocks {
                let nc = (n - jc).min(nc_max);
                let off = (pi * n_blocks + ji) * b_block;
                gebp_kernel(
                    a_pack.as_slice(),
                    &b_packed[off..],
                    c,
                    n,
                    ic,
                    jc,
                    mc_cur,
                    nc,
                    kc,
                    pc == 0,
                );
                jc += nc;
            }
            ic += mc_cur;
        }
        pc += kc;
    }
    PACK_CACHE.with(|c| {
        c.borrow_mut().a = Some(a_pack);
    });
}

/// Single-threaded AMX matmul_f32 (full GEBP).
#[cfg(target_arch = "aarch64")]
fn matmul_f32_amx_single(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    ensure_amx();

    // L1 constraint: KC × (MR+NR) × 4 ≤ L1D (64KB).
    // A panel (MC×KC) lives in L2 (4MB). Larger MC = fewer M-panels.
    // Cap to actual dimensions to avoid over-allocation for small matrices.
    let kc_max = k.min(if k > 256 { 512 } else { 256 });
    let mc_max = m.min(if m > 128 { 256 } else { MC }); // MC=256 → A panel in L2
    let nc_max = n.min(if n > 256 { 512 } else { 256 });
    // A: pairs (32*kc) + odd/rem strips (16*kc each). Same total as old layout.
    let a_need = mc_max.div_ceil(MR) * MR * kc_max;
    // B: 32-wide strips.
    let b_need = kc_max * nc_max.div_ceil(32) * 32;

    // Thread-local cache: reuse warm buffers across calls.
    let (mut a_pack, mut b_pack) = PACK_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        let a = cached_buf(&mut cache.a, a_need);
        let b = cached_buf(&mut cache.b, b_need);
        (a, b)
    });

    // Loop order: pc → jc → ic. Pack B once per (pc, jc), reuse across ic.
    // B stays in L2 while A panels are repacked per ic (smaller, NEON-fast).
    let mut pc = 0;
    while pc < k {
        let kc = (k - pc).min(kc_max);

        let mut jc = 0;
        while jc < n {
            let nc = (n - jc).min(nc_max);
            pack_b_32(b, n, pc, jc, kc, nc, b_pack.as_mut_slice());

            let mut ic = 0;
            while ic < m {
                let mc = (m - ic).min(mc_max);
                pack_a_gebp(a, k, ic, pc, mc, kc, a_pack.as_mut_slice());
                gebp_kernel(
                    a_pack.as_slice(),
                    b_pack.as_slice(),
                    c,
                    n,
                    ic,
                    jc,
                    mc,
                    nc,
                    kc,
                    pc == 0,
                );
                ic += mc;
            }
            jc += nc;
        }
        pc += kc;
    }

    // Return buffers to thread-local cache for reuse.
    PACK_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        cache.a = Some(a_pack);
        cache.b = Some(b_pack);
    });
}

// ---------------------------------------------------------------------------
// Packing: direct MR/NR-width strips (no repack needed)
// ---------------------------------------------------------------------------

/// Pack A[ic..ic+mc, pc..pc+kc] into MR-wide contiguous strips.
///
/// Layout: n_strips × kc × MR, where each strip is MR contiguous f32
/// per k step. Microkernel reads directly without repacking.
pub(super) fn pack_a_mr(
    a: &[f32],
    lda: usize,
    ic: usize,
    pc: usize,
    mc: usize,
    kc: usize,
    dst: &mut [f32],
) {
    let n_full = mc / MR;
    let rem = mc % MR;

    for s in 0..n_full {
        let base = s * kc * MR;
        let row_start = ic + s * MR;

        #[cfg(target_arch = "aarch64")]
        {
            pack_a_strip_neon(a, lda, row_start, pc, kc, &mut dst[base..]);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for i in 0..MR {
                let a_row = (row_start + i) * lda + pc;
                for p in 0..kc {
                    dst[base + p * MR + i] = a[a_row + p];
                }
            }
        }
    }

    if rem > 0 {
        let base = n_full * kc * MR;
        let row_start = ic + n_full * MR;
        // Zero the full strip first, then overwrite valid rows.
        for p in 0..kc {
            for i in 0..MR {
                dst[base + p * MR + i] = 0.0;
            }
        }
        for i in 0..rem {
            let a_row = (row_start + i) * lda + pc;
            for p in 0..kc {
                dst[base + p * MR + i] = a[a_row + p];
            }
        }
    }
}

/// NEON-accelerated pack: 4×4 transpose blocks for one MR-wide strip.
///
/// Loop order: p-outer, ig-inner. Each p step processes all 4 row-groups,
/// filling 4 complete output cache lines (64B each) before moving on.
/// This maximizes store-buffer coalescing vs the ig-outer order which
/// partially writes each cache line 4 separate times.
#[cfg(target_arch = "aarch64")]
fn pack_a_strip_neon(
    a: &[f32],
    lda: usize,
    row_start: usize,
    pc: usize,
    kc: usize,
    dst: &mut [f32],
) {
    use core::arch::aarch64::*;

    // Precompute row base offsets for all MR rows.
    let mut rows = [0usize; MR];
    for (i, row) in rows.iter_mut().enumerate() {
        *row = (row_start + i) * lda + pc;
    }

    let mut p = 0;
    while p + 4 <= kc {
        unsafe {
            let d = dst.as_mut_ptr();
            let ap = a.as_ptr();
            // Process all 4 row-groups for columns p..p+4.
            // Each group writes 4 floats to each of 4 cache lines;
            // after all 4 groups, each cache line is complete (16 floats = 64B).
            for ig in 0..4u32 {
                let i = (ig * 4) as usize;
                let r0 = vld1q_f32(ap.add(rows[i] + p));
                let r1 = vld1q_f32(ap.add(rows[i + 1] + p));
                let r2 = vld1q_f32(ap.add(rows[i + 2] + p));
                let r3 = vld1q_f32(ap.add(rows[i + 3] + p));

                let lo01 = vzip1q_f32(r0, r1);
                let hi01 = vzip2q_f32(r0, r1);
                let lo23 = vzip1q_f32(r2, r3);
                let hi23 = vzip2q_f32(r2, r3);

                let c0 = vreinterpretq_f32_f64(vzip1q_f64(
                    vreinterpretq_f64_f32(lo01),
                    vreinterpretq_f64_f32(lo23),
                ));
                let c1 = vreinterpretq_f32_f64(vzip2q_f64(
                    vreinterpretq_f64_f32(lo01),
                    vreinterpretq_f64_f32(lo23),
                ));
                let c2 = vreinterpretq_f32_f64(vzip1q_f64(
                    vreinterpretq_f64_f32(hi01),
                    vreinterpretq_f64_f32(hi23),
                ));
                let c3 = vreinterpretq_f32_f64(vzip2q_f64(
                    vreinterpretq_f64_f32(hi01),
                    vreinterpretq_f64_f32(hi23),
                ));

                vst1q_f32(d.add(p * MR + i), c0);
                vst1q_f32(d.add((p + 1) * MR + i), c1);
                vst1q_f32(d.add((p + 2) * MR + i), c2);
                vst1q_f32(d.add((p + 3) * MR + i), c3);
            }
        }
        p += 4;
    }

    // Remainder columns: scalar, all rows at once.
    while p < kc {
        for i in 0..MR {
            dst[p * MR + i] = a[rows[i] + p];
        }
        p += 1;
    }
}

/// Pack A for GEBP: interleaved pairs (128 bytes/k-step) + odd/remainder strips (64 bytes/k-step).
/// Layout: [pair0: 32*kc][pair1: 32*kc]...[odd: 16*kc][rem: 16*kc]
#[cfg(target_arch = "aarch64")]
fn pack_a_gebp(a: &[f32], lda: usize, ic: usize, pc: usize, mc: usize, kc: usize, dst: &mut [f32]) {
    let n_full_strips = mc / MR;
    let n_pairs = n_full_strips / 2;
    let has_odd = n_full_strips % 2 == 1;
    let has_rem = !mc.is_multiple_of(MR);

    // Pack full pairs using interleaved NEON packing.
    for pair in 0..n_pairs {
        let off = pair * kc * MR * 2;
        let row_start = ic + pair * 2 * MR;
        pack_a_interleaved_strip(a, lda, row_start, pc, kc, &mut dst[off..]);
    }

    let mut off = n_pairs * kc * MR * 2;

    // Pack odd full strip (16-wide).
    if has_odd {
        let row_start = ic + n_pairs * 2 * MR;
        pack_a_strip_neon(a, lda, row_start, pc, kc, &mut dst[off..]);
        off += kc * MR;
    }

    // Pack remainder strip (< MR rows, zero-filled).
    if has_rem {
        let row_start = ic + n_full_strips * MR;
        let rem = mc % MR;
        for p in 0..kc {
            for i in 0..MR {
                dst[off + p * MR + i] = 0.0;
            }
        }
        for i in 0..rem {
            let a_row = (row_start + i) * lda + pc;
            for p in 0..kc {
                dst[off + p * MR + i] = a[a_row + p];
            }
        }
    }
}

/// NEON-accelerated interleaved A pack for 2 strips (32 rows) with column offset.
/// Output stride = 32 floats (128 bytes) per k column for pair LDY.
#[cfg(target_arch = "aarch64")]
fn pack_a_interleaved_strip(
    a: &[f32],
    lda: usize,
    row_start: usize,
    pc: usize,
    kc: usize,
    dst: &mut [f32],
) {
    use core::arch::aarch64::*;

    let mut rows = [0usize; 32];
    for (i, row) in rows.iter_mut().enumerate() {
        *row = (row_start + i) * lda + pc;
    }

    let mut p = 0;
    while p + 4 <= kc {
        unsafe {
            let d = dst.as_mut_ptr();
            let ap = a.as_ptr();
            for ig in 0..8u32 {
                let i = (ig * 4) as usize;
                let r0 = vld1q_f32(ap.add(rows[i] + p));
                let r1 = vld1q_f32(ap.add(rows[i + 1] + p));
                let r2 = vld1q_f32(ap.add(rows[i + 2] + p));
                let r3 = vld1q_f32(ap.add(rows[i + 3] + p));
                let lo01 = vzip1q_f32(r0, r1);
                let hi01 = vzip2q_f32(r0, r1);
                let lo23 = vzip1q_f32(r2, r3);
                let hi23 = vzip2q_f32(r2, r3);
                let c0 = vreinterpretq_f32_f64(vzip1q_f64(
                    vreinterpretq_f64_f32(lo01),
                    vreinterpretq_f64_f32(lo23),
                ));
                let c1 = vreinterpretq_f32_f64(vzip2q_f64(
                    vreinterpretq_f64_f32(lo01),
                    vreinterpretq_f64_f32(lo23),
                ));
                let c2 = vreinterpretq_f32_f64(vzip1q_f64(
                    vreinterpretq_f64_f32(hi01),
                    vreinterpretq_f64_f32(hi23),
                ));
                let c3 = vreinterpretq_f32_f64(vzip2q_f64(
                    vreinterpretq_f64_f32(hi01),
                    vreinterpretq_f64_f32(hi23),
                ));
                vst1q_f32(d.add(p * 32 + i), c0);
                vst1q_f32(d.add((p + 1) * 32 + i), c1);
                vst1q_f32(d.add((p + 2) * 32 + i), c2);
                vst1q_f32(d.add((p + 3) * 32 + i), c3);
            }
        }
        p += 4;
    }
    while p < kc {
        for i in 0..32 {
            dst[p * 32 + i] = a[rows[i] + p];
        }
        p += 1;
    }
}

/// Pack B[pc..pc+kc, jc..jc+nc] into NR-wide contiguous strips.
#[allow(dead_code)]
pub(super) fn pack_b_nr(
    b: &[f32],
    ldb: usize,
    pc: usize,
    jc: usize,
    kc: usize,
    nc: usize,
    dst: &mut [f32],
) {
    let n_full = nc / NR;
    let rem = nc % NR;

    for s in 0..n_full {
        let base = s * kc * NR;
        let col_start = jc + s * NR;
        for p in 0..kc {
            let src_row = (pc + p) * ldb + col_start;
            let dst_off = base + p * NR;
            dst[dst_off..dst_off + NR].copy_from_slice(&b[src_row..src_row + NR]);
        }
    }

    if rem > 0 {
        let base = n_full * kc * NR;
        let col_start = jc + n_full * NR;
        for p in 0..kc {
            let src_row = (pc + p) * ldb + col_start;
            let dst_off = base + p * NR;
            dst[dst_off..dst_off + rem].copy_from_slice(&b[src_row..src_row + rem]);
            for j in rem..NR {
                dst[dst_off + j] = 0.0;
            }
        }
    }
}

/// Pack B[pc..pc+kc, jc..jc+nc] into 32-wide contiguous strips.
/// Layout: n_strips × kc × 32 floats. Zero-fills last strip if nc % 32 != 0.
fn pack_b_32(b: &[f32], ldb: usize, pc: usize, jc: usize, kc: usize, nc: usize, dst: &mut [f32]) {
    let n_full = nc / 32;
    let rem = nc % 32;
    for s in 0..n_full {
        let base = s * kc * 32;
        let col_start = jc + s * 32;
        for p in 0..kc {
            let src_row = (pc + p) * ldb + col_start;
            let dst_off = base + p * 32;
            dst[dst_off..dst_off + 32].copy_from_slice(&b[src_row..src_row + 32]);
        }
    }
    if rem > 0 {
        let base = n_full * kc * 32;
        let col_start = jc + n_full * 32;
        for p in 0..kc {
            let src_row = (pc + p) * ldb + col_start;
            let dst_off = base + p * 32;
            dst[dst_off..dst_off + rem].copy_from_slice(&b[src_row..src_row + rem]);
            for j in rem..32 {
                dst[dst_off + j] = 0.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GEBP kernel: direct micropanel pointers (no repack)
// ---------------------------------------------------------------------------

/// GEBP kernel for pair32 layout.
///
/// A pack layout: [pair0: 32*kc][pair1: 32*kc]...[odd: 16*kc][rem: 16*kc]
/// B pack layout: [strip0: 32*kc][strip1: 32*kc]... (32-wide strips)
///
/// Pair A × 32-wide B → microkernel_32x32_acc (6 ops/k-step).
/// Odd/rem A × 32-wide B → two 16×16 kernels per B strip.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub(super) fn gebp_kernel(
    a_pack: &[f32],
    b_pack: &[f32],
    c: &mut [f32],
    n: usize,
    ic: usize,
    jc: usize,
    mc: usize,
    nc: usize,
    kc: usize,
    first_k: bool,
) {
    let n_full_strips = mc / MR;
    let n_pairs = n_full_strips / 2;
    let has_odd = n_full_strips % 2 == 1;
    let has_rem = !mc.is_multiple_of(MR);
    let n_b32 = nc / 32;
    let b_rem = nc % 32;

    let b_ptr = b_pack.as_ptr();

    // --- Pair A strips × 32-wide B strips: 6 ops/k-step ---
    for pair in 0..n_pairs {
        let ir = pair * 2 * MR;
        let a_pair_off = pair * kc * MR * 2;
        let ap = unsafe { a_pack.as_ptr().add(a_pair_off) as *const u8 };

        for js in 0..n_b32 {
            let jr = js * 32;
            let bp = unsafe { b_ptr.add(js * kc * 32) as *const u8 };
            unsafe {
                let c0 = c.as_mut_ptr().add((ic + ir) * n + jc + jr);
                if first_k {
                    matrix::tile::microkernel_32x32_first(ap, bp, kc, 128);
                } else {
                    matrix::tile::preload_c(c0, n, 0);
                    matrix::tile::preload_c(c0.add(NR), n, 1);
                    matrix::tile::preload_c(c0.add(MR * n), n, 2);
                    matrix::tile::preload_c(c0.add(MR * n + NR), n, 3);
                    matrix::tile::microkernel_32x32_acc(
                        ap,
                        std::ptr::null(),
                        bp,
                        std::ptr::null(),
                        kc,
                        128,
                    );
                }
                matrix::tile::store_c(c0, n, 0);
                matrix::tile::store_c(c0.add(NR), n, 1);
                matrix::tile::store_c(c0.add(MR * n), n, 2);
                matrix::tile::store_c(c0.add(MR * n + NR), n, 3);
            }
        }

        // Remainder B columns (< 32): NEON edge kernel (A stride=32 is not AMX-compatible).
        if b_rem > 0 {
            let b_off = n_b32 * kc * 32;
            for s in 0..2usize {
                edge_kernel(
                    a_pack,
                    b_pack,
                    c,
                    n,
                    ic + ir + s * MR,
                    jc + n_b32 * 32,
                    a_pair_off,
                    32,
                    s * MR,
                    b_off,
                    32,
                    MR,
                    b_rem,
                    kc,
                );
            }
        }
    }

    // --- Odd and remainder A strips: MR-wide (16 floats/k-step) ---
    let tail_a_start = n_pairs * kc * MR * 2;
    let mut a_off = tail_a_start;
    let mut ir = n_pairs * 2 * MR;

    let n_tail = (if has_odd { 1 } else { 0 }) + (if has_rem { 1 } else { 0 });
    for tail_idx in 0..n_tail {
        let mr_actual = if tail_idx == 0 && has_odd {
            MR
        } else {
            mc % MR
        };
        let a_ptr = unsafe { a_pack.as_ptr().add(a_off) as *const u8 };

        for js in 0..n_b32 {
            let jr = js * 32;
            let bp = unsafe { b_ptr.add(js * kc * 32) as *const u8 };
            if mr_actual == MR {
                unsafe {
                    let c0 = c.as_mut_ptr().add((ic + ir) * n + jc + jr);
                    if first_k {
                        matrix::tile::microkernel_16x16_first(a_ptr, bp, kc, 128);
                    } else {
                        matrix::tile::preload_c(c0, n, 0);
                        matrix::tile::microkernel_16x16_acc(a_ptr, bp, kc, 128);
                    }
                    matrix::tile::store_c(c0, n, 0);
                    let c1 = c.as_mut_ptr().add((ic + ir) * n + jc + jr + NR);
                    if first_k {
                        matrix::tile::microkernel_16x16_first(a_ptr, bp.add(64), kc, 128);
                    } else {
                        matrix::tile::preload_c(c1, n, 0);
                        matrix::tile::microkernel_16x16_acc(a_ptr, bp.add(64), kc, 128);
                    }
                    matrix::tile::store_c(c1, n, 0);
                }
            } else {
                let b_base = js * kc * 32;
                edge_kernel(
                    a_pack,
                    b_pack,
                    c,
                    n,
                    ic + ir,
                    jc + jr,
                    a_off,
                    MR,
                    0,
                    b_base,
                    32,
                    mr_actual,
                    16,
                    kc,
                );
                edge_kernel(
                    a_pack,
                    b_pack,
                    c,
                    n,
                    ic + ir,
                    jc + jr + NR,
                    a_off,
                    MR,
                    0,
                    b_base + NR,
                    32,
                    mr_actual,
                    16,
                    kc,
                );
            }
        }

        if b_rem > 0 {
            let b_off = n_b32 * kc * 32;
            let jr = n_b32 * 32;
            let nr_lo = b_rem.min(NR);
            let nr_hi = b_rem.saturating_sub(NR);
            if mr_actual == MR && nr_lo == NR {
                unsafe {
                    let bp = b_ptr.add(b_off) as *const u8;
                    let cp = c.as_mut_ptr().add((ic + ir) * n + jc + jr);
                    matrix::tile::preload_c(cp, n, 0);
                    matrix::tile::microkernel_16x16_acc(a_ptr, bp, kc, 128);
                    matrix::tile::store_c(cp, n, 0);
                }
            } else {
                edge_kernel(
                    a_pack,
                    b_pack,
                    c,
                    n,
                    ic + ir,
                    jc + jr,
                    a_off,
                    MR,
                    0,
                    b_off,
                    32,
                    mr_actual,
                    nr_lo,
                    kc,
                );
            }
            if nr_hi > 0 {
                if mr_actual == MR && nr_hi == NR {
                    unsafe {
                        let bp = b_ptr.add(b_off + NR) as *const u8;
                        let cp = c.as_mut_ptr().add((ic + ir) * n + jc + jr + NR);
                        matrix::tile::preload_c(cp, n, 0);
                        matrix::tile::microkernel_16x16_acc(a_ptr, bp, kc, 128);
                        matrix::tile::store_c(cp, n, 0);
                    }
                } else {
                    edge_kernel(
                        a_pack,
                        b_pack,
                        c,
                        n,
                        ic + ir,
                        jc + jr + NR,
                        a_off,
                        MR,
                        0,
                        b_off + NR,
                        32,
                        mr_actual,
                        nr_hi,
                        kc,
                    );
                }
            }
        }

        a_off += kc * MR;
        ir += MR;
    }
}

/// NEON-accelerated edge kernel for partial tiles.
///
/// `a_off`: offset into a_pack for this A strip.
/// `a_stride`: floats per k-step in A (32 for interleaved pair, 16 for single strip).
/// `a_sub`: sub-offset within A stride (0 for strip0/single, 16 for strip1 in pair).
/// `b_off`: offset into b_pack for B data.
/// `b_stride`: floats per k-step in B (always 32 for 32-wide pack).
#[allow(clippy::too_many_arguments)]
fn edge_kernel(
    a_pack: &[f32],
    b_pack: &[f32],
    c: &mut [f32],
    n: usize,
    c_row: usize,
    c_col: usize,
    a_off: usize,
    a_stride: usize,
    a_sub: usize,
    b_off: usize,
    b_stride: usize,
    mr: usize,
    nr: usize,
    kc: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::*;
        unsafe {
            for i in 0..mr {
                let mut j = 0;
                while j + 4 <= nr {
                    let mut acc = vld1q_f32(c.as_ptr().add(c_row * n + c_col + i * n + j));
                    for p in 0..kc {
                        let av =
                            vdupq_n_f32(*a_pack.get_unchecked(a_off + p * a_stride + a_sub + i));
                        let bv = vld1q_f32(b_pack.as_ptr().add(b_off + p * b_stride + j));
                        acc = vfmaq_f32(acc, av, bv);
                    }
                    vst1q_f32(c.as_mut_ptr().add(c_row * n + c_col + i * n + j), acc);
                    j += 4;
                }
                while j < nr {
                    let mut acc = c[c_row * n + c_col + i * n + j];
                    for p in 0..kc {
                        acc += a_pack[a_off + p * a_stride + a_sub + i]
                            * b_pack[b_off + p * b_stride + j];
                    }
                    c[c_row * n + c_col + i * n + j] = acc;
                    j += 1;
                }
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..mr {
            for j in 0..nr {
                let mut acc = 0.0f32;
                for p in 0..kc {
                    acc +=
                        a_pack[a_off + p * a_stride + a_sub + i] * b_pack[b_off + p * b_stride + j];
                }
                c[c_row * n + c_col + i * n + j] += acc;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fallbacks
// ---------------------------------------------------------------------------

/// NEON matmul_f32 for small matrices.
///
/// 4×16 microkernel: 4 rows × 16 columns, 16 NEON accumulators.
/// Each k step: 1 A scalar broadcast × 4 B loads = 4 FMA per row = 16 FMA total.
/// 16 accumulators saturate NEON FMA throughput (4-cycle latency, 2 pipes).
#[cfg(target_arch = "aarch64")]
fn matmul_f32_neon(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    use core::arch::aarch64::*;
    unsafe {
        let mut i = 0;
        while i + 4 <= m {
            let mut j = 0;
            // 4×16 block: 16 accumulators, one B load serves 4 rows.
            while j + 16 <= n {
                // 4 rows × 4 NEON regs = 16 accumulators.
                let mut c00 = vld1q_f32(c.as_ptr().add(i * n + j));
                let mut c01 = vld1q_f32(c.as_ptr().add(i * n + j + 4));
                let mut c02 = vld1q_f32(c.as_ptr().add(i * n + j + 8));
                let mut c03 = vld1q_f32(c.as_ptr().add(i * n + j + 12));
                let mut c10 = vld1q_f32(c.as_ptr().add((i + 1) * n + j));
                let mut c11 = vld1q_f32(c.as_ptr().add((i + 1) * n + j + 4));
                let mut c12 = vld1q_f32(c.as_ptr().add((i + 1) * n + j + 8));
                let mut c13 = vld1q_f32(c.as_ptr().add((i + 1) * n + j + 12));
                let mut c20 = vld1q_f32(c.as_ptr().add((i + 2) * n + j));
                let mut c21 = vld1q_f32(c.as_ptr().add((i + 2) * n + j + 4));
                let mut c22 = vld1q_f32(c.as_ptr().add((i + 2) * n + j + 8));
                let mut c23 = vld1q_f32(c.as_ptr().add((i + 2) * n + j + 12));
                let mut c30 = vld1q_f32(c.as_ptr().add((i + 3) * n + j));
                let mut c31 = vld1q_f32(c.as_ptr().add((i + 3) * n + j + 4));
                let mut c32 = vld1q_f32(c.as_ptr().add((i + 3) * n + j + 8));
                let mut c33 = vld1q_f32(c.as_ptr().add((i + 3) * n + j + 12));

                for p in 0..k {
                    let b_ptr = b.as_ptr().add(p * n + j);
                    let b0 = vld1q_f32(b_ptr);
                    let b1 = vld1q_f32(b_ptr.add(4));
                    let b2 = vld1q_f32(b_ptr.add(8));
                    let b3 = vld1q_f32(b_ptr.add(12));

                    let a0 = vdupq_n_f32(*a.get_unchecked(i * k + p));
                    c00 = vfmaq_f32(c00, a0, b0);
                    c01 = vfmaq_f32(c01, a0, b1);
                    c02 = vfmaq_f32(c02, a0, b2);
                    c03 = vfmaq_f32(c03, a0, b3);

                    let a1 = vdupq_n_f32(*a.get_unchecked((i + 1) * k + p));
                    c10 = vfmaq_f32(c10, a1, b0);
                    c11 = vfmaq_f32(c11, a1, b1);
                    c12 = vfmaq_f32(c12, a1, b2);
                    c13 = vfmaq_f32(c13, a1, b3);

                    let a2 = vdupq_n_f32(*a.get_unchecked((i + 2) * k + p));
                    c20 = vfmaq_f32(c20, a2, b0);
                    c21 = vfmaq_f32(c21, a2, b1);
                    c22 = vfmaq_f32(c22, a2, b2);
                    c23 = vfmaq_f32(c23, a2, b3);

                    let a3 = vdupq_n_f32(*a.get_unchecked((i + 3) * k + p));
                    c30 = vfmaq_f32(c30, a3, b0);
                    c31 = vfmaq_f32(c31, a3, b1);
                    c32 = vfmaq_f32(c32, a3, b2);
                    c33 = vfmaq_f32(c33, a3, b3);
                }

                vst1q_f32(c.as_mut_ptr().add(i * n + j), c00);
                vst1q_f32(c.as_mut_ptr().add(i * n + j + 4), c01);
                vst1q_f32(c.as_mut_ptr().add(i * n + j + 8), c02);
                vst1q_f32(c.as_mut_ptr().add(i * n + j + 12), c03);
                vst1q_f32(c.as_mut_ptr().add((i + 1) * n + j), c10);
                vst1q_f32(c.as_mut_ptr().add((i + 1) * n + j + 4), c11);
                vst1q_f32(c.as_mut_ptr().add((i + 1) * n + j + 8), c12);
                vst1q_f32(c.as_mut_ptr().add((i + 1) * n + j + 12), c13);
                vst1q_f32(c.as_mut_ptr().add((i + 2) * n + j), c20);
                vst1q_f32(c.as_mut_ptr().add((i + 2) * n + j + 4), c21);
                vst1q_f32(c.as_mut_ptr().add((i + 2) * n + j + 8), c22);
                vst1q_f32(c.as_mut_ptr().add((i + 2) * n + j + 12), c23);
                vst1q_f32(c.as_mut_ptr().add((i + 3) * n + j), c30);
                vst1q_f32(c.as_mut_ptr().add((i + 3) * n + j + 4), c31);
                vst1q_f32(c.as_mut_ptr().add((i + 3) * n + j + 8), c32);
                vst1q_f32(c.as_mut_ptr().add((i + 3) * n + j + 12), c33);
                j += 16;
            }
            // 4×4 remainder columns.
            while j + 4 <= n {
                let mut c0 = vld1q_f32(c.as_ptr().add(i * n + j));
                let mut c1 = vld1q_f32(c.as_ptr().add((i + 1) * n + j));
                let mut c2 = vld1q_f32(c.as_ptr().add((i + 2) * n + j));
                let mut c3 = vld1q_f32(c.as_ptr().add((i + 3) * n + j));
                for p in 0..k {
                    let bv = vld1q_f32(b.as_ptr().add(p * n + j));
                    c0 = vfmaq_f32(c0, vdupq_n_f32(*a.get_unchecked(i * k + p)), bv);
                    c1 = vfmaq_f32(c1, vdupq_n_f32(*a.get_unchecked((i + 1) * k + p)), bv);
                    c2 = vfmaq_f32(c2, vdupq_n_f32(*a.get_unchecked((i + 2) * k + p)), bv);
                    c3 = vfmaq_f32(c3, vdupq_n_f32(*a.get_unchecked((i + 3) * k + p)), bv);
                }
                vst1q_f32(c.as_mut_ptr().add(i * n + j), c0);
                vst1q_f32(c.as_mut_ptr().add((i + 1) * n + j), c1);
                vst1q_f32(c.as_mut_ptr().add((i + 2) * n + j), c2);
                vst1q_f32(c.as_mut_ptr().add((i + 3) * n + j), c3);
                j += 4;
            }
            // Scalar tail columns.
            while j < n {
                for ii in 0..4 {
                    let mut acc = *c.get_unchecked((i + ii) * n + j);
                    for p in 0..k {
                        acc += a.get_unchecked((i + ii) * k + p) * b.get_unchecked(p * n + j);
                    }
                    *c.get_unchecked_mut((i + ii) * n + j) = acc;
                }
                j += 1;
            }
            i += 4;
        }
        // Remaining rows: 1 at a time with NEON.
        while i < m {
            let mut j = 0;
            while j + 4 <= n {
                let mut acc = vld1q_f32(c.as_ptr().add(i * n + j));
                for p in 0..k {
                    let a_val = vdupq_n_f32(*a.get_unchecked(i * k + p));
                    acc = vfmaq_f32(acc, a_val, vld1q_f32(b.as_ptr().add(p * n + j)));
                }
                vst1q_f32(c.as_mut_ptr().add(i * n + j), acc);
                j += 4;
            }
            while j < n {
                let mut acc = *c.get_unchecked(i * n + j);
                for p in 0..k {
                    acc += a.get_unchecked(i * k + p) * b.get_unchecked(p * n + j);
                }
                *c.get_unchecked_mut(i * n + j) = acc;
                j += 1;
            }
            i += 1;
        }
    }
}

#[allow(dead_code)]
fn matmul_f32_scalar(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
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

    #[test]
    fn matmul_f32_identity() {
        const N: usize = 64;
        let mut a = vec![0.0f32; N * N];
        let mut b = vec![0.0f32; N * N];
        let mut c = vec![0.0f32; N * N];

        for i in 0..N {
            for j in 0..N {
                a[i * N + j] = 1.0;
            }
            b[i * N + i] = 1.0;
        }

        matmul_f32(&a, &b, &mut c, N, N, N);

        for i in 0..N {
            for j in 0..N {
                assert!(
                    (c[i * N + j] - 1.0).abs() < 1e-4,
                    "mismatch at [{i},{j}]: {}",
                    c[i * N + j]
                );
            }
        }
    }

    #[test]
    fn matmul_f32_small() {
        const N: usize = 4;
        let a: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let b = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let mut c = vec![0.0f32; 16];
        matmul_f32(&a, &b, &mut c, N, N, N);
        for i in 0..16 {
            assert!((c[i] - a[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn matmul_f32_vs_naive() {
        const M: usize = 33;
        const N: usize = 35;
        const K: usize = 31;
        let a: Vec<f32> = (0..M * K).map(|i| (i % 7) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..K * N).map(|i| (i % 11) as f32 * 0.1).collect();
        let mut c_amx = vec![0.0f32; M * N];
        let mut c_ref = vec![0.0f32; M * N];

        matmul_f32(&a, &b, &mut c_amx, M, N, K);
        matmul_f32_scalar(&a, &b, &mut c_ref, M, N, K);

        for i in 0..M * N {
            assert!(
                (c_amx[i] - c_ref[i]).abs() < 1e-3,
                "mismatch at {i}: amx={} ref={}",
                c_amx[i],
                c_ref[i]
            );
        }
    }

    #[test]
    fn matmul_f32_large_parallel() {
        const M: usize = 256;
        const N: usize = 256;
        const K: usize = 256;
        let a: Vec<f32> = (0..M * K).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
        let b: Vec<f32> = (0..K * N).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let mut c_par = vec![0.0f32; M * N];
        let mut c_ref = vec![0.0f32; M * N];

        matmul_f32(&a, &b, &mut c_par, M, N, K);
        matmul_f32_scalar(&a, &b, &mut c_ref, M, N, K);

        let mut max_err = 0.0f32;
        for i in 0..M * N {
            let err = (c_par[i] - c_ref[i]).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(
            max_err < 0.1,
            "parallel matmul_f32 256x256: max_err={max_err}"
        );
    }
}
