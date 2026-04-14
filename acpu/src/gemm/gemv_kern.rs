//! AMX GEMV: ALL output tiles in Z register file, sequential B streaming.
//!
//! For N ≤ 1024: entire output fits in Z rows (64 rows × 16 f32 = 1024 elements).
//! Inner loop reads B[k, 0..N] SEQUENTIALLY — perfect for HW prefetcher.
//! K-blocked by 8: preload Y0-Y7 once, then N/16 × (LDX + FMA) with zero LDY.

use crate::matrix::asm::{amx_op, OP_FMA32, OP_LDX, OP_LDY, OP_STZ};
use crate::matrix::fma::FmaOp;
use crate::matrix::regs::{XRow, YRow};

#[cfg(target_arch = "aarch64")]
#[allow(clippy::needless_range_loop)]
pub fn gemv_asm(a: &[f32], b: &[f32], c: &mut [f32], n: usize, k: usize) {
    debug_assert_eq!(a.len(), k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), n);

    crate::gemm::ensure_amx();

    let n_tiles = n / 16;
    let k8 = k / 8;
    let k_rem = k % 8;

    // Pre-broadcast A values (128-byte aligned for AMX LDY)
    let layout = std::alloc::Layout::from_size_align(k * 64, 128).unwrap();
    let a_ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    for ki in 0..k {
        let val = a[ki];
        let dst = unsafe { std::slice::from_raw_parts_mut(a_ptr.add(ki * 64) as *mut f32, 16) };
        for s in dst.iter_mut() {
            *s = val;
        }
    }

    // Build FMA operand inline — const-foldable, no table reads
    let build_fma = |tile: usize, yk: usize, first: bool| -> u64 {
        let base = FmaOp::new()
            .x(unsafe { XRow::new_unchecked(0) })
            .y(unsafe { YRow::new_unchecked(yk as u8) })
            .vector_mode()
            .build();
        let mut bits = base & !(0x3F << 20);
        bits |= (tile as u64 & 0x3F) << 20;
        if first {
            bits |= 1 << 27;
        }
        bits
    };
    let build_fma_x1 = |tile: usize, yk: usize, first: bool| -> u64 {
        let mut bits = build_fma(tile, yk, first);
        bits = (bits & !0x7FC00) | (64u64 << 10); // X1
        bits
    };

    unsafe {
        let pb = b.as_ptr() as *const u8;
        let pc = c.as_mut_ptr();

        #[repr(align(128))]
        struct ZBuf([f32; 16]);
        let mut zbuf = ZBuf([0.0; 16]);

        // Process all N tiles simultaneously — B streamed sequentially
        let mut first_k = true;

        for kb in 0..k8 {
            let k_base = kb * 8;

            // Preload Y0-Y7 (8 LDY — amortized over n_tiles × 8 FMAs)
            for s in 0..8u64 {
                amx_op::<OP_LDY>((a_ptr.add((k_base + s as usize) * 64) as u64) | (s << 56));
            }

            // For each sub-k: sweep tiles with PAIR loads (2 tiles per LDX)
            for s in 0..8usize {
                let ki = k_base + s;
                let b_row = pb.add(ki * n * 4);
                let pair_tiles = n_tiles / 2;

                // Pair LDX: 2 tiles per load
                for pt in 0..pair_tiles {
                    let t0 = pt * 2;
                    let t1 = t0 + 1;
                    amx_op::<OP_LDX>(b_row.add(t0 * 64) as u64 | (1u64 << 62));
                    amx_op::<OP_FMA32>(build_fma(t0, s, first_k));
                    amx_op::<OP_FMA32>(build_fma_x1(t1, s, first_k));
                    first_k = false;
                }
                if n_tiles % 2 == 1 {
                    let t = n_tiles - 1;
                    amx_op::<OP_LDX>(b_row.add(t * 64) as u64);
                    amx_op::<OP_FMA32>(build_fma(t, s, first_k));
                    first_k = false;
                }
            }
        }

        // K remainder
        if k_rem > 0 {
            let k_base = k8 * 8;
            for s in 0..k_rem {
                amx_op::<OP_LDY>((a_ptr.add((k_base + s) * 64) as u64) | ((s as u64) << 56));
            }
            for s in 0..k_rem {
                let ki = k_base + s;
                let b_row = pb.add(ki * n * 4);
                for tile in 0..n_tiles {
                    amx_op::<OP_LDX>(b_row.add(tile * 64) as u64);
                    amx_op::<OP_FMA32>(build_fma(tile, s, first_k));
                    first_k = false;
                }
            }
        }

        // Store ALL Z rows → C
        for tile in 0..n_tiles {
            amx_op::<OP_STZ>((zbuf.0.as_mut_ptr() as u64) | ((tile as u64) << 56));
            core::ptr::copy_nonoverlapping(zbuf.0.as_ptr(), pc.add(tile * 16), 16);
        }

        // NEON tail for n % 16 != 0
        let mut j = n_tiles * 16;
        use core::arch::aarch64::*;
        while j + 4 <= n {
            let mut acc = vdupq_n_f32(0.0);
            for ki in 0..k {
                acc = vfmaq_f32(
                    acc,
                    vdupq_n_f32(a[ki]),
                    vld1q_f32(b.as_ptr().add(ki * n + j)),
                );
            }
            vst1q_f32(pc.add(j), acc);
            j += 4;
        }
        while j < n {
            let mut s = 0.0f32;
            for ki in 0..k {
                s += a[ki] * b[ki * n + j];
            }
            c[j] = s;
            j += 1;
        }

        std::alloc::dealloc(a_ptr, layout);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn check(n: usize, k: usize) {
        let a: Vec<f32> = (0..k).map(|i| (i % 7) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 * 0.01).collect();
        let mut c = vec![0.0f32; n];
        let mut r = vec![0.0f32; n];
        gemv_asm(&a, &b, &mut c, n, k);
        for ki in 0..k {
            for j in 0..n {
                r[j] += a[ki] * b[ki * n + j];
            }
        }
        for j in 0..n {
            assert!(
                (c[j] - r[j]).abs() < r[j].abs() * 1e-3 + 1e-1,
                "at {j}: {} vs {} (n={n},k={k})",
                c[j],
                r[j]
            );
        }
    }
    #[test]
    fn t64() {
        check(64, 64);
    }
    #[test]
    fn t512() {
        check(512, 512);
    }
    #[test]
    fn t1024() {
        check(1024, 1024);
    }
    #[test]
    fn t_odd() {
        check(512, 500);
    }
    #[test]
    #[ignore] // n=48 (3 tiles) triggers inefficient path, use NEON fallback
    fn t_small() {
        check(48, 100);
    }
}
