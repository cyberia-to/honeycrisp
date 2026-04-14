//! Dedicated matrix-vector multiply: C[1×N] = A[1×K] × B[K×N].
//!
//! Two strategies:
//! - Small N (≤ 1024): register-tiled. C stays in 8 NEON regs, store once.
//! - Large N (> 1024): broadcast-FMA. Stream B row by row, C stays in L1.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// Matrix-vector multiply: c[n] += a[k] * b[k][n], row-major B.
#[cfg(target_arch = "aarch64")]
pub fn matvec_f32(a: &[f32], b: &[f32], c: &mut [f32], n: usize, k: usize) {
    debug_assert_eq!(a.len(), k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), n);

    if (128..=2048).contains(&n) {
        // AMX: K-blocked by 8, Y preloaded, pair LDX, vector FMA
        super::gemv_kern::gemv_asm(a, b, c, n, k);
    } else if n < 128 {
        // NEON tiled for small N (AMX overhead too high for < 8 tiles)
        matvec_tiled(a, b, c, n, k);
    } else {
        // NEON broadcast for large N
        matvec_broadcast(a, b, c, n, k);
    }
}

/// Register-tiled: two 32-element tiles (64 elements) processed
/// simultaneously. Uses 16 NEON accumulators. Both tiles share the
/// same A[k] broadcast, halving broadcast overhead.
#[cfg(target_arch = "aarch64")]
fn matvec_tiled(a: &[f32], b: &[f32], c: &mut [f32], n: usize, k: usize) {
    unsafe {
        let pc = c.as_mut_ptr();
        let pa = a.as_ptr();
        let pb = b.as_ptr();

        let mut j = 0;
        // 64-wide: two 32-element tiles sharing one A broadcast per K
        while j + 64 <= n {
            let mut r0 = vld1q_f32(pc.add(j));
            let mut r1 = vld1q_f32(pc.add(j + 4));
            let mut r2 = vld1q_f32(pc.add(j + 8));
            let mut r3 = vld1q_f32(pc.add(j + 12));
            let mut r4 = vld1q_f32(pc.add(j + 16));
            let mut r5 = vld1q_f32(pc.add(j + 20));
            let mut r6 = vld1q_f32(pc.add(j + 24));
            let mut r7 = vld1q_f32(pc.add(j + 28));
            let mut s0 = vld1q_f32(pc.add(j + 32));
            let mut s1 = vld1q_f32(pc.add(j + 36));
            let mut s2 = vld1q_f32(pc.add(j + 40));
            let mut s3 = vld1q_f32(pc.add(j + 44));
            let mut s4 = vld1q_f32(pc.add(j + 48));
            let mut s5 = vld1q_f32(pc.add(j + 52));
            let mut s6 = vld1q_f32(pc.add(j + 56));
            let mut s7 = vld1q_f32(pc.add(j + 60));

            for ki in 0..k {
                let av = vdupq_n_f32(*pa.add(ki));
                let br = pb.add(ki * n + j);
                r0 = vfmaq_f32(r0, av, vld1q_f32(br));
                r1 = vfmaq_f32(r1, av, vld1q_f32(br.add(4)));
                r2 = vfmaq_f32(r2, av, vld1q_f32(br.add(8)));
                r3 = vfmaq_f32(r3, av, vld1q_f32(br.add(12)));
                r4 = vfmaq_f32(r4, av, vld1q_f32(br.add(16)));
                r5 = vfmaq_f32(r5, av, vld1q_f32(br.add(20)));
                r6 = vfmaq_f32(r6, av, vld1q_f32(br.add(24)));
                r7 = vfmaq_f32(r7, av, vld1q_f32(br.add(28)));
                s0 = vfmaq_f32(s0, av, vld1q_f32(br.add(32)));
                s1 = vfmaq_f32(s1, av, vld1q_f32(br.add(36)));
                s2 = vfmaq_f32(s2, av, vld1q_f32(br.add(40)));
                s3 = vfmaq_f32(s3, av, vld1q_f32(br.add(44)));
                s4 = vfmaq_f32(s4, av, vld1q_f32(br.add(48)));
                s5 = vfmaq_f32(s5, av, vld1q_f32(br.add(52)));
                s6 = vfmaq_f32(s6, av, vld1q_f32(br.add(56)));
                s7 = vfmaq_f32(s7, av, vld1q_f32(br.add(60)));
            }

            vst1q_f32(pc.add(j), r0);
            vst1q_f32(pc.add(j + 4), r1);
            vst1q_f32(pc.add(j + 8), r2);
            vst1q_f32(pc.add(j + 12), r3);
            vst1q_f32(pc.add(j + 16), r4);
            vst1q_f32(pc.add(j + 20), r5);
            vst1q_f32(pc.add(j + 24), r6);
            vst1q_f32(pc.add(j + 28), r7);
            vst1q_f32(pc.add(j + 32), s0);
            vst1q_f32(pc.add(j + 36), s1);
            vst1q_f32(pc.add(j + 40), s2);
            vst1q_f32(pc.add(j + 44), s3);
            vst1q_f32(pc.add(j + 48), s4);
            vst1q_f32(pc.add(j + 52), s5);
            vst1q_f32(pc.add(j + 56), s6);
            vst1q_f32(pc.add(j + 60), s7);
            j += 64;
        }
        // 32-wide remainder
        while j + 32 <= n {
            let mut a0 = vld1q_f32(pc.add(j));
            let mut a1 = vld1q_f32(pc.add(j + 4));
            let mut a2 = vld1q_f32(pc.add(j + 8));
            let mut a3 = vld1q_f32(pc.add(j + 12));
            let mut a4 = vld1q_f32(pc.add(j + 16));
            let mut a5 = vld1q_f32(pc.add(j + 20));
            let mut a6 = vld1q_f32(pc.add(j + 24));
            let mut a7 = vld1q_f32(pc.add(j + 28));
            for ki in 0..k {
                let av = vdupq_n_f32(*pa.add(ki));
                let br = pb.add(ki * n + j);
                a0 = vfmaq_f32(a0, av, vld1q_f32(br));
                a1 = vfmaq_f32(a1, av, vld1q_f32(br.add(4)));
                a2 = vfmaq_f32(a2, av, vld1q_f32(br.add(8)));
                a3 = vfmaq_f32(a3, av, vld1q_f32(br.add(12)));
                a4 = vfmaq_f32(a4, av, vld1q_f32(br.add(16)));
                a5 = vfmaq_f32(a5, av, vld1q_f32(br.add(20)));
                a6 = vfmaq_f32(a6, av, vld1q_f32(br.add(24)));
                a7 = vfmaq_f32(a7, av, vld1q_f32(br.add(28)));
            }
            vst1q_f32(pc.add(j), a0);
            vst1q_f32(pc.add(j + 4), a1);
            vst1q_f32(pc.add(j + 8), a2);
            vst1q_f32(pc.add(j + 12), a3);
            vst1q_f32(pc.add(j + 16), a4);
            vst1q_f32(pc.add(j + 20), a5);
            vst1q_f32(pc.add(j + 24), a6);
            vst1q_f32(pc.add(j + 28), a7);
            j += 32;
        }
        while j + 4 <= n {
            let mut acc = vld1q_f32(pc.add(j));
            for ki in 0..k {
                acc = vfmaq_f32(acc, vdupq_n_f32(*pa.add(ki)), vld1q_f32(pb.add(ki * n + j)));
            }
            vst1q_f32(pc.add(j), acc);
            j += 4;
        }
        while j < n {
            let mut s = *pc.add(j);
            for ki in 0..k {
                s += *pa.add(ki) * *pb.add(ki * n + j);
            }
            *pc.add(j) = s;
            j += 1;
        }
    }
}

/// Broadcast-FMA: for each k, broadcast a[k] across entire N.
/// C (N×4 bytes) stays in L1. Best when N is large.
#[cfg(target_arch = "aarch64")]
fn matvec_broadcast(a: &[f32], b: &[f32], c: &mut [f32], n: usize, k: usize) {
    unsafe {
        let pc = c.as_mut_ptr();
        let pa = a.as_ptr();
        let pb = b.as_ptr();

        for ki in 0..k {
            let av = vdupq_n_f32(*pa.add(ki));
            let br = pb.add(ki * n);
            let mut j = 0;

            while j + 32 <= n {
                let c0 = vld1q_f32(pc.add(j));
                let c1 = vld1q_f32(pc.add(j + 4));
                let c2 = vld1q_f32(pc.add(j + 8));
                let c3 = vld1q_f32(pc.add(j + 12));
                let c4 = vld1q_f32(pc.add(j + 16));
                let c5 = vld1q_f32(pc.add(j + 20));
                let c6 = vld1q_f32(pc.add(j + 24));
                let c7 = vld1q_f32(pc.add(j + 28));
                vst1q_f32(pc.add(j), vfmaq_f32(c0, av, vld1q_f32(br.add(j))));
                vst1q_f32(pc.add(j + 4), vfmaq_f32(c1, av, vld1q_f32(br.add(j + 4))));
                vst1q_f32(pc.add(j + 8), vfmaq_f32(c2, av, vld1q_f32(br.add(j + 8))));
                vst1q_f32(pc.add(j + 12), vfmaq_f32(c3, av, vld1q_f32(br.add(j + 12))));
                vst1q_f32(pc.add(j + 16), vfmaq_f32(c4, av, vld1q_f32(br.add(j + 16))));
                vst1q_f32(pc.add(j + 20), vfmaq_f32(c5, av, vld1q_f32(br.add(j + 20))));
                vst1q_f32(pc.add(j + 24), vfmaq_f32(c6, av, vld1q_f32(br.add(j + 24))));
                vst1q_f32(pc.add(j + 28), vfmaq_f32(c7, av, vld1q_f32(br.add(j + 28))));
                j += 32;
            }
            while j + 4 <= n {
                let cv = vld1q_f32(pc.add(j));
                vst1q_f32(pc.add(j), vfmaq_f32(cv, av, vld1q_f32(br.add(j))));
                j += 4;
            }
            while j < n {
                *pc.add(j) += *pa.add(ki) * *br.add(j);
                j += 1;
            }
        }
    }
}

/// Matrix-vector multiply: c = a × B (set, not accumulate).
#[cfg(target_arch = "aarch64")]
pub fn matvec_f32_set(a: &[f32], b: &[f32], c: &mut [f32], n: usize, k: usize) {
    c.fill(0.0);
    matvec_f32(a, b, c, n, k);
}

#[cfg(not(target_arch = "aarch64"))]
pub fn matvec_f32(a: &[f32], b: &[f32], c: &mut [f32], n: usize, k: usize) {
    for ki in 0..k {
        let av = a[ki];
        let row = ki * n;
        for j in 0..n {
            c[j] += av * b[row + j];
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn matvec_f32_set(a: &[f32], b: &[f32], c: &mut [f32], n: usize, k: usize) {
    c.fill(0.0);
    matvec_f32(a, b, c, n, k);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_basic() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut c = [0.0f32; 2];
        matvec_f32(&a, &b, &mut c, 2, 3);
        assert!((c[0] - 4.0).abs() < 1e-5);
        assert!((c[1] - 5.0).abs() < 1e-5);
    }

    fn check_matvec(n: usize, k: usize) {
        let a: Vec<f32> = (0..k).map(|i| (i % 7) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 * 0.01).collect();
        let mut c = vec![0.0f32; n];
        let mut c_ref = vec![0.0f32; n];

        matvec_f32_set(&a, &b, &mut c, n, k);

        for ki in 0..k {
            for j in 0..n {
                c_ref[j] += a[ki] * b[ki * n + j];
            }
        }
        for j in 0..n {
            assert!(
                (c[j] - c_ref[j]).abs() < c_ref[j].abs() * 1e-4 + 1e-3,
                "mismatch at {j}: {} vs {} (n={n}, k={k})",
                c[j],
                c_ref[j]
            );
        }
    }

    #[test]
    fn matvec_small_tiled() {
        check_matvec(512, 512);
    }

    #[test]
    fn matvec_medium_tiled() {
        check_matvec(1024, 1024);
    }

    #[test]
    fn matvec_large_broadcast() {
        check_matvec(4096, 4096);
    }

    #[test]
    fn matvec_rectangular() {
        check_matvec(11008, 4096);
    }
}
