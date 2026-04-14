//! Complex multiply-accumulate using FCMLA (ARMv8.3-A FEAT_FCMA).

/// Complex multiply-accumulate: `acc += a * b` (element-wise complex).
///
/// Inputs are interleaved complex: `[re0, im0, re1, im1, ...]`.
/// Each slice length must be even (pairs of `(re, im)`).
///
/// Uses FCMLA on aarch64 (ARMv8.3-A FEAT_FCMA, available on Apple M1+).
///
/// # Panics
///
/// Panics if any slice has odd length or if `a` and `b` differ in length.
pub fn complex_mul_acc(acc: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "a and b must have the same length");
    assert_eq!(a.len() % 2, 0, "complex slices must have even length");
    let n = acc.len().min(a.len());
    assert_eq!(n % 2, 0, "acc length must be even");

    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        // FCMLA processes 2 complex pairs (4 f32) per instruction.
        // Two FCMLA ops (rotate 0° and 90°) compute full complex mul-acc.
        // 4× unroll = 8 complex pairs per iteration.
        unsafe {
            let pa = a.as_ptr();
            let pb = b.as_ptr();
            let pc = acc.as_mut_ptr();

            while i + 16 <= n {
                let mut c0 = core::arch::aarch64::vld1q_f32(pc.add(i));
                let mut c1 = core::arch::aarch64::vld1q_f32(pc.add(i + 4));
                let mut c2 = core::arch::aarch64::vld1q_f32(pc.add(i + 8));
                let mut c3 = core::arch::aarch64::vld1q_f32(pc.add(i + 12));
                let a0 = core::arch::aarch64::vld1q_f32(pa.add(i));
                let a1 = core::arch::aarch64::vld1q_f32(pa.add(i + 4));
                let a2 = core::arch::aarch64::vld1q_f32(pa.add(i + 8));
                let a3 = core::arch::aarch64::vld1q_f32(pa.add(i + 12));
                let b0 = core::arch::aarch64::vld1q_f32(pb.add(i));
                let b1 = core::arch::aarch64::vld1q_f32(pb.add(i + 4));
                let b2 = core::arch::aarch64::vld1q_f32(pb.add(i + 8));
                let b3 = core::arch::aarch64::vld1q_f32(pb.add(i + 12));
                // FCMLA rotate 0°: acc.re += a.re*b.re, acc.im += a.re*b.im
                core::arch::asm!(
                    "fcmla {c0:v}.4s, {a0:v}.4s, {b0:v}.4s, #0",
                    "fcmla {c1:v}.4s, {a1:v}.4s, {b1:v}.4s, #0",
                    "fcmla {c2:v}.4s, {a2:v}.4s, {b2:v}.4s, #0",
                    "fcmla {c3:v}.4s, {a3:v}.4s, {b3:v}.4s, #0",
                    "fcmla {c0:v}.4s, {a0:v}.4s, {b0:v}.4s, #90",
                    "fcmla {c1:v}.4s, {a1:v}.4s, {b1:v}.4s, #90",
                    "fcmla {c2:v}.4s, {a2:v}.4s, {b2:v}.4s, #90",
                    "fcmla {c3:v}.4s, {a3:v}.4s, {b3:v}.4s, #90",
                    c0 = inout(vreg) c0, c1 = inout(vreg) c1,
                    c2 = inout(vreg) c2, c3 = inout(vreg) c3,
                    a0 = in(vreg) a0, a1 = in(vreg) a1,
                    a2 = in(vreg) a2, a3 = in(vreg) a3,
                    b0 = in(vreg) b0, b1 = in(vreg) b1,
                    b2 = in(vreg) b2, b3 = in(vreg) b3,
                );
                core::arch::aarch64::vst1q_f32(pc.add(i), c0);
                core::arch::aarch64::vst1q_f32(pc.add(i + 4), c1);
                core::arch::aarch64::vst1q_f32(pc.add(i + 8), c2);
                core::arch::aarch64::vst1q_f32(pc.add(i + 12), c3);
                i += 16;
            }
            while i + 4 <= n {
                let mut c0 = core::arch::aarch64::vld1q_f32(pc.add(i));
                let a0 = core::arch::aarch64::vld1q_f32(pa.add(i));
                let b0 = core::arch::aarch64::vld1q_f32(pb.add(i));
                core::arch::asm!(
                    "fcmla {c:v}.4s, {a:v}.4s, {b:v}.4s, #0",
                    "fcmla {c:v}.4s, {a:v}.4s, {b:v}.4s, #90",
                    c = inout(vreg) c0, a = in(vreg) a0, b = in(vreg) b0,
                );
                core::arch::aarch64::vst1q_f32(pc.add(i), c0);
                i += 4;
            }
        }
    }

    // Scalar tail / fallback
    while i + 1 < n {
        let ar = a[i];
        let ai = a[i + 1];
        let br = b[i];
        let bi = b[i + 1];
        acc[i] += ar * br - ai * bi;
        acc[i + 1] += ar * bi + ai * br;
        i += 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_mul_acc_basic() {
        // (1+2i) * (3+4i) = (1*3 - 2*4) + (1*4 + 2*3)i = -5 + 10i
        let a = [1.0f32, 2.0];
        let b = [3.0f32, 4.0];
        let mut acc = [0.0f32, 0.0];
        complex_mul_acc(&mut acc, &a, &b);
        assert!((acc[0] - (-5.0)).abs() < 1e-6);
        assert!((acc[1] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn complex_mul_acc_accumulates() {
        let a = [1.0f32, 0.0, 0.0, 1.0];
        let b = [1.0f32, 0.0, 0.0, 1.0];
        let mut acc = [10.0f32, 20.0, 30.0, 40.0];
        complex_mul_acc(&mut acc, &a, &b);
        // (1+0i)*(1+0i) = 1+0i => acc[0..2] = [11, 20]
        // (0+1i)*(0+1i) = -1+0i => acc[2..4] = [29, 40]
        assert!((acc[0] - 11.0).abs() < 1e-6);
        assert!((acc[1] - 20.0).abs() < 1e-6);
        assert!((acc[2] - 29.0).abs() < 1e-6);
        assert!((acc[3] - 40.0).abs() < 1e-6);
    }

    #[test]
    fn complex_mul_acc_large() {
        let n = 4096;
        let a: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| (i % 11) as f32 * 0.1).collect();
        let mut acc_neon = vec![0.0f32; n];
        let mut acc_scalar = vec![0.0f32; n];

        complex_mul_acc(&mut acc_neon, &a, &b);

        // Scalar reference
        for i in (0..n).step_by(2) {
            acc_scalar[i] += a[i] * b[i] - a[i + 1] * b[i + 1];
            acc_scalar[i + 1] += a[i] * b[i + 1] + a[i + 1] * b[i];
        }

        for i in 0..n {
            assert!(
                (acc_neon[i] - acc_scalar[i]).abs() < 1e-4,
                "mismatch at {}: {} vs {}",
                i,
                acc_neon[i],
                acc_scalar[i]
            );
        }
    }
}
