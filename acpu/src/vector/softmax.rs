// ---------------------------------------------------------------------------
// Compound vector operations -- softmax, normalize
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use super::math;
use super::reduce;

/// In-place softmax: x_i = exp(x_i - max) / sum(exp(x - max))
///
/// 2-pass: (1) find max via fast reduction, (2) fused exp + sum + divide.
/// Pass 2 reads input once, computes exp(x-max), accumulates sum, stores exp.
/// Then one final 16-wide scale pass divides by sum.
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    let len = x.len();
    let m = reduce::max(x);

    // Pass 1: exp(x-max) + accumulate sum, store exp values
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    let sum = {
        unsafe {
            let mv = vdupq_n_f32(m);
            let mut s0 = vdupq_n_f32(0.0);
            let mut s1 = vdupq_n_f32(0.0);
            let mut s2 = vdupq_n_f32(0.0);
            let mut s3 = vdupq_n_f32(0.0);
            let ptr = x.as_mut_ptr();
            while i + 16 <= len {
                let e0 = math::exp_neon(vsubq_f32(vld1q_f32(ptr.add(i)), mv));
                let e1 = math::exp_neon(vsubq_f32(vld1q_f32(ptr.add(i + 4)), mv));
                let e2 = math::exp_neon(vsubq_f32(vld1q_f32(ptr.add(i + 8)), mv));
                let e3 = math::exp_neon(vsubq_f32(vld1q_f32(ptr.add(i + 12)), mv));
                vst1q_f32(ptr.add(i), e0);
                vst1q_f32(ptr.add(i + 4), e1);
                vst1q_f32(ptr.add(i + 8), e2);
                vst1q_f32(ptr.add(i + 12), e3);
                s0 = vaddq_f32(s0, e0);
                s1 = vaddq_f32(s1, e1);
                s2 = vaddq_f32(s2, e2);
                s3 = vaddq_f32(s3, e3);
                i += 16;
            }
            while i + 4 <= len {
                let e = math::exp_neon(vsubq_f32(vld1q_f32(ptr.add(i)), mv));
                vst1q_f32(ptr.add(i), e);
                s0 = vaddq_f32(s0, e);
                i += 4;
            }
            let mut s = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
            while i < len {
                let e = math::exp_scalar(x[i] - m);
                x[i] = e;
                s += e;
                i += 1;
            }
            s
        }
    };

    #[cfg(not(target_arch = "aarch64"))]
    let sum = {
        let mut s = 0.0f32;
        for v in x.iter_mut() {
            let e = math::exp_scalar(*v - m);
            *v = e;
            s += e;
        }
        s
    };

    if sum == 0.0 {
        return;
    }

    // Pass 2: divide by sum — 16-wide
    let inv_s = 1.0 / sum;
    i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let inv_v = vdupq_n_f32(inv_s);
            let ptr = x.as_mut_ptr();
            while i + 16 <= len {
                vst1q_f32(ptr.add(i), vmulq_f32(vld1q_f32(ptr.add(i)), inv_v));
                vst1q_f32(ptr.add(i + 4), vmulq_f32(vld1q_f32(ptr.add(i + 4)), inv_v));
                vst1q_f32(ptr.add(i + 8), vmulq_f32(vld1q_f32(ptr.add(i + 8)), inv_v));
                vst1q_f32(
                    ptr.add(i + 12),
                    vmulq_f32(vld1q_f32(ptr.add(i + 12)), inv_v),
                );
                i += 16;
            }
            while i + 4 <= len {
                vst1q_f32(ptr.add(i), vmulq_f32(vld1q_f32(ptr.add(i)), inv_v));
                i += 4;
            }
        }
    }

    while i < len {
        x[i] *= inv_s;
        i += 1;
    }
}

/// RMS normalization: out_i = (x_i / rms) * weight_i
/// where rms = sqrt(mean(x^2) + eps)
pub fn normalize(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let len = x.len();
    assert_eq!(len, weight.len(), "normalize: x and weight length mismatch");
    assert!(out.len() >= len, "normalize: output buffer too small");

    if len == 0 {
        return;
    }

    // Pass 1: sum of squares — 4 accumulators, 16-wide unroll for ILP
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    let ss = {
        unsafe {
            let mut a0 = vdupq_n_f32(0.0);
            let mut a1 = vdupq_n_f32(0.0);
            let mut a2 = vdupq_n_f32(0.0);
            let mut a3 = vdupq_n_f32(0.0);
            while i + 16 <= len {
                let v0 = vld1q_f32(x.as_ptr().add(i));
                let v1 = vld1q_f32(x.as_ptr().add(i + 4));
                let v2 = vld1q_f32(x.as_ptr().add(i + 8));
                let v3 = vld1q_f32(x.as_ptr().add(i + 12));
                a0 = vfmaq_f32(a0, v0, v0);
                a1 = vfmaq_f32(a1, v1, v1);
                a2 = vfmaq_f32(a2, v2, v2);
                a3 = vfmaq_f32(a3, v3, v3);
                i += 16;
            }
            // 4-wide tail
            while i + 4 <= len {
                let v = vld1q_f32(x.as_ptr().add(i));
                a0 = vfmaq_f32(a0, v, v);
                i += 4;
            }
            let sum4 = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
            let mut s = vaddvq_f32(sum4);
            while i < len {
                s += x[i] * x[i];
                i += 1;
            }
            s
        }
    };

    #[cfg(not(target_arch = "aarch64"))]
    let ss = {
        let mut s = 0.0f32;
        for &v in x {
            s += v * v;
        }
        s
    };

    let rms_inv = 1.0 / (ss / len as f32 + eps).sqrt();

    // Pass 2: normalize and scale — 16-wide unroll, fused x*scale*weight
    i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let scale = vdupq_n_f32(rms_inv);
            while i + 16 <= len {
                let x0 = vld1q_f32(x.as_ptr().add(i));
                let x1 = vld1q_f32(x.as_ptr().add(i + 4));
                let x2 = vld1q_f32(x.as_ptr().add(i + 8));
                let x3 = vld1q_f32(x.as_ptr().add(i + 12));
                let w0 = vld1q_f32(weight.as_ptr().add(i));
                let w1 = vld1q_f32(weight.as_ptr().add(i + 4));
                let w2 = vld1q_f32(weight.as_ptr().add(i + 8));
                let w3 = vld1q_f32(weight.as_ptr().add(i + 12));
                // fused: out = (x * scale) * weight
                vst1q_f32(out.as_mut_ptr().add(i), vmulq_f32(vmulq_f32(x0, scale), w0));
                vst1q_f32(
                    out.as_mut_ptr().add(i + 4),
                    vmulq_f32(vmulq_f32(x1, scale), w1),
                );
                vst1q_f32(
                    out.as_mut_ptr().add(i + 8),
                    vmulq_f32(vmulq_f32(x2, scale), w2),
                );
                vst1q_f32(
                    out.as_mut_ptr().add(i + 12),
                    vmulq_f32(vmulq_f32(x3, scale), w3),
                );
                i += 16;
            }
            while i + 4 <= len {
                let vx = vld1q_f32(x.as_ptr().add(i));
                let vw = vld1q_f32(weight.as_ptr().add(i));
                vst1q_f32(out.as_mut_ptr().add(i), vmulq_f32(vmulq_f32(vx, scale), vw));
                i += 4;
            }
        }
    }

    while i < len {
        out[i] = x[i] * rms_inv * weight[i];
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_sums_to_one() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        softmax(&mut v);
        let s: f32 = v.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
        // should be monotonically increasing
        for i in 1..v.len() {
            assert!(v[i] >= v[i - 1]);
        }
    }

    #[test]
    fn test_softmax_single() {
        let mut v = vec![42.0];
        softmax(&mut v);
        assert!((v[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_equal() {
        let mut v = vec![1.0; 4];
        softmax(&mut v);
        for &val in &v {
            assert!((val - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn test_normalize_unit_weight() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        let mut out = vec![0.0; 4];
        normalize(&mut out, &x, &w, 1e-5);
        // After normalize with unit weights, the RMS of out should be ~1
        let ss: f32 = out.iter().map(|v| v * v).sum::<f32>() / out.len() as f32;
        assert!((ss - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_normalize_scaling() {
        let x = vec![2.0; 8];
        let w = vec![3.0; 8];
        let mut out = vec![0.0; 8];
        normalize(&mut out, &x, &w, 1e-5);
        // rms = 2.0, so normalized = 1.0, * weight 3.0 = 3.0
        for &v in &out {
            assert!((v - 3.0).abs() < 1e-3);
        }
    }
}
