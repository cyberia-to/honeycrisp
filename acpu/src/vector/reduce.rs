// ---------------------------------------------------------------------------
// Reduction operations -- NEON 4-way unrolled + scalar fallback
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// Sum of all elements. 8-accumulator, 32-wide unroll for NEON pipeline saturation.
pub fn sum(x: &[f32]) -> f32 {
    let len = x.len();
    if len == 0 {
        return 0.0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        let mut i = 0;
        unsafe {
            let mut a0 = vdupq_n_f32(0.0);
            let mut a1 = vdupq_n_f32(0.0);
            let mut a2 = vdupq_n_f32(0.0);
            let mut a3 = vdupq_n_f32(0.0);
            let mut a4 = vdupq_n_f32(0.0);
            let mut a5 = vdupq_n_f32(0.0);
            let mut a6 = vdupq_n_f32(0.0);
            let mut a7 = vdupq_n_f32(0.0);
            let p = x.as_ptr();
            while i + 32 <= len {
                a0 = vaddq_f32(a0, vld1q_f32(p.add(i)));
                a1 = vaddq_f32(a1, vld1q_f32(p.add(i + 4)));
                a2 = vaddq_f32(a2, vld1q_f32(p.add(i + 8)));
                a3 = vaddq_f32(a3, vld1q_f32(p.add(i + 12)));
                a4 = vaddq_f32(a4, vld1q_f32(p.add(i + 16)));
                a5 = vaddq_f32(a5, vld1q_f32(p.add(i + 20)));
                a6 = vaddq_f32(a6, vld1q_f32(p.add(i + 24)));
                a7 = vaddq_f32(a7, vld1q_f32(p.add(i + 28)));
                i += 32;
            }
            a0 = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
            a4 = vaddq_f32(vaddq_f32(a4, a5), vaddq_f32(a6, a7));
            a0 = vaddq_f32(a0, a4);
            while i + 4 <= len {
                a0 = vaddq_f32(a0, vld1q_f32(p.add(i)));
                i += 4;
            }
            let mut s = vaddvq_f32(a0);
            while i < len {
                s += *p.add(i);
                i += 1;
            }
            s
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        x.iter().sum()
    }
}

/// Maximum element. Returns -INF for empty slices. 8-acc 32-wide.
pub fn max(x: &[f32]) -> f32 {
    let len = x.len();
    if len == 0 {
        return f32::NEG_INFINITY;
    }

    #[cfg(target_arch = "aarch64")]
    {
        let mut i = 0;
        unsafe {
            let init = vdupq_n_f32(f32::NEG_INFINITY);
            let (mut a0, mut a1, mut a2, mut a3) = (init, init, init, init);
            let (mut a4, mut a5, mut a6, mut a7) = (init, init, init, init);
            let p = x.as_ptr();
            while i + 32 <= len {
                a0 = vmaxnmq_f32(a0, vld1q_f32(p.add(i)));
                a1 = vmaxnmq_f32(a1, vld1q_f32(p.add(i + 4)));
                a2 = vmaxnmq_f32(a2, vld1q_f32(p.add(i + 8)));
                a3 = vmaxnmq_f32(a3, vld1q_f32(p.add(i + 12)));
                a4 = vmaxnmq_f32(a4, vld1q_f32(p.add(i + 16)));
                a5 = vmaxnmq_f32(a5, vld1q_f32(p.add(i + 20)));
                a6 = vmaxnmq_f32(a6, vld1q_f32(p.add(i + 24)));
                a7 = vmaxnmq_f32(a7, vld1q_f32(p.add(i + 28)));
                i += 32;
            }
            a0 = vmaxnmq_f32(vmaxnmq_f32(a0, a1), vmaxnmq_f32(a2, a3));
            a4 = vmaxnmq_f32(vmaxnmq_f32(a4, a5), vmaxnmq_f32(a6, a7));
            a0 = vmaxnmq_f32(a0, a4);
            while i + 4 <= len {
                a0 = vmaxnmq_f32(a0, vld1q_f32(p.add(i)));
                i += 4;
            }
            let mut m = vmaxnmvq_f32(a0);
            while i < len {
                let v = *p.add(i);
                if v > m {
                    m = v;
                }
                i += 1;
            }
            m
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        x.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }
}

/// Minimum element. Returns +INF for empty slices. 8-acc 32-wide.
pub fn min(x: &[f32]) -> f32 {
    let len = x.len();
    if len == 0 {
        return f32::INFINITY;
    }

    #[cfg(target_arch = "aarch64")]
    {
        let mut i = 0;
        unsafe {
            let init = vdupq_n_f32(f32::INFINITY);
            let (mut a0, mut a1, mut a2, mut a3) = (init, init, init, init);
            let (mut a4, mut a5, mut a6, mut a7) = (init, init, init, init);
            let p = x.as_ptr();
            while i + 32 <= len {
                a0 = vminnmq_f32(a0, vld1q_f32(p.add(i)));
                a1 = vminnmq_f32(a1, vld1q_f32(p.add(i + 4)));
                a2 = vminnmq_f32(a2, vld1q_f32(p.add(i + 8)));
                a3 = vminnmq_f32(a3, vld1q_f32(p.add(i + 12)));
                a4 = vminnmq_f32(a4, vld1q_f32(p.add(i + 16)));
                a5 = vminnmq_f32(a5, vld1q_f32(p.add(i + 20)));
                a6 = vminnmq_f32(a6, vld1q_f32(p.add(i + 24)));
                a7 = vminnmq_f32(a7, vld1q_f32(p.add(i + 28)));
                i += 32;
            }
            a0 = vminnmq_f32(vminnmq_f32(a0, a1), vminnmq_f32(a2, a3));
            a4 = vminnmq_f32(vminnmq_f32(a4, a5), vminnmq_f32(a6, a7));
            a0 = vminnmq_f32(a0, a4);
            while i + 4 <= len {
                a0 = vminnmq_f32(a0, vld1q_f32(p.add(i)));
                i += 4;
            }
            let mut m = vminnmvq_f32(a0);
            while i < len {
                let v = *p.add(i);
                if v < m {
                    m = v;
                }
                i += 1;
            }
            m
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        x.iter().copied().fold(f32::INFINITY, f32::min)
    }
}

/// Dot product of two slices. 8-acc 32-wide. Panics if lengths differ.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot: length mismatch");
    let len = a.len();
    if len == 0 {
        return 0.0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        let mut i = 0;
        unsafe {
            let z = vdupq_n_f32(0.0);
            let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
            let (mut a4, mut a5, mut a6, mut a7) = (z, z, z, z);
            let pa = a.as_ptr();
            let pb = b.as_ptr();
            while i + 32 <= len {
                a0 = vfmaq_f32(a0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
                a1 = vfmaq_f32(a1, vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
                a2 = vfmaq_f32(a2, vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
                a3 = vfmaq_f32(a3, vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
                a4 = vfmaq_f32(a4, vld1q_f32(pa.add(i + 16)), vld1q_f32(pb.add(i + 16)));
                a5 = vfmaq_f32(a5, vld1q_f32(pa.add(i + 20)), vld1q_f32(pb.add(i + 20)));
                a6 = vfmaq_f32(a6, vld1q_f32(pa.add(i + 24)), vld1q_f32(pb.add(i + 24)));
                a7 = vfmaq_f32(a7, vld1q_f32(pa.add(i + 28)), vld1q_f32(pb.add(i + 28)));
                i += 32;
            }
            a0 = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
            a4 = vaddq_f32(vaddq_f32(a4, a5), vaddq_f32(a6, a7));
            a0 = vaddq_f32(a0, a4);
            while i + 4 <= len {
                a0 = vfmaq_f32(a0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
                i += 4;
            }
            let mut s = vaddvq_f32(a0);
            while i < len {
                s += *pa.add(i) * *pb.add(i);
                i += 1;
            }
            s
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

/// L2 norm: sqrt(sum(x_i^2)).
pub fn length(x: &[f32]) -> f32 {
    let len = x.len();
    if len == 0 {
        return 0.0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        let mut i = 0;
        unsafe {
            let mut a0 = vdupq_n_f32(0.0);
            let mut a1 = vdupq_n_f32(0.0);
            let mut a2 = vdupq_n_f32(0.0);
            let mut a3 = vdupq_n_f32(0.0);
            let p = x.as_ptr();
            while i + 16 <= len {
                let v0 = vld1q_f32(p.add(i));
                let v1 = vld1q_f32(p.add(i + 4));
                let v2 = vld1q_f32(p.add(i + 8));
                let v3 = vld1q_f32(p.add(i + 12));
                a0 = vfmaq_f32(a0, v0, v0);
                a1 = vfmaq_f32(a1, v1, v1);
                a2 = vfmaq_f32(a2, v2, v2);
                a3 = vfmaq_f32(a3, v3, v3);
                i += 16;
            }
            a0 = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
            while i + 4 <= len {
                let v = vld1q_f32(p.add(i));
                a0 = vfmaq_f32(a0, v, v);
                i += 4;
            }
            let mut s = vaddvq_f32(a0);
            while i < len {
                let v = *p.add(i);
                s += v * v;
                i += 1;
            }
            s.sqrt()
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        x.iter().map(|v| v * v).sum::<f32>().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((sum(&v) - 15.0).abs() < 1e-5);
    }

    #[test]
    fn test_sum_large() {
        let v: Vec<f32> = (0..4096).map(|i| (i % 10) as f32).collect();
        let expected: f32 = v.iter().sum();
        assert!((sum(&v) - expected).abs() < 1.0);
    }

    #[test]
    fn test_max_min() {
        let v = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        assert!((max(&v) - 9.0).abs() < 1e-5);
        assert!((min(&v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_dot() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot(&a, &b) - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_length() {
        let v = vec![3.0, 4.0];
        assert!((length(&v) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_empty() {
        assert_eq!(sum(&[]), 0.0);
        assert_eq!(max(&[]), f32::NEG_INFINITY);
        assert_eq!(min(&[]), f32::INFINITY);
        assert_eq!(length(&[]), 0.0);
    }
}
