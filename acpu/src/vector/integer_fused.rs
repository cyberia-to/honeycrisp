//! Fused integer operations — single-pass compute+reduce kernels.
//!
//! These combine multiple operations (sub+abs+sum, sub+square+sum, scale+accumulate)
//! into a single pass, saving memory traffic vs separate operations.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// Sum of absolute differences: sum(|a[i] - b[i]|).
/// Single pass — reads both arrays once, no intermediate storage.
/// Uses UABD + UADALP chain (unsigned absolute difference + pairwise add long).
pub fn sad_u8(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len(), "sad_u8: length mismatch");
    let len = a.len();
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let mut s0 = vdupq_n_u64(0);
            let mut s1 = vdupq_n_u64(0);
            let pa = a.as_ptr();
            let pb = b.as_ptr();
            while i + 32 <= len {
                let d0 = vabdq_u8(vld1q_u8(pa.add(i)), vld1q_u8(pb.add(i)));
                let d1 = vabdq_u8(vld1q_u8(pa.add(i + 16)), vld1q_u8(pb.add(i + 16)));
                let w0 = vpaddlq_u8(d0);
                let w1 = vpaddlq_u8(d1);
                let x0 = vpaddlq_u16(w0);
                let x1 = vpaddlq_u16(w1);
                s0 = vpadalq_u32(s0, x0);
                s1 = vpadalq_u32(s1, x1);
                i += 32;
            }
            s0 = vaddq_u64(s0, s1);
            while i + 16 <= len {
                let d = vabdq_u8(vld1q_u8(pa.add(i)), vld1q_u8(pb.add(i)));
                s0 = vpadalq_u32(s0, vpaddlq_u16(vpaddlq_u8(d)));
                i += 16;
            }
            let mut s = vaddvq_u64(s0);
            while i < len {
                s += (a[i] as i16 - b[i] as i16).unsigned_abs() as u64;
                i += 1;
            }
            return s;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut s = 0u64;
        for j in 0..len {
            s += (a[j] as i16 - b[j] as i16).unsigned_abs() as u64;
        }
        s
    }
}

/// Sum of squared differences: sum((a[i] - b[i])²) for i32 slices.
/// Single pass — no intermediate array. Widens to i64 to avoid overflow.
pub fn ssd_i32(a: &[i32], b: &[i32]) -> i64 {
    assert_eq!(a.len(), b.len(), "ssd_i32: length mismatch");
    let len = a.len();
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let mut s0 = vdupq_n_s64(0);
            let mut s1 = vdupq_n_s64(0);
            let pa = a.as_ptr();
            let pb = b.as_ptr();
            while i + 8 <= len {
                let d0 = vsubq_s32(vld1q_s32(pa.add(i)), vld1q_s32(pb.add(i)));
                let d1 = vsubq_s32(vld1q_s32(pa.add(i + 4)), vld1q_s32(pb.add(i + 4)));
                s0 = vmlal_s32(s0, vget_low_s32(d0), vget_low_s32(d0));
                s0 = vmlal_high_s32(s0, d0, d0);
                s1 = vmlal_s32(s1, vget_low_s32(d1), vget_low_s32(d1));
                s1 = vmlal_high_s32(s1, d1, d1);
                i += 8;
            }
            s0 = vaddq_s64(s0, s1);
            while i + 4 <= len {
                let d = vsubq_s32(vld1q_s32(pa.add(i)), vld1q_s32(pb.add(i)));
                s0 = vmlal_s32(s0, vget_low_s32(d), vget_low_s32(d));
                s0 = vmlal_high_s32(s0, d, d);
                i += 4;
            }
            let mut s = vaddvq_s64(s0);
            while i < len {
                let d = a[i] as i64 - b[i] as i64;
                s += d * d;
                i += 1;
            }
            return s;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut s = 0i64;
        for j in 0..len {
            let d = a[j] as i64 - b[j] as i64;
            s += d * d;
        }
        s
    }
}

/// Scale and accumulate: acc[i] += a[i] * scale.
/// Single pass with widening: i16 × i16 → i32 accumulated.
pub fn scale_acc_i16(acc: &mut [i32], a: &[i16], scale: i16) {
    let len = acc.len().min(a.len());
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let sv = vdup_n_s16(scale);
            let sq = vdupq_n_s16(scale);
            let pa = a.as_ptr();
            let pc = acc.as_mut_ptr();
            while i + 16 <= len {
                let a0 = vld1q_s16(pa.add(i));
                let a1 = vld1q_s16(pa.add(i + 8));
                let mut c0 = vld1q_s32(pc.add(i));
                let mut c1 = vld1q_s32(pc.add(i + 4));
                let mut c2 = vld1q_s32(pc.add(i + 8));
                let mut c3 = vld1q_s32(pc.add(i + 12));
                c0 = vmlal_s16(c0, vget_low_s16(a0), sv);
                c1 = vmlal_high_s16(c1, a0, sq);
                c2 = vmlal_s16(c2, vget_low_s16(a1), sv);
                c3 = vmlal_high_s16(c3, a1, sq);
                vst1q_s32(pc.add(i), c0);
                vst1q_s32(pc.add(i + 4), c1);
                vst1q_s32(pc.add(i + 8), c2);
                vst1q_s32(pc.add(i + 12), c3);
                i += 16;
            }
        }
    }

    while i < len {
        acc[i] += a[i] as i32 * scale as i32;
        i += 1;
    }
}

/// Fused sum of absolute values: sum(|x[i]|) for i8.
/// Combines abs + widen + accumulate in one pass.
pub fn sum_abs_i8(x: &[i8]) -> u64 {
    let len = x.len();
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let mut s0 = vdupq_n_u64(0);
            let mut s1 = vdupq_n_u64(0);
            let p = x.as_ptr() as *const u8;
            while i + 32 <= len {
                let v0 = vreinterpretq_u8_s8(vabsq_s8(vld1q_s8(p.add(i) as *const i8)));
                let v1 = vreinterpretq_u8_s8(vabsq_s8(vld1q_s8(p.add(i + 16) as *const i8)));
                s0 = vpadalq_u32(s0, vpaddlq_u16(vpaddlq_u8(v0)));
                s1 = vpadalq_u32(s1, vpaddlq_u16(vpaddlq_u8(v1)));
                i += 32;
            }
            s0 = vaddq_u64(s0, s1);
            while i + 16 <= len {
                let v = vreinterpretq_u8_s8(vabsq_s8(vld1q_s8(p.add(i) as *const i8)));
                s0 = vpadalq_u32(s0, vpaddlq_u16(vpaddlq_u8(v)));
                i += 16;
            }
            let mut s = vaddvq_u64(s0);
            while i < len {
                s += x[i].unsigned_abs() as u64;
                i += 1;
            }
            return s;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        x.iter().map(|v| v.unsigned_abs() as u64).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sad_u8() {
        let a: Vec<u8> = (0..256).map(|i| (i % 200) as u8).collect();
        let b: Vec<u8> = (0..256).map(|i| ((i + 50) % 200) as u8).collect();
        let expected: u64 = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u64)
            .sum();
        assert_eq!(sad_u8(&a, &b), expected);
    }

    #[test]
    fn test_ssd_i32() {
        let a: Vec<i32> = (0..64).map(|i| i * 10).collect();
        let b: Vec<i32> = (0..64).map(|i| i * 10 + 3).collect();
        let expected: i64 = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| {
                let d = x as i64 - y as i64;
                d * d
            })
            .sum();
        assert_eq!(ssd_i32(&a, &b), expected);
    }

    #[test]
    fn test_scale_acc_i16() {
        let a: Vec<i16> = (0..32).map(|i| (i * 5 - 80) as i16).collect();
        let mut acc = vec![100i32; 32];
        let mut expected = vec![100i32; 32];
        let scale: i16 = 7;
        for i in 0..32 {
            expected[i] += a[i] as i32 * scale as i32;
        }
        scale_acc_i16(&mut acc, &a, scale);
        assert_eq!(acc, expected);
    }

    #[test]
    fn test_sum_abs_i8() {
        let v: Vec<i8> = vec![3, -7, 2, -1, 5, -127];
        let expected: u64 = v.iter().map(|x| x.unsigned_abs() as u64).sum();
        assert_eq!(sum_abs_i8(&v), expected);
    }
}
