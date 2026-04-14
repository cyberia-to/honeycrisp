//! Integer SIMD operations — NEON vectorized, stay in integer domain.
//!
//! Kernels for quantized inference: dot products, reductions, multiply-accumulate.
//! All use 8-accumulator unrolling for pipeline saturation.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

// ── i32 reductions ──────────────────────────────────────────────────────────

/// Sum of all i32 elements. 8-accumulator i32, widen to i64 at end.
pub fn sum_i32(x: &[i32]) -> i64 {
    let len = x.len();
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let z = vdupq_n_s32(0);
            let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
            let (mut a4, mut a5, mut a6, mut a7) = (z, z, z, z);
            let p = x.as_ptr();
            while i + 32 <= len {
                a0 = vaddq_s32(a0, vld1q_s32(p.add(i)));
                a1 = vaddq_s32(a1, vld1q_s32(p.add(i + 4)));
                a2 = vaddq_s32(a2, vld1q_s32(p.add(i + 8)));
                a3 = vaddq_s32(a3, vld1q_s32(p.add(i + 12)));
                a4 = vaddq_s32(a4, vld1q_s32(p.add(i + 16)));
                a5 = vaddq_s32(a5, vld1q_s32(p.add(i + 20)));
                a6 = vaddq_s32(a6, vld1q_s32(p.add(i + 24)));
                a7 = vaddq_s32(a7, vld1q_s32(p.add(i + 28)));
                i += 32;
            }
            a0 = vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3));
            a4 = vaddq_s32(vaddq_s32(a4, a5), vaddq_s32(a6, a7));
            a0 = vaddq_s32(a0, a4);
            while i + 4 <= len {
                a0 = vaddq_s32(a0, vld1q_s32(p.add(i)));
                i += 4;
            }
            // Widen to i64 only at the end
            let wide = vpaddlq_s32(a0);
            let mut s = vaddvq_s64(wide);
            while i < len {
                s += *p.add(i) as i64;
                i += 1;
            }
            s
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut s = 0i64;
        for &v in x {
            s += v as i64;
        }
        s
    }
}

/// Maximum i32 element. 8-accumulator, 32-wide.
pub fn max_i32(x: &[i32]) -> i32 {
    if x.is_empty() {
        return i32::MIN;
    }

    #[cfg(target_arch = "aarch64")]
    {
        let mut i = 0;
        unsafe {
            let init = vdupq_n_s32(i32::MIN);
            let (mut a0, mut a1, mut a2, mut a3) = (init, init, init, init);
            let (mut a4, mut a5, mut a6, mut a7) = (init, init, init, init);
            let p = x.as_ptr();
            while i + 32 <= x.len() {
                a0 = vmaxq_s32(a0, vld1q_s32(p.add(i)));
                a1 = vmaxq_s32(a1, vld1q_s32(p.add(i + 4)));
                a2 = vmaxq_s32(a2, vld1q_s32(p.add(i + 8)));
                a3 = vmaxq_s32(a3, vld1q_s32(p.add(i + 12)));
                a4 = vmaxq_s32(a4, vld1q_s32(p.add(i + 16)));
                a5 = vmaxq_s32(a5, vld1q_s32(p.add(i + 20)));
                a6 = vmaxq_s32(a6, vld1q_s32(p.add(i + 24)));
                a7 = vmaxq_s32(a7, vld1q_s32(p.add(i + 28)));
                i += 32;
            }
            a0 = vmaxq_s32(vmaxq_s32(a0, a1), vmaxq_s32(a2, a3));
            a4 = vmaxq_s32(vmaxq_s32(a4, a5), vmaxq_s32(a6, a7));
            a0 = vmaxq_s32(a0, a4);
            while i + 4 <= x.len() {
                a0 = vmaxq_s32(a0, vld1q_s32(p.add(i)));
                i += 4;
            }
            let mut m = vmaxvq_s32(a0);
            while i < x.len() {
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
    x.iter().copied().max().unwrap_or(i32::MIN)
}

/// Minimum i32 element. 8-accumulator, 32-wide.
pub fn min_i32(x: &[i32]) -> i32 {
    if x.is_empty() {
        return i32::MAX;
    }

    #[cfg(target_arch = "aarch64")]
    {
        let mut i = 0;
        unsafe {
            let init = vdupq_n_s32(i32::MAX);
            let (mut a0, mut a1, mut a2, mut a3) = (init, init, init, init);
            let (mut a4, mut a5, mut a6, mut a7) = (init, init, init, init);
            let p = x.as_ptr();
            while i + 32 <= x.len() {
                a0 = vminq_s32(a0, vld1q_s32(p.add(i)));
                a1 = vminq_s32(a1, vld1q_s32(p.add(i + 4)));
                a2 = vminq_s32(a2, vld1q_s32(p.add(i + 8)));
                a3 = vminq_s32(a3, vld1q_s32(p.add(i + 12)));
                a4 = vminq_s32(a4, vld1q_s32(p.add(i + 16)));
                a5 = vminq_s32(a5, vld1q_s32(p.add(i + 20)));
                a6 = vminq_s32(a6, vld1q_s32(p.add(i + 24)));
                a7 = vminq_s32(a7, vld1q_s32(p.add(i + 28)));
                i += 32;
            }
            a0 = vminq_s32(vminq_s32(a0, a1), vminq_s32(a2, a3));
            a4 = vminq_s32(vminq_s32(a4, a5), vminq_s32(a6, a7));
            a0 = vminq_s32(a0, a4);
            while i + 4 <= x.len() {
                a0 = vminq_s32(a0, vld1q_s32(p.add(i)));
                i += 4;
            }
            let mut m = vminvq_s32(a0);
            while i < x.len() {
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
    x.iter().copied().min().unwrap_or(i32::MAX)
}

// ── i8 dot product (SDOT — FEAT_DotProd) ────────────────────────────────────

/// Dot product of two i8 slices, returning i32 accumulator.
///
/// Uses SDOT instruction (4×i8 × 4×i8 → i32, 4 lanes per instruction).
/// Each SDOT computes 16 multiply-adds. 4× unroll = 64 elements per iteration.
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len(), "dot_i8: length mismatch");
    let len = a.len();
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let z = vdupq_n_s32(0);
            let (mut acc0, mut acc1, mut acc2, mut acc3) = (z, z, z, z);
            let pa = a.as_ptr() as *const u8;
            let pb = b.as_ptr() as *const u8;

            while i + 64 <= len {
                // SDOT: 4×(4×i8 × 4×i8) → 4×i32 accumulated
                core::arch::asm!(
                    "ldp q4, q5, [{pa}]",
                    "ldp q6, q7, [{pa}, #32]",
                    "ldp q8, q9, [{pb}]",
                    "ldp q10, q11, [{pb}, #32]",
                    ".inst 0x4e889480", // sdot v0.4s, v4.16b, v8.16b
                    ".inst 0x4e8994a1", // sdot v1.4s, v5.16b, v9.16b
                    ".inst 0x4e8a94c2", // sdot v2.4s, v6.16b, v10.16b
                    ".inst 0x4e8b94e3", // sdot v3.4s, v7.16b, v11.16b
                    pa = in(reg) pa.add(i),
                    pb = in(reg) pb.add(i),
                    inout("v0") acc0, inout("v1") acc1,
                    inout("v2") acc2, inout("v3") acc3,
                    out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                    out("v8") _, out("v9") _, out("v10") _, out("v11") _,
                );
                i += 64;
            }
            acc0 = vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));
            while i + 16 <= len {
                core::arch::asm!(
                    "ldr q4, [{pa}]",
                    "ldr q5, [{pb}]",
                    ".inst 0x4e859480", // sdot v0.4s, v4.16b, v5.16b
                    pa = in(reg) pa.add(i),
                    pb = in(reg) pb.add(i),
                    inout("v0") acc0,
                    out("v4") _, out("v5") _,
                );
                i += 16;
            }
            let mut s = vaddvq_s32(acc0);
            while i < len {
                s += a[i] as i32 * b[i] as i32;
                i += 1;
            }
            s
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut s = 0i32;
        for j in 0..len {
            s += a[j] as i32 * b[j] as i32;
        }
        s
    }
}

// ── i16 multiply-accumulate (widening) ──────────────────────────────────────

/// Multiply-accumulate i16 vectors into i32 accumulator: acc += a * b.
///
/// Uses SMLAL/SMLAL2 (signed widening multiply-add long).
/// Each instruction: 4×i16 × 4×i16 → 4×i32 accumulated.
pub fn macc_i16(acc: &mut [i32], a: &[i16], b: &[i16]) {
    assert_eq!(a.len(), b.len(), "macc_i16: a/b length mismatch");
    let len = a.len().min(acc.len());
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let pa = a.as_ptr();
            let pb = b.as_ptr();
            let pc = acc.as_mut_ptr();

            while i + 16 <= len {
                // Load 8 x i16 from a and b (two registers)
                let a_lo = vld1q_s16(pa.add(i));
                let a_hi = vld1q_s16(pa.add(i + 8));
                let b_lo = vld1q_s16(pb.add(i));
                let b_hi = vld1q_s16(pb.add(i + 8));

                // Load existing accumulators
                let mut c0 = vld1q_s32(pc.add(i));
                let mut c1 = vld1q_s32(pc.add(i + 4));
                let mut c2 = vld1q_s32(pc.add(i + 8));
                let mut c3 = vld1q_s32(pc.add(i + 12));

                // Widening multiply-accumulate
                c0 = vmlal_s16(c0, vget_low_s16(a_lo), vget_low_s16(b_lo));
                c1 = vmlal_high_s16(c1, a_lo, b_lo);
                c2 = vmlal_s16(c2, vget_low_s16(a_hi), vget_low_s16(b_hi));
                c3 = vmlal_high_s16(c3, a_hi, b_hi);

                vst1q_s32(pc.add(i), c0);
                vst1q_s32(pc.add(i + 4), c1);
                vst1q_s32(pc.add(i + 8), c2);
                vst1q_s32(pc.add(i + 12), c3);
                i += 16;
            }
        }
    }

    while i < len {
        acc[i] += a[i] as i32 * b[i] as i32;
        i += 1;
    }
}

// ── i32 elementwise ─────────────────────────────────────────────────────────

/// Elementwise i32 add: dst[i] = a[i] + b[i]. 16-wide.
pub fn add_i32(dst: &mut [i32], a: &[i32], b: &[i32]) {
    let len = dst.len().min(a.len()).min(b.len());
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let pa = a.as_ptr();
            let pb = b.as_ptr();
            let pd = dst.as_mut_ptr();
            while i + 16 <= len {
                let a0 = vld1q_s32(pa.add(i));
                let a1 = vld1q_s32(pa.add(i + 4));
                let a2 = vld1q_s32(pa.add(i + 8));
                let a3 = vld1q_s32(pa.add(i + 12));
                vst1q_s32(pd.add(i), vaddq_s32(a0, vld1q_s32(pb.add(i))));
                vst1q_s32(pd.add(i + 4), vaddq_s32(a1, vld1q_s32(pb.add(i + 4))));
                vst1q_s32(pd.add(i + 8), vaddq_s32(a2, vld1q_s32(pb.add(i + 8))));
                vst1q_s32(pd.add(i + 12), vaddq_s32(a3, vld1q_s32(pb.add(i + 12))));
                i += 16;
            }
        }
    }

    while i < len {
        dst[i] = a[i].wrapping_add(b[i]);
        i += 1;
    }
}

/// Elementwise i32 multiply: dst[i] = a[i] * b[i]. 32-wide.
pub fn mul_i32(dst: &mut [i32], a: &[i32], b: &[i32]) {
    let len = dst.len().min(a.len()).min(b.len());
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let pa = a.as_ptr();
            let pb = b.as_ptr();
            let pd = dst.as_mut_ptr();
            while i + 32 <= len {
                let a0 = vld1q_s32(pa.add(i));
                let a1 = vld1q_s32(pa.add(i + 4));
                let a2 = vld1q_s32(pa.add(i + 8));
                let a3 = vld1q_s32(pa.add(i + 12));
                let a4 = vld1q_s32(pa.add(i + 16));
                let a5 = vld1q_s32(pa.add(i + 20));
                let a6 = vld1q_s32(pa.add(i + 24));
                let a7 = vld1q_s32(pa.add(i + 28));
                vst1q_s32(pd.add(i), vmulq_s32(a0, vld1q_s32(pb.add(i))));
                vst1q_s32(pd.add(i + 4), vmulq_s32(a1, vld1q_s32(pb.add(i + 4))));
                vst1q_s32(pd.add(i + 8), vmulq_s32(a2, vld1q_s32(pb.add(i + 8))));
                vst1q_s32(pd.add(i + 12), vmulq_s32(a3, vld1q_s32(pb.add(i + 12))));
                vst1q_s32(pd.add(i + 16), vmulq_s32(a4, vld1q_s32(pb.add(i + 16))));
                vst1q_s32(pd.add(i + 20), vmulq_s32(a5, vld1q_s32(pb.add(i + 20))));
                vst1q_s32(pd.add(i + 24), vmulq_s32(a6, vld1q_s32(pb.add(i + 24))));
                vst1q_s32(pd.add(i + 28), vmulq_s32(a7, vld1q_s32(pb.add(i + 28))));
                i += 32;
            }
            while i + 4 <= len {
                vst1q_s32(
                    pd.add(i),
                    vmulq_s32(vld1q_s32(pa.add(i)), vld1q_s32(pb.add(i))),
                );
                i += 4;
            }
        }
    }

    while i < len {
        dst[i] = a[i].wrapping_mul(b[i]);
        i += 1;
    }
}

// ── i8 abs max (for quantization range finding) ─────────────────────────────

/// Find maximum absolute value in i8 slice. Used for symmetric quantization
/// scale computation.
pub fn absmax_i8(x: &[i8]) -> u8 {
    if x.is_empty() {
        return 0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        let mut i = 0;
        unsafe {
            let mut m0 = vdupq_n_u8(0);
            let mut m1 = vdupq_n_u8(0);
            let p = x.as_ptr() as *const u8;
            while i + 32 <= x.len() {
                let v0 = vabsq_s8(vld1q_s8(p.add(i) as *const i8));
                let v1 = vabsq_s8(vld1q_s8(p.add(i + 16) as *const i8));
                m0 = vmaxq_u8(m0, vreinterpretq_u8_s8(v0));
                m1 = vmaxq_u8(m1, vreinterpretq_u8_s8(v1));
                i += 32;
            }
            m0 = vmaxq_u8(m0, m1);
            while i + 16 <= x.len() {
                let v = vabsq_s8(vld1q_s8(p.add(i) as *const i8));
                m0 = vmaxq_u8(m0, vreinterpretq_u8_s8(v));
                i += 16;
            }
            let mut m = vmaxvq_u8(m0);
            while i < x.len() {
                let v = x[i].unsigned_abs();
                if v > m {
                    m = v;
                }
                i += 1;
            }
            m
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    x.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_i32() {
        let v: Vec<i32> = (0..4096).map(|i| (i % 100) as i32 - 50).collect();
        let expected: i64 = v.iter().map(|&x| x as i64).sum();
        assert_eq!(sum_i32(&v), expected);
    }

    #[test]
    fn test_max_min_i32() {
        let v = vec![3, -1, 4, 1, -5, 9, 2, 6, -8];
        assert_eq!(max_i32(&v), 9);
        assert_eq!(min_i32(&v), -8);
    }

    #[test]
    fn test_dot_i8() {
        let a: Vec<i8> = (0..128).map(|i| (i % 7 - 3) as i8).collect();
        let b: Vec<i8> = (0..128).map(|i| (i % 5 - 2) as i8).collect();
        let expected: i32 = a.iter().zip(&b).map(|(&x, &y)| x as i32 * y as i32).sum();
        assert_eq!(dot_i8(&a, &b), expected);
    }

    #[test]
    fn test_macc_i16() {
        let a: Vec<i16> = (0..64).map(|i| (i * 3 - 100) as i16).collect();
        let b: Vec<i16> = (0..64).map(|i| (i * 2 - 50) as i16).collect();
        let mut acc = vec![0i32; 64];
        let mut expected = vec![0i32; 64];
        for i in 0..64 {
            expected[i] = a[i] as i32 * b[i] as i32;
        }
        macc_i16(&mut acc, &a, &b);
        assert_eq!(acc, expected);
    }

    #[test]
    fn test_add_mul_i32() {
        let a: Vec<i32> = (0..32).map(|i| i * 3).collect();
        let b: Vec<i32> = (0..32).map(|i| i * 2 + 1).collect();
        let mut dst = vec![0i32; 32];
        add_i32(&mut dst, &a, &b);
        for i in 0..32 {
            assert_eq!(dst[i], a[i] + b[i]);
        }
        mul_i32(&mut dst, &a, &b);
        for i in 0..32 {
            assert_eq!(dst[i], a[i] * b[i]);
        }
    }

    #[test]
    fn test_absmax_i8() {
        let v: Vec<i8> = vec![3, -7, 2, -128, 5, 127];
        assert_eq!(absmax_i8(&v), 128);
    }
}
