//! FP16 (IEEE 754 half-precision) conversion and arithmetic.
//!
//! Uses inline NEON assembly on aarch64 for single-element and bulk
//! conversions, with a portable software fallback for other targets.

/// Convert a single fp16 value (stored as `u16`) to `f32`.
///
/// On aarch64 this emits scalar `fcvt s, h` via inline asm.
#[inline(always)]
pub fn fp16_to_f32(v: u16) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        let result: u32;
        unsafe {
            std::arch::asm!(
                "fmov h0, {src:w}",
                "fcvt s0, h0",
                "fmov {dst:w}, s0",
                src = in(reg) v as u32,
                dst = out(reg) result,
                out("v0") _,
            );
        }
        f32::from_bits(result)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        fp16_to_f32_soft(v)
    }
}

/// Convert a single `f32` value to fp16 (stored as `u16`).
///
/// On aarch64 this emits scalar `fcvt h, s` via inline asm.
#[inline(always)]
pub fn f32_to_fp16(v: f32) -> u16 {
    #[cfg(target_arch = "aarch64")]
    {
        let result: u32;
        unsafe {
            std::arch::asm!(
                "fmov s0, {src:w}",
                "fcvt h0, s0",
                "fmov {dst:w}, s0",
                src = in(reg) v.to_bits(),
                dst = out(reg) result,
                out("v0") _,
            );
        }
        result as u16
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        f32_to_fp16_soft(v)
    }
}

/// Bulk convert fp16 to f32.
///
/// On aarch64 the hot path processes 32 elements per iteration (4x unrolled
/// NEON `fcvtl`/`fcvtl2`), with an 8-element tail loop and scalar remainder.
pub fn cast_f16_f32(dst: &mut [f32], src: &[u16]) {
    let n = dst.len().min(src.len());

    #[cfg(target_arch = "aarch64")]
    {
        let mut i = 0;

        // 4x unrolled: 32 fp16 -> 32 f32 per iteration
        while i + 32 <= n {
            unsafe {
                let s = src.as_ptr().add(i);
                let d = dst.as_mut_ptr().add(i);
                std::arch::asm!(
                    "ldp q0, q1, [{s}]",        // load 16 fp16 (32 bytes)
                    "ldp q2, q3, [{s}, #32]",   // load 16 fp16 (32 bytes)
                    "fcvtl v4.4s, v0.4h",
                    "fcvtl2 v5.4s, v0.8h",
                    "fcvtl v6.4s, v1.4h",
                    "fcvtl2 v7.4s, v1.8h",
                    "fcvtl v16.4s, v2.4h",
                    "fcvtl2 v17.4s, v2.8h",
                    "fcvtl v18.4s, v3.4h",
                    "fcvtl2 v19.4s, v3.8h",
                    "stp q4, q5, [{d}]",        // store 8 f32
                    "stp q6, q7, [{d}, #32]",   // store 8 f32
                    "stp q16, q17, [{d}, #64]", // store 8 f32
                    "stp q18, q19, [{d}, #96]", // store 8 f32
                    s = in(reg) s,
                    d = in(reg) d,
                    out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                    out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                    out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                );
            }
            i += 32;
        }

        // Tail: 8 elements at a time
        while i + 8 <= n {
            unsafe {
                std::arch::asm!(
                    "ldr q0, [{src}]",
                    "fcvtl v1.4s, v0.4h",
                    "fcvtl2 v2.4s, v0.8h",
                    "stp q1, q2, [{dst}]",
                    src = in(reg) src.as_ptr().add(i),
                    dst = in(reg) dst.as_mut_ptr().add(i),
                    out("v0") _, out("v1") _, out("v2") _,
                );
            }
            i += 8;
        }

        // Scalar remainder
        for j in i..n {
            dst[j] = fp16_to_f32(src[j]);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            dst[i] = fp16_to_f32_soft(src[i]);
        }
    }
}

/// Bulk convert f32 to fp16.
///
/// On aarch64 the hot path processes 32 elements per iteration (4x unrolled
/// NEON `fcvtn`/`fcvtn2`), with an 8-element tail loop and scalar remainder.
pub fn cast_f32_f16(dst: &mut [u16], src: &[f32]) {
    let n = dst.len().min(src.len());

    #[cfg(target_arch = "aarch64")]
    {
        let mut i = 0;

        // 4x unrolled: 32 f32 -> 32 fp16 per iteration
        while i + 32 <= n {
            unsafe {
                let s = src.as_ptr().add(i);
                let d = dst.as_mut_ptr().add(i);
                std::arch::asm!(
                    "ldp q0, q1, [{s}]",        // 8 f32
                    "ldp q2, q3, [{s}, #32]",   // 8 f32
                    "ldp q4, q5, [{s}, #64]",   // 8 f32
                    "ldp q6, q7, [{s}, #96]",   // 8 f32
                    "fcvtn v0.4h, v0.4s",
                    "fcvtn2 v0.8h, v1.4s",
                    "fcvtn v2.4h, v2.4s",
                    "fcvtn2 v2.8h, v3.4s",
                    "fcvtn v4.4h, v4.4s",
                    "fcvtn2 v4.8h, v5.4s",
                    "fcvtn v6.4h, v6.4s",
                    "fcvtn2 v6.8h, v7.4s",
                    "stp q0, q2, [{d}]",        // 16 fp16
                    "stp q4, q6, [{d}, #32]",   // 16 fp16
                    s = in(reg) s,
                    d = in(reg) d,
                    out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                    out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                );
            }
            i += 32;
        }

        // Tail: 8 elements at a time
        while i + 8 <= n {
            unsafe {
                std::arch::asm!(
                    "ldp q0, q1, [{src}]",
                    "fcvtn v2.4h, v0.4s",
                    "fcvtn2 v2.8h, v1.4s",
                    "str q2, [{dst}]",
                    src = in(reg) src.as_ptr().add(i),
                    dst = in(reg) dst.as_mut_ptr().add(i),
                    out("v0") _, out("v1") _, out("v2") _,
                );
            }
            i += 8;
        }

        // Scalar remainder
        for j in i..n {
            dst[j] = f32_to_fp16(src[j]);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            dst[i] = f32_to_fp16_soft(src[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Software fallback (used on non-aarch64 and in tests)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn fp16_to_f32_soft(v: u16) -> f32 {
    let sign = (v >> 15) & 1;
    let exp = (v >> 10) & 0x1F;
    let frac = v & 0x3FF;
    let sign_f = if sign != 0 { -1.0f32 } else { 1.0f32 };

    if exp == 0 {
        // Subnormal or zero
        sign_f * (frac as f32 / 1024.0) * 2.0f32.powi(-14)
    } else if exp == 31 {
        if frac == 0 {
            sign_f * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign_f * (1.0 + frac as f32 / 1024.0) * 2.0f32.powi(exp as i32 - 15)
    }
}

#[allow(dead_code)]
fn f32_to_fp16_soft(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let frac = bits & 0x7FFFFF;

    // NaN / Inf passthrough
    if ((bits >> 23) & 0xFF) == 0xFF {
        if frac != 0 {
            return ((sign << 15) | 0x7C00 | (frac >> 13).max(1)) as u16;
        }
        return ((sign << 15) | 0x7C00) as u16;
    }

    if exp > 15 {
        // Overflow -> Inf
        ((sign << 15) | 0x7C00) as u16
    } else if exp < -14 {
        let shift = (12 - exp) as u32;
        if shift >= 24 {
            (sign << 15) as u16 // flush to zero
        } else {
            ((sign << 15) | ((0x800000 | frac) >> shift)) as u16
        }
    } else {
        ((sign << 15) | (((exp + 15) as u32) << 10) | (frac >> 13)) as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_normal() {
        let values = [0.0f32, 1.0, -1.0, 0.5, 65504.0, -65504.0];
        for &v in &values {
            let h = f32_to_fp16(v);
            let back = fp16_to_f32(h);
            assert!(
                (back - v).abs() < 1e-3 || (v == 0.0 && back == 0.0),
                "roundtrip failed for {v}: got {back}"
            );
        }
    }

    #[test]
    fn bulk_roundtrip() {
        let src: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.1).collect();
        let mut fp16 = vec![0u16; 100];
        let mut dst = vec![0.0f32; 100];
        cast_f32_f16(&mut fp16, &src);
        cast_f16_f32(&mut dst, &fp16);
        for i in 0..100 {
            assert!(
                (dst[i] - src[i]).abs() < 0.01,
                "mismatch at {i}: expected {}, got {}",
                src[i],
                dst[i]
            );
        }
    }
}
