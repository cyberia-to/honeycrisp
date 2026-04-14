//! BF16 (bfloat16) conversion routines.
//!
//! BF16 is the upper 16 bits of an IEEE 754 f32, giving the same exponent
//! range as f32 but only 8 bits of mantissa (7 explicit + 1 implicit).
//!
//! On aarch64 with BF16 ISA support, bulk f32->bf16 uses `bfcvtn`/`bfcvtn2`.
//! The reverse direction (bf16->f32) is always a simple shift.

/// Convert a single bf16 value (stored as `u16`) to `f32`.
///
/// This is a zero-cost bit shift: bf16 is the upper 16 bits of f32.
#[inline(always)]
pub fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

/// Convert a single `f32` to bf16 with round-to-nearest-even.
#[inline(always)]
pub fn f32_to_bf16(v: f32) -> u16 {
    f32_to_bf16_rne(v)
}

/// Bulk convert bf16 to f32.
///
/// Since bf16->f32 is just a shift, the NEON path uses bit-shift instructions
/// to process 4 elements at a time via `ushll`.
pub fn cast_bf16_f32(dst: &mut [f32], src: &[u16]) {
    let n = dst.len().min(src.len());

    #[cfg(target_arch = "aarch64")]
    {
        let mut i = 0;

        // 16 elements per iteration (4x unrolled)
        while i + 16 <= n {
            unsafe {
                let s = src.as_ptr().add(i);
                let d = dst.as_mut_ptr().add(i);
                std::arch::asm!(
                    "ldp q0, q1, [{s}]",         // 16 x u16
                    "shll v2.4s, v0.4h, #16",    // low 4 of first 8
                    "shll2 v3.4s, v0.8h, #16",   // high 4 of first 8
                    "shll v4.4s, v1.4h, #16",    // low 4 of second 8
                    "shll2 v5.4s, v1.8h, #16",   // high 4 of second 8
                    "stp q2, q3, [{d}]",
                    "stp q4, q5, [{d}, #32]",
                    s = in(reg) s,
                    d = in(reg) d,
                    out("v0") _, out("v1") _,
                    out("v2") _, out("v3") _, out("v4") _, out("v5") _,
                );
            }
            i += 16;
        }

        // Tail: 4 at a time
        while i + 4 <= n {
            unsafe {
                std::arch::asm!(
                    "ldr d0, [{s}]",             // 4 x u16
                    "shll v1.4s, v0.4h, #16",
                    "str q1, [{d}]",
                    s = in(reg) src.as_ptr().add(i),
                    d = in(reg) dst.as_mut_ptr().add(i),
                    out("v0") _, out("v1") _,
                );
            }
            i += 4;
        }

        for j in i..n {
            dst[j] = bf16_to_f32(src[j]);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            dst[i] = bf16_to_f32(src[i]);
        }
    }
}

/// Bulk convert f32 to bf16 with round-to-nearest-even.
///
/// On aarch64 with BF16 extension, this uses `bfcvtn`/`bfcvtn2` to narrow
/// 8 f32 values into 8 bf16 packed in a NEON register. When the BF16
/// instruction is not available, falls back to the scalar RNE path.
pub fn cast_f32_bf16(dst: &mut [u16], src: &[f32]) {
    let n = dst.len().min(src.len());

    // Note: `bfcvtn` requires FEAT_BF16 (Apple M1+, Cortex-A510+).
    // We gate on target_feature at compile time. For a runtime check
    // you would use std::arch::is_aarch64_feature_detected!("bf16").
    #[cfg(all(target_arch = "aarch64", target_feature = "bf16"))]
    {
        let mut i = 0;

        // 16 elements per iteration
        while i + 16 <= n {
            unsafe {
                let s = src.as_ptr().add(i);
                let d = dst.as_mut_ptr().add(i);
                std::arch::asm!(
                    "ldp q0, q1, [{s}]",          // 8 f32
                    "ldp q2, q3, [{s}, #32]",     // 8 f32
                    ".inst 0x0ea16800",            // bfcvtn v0.4h, v0.4s  (encoding)
                    ".inst 0x4ea16820",            // bfcvtn2 v0.8h, v1.4s
                    ".inst 0x0ea16842",            // bfcvtn v2.4h, v2.4s
                    ".inst 0x4ea16862",            // bfcvtn2 v2.8h, v3.4s
                    "stp q0, q2, [{d}]",
                    s = in(reg) s,
                    d = in(reg) d,
                    out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                );
            }
            i += 16;
        }

        for j in i..n {
            dst[j] = f32_to_bf16_rne(src[j]);
        }
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "bf16")))]
    {
        for i in 0..n {
            dst[i] = f32_to_bf16_rne(src[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Round-to-nearest-even for f32 -> bf16
// ---------------------------------------------------------------------------

/// Software round-to-nearest-even bf16 conversion.
///
/// Adds a rounding bias that depends on the LSB of the bf16 result to
/// implement banker's rounding, then truncates.
fn f32_to_bf16_rne(v: f32) -> u16 {
    let bits = v.to_bits();

    // NaN: preserve sign + set quiet bit, keep some payload
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16; // quiet NaN
    }

    // Round-to-nearest-even: add rounding bias
    // The bit at position 16 of the f32 is the LSB of the bf16 mantissa.
    // We add 0x7FFF + bit[16] to round to nearest even.
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    let rounded = bits.wrapping_add(rounding_bias);
    (rounded >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_bf16() {
        let values = [
            0.0f32,
            1.0,
            -1.0,
            3.14,
            -100.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        for &v in &values {
            let h = f32_to_bf16(v);
            let back = bf16_to_f32(h);
            if v.is_finite() {
                assert!(
                    (back - v).abs() / v.abs().max(1.0) < 0.01,
                    "roundtrip failed for {v}: got {back}"
                );
            } else {
                assert_eq!(back, v);
            }
        }
    }

    #[test]
    fn bulk_bf16() {
        let src: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.5).collect();
        let mut bf = vec![0u16; 64];
        let mut dst = vec![0.0f32; 64];
        cast_f32_bf16(&mut bf, &src);
        cast_bf16_f32(&mut dst, &bf);
        for i in 0..64 {
            assert!(
                (dst[i] - src[i]).abs() < 0.5,
                "mismatch at {i}: expected {}, got {}",
                src[i],
                dst[i]
            );
        }
    }

    #[test]
    fn rne_rounding() {
        // 1.0 in f32 = 0x3F80_0000, bf16 = 0x3F80 (exact)
        assert_eq!(f32_to_bf16(1.0), 0x3F80);
        // Verify zero
        assert_eq!(f32_to_bf16(0.0), 0x0000);
    }
}
