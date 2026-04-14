//! Quantization operations: f32 <-> int8 with scale/zero-point.
//!
//! Uses NEON vectorized paths on aarch64 for bulk conversions.

/// Quantize f32 to int8: `dst[i] = clamp(round(src[i] / scale), -128, 127)`.
///
/// `scale` is the quantization step size. Typically computed as
/// `(max - min) / 255` for asymmetric or `max(abs) / 127` for symmetric.
pub fn cast_f32_i8(dst: &mut [i8], src: &[f32], scale: f32) {
    let n = dst.len().min(src.len());
    let inv_scale = 1.0 / scale;

    #[cfg(target_arch = "aarch64")]
    {
        let mut i = 0;

        // Process 16 f32 -> 16 i8 per iteration
        while i + 16 <= n {
            unsafe {
                let s = src.as_ptr().add(i);
                let d = dst.as_mut_ptr().add(i);
                std::arch::asm!(
                    // Broadcast inv_scale into v31
                    "dup v31.4s, {inv:w}",
                    // Load 16 x f32
                    "ldp q0, q1, [{s}]",
                    "ldp q2, q3, [{s}, #32]",
                    // Multiply by inv_scale
                    "fmul v0.4s, v0.4s, v31.4s",
                    "fmul v1.4s, v1.4s, v31.4s",
                    "fmul v2.4s, v2.4s, v31.4s",
                    "fmul v3.4s, v3.4s, v31.4s",
                    // Round to nearest integer (f32)
                    "frintn v0.4s, v0.4s",
                    "frintn v1.4s, v1.4s",
                    "frintn v2.4s, v2.4s",
                    "frintn v3.4s, v3.4s",
                    // Convert to i32
                    "fcvtzs v0.4s, v0.4s",
                    "fcvtzs v1.4s, v1.4s",
                    "fcvtzs v2.4s, v2.4s",
                    "fcvtzs v3.4s, v3.4s",
                    // Narrow i32 -> i16 (saturating)
                    "sqxtn v0.4h, v0.4s",
                    "sqxtn2 v0.8h, v1.4s",
                    "sqxtn v2.4h, v2.4s",
                    "sqxtn2 v2.8h, v3.4s",
                    // Narrow i16 -> i8 (saturating)
                    "sqxtn v0.8b, v0.8h",
                    "sqxtn2 v0.16b, v2.8h",
                    // Store 16 x i8
                    "str q0, [{d}]",
                    inv = in(reg) inv_scale.to_bits(),
                    s = in(reg) s,
                    d = in(reg) d,
                    out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                    out("v31") _,
                );
            }
            i += 16;
        }

        // Scalar remainder
        for j in i..n {
            let q = (src[j] * inv_scale).round();
            dst[j] = q.clamp(-128.0, 127.0) as i8;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            let q = (src[i] * inv_scale).round();
            dst[i] = q.clamp(-128.0, 127.0) as i8;
        }
    }
}

/// Dequantize int8 to f32: `dst[i] = (src[i] - zero) * scale`.
///
/// `zero` is the zero-point offset. For symmetric quantization, `zero = 0`.
pub fn cast_i8_f32(dst: &mut [f32], src: &[i8], scale: f32, zero: i8) {
    let n = dst.len().min(src.len());

    #[cfg(target_arch = "aarch64")]
    {
        let mut i = 0;

        // Process 16 i8 -> 16 f32 per iteration
        while i + 16 <= n {
            unsafe {
                let s = src.as_ptr().add(i);
                let d = dst.as_mut_ptr().add(i);
                std::arch::asm!(
                    // Broadcast scale and zero
                    "dup v30.4s, {sc:w}",
                    "dup v31.16b, {z:w}",
                    // Load 16 x i8
                    "ldr q0, [{s}]",
                    // Subtract zero point (i8 saturating)
                    "sqsub v0.16b, v0.16b, v31.16b",
                    // Widen i8 -> i16
                    "sxtl v1.8h, v0.8b",
                    "sxtl2 v2.8h, v0.16b",
                    // Widen i16 -> i32
                    "sxtl v3.4s, v1.4h",
                    "sxtl2 v4.4s, v1.8h",
                    "sxtl v5.4s, v2.4h",
                    "sxtl2 v6.4s, v2.8h",
                    // Convert i32 -> f32
                    "scvtf v3.4s, v3.4s",
                    "scvtf v4.4s, v4.4s",
                    "scvtf v5.4s, v5.4s",
                    "scvtf v6.4s, v6.4s",
                    // Multiply by scale
                    "fmul v3.4s, v3.4s, v30.4s",
                    "fmul v4.4s, v4.4s, v30.4s",
                    "fmul v5.4s, v5.4s, v30.4s",
                    "fmul v6.4s, v6.4s, v30.4s",
                    // Store 16 x f32
                    "stp q3, q4, [{d}]",
                    "stp q5, q6, [{d}, #32]",
                    sc = in(reg) scale.to_bits(),
                    z = in(reg) zero as i32,
                    s = in(reg) s,
                    d = in(reg) d,
                    out("v0") _, out("v1") _, out("v2") _,
                    out("v3") _, out("v4") _, out("v5") _, out("v6") _,
                    out("v30") _, out("v31") _,
                );
            }
            i += 16;
        }

        // Scalar remainder
        for j in i..n {
            dst[j] = ((src[j] as i32 - zero as i32) as f32) * scale;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            dst[i] = ((src[i] as i32 - zero as i32) as f32) * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_dequantize_symmetric() {
        let src: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let scale = 0.1_f32 / 127.0 * 16.0; // rough scale
        let mut qi8 = vec![0i8; 32];
        let mut dst = vec![0.0f32; 32];

        cast_f32_i8(&mut qi8, &src, scale);
        cast_i8_f32(&mut dst, &qi8, scale, 0);

        for i in 0..32 {
            let err = (dst[i] - src[i]).abs();
            assert!(
                err < scale * 1.5, // within ~1 quantization step
                "mismatch at {i}: src={}, dst={}, err={}",
                src[i],
                dst[i],
                err,
            );
        }
    }

    #[test]
    fn clamp_overflow() {
        let src = [1000.0f32, -1000.0];
        let mut dst = [0i8; 2];
        cast_f32_i8(&mut dst, &src, 1.0);
        assert_eq!(dst[0], 127);
        assert_eq!(dst[1], -128);
    }
}
