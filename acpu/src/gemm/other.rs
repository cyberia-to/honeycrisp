//! Non-f32 GEMM variants: matmul_f16 (fp16), matmul_bf16 (bf16), matmul_i8 (int8).
//!
//! matmul_f16: AMX FMA16 accelerated (32×32 microkernel, 2048 FLOPs/instruction).
//! matmul_bf16, matmul_i8: scalar reference (AMX MAC16 planned).

/// Half-precision matrix multiply: C += A × B (fp16 in, fp32 accum).
///
/// Converts fp16→f32 then dispatches to AMX sgemm for full GEBP + parallel.
/// Direct FMA16 microkernel available in matrix::tile_f16 for future integration.
///
/// # Panics
/// Panics if slice lengths do not match dimensions.
pub fn matmul_f16(a: &[u16], b: &[u16], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    let mut a_f32 = vec![0f32; m * k];
    let mut b_f32 = vec![0f32; k * n];
    crate::cast_f16_f32(&mut a_f32, a);
    crate::cast_f16_f32(&mut b_f32, b);
    super::matmul_f32(&a_f32, &b_f32, c, m, n, k);
}

// Direct FMA16 microkernel exists in matrix::tile_f16 for future GEBP integration.
// Currently convert+sgemm is faster due to sgemm's cache blocking and parallelism.

/// BFloat16 matrix multiply: C += A × B (bf16 in, fp32 accum).
///
/// Converts bf16→f32 then dispatches to AMX sgemm. No native bf16 AMX on M1-M4.
pub fn matmul_bf16(a: &[u16], b: &[u16], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    // Convert bf16 → f32
    let mut a_f32 = vec![0f32; m * k];
    let mut b_f32 = vec![0f32; k * n];
    crate::cast_bf16_f32(&mut a_f32, a);
    crate::cast_bf16_f32(&mut b_f32, b);

    // Dispatch to AMX sgemm
    super::matmul_f32(&a_f32, &b_f32, c, m, n, k);
}

/// Int8 quantised matrix multiply: C += scale × (A - zero) × (B - zero).
///
/// Dequantizes i8→f32, then dispatches to AMX sgemm. MAC16 i16 path planned.
#[allow(clippy::too_many_arguments)]
pub fn matmul_i8(
    a: &[i8],
    b: &[i8],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    scale: f32,
    zero: i8,
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    let z = zero as f32;
    let mut a_f32 = vec![0f32; m * k];
    let mut b_f32 = vec![0f32; k * n];
    for i in 0..a.len() {
        a_f32[i] = (a[i] as f32 - z) * scale.sqrt();
    }
    for i in 0..b.len() {
        b_f32[i] = (b[i] as f32 - z) * scale.sqrt();
    }
    super::matmul_f32(&a_f32, &b_f32, c, m, n, k);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_i8_basic() {
        let a: Vec<i8> = vec![1, 2, 3, 4];
        let b: Vec<i8> = vec![1, 0, 0, 1];
        let mut c = vec![0.0f32; 4];
        matmul_i8(&a, &b, &mut c, 2, 2, 2, 1.0, 0);
        assert!((c[0] - 1.0).abs() < 1e-5);
        assert!((c[1] - 2.0).abs() < 1e-5);
        assert!((c[2] - 3.0).abs() < 1e-5);
        assert!((c[3] - 4.0).abs() < 1e-5);
    }
}
