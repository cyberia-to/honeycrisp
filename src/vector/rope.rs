// ---------------------------------------------------------------------------
// Rotary Positional Embedding (RoPE) -- NEON fast-path + scalar fallback
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// Apply rotary positional embedding.
///
/// `x` contains pairs of values (x0, x1) for each dimension pair.
/// `freqs` contains the base frequencies for each pair (length = x.len() / 2).
/// `pos` is the token position.
///
/// For each pair (x0, x1) with frequency f:
///   theta = pos * f
///   out[2*i]   = x0 * cos(theta) - x1 * sin(theta)
///   out[2*i+1] = x0 * sin(theta) + x1 * cos(theta)
pub fn rotate(out: &mut [f32], x: &[f32], freqs: &[f32], pos: usize) {
    let dim = x.len();
    assert!(dim % 2 == 0, "rotate: dimension must be even");
    let n_pairs = dim / 2;
    assert!(freqs.len() >= n_pairs, "rotate: not enough frequencies");
    assert!(out.len() >= dim, "rotate: output buffer too small");

    let pos_f = pos as f32;
    let mut i = 0; // pair index

    #[cfg(target_arch = "aarch64")]
    {
        // Process 2 pairs at a time (= 4 f32 values from x)
        unsafe {
            while i + 2 <= n_pairs {
                // Load two frequencies
                let f0 = freqs[i];
                let f1 = freqs[i + 1];
                let theta0 = pos_f * f0;
                let theta1 = pos_f * f1;

                let cos0 = theta0.cos();
                let sin0 = theta0.sin();
                let cos1 = theta1.cos();
                let sin1 = theta1.sin();

                // Load 4 values: [x0_a, x1_a, x0_b, x1_b]
                let xv = vld1q_f32(x.as_ptr().add(i * 2));

                // Build cos/sin vectors matching the interleaved layout
                let cos_v = {
                    let arr: [f32; 4] = [cos0, cos0, cos1, cos1];
                    vld1q_f32(arr.as_ptr())
                };
                let sin_v = {
                    let arr: [f32; 4] = [sin0, sin0, sin1, sin1];
                    vld1q_f32(arr.as_ptr())
                };

                // Build the "swapped and negated" version:
                // [-x1_a, x0_a, -x1_b, x0_b]
                // We negate odd-indexed originals and swap pairs.
                let x_arr: [f32; 4] = [
                    vgetq_lane_f32::<0>(xv),
                    vgetq_lane_f32::<1>(xv),
                    vgetq_lane_f32::<2>(xv),
                    vgetq_lane_f32::<3>(xv),
                ];
                let swapped_arr: [f32; 4] = [-x_arr[1], x_arr[0], -x_arr[3], x_arr[2]];
                let swapped = vld1q_f32(swapped_arr.as_ptr());

                // result = x * cos + swapped * sin
                let result = vfmaq_f32(vmulq_f32(xv, cos_v), swapped, sin_v);
                vst1q_f32(out.as_mut_ptr().add(i * 2), result);

                i += 2;
            }
        }
    }

    // Scalar tail / fallback
    while i < n_pairs {
        let theta = pos_f * freqs[i];
        let (sin_t, cos_t) = theta.sin_cos();
        let x0 = x[2 * i];
        let x1 = x[2 * i + 1];
        out[2 * i] = x0 * cos_t - x1 * sin_t;
        out[2 * i + 1] = x0 * sin_t + x1 * cos_t;
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_identity_at_zero() {
        // At pos=0, theta=0, cos=1, sin=0, so output == input.
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let freqs = vec![0.1, 0.01, 0.001, 0.0001];
        let mut out = vec![0.0; 8];
        rotate(&mut out, &x, &freqs, 0);
        for i in 0..x.len() {
            assert!(
                (out[i] - x[i]).abs() < 1e-5,
                "mismatch at {}: {} vs {}",
                i,
                out[i],
                x[i]
            );
        }
    }

    #[test]
    fn test_rope_preserves_norm() {
        // RoPE is a rotation, so the norm of each pair should be preserved.
        let x = vec![3.0, 4.0, 1.0, 0.0];
        let freqs = vec![1.0, 2.0];
        let mut out = vec![0.0; 4];
        rotate(&mut out, &x, &freqs, 7);

        let norm_in_0 = (x[0] * x[0] + x[1] * x[1]).sqrt();
        let norm_out_0 = (out[0] * out[0] + out[1] * out[1]).sqrt();
        assert!(
            (norm_in_0 - norm_out_0).abs() < 1e-4,
            "norm not preserved: {} vs {}",
            norm_in_0,
            norm_out_0
        );

        let norm_in_1 = (x[2] * x[2] + x[3] * x[3]).sqrt();
        let norm_out_1 = (out[2] * out[2] + out[3] * out[3]).sqrt();
        assert!(
            (norm_in_1 - norm_out_1).abs() < 1e-4,
            "norm not preserved: {} vs {}",
            norm_in_1,
            norm_out_1
        );
    }

    #[test]
    fn test_rope_rotation() {
        // Verify explicit rotation for a single pair.
        let x = vec![1.0, 0.0];
        let freq = 1.0;
        let pos = 1;
        let theta = pos as f32 * freq;
        let mut out = vec![0.0; 2];
        rotate(&mut out, &x, &[freq], pos);
        assert!((out[0] - theta.cos()).abs() < 1e-5);
        assert!((out[1] - theta.sin()).abs() < 1e-5);
    }

    #[test]
    fn test_rope_odd_pairs() {
        // 3 pairs = 6 elements, tests scalar tail when NEON handles 2 pairs.
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let freqs = vec![0.5, 1.0, 1.5];
        let mut out = vec![0.0; 6];
        rotate(&mut out, &x, &freqs, 3);
        // Just verify norms are preserved
        for p in 0..3 {
            let ni = (x[2 * p] * x[2 * p] + x[2 * p + 1] * x[2 * p + 1]).sqrt();
            let no = (out[2 * p] * out[2 * p] + out[2 * p + 1] * out[2 * p + 1]).sqrt();
            assert!((ni - no).abs() < 1e-4, "pair {}: {} vs {}", p, ni, no);
        }
    }
}
