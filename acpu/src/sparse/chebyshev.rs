//! Chebyshev polynomial approximation of the graph heat kernel H_τ(L).
//!
//! H_τ(L) x ≈ Σ_{k=0}^{K} c_k(τ) · T_k(L̃) x
//!
//! where L̃ = (2L − λ_max I) / λ_max  maps eigenvalues to [-1,1]
//! (λ_max = 2 for the normalized Laplacian),
//! T_k are Chebyshev polynomials (T_0=1, T_1=x, T_{k+1}=2xT_k − T_{k-1}),
//! and c_k(τ) = exp(−τ) · I_k(τ) · (2 − δ_{k,0}) are coefficients from
//! the modified Bessel functions of the first kind.
//!
//! Reference: Shuman et al., "The Emerging Field of Signal Processing on Graphs",
//! IEEE Signal Processing Magazine 2013 — Algorithm 1.

use super::csr_matvec_set;

// ── Modified Bessel I_k ───────────────────────────────────────────────────────

/// Modified Bessel function I_k(τ) via Miller's backward recurrence.
///
/// Uses the identity: I_0(τ) + 2 * Σ_{j=1}^∞ I_j(τ) = exp(τ).
/// In Miller's method we compute un-normalized ratios f[j] ∝ I_j(τ) by
/// backward recurrence from k_max, then normalize so the above sum holds.
pub fn bessel_i(k: usize, tau: f32) -> f32 {
    if tau == 0.0 {
        return if k == 0 { 1.0 } else { 0.0 };
    }

    let tau_d = tau as f64;
    let k_max = (2 * k + 60).max(80);

    // f[j] will hold the un-normalized value proportional to I_j(τ).
    // Backward recurrence: I_{j-1} = (2j/τ) I_j + I_{j+1}.
    // We store only two consecutive values.
    let mut f_next = 0.0f64; // f_{j+1}
    let mut f_curr = 1.0f64; // f_j

    let mut f_k = 0.0f64;
    // Normalization: sum = f[0] + 2*(f[1] + f[2] + ... + f[k_max]).
    // We accumulate in the loop as we go from k_max down to 0.
    let mut norm_sum = 0.0f64;
    let mut scale_acc = 1.0f64; // tracks the accumulated rescaling

    // Rescaling for overflow prevention.
    let rescale = |f_next: &mut f64,
                   f_curr: &mut f64,
                   f_k: &mut f64,
                   norm_sum: &mut f64,
                   scale_acc: &mut f64| {
        let s = 1.0 / f_curr.abs().max(1e-300);
        *f_next *= s;
        *f_curr *= s;
        *f_k *= s;
        *norm_sum *= s;
        *scale_acc *= s;
    };

    for j in (0..k_max).rev() {
        // Backward recurrence step: f_{j} = (2(j+1)/τ) f_{j+1} + f_{j+2}
        let f_prev = (2.0 * (j + 1) as f64 / tau_d) * f_curr + f_next;
        f_next = f_curr;
        f_curr = f_prev;

        if f_curr.abs() > 1e150 {
            rescale(
                &mut f_next,
                &mut f_curr,
                &mut f_k,
                &mut norm_sum,
                &mut scale_acc,
            );
        }

        // Record the value for the requested index k.
        // Note: after computing f_prev (which is f_j), f_curr = f_j.
        if j == k {
            f_k = f_curr;
        }

        // Accumulate normalization: sum = f[0] + 2*(f[1] + f[2] + ...).
        if j == 0 {
            norm_sum += f_curr; // coefficient 1 for j=0
        } else {
            norm_sum += 2.0 * f_curr; // coefficient 2 for j≥1
        }
    }

    // real I_k = f_k * exp(τ) / norm_sum
    // (norm_sum in un-normalized units corresponds to exp(τ) in real units)
    if norm_sum.abs() < 1e-300 {
        return 0.0;
    }
    let result = f_k * tau_d.exp() / norm_sum;
    result as f32
}

// ── Chebyshev coefficients ────────────────────────────────────────────────────

/// Compute Chebyshev coefficients c_k(τ) for the heat kernel exp(−τ ℒ) on the
/// normalized Laplacian ℒ with λ_max = 2.
///
/// Using L̃ = ℒ − I (eigenvalues in [-1, 1]):
///   exp(−τ ℒ) = exp(−τ) exp(−τ L̃)
/// Chebyshev expansion of exp(−τ x) for x ∈ [−1,1]:
///   exp(−τ x) = I_0(τ) T_0(x) + 2 Σ_{k≥1} (−1)^k I_k(τ) T_k(x)
///
/// So c_0 = exp(−τ) · I_0(τ),  c_k = exp(−τ) · 2 · (−1)^k · I_k(τ) for k ≥ 1.
pub fn chebyshev_coeffs(tau: f32, order: usize) -> Vec<f32> {
    let exp_neg_tau = (-tau).exp();
    (0..=order)
        .map(|k| {
            let factor = if k == 0 { 1.0 } else { 2.0 };
            let sign = if k % 2 == 0 { 1.0f32 } else { -1.0 };
            exp_neg_tau * factor * sign * bessel_i(k, tau)
        })
        .collect()
}

// ── Chebyshev matvec ──────────────────────────────────────────────────────────

/// Apply the Chebyshev heat-kernel approximation H_τ(L) to vector x.
///
/// Writes result into y.  t0 and t1 are scratch buffers of length n.
///
/// The normalized-Laplacian heat kernel maps to [-1,1] via
///   L̃ = (2L − λ_max I) / λ_max,   λ_max = 2  for the normalized Laplacian.
/// Hence L̃ x = L x − x = (x − D^{-½}AD^{-½}x) − x = −D^{-½}AD^{-½}x.
///
/// # Arguments
/// * `row_ptr`, `col_idx`, `values` — CSR of the raw adjacency matrix A
/// * `d_inv_sqrt` — D^{-1/2} diagonal vector
/// * `x` — input vector (length n)
/// * `y` — output vector (length n), overwritten
/// * `t0`, `t1` — caller-provided scratch, each length n
/// * `tau` — heat-kernel time scale τ
/// * `order` — Chebyshev truncation order K (typically 20)
#[allow(clippy::too_many_arguments)]
pub fn chebyshev_matvec(
    row_ptr: &[u32],
    col_idx: &[u32],
    values: &[f32],
    d_inv_sqrt: &[f32],
    x: &[f32],
    y: &mut [f32],
    t0: &mut [f32],
    t1: &mut [f32],
    tau: f32,
    order: usize,
) {
    let n = x.len();
    debug_assert_eq!(y.len(), n);
    debug_assert_eq!(t0.len(), n);
    debug_assert_eq!(t1.len(), n);

    let coeffs = chebyshev_coeffs(tau, order);

    // One internal allocation for A·z scratch (avoids the aliasing problem).
    let mut az = vec![0.0f32; n];

    // T_0 = x
    t0.copy_from_slice(x);

    // T_1 = L̃ x = −D^{-½} A D^{-½} x
    // Step 1: az = D^{-½} x
    for i in 0..n {
        az[i] = d_inv_sqrt[i] * x[i];
    }
    // Step 2: t1 = A · az
    csr_matvec_set(row_ptr, col_idx, values, &az, t1);
    // Step 3: t1[i] = −d_inv_sqrt[i] * t1[i]
    for i in 0..n {
        t1[i] = -d_inv_sqrt[i] * t1[i];
    }

    // Accumulate y = c_0 T_0 + c_1 T_1
    let c1 = if order >= 1 { coeffs[1] } else { 0.0 };
    for i in 0..n {
        y[i] = coeffs[0] * t0[i] + c1 * t1[i];
    }

    if order < 2 {
        return;
    }

    // Chebyshev recurrence: T_k = 2 L̃ T_{k-1} − T_{k-2}
    //
    // We maintain two rolling buffers:
    //   even k → T_{k-2} is in t0, T_{k-1} is in t1, write T_k into t0
    //   odd  k → T_{k-2} is in t1, T_{k-1} is in t0, write T_k into t1
    //
    // (k starts at 2, so k==2 is even: T_0 in t0, T_1 in t1 → write T_2 into t0)

    for k in 2..=order {
        let ck = coeffs[k];
        if k % 2 == 0 {
            // T_{k-2} in t0, T_{k-1} in t1. Compute T_k into t0.
            // az = D^{-½} t1
            for i in 0..n {
                az[i] = d_inv_sqrt[i] * t1[i];
            }
            // az2 = A · az  (need a second scratch — reuse y momentarily)
            // But y holds our running accumulator! Save by NOT clobbering y.
            // Instead use t0 as az2 output: T_k[i] = 2*(−d_inv sqrt[i]*az2[i]) − t0[i]
            // We need t0 (T_{k-2}) to compute T_k, but we'd write into it simultaneously.
            // Solution: compute Az into az (reuse az as Az output by doing in two passes).
            // Pass 1: az = D^{-½} t1  (done above)
            // Pass 2: compute A·az into a fresh location (az itself is the input, need output).
            // Let's use a separate local vec:
            let mut az2 = vec![0.0f32; n];
            csr_matvec_set(row_ptr, col_idx, values, &az, &mut az2);
            // T_k[i] = −2*d_inv_sqrt[i]*az2[i] − t0[i]  (T_{k-2} is in t0)
            for i in 0..n {
                t0[i] = -2.0 * d_inv_sqrt[i] * az2[i] - t0[i];
            }
            // Accumulate y += c_k * T_k
            for i in 0..n {
                y[i] += ck * t0[i];
            }
        } else {
            // T_{k-2} in t1, T_{k-1} in t0. Compute T_k into t1.
            for i in 0..n {
                az[i] = d_inv_sqrt[i] * t0[i];
            }
            let mut az2 = vec![0.0f32; n];
            csr_matvec_set(row_ptr, col_idx, values, &az, &mut az2);
            for i in 0..n {
                t1[i] = -2.0 * d_inv_sqrt[i] * az2[i] - t1[i];
            }
            for i in 0..n {
                y[i] += ck * t1[i];
            }
        }
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bessel_i0_known_values() {
        // I_0(0) = 1
        assert!((bessel_i(0, 0.0) - 1.0).abs() < 1e-5, "I_0(0)");
        // I_0(1) ≈ 1.2660658777520082
        assert!(
            (bessel_i(0, 1.0) - 1.266_065_9).abs() < 1e-4,
            "I_0(1) = {}",
            bessel_i(0, 1.0)
        );
        // I_1(1) ≈ 0.5651591039924851
        assert!(
            (bessel_i(1, 1.0) - 0.565_159_1).abs() < 1e-4,
            "I_1(1) = {}",
            bessel_i(1, 1.0)
        );
        // I_0(5) ≈ 27.2399
        assert!(
            (bessel_i(0, 5.0) - 27.2399).abs() < 0.05,
            "I_0(5) = {}",
            bessel_i(0, 5.0)
        );
    }

    #[test]
    fn chebyshev_coeffs_at_x1() {
        // At x=1 (T_k(1) = 1), Σ c_k = exp(-τ) * exp(-τ·1) = exp(-2τ).
        // Proof: exp(-τ x)|_{x=1} = exp(-τ) = I_0 + 2Σ_{k≥1}(-1)^k I_k,
        // so Σ c_k = exp(-τ) * (I_0 - 2I_1 + 2I_2 - ...) = exp(-τ) * exp(-τ) = exp(-2τ).
        for tau in [0.5f32, 1.0, 3.0] {
            let coeffs = chebyshev_coeffs(tau, 60);
            let sum: f32 = coeffs.iter().sum();
            let expected = (-2.0 * tau).exp();
            assert!(
                (sum - expected).abs() < 0.01,
                "Σ c_k at tau={tau}: got {sum}, expected exp(-2τ) = {expected}"
            );
        }
    }

    #[test]
    fn chebyshev_matvec_path4_smooth() {
        // Path graph 0-1-2-3. Apply heat kernel to delta at node 0.
        // The Chebyshev approximation of exp(-τ ℒ) is a smoothing operator:
        //   - It is symmetric positive semi-definite (eigenvalues in [0,1]).
        //   - It spreads mass from highly-connected to less-connected nodes.
        //   - Total ℓ2 norm decreases: ‖y‖ ≤ ‖x‖.
        let row_ptr = vec![0u32, 1, 3, 5, 6];
        let col_idx = vec![1u32, 0, 2, 1, 3, 2];
        let values = vec![1.0f32; 6];
        let n = 4;

        // Degrees: [1, 2, 2, 1]
        let d_inv_sqrt = vec![1.0f32, 1.0 / 2.0f32.sqrt(), 1.0 / 2.0f32.sqrt(), 1.0];
        let x = vec![1.0f32, 0.0, 0.0, 0.0];
        let mut y = vec![0.0f32; n];
        let mut t0 = vec![0.0f32; n];
        let mut t1 = vec![0.0f32; n];

        chebyshev_matvec(
            &row_ptr,
            &col_idx,
            &values,
            &d_inv_sqrt,
            &x,
            &mut y,
            &mut t0,
            &mut t1,
            1.0,
            20,
        );

        // Mass should spread: node 1 gains.
        assert!(y[1] > 1e-4, "y[1] should receive diffused mass: {}", y[1]);

        // Node 0 should decrease from 1.0.
        assert!(y[0] < 0.99, "y[0] should decrease: {}", y[0]);
        assert!(y[0] > 0.0, "y[0] should remain non-negative: {}", y[0]);

        // ‖y‖_2 ≤ ‖x‖_2 = 1.0 (heat kernel contracts).
        let norm_y: f32 = y.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(norm_y <= 1.01, "‖y‖₂ = {norm_y} should be ≤ ‖x‖₂ = 1");
    }
}
