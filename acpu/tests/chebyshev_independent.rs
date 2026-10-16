//! Independent analytic heat controls; no production coefficient/recurrence
//! values are used to construct the expected matrix exponential.
use acpu::chebyshev_matvec;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[test]
fn weighted_four_cycle_matches_its_exact_spectral_heat_kernel() {
    // A four-cycle with weight2 has degree4 and exactly representable D^-1/2.
    // Its normalized Laplacian has Fourier eigenvalues [0,1,2,1].
    let rows = [0, 2, 4, 6, 8];
    let columns = [1, 3, 0, 2, 1, 3, 0, 2];
    let values = [2.0; 8];
    let diagonal = [0.5; 4];
    for x in [[1.0, -2.0, 3.0, 5.0], [1.0, -1.0, 1.0, -1.0], [3.0; 4]] {
        let x64 = x.map(f64::from);
        let mean = x64.iter().sum::<f64>() / 4.0;
        let alternating = (x64[0] - x64[1] + x64[2] - x64[3]) / 4.0;
        let scale = x64.iter().map(|x| x.abs()).sum::<f64>();
        for tau in [0.0_f32, 0.125, 1.0, 5.0, 100.0, 104.0, 1000.0, 10000.0] {
            let time = f64::from(tau);
            let order = (16.0 * (time + 1.0).sqrt()).ceil() as usize + 16;
            // Separate the truncation bound from the observed f32 tolerance.
            // Positive Bessel normalization and an exponential Chernoff bound
            // give an absolute omitted coefficient mass no larger than this.
            if tau != 0.0 {
                let n = (order + 1) as f64;
                let u = n / time;
                let bound = 2.0 * (n * (u / (1.0 + u.hypot(1.0)) - u.asinh())).exp();
                assert!(bound < 1e-10, "tau={tau}, order={order}, tail={bound}");
            }
            let mut output = [91.0; 4];
            let mut a = [92.0; 4];
            let mut b = [93.0; 4];
            chebyshev_matvec(
                &rows,
                &columns,
                &values,
                &diagonal,
                &x,
                &mut output,
                &mut a,
                &mut b,
                tau,
                order,
            );
            for i in 0..4 {
                let mode2 = if i % 2 == 0 {
                    alternating
                } else {
                    -alternating
                };
                let expected =
                    mean + mode2 * (-2.0 * time).exp() + (x64[i] - mean - mode2) * (-time).exp();
                let error = (f64::from(output[i]) - expected).abs();
                assert!(output[i].is_finite());
                assert!(
                    error <= 4e-6 * scale,
                    "tau={tau}, x={x:?}, i={i}, error={error}"
                );
            }
        }
    }
}

#[test]
fn numerical_refusals_preserve_all_output_and_scratch_bits() {
    for (tau, order) in [
        (f32::NAN, 12),
        (f32::INFINITY, 12),
        (-1.0, 12),
        (f32::MAX, 12),
        (1.0, usize::MAX),
        (1.0, 65_536),
    ] {
        let original = [f32::from_bits(0x7fc0_1234), -0.0, 17.0, -19.0];
        let mut output = original;
        let mut a = original;
        let mut b = original;
        let result = catch_unwind(AssertUnwindSafe(|| {
            chebyshev_matvec(
                &[0, 2, 4, 6, 8],
                &[1, 3, 0, 2, 1, 3, 0, 2],
                &[2.0; 8],
                &[0.5; 4],
                &[1.0; 4],
                &mut output,
                &mut a,
                &mut b,
                tau,
                order,
            );
        }));
        assert!(result.is_err(), "tau={tau}, order={order}");
        assert_eq!(output.map(f32::to_bits), original.map(f32::to_bits));
        assert_eq!(a.map(f32::to_bits), original.map(f32::to_bits));
        assert_eq!(b.map(f32::to_bits), original.map(f32::to_bits));
    }
}
