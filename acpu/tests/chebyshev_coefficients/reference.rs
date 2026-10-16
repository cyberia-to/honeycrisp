//! Independent defining-series/quadrature controls; no backward recurrence.
use std::f64::consts::PI;

fn add(sum: &mut f64, correction: &mut f64, value: f64) {
    let term = value - *correction;
    let next = *sum + term;
    *correction = (next - *sum) - term;
    *sum = next;
}

// DLMF10.25.2; tau<=700 keeps the unscaled positive series inside f64.
pub fn series(tau: f64, order: usize) -> f64 {
    assert!((0.0..=700.0).contains(&tau));
    let mut term = 1.0;
    for j in 1..=order {
        term *= tau / (2.0 * j as f64);
    }
    let mut sum = term;
    let mut correction = 0.0;
    for j in 1..=10_000 {
        term *= (tau * 0.5) * (tau * 0.5) / (j as f64 * (j + order) as f64);
        add(&mut sum, &mut correction, term);
        if term <= sum * 2.0_f64.powi(-55) {
            return sum * (-tau).exp();
        }
    }
    panic!("positive series did not converge");
}

// DLMF10.32.3, u=sqrt(tau)*theta. The original root-owned oracle uses the
// same primary integral; this local control is separately hashed and executed.
fn integral(tau: f64, order: usize, panels: usize) -> (f64, f64) {
    let root = tau.sqrt();
    let upper = (PI * root).min(24.0);
    let h = upper / panels as f64;
    let mut sum = 0.0;
    let mut correction = 0.0;
    for j in 0..=panels {
        let u = j as f64 * h;
        let sine = (u / (2.0 * root)).sin();
        let weight = if j == 0 || j == panels {
            1.0
        } else if j % 2 == 0 {
            2.0
        } else {
            4.0
        };
        add(
            &mut sum,
            &mut correction,
            weight * (-2.0 * tau * sine * sine).exp() * (order as f64 * u / root).cos(),
        );
    }
    let alpha = 2.0 / (PI * PI);
    let tail = if upper < PI * root {
        (-alpha * upper * upper).exp() / (2.0 * alpha * upper * PI * root)
    } else {
        0.0
    };
    (sum * h / (3.0 * PI * root), tail)
}

pub fn quadrature(tau: f64, order: usize) -> (Option<f64>, f64) {
    let (coarse, _) = integral(tau, order, 8192);
    let (fine, tail) = integral(tau, order, 16384);
    let disagreement = (coarse - fine).abs();
    // A deliberately conservative numerical resolution screen, not a proof
    // of the libm/quadrature error: reject cancellation-floor estimates.
    let uncertainty = disagreement * 16.0 + tail + 2.0e-14 / tau.sqrt();
    let allowed = fine.abs() * 2.0_f64.powi(-25);
    (
        (fine > 0.0 && uncertainty <= allowed).then_some(fine),
        uncertainty,
    )
}
