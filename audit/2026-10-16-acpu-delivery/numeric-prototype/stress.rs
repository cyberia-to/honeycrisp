//! Structural consistency/limit stress; these checks are not independent accuracy oracles.
mod recurrence;
use recurrence::{coefficients, Error, DEFAULT};

fn main() {
    let times = [
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        2.0_f32.powi(-20),
        0.01,
        0.1,
        0.125,
        0.5,
        1.0,
        3.0,
        10.0,
        80.0,
        100.0,
        104.0,
        500.0,
        700.0,
        1e3,
        1e4,
        1e5,
        1e6,
        1e7,
        1e8,
        1e9,
        2e9,
        4e9,
        1e10,
        1e12,
        f32::MAX,
    ];
    let orders = [0, 1, 2, 5, 20, 40, 80, 160, 320, 1024, 4096];
    let mut accepted = 0;
    let mut refused = 0;
    let mut max_fraction = 0.0_f64;
    let mut max_steps = 0;
    let mut max_rounds = 0;
    let mut max_gap = 0.0_f64;
    let mut max_tail = 0.0_f64;
    let mut max_depth = 0;
    println!("{{\"rows\":[");
    for (index, (tau, order)) in times
        .into_iter()
        .flat_map(|t| orders.into_iter().map(move |k| (t, k)))
        .enumerate()
    {
        if index != 0 {
            println!(",");
        }
        match coefficients(tau, order, DEFAULT) {
            Ok(result) => {
                accepted += 1;
                max_steps = max_steps.max(result.steps);
                max_depth = max_depth.max(result.depth);
                max_rounds = max_rounds.max(result.rounds);
                max_gap = max_gap.max(result.largest_gap);
                max_tail = max_tail.max(result.tail_estimate);
                let extended = coefficients(tau, order + 7, DEFAULT);
                let mut compared = false;
                if let Ok(other) = extended {
                    compared = true;
                    for (k, a) in result.coefficients.iter().enumerate() {
                        let b = other.coefficients[k];
                        let scale = f64::from(a.abs().max(b.abs())) * 2.0_f64.powi(-21)
                            + f64::from(f32::from_bits(1));
                        let fraction = (f64::from(*a) - f64::from(b)).abs() / scale;
                        max_fraction = max_fraction.max(fraction);
                        assert!(
                            fraction <= 1.0,
                            "cross-order tau={tau} order={order} k={k} a={a} b={b}"
                        );
                    }
                }
                for (k, c) in result.coefficients.iter().enumerate() {
                    assert!(c.is_finite() && c.abs() <= 1.0);
                    if *c != 0.0 {
                        assert_eq!(c.is_sign_negative(), k % 2 == 1);
                    }
                }
                let mass: f64 = result.coefficients.iter().map(|c| f64::from(c.abs())).sum();
                assert!(mass <= 1.0 + 2.0_f64.powi(-21));
                println!("{{\"tau\":{tau:e},\"order\":{order},\"status\":\"ok\",\"depth\":{},\"steps\":{},\"rounds\":{},\"cross_order_compared\":{compared}}}", result.depth, result.steps, result.rounds);
            }
            Err(error) => {
                assert_eq!(error, Error::WorkLimit, "tau={tau} order={order}");
                refused += 1;
                println!("{{\"tau\":{tau:e},\"order\":{order},\"status\":\"WorkLimit\"}}");
            }
        }
    }
    println!("],\"accepted\":{accepted},\"refused\":{refused},\"max_cross_order_fraction\":{max_fraction:.17e},\"max_steps\":{max_steps},\"max_depth\":{max_depth},\"max_rounds\":{max_rounds},\"max_gap_fraction\":{max_gap:.17e},\"max_tail_estimate\":{max_tail:.17e}}}");
}
