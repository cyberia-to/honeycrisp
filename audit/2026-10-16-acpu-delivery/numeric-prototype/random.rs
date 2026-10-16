mod recurrence;
mod reference;
use recurrence::{coefficients, Error, Limits, DEFAULT};

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn main() {
    let mut seed = 0x319f_a27d_2e88_4c15;
    let mut cases = Vec::new();
    for _ in 0..256 {
        let bits = 1 + (next(&mut seed) % u64::from(700.0_f32.to_bits())) as u32;
        cases.push((
            f32::from_bits(bits),
            (next(&mut seed) % 513) as usize,
            DEFAULT,
        ));
    }
    for _ in 0..96 {
        let unit = (next(&mut seed) >> 11) as f64 / ((1_u64 << 53) as f64);
        let tau = (700.0_f64.ln() + unit * (4.0e9_f64 / 700.0).ln()).exp() as f32;
        cases.push((tau, (next(&mut seed) % 4097) as usize, DEFAULT));
    }
    for tau in [1e5, 1e6, 1e7, 1e8, 1e9, 4e9] {
        for order in [0, 1, 20, 320, 4096] {
            cases.push((tau, order, DEFAULT));
        }
    }
    for tau in [0.125, 3.0, 20.0, 100.0, 500.0, 700.0] {
        for order in 15..=340 {
            cases.push((tau, order, DEFAULT));
        }
    }
    cases.push((
        1e10,
        0,
        Limits {
            steps: 2_000_000,
            ..DEFAULT
        },
    ));
    let mut compared = 0;
    let mut series_cases = 0;
    let mut quadrature_cases = 0;
    let mut unresolved = 0;
    let mut refused = 0;
    let mut max_fraction = 0.0_f64;
    let mut subnormal_compared = 0;
    println!("{{\"rows\":[");
    for (index, (tau, order, limits)) in cases.iter().copied().enumerate() {
        if index != 0 {
            println!(",");
        }
        let output = match coefficients(tau, order, limits) {
            Ok(output) => output,
            Err(error) => {
                assert_eq!(error, Error::WorkLimit);
                refused += 1;
                println!("{{\"tau\":{tau:e},\"order\":{order},\"status\":\"WorkLimit\"}}");
                continue;
            }
        };
        // Read all output diagnostics so this independently compiled harness is warning-free.
        assert!(output.steps <= limits.steps && output.rounds > 0 && output.depth >= order);
        assert!(output.tail_estimate <= 2.0_f64.powi(-40) && output.largest_gap <= 1.0);
        let (q, method, uncertainty) = if tau <= 700.0 {
            series_cases += 1;
            (
                Some(reference::series(f64::from(tau), order)),
                "series",
                0.0,
            )
        } else {
            quadrature_cases += 1;
            let (q, uncertainty) = reference::quadrature(f64::from(tau), order);
            (q, "integral", uncertainty)
        };
        let value = f64::from(output.coefficients[order]);
        let reference = q.map(|q| {
            q * if order == 0 {
                1.0
            } else if order % 2 == 0 {
                2.0
            } else {
                -2.0
            }
        });
        let fraction = reference.map(|r| {
            compared += 1;
            if r != 0.0 && r.abs() < f64::from(f32::MIN_POSITIVE) {
                subnormal_compared += 1;
            }
            let target = r.abs() * 2.0_f64.powi(-21) + f64::from(f32::from_bits(1));
            let fraction = (value - r).abs() / target;
            assert!(
                fraction <= 1.0,
                "tau={tau} order={order} value={value} reference={r} fraction={fraction}"
            );
            max_fraction = max_fraction.max(fraction);
            fraction
        });
        if q.is_none() {
            unresolved += 1;
        }
        println!("{{\"tau\":{tau:e},\"order\":{order},\"status\":\"ok\",\"value\":{value:.17e},\"reference_method\":\"{method}\",\"reference\":{},\"reference_uncertainty_estimate\":{uncertainty:.17e},\"target_fraction\":{}}}",
            reference.map_or("null".into(), |r| format!("{r:.17e}")), fraction.map_or("null".into(), |r| format!("{r:.17e}")));
    }
    let maximum = coefficients(0.0, DEFAULT.coefficients - 1, DEFAULT).unwrap();
    assert_eq!(maximum.coefficients.len(), DEFAULT.coefficients);
    assert_eq!(
        coefficients(0.0, DEFAULT.coefficients, DEFAULT).err(),
        Some(Error::OutputLimit)
    );
    let raised = coefficients(
        0.0,
        DEFAULT.coefficients,
        Limits {
            coefficients: DEFAULT.coefficients + 1,
            ..DEFAULT
        },
    )
    .unwrap();
    assert_eq!(raised.coefficients.len(), DEFAULT.coefficients + 1);
    assert_eq!(coefficients(1e10, 0, DEFAULT).err(), Some(Error::WorkLimit));
    println!("],\"cases\":{},\"compared\":{compared},\"series_cases\":{series_cases},\"quadrature_cases\":{quadrature_cases},\"unresolved\":{unresolved},\"refused\":{refused},\"max_target_fraction\":{max_fraction:.17e},\"subnormal_compared\":{subnormal_compared},\"custom_limit_controls\":4}}", cases.len());
}
