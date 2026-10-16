mod recurrence;
use recurrence::{coefficients, Error, Limits, DEFAULT};
use std::io::{self, Read};

fn expect_error(tau: f32, order: usize, limits: Limits, expected: Error) {
    assert_eq!(coefficients(tau, order, limits).err(), Some(expected));
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut accepted = 0;
    let mut compared = 0;
    let mut refused = 0;
    let mut unresolved = 0;
    let mut max_error = 0.0_f64;
    let mut max_work = 0;
    println!("{{\"rows\":[");
    for (index, line) in input.lines().skip(1).enumerate() {
        let fields: Vec<_> = line.split('\t').collect();
        let tau = fields[0].parse::<f32>().unwrap();
        let order = fields[1].parse::<usize>().unwrap();
        let reference = fields[2].parse::<f64>().ok();
        if index != 0 {
            println!(",");
        }
        match coefficients(tau, order, DEFAULT) {
            Ok(result) => {
                accepted += 1;
                max_work = max_work.max(result.steps);
                let value = f64::from(result.coefficients[order]);
                let error = reference.map(|r| {
                    let target = r.abs() * 2.0_f64.powi(-21) + f64::from(f32::from_bits(1));
                    let fraction = (value - r).abs() / target;
                    assert!(fraction <= 1.0, "tau={tau}, order={order}, value={value}, reference={r}, fraction={fraction}");
                    compared += 1;
                    max_error = max_error.max(fraction);
                    fraction
                });
                if reference.is_none() {
                    unresolved += 1;
                }
                println!("{{\"tau\":{tau:e},\"order\":{order},\"status\":\"ok\",\"value\":{value:.17e},\"depth\":{},\"steps\":{},\"rounds\":{},\"tail_estimate\":{:.17e},\"gap_fraction\":{:.17e},\"target_fraction\":{}}}",
                    result.depth, result.steps, result.rounds, result.tail_estimate, result.largest_gap,
                    error.map_or("null".into(), |e| format!("{e:.17e}")));
                assert_eq!(
                    coefficients(
                        tau,
                        order,
                        Limits {
                            steps: result.steps,
                            ..DEFAULT
                        }
                    )
                    .unwrap()
                    .coefficients,
                    result.coefficients
                );
                if result.steps != 0 {
                    expect_error(
                        tau,
                        order,
                        Limits {
                            steps: result.steps - 1,
                            ..DEFAULT
                        },
                        Error::WorkLimit,
                    );
                }
            }
            Err(error) => {
                assert_eq!(
                    error,
                    Error::WorkLimit,
                    "unexpected tau={tau} order={order}"
                );
                refused += 1;
                println!("{{\"tau\":{tau:e},\"order\":{order},\"status\":\"WorkLimit\"}}");
            }
        }
    }
    for tau in [-1.0, f32::NEG_INFINITY, f32::INFINITY, f32::NAN] {
        expect_error(tau, 0, DEFAULT, Error::InvalidTime);
    }
    expect_error(1.0, usize::MAX, DEFAULT, Error::OrderOverflow);
    expect_error(1.0, 65_536, DEFAULT, Error::OutputLimit);
    expect_error(
        1.0,
        0,
        Limits {
            steps: 0,
            ..DEFAULT
        },
        Error::WorkLimit,
    );
    expect_error(f32::MAX, 0, DEFAULT, Error::WorkLimit);
    let tiny = coefficients(f32::from_bits(1), 4, DEFAULT).unwrap();
    assert_eq!(tiny.coefficients, [1.0, -f32::from_bits(1), 0.0, 0.0, 0.0]);
    assert_eq!(coefficients(-0.0, 0, DEFAULT).unwrap().coefficients, [1.0]);
    println!("],\"accepted\":{accepted},\"compared\":{compared},\"refused\":{refused},\"unresolved\":{unresolved},\"max_target_fraction\":{max_error:.17e},\"max_steps\":{max_work},\"admission_controls\":10}}");
}
