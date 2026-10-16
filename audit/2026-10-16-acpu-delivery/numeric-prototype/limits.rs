//! Full-buffer admission controls; no accuracy claim at mathematically tiny orders.
mod recurrence;
use recurrence::{coefficients, Error, Limits, DEFAULT};

fn main() {
    let prefix = coefficients(1.0, 20, DEFAULT).unwrap();
    let full = coefficients(1.0, DEFAULT.coefficients - 1, DEFAULT).unwrap();
    assert_eq!(full.coefficients.len(), DEFAULT.coefficients);
    assert_eq!(
        &full.coefficients[..prefix.coefficients.len()],
        prefix.coefficients
    );
    for (order, c) in full.coefficients.iter().enumerate() {
        assert!(c.is_finite() && c.abs() <= 1.0);
        if *c != 0.0 {
            assert_eq!(c.is_sign_negative(), order % 2 == 1);
        }
    }
    assert_eq!(
        coefficients(1.0, DEFAULT.coefficients, DEFAULT).err(),
        Some(Error::OutputLimit)
    );
    let raised = coefficients(
        1.0,
        DEFAULT.coefficients,
        Limits {
            coefficients: DEFAULT.coefficients + 1,
            ..DEFAULT
        },
    )
    .unwrap();
    assert_eq!(
        &raised.coefficients[..full.coefficients.len()],
        full.coefficients
    );
    assert_eq!(
        coefficients(
            0.0,
            usize::MAX - 1,
            Limits {
                coefficients: usize::MAX,
                steps: u64::MAX
            }
        )
        .err(),
        Some(Error::Allocation)
    );
    assert!(full.largest_gap <= 1.0 && full.tail_estimate <= 2.0_f64.powi(-40));
    println!("{{\"positive_time_maximum_count\":{},\"depth\":{},\"steps\":{},\"rounds\":{},\"prefix_count\":{},\"custom_count\":{},\"capacity_overflow\":\"Allocation\"}}",
        full.coefficients.len(), full.depth, full.steps, full.rounds,
        prefix.coefficients.len(), raised.coefficients.len());
}
