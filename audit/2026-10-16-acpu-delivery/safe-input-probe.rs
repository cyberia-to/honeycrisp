//! Existing APIs only. Never call malformed CSR or the empty recursive polynomial.
fn report<T: std::fmt::Debug>(name: &str, f: impl FnOnce() -> T + std::panic::UnwindSafe) {
    match std::panic::catch_unwind(f) {
        Ok(value) => println!("{name}: returned {value:?}"),
        Err(_) => println!("{name}: panicked"),
    }
}

fn main() {
    println!("debug_assertions={}", cfg!(debug_assertions));
    report("ntt non-power-of-two", || {
        let mut values = [1, 2, 3];
        acpu::field::ntt_forward(&mut values, 1);
        values
    });
    report("bit reversal non-power-of-two", || {
        let mut values = [1, 2, 3];
        acpu::field::bit_reverse_permute(&mut values);
        values
    });
    report("poly non-power-of-two", || {
        acpu::field::multilinear_eval(&[1, 2, 3], &[0, 0])
    });
    report("poly too few coordinates", || {
        acpu::field::multilinear_eval(&[1, 2, 3, 4], &[0])
    });
    report("poly extra coordinates", || {
        acpu::field::multilinear_eval(&[1, 2, 3, 4], &[0, 0, 5])
    });
    for tau in [1.0, 100.0, 104.0] {
        let coeffs = acpu::sparse::chebyshev::chebyshev_coeffs(tau, 20);
        println!(
            "chebyshev tau={tau}: first={:?}; finite={}/{}",
            coeffs[0],
            coeffs.iter().filter(|c| c.is_finite()).count(),
            coeffs.len()
        );
    }
}
