//! Measure the per-call cost of opening + closing a `Stream` on M4 Max.

use acpu::streaming::Stream;
use std::time::Instant;

fn main() {
    if !acpu::probe::scan().has_sme {
        eprintln!("skip: FEAT_SME not present");
        return;
    }
    let n = 100_000usize;
    let t0 = Instant::now();
    for _ in 0..n {
        let s = Stream::new().unwrap();
        std::hint::black_box(&s);
        drop(s);
    }
    let dt = t0.elapsed();
    println!(
        "Stream open+close: {} iters in {:?} => {:.0} ns/call",
        n,
        dt,
        dt.as_secs_f64() * 1e9 / n as f64
    );
}
