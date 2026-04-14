//! Profile sgemm breakdown: packing vs compute vs accumulate.

use std::time::Instant;

fn main() {
    const M: usize = 2048;
    const N: usize = 2048;
    const K: usize = 2048;

    let mut seed: u64 = 0xDEAD_BEEF;
    let mut randf = || -> f32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32) / (u32::MAX as f32) - 0.5
    };

    let a: Vec<f32> = (0..M * K).map(|_| randf()).collect();
    let b: Vec<f32> = (0..K * N).map(|_| randf()).collect();

    // Warm up.
    let mut c = vec![0.0f32; M * N];
    acpu::matmul_f32(&a, &b, &mut c, M, N, K);

    // Time it.
    let mut times = Vec::new();
    for _ in 0..10 {
        let mut c = vec![0.0f32; M * N];
        let t0 = Instant::now();
        acpu::matmul_f32(&a, &b, &mut c, M, N, K);
        times.push(t0.elapsed());
    }
    times.sort();
    let median = times[5];
    let gflops = 2.0 * M as f64 * N as f64 * K as f64 / median.as_nanos() as f64;
    println!("sgemm {}×{}: {:.1?} = {:.0} GFLOPS", M, N, median, gflops);
}
