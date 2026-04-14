//! Compare NEON vs AMX paths for specific sizes.
use std::time::Instant;

fn median_of(times: &mut [u64]) -> u64 {
    times.sort();
    times[times.len() / 2]
}

fn main() {
    // Force NEON path by calling the function directly.
    // We can't access sgemm_neon directly, so test by adjusting dispatch.
    // Instead, compare current sgemm vs scalar to isolate AMX perf.
    for &sz in &[32usize, 48, 64, 96, 128] {
        let a: Vec<f32> = (0..sz * sz).map(|i| (i % 7) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..sz * sz).map(|i| (i % 11) as f32 * 0.1).collect();
        let mut c = vec![0.0f32; sz * sz];
        let iters = 100;

        acpu::matmul_f32(&a, &b, &mut c, sz, sz, sz);
        let mut t = vec![0u64; iters];
        for i in 0..iters {
            c.fill(0.0);
            let s = Instant::now();
            acpu::matmul_f32(&a, &b, &mut c, sz, sz, sz);
            t[i] = s.elapsed().as_nanos() as u64;
        }
        let ns = median_of(&mut t);
        let ops = 2.0 * (sz as f64).powi(3);
        let gf = ops / ns as f64;

        // Breakdown: how much time in pack vs compute?
        // Can't measure directly, but we can compare.
        eprintln!("{:>5}: {:>8} ns  {:>8.2} GFLOPS", sz, ns, gf);
    }
}
