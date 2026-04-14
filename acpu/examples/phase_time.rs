//! Time each phase of sgemm for 64×64 to find the bottleneck.
use std::time::Instant;

fn median_of(times: &mut [u64]) -> u64 {
    times.sort();
    times[times.len() / 2]
}

fn main() {
    for &sz in &[32usize, 64, 128, 256] {
        let a: Vec<f32> = (0..sz * sz).map(|i| (i % 7) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..sz * sz).map(|i| (i % 11) as f32 * 0.1).collect();
        let mut c = vec![0.0f32; sz * sz];
        let iters = 100;

        // Warmup
        acpu::matmul_f32(&a, &b, &mut c, sz, sz, sz);

        // Total sgemm time
        let mut t = vec![0u64; iters];
        for i in 0..iters {
            c.fill(0.0);
            let s = Instant::now();
            acpu::matmul_f32(&a, &b, &mut c, sz, sz, sz);
            t[i] = s.elapsed().as_nanos() as u64;
        }
        let total = median_of(&mut t);

        // Time just c.fill(0.0) to subtract
        for i in 0..iters {
            let s = Instant::now();
            c.fill(0.0);
            std::hint::black_box(&c);
            t[i] = s.elapsed().as_nanos() as u64;
        }
        let fill = median_of(&mut t);

        let ops = 2.0 * (sz as f64).powi(3);
        let net = total.saturating_sub(fill);
        eprintln!(
            "{:>4}×{:<4}  total={:>6}ns  fill={:>4}ns  net={:>6}ns  {:>.1} GFLOPS",
            sz,
            sz,
            total,
            fill,
            net,
            ops / net as f64
        );
    }
}
