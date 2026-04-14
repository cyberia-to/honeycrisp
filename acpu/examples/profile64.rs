//! Profile 64×64: isolate microkernel vs preload/store C overhead.
use std::time::Instant;

fn median(t: &mut [u64]) -> u64 {
    t.sort();
    t[t.len() / 2]
}

fn main() {
    let n = 64usize;
    let k = 64;
    let a: Vec<f32> = (0..n * k).map(|i| (i % 7) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 * 0.1).collect();
    let mut c = vec![0.0f32; n * n];
    let iters = 500;

    // Full sgemm (includes pack + preload + compute + store)
    acpu::matmul_f32(&a, &b, &mut c, n, n, k);
    let mut t = vec![0u64; iters];
    for i in 0..iters {
        c.fill(0.0);
        let s = Instant::now();
        acpu::matmul_f32(&a, &b, &mut c, n, n, k);
        t[i] = s.elapsed().as_nanos() as u64;
    }
    eprintln!(
        "full sgemm 64×64: {:>5} ns  {:.1} GFLOPS",
        median(&mut t),
        2.0 * 64f64.powi(3) / median(&mut t) as f64
    );

    // Just pack_a (indirect via calling sgemm with k=0 doesn't work).
    // Instead: measure sgemm for different k to extrapolate pack cost.
    for &kk in &[8, 16, 32, 64] {
        let a2: Vec<f32> = (0..n * kk).map(|i| (i % 7) as f32 * 0.1).collect();
        let b2: Vec<f32> = (0..kk * n).map(|i| (i % 11) as f32 * 0.1).collect();
        let mut c2 = vec![0.0f32; n * n];
        acpu::matmul_f32(&a2, &b2, &mut c2, n, n, kk);
        for i in 0..iters {
            c2.fill(0.0);
            let s = Instant::now();
            acpu::matmul_f32(&a2, &b2, &mut c2, n, n, kk);
            t[i] = s.elapsed().as_nanos() as u64;
        }
        let ns = median(&mut t);
        let gf = 2.0 * n as f64 * n as f64 * kk as f64 / ns as f64;
        eprintln!(
            "sgemm 64×64×k={:<3}: {:>5} ns  {:.1} GFLOPS  ({:.1} ns/k-step)",
            kk,
            ns,
            gf,
            (ns as f64) / kk as f64
        );
    }
    // If ns/k-step is constant, overhead is in the per-k-step ops.
    // If it decreases with k, fixed overhead (pack/preload) dominates.
}
