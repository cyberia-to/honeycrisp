//! Compare single-threaded GEBP vs parallel at 1024.
use std::time::Instant;

fn median(t: &mut [u64]) -> u64 {
    t.sort();
    t[t.len() / 2]
}

fn main() {
    for &sz in &[256, 512, 768, 1024] {
        let a: Vec<f32> = (0..sz * sz).map(|i| (i % 7) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..sz * sz).map(|i| (i % 11) as f32 * 0.1).collect();
        let mut c = vec![0.0f32; sz * sz];
        acpu::matmul_f32(&a, &b, &mut c, sz, sz, sz);
        let iters = if sz >= 1024 { 5 } else { 10 };
        let mut t = vec![0u64; iters];
        for i in 0..iters {
            c.fill(0.0);
            let s = Instant::now();
            acpu::matmul_f32(&a, &b, &mut c, sz, sz, sz);
            t[i] = s.elapsed().as_nanos() as u64;
        }
        let ns = median(&mut t);
        let gf = 2.0 * (sz as f64).powi(3) / ns as f64;
        let flops = 2 * sz * sz * sz;
        let parallel = flops > 200_000_000;
        eprintln!(
            "{:>5}: {:>8} ns  {:>8.1} GFLOPS  {}",
            sz,
            ns,
            gf,
            if parallel { "parallel" } else { "single" }
        );
    }
}
