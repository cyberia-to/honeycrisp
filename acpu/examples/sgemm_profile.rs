use std::time::Instant;

fn main() {
    let sz = 1024usize;
    let a: Vec<f32> = (0..sz * sz).map(|i| (i % 7) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..sz * sz).map(|i| (i % 11) as f32 * 0.1).collect();
    let mut c = vec![0.0f32; sz * sz];

    // warmup
    for _ in 0..3 {
        c.fill(0.0);
        acpu::matmul_f32_set(&a, &b, &mut c, sz, sz, sz);
    }

    let mut best = u128::MAX;
    for _ in 0..20 {
        let t = Instant::now();
        acpu::matmul_f32_set(&a, &b, &mut c, sz, sz, sz);
        let elapsed = t.elapsed().as_nanos();
        best = best.min(elapsed);
    }
    let gf = 2.0 * (sz as f64).powi(3) / best as f64;
    eprintln!(
        "sgemm {sz}×{sz}: {:.1}us = {:.0} GF",
        best as f64 / 1000.0,
        gf
    );

    // Compare with single-threaded to isolate threading overhead
    // Use a smaller size that doesn't multithread
    let sz2 = 512usize;
    let a2: Vec<f32> = (0..sz2 * sz2).map(|i| (i % 7) as f32 * 0.1).collect();
    let b2: Vec<f32> = (0..sz2 * sz2).map(|i| (i % 11) as f32 * 0.1).collect();
    let mut c2 = vec![0.0f32; sz2 * sz2];
    for _ in 0..3 {
        acpu::matmul_f32_set(&a2, &b2, &mut c2, sz2, sz2, sz2);
    }
    let mut best2 = u128::MAX;
    for _ in 0..30 {
        let t = Instant::now();
        acpu::matmul_f32_set(&a2, &b2, &mut c2, sz2, sz2, sz2);
        best2 = best2.min(t.elapsed().as_nanos());
    }
    let gf2 = 2.0 * (sz2 as f64).powi(3) / best2 as f64;
    eprintln!(
        "sgemm {sz2}×{sz2}: {:.1}us = {:.0} GF (single-core reference)",
        best2 as f64 / 1000.0,
        gf2
    );
    eprintln!(
        "1024 per-core: {:.0} GF ({:.1}× scaling over 8 cores)",
        gf / 8.0,
        gf / gf2
    );
}
