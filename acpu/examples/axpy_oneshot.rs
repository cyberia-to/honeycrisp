use std::io::Write;
use std::time::Instant;
fn main() {
    let caps = acpu::scan();
    println!("chip: {}", caps.chip);
    if !caps.has_sme {
        return;
    }
    for &n in &[64usize, 256, 1024, 16384, 262144] {
        println!("size = {n}");
        std::io::stdout().flush().ok();
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let mut y: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let a = 1.5f32;
        for _ in 0..3 {
            acpu::streaming::kern::axpy_f32(a, &x, &mut y).unwrap();
        }
        println!("  warmed");
        std::io::stdout().flush().ok();
        let iters = if n < 1024 {
            2000
        } else if n < 16384 {
            500
        } else {
            50
        };
        let s = Instant::now();
        for _ in 0..iters {
            acpu::streaming::kern::axpy_f32(a, &x, &mut y).unwrap();
        }
        let per = s.elapsed().as_nanos() as u64 / iters as u64;
        let bw = (2.0 * 4.0 * n as f64) / per as f64; // bytes / ns = GB/s
        println!("  per_call = {per} ns   bw = {bw:.1} GB/s");
        std::io::stdout().flush().ok();
    }
}
