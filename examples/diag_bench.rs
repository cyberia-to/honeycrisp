//! Diagnostic: time one sgemm call, print immediately.
use std::time::Instant;

fn main() {
    let n = 16usize;
    let a: Vec<f32> = (0..n * n).map(|i| (i % 7) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..n * n).map(|i| (i % 11) as f32 * 0.1).collect();
    let mut c = vec![0.0f32; n * n];

    eprintln!("starting warmup...");
    acpu::matmul_f32(&a, &b, &mut c, n, n, n);
    eprintln!("warmup done");

    c.fill(0.0);
    let t = Instant::now();
    acpu::matmul_f32(&a, &b, &mut c, n, n, n);
    let ns = t.elapsed().as_nanos();
    eprintln!("16x16 sgemm: {} ns", ns);

    let n = 64usize;
    let a: Vec<f32> = (0..n * n).map(|i| (i % 7) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..n * n).map(|i| (i % 11) as f32 * 0.1).collect();
    let mut c = vec![0.0f32; n * n];

    acpu::matmul_f32(&a, &b, &mut c, n, n, n);
    c.fill(0.0);
    let t = Instant::now();
    acpu::matmul_f32(&a, &b, &mut c, n, n, n);
    let ns = t.elapsed().as_nanos();
    eprintln!("64x64 sgemm: {} ns", ns);

    let n = 256usize;
    let a: Vec<f32> = (0..n * n).map(|i| (i % 7) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..n * n).map(|i| (i % 11) as f32 * 0.1).collect();
    let mut c = vec![0.0f32; n * n];

    acpu::matmul_f32(&a, &b, &mut c, n, n, n);
    c.fill(0.0);
    let t = Instant::now();
    acpu::matmul_f32(&a, &b, &mut c, n, n, n);
    let ns = t.elapsed().as_nanos();
    eprintln!("256x256 sgemm: {} ns", ns);
    eprintln!("done");
}
