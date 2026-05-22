use std::time::Instant;
fn main() {
    let caps = acpu::scan();
    if !caps.has_sme {
        return;
    }
    for &n in &[64usize, 128, 256, 512] {
        let len = n * n;
        let a = vec![0.1f32; len];
        let b = vec![0.2f32; len];
        let mut c = vec![0.0f32; len];
        for _ in 0..3 {
            acpu::sme::matmul_f32_sme_set(&a, &b, &mut c, n, n, n).unwrap();
        }
        let iters = 20;
        let s = Instant::now();
        for _ in 0..iters {
            acpu::sme::matmul_f32_sme_set(&a, &b, &mut c, n, n, n).unwrap();
        }
        let elapsed_ns = s.elapsed().as_nanos() as u64;
        let per_call_ns = elapsed_ns / iters;
        let flops = 2u64 * n as u64 * n as u64 * n as u64;
        let gf = (flops as f64) / (per_call_ns as f64);
        println!("n={n:4} per_call={per_call_ns:8}ns  {gf:6.1} GF/s");
    }
}
