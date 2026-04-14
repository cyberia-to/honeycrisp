use std::time::Instant;

fn main() {
    let n: usize = 4096;
    // Same data as bench_all
    let src: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * 6.0 - 3.0).collect();
    let mut dst = vec![0.0f32; n];
    // Extra buffers like bench_all has (compete for cache)
    let _extra1 = vec![0.0f32; n];
    let _extra2 = vec![0.0f32; n];

    // warmup
    acpu::vector::math::exp_to(&src, &mut dst);

    // Median like bench_all
    let mut times: Vec<u64> = Vec::with_capacity(200);
    for _ in 0..200 {
        let t = Instant::now();
        acpu::vector::math::exp_to(&src, &mut dst);
        times.push(t.elapsed().as_nanos() as u64);
    }
    times.sort();
    let median = times[times.len() / 2];
    let min = times[0];
    eprintln!("exp_to: median={}ns min={}ns (n={})", median, min, n);
    eprintln!(
        "  p5={}ns p25={}ns p75={}ns p95={}ns",
        times[10], times[50], times[150], times[190]
    );
}
