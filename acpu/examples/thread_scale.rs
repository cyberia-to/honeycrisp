//! Thread scaling test for 1024×1024.
//! Tests 1, 2, 4, 8 threads to find optimal parallelism.

use std::time::Instant;

const N: usize = 1024;
const ITERS: usize = 20;

fn main() {
    let a: Vec<f32> = (0..N * N).map(|i| (i % 7) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..N * N).map(|i| (i % 11) as f32 * 0.1).collect();
    let mut c = vec![0.0f32; N * N];

    let caps = acpu::scan();
    println!("chip: {}, p_cores: {}", caps.chip, caps.p_cores);
    println!("matrix: {}×{}\n", N, N);
    println!("{:>8}  {:>8}  {:>8}", "threads", "µs", "GFLOPS");
    println!("{}", "-".repeat(30));

    // Single-threaded baseline
    {
        let mut times = Vec::with_capacity(ITERS);
        for _ in 0..5 {
            c.fill(0.0);
            acpu::matmul_f32(&a, &b, &mut c, N, N, N);
        }
        for _ in 0..ITERS {
            c.fill(0.0);
            let t = Instant::now();
            acpu::matmul_f32(&a, &b, &mut c, N, N, N);
            times.push(t.elapsed().as_nanos() as u64);
        }
        times.sort();
        let ns = times[ITERS / 2];
        let gf = 2.0 * (N as f64).powi(3) / ns as f64;
        println!("{:>8}  {:>8.1}  {:>8.1}", "auto", ns as f64 / 1000.0, gf);
    }

    // Manual thread counts
    for &n_threads in &[1usize, 2, 4, 8] {
        let mut times = Vec::with_capacity(ITERS);

        for _ in 0..5 {
            c.fill(0.0);
            run_parallel(&a, &b, &mut c, n_threads);
        }

        for _ in 0..ITERS {
            c.fill(0.0);
            let t = Instant::now();
            run_parallel(&a, &b, &mut c, n_threads);
            times.push(t.elapsed().as_nanos() as u64);
        }
        times.sort();
        let ns = times[ITERS / 2];
        let gf = 2.0 * (N as f64).powi(3) / ns as f64;
        println!("{:>8}  {:>8.1}  {:>8.1}", n_threads, ns as f64 / 1000.0, gf);
    }
}

fn run_parallel(a: &[f32], b: &[f32], c: &mut [f32], n_threads: usize) {
    if n_threads <= 1 {
        acpu::matmul_f32(a, b, c, N, N, N);
        return;
    }

    let mr = 16usize;
    let base = (N / n_threads / mr) * mr;
    let rows_per = if base == 0 { mr } else { base };

    std::thread::scope(|s| {
        let mut c_rest: &mut [f32] = c;
        let mut m_start = 0;

        while m_start < N {
            let m_this = if N - m_start <= rows_per + mr {
                N - m_start
            } else {
                rows_per
            };

            let (c_chunk, rest) = c_rest.split_at_mut(m_this * N);
            c_rest = rest;
            let a_slice = &a[m_start * N..(m_start + m_this) * N];

            s.spawn(move || {
                let _ = acpu::sync::affinity::pin_p_core();
                acpu::matmul_f32(a_slice, b, c_chunk, m_this, N, N);
            });

            m_start += m_this;
        }
    });
}
