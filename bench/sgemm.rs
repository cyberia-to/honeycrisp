//! SGEMM benchmark: full spectrum + mixed precision + thread scaling.
#[path = "common.rs"]
mod common;
use common::*;
use std::time::Instant;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {}

// ── helpers ──────────────────────────────────────────────────────────────────

fn gflops(m: usize, n: usize, k: usize, ns: u64) -> f64 {
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    flops / ns as f64
}

fn iters_for(n: usize) -> usize {
    if n >= 2048 {
        8
    } else if n >= 512 {
        20
    } else if n >= 128 {
        100
    } else {
        500
    }
}

fn apple_sgemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
}

// ── 1. SGEMM spectrum ───────────────────────────────────────────────────────

fn bench_spectrum(score: &mut Score) {
    let sizes: &[usize] = &[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

    // Apple ALL sizes first — Accelerate deadlocks after acpu's thread pool spawns
    let mut apple_results: Vec<(usize, u64)> = Vec::new();
    for &n in sizes {
        let len = n * n;
        let a = vec![0.1f32; len];
        let b = vec![0.2f32; len];
        let mut c = vec![0.0f32; len];
        let iters = iters_for(n);
        for _ in 0..3 {
            apple_sgemm(&a, &b, &mut c, n, n, n);
        }
        apple_results.push((n, best_of(|| apple_sgemm(&a, &b, &mut c, n, n, n), iters)));
    }

    // acpu ALL sizes second
    score.hdr("SGEMM spectrum (square M=N=K)");
    for &n in sizes {
        let len = n * n;
        let a = vec![0.1f32; len];
        let b = vec![0.2f32; len];
        let mut c = vec![0.0f32; len];
        let iters = iters_for(n);
        for _ in 0..3 {
            acpu::matmul_f32_set(&a, &b, &mut c, n, n, n);
        }
        let acpu_ns = best_of(|| acpu::matmul_f32_set(&a, &b, &mut c, n, n, n), iters);
        let apple_ns = apple_results.iter().find(|r| r.0 == n).unwrap().1;
        let label = format!("sgemm {}×{}", n, n);
        score.row_gf(&label, gflops(n, n, n, acpu_ns), gflops(n, n, n, apple_ns));
    }
}

// ── 2. Mixed-precision GEMM ─────────────────────────────────────────────────

fn bench_mixed() {
    println!();
    println!("--- MIXED-PRECISION GEMM (256×256) ---");
    println!("{:<28} {:>10} {:>8}", "operation", "GFLOPS", "note");

    let n: usize = 256;
    let len = n * n;
    let flops = 2.0 * (n as f64).powi(3);
    let a_f32 = vec![0.1f32; len];
    let b_f32 = vec![0.2f32; len];

    // fp16
    {
        let mut a16 = vec![0u16; len];
        let mut b16 = vec![0u16; len];
        acpu::cast_f32_f16(&mut a16, &a_f32);
        acpu::cast_f32_f16(&mut b16, &b_f32);
        let mut c = vec![0.0f32; len];
        for _ in 0..3 {
            acpu::matmul_f16(&a16, &b16, &mut c, n, n, n);
        }
        let ns = best_of(|| acpu::matmul_f16(&a16, &b16, &mut c, n, n, n), 100);
        let gf = flops / ns as f64;
        println!("{:<28} {:>8.1} GF   {}", "hgemm fp16", gf, "convert+sgemm");
    }

    // bf16
    {
        let mut a16 = vec![0u16; len];
        let mut b16 = vec![0u16; len];
        acpu::cast_f32_bf16(&mut a16, &a_f32);
        acpu::cast_f32_bf16(&mut b16, &b_f32);
        let mut c = vec![0.0f32; len];
        for _ in 0..3 {
            acpu::matmul_bf16(&a16, &b16, &mut c, n, n, n);
        }
        let ns = best_of(|| acpu::matmul_bf16(&a16, &b16, &mut c, n, n, n), 100);
        let gf = flops / ns as f64;
        println!("{:<28} {:>8.1} GF   {}", "bgemm bf16", gf, "convert+sgemm");
    }

    // i8
    {
        let mut a8 = vec![0i8; len];
        let mut b8 = vec![0i8; len];
        acpu::cast_f32_i8(&mut a8, &a_f32, 127.0);
        acpu::cast_f32_i8(&mut b8, &b_f32, 127.0);
        let mut c = vec![0.0f32; len];
        for _ in 0..3 {
            acpu::matmul_i8(&a8, &b8, &mut c, n, n, n, 1.0 / 127.0, 0);
        }
        let ns = best_of(
            || acpu::matmul_i8(&a8, &b8, &mut c, n, n, n, 1.0 / 127.0, 0),
            100,
        );
        let gf = flops / ns as f64;
        println!("{:<28} {:>8.1} GF   {}", "qgemm i8", gf, "deq+sgemm");
    }
}

// ── 3. Thread scaling ───────────────────────────────────────────────────────

fn bench_thread_scaling() {
    println!();
    println!("--- THREAD SCALING (sgemm 1024×1024) ---");
    println!("{:<28} {:>10} {:>10}", "config", "total GF", "GF/core");

    let n: usize = 1024;
    let len = n * n;
    let a = vec![0.1f32; len];
    let b = vec![0.2f32; len];
    let mut c = vec![0.0f32; len];
    let flops = 2.0 * (n as f64).powi(3);
    let p_cores = acpu::scan().p_cores as usize;

    // warmup
    for _ in 0..3 {
        acpu::matmul_f32_set(&a, &b, &mut c, n, n, n);
    }

    // measure total time for 10 sequential calls
    let start = Instant::now();
    for _ in 0..10 {
        acpu::matmul_f32_set(&a, &b, &mut c, n, n, n);
    }
    let total_ns = start.elapsed().as_nanos() as f64;
    let avg_ns = total_ns / 10.0;
    let total_gf = flops / avg_ns;
    let gf_per_core = total_gf / p_cores as f64;

    println!(
        "{:<28} {:>8.1} GF {:>8.1} GF",
        format!("{} P-cores", p_cores),
        total_gf,
        gf_per_core,
    );
}

// ── main ─────────────────────────────────────────────────────────────────────

fn main() {
    println!("acpu SGEMM benchmark");
    println!("chip: {:?}", acpu::scan().chip);
    println!();

    let mut score = Score::new();

    bench_spectrum(&mut score);
    bench_mixed();
    bench_thread_scaling();

    score.summary();
}
