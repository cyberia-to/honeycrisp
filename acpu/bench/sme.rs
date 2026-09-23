//! SME f32 matmul benchmark — vs AMX path and Apple Accelerate cblas_sgemm.
//!
//! Skipped on chips without FEAT_SME (M3 and earlier).

#[path = "common.rs"]
mod common;
use common::*;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {}

fn gflops(m: usize, n: usize, k: usize, ns: u64) -> f64 {
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    flops / ns as f64
}

fn iters_for(n: usize) -> usize {
    if n >= 2048 {
        4
    } else if n >= 512 {
        15
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

fn main() {
    let caps = acpu::scan();
    println!("acpu SME benchmark");
    println!("chip: {}", caps.chip);
    if !caps.has_sme {
        println!("SKIP: FEAT_SME not present");
        return;
    }
    println!("SVL: {} B   AMX ver: {}", caps.svl_bytes, caps.amx_ver);
    println!();

    let sizes: &[usize] = &[16, 32, 64, 128, 256, 512, 1024, 2048];

    println!(
        "{:<14} {:>10} {:>10} {:>10} {:>10}",
        "size", "SME GF", "AMX GF", "Apple GF", "SME/AMX"
    );

    let mut wins = 0;
    let mut ties = 0;
    let mut losses = 0;

    // Apple first (matches the sgemm.rs pattern — Accelerate deadlocks
    // after acpu's thread pool spawns).
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

    for &n in sizes {
        let len = n * n;
        let a = vec![0.1f32; len];
        let b = vec![0.2f32; len];
        let mut c_sme = vec![0.0f32; len];
        let mut c_amx = vec![0.0f32; len];
        let iters = iters_for(n);

        // SME path
        for _ in 0..3 {
            acpu::sme::matmul_f32_sme_set(&a, &b, &mut c_sme, n, n, n).unwrap();
        }
        let sme_ns = best_of(
            || {
                acpu::sme::matmul_f32_sme_set(&a, &b, &mut c_sme, n, n, n).unwrap();
            },
            iters,
        );

        // AMX path (existing)
        for _ in 0..3 {
            acpu::matmul_f32_set(&a, &b, &mut c_amx, n, n, n);
        }
        let amx_ns = best_of(|| acpu::matmul_f32_set(&a, &b, &mut c_amx, n, n, n), iters);

        let apple_ns = apple_results.iter().find(|r| r.0 == n).unwrap().1;

        let sme_gf = gflops(n, n, n, sme_ns);
        let amx_gf = gflops(n, n, n, amx_ns);
        let apple_gf = gflops(n, n, n, apple_ns);
        let ratio = sme_gf / amx_gf;

        let status = if ratio > 1.05 {
            wins += 1;
            "WIN"
        } else if ratio > 0.95 {
            ties += 1;
            "TIE"
        } else {
            losses += 1;
            "LOSS"
        };

        println!(
            "{:<14} {:>10.1} {:>10.1} {:>10.1} {:>9.2}× {}",
            format!("{}×{}×{}", n, n, n),
            sme_gf,
            amx_gf,
            apple_gf,
            ratio,
            status,
        );

        // Sanity: SME and AMX must agree.
        let mut diff_max = 0.0f32;
        for (a_, b_) in c_sme.iter().zip(c_amx.iter()) {
            let d = (a_ - b_).abs();
            if d > diff_max {
                diff_max = d;
            }
        }
        if diff_max > 1e-2 {
            println!("  WARN: SME vs AMX max diff = {diff_max}");
        }
    }

    println!();
    println!("SME vs AMX: wins={wins} ties={ties} losses={losses}");
}
