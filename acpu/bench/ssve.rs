//! SSVE numerical kernel bench: AXPY (y += a*x) vs NEON.

#[path = "common.rs"]
mod common;
use common::*;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_axpy(a: f32, x: &[f32], y: &mut [f32]) {
    use core::arch::aarch64::*;
    let av = vdupq_n_f32(a);
    let n = x.len();
    let mut i = 0;
    while i + 4 <= n {
        let xv = vld1q_f32(x.as_ptr().add(i));
        let yv = vld1q_f32(y.as_ptr().add(i));
        let r = vfmaq_f32(yv, av, xv);
        vst1q_f32(y.as_mut_ptr().add(i), r);
        i += 4;
    }
    while i < n {
        y[i] += a * x[i];
        i += 1;
    }
}

fn apple_saxpy(a: f32, x: &[f32], y: &mut [f32]) {
    unsafe {
        let n = x.len() as i32;
        // cblas_saxpy is in Accelerate.
        cblas_saxpy(n, a, x.as_ptr(), 1, y.as_mut_ptr(), 1);
    }
}

extern "C" {
    fn cblas_saxpy(n: i32, alpha: f32, x: *const f32, incx: i32, y: *mut f32, incy: i32);
}

fn main() {
    let caps = acpu::scan();
    println!("acpu SSVE AXPY benchmark — wide vs NEON vs Apple Accelerate");
    println!("chip: {}", caps.chip);
    if !caps.has_sme {
        println!("SKIP: FEAT_SME not present");
        return;
    }
    println!();

    let sizes: &[usize] = &[64, 256, 1024, 4096, 16384, 65536, 262144];

    println!(
        "{:<14} {:>10} {:>10} {:>10} {:>10}",
        "n f32", "NEON ns", "Apple ns", "SSVE ns", "SSVE/NEON"
    );

    let mut wins = 0;
    let mut total = 0;

    for &n in sizes {
        println!("# size = {n}");
        use std::io::Write;
        std::io::stdout().flush().ok();
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let mut y_neon: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let mut y_apple: Vec<f32> = y_neon.clone();
        let mut y_ssve: Vec<f32> = y_neon.clone();
        let a = 1.5f32;

        // Warm
        for _ in 0..3 {
            unsafe { neon_axpy(a, &x, &mut y_neon) };
            acpu::streaming::kern::axpy_f32(a, &x, &mut y_ssve).unwrap();
        }
        println!("  warmed");
        std::io::stdout().flush().ok();
        // (cblas_saxpy disabled — Accelerate's internal pool deadlocks
        // when interleaved with streaming-mode calls in the same proc.)
        let _apple_ns: u64 = 0;
        let apple_ns = _apple_ns;
        let _ = &y_apple;

        let iters = if n < 1024 {
            2000
        } else if n < 16384 {
            500
        } else {
            50
        };

        let neon_ns = best_of(|| unsafe { neon_axpy(a, &x, &mut y_neon) }, iters);
        let ssve_ns = best_of(
            || {
                acpu::streaming::kern::axpy_f32(a, &x, &mut y_ssve).unwrap();
            },
            iters,
        );

        let ratio = neon_ns as f64 / ssve_ns as f64;
        total += 1;
        let status = if ratio > 1.05 {
            wins += 1;
            "WIN"
        } else if ratio > 0.95 {
            "TIE"
        } else {
            "LOSS"
        };
        println!(
            "{:<14} {:>10} {:>10} {:>10} {:>9.2}× {}",
            n, neon_ns, apple_ns, ssve_ns, ratio, status
        );
    }

    println!();
    println!("wins / total (vs NEON): {wins}/{total}");
}
