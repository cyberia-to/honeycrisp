//! AMX coprocessor utilization: bandwidth and compute throughput.
#[path = "common.rs"]
mod common;
use std::time::Instant;

use acpu::matrix::asm::{
    amx_op, OP_FMA16, OP_FMA32, OP_FMA64, OP_LDX, OP_LDY, OP_LDZ, OP_MAC16, OP_STZ,
};

/// 64-byte-aligned buffer (one AMX register row = 64 bytes = 16 f32).
#[repr(align(128))]
struct A128([f32; 64]);

const CLOCK_GHZ: f64 = 3.228;
const ITERS: usize = 100;
const OPS_PER_ITER: usize = 50;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn med(t: &mut [u64]) -> u64 {
    common::med(t)
}

/// Run `body` (which does OPS_PER_ITER AMX ops) ITERS times, return median
/// ns for a single op.
fn bench_amx_op<F: FnMut()>(mut body: F) -> u64 {
    let mut times = vec![0u64; ITERS];
    for t in times.iter_mut() {
        let s = Instant::now();
        body();
        *t = s.elapsed().as_nanos() as u64;
    }
    med(&mut times) / OPS_PER_ITER as u64
}

// ---------------------------------------------------------------------------
// Bandwidth benchmark
// ---------------------------------------------------------------------------

fn bench_bandwidth() {
    println!("--- AMX BANDWIDTH ---");
    println!(
        "{:<16} {:>8} {:>10} {:>10} {:>8}",
        "operation", "ns/op", "GB/s", "peak GB/s", "util%"
    );

    // Peak: 64 bytes per op × clock frequency
    let peak_gbs = 64.0 * CLOCK_GHZ;

    let mut buf = A128([0.0f32; 64]);
    let ptr = buf.0.as_mut_ptr() as *mut u8;

    // Warm up AMX
    {
        let mut a = vec![0.0f32; 16 * 16];
        let mut b = vec![0.0f32; 16 * 16];
        let mut c = vec![0.0f32; 16 * 16];
        for (i, v) in a.iter_mut().enumerate() {
            *v = i as f32;
        }
        for (i, v) in b.iter_mut().enumerate() {
            *v = i as f32;
        }
        acpu::matmul_f32(&a, &b, &mut c, 16, 16, 16);
    }

    let amx = acpu::Matrix::new().expect("AMX init");

    // LDX
    let ns = bench_amx_op(|| unsafe {
        for _ in 0..OPS_PER_ITER {
            amx_op::<OP_LDX>(ptr as u64);
        }
    });
    let gbs = 64.0 / ns as f64;
    println!(
        "{:<16} {:>8} {:>10.1} {:>10.1} {:>7.1}%",
        "LDX",
        ns,
        gbs,
        peak_gbs,
        gbs / peak_gbs * 100.0
    );

    // LDX pair (rows 0+1)
    let ns = bench_amx_op(|| unsafe {
        for _ in 0..OPS_PER_ITER {
            amx_op::<OP_LDX>(ptr as u64);
            amx_op::<OP_LDX>(ptr as u64 | (1 << 56));
        }
    });
    // Two loads per "op"
    let gbs = 128.0 / ns as f64;
    println!(
        "{:<16} {:>8} {:>10.1} {:>10.1} {:>7.1}%",
        "LDX pair",
        ns,
        gbs,
        peak_gbs * 2.0,
        gbs / (peak_gbs * 2.0) * 100.0
    );

    // LDY
    let ns = bench_amx_op(|| unsafe {
        for _ in 0..OPS_PER_ITER {
            amx_op::<OP_LDY>(ptr as u64);
        }
    });
    let gbs = 64.0 / ns as f64;
    println!(
        "{:<16} {:>8} {:>10.1} {:>10.1} {:>7.1}%",
        "LDY",
        ns,
        gbs,
        peak_gbs,
        gbs / peak_gbs * 100.0
    );

    // LDZ
    let ns = bench_amx_op(|| unsafe {
        for _ in 0..OPS_PER_ITER {
            amx_op::<OP_LDZ>(ptr as u64);
        }
    });
    let gbs = 64.0 / ns as f64;
    println!(
        "{:<16} {:>8} {:>10.1} {:>10.1} {:>7.1}%",
        "LDZ",
        ns,
        gbs,
        peak_gbs,
        gbs / peak_gbs * 100.0
    );

    // STZ
    let ns = bench_amx_op(|| unsafe {
        for _ in 0..OPS_PER_ITER {
            amx_op::<OP_STZ>(ptr as u64);
        }
    });
    let gbs = 64.0 / ns as f64;
    println!(
        "{:<16} {:>8} {:>10.1} {:>10.1} {:>7.1}%",
        "STZ",
        ns,
        gbs,
        peak_gbs,
        gbs / peak_gbs * 100.0
    );

    drop(amx);
}

// ---------------------------------------------------------------------------
// Compute benchmark
// ---------------------------------------------------------------------------

fn bench_compute() {
    println!();
    println!("--- AMX COMPUTE ---");
    println!(
        "{:<20} {:>8} {:>10} {:>10} {:>8}",
        "operation", "ns/op", "GFLOPS", "peak GF", "util%"
    );

    let mut buf = A128([0.0f32; 64]);
    let ptr = buf.0.as_mut_ptr() as *mut u8;

    // Warm up AMX
    {
        let mut a = vec![0.0f32; 16 * 16];
        let mut b = vec![0.0f32; 16 * 16];
        let mut c = vec![0.0f32; 16 * 16];
        for (i, v) in a.iter_mut().enumerate() {
            *v = i as f32;
        }
        for (i, v) in b.iter_mut().enumerate() {
            *v = i as f32;
        }
        acpu::matmul_f32(&a, &b, &mut c, 16, 16, 16);
    }

    let amx = acpu::Matrix::new().expect("AMX init");

    // Preload X and Y rows so FMA has something to work with
    unsafe {
        amx_op::<OP_LDX>(ptr as u64);
        amx_op::<OP_LDY>(ptr as u64);
    }

    // FMA32: 16×16 outer product = 16*16*2 = 512 FLOPs per op
    let fma32_flops = 16.0 * 16.0 * 2.0;
    let fma32_peak = fma32_flops * CLOCK_GHZ; // GFLOPS at 1 op/cycle
    let ns = bench_amx_op(|| unsafe {
        for _ in 0..OPS_PER_ITER {
            amx_op::<OP_FMA32>(0u64);
        }
    });
    let gf = fma32_flops / ns as f64;
    println!(
        "{:<20} {:>8} {:>10.1} {:>10.1} {:>7.1}%",
        "FMA32 16x16",
        ns,
        gf,
        fma32_peak,
        gf / fma32_peak * 100.0
    );

    // FMA16: 32×32 outer product = 32*32*2 = 2048 FLOPs per op
    let fma16_flops = 32.0 * 32.0 * 2.0;
    let fma16_peak = fma16_flops * CLOCK_GHZ;
    let ns = bench_amx_op(|| unsafe {
        for _ in 0..OPS_PER_ITER {
            amx_op::<OP_FMA16>(0u64);
        }
    });
    let gf = fma16_flops / ns as f64;
    println!(
        "{:<20} {:>8} {:>10.1} {:>10.1} {:>7.1}%",
        "FMA16 32x32",
        ns,
        gf,
        fma16_peak,
        gf / fma16_peak * 100.0
    );

    // FMA64: 8×8 outer product = 8*8*2 = 128 FLOPs per op
    let fma64_flops = 8.0 * 8.0 * 2.0;
    let fma64_peak = fma64_flops * CLOCK_GHZ;
    let ns = bench_amx_op(|| unsafe {
        for _ in 0..OPS_PER_ITER {
            amx_op::<OP_FMA64>(0u64);
        }
    });
    let gf = fma64_flops / ns as f64;
    println!(
        "{:<20} {:>8} {:>10.1} {:>10.1} {:>7.1}%",
        "FMA64 8x8",
        ns,
        gf,
        fma64_peak,
        gf / fma64_peak * 100.0
    );

    // MAC16 i16: 32×32 outer product = 32*32*2 = 2048 integer ops per instruction
    let mac16_ops = 32.0 * 32.0 * 2.0;
    let mac16_peak = mac16_ops * CLOCK_GHZ;
    let ns = bench_amx_op(|| unsafe {
        for _ in 0..OPS_PER_ITER {
            amx_op::<OP_MAC16>(0u64);
        }
    });
    let gops = mac16_ops / ns as f64;
    println!(
        "{:<20} {:>8} {:>10.1} {:>10.1} {:>7.1}%",
        "MAC16 i16",
        ns,
        gops,
        mac16_peak,
        gops / mac16_peak * 100.0
    );

    drop(amx);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== AMX utilization benchmark ===");
    println!(
        "clock assumption: {:.3} GHz (adjust CLOCK_GHZ if different)",
        CLOCK_GHZ
    );
    println!(
        "{} iterations x {} ops per sample, median timing",
        ITERS, OPS_PER_ITER
    );
    println!();
    bench_bandwidth();
    bench_compute();
}
