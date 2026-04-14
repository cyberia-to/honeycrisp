//! Integer SIMD benchmark: NEON throughput for i8/i16/i32 operations.
//! Comparison: NEON vectorized (acpu) vs scalar baseline.

#[path = "common.rs"]
mod common;
use common::*;
use std::time::Instant;

const N: usize = 4096;

fn scalar_sum_i32(x: &[i32]) -> i64 {
    x.iter().map(|&v| v as i64).sum()
}

fn scalar_max_i32(x: &[i32]) -> i32 {
    x.iter().copied().max().unwrap_or(i32::MIN)
}

fn scalar_min_i32(x: &[i32]) -> i32 {
    x.iter().copied().min().unwrap_or(i32::MAX)
}

fn scalar_dot_i8(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b).map(|(&x, &y)| x as i32 * y as i32).sum()
}

fn scalar_macc_i16(acc: &mut [i32], a: &[i16], b: &[i16]) {
    for i in 0..acc.len().min(a.len()).min(b.len()) {
        acc[i] += a[i] as i32 * b[i] as i32;
    }
}

fn scalar_add_i32(dst: &mut [i32], a: &[i32], b: &[i32]) {
    for i in 0..dst.len().min(a.len()).min(b.len()) {
        dst[i] = a[i].wrapping_add(b[i]);
    }
}

fn scalar_mul_i32(dst: &mut [i32], a: &[i32], b: &[i32]) {
    for i in 0..dst.len().min(a.len()).min(b.len()) {
        dst[i] = a[i].wrapping_mul(b[i]);
    }
}

fn scalar_absmax_i8(x: &[i8]) -> u8 {
    x.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0)
}

fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(60));
        eprintln!("!!! 60s TIMEOUT !!!");
        std::process::exit(1);
    });

    let caps = acpu::probe::scan();
    println!("acpu integer SIMD benchmark — {:?}", caps.chip);

    let mut score = Score::vs("scalar");

    // ── test data ───────────────────────────────────────────────────────
    let i32_a: Vec<i32> = (0..N).map(|i| (i as i32 * 7 - 2000)).collect();
    let i32_b: Vec<i32> = (0..N).map(|i| (i as i32 * 3 + 500)).collect();
    let i8_a: Vec<i8> = (0..N).map(|i| ((i * 7) % 255) as i8).collect();
    let i8_b: Vec<i8> = (0..N).map(|i| ((i * 13) % 255) as i8).collect();
    let i16_a: Vec<i16> = (0..N).map(|i| (i as i16 * 5 - 2000)).collect();
    let i16_b: Vec<i16> = (0..N).map(|i| (i as i16 * 3 + 1000)).collect();
    let mut dst_i32 = vec![0i32; N];
    let mut acc_i32 = vec![0i32; N];

    // ── i32 reductions ──────────────────────────────────────────────────
    score.hdr("I32 REDUCTIONS (4096 elements)");

    score.row(
        "sum_i32",
        ns(|| {
            std::hint::black_box(acpu::vector::integer::sum_i32(&i32_a));
        }),
        ns(|| {
            std::hint::black_box(scalar_sum_i32(&i32_a));
        }),
    );
    score.row(
        "max_i32",
        ns(|| {
            std::hint::black_box(acpu::vector::integer::max_i32(&i32_a));
        }),
        ns(|| {
            std::hint::black_box(scalar_max_i32(&i32_a));
        }),
    );
    score.row(
        "min_i32",
        ns(|| {
            std::hint::black_box(acpu::vector::integer::min_i32(&i32_a));
        }),
        ns(|| {
            std::hint::black_box(scalar_min_i32(&i32_a));
        }),
    );

    // ── i8 dot product (SDOT) ───────────────────────────────────────────
    score.hdr("I8 DOT PRODUCT (SDOT, 4096 elements)");

    let neon_i8 = ns(|| {
        std::hint::black_box(acpu::vector::integer::dot_i8(&i8_a, &i8_b));
    });
    let scalar_i8 = ns(|| {
        std::hint::black_box(scalar_dot_i8(&i8_a, &i8_b));
    });
    score.row("dot_i8", neon_i8, scalar_i8);

    // throughput
    let ops = N as f64; // N multiply-adds
    let gops = ops / neon_i8 as f64;
    println!("  throughput: {:.1} Gop/s ({} i8×i8→i32 MADs)", gops, N);

    // ── i16 multiply-accumulate ─────────────────────────────────────────
    score.hdr("I16 MULTIPLY-ACCUMULATE (SMLAL, 4096 elements)");

    score.row(
        "macc_i16",
        ns(|| {
            acc_i32.fill(0);
            acpu::vector::integer::macc_i16(&mut acc_i32, &i16_a, &i16_b);
            std::hint::black_box(&acc_i32);
        }),
        ns(|| {
            acc_i32.fill(0);
            scalar_macc_i16(&mut acc_i32, &i16_a, &i16_b);
            std::hint::black_box(&acc_i32);
        }),
    );

    // ── i32 elementwise ─────────────────────────────────────────────────
    score.hdr("I32 ELEMENTWISE (4096 elements)");

    score.row(
        "add_i32",
        ns(|| {
            acpu::vector::integer::add_i32(&mut dst_i32, &i32_a, &i32_b);
            std::hint::black_box(&dst_i32);
        }),
        ns(|| {
            scalar_add_i32(&mut dst_i32, &i32_a, &i32_b);
            std::hint::black_box(&dst_i32);
        }),
    );
    score.row(
        "mul_i32",
        ns(|| {
            acpu::vector::integer::mul_i32(&mut dst_i32, &i32_a, &i32_b);
            std::hint::black_box(&dst_i32);
        }),
        ns(|| {
            scalar_mul_i32(&mut dst_i32, &i32_a, &i32_b);
            std::hint::black_box(&dst_i32);
        }),
    );

    // ── i8 absmax ────────────────────────────────────────────────────────
    score.hdr("I8 ABSMAX (4096 elements)");

    score.row(
        "absmax_i8",
        ns(|| {
            std::hint::black_box(acpu::vector::integer::absmax_i8(&i8_a));
        }),
        ns(|| {
            std::hint::black_box(scalar_absmax_i8(&i8_a));
        }),
    );

    // ── fused operations (single-pass compute+reduce) ────────────────────
    let u8_a: Vec<u8> = (0..N).map(|i| (i % 200) as u8).collect();
    let u8_b: Vec<u8> = (0..N).map(|i| ((i * 3 + 50) % 200) as u8).collect();

    score.hdr("FUSED OPS (single-pass, 4096 elements)");

    score.row(
        "sad_u8",
        ns(|| {
            std::hint::black_box(acpu::vector::integer_fused::sad_u8(&u8_a, &u8_b));
        }),
        ns(|| {
            let mut s = 0u64;
            for j in 0..N {
                s += (u8_a[j] as i16 - u8_b[j] as i16).unsigned_abs() as u64;
            }
            std::hint::black_box(s);
        }),
    );

    score.row(
        "ssd_i32",
        ns(|| {
            std::hint::black_box(acpu::vector::integer_fused::ssd_i32(&i32_a, &i32_b));
        }),
        ns(|| {
            let mut s = 0i64;
            for j in 0..N {
                let d = i32_a[j] as i64 - i32_b[j] as i64;
                s += d * d;
            }
            std::hint::black_box(s);
        }),
    );

    let mut scale_acc = vec![0i32; N];
    score.row(
        "scale_acc_i16",
        ns(|| {
            scale_acc.fill(0);
            acpu::vector::integer_fused::scale_acc_i16(&mut scale_acc, &i16_a, 7);
            std::hint::black_box(&scale_acc);
        }),
        ns(|| {
            scale_acc.fill(0);
            for j in 0..N {
                scale_acc[j] += i16_a[j] as i32 * 7;
            }
            std::hint::black_box(&scale_acc);
        }),
    );

    score.row(
        "sum_abs_i8",
        ns(|| {
            std::hint::black_box(acpu::vector::integer_fused::sum_abs_i8(&i8_a));
        }),
        ns(|| {
            let s: u64 = i8_a.iter().map(|v| v.unsigned_abs() as u64).sum();
            std::hint::black_box(s);
        }),
    );

    println!();
    score.summary();
}
