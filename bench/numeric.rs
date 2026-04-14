//! Numeric extensions benchmark: acpu vs Apple Accelerate.
//! Complex mul-acc vs vDSP_zvmul, conversions vs vImageConvert, RoPE.

#[path = "common.rs"]
mod common;
use common::*;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    // vDSP element-wise multiply (used as baseline for complex)
    fn vDSP_vmul(a: *const f32, ia: i64, b: *const f32, ib: i64, c: *mut f32, ic: i64, n: u64);
    fn vDSP_vsub(b: *const f32, ib: i64, a: *const f32, ia: i64, c: *mut f32, ic: i64, n: u64);
    fn vDSP_vadd(a: *const f32, ia: i64, b: *const f32, ib: i64, c: *mut f32, ic: i64, n: u64);
}

const N: usize = 4096;

fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(60));
        eprintln!("WATCHDOG: 60s timeout");
        std::process::exit(1);
    });

    println!("acpu numeric benchmark vs Apple Accelerate");
    println!();

    let mut score = Score::new(); // "apple" — correct here

    // ── complex mul-acc: acpu FCMLA vs vDSP_zvmul ────────────────────

    score.hdr("COMPLEX MUL-ACC (2048 complex pairs)");
    {
        // acpu: interleaved [re,im,re,im,...] = 4096 f32
        let a: Vec<f32> = (0..N).map(|i| (i % 7) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..N).map(|i| (i % 11) as f32 * 0.1).collect();
        let mut acc = vec![0.0f32; N];

        let t_acpu = ns(|| {
            acc.fill(0.0);
            acpu::numeric::complex::complex_mul_acc(&mut acc, &a, &b);
            std::hint::black_box(&acc);
        });

        // Apple baseline: manual complex mul via vDSP_vmul + vDSP_vsub + vDSP_vadd
        // (a_re*b_re - a_im*b_im) + i*(a_re*b_im + a_im*b_re)
        let half = N / 2;
        let a_re: Vec<f32> = (0..half).map(|i| a[2 * i]).collect();
        let a_im: Vec<f32> = (0..half).map(|i| a[2 * i + 1]).collect();
        let b_re: Vec<f32> = (0..half).map(|i| b[2 * i]).collect();
        let b_im: Vec<f32> = (0..half).map(|i| b[2 * i + 1]).collect();
        let mut t1 = vec![0.0f32; half];
        let mut t2 = vec![0.0f32; half];
        let mut c_re = vec![0.0f32; half];
        let mut c_im = vec![0.0f32; half];
        let hn = half as u64;

        let t_apple = ns(|| unsafe {
            // c_re = a_re*b_re - a_im*b_im
            vDSP_vmul(a_re.as_ptr(), 1, b_re.as_ptr(), 1, t1.as_mut_ptr(), 1, hn);
            vDSP_vmul(a_im.as_ptr(), 1, b_im.as_ptr(), 1, t2.as_mut_ptr(), 1, hn);
            vDSP_vsub(t2.as_ptr(), 1, t1.as_ptr(), 1, c_re.as_mut_ptr(), 1, hn);
            // c_im = a_re*b_im + a_im*b_re
            vDSP_vmul(a_re.as_ptr(), 1, b_im.as_ptr(), 1, t1.as_mut_ptr(), 1, hn);
            vDSP_vmul(a_im.as_ptr(), 1, b_re.as_ptr(), 1, t2.as_mut_ptr(), 1, hn);
            vDSP_vadd(t1.as_ptr(), 1, t2.as_ptr(), 1, c_im.as_mut_ptr(), 1, hn);
            std::hint::black_box(&c_re);
            std::hint::black_box(&c_im);
        });

        score.row("complex_mul_acc", t_acpu, t_apple);
    }

    // ── f32↔f16 conversions (absolute, no Apple equiv for bulk f16) ──

    score.hdr("CONVERSIONS (4096 elements)");
    {
        let src: Vec<f32> = (0..N).map(|i| i as f32 * 0.01).collect();
        let mut f16_buf = vec![0u16; N];
        let mut f32_dst = vec![0.0f32; N];

        // f32 → f16: uses hardware fcvtn
        let t_f32_f16 = best_of(
            || {
                acpu::cast_f32_f16(&mut f16_buf, &src);
                std::hint::black_box(&f16_buf);
            },
            200,
        );
        let gb_f16 = N as f64 * 4.0 / t_f32_f16 as f64;
        println!(
            "  {:<28} {:>8}ns  ({:.1} GB/s, hardware fcvtn)",
            "f32→f16", t_f32_f16, gb_f16
        );

        // f16 → f32: uses hardware fcvtl
        acpu::cast_f32_f16(&mut f16_buf, &src);
        let t_f16_f32 = best_of(
            || {
                acpu::cast_f16_f32(&mut f32_dst, &f16_buf);
                std::hint::black_box(&f32_dst);
            },
            200,
        );
        let gb_f32 = N as f64 * 4.0 / t_f16_f32 as f64;
        println!(
            "  {:<28} {:>8}ns  ({:.1} GB/s, hardware fcvtl)",
            "f16→f32", t_f16_f32, gb_f32
        );

        // bf16 and i8 — no Apple equivalent, show absolute throughput
        let mut bf16_buf = vec![0u16; N];
        let mut i8_buf = vec![0i8; N];

        let t_bf16 = ns(|| {
            acpu::cast_f32_bf16(&mut bf16_buf, &src);
            acpu::cast_bf16_f32(&mut f32_dst, &bf16_buf);
            std::hint::black_box(&f32_dst);
        });
        let bytes = N as f64 * 4.0 * 2.0;
        println!(
            "  {:<28} {:>8}ns  ({:.1} GB/s, no Apple equiv)",
            "bf16 round-trip",
            t_bf16,
            bytes / t_bf16 as f64
        );

        let t_i8 = ns(|| {
            acpu::cast_f32_i8(&mut i8_buf, &src, 0.1);
            acpu::cast_i8_f32(&mut f32_dst, &i8_buf, 0.1, 0);
            std::hint::black_box(&f32_dst);
        });
        println!(
            "  {:<28} {:>8}ns  ({:.1} GB/s, no Apple equiv)",
            "i8 quant round-trip",
            t_i8,
            bytes / t_i8 as f64
        );
    }

    // ── RoPE — no Apple equivalent ───────────────────────────────────

    println!();
    println!("--- ROPE (no Apple equivalent) ---");
    {
        let dim = N;
        let x: Vec<f32> = (0..dim).map(|i| i as f32 * 0.01).collect();
        let freqs: Vec<f32> = (0..dim / 2)
            .map(|i| 1.0 / 10000f32.powf(2.0 * i as f32 / dim as f32))
            .collect();
        let mut out = vec![0.0f32; dim];
        let t = ns(|| {
            acpu::vector::rotate(&mut out, &x, &freqs, 42);
            std::hint::black_box(&out);
        });
        println!("  rotate {dim}: {t}ns");
    }

    println!();
    score.summary();
}
