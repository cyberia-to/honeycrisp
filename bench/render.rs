//! Rendering operations benchmark: acpu vs Apple Accelerate.
//! Uses vvrsqrtf, vvrecf, vDSP_vclip, vDSP_vintb as Apple baselines.

#[path = "common.rs"]
mod common;
use common::*;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn vvrsqrtf(y: *mut f32, x: *const f32, n: *const i32);
    fn vvrecf(y: *mut f32, x: *const f32, n: *const i32);
    fn vDSP_vclip(
        a: *const f32,
        ia: i64,
        lo: *const f32,
        hi: *const f32,
        c: *mut f32,
        ic: i64,
        n: u64,
    );
    fn vDSP_vintb(
        a: *const f32,
        ia: i64,
        b: *const f32,
        ib: i64,
        t: *const f32,
        c: *mut f32,
        ic: i64,
        n: u64,
    );
}

fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(60));
        eprintln!("WATCHDOG: 60s timeout");
        std::process::exit(1);
    });

    let n: usize = 4096;
    let nn = n as i32;
    let nu = n as u64;

    let src: Vec<f32> = (1..=n).map(|i| i as f32 * 0.01 + 0.1).collect();
    let a_vec: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
    let b_vec: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 + 1.0).collect();
    let wide: Vec<f32> = (0..n).map(|i| (i as f32) - 2048.0).collect();

    let mut score = Score::new(); // "apple" column — correct here

    score.hdr("RENDERING OPS vs Apple Accelerate (4096 f32)");

    // ── rsqrt: acpu vs vvrsqrtf ──────────────────────────────────────
    {
        let mut buf = src.clone();
        let mut apple_dst = vec![0.0f32; n];

        let t_acpu = ns(|| {
            buf.copy_from_slice(&src);
            acpu::vector::render::rsqrt(&mut buf);
            std::hint::black_box(&buf);
        });
        let t_apple = ns(|| unsafe {
            vvrsqrtf(apple_dst.as_mut_ptr(), src.as_ptr(), &nn);
            std::hint::black_box(&apple_dst);
        });
        score.row("rsqrt", t_acpu, t_apple);
    }

    // ── recip: acpu vs vvrecf (both out-of-place) ──────────────────
    {
        let mut acpu_dst = vec![0.0f32; n];
        let mut apple_dst = vec![0.0f32; n];

        let t_acpu = ns(|| {
            acpu::vector::render::recip_to(&src, &mut acpu_dst);
            std::hint::black_box(&acpu_dst);
        });
        let t_apple = ns(|| unsafe {
            vvrecf(apple_dst.as_mut_ptr(), src.as_ptr(), &nn);
            std::hint::black_box(&apple_dst);
        });
        score.row("recip", t_acpu, t_apple);
    }

    // ── clamp: acpu vs vDSP_vclip (both out-of-place) ────────────────
    {
        let mut acpu_dst = vec![0.0f32; n];
        let mut apple_dst = vec![0.0f32; n];
        let lo = -100.0f32;
        let hi = 100.0f32;

        let t_acpu = ns(|| {
            acpu::vector::render::clamp_to(&wide, &mut acpu_dst, lo, hi);
            std::hint::black_box(&acpu_dst);
        });
        let t_apple = ns(|| unsafe {
            vDSP_vclip(wide.as_ptr(), 1, &lo, &hi, apple_dst.as_mut_ptr(), 1, nu);
            std::hint::black_box(&apple_dst);
        });
        score.row("clamp", t_acpu, t_apple);
    }

    // ── lerp: acpu vs vDSP_vintb ─────────────────────────────────────
    {
        let mut dst = vec![0.0f32; n];
        let mut apple_dst = vec![0.0f32; n];
        let t = 0.5f32;

        let t_acpu = ns(|| {
            acpu::vector::render::lerp(&mut dst, &a_vec, &b_vec, t);
            std::hint::black_box(&dst);
        });
        let t_apple = ns(|| unsafe {
            vDSP_vintb(
                a_vec.as_ptr(),
                1,
                b_vec.as_ptr(),
                1,
                &t,
                apple_dst.as_mut_ptr(),
                1,
                nu,
            );
            std::hint::black_box(&apple_dst);
        });
        score.row("lerp", t_acpu, t_apple);
    }

    println!();
    score.summary();
}
