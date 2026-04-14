//! AI inference pipeline benchmark: acpu vs Apple Accelerate.
//! Measures operations relevant to CPU-side transformer inference.

#[path = "common.rs"]
mod common;
use common::*;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        o: i32,
        ta: i32,
        tb: i32,
        m: i32,
        n: i32,
        k: i32,
        a: f32,
        ap: *const f32,
        lda: i32,
        bp: *const f32,
        ldb: i32,
        b: f32,
        cp: *mut f32,
        ldc: i32,
    );
}

fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(60));
        eprintln!("!!! 60s TIMEOUT !!!");
        std::process::exit(1);
    });

    let caps = acpu::probe::scan();
    println!("acpu AI inference benchmark — {:?}", caps.chip);
    println!();

    // ── 1. SGEMM at inference-relevant sizes (acpu vs Apple) ─────────────
    // Apple Accelerate runs FIRST to avoid BLASStateRetain deadlock

    println!("--- SGEMM (inference sizes, acpu vs Apple cblas) ---");
    println!(
        "{:<30} {:>10} {:>10} {:>10} {:>7}",
        "operation", "acpu GF", "apple GF", "time", "ratio"
    );

    let sizes: [(usize, usize, usize, &str); 4] = [
        (4096, 4096, 4096, "FFN weight (4K×4K)"),
        (4096, 11008, 4096, "llama FFN up (4K×11K)"),
        (1, 4096, 4096, "single-token matvec"),
        (32, 4096, 128, "attention QK^T (32 heads)"),
    ];

    // Apple first (all sizes)
    let mut apple_results = Vec::new();
    for &(m, n, k, _) in &sizes {
        let a = vec![0.1f32; m * k];
        let b = vec![0.1f32; k * n];
        let mut c = vec![0.0f32; m * n];
        let iters = if m * n * k > 100_000_000 { 3 } else { 20 };
        for _ in 0..2 {
            unsafe {
                cblas_sgemm(
                    101,
                    111,
                    111,
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
        let t = best_of(
            || unsafe {
                cblas_sgemm(
                    101,
                    111,
                    111,
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
                std::hint::black_box(&c);
            },
            iters,
        );
        apple_results.push(t);
    }

    // acpu second
    for (idx, &(m, n, k, label)) in sizes.iter().enumerate() {
        let a = vec![0.1f32; m * k];
        let b = vec![0.1f32; k * n];
        let mut c = vec![0.0f32; m * n];
        acpu::matmul_f32_set(&a, &b, &mut c, m, n, k);
        let iters = if m * n * k > 100_000_000 { 3 } else { 20 };
        let t = best_of(
            || {
                acpu::matmul_f32_set(&a, &b, &mut c, m, n, k);
                std::hint::black_box(&c);
            },
            iters,
        );
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let acpu_gf = flops / t as f64;
        let apple_gf = flops / apple_results[idx] as f64;
        let ratio = acpu_gf / apple_gf;
        let marker = if ratio > 1.05 {
            "←"
        } else if ratio >= 0.95 {
            "≈"
        } else {
            ""
        };
        println!(
            "  {:<28} {:>8.0} GF {:>8.0} GF {:>6.1}ms {:>5.2}×{}",
            label,
            acpu_gf,
            apple_gf,
            t as f64 / 1e6,
            ratio,
            marker
        );
    }

    // ── 1b. MATVEC (single-token inference) ────────────────────────────

    println!();
    println!("--- MATVEC (1×K×N, single-token, acpu vs Apple) ---");
    println!(
        "{:<30} {:>10} {:>10} {:>8} {:>7}",
        "shape", "acpu", "apple", "time", "ratio"
    );

    // Apple matvec first
    let mv_sizes: [(usize, &str); 4] = [
        (512, "1×512×512"),
        (1024, "1×1024×1024"),
        (4096, "1×4096×4096"),
        (11008, "1×4096×11008"),
    ];
    let mut apple_mv = Vec::new();
    for &(dim, _) in &mv_sizes {
        let k = if dim == 11008 { 4096 } else { dim };
        let n = dim;
        let a = vec![0.1f32; k];
        let b = vec![0.1f32; k * n];
        let mut c = vec![0.0f32; n];
        for _ in 0..3 {
            unsafe {
                cblas_sgemm(
                    101,
                    111,
                    111,
                    1,
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
        let t = best_of(
            || unsafe {
                cblas_sgemm(
                    101,
                    111,
                    111,
                    1,
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
                std::hint::black_box(&c);
            },
            50,
        );
        apple_mv.push(t);
    }

    // acpu matvec
    for (idx, &(dim, label)) in mv_sizes.iter().enumerate() {
        let k = if dim == 11008 { 4096 } else { dim };
        let n = dim;
        let a = vec![0.1f32; k];
        let b = vec![0.1f32; k * n];
        let mut c = vec![0.0f32; n];
        acpu::gemm::matvec::matvec_f32_set(&a, &b, &mut c, n, k);
        let t = best_of(
            || {
                acpu::gemm::matvec::matvec_f32_set(&a, &b, &mut c, n, k);
                std::hint::black_box(&c);
            },
            50,
        );
        let flops = 2.0 * k as f64 * n as f64;
        let acpu_gf = flops / t as f64;
        let apple_gf = flops / apple_mv[idx] as f64;
        let ratio = acpu_gf / apple_gf;
        let marker = if ratio > 1.05 {
            "←"
        } else if ratio >= 0.95 {
            "≈"
        } else {
            ""
        };
        println!(
            "  {:<28} {:>8.1} GF {:>8.1} GF {:>6.0}μs {:>5.2}×{}",
            label,
            acpu_gf,
            apple_gf,
            t as f64 / 1e3,
            ratio,
            marker
        );
    }

    // ── 2. QUANTIZED DOT PRODUCT ─────────────────────────────────────────

    println!();
    println!("--- QUANTIZED OPS ---");

    let n = 4096usize;
    let i8_a: Vec<i8> = (0..n).map(|i| ((i * 7) % 255) as i8).collect();
    let i8_b: Vec<i8> = (0..n).map(|i| ((i * 13) % 255) as i8).collect();

    let dot_ns = best_of(
        || {
            std::hint::black_box(acpu::vector::integer::dot_i8(&i8_a, &i8_b));
        },
        500,
    );
    println!(
        "  i8 dot (SDOT, {n}):    {:>5}ns  ({:.1} Gop/s)",
        dot_ns,
        n as f64 / dot_ns as f64
    );

    let u8_a: Vec<u8> = (0..n).map(|i| (i % 200) as u8).collect();
    let u8_b: Vec<u8> = (0..n).map(|i| ((i * 3 + 50) % 200) as u8).collect();
    let sad_ns = best_of(
        || {
            std::hint::black_box(acpu::vector::integer_fused::sad_u8(&u8_a, &u8_b));
        },
        500,
    );
    println!(
        "  u8 SAD ({n}):          {:>5}ns  ({:.1} Gop/s)",
        sad_ns,
        n as f64 / sad_ns as f64
    );

    // ── 3. ACTIVATIONS ───────────────────────────────────────────────────

    println!();
    println!("--- ACTIVATIONS (4096 f32) ---");

    let src: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * 6.0 - 3.0).collect();
    let mut buf = vec![0.0f32; n];

    for &(name, f) in &[
        ("gelu", acpu::vector::gelu as fn(&mut [f32])),
        ("silu", acpu::vector::silu as fn(&mut [f32])),
        ("sigmoid", acpu::vector::sigmoid as fn(&mut [f32])),
    ] {
        let t = best_of(
            || {
                buf.copy_from_slice(&src);
                f(&mut buf);
                std::hint::black_box(&buf);
            },
            200,
        );
        println!(
            "  {:<12} {:>5}ns  ({:.0} Melem/s)",
            name,
            t,
            n as f64 / t as f64 * 1000.0
        );
    }

    // ── 4. RMSNORM + ROPE PIPELINE ───────────────────────────────────────

    println!();
    println!("--- NORM + ROPE PIPELINE (dim=4096) ---");

    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 20.0).collect();
    let w: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.1).collect();
    let freqs: Vec<f32> = (0..n / 2)
        .map(|i| 1.0 / 10000f32.powf(2.0 * i as f32 / n as f32))
        .collect();
    let mut norm_out = vec![0.0f32; n];
    let mut rope_out = vec![0.0f32; n];

    let norm_ns = best_of(
        || {
            acpu::vector::normalize(&mut norm_out, &x, &w, 1e-5);
            std::hint::black_box(&norm_out);
        },
        200,
    );

    let rope_ns = best_of(
        || {
            acpu::vector::rotate(&mut rope_out, &x, &freqs, 42);
            std::hint::black_box(&rope_out);
        },
        200,
    );

    let pipeline_ns = norm_ns + rope_ns;
    println!("  rmsnorm:     {:>5}ns", norm_ns);
    println!("  rope:        {:>5}ns", rope_ns);
    println!("  pipeline:    {:>5}ns", pipeline_ns);

    // ── 5. SOFTMAX AT ATTENTION SIZES ────────────────────────────────────

    println!();
    println!("--- SOFTMAX (attention sequence lengths) ---");

    for &seq_len in &[128, 512, 2048, 8192] {
        let mut attn: Vec<f32> = (0..seq_len).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let t = best_of(
            || {
                attn.copy_from_slice(
                    &(0..seq_len)
                        .map(|i| (i as f32) * 0.01 - 5.0)
                        .collect::<Vec<_>>(),
                );
                acpu::vector::softmax(&mut attn);
                std::hint::black_box(&attn);
            },
            200,
        );
        println!(
            "  seq_len={:<5} {:>6}ns  ({:.0}ns/elem)",
            seq_len,
            t,
            t as f64 / seq_len as f64
        );
    }

    println!();
    println!("=== end ===");
}
