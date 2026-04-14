//! ANE driver benchmark — measures pure driver overhead
//!
//! Run: cargo run --release --example bench

use rane::{cast_f16_f32, cast_f32_f16, Buffer, Program};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("ANE Driver Benchmark\n");

    // ── Surface creation ──
    for size_kb in [1, 64, 1024] {
        let bytes = size_kb * 1024;
        let iters = 100;
        let t0 = Instant::now();
        for _ in 0..iters {
            let _s = Buffer::new(bytes).unwrap();
        }
        let avg = t0.elapsed().as_secs_f64() / iters as f64 * 1000.0;
        println!("Surface create ({:>4} KB): {:.3} ms", size_kb, avg);
    }
    println!();

    // ── MIL compile ──
    for (ic, oc, seq) in [(32, 32, 32), (64, 64, 64), (128, 128, 64)] {
        let iters = 10;
        let p = rane::mil::matmul(ic, oc, seq);
        let t0 = Instant::now();
        for _ in 0..iters {
            let _m = Program::compile(&p, &[]).unwrap();
        }
        let avg = t0.elapsed().as_secs_f64() / iters as f64 * 1000.0;
        println!("Compile matmul({ic}x{oc}, seq={seq}): {:.1} ms", avg);
    }
    println!();

    // ── Load / unload ──
    {
        let p = rane::mil::matmul(64, 64, 64);
        let iters = 20;
        let t0 = Instant::now();
        for _ in 0..iters {
            let mut m = Program::compile(&p, &[]).unwrap();
            m.load().unwrap();
            m.unload().unwrap();
        }
        let avg = t0.elapsed().as_secs_f64() / iters as f64 * 1000.0;
        println!("Compile+load+unload (64x64): {:.1} ms", avg);
    }
    println!();

    // ── Dispatch overhead ──
    {
        let ic = 64;
        let oc = 64;
        let seq = 64;
        let p = rane::mil::matmul(ic, oc, seq);
        let mut model = Program::compile(&p, &[]).unwrap();
        model.load().unwrap();

        let input = Buffer::new(p.input_size()).unwrap();
        let output = Buffer::new(p.output_size()).unwrap();

        // warmup
        for _ in 0..5 {
            model.run(&input, &output).unwrap();
        }

        let iters = 100;
        let t0 = Instant::now();
        for _ in 0..iters {
            model.run(&input, &output).unwrap();
        }
        let avg = t0.elapsed().as_secs_f64() / iters as f64 * 1000.0;
        println!("Dispatch overhead (64x64 matmul): {:.3} ms", avg);

        // Larger matmul
        let p2 = rane::mil::matmul(256, 256, 64);
        let mut m2 = Program::compile(&p2, &[]).unwrap();
        m2.load().unwrap();
        let in2 = Buffer::new(p2.input_size()).unwrap();
        let out2 = Buffer::new(p2.output_size()).unwrap();
        for _ in 0..5 {
            m2.run(&in2, &out2).unwrap();
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            m2.run(&in2, &out2).unwrap();
        }
        let avg = t0.elapsed().as_secs_f64() / iters as f64 * 1000.0;
        println!("Dispatch overhead (256x256 matmul): {:.3} ms", avg);
    }
    println!();

    // ── fp16 conversion throughput ──
    {
        let n = 16 * 1024 * 1024; // 16M elements
        let src_f32: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let mut dst_f16 = vec![0u16; n];
        let mut dst_f32 = vec![0.0f32; n];

        let iters = 10;

        // f32 → fp16
        let t0 = Instant::now();
        for _ in 0..iters {
            cast_f32_f16(&mut dst_f16, &src_f32);
        }
        let elapsed = t0.elapsed().as_secs_f64() / iters as f64;
        let gbps = (n * 4) as f64 / elapsed / 1e9;
        println!(
            "f32→fp16 (16M): {:.2} ms, {:.1} GB/s",
            elapsed * 1000.0,
            gbps
        );

        // fp16 → f32
        let t0 = Instant::now();
        for _ in 0..iters {
            cast_f16_f32(&mut dst_f32, &dst_f16);
        }
        let elapsed = t0.elapsed().as_secs_f64() / iters as f64;
        let gbps = (n * 2) as f64 / elapsed / 1e9;
        println!(
            "fp16→f32 (16M): {:.2} ms, {:.1} GB/s",
            elapsed * 1000.0,
            gbps
        );
    }

    Ok(())
}
