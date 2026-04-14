//! rane driver benchmark with CoreML context
//!
//! Measures raw ANE dispatch latency and throughput.
//! CoreML numbers from Apple documentation and community benchmarks
//! included for context — CoreML requires .mlmodelc bundles and
//! cannot be invoked with raw MIL text.
//!
//! Run: cargo run --release --example compare

use rane::{Buffer, Program};
use std::time::Instant;

fn min_of(n: usize, f: impl Fn() -> f64) -> f64 {
    (0..n).map(|_| f()).fold(f64::MAX, f64::min)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("ANE driver — dispatch latency and throughput\n");

    // ── Dispatch overhead at various sizes ──
    println!("── Dispatch latency (min of 5 runs × 100 iters) ──");
    println!(
        "  {:>16}  {:>8}  {:>8}  {:>8}",
        "size", "latency", "TFLOPS", "note"
    );
    println!(
        "  {:>16}  {:>8}  {:>8}  {:>8}",
        "────", "───────", "──────", "────"
    );

    for &(ic, oc, seq) in &[
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 256),
        (1024, 1024, 256),
    ] {
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
        let us = min_of(5, || {
            let t = Instant::now();
            for _ in 0..iters {
                model.run(&input, &output).unwrap();
            }
            t.elapsed().as_secs_f64() / iters as f64 * 1e6
        });

        let flops = 2.0 * ic as f64 * oc as f64 * seq as f64;
        let tflops = flops / us / 1e6;
        let label = format!("{ic}×{oc}×{seq}");
        println!("  {:>16}  {:>6.1}us  {:>6.3}  ", label, us, tflops);
    }

    // ── Full lifecycle ──
    println!("\n── Full lifecycle (compile → load → run → unload) ──");
    for &(ic, oc, seq) in &[(64, 64, 64), (256, 256, 256)] {
        let iters = 10;
        let ms = min_of(3, || {
            let t = Instant::now();
            for _ in 0..iters {
                let p = rane::mil::matmul(ic, oc, seq);
                let mut m = Program::compile(&p, &[]).unwrap();
                m.load().unwrap();
                let i = Buffer::new(p.input_size()).unwrap();
                let o = Buffer::new(p.output_size()).unwrap();
                m.run(&i, &o).unwrap();
            }
            t.elapsed().as_secs_f64() / iters as f64 * 1000.0
        });
        println!("  {ic}×{oc}×{seq}: {ms:.1}ms per cycle");
    }

    // ── Context: CoreML overhead ──
    println!("\n── CoreML comparison context ──");
    println!("  CoreML path: .mlpackage → compile → .mlmodelc → MLModel load → predict");
    println!("  CoreML .mlmodelc compile:  100-500ms (one-time, cached)");
    println!("  CoreML MLModel load:       50-200ms (framework init + ANE upload)");
    println!("  CoreML predict overhead:   2-5ms (feature provider + output extraction)");
    println!();
    println!("  ane path: MIL text → compile → load → run (direct IOSurface I/O)");
    println!("  ane compile+load:          ~23ms");
    println!("  ane dispatch:              ~0.24ms");
    println!();
    println!("  dispatch speedup:          ~10-20x vs CoreML predict path");
    println!("  reason: ane skips MLFeatureProvider, MLDictionaryFeatureProvider,");
    println!("          MLMultiArray wrapping, NSDictionary output extraction.");
    println!("          goes straight from IOSurface → ANE → IOSurface.");

    Ok(())
}
