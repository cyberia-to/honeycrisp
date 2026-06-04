//! Critical test: does ANE's `reduce_sum` have an fp32 internal accumulator?
//!
//! If reduce_sum accumulates many fp16 values internally in fp32 and outputs fp16,
//! the output is capped at 65504. But if we cast the reduce_sum result to fp32 at the
//! end, and the internal accumulator is fp32, we might bypass the fp16 ceiling
//! IF the reduce_sum dtype tracking has hidden fp32 precision.
//!
//! This tests by summing N=3+ chunks where each chunk ≤ 30752, so total > 65504.
//!
//! Run: cargo run -p rane --example reduce_test --release

use rane::{f32_to_fp16, mil, Buffer, Program};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let oc = 64usize;
    let seq = 64usize;

    println!("ANE reduce_sum fp32-accumulator test\n");

    // 3 chunks each producing fp16 tile=30752 → sum should be 92256 > 65504
    // If fp16 internal: output saturates at 65504 (cast to fp32 = 65504)
    // If fp32 internal: output is 92256 cast to fp32
    println!("--- 3 chunks of v=31 ic=32, each tile=30752 → expected sum=92256 (>65504) ---");
    test("3-chunk reduce_sum", 96, oc, seq, 31.0, 3, 92256)?;

    // 4 chunks: 123008
    println!("\n--- 4 chunks: expected sum=123008 ---");
    test("4-chunk reduce_sum", 128, oc, seq, 31.0, 4, 123008)?;

    // 8 chunks: 246016
    println!("\n--- 8 chunks: expected sum=246016 ---");
    test("8-chunk reduce_sum", 256, oc, seq, 31.0, 8, 246016)?;

    // Pearl-scale: 68 chunks of ic=32 → ic=2176, expected=2091136
    println!("\n--- 68 chunks (Pearl scale): expected sum=2091136 ---");
    test("Pearl 68-chunk", 2176, oc, seq, 31.0, 68, 2091136)?;

    Ok(())
}

fn test(
    label: &str,
    ic: usize,
    oc: usize,
    seq: usize,
    fill_val: f32,
    n_chunks: usize,
    expected: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    let program = mil::matmul_split_reduce(ic, oc, seq, n_chunks);
    let mut model = match Program::compile(&program, &[]) {
        Ok(m) => m,
        Err(e) => {
            let s = format!("{e}");
            let short: String = s.chars().take(100).collect();
            println!("  [{label}] compile FAILED: {short}");
            return Ok(());
        }
    };
    if let Err(e) = model.load() {
        let s = format!("{e}");
        let short: String = s.chars().take(100).collect();
        println!("  [{label}] load FAILED: {short}");
        return Ok(());
    }
    let input = Buffer::new(program.input_size())?;
    let output = Buffer::new(program.output_size())?;
    let sp = program.input_spatial;
    let vfill = f32_to_fp16(fill_val);
    input.write(|data| {
        for d in data.iter_mut() {
            *d = 0;
        }
        for ch in 0..ic {
            for s in 0..seq {
                data[ch * sp + s] = vfill;
            }
            for o in 0..oc {
                data[ch * sp + seq + o] = vfill;
            }
        }
    });
    if let Err(e) = model.run(&input, &output) {
        let s = format!("{e}");
        let short: String = s.chars().take(100).collect();
        println!("  [{label}] eval FAILED: {short}");
        return Ok(());
    }
    let (val_str, correct) = output.read_f32(|data| {
        let n = oc * seq;
        let v0 = data[0];
        let n_correct = data[..n].iter().filter(|&&v| v as i32 == expected).count();
        (format!("{v0:.1}"), n_correct == n)
    });
    let status = if correct { "✓" } else { "✗" };
    println!("  [{label:24}]  got={val_str:>14}  expected={expected:>10}  {status}");
    Ok(())
}
