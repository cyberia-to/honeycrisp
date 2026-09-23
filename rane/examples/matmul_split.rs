//! ANE matmul SPLIT-AND-SUM probe.
//!
//! Hypothesis: ANE's ~32768 fp16 overflow is in the MATMUL accumulator only.
//! ANE's `add` op may have separate fp32 hardware that can handle larger sums.
//!
//! Strategy:
//!   1. Split ic into chunks small enough that each chunk's fp16 tile < 32768
//!   2. Cast each chunk to fp32
//!   3. Sum fp32 chunks via MIL `add` ops
//!   4. Output is fp32 — final sum can exceed 32768
//!
//! If this works, Pearl-scale matmul (ic=2176, max_tile≈2M) becomes feasible on ANE.
//!
//! Run: cargo run -p rane --example matmul_split --release

use rane::mil::{self, OutputDtype};
use rane::{f32_to_fp16, Buffer, Program};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let oc = 64usize;
    let seq = 64usize;

    // Dump generated MIL once
    if std::env::var("DUMP_MIL").is_ok() {
        let p = mil::matmul_split(64, oc, seq, 2);
        println!("---- MIL for split(64,64,64,2) ----\n{}", p.text);
        return Ok(());
    }

    println!("ANE split-matmul + fp32 sum probe");
    println!("  hypothesis: matmul fp16 accum saturates at ~32768, but fp32 add is unconstrained");
    println!("  oc={oc}, seq={seq}\n");

    // ── case 0: structural test with fp16 (small value) ──────────────────────────
    // Tests whether multi-matmul + add compiles at all on ANE.
    println!("--- structural test: ic=64, v=2, expected=256 (fp16 sum) ---");
    test_split("fp16 1chunk", 64, oc, seq, 2.0, 1, 256, true)?;
    test_split("fp16 2chunk", 64, oc, seq, 2.0, 2, 256, true)?;
    test_split("fp32 2chunk", 64, oc, seq, 2.0, 2, 256, false)?;

    // ── case 1: ic=64, v=31 (full=61504 > 32768, chunk_2 each=30752 < 32768) ────
    println!("\n--- ic=64, v=31, expected=61504 (single fail, 2-chunk should win) ---");
    test_split("single fp32", 64, oc, seq, 31.0, 1, 61504, false)?;
    test_split("fp16 2chunk", 64, oc, seq, 31.0, 2, 61504, true)?;
    test_split("fp32 2chunk", 64, oc, seq, 31.0, 2, 61504, false)?;
    test_split("fp32 4chunk", 64, oc, seq, 31.0, 4, 61504, false)?;

    // ── case 2: ic=128, v=31 ─────────────────────────────────────────────────────
    println!("\n--- ic=128, v=31, expected=123008 ---");
    test_split("fp32 4chunk", 128, oc, seq, 31.0, 4, 123008, false)?;

    // ── case 3: Pearl-scale ────────────────────────────────────────────────────
    println!("\n--- ic=2176, v=31, expected=2091136 (Pearl full rank) ---");
    test_split("fp32 68chunk", 2176, oc, seq, 31.0, 68, 2091136, false)?;

    Ok(())
}

fn test_split(
    label: &str,
    ic: usize,
    oc: usize,
    seq: usize,
    fill_val: f32,
    n_chunks: usize,
    expected: i32,
    use_fp16_sum: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let program = if n_chunks == 1 && !use_fp16_sum {
        mil::matmul_cast(ic, oc, seq, OutputDtype::Fp32)
    } else if use_fp16_sum {
        mil::matmul_split_fp16(ic, oc, seq, n_chunks)
    } else {
        mil::matmul_split(ic, oc, seq, n_chunks)
    };

    let mut model = match Program::compile(&program, &[]) {
        Ok(m) => m,
        Err(e) => {
            println!(
                "  [{label:14}]  compile FAILED: {}",
                short_err(&format!("{e}"))
            );
            return Ok(());
        }
    };
    if let Err(e) = model.load() {
        println!(
            "  [{label:14}]  load FAILED: {}",
            short_err(&format!("{e}"))
        );
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
        println!(
            "  [{label:14}]  eval FAILED: {}",
            short_err(&format!("{e}"))
        );
        return Ok(());
    }

    let (val_str, correct) = if program.output_dtype == OutputDtype::Fp16 {
        output.read(|data| {
            let n = oc * seq;
            let v0 = rane::fp16_to_f32(data[0]);
            let n_correct = data[..n]
                .iter()
                .filter(|&&v| rane::fp16_to_f32(v) as i32 == expected)
                .count();
            (format!("{v0:.1}"), n_correct == n)
        })
    } else {
        output.read_f32(|data| {
            let n = oc * seq;
            let v0 = data[0];
            let n_correct = data[..n].iter().filter(|&&v| v as i32 == expected).count();
            (format!("{v0:.1}"), n_correct == n)
        })
    };

    let status = if correct {
        "ALL CORRECT ✓"
    } else {
        "WRONG ✗"
    };
    println!("  [{label:14}]  got={val_str:>12}  expected={expected:>8}  → {status}");
    Ok(())
}

fn short_err(s: &str) -> String {
    // Trim noisy NSError text; keep first 100 chars after a status keyword if present
    if let Some(idx) = s.find("status=") {
        let tail: String = s[idx..].chars().take(60).collect();
        return tail;
    }
    if let Some(idx) = s.find("err=(") {
        let tail: String = s[idx..].chars().take(80).collect();
        return tail;
    }
    s.chars().take(120).collect()
}
