//! ANE fp16 accumulator overflow probe.
//!
//! Determines ANE's actual fp16 matmul overflow threshold.
//!
//! FINDINGS:
//!   - int32 cast: rejected by ANECCompile (InvalidMILProgram).
//!   - fp32 cast: compiles OK but gives SAME result as fp16 (cast is post-matmul).
//!   - ANE fp16 matmul overflows to inf at tile > ~32768 (NOT at fp16 max 65504).
//!   - The overflow is in the ANE's internal fp16 accumulator, not the output format.
//!
//! CONSEQUENCE for Pearl PoW:
//!   Pearl tiles reach 31*31*2176 ≈ 2,091,136 — 64× above the ~32768 overflow threshold.
//!   ANE is definitively excluded from Pearl PoW. No MIL-level workaround exists.
//!
//! Run: cargo run -p rane --example matmul_cast --release

use rane::mil::{self, OutputDtype};
use rane::{f32_to_fp16, fp16_to_f32, Buffer, Program};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let oc = 64usize;
    let seq = 64usize;

    println!("ANE fp16 accumulator overflow probe");
    println!("  oc={oc}, seq={seq}");
    println!("  fp32 cast: compiles OK but output == fp16 output (no fp32 accumulator)\n");

    // ── threshold: ic=64, vary fill_val ─────────────────────────────────────────
    // Overflow at ~32768: v=22 tile=30976 works; v=23 tile=33856 fails.
    let ic = 64usize;
    println!("--- threshold scan ic=64 ---");
    for &v in &[16.0f32, 20.0, 22.0, 23.0, 24.0] {
        let expected_i = (v * v * ic as f32) as i32;
        run_variant_val(
            &format!("fp16 v={v}"),
            ic,
            oc,
            seq,
            OutputDtype::Fp16,
            expected_i,
            v,
        )?;
        run_variant_val(
            &format!("fp32 v={v}"),
            ic,
            oc,
            seq,
            OutputDtype::Fp32,
            expected_i,
            v,
        )?;
    }

    // ── threshold: vary ic with v=31 ─────────────────────────────────────────────
    // ic=32 tile=30752 works; ic=40 tile=38440 fails.
    println!("\n--- threshold scan v=31 ---");
    for &ic2 in &[8usize, 16, 24, 32, 40, 64] {
        let expected = 31i32 * 31 * ic2 as i32;
        run_variant_val(
            &format!("fp16 ic={ic2}"),
            ic2,
            oc,
            seq,
            OutputDtype::Fp16,
            expected,
            31.0,
        )?;
    }

    // ── Pearl scale ───────────────────────────────────────────────────────────────
    println!("\n--- Pearl scale (ic=2176) ---");
    let expected = 31i32 * 31 * 2176i32;
    run_variant_val(
        "fp32 ic=2176",
        2176,
        oc,
        seq,
        OutputDtype::Fp32,
        expected,
        31.0,
    )?;

    Ok(())
}

fn run_variant_val(
    label: &str,
    ic: usize,
    oc: usize,
    seq: usize,
    dtype: OutputDtype,
    expected_tile: i32,
    fill_val: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let program = mil::matmul_cast(ic, oc, seq, dtype);
    let sp = program.input_spatial;

    let mut model = match Program::compile(&program, &[]) {
        Ok(m) => m,
        Err(e) => {
            println!("  [{label}] compile FAILED: {e}");
            return Ok(());
        }
    };
    if let Err(e) = model.load() {
        println!("  [{label}] load FAILED: {e}");
        return Ok(());
    }

    let input = Buffer::new(program.input_size())?;
    let output = Buffer::new(program.output_size())?;

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
        println!("  [{label}] eval FAILED: {e}");
        return Ok(());
    }

    let (first_val_str, correct, n_correct, n_total) = match dtype {
        OutputDtype::Fp16 => output.read(|data| {
            let n = oc * seq;
            let v0 = fp16_to_f32(data[0]);
            let nc = data[..n]
                .iter()
                .filter(|&&v| fp16_to_f32(v) as i32 == expected_tile)
                .count();
            (format!("{v0:.1}"), v0 as i32 == expected_tile, nc, n)
        }),
        OutputDtype::Fp32 => output.read_f32(|data| {
            let n = oc * seq;
            let v0 = data[0];
            let nc = data[..n]
                .iter()
                .filter(|&&v| v as i32 == expected_tile)
                .count();
            (format!("{v0:.1}"), v0 as i32 == expected_tile, nc, n)
        }),
        OutputDtype::Int32 => output.read_i32(|data| {
            let n = oc * seq;
            let v0 = data[0];
            let nc = data[..n].iter().filter(|&&v| v == expected_tile).count();
            (format!("{v0}"), v0 == expected_tile, nc, n)
        }),
    };

    let status = if correct { "✓" } else { "✗" };
    println!(
        "  [{label:18}]  got={first_val_str:>10}  expected={expected_tile:>8}  {n_correct}/{n_total}  {status}"
    );
    Ok(())
}
