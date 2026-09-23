//! TEST: does ANE matmul → cast(int16/int8) use a DIFFERENT accumulator path?
//!
//! If cast(int16) reads from the int32 accumulator directly (vs casting fp16),
//! values that overflow fp16 (>65504) should still be representable mod 2^16.
//!
//! Specifically:
//!   matmul produces int32 accumulator value K (e.g. 75000)
//!   - fp16 cast: 75000 > 65504 → saturates to inf
//!   - int16 cast: K & 0xFFFF = 9464 (wrapped) OR K clamped to 32767
//!   - If we see wrap-around, we have access to true int32 accumulator!
//!
//! Run: cargo run -p rane --example int_cast_test --release

use rane::mil::{self, OutputDtype};
use rane::{f32_to_fp16, fp16_to_f32, Buffer, Program};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let oc = 64usize;
    let seq = 64usize;

    println!("ANE matmul → int output probe");
    println!("  hypothesis: int8/int16 cast reads from int32 accumulator (not fp16)\n");

    // ── Case 1: small value (no fp16 overflow) ───────────────────────────────────
    // v=1, ic=64: tile = 1*1*64 = 64. Should be exact in all dtypes.
    println!("--- v=1, ic=64, expected=64 ---");
    test_dtype("fp16", 64, oc, seq, 1.0, 64, OutputDtype::Fp16)?;
    test_dtype("int16", 64, oc, seq, 1.0, 64, OutputDtype::Int16)?;
    test_dtype("int8", 64, oc, seq, 1.0, 64, OutputDtype::Int8)?;
    test_dtype("uint8", 64, oc, seq, 1.0, 64, OutputDtype::UInt8)?;

    // ── Case 2: moderate (fits fp16, exceeds int8 range) ──────────────────────────
    // v=10, ic=64: tile = 100*64 = 6400. fp16 ✓. int16 ✓. int8 (saturates).
    println!("\n--- v=10, ic=64, expected=6400 ---");
    test_dtype("fp16", 64, oc, seq, 10.0, 6400, OutputDtype::Fp16)?;
    test_dtype("int16", 64, oc, seq, 10.0, 6400, OutputDtype::Int16)?;
    test_dtype("int8", 64, oc, seq, 10.0, 6400, OutputDtype::Int8)?;

    // ── Case 3: matmul accumulator boundary (fp16 still fits but matmul fails) ─
    // v=23, ic=64: tile = 529*64 = 33856. fp16 says inf (matmul accum cap ~32768)
    // If int16 cast bypasses matmul fp16 accumulator, we'd get 33856 (overflows int16 too)
    // or 33856 mod 65536 = 33856 (fits int32 mod, but int16 saturates or wraps)
    println!("\n--- v=23, ic=64, expected=33856 (fp16 should be inf) ---");
    test_dtype("fp16", 64, oc, seq, 23.0, 33856, OutputDtype::Fp16)?;
    test_dtype("int16", 64, oc, seq, 23.0, 33856, OutputDtype::Int16)?;
    test_dtype("int8", 64, oc, seq, 23.0, 33856, OutputDtype::Int8)?;

    // ── Case 4: Pearl-level overflow (way past fp16) ─────────────────────────────
    // v=31, ic=128: tile = 961*128 = 123008. fp16 inf.
    // int32 mod 2^16 = 123008 & 0xFFFF = 57472. int8 = 123008 & 0xFF = 0.
    println!("\n--- v=31, ic=128, expected=123008 ---");
    test_dtype("fp16", 128, oc, seq, 31.0, 123008, OutputDtype::Fp16)?;
    test_dtype("int16", 128, oc, seq, 31.0, 123008, OutputDtype::Int16)?;
    test_dtype("int8", 128, oc, seq, 31.0, 123008, OutputDtype::Int8)?;

    Ok(())
}

fn test_dtype(
    label: &str,
    ic: usize,
    oc: usize,
    seq: usize,
    fill_val: f32,
    expected: i32,
    dtype: OutputDtype,
) -> Result<(), Box<dyn std::error::Error>> {
    let program = mil::matmul_cast(ic, oc, seq, dtype);
    let mut model = match Program::compile(&program, &[]) {
        Ok(m) => m,
        Err(e) => {
            println!(
                "  [{label:6}] compile FAILED: {}",
                format!("{e}").chars().take(80).collect::<String>()
            );
            return Ok(());
        }
    };
    if let Err(e) = model.load() {
        println!(
            "  [{label:6}] load FAILED: {}",
            format!("{e}").chars().take(80).collect::<String>()
        );
        return Ok(());
    }
    let input = Buffer::new(program.input_size())?;
    let output = Buffer::new(program.output_size())?;
    let sp = program.input_spatial;
    let vf = f32_to_fp16(fill_val);
    input.write(|data| {
        for d in data.iter_mut() {
            *d = 0;
        }
        for ch in 0..ic {
            for s in 0..seq {
                data[ch * sp + s] = vf;
            }
            for o in 0..oc {
                data[ch * sp + seq + o] = vf;
            }
        }
    });
    if let Err(e) = model.run(&input, &output) {
        println!(
            "  [{label:6}] eval FAILED: {}",
            format!("{e}").chars().take(80).collect::<String>()
        );
        return Ok(());
    }
    let (got_str, n) = (oc * seq, oc * seq);
    let val: String;
    let n_check: usize;
    match dtype {
        OutputDtype::Fp16 => {
            (val, n_check) = output.read(|data| {
                let v0 = fp16_to_f32(data[0]);
                let nc = data[..n]
                    .iter()
                    .filter(|&&v| fp16_to_f32(v) as i32 == expected)
                    .count();
                (format!("{v0:.1}"), nc)
            });
        }
        OutputDtype::Int16 => {
            (val, n_check) = output.read_i16(|data| {
                let v0 = data[0] as i32;
                let nc = data[..n].iter().filter(|&&v| v as i32 == expected).count();
                (format!("{v0}"), nc)
            });
        }
        OutputDtype::Int8 => {
            (val, n_check) = output.read_i8(|data| {
                let v0 = data[0] as i32;
                let nc = data[..n].iter().filter(|&&v| v as i32 == expected).count();
                (format!("{v0}"), nc)
            });
        }
        OutputDtype::UInt8 => {
            (val, n_check) = output.read_bytes(|data| {
                let v0 = data[0] as i32;
                let nc = data[..n]
                    .iter()
                    .filter(|&&v| v as i32 == (expected & 0xff))
                    .count();
                (format!("{v0}"), nc)
            });
        }
        _ => {
            (val, n_check) = (format!("?"), 0);
        }
    }
    let _ = got_str;
    println!("  [{label:6}]  got={val:>10}  expected={expected:>8}  ({n_check}/{n})");
    Ok(())
}
