//! Probe ANE matmul accumulator precision: fp16 or fp32?
//!
//! Method: feed int-valued fp16 inputs, run matmul on ANE, compare to scalar i64 reference.
//! - fp16 accumulator → small-K errors (partial sums lose mantissa bits)
//! - fp32 accumulator → exact until final fp16 cast rounds the result
//!
//! Run: cargo run -p rane --example ane_accum_precision --release

use rane::mil;
use rane::{f32_to_fp16, fp16_to_f32, Buffer, Program};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ANE matmul accumulator precision probe ===\n");
    println!("Setup: A_int * B_int via fp16 inputs, compare ANE result to i64 reference.\n");

    // Test multiple K values
    let cases: &[(usize, &str)] = &[
        (64, "K=64 (single ANE tile)"),
        (128, "K=128"),
        (256, "K=256"),
        (512, "K=512"),
        (1024, "K=1024"),
        (2048, "K=2048"),
    ];

    // Test multiple integer ranges
    let ranges: &[(i32, i32, &str)] = &[
        (-8, 7, "int4   [-8, 7]"),
        (-32, 31, "int6   [-32, 31]"),
        (-64, 63, "int7   [-64, 63]   (Pearl spec)"),
    ];

    for (lo, hi, range_label) in ranges {
        println!("=== range: {} ===", range_label);
        for &(k, label) in cases {
            run_one(k, *lo, *hi, label)?;
        }
        println!();
    }

    Ok(())
}

fn run_one(k: usize, lo: i32, hi: i32, label: &str) -> Result<(), Box<dyn std::error::Error>> {
    // matmul(ic = K, oc = 64, seq = 64): one matmul with K dot products in parallel
    // Layout: [1, ic=K, 1, seq+oc=128]
    //   x[k][s] = A[k] for s in 0..seq  (broadcast A across seq axis)
    //   x[k][seq+o] = B[k] for o in 0..oc  (broadcast B across oc axis)
    // Output[s][o] = sum_k A[k] * B[k] = same value for all (s, o)
    let ic = k;
    let oc = 64;
    let seq = 64;
    let program = mil::matmul(ic, oc, seq);
    let mut model = Program::compile(&program, &[])?;
    model.load()?;

    let input = Buffer::new(program.input_size())?;
    let output = Buffer::new(program.output_size())?;

    let range = (hi - lo + 1) as u32;
    let gen = |seed: u32| -> i32 {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1442695041);
        s ^= s >> 16;
        s = s.wrapping_mul(0x85ebca6b);
        s ^= s >> 13;
        s = s.wrapping_mul(0xc2b2ae35);
        s ^= s >> 16;
        ((s % range) as i32) + lo
    };

    let mut a_vals = vec![0i32; ic];
    let mut b_vals = vec![0i32; ic];
    for kk in 0..ic {
        a_vals[kk] = gen(kk as u32);
        b_vals[kk] = gen((kk as u32) + 0x1000);
    }
    let scalar_ref: i64 = (0..ic).map(|kk| (a_vals[kk] as i64) * (b_vals[kk] as i64)).sum();

    input.write(|data| {
        let sp = seq + oc;
        for kk in 0..ic {
            let a_fp = f32_to_fp16(a_vals[kk] as f32);
            let b_fp = f32_to_fp16(b_vals[kk] as f32);
            for s in 0..seq {
                data[kk * sp + s] = a_fp;
            }
            for o in 0..oc {
                data[kk * sp + seq + o] = b_fp;
            }
        }
    });

    model.run(&input, &output)?;

    let mut ane_val: f32 = 0.0;
    let mut all_equal = true;
    output.read(|data| {
        ane_val = fp16_to_f32(data[0]);
        for i in 1..(seq * oc) {
            if fp16_to_f32(data[i]) != ane_val {
                all_equal = false;
            }
        }
    });

    let err = (ane_val - scalar_ref as f32).abs();
    let exact_int = ane_val.fract() == 0.0 && (ane_val as i64) == scalar_ref;
    let fp16_ref = f32_to_fp16(scalar_ref as f32);
    let exact_fp16 = fp16_ref == f32_to_fp16(ane_val);
    let oor = scalar_ref.unsigned_abs() > 65504;
    let status = if exact_int { "INT-EXACT " }
                 else if exact_fp16 { "FP16-EXACT" }
                 else if oor { "OOR-fp16  " }
                 else { "MISMATCH  " };
    let label_pad = format!("  {label}");
    println!(
        "{label_pad:<32}ref={scalar_ref:>8}  ane={ane_val:>11.2}  err={err:>9.2}  {status}{}",
        if !all_equal { " [VARYING-OUTPUTS]" } else { "" }
    );

    Ok(())
}
