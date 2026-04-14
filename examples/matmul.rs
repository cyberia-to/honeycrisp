//! Minimal ANE matmul demo — pure Rust, zero dependencies
//!
//! Compiles a 64×64 matmul on ANE, runs with identity matrix,
//! verifies output matches input.
//!
//! Run: cargo run --example matmul

use rane::mil;
use rane::{f32_to_fp16, fp16_to_f32, Buffer, Program};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("ANE Matmul — Pure Rust\n");

    // Build MIL program: y = x @ W
    // Input: [1, 64, 1, 128] fp16 (64 activations + 64 weights per channel)
    // Output: [1, 64, 1, 64] fp16
    let ic = 64;
    let oc = 64;
    let seq = 64;
    let program = mil::matmul(ic, oc, seq);
    let (in_ch, in_sp) = program.input_shape();
    let (out_ch, out_sp) = program.output_shape();
    println!("  MIL: matmul({ic}x{oc}, seq={seq})");
    println!(
        "  Input:  [1, {in_ch}, 1, {in_sp}] fp16 ({} KB)",
        program.input_size() / 1024
    );
    println!(
        "  Output: [1, {out_ch}, 1, {out_sp}] fp16 ({} KB)\n",
        program.output_size() / 1024
    );

    // Compile
    print!("  Compiling...");
    let mut model = Program::compile(&program, &[])?;
    println!(" OK");

    // Load into ANE
    print!("  Loading...");
    model.load()?;
    println!(" OK");

    // Prepare I/O surfaces
    let input = Buffer::new(program.input_size())?;
    let output = Buffer::new(program.output_size())?;

    // Fill input: activations = 1.0, weights = identity matrix
    input.write(|data| {
        for ch in 0..ic {
            for s in 0..seq {
                data[ch * in_sp + s] = f32_to_fp16(1.0);
            }
            for o in 0..oc {
                data[ch * in_sp + seq + o] = if ch == o { f32_to_fp16(1.0) } else { 0 };
            }
        }
    });
    println!("  Input: all 1.0 activations, identity weight matrix");

    // Run on ANE
    print!("  Evaluating on ANE...");
    model.run(&input, &output)?;
    println!(" OK\n");

    // Read output
    output.read(|data| {
        print!("  Output[0..8] = [");
        for i in 0..8 {
            if i > 0 {
                print!(", ");
            }
            print!("{:.1}", fp16_to_f32(data[i]));
        }
        println!("]");

        // Verify: identity matmul should give all 1.0 (each output = sum of row × identity col = 1.0)
        // Actually: [SEQ,IC] @ [IC,OC] with identity → output = ones × I = ones
        // But the matmul is (seq=64) @ (ic=64,oc=64), so each output is dot(ones_64, identity_col) = 1.0
        let all_ones = data[..out_ch * out_sp]
            .iter()
            .all(|&v| fp16_to_f32(v) == 1.0);
        if all_ones {
            println!("\n  VERIFIED: all {} output values = 1.0", out_ch * out_sp);
            println!("  Pure Rust → MIL → ANE bytecode → ANE hardware → correct result");
        } else {
            let nonzero: Vec<_> = data[..8].iter().map(|&v| fp16_to_f32(v)).collect();
            println!("\n  Output values: {:?}", nonzero);
        }
    });

    // Cleanup is automatic (Drop)
    Ok(())
}
