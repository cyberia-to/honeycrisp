/// Verify compile_anec path: binary net.plist → load → evaluate on ANE hardware.
///
/// Uses a minimal 16-in × 8-out Int8 1x1 Conv ANEC IR model.
/// IOSurface layout: ANEC interleaved, 4 channels per group, 64 bytes/group.
use rane::{AneError, Buffer, Program};
use std::fs;

fn main() -> Result<(), AneError> {
    let plist_path = "/tmp/anecir_minimal/net.plist";
    let weights_path = "/tmp/anecir_minimal/net.additional.weights";

    let plist_data = fs::read(plist_path).expect("net.plist not found at /tmp/anecir_minimal/");
    let weights_data = fs::read(weights_path).expect("weights not found");

    println!(
        "plist={} bytes  weights={} bytes",
        plist_data.len(),
        weights_data.len()
    );

    // Compile via ANEC IR path (isMILModel=0, invokes ANECCompile)
    let mut prog =
        Program::compile_anec(&plist_data, &[("net.additional.weights", &weights_data)])?;
    println!("compile ok");

    prog.load()?;
    println!("load ok");

    // ANEC interleaved layout: 4 channels per group, 64 bytes per group.
    //   16 in → 4 groups × 64 = 256 bytes
    //    8 out → 2 groups × 64 = 128 bytes
    let input = Buffer::with_anec_channels(16)?;
    let output = Buffer::with_anec_channels(8)?;

    // Fill input: channel ch = (ch+1) as fp16  [1.0 .. 16.0]
    // Each group is ANEC_GROUP_STRIDE bytes = 32 u16 slots; only slots [0..3] carry data.
    input.write(|buf| {
        let stride = Buffer::ANEC_GROUP_STRIDE / 2; // bytes→u16 slots
        for g in 0..4usize {
            for c in 0..4usize {
                buf[g * stride + c] = rane::f32_to_fp16((g * 4 + c + 1) as f32);
            }
        }
    });

    // Sentinel output to distinguish ANE writes from un-touched memory
    output.write(|buf| {
        for v in buf.iter_mut() {
            *v = 0xDEAD_u16;
        }
    });

    unsafe { prog.run_direct(input.as_raw(), output.as_raw())? };
    println!("evaluate ok");

    // Read 8 output channels from their interleaved positions
    output.read(|buf| {
        let stride = Buffer::ANEC_GROUP_STRIDE / 2;
        print!("output [0..7] (fp32): ");
        for g in 0..2usize {
            for c in 0..4usize {
                print!("{:.4} ", rane::fp16_to_f32(buf[g * stride + c]));
            }
        }
        println!();
        let sentinel = buf.iter().filter(|&&v| v == 0xDEAD_u16).count();
        println!("unchanged sentinel words: {} / {}", sentinel, buf.len());
    });

    prog.unload()?;
    println!("unload ok");

    Ok(())
}
