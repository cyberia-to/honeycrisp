//! Scan AMX opcodes 18-31 for undocumented instructions.
//!
//! Run with: `cargo run --example amx_opcode_scan --release`

use acpu::matrix::asm::{amx_op, OP_LDX, OP_LDY, OP_LDZ, OP_STZ};
use acpu::Matrix;
use std::alloc::{alloc_zeroed, dealloc, Layout};

unsafe fn aligned_f32(n: usize) -> *mut f32 {
    let layout = Layout::from_size_align(n * 4, 64).unwrap();
    alloc_zeroed(layout) as *mut f32
}

unsafe fn free_f32(ptr: *mut f32, n: usize) {
    let layout = Layout::from_size_align(n * 4, 64).unwrap();
    dealloc(ptr as *mut u8, layout);
}

fn main() {
    println!("=== AMX opcode scan (18-31) ===\n");

    let _ctx = Matrix::new().expect("AMX not available");

    unsafe {
        let x_buf = aligned_f32(16);
        let y_buf = aligned_f32(16);
        let zero_buf = aligned_f32(16);
        let z_buf = aligned_f32(64 * 16);

        // Load known data.
        for i in 0..16 {
            *x_buf.add(i) = (i + 1) as f32;
            *y_buf.add(i) = 1.0;
        }

        for opcode in 18u32..=31 {
            // Zero Z.
            for row in 0u8..64 {
                amx_op::<OP_LDZ>((zero_buf as u64) | ((row as u64) << 56));
            }

            // Load X[0] and Y[0].
            amx_op::<OP_LDX>((x_buf as u64) | (0u64 << 56));
            amx_op::<OP_LDY>((y_buf as u64) | (0u64 << 56));

            // Try this opcode with operand 0 (simplest).
            // Catch SIGILL — if opcode doesn't exist, it will crash.
            // We can't catch SIGILL easily in Rust, so we just try and see.
            println!("Trying opcode {}...", opcode);

            match opcode {
                18 => amx_op::<18>(0),
                19 => amx_op::<19>(0),
                20 => amx_op::<20>(0),
                21 => amx_op::<21>(0),
                22 => amx_op::<22>(0),
                23 => amx_op::<23>(0),
                24 => amx_op::<24>(0),
                25 => amx_op::<25>(0),
                26 => amx_op::<26>(0),
                27 => amx_op::<27>(0),
                28 => amx_op::<28>(0),
                29 => amx_op::<29>(0),
                30 => amx_op::<30>(0),
                31 => amx_op::<31>(0),
                _ => unreachable!(),
            }

            println!("  opcode {} executed (no SIGILL)", opcode);

            // Check if Z changed.
            let mut nz = 0;
            for row in 0u8..64 {
                let dst = z_buf.add(row as usize * 16) as *mut u8;
                amx_op::<OP_STZ>((dst as u64) | ((row as u64) << 56));
            }
            for i in 0..(64 * 16) {
                if (*z_buf.add(i)).to_bits() != 0 {
                    nz += 1;
                }
            }
            if nz > 0 {
                println!("  *** Z modified! {} non-zero elements ***", nz);
                // Print first non-zero Z row.
                for row in 0..64 {
                    let base = z_buf.add(row * 16);
                    let mut has = false;
                    for i in 0..16 {
                        if (*base.add(i)).to_bits() != 0 {
                            has = true;
                            break;
                        }
                    }
                    if has {
                        print!("  Z[{:2}] = [", row);
                        for i in 0..16 {
                            if i > 0 {
                                print!(", ");
                            }
                            print!("{:.1}", *base.add(i));
                        }
                        println!("]");
                        break; // just first row
                    }
                }
            } else {
                println!("  Z unchanged");
            }
        }

        free_f32(x_buf, 16);
        free_f32(y_buf, 16);
        free_f32(zero_buf, 16);
        free_f32(z_buf, 64 * 16);
    }
}
