//! Full AMX reverse engineering: enumerate all opcodes and operand modes.
//!
//! For each valid opcode: test what registers change, measure throughput,
//! probe operand bits.
//!
//! Run with: `cargo run --example amx_reverse --release`

use acpu::matrix::asm::{amx_op, OP_LDX, OP_LDY, OP_LDZ, OP_STX, OP_STY, OP_STZ};
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

/// Snapshot: save all X, Y, Z registers.
struct Snapshot {
    x: [[u32; 16]; 8],  // 8 X rows × 16 u32
    y: [[u32; 16]; 8],  // 8 Y rows × 16 u32
    z: [[u32; 16]; 64], // 64 Z rows × 16 u32
}

impl Snapshot {
    unsafe fn capture(buf: *mut u8) -> Self {
        let mut s = Snapshot {
            x: [[0; 16]; 8],
            y: [[0; 16]; 8],
            z: [[0; 16]; 64],
        };
        for r in 0u8..8 {
            amx_op::<OP_STX>((buf as u64) | ((r as u64) << 56));
            let p = buf as *const u32;
            for i in 0..16 {
                s.x[r as usize][i] = *p.add(i);
            }
        }
        for r in 0u8..8 {
            amx_op::<OP_STY>((buf as u64) | ((r as u64) << 56));
            let p = buf as *const u32;
            for i in 0..16 {
                s.y[r as usize][i] = *p.add(i);
            }
        }
        for r in 0u8..64 {
            amx_op::<OP_STZ>((buf as u64) | ((r as u64) << 56));
            let p = buf as *const u32;
            for i in 0..16 {
                s.z[r as usize][i] = *p.add(i);
            }
        }
        s
    }

    fn diff(&self, other: &Snapshot) -> Vec<String> {
        let mut diffs = Vec::new();
        for r in 0..8 {
            for i in 0..16 {
                if self.x[r][i] != other.x[r][i] {
                    diffs.push(format!(
                        "X[{}][{}]: {:08x}→{:08x}",
                        r, i, other.x[r][i], self.x[r][i]
                    ));
                }
            }
        }
        for r in 0..8 {
            for i in 0..16 {
                if self.y[r][i] != other.y[r][i] {
                    diffs.push(format!(
                        "Y[{}][{}]: {:08x}→{:08x}",
                        r, i, other.y[r][i], self.y[r][i]
                    ));
                }
            }
        }
        for r in 0..64 {
            for i in 0..16 {
                if self.z[r][i] != other.z[r][i] {
                    diffs.push(format!(
                        "Z[{}][{}]: {:08x}→{:08x}",
                        r, i, other.z[r][i], self.z[r][i]
                    ));
                }
            }
        }
        diffs
    }

    fn changed_regs(&self, other: &Snapshot) -> (bool, bool, bool) {
        let mut x_changed = false;
        let mut y_changed = false;
        let mut z_changed = false;
        for r in 0..8 {
            for i in 0..16 {
                if self.x[r][i] != other.x[r][i] {
                    x_changed = true;
                }
            }
        }
        for r in 0..8 {
            for i in 0..16 {
                if self.y[r][i] != other.y[r][i] {
                    y_changed = true;
                }
            }
        }
        for r in 0..64 {
            for i in 0..16 {
                if self.z[r][i] != other.z[r][i] {
                    z_changed = true;
                }
            }
        }
        (x_changed, y_changed, z_changed)
    }

    fn z_nonzero_rows(&self) -> Vec<usize> {
        let mut rows = Vec::new();
        for r in 0..64 {
            if self.z[r].iter().any(|&v| v != 0) {
                rows.push(r);
            }
        }
        rows
    }
}

fn main() {
    println!("=== AMX Full Reverse Engineering ===\n");

    let _ctx = Matrix::new().expect("AMX not available");

    unsafe {
        let buf = aligned_f32(16);
        let x_data = aligned_f32(16);
        let y_data = aligned_f32(16);

        // Load known patterns into X[0] and Y[0].
        for i in 0..16 {
            *x_data.add(i) = (i + 1) as f32;
            *y_data.add(i) = (16 + i + 1) as f32;
        }

        println!("=== Phase 1: Opcode map (0-22) with operand=0 ===\n");
        println!(
            "{:>3} | {:>4} {:>4} {:>4} | Z rows | notes",
            "op", "X?", "Y?", "Z?"
        );
        println!("{}", "-".repeat(70));

        for opcode in 0u32..=22 {
            // Reset state: load known data.
            let zero = aligned_f32(16);
            for r in 0u8..64 {
                amx_op::<OP_LDZ>((zero as u64) | ((r as u64) << 56));
            }
            for r in 0u8..8 {
                amx_op::<OP_LDX>((x_data as u64) | ((r as u64) << 56));
                amx_op::<OP_LDY>((y_data as u64) | ((r as u64) << 56));
            }
            free_f32(zero, 16);

            let before = Snapshot::capture(buf as *mut u8);

            // Execute opcode with operand=0.
            match opcode {
                0 => amx_op::<0>(x_data as u64), // ldx: ptr in operand
                1 => amx_op::<1>(y_data as u64), // ldy
                2 => amx_op::<2>(buf as u64),    // stx
                3 => amx_op::<3>(buf as u64),    // sty
                4 => amx_op::<4>(buf as u64),    // ldz
                5 => amx_op::<5>(buf as u64),    // stz
                6 => amx_op::<6>(buf as u64),    // ldzi
                7 => amx_op::<7>(buf as u64),    // stzi
                8 => amx_op::<8>(0u64),
                9 => amx_op::<9>(0u64),
                10 => amx_op::<10>(0u64),
                11 => amx_op::<11>(0u64),
                12 => amx_op::<12>(0u64),
                13 => amx_op::<13>(0u64),
                14 => amx_op::<14>(0u64),
                15 => amx_op::<15>(0u64),
                16 => amx_op::<16>(0u64),
                // 17 = set/clr, skip
                18 => amx_op::<18>(0u64),
                19 => amx_op::<19>(0u64),
                20 => amx_op::<20>(0u64),
                21 => amx_op::<21>(0u64),
                22 => amx_op::<22>(0u64),
                _ => continue,
            }

            let after = Snapshot::capture(buf as *mut u8);
            let (xc, yc, zc) = after.changed_regs(&before);
            let z_rows = after.z_nonzero_rows();

            let known = match opcode {
                0 => "ldx",
                1 => "ldy",
                2 => "stx",
                3 => "sty",
                4 => "ldz",
                5 => "stz",
                6 => "ldzi",
                7 => "stzi",
                8 => "extrx",
                9 => "extry",
                10 => "fma64",
                11 => "fms64",
                12 => "fma32",
                13 => "fms32",
                14 => "mac16",
                15 => "fma16",
                16 => "fms16",
                18..=22 => "???",
                _ => "skip",
            };

            println!(
                "{:>3} | {:>4} {:>4} {:>4} | {:?} | {}",
                opcode,
                if xc { "YES" } else { "-" },
                if yc { "YES" } else { "-" },
                if zc { "YES" } else { "-" },
                if z_rows.len() <= 8 {
                    format!("{:?}", z_rows)
                } else {
                    format!("{} rows", z_rows.len())
                },
                known
            );
        }

        println!("\n=== Phase 2: Operand bit probe for fma32 (opcode 12) ===\n");
        println!("Testing each bit 0-63 of the operand independently.\n");

        for bit in 0..64u32 {
            let operand: u64 = 1u64 << bit;

            // Zero Z, load X/Y.
            let zero = aligned_f32(16);
            for r in 0u8..64 {
                amx_op::<OP_LDZ>((zero as u64) | ((r as u64) << 56));
            }
            free_f32(zero, 16);
            for r in 0u8..8 {
                amx_op::<OP_LDX>((x_data as u64) | ((r as u64) << 56));
                amx_op::<OP_LDY>((y_data as u64) | ((r as u64) << 56));
            }

            amx_op::<12>(operand);

            let after = Snapshot::capture(buf as *mut u8);
            let z_rows = after.z_nonzero_rows();
            let z_stride = if z_rows.len() >= 2 {
                z_rows[1] - z_rows[0]
            } else {
                0
            };

            // Check first non-zero Z row value.
            let first_val = if let Some(&r) = z_rows.first() {
                format!("Z[{}][0]={:.1}", r, f32::from_bits(after.z[r][0]))
            } else {
                "empty".to_string()
            };

            if z_rows.is_empty() {
                println!("bit {:2}: Z empty (operand=0x{:016x})", bit, operand);
            } else {
                println!(
                    "bit {:2}: {} Z rows, stride={}, {} (operand=0x{:016x})",
                    bit,
                    z_rows.len(),
                    z_stride,
                    first_val,
                    operand
                );
            }
        }

        println!("\n=== Phase 3: Unknown opcodes 18-22 with operand bit scan ===\n");

        for opcode in 18u32..=22 {
            println!("--- Opcode {} ---", opcode);

            // Test with operand = skip_z (bit 27) to see if it behaves like FMA.
            let zero = aligned_f32(16);
            for r in 0u8..64 {
                amx_op::<OP_LDZ>((zero as u64) | ((r as u64) << 56));
            }
            free_f32(zero, 16);
            for r in 0u8..8 {
                amx_op::<OP_LDX>((x_data as u64) | ((r as u64) << 56));
                amx_op::<OP_LDY>((y_data as u64) | ((r as u64) << 56));
            }

            // Try with skip_z bit.
            let operand_skip = 1u64 << 27;
            match opcode {
                18 => amx_op::<18>(operand_skip),
                19 => amx_op::<19>(operand_skip),
                20 => amx_op::<20>(operand_skip),
                21 => amx_op::<21>(operand_skip),
                22 => amx_op::<22>(operand_skip),
                _ => {}
            }

            let after = Snapshot::capture(buf as *mut u8);
            let z_rows = after.z_nonzero_rows();
            let z_stride = if z_rows.len() >= 2 {
                z_rows[1] - z_rows[0]
            } else {
                0
            };
            println!("  skip_z: {} Z rows, stride={}", z_rows.len(), z_stride);

            if let Some(&r) = z_rows.first() {
                print!("  Z[{}] f32 = [", r);
                for i in 0..16 {
                    if i > 0 {
                        print!(", ");
                    }
                    print!("{:.1}", f32::from_bits(after.z[r][i]));
                }
                println!("]");
                print!("  Z[{}] hex = [", r);
                for i in 0..16 {
                    if i > 0 {
                        print!(" ");
                    }
                    print!("{:08x}", after.z[r][i]);
                }
                println!("]");
            }

            // Try vector mode (bit 63).
            for r in 0u8..64 {
                let zero2 = aligned_f32(16);
                amx_op::<OP_LDZ>((zero2 as u64) | ((r as u64) << 56));
                free_f32(zero2, 16);
            }

            let operand_vec = (1u64 << 63) | (1u64 << 27);
            match opcode {
                18 => amx_op::<18>(operand_vec),
                19 => amx_op::<19>(operand_vec),
                20 => amx_op::<20>(operand_vec),
                21 => amx_op::<21>(operand_vec),
                22 => amx_op::<22>(operand_vec),
                _ => {}
            }

            let after_vec = Snapshot::capture(buf as *mut u8);
            let z_rows_vec = after_vec.z_nonzero_rows();
            println!("  vector: {} Z rows", z_rows_vec.len());

            if let Some(&r) = z_rows_vec.first() {
                print!("  Z[{}] f32 = [", r);
                for i in 0..16 {
                    if i > 0 {
                        print!(", ");
                    }
                    print!("{:.1}", f32::from_bits(after_vec.z[r][i]));
                }
                println!("]");
            }
            println!();
        }

        free_f32(buf, 16);
        free_f32(x_data, 16);
        free_f32(y_data, 16);
    }

    println!("=== Done ===");
}
