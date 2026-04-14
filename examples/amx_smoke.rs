//! AMX fma32 verification — confirm outer product correctness on real hardware.
//!
//! Run with: `cargo run --example amx_smoke --release`

use acpu::matrix::asm::{amx_op, OP_FMA32, OP_LDX, OP_LDY, OP_LDZ, OP_STZ};
use acpu::matrix::fma::{fma_acc, fma_first};
use acpu::matrix::regs::{XRow, YRow};
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
    println!("=== AMX fma32 verification ===\n");

    let _ctx = Matrix::new().expect("AMX not available");
    let mut all_pass = true;

    unsafe {
        let x_buf = aligned_f32(16);
        let y_buf = aligned_f32(16);
        let zero_buf = aligned_f32(16);
        let z_buf = aligned_f32(16 * 16);

        // --- Test 1: basic outer product ---
        // X[0] = [1..16], Y[0] = [1;16], skip_z
        // Expected: Z[j*4][i] = (i+1) for all j

        for i in 0..16 {
            *x_buf.add(i) = (i + 1) as f32;
        }
        for j in 0..16 {
            *y_buf.add(j) = 1.0;
        }
        for row in 0u8..64 {
            amx_op::<OP_LDZ>((zero_buf as u64) | ((row as u64) << 56));
        }
        amx_op::<OP_LDX>((x_buf as u64) | (0u64 << 56));
        amx_op::<OP_LDY>((y_buf as u64) | (0u64 << 56));

        let op = fma_first(XRow::new_unchecked(0), YRow::new_unchecked(0), 0);
        amx_op::<OP_FMA32>(op);

        // Store tile 0: Z rows 0,4,8,...,60
        for j in 0u8..16 {
            let z_row = j * 4;
            let dst = z_buf.add(j as usize * 16) as *mut u8;
            amx_op::<OP_STZ>((dst as u64) | ((z_row as u64) << 56));
        }

        let mut pass = true;
        for j in 0..16 {
            for i in 0..16 {
                let val = *z_buf.add(j * 16 + i);
                let expected = (i + 1) as f32;
                if (val - expected).abs() > 1e-5 {
                    pass = false;
                    println!(
                        "  MISMATCH Z[{}][{}]: got {} expected {}",
                        j * 4,
                        i,
                        val,
                        expected
                    );
                }
            }
        }
        println!(
            "Test 1 (outer product X=[1..16] * Y=[1;16]):  {}",
            if pass { "PASS" } else { "FAIL" }
        );
        all_pass &= pass;

        // --- Test 2: accumulate ---
        // Z already has X*Y from test 1. Do Z += X*Y again.
        // Expected: Z[j*4][i] = 2*(i+1)

        let op_acc = fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0);
        amx_op::<OP_FMA32>(op_acc);

        for j in 0u8..16 {
            let z_row = j * 4;
            let dst = z_buf.add(j as usize * 16) as *mut u8;
            amx_op::<OP_STZ>((dst as u64) | ((z_row as u64) << 56));
        }

        let mut pass2 = true;
        for j in 0..16 {
            for i in 0..16 {
                let val = *z_buf.add(j * 16 + i);
                let expected = 2.0 * (i + 1) as f32;
                if (val - expected).abs() > 1e-4 {
                    pass2 = false;
                }
            }
        }
        println!(
            "Test 2 (accumulate: Z += X*Y):                {}",
            if pass2 { "PASS" } else { "FAIL" }
        );
        all_pass &= pass2;

        // --- Test 3: different registers ---
        // X[3] = [10;16], Y[5] = [0.5;16]
        // Expected: Z[j*4][i] = 5.0

        for row in 0u8..64 {
            amx_op::<OP_LDZ>((zero_buf as u64) | ((row as u64) << 56));
        }
        for i in 0..16 {
            *x_buf.add(i) = 10.0;
        }
        amx_op::<OP_LDX>((x_buf as u64) | (3u64 << 56));
        for j in 0..16 {
            *y_buf.add(j) = 0.5;
        }
        amx_op::<OP_LDY>((y_buf as u64) | (5u64 << 56));

        let op3 = fma_first(XRow::new_unchecked(3), YRow::new_unchecked(5), 0);
        amx_op::<OP_FMA32>(op3);

        for j in 0u8..16 {
            let z_row = j * 4;
            let dst = z_buf.add(j as usize * 16) as *mut u8;
            amx_op::<OP_STZ>((dst as u64) | ((z_row as u64) << 56));
        }

        let mut pass3 = true;
        for j in 0..16 {
            for i in 0..16 {
                let val = *z_buf.add(j * 16 + i);
                if (val - 5.0).abs() > 1e-5 {
                    pass3 = false;
                }
            }
        }
        println!(
            "Test 3 (X[3]=[10;16] * Y[5]=[0.5;16]):       {}",
            if pass3 { "PASS" } else { "FAIL" }
        );
        all_pass &= pass3;

        // --- Test 4: tile select ---
        // Write to tile 1 (z_row offset=1), verify tile 0 is untouched.
        for row in 0u8..64 {
            amx_op::<OP_LDZ>((zero_buf as u64) | ((row as u64) << 56));
        }
        for i in 0..16 {
            *x_buf.add(i) = 7.0;
        }
        amx_op::<OP_LDX>((x_buf as u64) | (0u64 << 56));
        for j in 0..16 {
            *y_buf.add(j) = 3.0;
        }
        amx_op::<OP_LDY>((y_buf as u64) | (0u64 << 56));

        let op4 = fma_first(XRow::new_unchecked(0), YRow::new_unchecked(0), 1);
        amx_op::<OP_FMA32>(op4);

        // Read tile 0 (rows 0,4,...,60) — should be zero.
        let mut tile0_clean = true;
        for j in 0u8..16 {
            let z_row = j * 4;
            let dst = z_buf.add(j as usize * 16) as *mut u8;
            amx_op::<OP_STZ>((dst as u64) | ((z_row as u64) << 56));
        }
        for j in 0..16 {
            for i in 0..16 {
                if (*z_buf.add(j * 16 + i)).to_bits() != 0 {
                    tile0_clean = false;
                }
            }
        }

        // Read tile 1 (rows 1,5,...,61) — should be 21.0.
        let mut tile1_correct = true;
        for j in 0u8..16 {
            let z_row = j * 4 + 1;
            let dst = z_buf.add(j as usize * 16) as *mut u8;
            amx_op::<OP_STZ>((dst as u64) | ((z_row as u64) << 56));
        }
        for j in 0..16 {
            for i in 0..16 {
                let val = *z_buf.add(j * 16 + i);
                if (val - 21.0).abs() > 1e-4 {
                    tile1_correct = false;
                }
            }
        }

        let pass4 = tile0_clean && tile1_correct;
        println!(
            "Test 4 (tile isolation: tile0=0, tile1=21):    {}",
            if pass4 { "PASS" } else { "FAIL" }
        );
        if !tile0_clean {
            println!("  tile0 NOT clean");
        }
        if !tile1_correct {
            println!("  tile1 NOT correct");
        }
        all_pass &= pass4;

        // --- Test 5: rank-K microkernel ---
        // Multiply A[16×4] × B[4×16] using 4 rank-1 updates.
        // A column-major, B row-major.
        for row in 0u8..64 {
            amx_op::<OP_LDZ>((zero_buf as u64) | ((row as u64) << 56));
        }

        // A: 4 columns of 16 rows. A[i][p] = (i+1) * (p+1).
        // B: 4 rows of 16 cols. B[p][j] = 1.0 (identity-like).
        let a_panel = aligned_f32(4 * 16);
        let b_panel = aligned_f32(4 * 16);

        for p in 0..4 {
            for i in 0..16 {
                *a_panel.add(p * 16 + i) = ((i + 1) * (p + 1)) as f32;
            }
            for j in 0..16 {
                *b_panel.add(p * 16 + j) = if j == p { 1.0 } else { 0.0 };
            }
        }

        // 4 rank-1 updates.
        for p in 0..4u8 {
            amx_op::<OP_LDX>((b_panel.add(p as usize * 16) as u64) | ((p as u64) << 56));
            amx_op::<OP_LDY>((a_panel.add(p as usize * 16) as u64) | ((p as u64) << 56));
        }
        // First update: skip_z.
        amx_op::<OP_FMA32>(fma_first(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
        // Remaining: accumulate.
        for p in 1..4u8 {
            amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(p), YRow::new_unchecked(p), 0));
        }

        for j in 0u8..16 {
            let z_row = j * 4;
            let dst = z_buf.add(j as usize * 16) as *mut u8;
            amx_op::<OP_STZ>((dst as u64) | ((z_row as u64) << 56));
        }

        // Z[j][i] = sum_p X_p[i] * Y_p[j] = sum_p B[p][i] * A[j][p]
        // B is identity-like (B[p][i]=1 iff i==p, for p<4), so:
        // Z[j][i] = A[j][i] = (j+1)*(i+1) for i < 4, else 0.
        let mut pass5 = true;
        for j in 0..16 {
            for i in 0..16 {
                let val = *z_buf.add(j * 16 + i);
                let expected = if i < 4 {
                    ((i + 1) * (j + 1)) as f32
                } else {
                    0.0
                };
                if (val - expected).abs() > 1e-3 {
                    pass5 = false;
                    if i < 2 && j < 2 {
                        println!(
                            "  MISMATCH C[{}][{}]: got {} expected {}",
                            i, j, val, expected
                        );
                    }
                }
            }
        }
        println!(
            "Test 5 (rank-4 microkernel A[16x4]*B[4x16]):  {}",
            if pass5 { "PASS" } else { "FAIL" }
        );
        all_pass &= pass5;

        free_f32(x_buf, 16);
        free_f32(y_buf, 16);
        free_f32(zero_buf, 16);
        free_f32(z_buf, 16 * 16);
        free_f32(a_panel, 4 * 16);
        free_f32(b_panel, 4 * 16);
    }

    println!(
        "\n=== {} ===",
        if all_pass {
            "ALL TESTS PASSED"
        } else {
            "SOME TESTS FAILED"
        }
    );
    if !all_pass {
        std::process::exit(1);
    }
}
