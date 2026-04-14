//! Step-by-step trace of sgemm 64×64 — find the hidden overhead.
//! Reproduces the exact GEBP path with timing at each step.
//!
//! Run: cargo run --example trace64 --release

use std::time::Instant;

const N: usize = 64;
const MR: usize = 16;
const NR: usize = 16;
const ITERS: usize = 10_000;

fn median_ns<F: FnMut()>(mut f: F) -> u64 {
    for _ in 0..200 {
        f();
    }
    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        f();
        times.push(t.elapsed().as_nanos() as u64);
    }
    times.sort();
    times[ITERS / 2]
}

/// NEON 4×4 transpose pack (same as pack_a_strip_neon in gemm/mod.rs)
#[cfg(target_arch = "aarch64")]
unsafe fn pack_a_neon(a: &[f32], dst: &mut [f32], n_mr: usize, k: usize) {
    use core::arch::aarch64::*;
    for s in 0..n_mr {
        let row_start = s * MR;
        let base = s * k * MR;
        for ig in 0..(MR / 4) {
            let ii = ig * 4;
            let a0 = (row_start + ii) * k;
            let a1 = (row_start + ii + 1) * k;
            let a2 = (row_start + ii + 2) * k;
            let a3 = (row_start + ii + 3) * k;
            let mut p = 0;
            while p + 4 <= k {
                let r0 = vld1q_f32(a.as_ptr().add(a0 + p));
                let r1 = vld1q_f32(a.as_ptr().add(a1 + p));
                let r2 = vld1q_f32(a.as_ptr().add(a2 + p));
                let r3 = vld1q_f32(a.as_ptr().add(a3 + p));
                let lo01 = vzip1q_f32(r0, r1);
                let hi01 = vzip2q_f32(r0, r1);
                let lo23 = vzip1q_f32(r2, r3);
                let hi23 = vzip2q_f32(r2, r3);
                let c0 = vreinterpretq_f32_f64(vzip1q_f64(
                    vreinterpretq_f64_f32(lo01),
                    vreinterpretq_f64_f32(lo23),
                ));
                let c1 = vreinterpretq_f32_f64(vzip2q_f64(
                    vreinterpretq_f64_f32(lo01),
                    vreinterpretq_f64_f32(lo23),
                ));
                let c2 = vreinterpretq_f32_f64(vzip1q_f64(
                    vreinterpretq_f64_f32(hi01),
                    vreinterpretq_f64_f32(hi23),
                ));
                let c3 = vreinterpretq_f32_f64(vzip2q_f64(
                    vreinterpretq_f64_f32(hi01),
                    vreinterpretq_f64_f32(hi23),
                ));
                let d = dst.as_mut_ptr().add(base);
                vst1q_f32(d.add(p * MR + ii), c0);
                vst1q_f32(d.add((p + 1) * MR + ii), c1);
                vst1q_f32(d.add((p + 2) * MR + ii), c2);
                vst1q_f32(d.add((p + 3) * MR + ii), c3);
                p += 4;
            }
        }
    }
}

/// Copy B into NR-wide strips (same as pack_b_nr in gemm/mod.rs)
fn pack_b_copy(b: &[f32], dst: &mut [f32], n_nr: usize, k: usize) {
    for s in 0..n_nr {
        let col_start = s * NR;
        let base = s * k * NR;
        for p in 0..k {
            let src = p * N + col_start;
            let d = base + p * NR;
            dst[d..d + NR].copy_from_slice(&b[src..src + NR]);
        }
    }
}

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

fn main() {
    let a: Vec<f32> = (0..N * N).map(|i| (i % 7) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..N * N).map(|i| (i % 11) as f32 * 0.1).collect();

    let n_mr = N / MR; // 4
    let n_nr = N / NR; // 4

    println!("=== trace64: sgemm 64×64 step-by-step ===\n");

    // 1. alloc_zeroed 32KB (a_pack 16KB + b_pack 16KB)
    let t_alloc = median_ns(|| {
        let la = std::alloc::Layout::from_size_align(N * N * 4, 64).unwrap();
        let lb = std::alloc::Layout::from_size_align(N * N * 4, 64).unwrap();
        let pa = unsafe { std::alloc::alloc_zeroed(la) };
        let pb = unsafe { std::alloc::alloc_zeroed(lb) };
        std::hint::black_box(pa);
        std::hint::black_box(pb);
        unsafe {
            std::alloc::dealloc(pa, la);
            std::alloc::dealloc(pb, lb);
        }
    });

    // 2. A pack NEON (into pre-allocated stack buffer)
    #[repr(align(64))]
    struct Buf64([f32; N * N]);

    let mut a_pack = Buf64([0.0; N * N]);
    let t_apack = median_ns(|| {
        unsafe { pack_a_neon(&a, &mut a_pack.0, n_mr, N) };
        std::hint::black_box(&a_pack);
    });

    // 3. B pack (copy_from_slice into pre-allocated stack buffer)
    let mut b_pack = Buf64([0.0; N * N]);
    let t_bpack = median_ns(|| {
        pack_b_copy(&b, &mut b_pack.0, n_nr, N);
        std::hint::black_box(&b_pack);
    });

    // 4. AMX set + clr
    let t_setclr = median_ns(|| {
        let ctx = acpu::Matrix::new().unwrap();
        std::hint::black_box(&ctx);
        drop(ctx);
    });

    // 5. Compute only: preload + µk64_acc + store (pre-packed data)
    let mut c_buf = vec![0.0f32; N * N];
    // Make sure packs are populated
    unsafe { pack_a_neon(&a, &mut a_pack.0, n_mr, N) };
    pack_b_copy(&b, &mut b_pack.0, n_nr, N);

    let t_compute = median_ns(|| unsafe {
        use acpu::matrix::asm::*;
        amx_set();

        for ir in 0..n_mr {
            let ap = a_pack.0.as_ptr().add(ir * N * MR) as *const u8;
            let cp = c_buf.as_mut_ptr().add(ir * MR * N);

            // Preload 4 C tiles
            for t in 0u8..4 {
                acpu::matrix::tile::preload_c(cp.add(t as usize * NR), N, t);
            }

            // µkernel 16×64
            acpu::matrix::tile::microkernel_16x64_acc(
                ap,
                b_pack.0.as_ptr().add(0 * N * NR) as *const u8,
                b_pack.0.as_ptr().add(1 * N * NR) as *const u8,
                b_pack.0.as_ptr().add(2 * N * NR) as *const u8,
                b_pack.0.as_ptr().add(3 * N * NR) as *const u8,
                N,
                64,
            );

            // Store 4 C tiles
            for t in 0u8..4 {
                acpu::matrix::tile::store_c(cp.add(t as usize * NR), N, t);
            }
        }

        amx_clr();
        std::hint::black_box(&c_buf);
    });

    // 6. Combined: alloc + pack_a + pack_b + set + compute + clr + dealloc
    let t_combined = median_ns(|| unsafe {
        // Alloc
        let la = std::alloc::Layout::from_size_align(N * N * 4, 64).unwrap();
        let lb = std::alloc::Layout::from_size_align(N * N * 4, 64).unwrap();
        let pa = std::alloc::alloc_zeroed(la) as *mut f32;
        let pb = std::alloc::alloc_zeroed(lb) as *mut f32;
        let ap = std::slice::from_raw_parts_mut(pa, N * N);
        let bp = std::slice::from_raw_parts_mut(pb, N * N);

        // Pack
        pack_a_neon(&a, ap, n_mr, N);
        pack_b_copy(&b, bp, n_nr, N);

        // Compute
        use acpu::matrix::asm::*;
        amx_set();

        for ir in 0..n_mr {
            let a_ptr = ap.as_ptr().add(ir * N * MR) as *const u8;
            let cp = c_buf.as_mut_ptr().add(ir * MR * N);
            for t in 0u8..4 {
                acpu::matrix::tile::preload_c(cp.add(t as usize * NR), N, t);
            }
            acpu::matrix::tile::microkernel_16x64_acc(
                a_ptr,
                bp.as_ptr().add(0 * N * NR) as *const u8,
                bp.as_ptr().add(1 * N * NR) as *const u8,
                bp.as_ptr().add(2 * N * NR) as *const u8,
                bp.as_ptr().add(3 * N * NR) as *const u8,
                N,
                64,
            );
            for t in 0u8..4 {
                acpu::matrix::tile::store_c(cp.add(t as usize * NR), N, t);
            }
        }

        amx_clr();

        std::alloc::dealloc(pa as *mut u8, la);
        std::alloc::dealloc(pb as *mut u8, lb);
        std::hint::black_box(&c_buf);
    });

    // 7. Actual sgemm
    let t_sgemm = median_ns(|| {
        c_buf.fill(0.0);
        acpu::matmul_f32(&a, &b, &mut c_buf, N, N, N);
        std::hint::black_box(&c_buf);
    });

    // 8. Accelerate
    let t_accel = median_ns(|| {
        c_buf.fill(0.0);
        unsafe {
            cblas_sgemm(
                101,
                111,
                111,
                N as i32,
                N as i32,
                N as i32,
                1.0,
                a.as_ptr(),
                N as i32,
                b.as_ptr(),
                N as i32,
                0.0,
                c_buf.as_mut_ptr(),
                N as i32,
            );
        }
        std::hint::black_box(&c_buf);
    });

    // 9. Compute without B packing — load B directly with stride
    let t_nopack_b = median_ns(|| unsafe {
        use acpu::matrix::asm::*;
        use acpu::matrix::fma::*;
        use acpu::matrix::regs::*;
        amx_set();

        for ir in 0..n_mr {
            let ap = a_pack.0.as_ptr().add(ir * N * MR) as *const u8;
            let cp = c_buf.as_mut_ptr().add(ir * MR * N);

            for t in 0u8..4 {
                acpu::matrix::tile::preload_c(cp.add(t as usize * NR), N, t);
            }

            // Direct B loads with stride=N*4, no packing
            let b_stride = N * 4; // bytes
            for p in 0..N {
                // LDY: packed A column
                amx_op::<OP_LDY>(ap.add(p * 64) as u64);
                // LDX: 4 B rows directly from b[] with stride
                for t in 0u8..4 {
                    let b_addr = b.as_ptr().add(p * N + t as usize * NR) as *const u8;
                    amx_op::<OP_LDX>((b_addr as u64) | ((t as u64) << 56));
                }
                // FMA all 4 tiles
                for t in 0u8..4 {
                    amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(t), YRow::new_unchecked(0), t));
                }
            }

            for t in 0u8..4 {
                acpu::matrix::tile::store_c(cp.add(t as usize * NR), N, t);
            }
        }

        amx_clr();
        std::hint::black_box(&c_buf);
    });

    // 10. Absolute minimum: just A pack + compute with direct B, stack buffers
    let t_minimal = median_ns(|| unsafe {
        use acpu::matrix::asm::*;
        use acpu::matrix::fma::*;
        use acpu::matrix::regs::*;

        // Pack A on stack (pre-allocated, no zero-fill needed)
        pack_a_neon(&a, &mut a_pack.0, n_mr, N);

        amx_set();

        for ir in 0..n_mr {
            let ap = a_pack.0.as_ptr().add(ir * N * MR) as *const u8;
            let cp = c_buf.as_mut_ptr().add(ir * MR * N);

            for t in 0u8..4 {
                acpu::matrix::tile::preload_c(cp.add(t as usize * NR), N, t);
            }

            for p in 0..N {
                amx_op::<OP_LDY>(ap.add(p * 64) as u64);
                for t in 0u8..4 {
                    let b_addr = b.as_ptr().add(p * N + t as usize * NR) as *const u8;
                    amx_op::<OP_LDX>((b_addr as u64) | ((t as u64) << 56));
                }
                for t in 0u8..4 {
                    amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(t), YRow::new_unchecked(0), t));
                }
            }

            for t in 0u8..4 {
                acpu::matrix::tile::store_c(cp.add(t as usize * NR), N, t);
            }
        }

        amx_clr();
        std::hint::black_box(&c_buf);
    });

    println!("{:>35}  {:>6} ns", "alloc_zeroed 32KB + dealloc", t_alloc);
    println!("{:>35}  {:>6} ns", "A pack NEON (stack buf)", t_apack);
    println!("{:>35}  {:>6} ns", "B pack copy (stack buf)", t_bpack);
    println!("{:>35}  {:>6} ns", "AMX set+clr", t_setclr);
    println!("{:>35}  {:>6} ns", "compute only (pre-packed)", t_compute);
    println!(
        "{:>35}  {:>6} ns",
        "compute no B pack (direct LDX)", t_nopack_b
    );
    println!();

    let sum = t_alloc + t_apack + t_bpack + t_setclr + t_compute;
    println!("{:>35}  {:>6} ns", "sum of parts", sum);
    println!(
        "{:>35}  {:>6} ns",
        "combined (alloc+pack+compute)", t_combined
    );
    println!(
        "{:>35}  {:>6} ns",
        "minimal (A pack+direct B+compute)", t_minimal
    );
    println!("{:>35}  {:>6} ns", "sgemm (full path)", t_sgemm);
    println!("{:>35}  {:>6} ns", "Accelerate", t_accel);

    println!(
        "\n{:>35}  {:>5.1}x",
        "sgemm / Accelerate",
        t_sgemm as f64 / t_accel as f64
    );
    println!(
        "{:>35}  {:>5.1}x",
        "minimal / Accelerate",
        t_minimal as f64 / t_accel as f64
    );
    println!(
        "{:>35}  {:>5.1}x",
        "sgemm / minimal",
        t_sgemm as f64 / t_minimal as f64
    );
    println!(
        "\n{:>35}  {:>6} ns",
        "hidden overhead (sgemm - sum)",
        t_sgemm as i64 - sum as i64
    );
    println!(
        "{:>35}  {:>6} ns",
        "hidden overhead (sgemm - combined)",
        t_sgemm as i64 - t_combined as i64
    );

    // 11. Combined but with STACK buffers — test cache conflict hypothesis
    let t_stack_combined = median_ns(|| unsafe {
        use acpu::matrix::asm::*;

        #[repr(align(64))]
        struct S([f32; N * N]);

        let mut sa = S([0.0; N * N]);
        let mut sb = S([0.0; N * N]);

        pack_a_neon(&a, &mut sa.0, n_mr, N);
        pack_b_copy(&b, &mut sb.0, n_nr, N);

        amx_set();

        for ir in 0..n_mr {
            let ap = sa.0.as_ptr().add(ir * N * MR) as *const u8;
            let cp = c_buf.as_mut_ptr().add(ir * MR * N);
            for t in 0u8..4 {
                acpu::matrix::tile::preload_c(cp.add(t as usize * NR), N, t);
            }
            acpu::matrix::tile::microkernel_16x64_acc(
                ap,
                sb.0.as_ptr().add(0 * N * NR) as *const u8,
                sb.0.as_ptr().add(1 * N * NR) as *const u8,
                sb.0.as_ptr().add(2 * N * NR) as *const u8,
                sb.0.as_ptr().add(3 * N * NR) as *const u8,
                N,
                64,
            );
            for t in 0u8..4 {
                acpu::matrix::tile::store_c(cp.add(t as usize * NR), N, t);
            }
        }

        amx_clr();
        std::hint::black_box(&c_buf);
    });

    println!(
        "\n{:>35}  {:>6} ns  (cache conflict test)",
        "combined STACK bufs", t_stack_combined
    );
    println!("{:>35}  {:>6} ns", "combined HEAP bufs", t_combined);
    println!(
        "{:>35}  {:>5.1}x",
        "heap / stack",
        t_combined as f64 / t_stack_combined as f64
    );

    // 12. Minimal but with fresh stack zero-init each iteration
    let t_minimal_zeroinit = median_ns(|| unsafe {
        use acpu::matrix::asm::*;
        use acpu::matrix::fma::*;
        use acpu::matrix::regs::*;

        #[repr(align(64))]
        struct S([f32; N * N]);
        let mut sa = S([0.0; N * N]); // zero-init inside loop

        pack_a_neon(&a, &mut sa.0, n_mr, N);
        amx_set();
        for ir in 0..n_mr {
            let ap = sa.0.as_ptr().add(ir * N * MR) as *const u8;
            let cp = c_buf.as_mut_ptr().add(ir * MR * N);
            for t in 0u8..4 {
                acpu::matrix::tile::preload_c(cp.add(t as usize * NR), N, t);
            }
            for p in 0..N {
                amx_op::<OP_LDY>(ap.add(p * 64) as u64);
                for t in 0u8..4 {
                    let b_addr = b.as_ptr().add(p * N + t as usize * NR) as *const u8;
                    amx_op::<OP_LDX>((b_addr as u64) | ((t as u64) << 56));
                }
                for t in 0u8..4 {
                    amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(t), YRow::new_unchecked(0), t));
                }
            }
            for t in 0u8..4 {
                acpu::matrix::tile::store_c(cp.add(t as usize * NR), N, t);
            }
        }
        amx_clr();
        std::hint::black_box(&c_buf);
    });

    // 13. Stack combined but with MaybeUninit (no zero-init)
    let t_stack_nozero = median_ns(|| unsafe {
        use acpu::matrix::asm::*;
        use core::mem::MaybeUninit;

        #[repr(align(64))]
        struct S([MaybeUninit<f32>; N * N]);
        let mut sa: S = MaybeUninit::uninit().assume_init();
        let mut sb: S = MaybeUninit::uninit().assume_init();
        let sa_slice = &mut *(sa.0.as_mut_ptr() as *mut [f32; N * N]);
        let sb_slice = &mut *(sb.0.as_mut_ptr() as *mut [f32; N * N]);

        pack_a_neon(&a, sa_slice, n_mr, N);
        pack_b_copy(&b, sb_slice, n_nr, N);

        amx_set();
        for ir in 0..n_mr {
            let ap = sa_slice.as_ptr().add(ir * N * MR) as *const u8;
            let cp = c_buf.as_mut_ptr().add(ir * MR * N);
            for t in 0u8..4 {
                acpu::matrix::tile::preload_c(cp.add(t as usize * NR), N, t);
            }
            acpu::matrix::tile::microkernel_16x64_acc(
                ap,
                sb_slice.as_ptr().add(0 * N * NR) as *const u8,
                sb_slice.as_ptr().add(1 * N * NR) as *const u8,
                sb_slice.as_ptr().add(2 * N * NR) as *const u8,
                sb_slice.as_ptr().add(3 * N * NR) as *const u8,
                N,
                64,
            );
            for t in 0u8..4 {
                acpu::matrix::tile::store_c(cp.add(t as usize * NR), N, t);
            }
        }
        amx_clr();
        std::hint::black_box(&c_buf);
    });

    // 14. Minimal with fused µkernel (packed B, but reuse external buffer)
    let t_minimal_fused = median_ns(|| unsafe {
        use acpu::matrix::asm::*;

        pack_a_neon(&a, &mut a_pack.0, n_mr, N);
        pack_b_copy(&b, &mut b_pack.0, n_nr, N);

        amx_set();
        for ir in 0..n_mr {
            let ap = a_pack.0.as_ptr().add(ir * N * MR) as *const u8;
            let cp = c_buf.as_mut_ptr().add(ir * MR * N);
            for t in 0u8..4 {
                acpu::matrix::tile::preload_c(cp.add(t as usize * NR), N, t);
            }
            acpu::matrix::tile::microkernel_16x64_acc(
                ap,
                b_pack.0.as_ptr().add(0 * N * NR) as *const u8,
                b_pack.0.as_ptr().add(1 * N * NR) as *const u8,
                b_pack.0.as_ptr().add(2 * N * NR) as *const u8,
                b_pack.0.as_ptr().add(3 * N * NR) as *const u8,
                N,
                64,
            );
            for t in 0u8..4 {
                acpu::matrix::tile::store_c(cp.add(t as usize * NR), N, t);
            }
        }
        amx_clr();
        std::hint::black_box(&c_buf);
    });

    println!("\n--- triangulation ---");
    println!("{:>35}  {:>6} ns", "minimal (ext buf, direct B)", t_minimal);
    println!(
        "{:>35}  {:>6} ns",
        "minimal + zero-init A each iter", t_minimal_zeroinit
    );
    println!(
        "{:>35}  {:>6} ns",
        "minimal + fused µk + B pack (ext)", t_minimal_fused
    );
    println!(
        "{:>35}  {:>6} ns",
        "stack combined (zero-init + pack)", t_stack_combined
    );
    println!(
        "{:>35}  {:>6} ns",
        "stack no-zero (MaybeUninit + pack)", t_stack_nozero
    );
    println!("{:>35}  {:>6} ns", "sgemm", t_sgemm);
    println!("{:>35}  {:>6} ns", "Accelerate", t_accel);
}
