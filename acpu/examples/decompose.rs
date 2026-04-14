//! Component-level overhead decomposition for sgemm.
//! Isolates: packing, AMX compute, accumulation, Accelerate reference.
//!
//! Run: cargo run --example decompose --release

use std::time::Instant;

const ITERS: usize = 10_000;
const MR: usize = 16;

fn median_ns<F: FnMut()>(mut f: F) -> u64 {
    // Warmup.
    for _ in 0..100 {
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
    for &n in &[64usize, 128, 256] {
        decompose(n);
        println!();
    }
}

fn decompose(n: usize) {
    let a: Vec<f32> = (0..n * n).map(|i| (i % 7) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..n * n).map(|i| (i % 11) as f32 * 0.1).collect();

    let n_mr = n.div_ceil(MR);
    let n_tiles = n_mr * n_mr; // total 16×16 tiles for n×n

    println!("=== n={n} ({n_tiles} tiles, k={n}) ===");
    println!("{:>32}  {:>7}  {:>5}", "component", "ns", "%sgemm");

    // --- Full sgemm (reference) ---
    let mut c = vec![0.0f32; n * n];
    let sgemm_ns = median_ns(|| {
        c.fill(0.0);
        acpu::matmul_f32(&a, &b, &mut c, n, n, n);
        std::hint::black_box(&c);
    });

    // --- Accelerate ---
    let accel_ns = median_ns(|| {
        c.fill(0.0);
        unsafe {
            cblas_sgemm(
                101,
                111,
                111,
                n as i32,
                n as i32,
                n as i32,
                1.0,
                a.as_ptr(),
                n as i32,
                b.as_ptr(),
                n as i32,
                0.0,
                c.as_mut_ptr(),
                n as i32,
            );
        }
        std::hint::black_box(&c);
    });

    // --- AMX set+clr ---
    let setclr_ns = median_ns(|| {
        let ctx = acpu::Matrix::new().unwrap();
        std::hint::black_box(&ctx);
        drop(ctx);
    });

    // --- A pack: scalar (small.rs style) ---
    let pack_scalar_ns = {
        use core::mem::MaybeUninit;

        #[repr(align(64))]
        struct APack([MaybeUninit<f32>; 256 * 256]);

        let t = median_ns(|| {
            let mut buf: APack = unsafe { MaybeUninit::uninit().assume_init() };
            let ap = &mut buf.0[..];
            for s in 0..n_mr {
                let rs = s * MR;
                let rows = MR.min(n - rs);
                let base = s * n * MR;
                for i in 0..rows {
                    let a_row = (rs + i) * n;
                    for p in 0..n {
                        ap[base + p * MR + i] = MaybeUninit::new(a[a_row + p]);
                    }
                }
                for i in rows..MR {
                    for p in 0..n {
                        ap[base + p * MR + i] = MaybeUninit::new(0.0);
                    }
                }
            }
            std::hint::black_box(&buf);
        });
        t
    };

    // --- A pack: NEON (pack_a_strip_neon style) ---
    let pack_neon_ns = {
        #[repr(align(64))]
        struct APack([f32; 256 * 256]);

        let t = median_ns(|| {
            let mut buf = APack([0.0f32; 256 * 256]);
            let dst = &mut buf.0[..];
            for s in 0..n_mr {
                let row_start = s * MR;
                let base = s * n * MR;
                unsafe {
                    use core::arch::aarch64::*;
                    for ig in 0..(MR / 4) {
                        let ii = ig * 4;
                        let a0 = (row_start + ii) * n;
                        let a1 = (row_start + ii + 1) * n;
                        let a2 = (row_start + ii + 2) * n;
                        let a3 = (row_start + ii + 3) * n;

                        let mut p = 0;
                        while p + 4 <= n {
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
            std::hint::black_box(&buf);
        });
        t
    };

    // --- AMX compute: per-instruction LDX+LDY+FMA (small.rs inner loop) ---
    let compute_perop_ns = {
        #[repr(align(64))]
        struct Buf([f32; 256 * 16]);

        let a_pack = Buf([1.0f32; 256 * 16]);
        let b_pack = Buf([1.0f32; 256 * 16]);

        median_ns(|| unsafe {
            use acpu::matrix::asm::*;
            use acpu::matrix::fma::*;
            use acpu::matrix::regs::*;
            amx_set();

            for _tile in 0..n_tiles {
                let ap = a_pack.0.as_ptr() as *const u8;
                let bp = b_pack.0.as_ptr() as *const u8;

                amx_op::<OP_LDX>(bp as u64);
                amx_op::<OP_LDY>(ap as u64);
                amx_op::<OP_FMA32>(fma_first(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));

                for p in 1..n {
                    amx_op::<OP_LDX>(bp.add(p * 64) as u64);
                    amx_op::<OP_LDY>(ap.add(p * 64) as u64);
                    amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
                }
            }

            amx_clr();
        })
    };

    // --- AMX compute: microkernel_16x16 (batched loads per 8) ---
    let compute_uk16_ns = {
        #[repr(align(64))]
        struct Buf([f32; 256 * 16]);

        let a_pack = Buf([1.0f32; 256 * 16]);
        let b_pack = Buf([1.0f32; 256 * 16]);

        median_ns(|| unsafe {
            use acpu::matrix::asm::*;
            amx_set();
            for _tile in 0..n_tiles {
                acpu::matrix::tile::microkernel_16x16(
                    a_pack.0.as_ptr() as *const u8,
                    b_pack.0.as_ptr() as *const u8,
                    n,
                );
            }
            amx_clr();
        })
    };

    // --- AMX compute: microkernel_16x64 (fused 4-tile, Y reuse) ---
    let compute_uk64_ns = {
        #[repr(align(64))]
        struct Buf([f32; 256 * 16]);

        let ap = Buf([1.0f32; 256 * 16]);
        let b0 = Buf([1.0f32; 256 * 16]);
        let b1 = Buf([1.0f32; 256 * 16]);
        let b2 = Buf([1.0f32; 256 * 16]);
        let b3 = Buf([1.0f32; 256 * 16]);

        // n_tiles/4 batches of 4 tiles = same total tiles
        let batches = n_tiles / 4;
        let leftover = n_tiles % 4;

        median_ns(|| unsafe {
            use acpu::matrix::asm::*;
            amx_set();
            for _ in 0..batches {
                acpu::matrix::tile::microkernel_16x64(
                    ap.0.as_ptr() as *const u8,
                    b0.0.as_ptr() as *const u8,
                    b1.0.as_ptr() as *const u8,
                    b2.0.as_ptr() as *const u8,
                    b3.0.as_ptr() as *const u8,
                    n,
                );
            }
            for _ in 0..leftover {
                acpu::matrix::tile::microkernel_16x16(
                    ap.0.as_ptr() as *const u8,
                    b0.0.as_ptr() as *const u8,
                    n,
                );
            }
            amx_clr();
        })
    };

    // --- B pack scalar (small.rs style: NR-wide strips) ---
    let pack_b_ns = {
        use core::mem::MaybeUninit;

        #[repr(align(64))]
        struct BPack([MaybeUninit<f32>; 256 * 256]);

        let t = median_ns(|| {
            let mut buf: BPack = unsafe { MaybeUninit::uninit().assume_init() };
            let bp = &mut buf.0[..];
            for s in 0..n_mr {
                let cs = s * MR;
                let cols = MR.min(n - cs);
                let base = s * n * MR;
                for p in 0..n {
                    let src = p * n + cs;
                    let dst = base + p * MR;
                    for j in 0..cols {
                        bp[dst + j] = MaybeUninit::new(b[src + j]);
                    }
                }
            }
            std::hint::black_box(&buf);
        });
        t
    };

    // --- Preload + compute + store (new path) ---
    let preload_path_ns = {
        #[repr(align(64))]
        struct Buf([f32; 256 * 16]);

        let ap = Buf([1.0f32; 256 * 16]);
        let b0 = Buf([1.0f32; 256 * 16]);
        let b1 = Buf([1.0f32; 256 * 16]);
        let b2 = Buf([1.0f32; 256 * 16]);
        let b3 = Buf([1.0f32; 256 * 16]);
        let mut c_buf = vec![0.0f32; n * n];

        let batches = n_tiles / 4;

        median_ns(|| unsafe {
            use acpu::matrix::asm::*;
            amx_set();
            for batch in 0..batches {
                let ir = batch / (n_mr / 4);
                let jr4 = batch % (n_mr / 4);
                let cp = c_buf.as_mut_ptr().add(ir * MR * n + jr4 * 4 * MR);
                for t in 0u8..4 {
                    acpu::matrix::tile::preload_c(cp.add(t as usize * MR), n, t);
                }
                acpu::matrix::tile::microkernel_16x64_acc(
                    ap.0.as_ptr() as *const u8,
                    b0.0.as_ptr() as *const u8,
                    b1.0.as_ptr() as *const u8,
                    b2.0.as_ptr() as *const u8,
                    b3.0.as_ptr() as *const u8,
                    n,
                    64,
                );
                for t in 0u8..4 {
                    acpu::matrix::tile::store_c(cp.add(t as usize * MR), n, t);
                }
            }
            amx_clr();
            std::hint::black_box(&c_buf);
        })
    };

    // --- C accumulate: old path (STZ → temp → NEON add) ---
    let accum_old_ns = {
        let mut c_buf = vec![0.0f32; n * n];

        median_ns(|| unsafe {
            use acpu::matrix::asm::*;
            amx_set();

            for t in 0..n_tiles {
                let ir = t / n_mr;
                let jr = t % n_mr;
                let cp = c_buf.as_mut_ptr().add(ir * MR * n + jr * MR);
                acpu::matrix::tile::accumulate_tile(cp, n, 0);
            }

            amx_clr();
            std::hint::black_box(&c_buf);
        })
    };

    // --- Print results ---
    let row = |name: &str, ns: u64| {
        let pct = ns as f64 / sgemm_ns as f64 * 100.0;
        println!("{:>32}  {:>7}  {:>4.0}%", name, ns, pct);
    };

    row("AMX set+clr", setclr_ns);
    row("A pack scalar", pack_scalar_ns);
    row("A pack NEON", pack_neon_ns);
    row("B pack scalar", pack_b_ns);
    row("compute per-op (LDX+LDY+FMA)", compute_perop_ns);
    row("compute µkernel_16x16", compute_uk16_ns);
    row("compute µkernel_16x64 (fused)", compute_uk64_ns);
    row("preload+µk64_acc+store (new)", preload_path_ns);
    row("accum old (STZ→tmp→NEON add)", accum_old_ns);
    println!("{:>32}  {:>7}  {:>4.0}%", "----------", "---", "");
    row("sgemm (full)", sgemm_ns);
    row("Accelerate", accel_ns);

    let gap = sgemm_ns as f64 / accel_ns as f64;
    println!("{:>32}  {:>7.1}x", "gap (acpu/accel)", gap);
}
