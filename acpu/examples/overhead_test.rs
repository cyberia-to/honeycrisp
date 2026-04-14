//! Measure overhead of each sgemm component at 64×64.

use std::time::Instant;

fn main() {
    const N: usize = 64;
    const ITERS: usize = 10000;

    let a = vec![1.0f32; N * N];
    let b = vec![1.0f32; N * N];

    // 1. Baseline: just call sgemm.
    let mut c = vec![0.0f32; N * N];
    let t0 = Instant::now();
    for _ in 0..ITERS {
        c.fill(0.0);
        acpu::matmul_f32(&a, &b, &mut c, N, N, N);
    }
    let sgemm_ns = t0.elapsed().as_nanos() / ITERS as u128;
    println!("sgemm 64x64:         {} ns", sgemm_ns);

    // 2. Just Matrix new/drop.
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let ctx = acpu::Matrix::new().unwrap();
        drop(ctx);
    }
    let ctx_ns = t0.elapsed().as_nanos() / ITERS as u128;
    println!("Matrix new+drop:     {} ns", ctx_ns);

    // 3. detect() call.
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = acpu::scan();
    }
    let detect_ns = t0.elapsed().as_nanos() / ITERS as u128;
    println!("detect() cached:     {} ns", detect_ns);

    // 4. AlignedBuf alloc+free (simulated via alloc API).
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let layout = std::alloc::Layout::from_size_align(256 * 512 * 4, 64).unwrap();
        let p = unsafe { std::alloc::alloc_zeroed(layout) };
        unsafe { std::alloc::dealloc(p, layout) };
    }
    let alloc_ns = t0.elapsed().as_nanos() / ITERS as u128;
    println!("alloc 512KB+free:    {} ns", alloc_ns);

    // 5. Small alloc.
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let layout = std::alloc::Layout::from_size_align(64 * 256 * 4, 64).unwrap();
        let p = unsafe { std::alloc::alloc_zeroed(layout) };
        unsafe { std::alloc::dealloc(p, layout) };
    }
    let small_alloc_ns = t0.elapsed().as_nanos() / ITERS as u128;
    println!("alloc 64KB+free:     {} ns", small_alloc_ns);

    // 6. Pure AMX: set, 16 tiles × 64 rank-1, clr.
    let t0 = Instant::now();
    for _ in 0..ITERS {
        unsafe {
            use acpu::matrix::asm::*;
            use acpu::matrix::fma::*;
            use acpu::matrix::regs::*;
            amx_set();
            // 16 tiles (4×4), each with k=64 rank-1 updates.
            // Just do 4 tiles × 64 updates as minimal compute.
            for _ in 0..64 {
                amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
            }
            amx_clr();
        }
    }
    let pure_amx_ns = t0.elapsed().as_nanos() / ITERS as u128;
    println!("pure AMX 64 FMA32:   {} ns", pure_amx_ns);

    let overhead = sgemm_ns - pure_amx_ns;
    println!(
        "\noverhead:            {} ns ({:.0}%)",
        overhead,
        overhead as f64 / sgemm_ns as f64 * 100.0
    );
}
