//! Simple 64x64 matrix multiply example using acpu::matmul_f32.
//!
//! Run with: `cargo run --example matmul --release`

fn main() {
    const N: usize = 64;

    // A = all ones.
    let a = vec![1.0f32; N * N];

    // B = identity matrix.
    let mut b = vec![0.0f32; N * N];
    for i in 0..N {
        b[i * N + i] = 1.0;
    }

    // C = zero (sgemm accumulates: C += A * B).
    let mut c = vec![0.0f32; N * N];

    let start = std::time::Instant::now();
    acpu::matmul_f32(&a, &b, &mut c, N, N, N);
    let elapsed = start.elapsed();

    // Verify: C should equal A (all ones) since B is identity.
    let mut max_err = 0.0f32;
    for i in 0..N * N {
        let err = (c[i] - 1.0).abs();
        if err > max_err {
            max_err = err;
        }
    }

    println!("acpu::matmul_f32 {N}x{N} x {N}x{N}");
    println!("  elapsed:   {:.3?}", elapsed);
    println!("  max error: {:.2e}", max_err);

    if max_err < 1e-5 {
        println!("  result:    PASS");
    } else {
        println!("  result:    FAIL");
        std::process::exit(1);
    }
}
