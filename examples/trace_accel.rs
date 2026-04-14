//! Loop cblas_sgemm for profiling. Usage: trace_accel [size] [iters]
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
    let sz = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(64);
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5_000_000);

    let a = vec![1.0f32; (sz * sz) as usize];
    let b = vec![1.0f32; (sz * sz) as usize];
    let mut c = vec![0.0f32; (sz * sz) as usize];

    eprintln!(
        "looping {}×{} for {} iters (pid {})",
        sz,
        sz,
        iters,
        std::process::id()
    );
    for _ in 0..iters {
        c.fill(0.0);
        unsafe {
            cblas_sgemm(
                101,
                111,
                111,
                sz,
                sz,
                sz,
                1.0,
                a.as_ptr(),
                sz,
                b.as_ptr(),
                sz,
                0.0,
                c.as_mut_ptr(),
                sz,
            );
        }
        std::hint::black_box(&c);
    }
    eprintln!("done");
}
