//! Core topology benchmark: P vs E performance, cross-core latency.
#[path = "common.rs"]
mod common;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

// ─────────────────────────────────────────────────────────────────────────────
// P-core vs E-core workload runner
// ─────────────────────────────────────────────────────────────────────────────

fn run_on_core(
    pin: fn() -> acpu::Result<()>,
    work: Arc<dyn Fn() + Send + Sync>,
    warmup: usize,
    reps: usize,
) -> u64 {
    std::thread::spawn(move || {
        let _ = pin();
        std::thread::sleep(std::time::Duration::from_millis(10));
        for _ in 0..warmup {
            work();
        }
        let mut t = Vec::with_capacity(reps);
        for _ in 0..reps {
            let s = Instant::now();
            work();
            t.push(s.elapsed().as_nanos() as u64);
        }
        common::med(&mut t)
    })
    .join()
    .unwrap()
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-core latency (atomic ping-pong)
// ─────────────────────────────────────────────────────────────────────────────

fn ping_pong(pin_a: fn() -> acpu::Result<()>, pin_b: fn() -> acpu::Result<()>, rounds: u64) -> f64 {
    let flag = Arc::new(AtomicU64::new(0));
    let f2 = flag.clone();

    let handle = std::thread::spawn(move || {
        let _ = pin_b();
        for i in 0..rounds {
            while f2.load(Ordering::Acquire) != i * 2 + 1 {
                std::hint::spin_loop();
            }
            f2.store(i * 2 + 2, Ordering::Release);
        }
    });

    let _ = pin_a();
    std::thread::sleep(std::time::Duration::from_millis(1));

    let start = Instant::now();
    for i in 0..rounds {
        flag.store(i * 2 + 1, Ordering::Release);
        while flag.load(Ordering::Acquire) != i * 2 + 2 {
            std::hint::spin_loop();
        }
    }
    let elapsed = start.elapsed().as_nanos() as f64;
    handle.join().unwrap();
    elapsed / rounds as f64 / 2.0
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    eprintln!("╔═══════════════════════════════════════════════════════════════╗");
    eprintln!("║  CORE TOPOLOGY BENCHMARK                                     ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════╝");

    // ── P-CORE vs E-CORE ─────────────────────────────────────────────────

    eprintln!("\n  P-CORE vs E-CORE (same workload, different core class)");
    eprintln!(
        "  {:<18} {:>10} {:>10} {:>8}",
        "workload", "P-core", "E-core", "P/E"
    );
    eprintln!("  {}", "─".repeat(50));

    let workloads: Vec<(&str, Arc<dyn Fn() + Send + Sync>)> = vec![
        (
            "sum 1M",
            Arc::new({
                let s: Vec<f32> = vec![1.0; 1024 * 1024];
                move || {
                    std::hint::black_box(acpu::vector::reduce::sum(&s));
                }
            }),
        ),
        (
            "exp 64K",
            Arc::new({
                let s: Vec<f32> = (0..65536).map(|i| (i % 100) as f32 * 0.01).collect();
                move || {
                    let mut b = s.clone();
                    acpu::vector::math::exp(&mut b);
                    std::hint::black_box(&b);
                }
            }),
        ),
        (
            "sgemm 256",
            Arc::new({
                let sz = 256;
                let a: Vec<f32> = (0..sz * sz).map(|i| (i % 7) as f32 * 0.1).collect();
                let b: Vec<f32> = (0..sz * sz).map(|i| (i % 11) as f32 * 0.1).collect();
                move || {
                    let mut c = vec![0f32; sz * sz];
                    acpu::matmul_f32(&a, &b, &mut c, sz, sz, sz);
                    std::hint::black_box(&c);
                }
            }),
        ),
    ];

    for (name, work) in &workloads {
        let p_ns = run_on_core(acpu::sync::affinity::pin_p_core, work.clone(), 30, 100);
        let e_ns = run_on_core(acpu::sync::affinity::pin_e_core, work.clone(), 30, 100);
        let ratio = p_ns as f64 / e_ns as f64;
        eprintln!("  {:<18} {:>9}ns {:>9}ns {:>6.2}×", name, p_ns, e_ns, ratio);
    }

    // ── CROSS-CORE LATENCY ───────────────────────────────────────────────

    eprintln!("\n  CROSS-CORE LATENCY (atomic ping-pong, one-way)");
    eprintln!("  {:<18} {:>10}", "path", "latency");
    eprintln!("  {}", "─".repeat(30));

    let rounds = 50_000u64;
    let pp = ping_pong(
        acpu::sync::affinity::pin_p_core,
        acpu::sync::affinity::pin_p_core,
        rounds,
    );
    let pe = ping_pong(
        acpu::sync::affinity::pin_p_core,
        acpu::sync::affinity::pin_e_core,
        rounds,
    );

    eprintln!("  {:<18} {:>8.1}ns", "P ↔ P", pp);
    eprintln!("  {:<18} {:>8.1}ns", "P ↔ E", pe);
}
