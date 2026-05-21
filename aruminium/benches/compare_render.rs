//! Render dispatch comparison: aruminium vs metal-rs vs wgpu (Metal backend)
//!
//! Answers the question: is aruminium fast relative to industry references?
//! - metal-rs: most-used Rust Metal wrapper; also powers wgpu's Metal backend
//! - wgpu: what Bevy uses; adds cross-platform abstraction overhead on top of Metal

mod aruminium;
mod metal_rs;
mod shaders;
mod wgpu_bench;

fn us(secs: f64) -> f64 {
    secs * 1_000_000.0
}

fn min_of<F: Fn() -> f64>(n: usize, f: F) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..n {
        best = best.min(f());
    }
    best
}

fn main() {
    println!("=== Render dispatch: aruminium vs metal-rs vs wgpu (Metal) ===\n");
    println!(
        "{:<28} {:>12} {:>12} {:>12}",
        "Test", "aruminium", "metal-rs", "wgpu"
    );
    println!("{}", "-".repeat(66));

    // Render dispatch: single draw per command buffer, sync round-trip
    let iters = 500;
    let a = min_of(3, || aruminium::render_pass_overhead(iters));
    let m = min_of(3, || metal_rs::render_pass_overhead(iters));
    let w = min_of(3, || wgpu_bench::render_pass_overhead(iters));
    println!(
        "{:<28} {:>10.1} us {:>10.1} us {:>10.1} us",
        "Render dispatch",
        us(a),
        us(m),
        us(w)
    );
    println!(
        "{:<28} {:>12} {:>10.2}x {:>10.2}x",
        "  ratio vs aruminium",
        "",
        m / a,
        w / a
    );

    println!();

    // Draw batch: 100 draws per command buffer, amortized cost per draw
    let a = min_of(3, || aruminium::render_batch_encode(100, 200));
    let m = min_of(3, || metal_rs::render_batch_encode(100, 200));
    let w = min_of(3, || wgpu_bench::render_batch_encode(100, 200));
    println!(
        "{:<28} {:>10.2} us {:>10.2} us {:>10.2} us",
        "Draw batch/op (100 draws)",
        us(a),
        us(m),
        us(w)
    );
    println!(
        "{:<28} {:>12} {:>10.2}x {:>10.2}x",
        "  ratio vs aruminium",
        "",
        m / a,
        w / a
    );

    println!("\n{}", "-".repeat(66));
    println!("ratio > 1.0 = competitor is slower than aruminium");
    println!("wgpu overhead includes: cross-platform validation + Metal translation layer");
}
