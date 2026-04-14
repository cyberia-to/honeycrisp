//! Performance comparison: aruminium (direct FFI) vs objc2-metal (ObjC bindings)
//!
//! Measures: device discovery, buffer creation, shader compilation,
//! dispatch overhead, batch encoding, pipelining, inference sim, SAXPY throughput.

mod aruminium;
mod objc2;
mod shaders;

fn us(secs: f64) -> f64 {
    secs * 1_000_000.0
}

fn ms(secs: f64) -> f64 {
    secs * 1_000.0
}

/// Run a benchmark N times, return the minimum (most stable measurement).
fn min_of<F: Fn() -> f64>(n: usize, f: F) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..n {
        best = best.min(f());
    }
    best
}

fn main() {
    println!("=== aruminium vs objc2-metal Performance Comparison ===\n");
    println!(
        "{:<30} {:>12} {:>12} {:>10}",
        "Test", "aruminium", "objc2", "ratio"
    );
    println!("{}", "-".repeat(66));

    // 1. Device discovery
    let iters = 1000;
    let r = aruminium::device_discovery(iters);
    let o = objc2::device_discovery(iters);
    println!(
        "{:<30} {:>10.2} us {:>10.2} us {:>9.2}x",
        "Device discovery",
        us(r),
        us(o),
        o / r
    );

    // 2. Buffer creation (4 KB)
    let iters = 5000;
    let r = aruminium::buffer_creation(iters, 4096);
    let o = objc2::buffer_creation(iters, 4096);
    println!(
        "{:<30} {:>10.2} us {:>10.2} us {:>9.2}x",
        "Buffer create (4 KB)",
        us(r),
        us(o),
        o / r
    );

    // 3. Buffer creation (16 MB)
    let iters = 500;
    let r = aruminium::buffer_creation(iters, 16 * 1024 * 1024);
    let o = objc2::buffer_creation(iters, 16 * 1024 * 1024);
    println!(
        "{:<30} {:>10.2} us {:>10.2} us {:>9.2}x",
        "Buffer create (16 MB)",
        us(r),
        us(o),
        o / r
    );

    // 4. Shader compilation
    let iters = 100;
    let r = aruminium::shader_compile(iters);
    let o = objc2::shader_compile(iters);
    println!(
        "{:<30} {:>10.2} ms {:>10.2} ms {:>9.2}x",
        "Shader compile (SAXPY)",
        ms(r),
        ms(o),
        o / r
    );

    // 4b. Encode overhead (CPU only, no per-iter GPU wait) — min of 3
    let iters = 5000;
    let r = min_of(3, || aruminium::encode_overhead(iters));
    let o = min_of(3, || objc2::encode_overhead(iters));
    println!(
        "{:<30} {:>10.2} us {:>10.2} us {:>9.2}x",
        "Encode (wrapper)",
        us(r),
        us(o),
        o / r
    );
    let ru = min_of(3, || aruminium::encode_unchecked(iters));
    println!(
        "{:<30} {:>10.2} us {:>10.2} us {:>9.2}x",
        "Encode (unchecked)",
        us(ru),
        us(o),
        o / ru
    );

    // 4c. Batch encode — fair comparison: both sides batch 100 dispatches per cmd
    let batch_iters = 200;
    let re = min_of(3, || aruminium::encode_encoder(100, batch_iters));
    let rb = min_of(3, || aruminium::batch_encode(100, batch_iters));
    let ob = min_of(3, || objc2::batch_encode(100, batch_iters));
    println!(
        "{:<30} {:>10.2} us {:>10.2} us {:>9.2}x",
        "Batch/op (encoder)",
        us(re),
        us(ob),
        ob / re
    );
    println!(
        "{:<30} {:>10.2} us {:>10.2} us {:>9.2}x",
        "Batch/op (IMP)",
        us(rb),
        us(ob),
        ob / rb
    );

    // 5. Dispatch overhead (tiny kernel, sync) — min of 3 runs for stability
    let iters = 1000;
    let r = min_of(3, || aruminium::dispatch_overhead(iters));
    let o = min_of(3, || objc2::dispatch_overhead(iters));
    println!(
        "{:<30} {:>10.2} us {:>10.2} us {:>9.2}x",
        "Dispatch (wrapper)",
        us(r),
        us(o),
        o / r
    );

    let raw = min_of(3, || aruminium::dispatch_raw(iters));
    println!(
        "{:<30} {:>10.2} us {:>10.2} us {:>9.2}x",
        "Dispatch (raw msgSend)",
        us(raw),
        us(o),
        o / raw
    );

    let imp = min_of(3, || aruminium::dispatch_imp(iters));
    println!(
        "{:<30} {:>10.2} us {:>10.2} us {:>9.2}x",
        "Dispatch (IMP+autorelease)",
        us(imp),
        us(o),
        o / imp
    );

    let unc = min_of(3, || aruminium::dispatch_unchecked(iters));
    println!(
        "{:<30} {:>10.2} us {:>10.2} us {:>9.2}x",
        "Dispatch (unchecked)",
        us(unc),
        us(o),
        o / unc
    );

    let ar = min_of(3, || aruminium::dispatch_autoreleased(iters));
    println!(
        "{:<30} {:>10.2} us {:>10.2} us {:>9.2}x",
        "Dispatch (autoreleased)",
        us(ar),
        us(o),
        o / ar
    );

    // 5f. Pipelined: overlap GPU wait with next encode
    let piped = min_of(3, || aruminium::dispatch_pipelined(iters));
    println!(
        "{:<30} {:>10.2} us {:>10.2} us {:>9.2}x",
        "Dispatch (pipelined)",
        us(piped),
        us(o),
        o / piped
    );

    // 5g. Inference simulation: 3 kernels × 100 layers, batched
    let inf_r = min_of(3, || aruminium::inference_sim(100, 100));
    let inf_o = min_of(3, || objc2::inference_sim(100, 100));
    println!(
        "{:<30} {:>10.2} ms {:>10.2} ms {:>9.2}x",
        "Inference (3x100 layers)",
        ms(inf_r),
        ms(inf_o),
        inf_o / inf_r
    );

    // 6. Large compute (SAXPY 16M floats)
    let n = 16 * 1024 * 1024;
    let iters = 100;
    let r = aruminium::large_compute(iters, n);
    let o = objc2::large_compute(iters, n);
    let bw_r = (n as f64 * 4.0 * 3.0) / r / 1e9;
    let bw_o = (n as f64 * 4.0 * 3.0) / o / 1e9;
    println!(
        "{:<30} {:>10.2} ms {:>10.2} ms {:>9.2}x",
        "SAXPY 16M floats",
        ms(r),
        ms(o),
        o / r
    );
    println!(
        "{:<30} {:>8.1} GB/s {:>8.1} GB/s",
        "  └ bandwidth", bw_r, bw_o
    );

    // 6b. SAXPY with Dispatch
    let ri = aruminium::large_compute_imp(iters, n);
    let bw_ri = (n as f64 * 4.0 * 3.0) / ri / 1e9;
    println!(
        "{:<30} {:>10.2} ms {:>10.2} ms {:>9.2}x",
        "SAXPY 16M (IMP+unretained)",
        ms(ri),
        ms(o),
        o / ri
    );
    println!(
        "{:<30} {:>8.1} GB/s {:>8.1} GB/s",
        "  └ bandwidth", bw_ri, bw_o
    );

    println!("\n{}", "-".repeat(66));
    println!("ratio > 1.0 = aruminium faster, < 1.0 = objc2 faster");
    println!("Note: GPU compute time dominates large workloads.");
    println!("      Binding overhead visible only in dispatch/creation paths.");
}
