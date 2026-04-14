//! Memory subsystem benchmark: latency, bandwidth, prefetch.
#[path = "common.rs"]
mod common;

use std::time::Instant;

// ---------------------------------------------------------------------------
// Pointer-chase memory latency
// ---------------------------------------------------------------------------

fn chase_latency_ns(size_bytes: usize) -> f64 {
    let n = size_bytes / std::mem::size_of::<usize>();
    let mut arr: Vec<usize> = (0..n).collect();
    // Sattolo: guaranteed single-cycle random permutation
    let mut rng: u64 = 0xdeadbeef12345678;
    for i in (1..n).rev() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng >> 33) as usize % i;
        arr.swap(i, j);
    }
    // warmup
    let mut idx = 0usize;
    for _ in 0..n {
        idx = arr[idx];
    }
    let iters = 1_000_000usize.min(n * 10).max(n);
    let s = Instant::now();
    for _ in 0..iters {
        idx = unsafe { *arr.get_unchecked(idx) };
    }
    let elapsed = s.elapsed().as_nanos() as f64;
    std::hint::black_box(idx);
    elapsed / iters as f64
}

// ---------------------------------------------------------------------------
// STREAM bandwidth kernels (NEON intrinsics, 16-wide unroll)
// ---------------------------------------------------------------------------

const STREAM_N: usize = 4 * 1024 * 1024; // 4M f32 = 16MB per array

fn stream_copy(a: &[f32], c: &mut [f32]) {
    c.copy_from_slice(a);
    std::hint::black_box(&c);
}

fn stream_scale(a: &[f32], c: &mut [f32], scalar: f32) {
    let n = a.len();
    unsafe {
        use core::arch::aarch64::*;
        let sv = vdupq_n_f32(scalar);
        let pa = a.as_ptr();
        let pc = c.as_mut_ptr() as *mut u8;
        let mut i = 0;
        while i + 32 <= n {
            // Prefetch ahead
            core::arch::asm!(
                "prfm pldl1strm, [{pa}, #512]",
                pa = in(reg) pa.add(i),
            );
            let r0 = vmulq_f32(sv, vld1q_f32(pa.add(i)));
            let r1 = vmulq_f32(sv, vld1q_f32(pa.add(i + 4)));
            let r2 = vmulq_f32(sv, vld1q_f32(pa.add(i + 8)));
            let r3 = vmulq_f32(sv, vld1q_f32(pa.add(i + 12)));
            let r4 = vmulq_f32(sv, vld1q_f32(pa.add(i + 16)));
            let r5 = vmulq_f32(sv, vld1q_f32(pa.add(i + 20)));
            let r6 = vmulq_f32(sv, vld1q_f32(pa.add(i + 24)));
            let r7 = vmulq_f32(sv, vld1q_f32(pa.add(i + 28)));
            // Non-temporal store pairs (STNP avoids read-for-ownership)
            core::arch::asm!(
                "stnp q0, q1, [{p}]",
                "stnp q2, q3, [{p}, #32]",
                "stnp q4, q5, [{p}, #64]",
                "stnp q6, q7, [{p}, #96]",
                p = in(reg) pc.add(i * 4),
                in("v0") r0, in("v1") r1, in("v2") r2, in("v3") r3,
                in("v4") r4, in("v5") r5, in("v6") r6, in("v7") r7,
            );
            i += 32;
        }
        while i + 4 <= n {
            vst1q_f32(
                pc.add(i * 4) as *mut f32,
                vmulq_f32(sv, vld1q_f32(pa.add(i))),
            );
            i += 4;
        }
    }
    std::hint::black_box(&c);
}

fn stream_add(a: &[f32], b: &[f32], c: &mut [f32]) {
    let n = a.len();
    unsafe {
        use core::arch::aarch64::*;
        let pa = a.as_ptr();
        let pb = b.as_ptr();
        let pc = c.as_mut_ptr() as *mut u8;
        let mut i = 0;
        while i + 32 <= n {
            core::arch::asm!(
                "prfm pldl1strm, [{pa}, #512]",
                "prfm pldl1strm, [{pb}, #512]",
                pa = in(reg) pa.add(i),
                pb = in(reg) pb.add(i),
            );
            let r0 = vaddq_f32(vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            let r1 = vaddq_f32(vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
            let r2 = vaddq_f32(vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
            let r3 = vaddq_f32(vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
            let r4 = vaddq_f32(vld1q_f32(pa.add(i + 16)), vld1q_f32(pb.add(i + 16)));
            let r5 = vaddq_f32(vld1q_f32(pa.add(i + 20)), vld1q_f32(pb.add(i + 20)));
            let r6 = vaddq_f32(vld1q_f32(pa.add(i + 24)), vld1q_f32(pb.add(i + 24)));
            let r7 = vaddq_f32(vld1q_f32(pa.add(i + 28)), vld1q_f32(pb.add(i + 28)));
            core::arch::asm!(
                "stnp q0, q1, [{p}]",
                "stnp q2, q3, [{p}, #32]",
                "stnp q4, q5, [{p}, #64]",
                "stnp q6, q7, [{p}, #96]",
                p = in(reg) pc.add(i * 4),
                in("v0") r0, in("v1") r1, in("v2") r2, in("v3") r3,
                in("v4") r4, in("v5") r5, in("v6") r6, in("v7") r7,
            );
            i += 32;
        }
        while i + 4 <= n {
            vst1q_f32(
                pc.add(i * 4) as *mut f32,
                vaddq_f32(vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i))),
            );
            i += 4;
        }
    }
    std::hint::black_box(&c);
}

fn stream_triad(a: &[f32], b: &[f32], d: &mut [f32], scalar: f32) {
    let n = a.len();
    unsafe {
        use core::arch::aarch64::*;
        let sv = vdupq_n_f32(scalar);
        let pa = a.as_ptr();
        let pb = b.as_ptr();
        let pd = d.as_mut_ptr() as *mut u8;
        let mut i = 0;
        while i + 32 <= n {
            core::arch::asm!(
                "prfm pldl1strm, [{pa}, #512]",
                "prfm pldl1strm, [{pb}, #512]",
                pa = in(reg) pa.add(i),
                pb = in(reg) pb.add(i),
            );
            let r0 = vfmaq_f32(vld1q_f32(pa.add(i)), sv, vld1q_f32(pb.add(i)));
            let r1 = vfmaq_f32(vld1q_f32(pa.add(i + 4)), sv, vld1q_f32(pb.add(i + 4)));
            let r2 = vfmaq_f32(vld1q_f32(pa.add(i + 8)), sv, vld1q_f32(pb.add(i + 8)));
            let r3 = vfmaq_f32(vld1q_f32(pa.add(i + 12)), sv, vld1q_f32(pb.add(i + 12)));
            let r4 = vfmaq_f32(vld1q_f32(pa.add(i + 16)), sv, vld1q_f32(pb.add(i + 16)));
            let r5 = vfmaq_f32(vld1q_f32(pa.add(i + 20)), sv, vld1q_f32(pb.add(i + 20)));
            let r6 = vfmaq_f32(vld1q_f32(pa.add(i + 24)), sv, vld1q_f32(pb.add(i + 24)));
            let r7 = vfmaq_f32(vld1q_f32(pa.add(i + 28)), sv, vld1q_f32(pb.add(i + 28)));
            core::arch::asm!(
                "stnp q0, q1, [{p}]",
                "stnp q2, q3, [{p}, #32]",
                "stnp q4, q5, [{p}, #64]",
                "stnp q6, q7, [{p}, #96]",
                p = in(reg) pd.add(i * 4),
                in("v0") r0, in("v1") r1, in("v2") r2, in("v3") r3,
                in("v4") r4, in("v5") r5, in("v6") r6, in("v7") r7,
            );
            i += 32;
        }
        while i + 4 <= n {
            vst1q_f32(
                pd.add(i * 4) as *mut f32,
                vfmaq_f32(vld1q_f32(pa.add(i)), sv, vld1q_f32(pb.add(i))),
            );
            i += 4;
        }
    }
    std::hint::black_box(&d);
}

// ---------------------------------------------------------------------------
// Prefetch impact — strided access over 16MB
// ---------------------------------------------------------------------------

fn strided_sum_no_prefetch(data: &[f32], stride: usize) -> u64 {
    let n = data.len();
    common::ns(|| unsafe {
        let p = data.as_ptr();
        let mut acc = 0f32;
        let mut i = 0;
        while i < n {
            acc += *p.add(i);
            i += stride;
        }
        std::hint::black_box(acc);
    })
}

fn strided_sum_prefetch(data: &[f32], stride: usize) -> u64 {
    let n = data.len();
    let ahead = stride * 8;
    common::ns(|| unsafe {
        let p = data.as_ptr();
        let mut acc = 0f32;
        let mut i = 0;
        while i + ahead < n {
            acpu::sync::prefetch::prefetch_l2(p.add(i + ahead) as *const u8);
            acc += *p.add(i);
            i += stride;
        }
        while i < n {
            acc += *p.add(i);
            i += stride;
        }
        std::hint::black_box(acc);
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    // 180s hard timeout
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(180));
        eprintln!("\n!!! 180s TIMEOUT !!!");
        std::process::exit(1);
    });

    let caps = acpu::probe::scan();
    eprintln!(
        "=== memory subsystem benchmark — {:?} ({}P+{}E) ===",
        caps.chip, caps.p_cores, caps.e_cores
    );

    // ── MEMORY LATENCY ───────────────────────────────────────────────────
    // Reference: Apple M1Pro published/measured specs
    eprintln!("\n  MEMORY LATENCY (pointer chasing, random access)");
    eprintln!(
        "  {:<18} {:>10} {:>10} {:>10}",
        "level", "size", "measured", "reference"
    );
    eprintln!("  {}", "-".repeat(52));
    // (size_bytes, label, reference_ns) — reference from Anandtech/Chips&Cheese M1Pro measurements
    for &(bytes, label, ref_ns) in &[
        (16 * 1024, "L1  16KB", "~1.3ns"),
        (128 * 1024, "L2  128KB", "~4ns"),
        (1024 * 1024, "L2  1MB", "~6ns"),
        (4 * 1024 * 1024, "L3  4MB", "~10ns"),
        (32 * 1024 * 1024, "L3  32MB", "~30-90ns"),
        (128 * 1024 * 1024, "DRAM 128MB", "~100ns"),
    ] {
        let lat = chase_latency_ns(bytes);
        eprintln!(
            "  {:<18} {:>8}KB {:>8.1}ns {:>10}",
            label,
            bytes / 1024,
            lat,
            ref_ns
        );
    }

    // ── STREAM BANDWIDTH ─────────────────────────────────────────────────
    // Methodology: pin to P-core, warmup 10 rounds, measure 3 rounds of
    // 20 iterations each, take best iteration across all rounds.
    // Reports best + coefficient of variation (CV) for reliability.
    eprintln!("\n  STREAM BANDWIDTH (4M f32 = 16MB, pinned P-core)");
    eprintln!(
        "  {:<12} {:>8} {:>8} {:>7} {:>6} {:>6}",
        "kernel", "best", "ref", "ratio", "CV%", ""
    );
    eprintln!("  {}", "-".repeat(52));

    // Pin to P-core for stable measurements
    let _ = acpu::sync::affinity::pin_p_core();

    let a_arr: Vec<f32> = (0..STREAM_N).map(|i| (i % 1000) as f32 * 0.001).collect();
    let b_arr: Vec<f32> = (0..STREAM_N).map(|i| (i % 997) as f32 * 0.001).collect();
    let mut c_arr = vec![0f32; STREAM_N];
    let mut d_arr = vec![0f32; STREAM_N];
    let scalar = 3.14159f32;
    let bytes = STREAM_N as f64 * 4.0;

    let specs: [(&str, f64, f64); 4] = [
        ("copy", 2.0, 96.0),
        ("scale", 2.0, 64.0),
        ("add", 3.0, 80.0),
        ("triad", 3.0, 80.0),
    ];

    // Warmup: 10 rounds of all kernels to stabilize thermal
    for _ in 0..10 {
        stream_copy(&a_arr, &mut c_arr);
        stream_scale(&a_arr, &mut c_arr, scalar);
        stream_add(&a_arr, &b_arr, &mut c_arr);
        stream_triad(&a_arr, &b_arr, &mut d_arr, scalar);
    }

    // Measure: 3 rounds × 20 iterations, collect all timings
    let rounds = 3;
    let iters_per = 20;
    let mut all_times: [[Vec<u64>; 4]; 1] = [Default::default()];
    let times = &mut all_times[0];
    for k in 0..4 {
        times[k] = Vec::with_capacity(rounds * iters_per);
    }

    for _ in 0..rounds {
        for _ in 0..iters_per {
            let s = Instant::now();
            stream_copy(&a_arr, &mut c_arr);
            times[0].push(s.elapsed().as_nanos() as u64);

            let s = Instant::now();
            stream_scale(&a_arr, &mut c_arr, scalar);
            times[1].push(s.elapsed().as_nanos() as u64);

            let s = Instant::now();
            stream_add(&a_arr, &b_arr, &mut c_arr);
            times[2].push(s.elapsed().as_nanos() as u64);

            let s = Instant::now();
            stream_triad(&a_arr, &b_arr, &mut d_arr, scalar);
            times[3].push(s.elapsed().as_nanos() as u64);
        }
        // Brief pause between rounds to let thermal stabilize
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    for k in 0..4 {
        let (name, factor, ref_gbs) = specs[k];
        let t = &times[k];
        let best = *t.iter().min().unwrap();
        let mean = t.iter().sum::<u64>() as f64 / t.len() as f64;
        let variance = t
            .iter()
            .map(|&x| {
                let d = x as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / t.len() as f64;
        let stddev = variance.sqrt();
        let cv = stddev / mean * 100.0; // coefficient of variation

        let best_gbs = factor * bytes / best as f64;
        let pct = best_gbs / ref_gbs * 100.0;
        let mark = if pct >= 95.0 {
            "✓"
        } else if pct >= 80.0 {
            "~"
        } else {
            "✗"
        };
        eprintln!(
            "  {:<12} {:>6.1} {:>6.1} {:>5.0}% {:>5.1} {:>4}",
            name, best_gbs, ref_gbs, pct, cv, mark
        );
    }
    eprintln!("  ({rounds}×{iters_per} iters, pinned P-core, CV = coeff of variation)");

    // Reset affinity
    let _ = acpu::sync::affinity::pin_any();

    // ── PREFETCH IMPACT ──────────────────────────────────────────────────
    eprintln!("\n  PREFETCH IMPACT (stride access over 16MB)");
    eprintln!("  {:<18} {:>10} {:>10}", "mode", "GB/s", "speedup");
    eprintln!("  {}", "-".repeat(40));

    let pn = 4 * 1024 * 1024usize;
    let pdata: Vec<f32> = vec![1.0; pn];
    let stride = 256usize; // 1KB stride — defeats simple HW prefetcher

    let t_no = strided_sum_no_prefetch(&pdata, stride);
    let elements_touched = pn / stride;
    let bytes_touched = elements_touched as f64 * 64.0; // each touches a cache line
    let bw_no = bytes_touched / t_no as f64;

    let t_pf = strided_sum_prefetch(&pdata, stride);
    let bw_pf = bytes_touched / t_pf as f64;

    eprintln!("  {:<18} {:>9.1} {:>10}", "no prefetch", bw_no, "baseline");
    eprintln!(
        "  {:<18} {:>9.1} {:>8.2}x",
        "prefetch_l2",
        bw_pf,
        bw_pf / bw_no
    );

    eprintln!("\ndone.");
}
