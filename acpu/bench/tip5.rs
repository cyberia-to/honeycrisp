//! Tip5 throughput benchmark — acpu's scalar Tip5 vs `twenty-first`.
//!
//! Why measure: `twenty-first` is the reference impl that downstream
//! provers (Triton VM, nockchain) use today. acpu offers a drop-in
//! replacement with no Montgomery-form leakage at the API boundary;
//! this bench confirms the swap doesn't regress per-call cost.

#[path = "common.rs"]
mod common;
use common::*;

use acpu::field::tip5::{
    tip5_hash_pair, tip5_hash_pair_n, tip5_hash_pair_n_batch4, tip5_hash_varlen, tip5_permute,
    STATE_SIZE,
};
use twenty_first::prelude::{BFieldElement, Digest, Tip5};

fn ref_permute(state: [u64; STATE_SIZE]) -> [u64; STATE_SIZE] {
    let mut t5 = Tip5 {
        state: state.map(BFieldElement::new),
    };
    t5.permutation();
    t5.state.map(|b| b.value())
}

fn ref_hash_pair(left: [u64; 5], right: [u64; 5]) -> [u64; 5] {
    let l = Digest::new(left.map(BFieldElement::new));
    let r = Digest::new(right.map(BFieldElement::new));
    Tip5::hash_pair(l, r).values().map(|b| b.value())
}

fn ref_hash_varlen(input: &[u64]) -> [u64; 5] {
    let buf: Vec<BFieldElement> = input.iter().copied().map(BFieldElement::new).collect();
    Tip5::hash_varlen(&buf).values().map(|b| b.value())
}

fn main() {
    let caps = acpu::scan();
    println!("acpu Tip5 benchmark — acpu vs twenty-first");
    println!("chip: {}", caps.chip);
    println!();

    let mut score = Score::vs("ty-first");

    // ── permutation ──────────────────────────────────────────────────────
    score.hdr("Tip5 permutation");
    let seed: [u64; STATE_SIZE] = core::array::from_fn(|i| (i as u64 + 1).wrapping_mul(0xDEAD_BEEF));

    // Sanity check: outputs match.
    let mut acpu_buf = seed;
    let mut ref_buf = seed;
    tip5_permute(&mut acpu_buf);
    let ref_out = ref_permute(ref_buf);
    assert_eq!(acpu_buf, ref_out, "permute output diverges");
    let _ = &mut ref_buf;

    // Batch many permutes per timed iteration — timer resolution ~40ns
    // would otherwise mask the difference between the two paths.
    const BATCH: usize = 1024;
    let acpu_ns = best_of(
        || {
            let mut s = seed;
            for _ in 0..BATCH {
                tip5_permute(&mut s);
            }
            std::hint::black_box(s);
        },
        200,
    );
    let ref_ns = best_of(
        || {
            let mut s = seed;
            for _ in 0..BATCH {
                s = ref_permute(s);
            }
            std::hint::black_box(s);
        },
        200,
    );
    let acpu_per = acpu_ns / BATCH as u64;
    let ref_per = ref_ns / BATCH as u64;
    score.row(
        &format!("permute (batched ×{BATCH})"),
        acpu_per,
        ref_per,
    );

    println!(
        "  acpu throughput  = {:.2} M perm/s",
        1000.0 / acpu_per as f64
    );
    println!(
        "  twenty-first     = {:.2} M perm/s",
        1000.0 / ref_per as f64
    );

    // ── hash_pair ────────────────────────────────────────────────────────
    score.hdr("Tip5 hash_pair (Merkle inner node)");
    let l: [u64; 5] = [1, 2, 3, 4, 5];
    let r: [u64; 5] = [6, 7, 8, 9, 10];
    assert_eq!(tip5_hash_pair(l, r), ref_hash_pair(l, r));
    let acpu_ns = best_of(
        || {
            let d = tip5_hash_pair(l, r);
            std::hint::black_box(d);
        },
        50_000,
    );
    let ref_ns = best_of(
        || {
            let d = ref_hash_pair(l, r);
            std::hint::black_box(d);
        },
        50_000,
    );
    score.row("hash_pair", acpu_ns, ref_ns);

    // ── hash_varlen ──────────────────────────────────────────────────────
    score.hdr("Tip5 hash_varlen (sponge absorb)");
    for &n in &[0usize, 1, 10, 100, 1000] {
        let input: Vec<u64> = (0..n as u64).collect();
        assert_eq!(tip5_hash_varlen(&input), ref_hash_varlen(&input));
        let acpu_ns = best_of(
            || {
                let d = tip5_hash_varlen(&input);
                std::hint::black_box(d);
            },
            if n > 100 { 5_000 } else { 30_000 },
        );
        let ref_ns = best_of(
            || {
                let d = ref_hash_varlen(&input);
                std::hint::black_box(d);
            },
            if n > 100 { 5_000 } else { 30_000 },
        );
        score.row(&format!("hash_varlen[{n}]"), acpu_ns, ref_ns);
    }

    // ── Merkle layer throughput ──────────────────────────────────────────
    score.hdr("Tip5 Merkle layer (1024 leaves → 512 inner nodes)");
    let leaves: Vec<[u64; 5]> = (0..1024)
        .map(|i| core::array::from_fn(|j| (i * 5 + j) as u64))
        .collect();
    let mut out = vec![[0u64; 5]; 512];

    let acpu_ns = best_of(
        || {
            for i in 0..512 {
                out[i] = tip5_hash_pair(leaves[i * 2], leaves[i * 2 + 1]);
            }
            std::hint::black_box(&out);
        },
        200,
    );
    let ref_ns = best_of(
        || {
            for i in 0..512 {
                out[i] = ref_hash_pair(leaves[i * 2], leaves[i * 2 + 1]);
            }
            std::hint::black_box(&out);
        },
        200,
    );
    score.row("layer-512", acpu_ns, ref_ns);
    let acpu_mhash = 512.0e9 / acpu_ns as f64 / 1_000_000.0;
    let ref_mhash = 512.0e9 / ref_ns as f64 / 1_000_000.0;
    println!("  acpu  Merkle layer = {acpu_mhash:.2} M hashes/s");
    println!("  ty-first Merkle layer = {ref_mhash:.2} M hashes/s");

    // ── Batched Merkle layer ─────────────────────────────────────────────
    score.hdr("Tip5 Merkle layer — batched API (1024 leaves → 512 nodes)");
    let pairs: Vec<([u64; 5], [u64; 5])> = (0..512)
        .map(|i| {
            (
                core::array::from_fn(|j| (i * 10 + j) as u64),
                core::array::from_fn(|j| (i * 10 + 5 + j) as u64),
            )
        })
        .collect();
    let mut out_a = vec![[0u64; 5]; 512];
    let mut out_b = vec![[0u64; 5]; 512];

    // Sequential through the batched API
    let n_ns = best_of(
        || {
            tip5_hash_pair_n(&pairs, &mut out_a);
            std::hint::black_box(&out_a);
        },
        200,
    );
    let n4_ns = best_of(
        || {
            tip5_hash_pair_n_batch4(&pairs, &mut out_b);
            std::hint::black_box(&out_b);
        },
        200,
    );

    // Correctness: batch4 must agree with sequential
    assert_eq!(out_a, out_b, "batch4 diverges from sequential");

    score.row("hash_pair_n (seq)", n_ns, n_ns);
    score.row("hash_pair_n_batch4", n4_ns, n_ns);
    let n_mhash = 512e9 / n_ns as f64 / 1_000_000.0;
    let n4_mhash = 512e9 / n4_ns as f64 / 1_000_000.0;
    println!("  hash_pair_n      = {n_mhash:.2} M hashes/s");
    println!("  hash_pair_n_b4   = {n4_mhash:.2} M hashes/s  ({:.2}× over seq)", n_ns as f64 / n4_ns as f64);

    // ── Batched Goldilocks multiply (SSVE vs scalar) ────────────────────
    score.hdr("Goldilocks raw_mul: SSVE batch8 vs scalar (8 mults / call)");
    if caps.has_sme {
        use acpu::field::tip5::simd::raw_mul_batch8;
        // 8 elements per call. Compare against 8 scalar Montgomery muls.
        let a: [u64; 8] = core::array::from_fn(|i| (i as u64 + 1).wrapping_mul(0xDEAD_BEEF));
        let b: [u64; 8] = core::array::from_fn(|i| (i as u64 + 17).wrapping_mul(0xCAFE_BABE));
        let mut out = [0u64; 8];

        // Warm
        for _ in 0..3 {
            raw_mul_batch8(&a, &b, &mut out).unwrap();
        }

        // Batched per timed iteration so the timer can resolve it.
        const N_CALLS: usize = 1000;

        // SSVE: each call is one SMSTART + 8 muls + one SMSTOP.
        let sme_ns = best_of(
            || {
                for _ in 0..N_CALLS {
                    raw_mul_batch8(&a, &b, &mut out).unwrap();
                }
                std::hint::black_box(&out);
            },
            200,
        );

        const P: u64 = 0xffff_ffff_0000_0001;
        fn montyred(x: u128) -> u64 {
            let xl = x as u64;
            let xh = (x >> 64) as u64;
            let (a, e) = xl.overflowing_add(xl << 32);
            let b = a.wrapping_sub(a >> 32).wrapping_sub(e as u64);
            let (r, c) = xh.overflowing_sub(b);
            r.wrapping_sub((1 + !P) * c as u64)
        }
        let scalar_ns = best_of(
            || {
                let mut o = [0u64; 8];
                for _ in 0..N_CALLS {
                    for i in 0..8 {
                        o[i] = montyred((a[i] as u128) * (b[i] as u128));
                    }
                    std::hint::black_box(&o);
                }
            },
            200,
        );

        let sme_per_call = sme_ns / N_CALLS as u64;
        let scalar_per_call = scalar_ns / N_CALLS as u64;
        score.row("8× raw_mul (per call)", sme_per_call, scalar_per_call);
        println!(
            "  per-mul SSVE   = {:.2} ns ({} calls)",
            sme_per_call as f64 / 8.0,
            N_CALLS
        );
        println!(
            "  per-mul scalar = {:.2} ns ({} calls)",
            scalar_per_call as f64 / 8.0,
            N_CALLS
        );

        // Streaming-mode-amortized variant: enter Stream once, do N_CALLS
        // multiplies, exit once. Strips out the SMSTART/SMSTOP cost.
        let amort_ns = best_of(
            || {
                let stream = acpu::Stream::new().unwrap();
                for _ in 0..N_CALLS {
                    unsafe {
                        // Re-uses the same asm but skips the Stream::new path.
                        acpu::field::tip5::simd::raw_mul_batch8_in_stream(
                            &stream, &a, &b, &mut out,
                        );
                    }
                }
                drop(stream);
                std::hint::black_box(&out);
            },
            200,
        );
        let amort_per_call = amort_ns / N_CALLS as u64;
        println!(
            "  per-mul SSVE (amort) = {:.2} ns  → {:.2} M-mul/s",
            amort_per_call as f64 / 8.0,
            8000.0 / amort_per_call as f64
        );
    } else {
        println!("  SKIP: FEAT_SME not present");
    }

    score.summary();
}
