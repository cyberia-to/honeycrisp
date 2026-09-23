//! Tip5 microbench — ns/permutation on the P-core, vs `twenty_first::Tip5`.
//!
//! The hot path for downstream consumers (trisha mining) is the bare
//! permutation. Numbers are reported in ns per `tip5_permute` call on a
//! 16-element state, plus throughput for the batched variant and `hash_pair`.

#[path = "common.rs"]
mod common;
use common::*;

use acpu::field::tip5::{
    tip5_hash_pair, tip5_hash_varlen, tip5_permute, tip5_permute_batch, tip5_permute_sme,
};
use twenty_first::prelude::{BFieldElement, Digest, Tip5};

fn rand_state(seed: u64) -> [u64; 16] {
    let mut s = [0u64; 16];
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    for v in s.iter_mut() {
        x = x.wrapping_mul(0x6C62_272E_07BB_0142).wrapping_add(1);
        // Keep canonical so twenty-first paths see realistic inputs.
        *v = BFieldElement::new(x).raw_u64();
    }
    s
}

fn main() {
    // Pin to a P-core for stable numbers.
    let _ = acpu::sync::affinity::pin_p_core();

    let caps = acpu::probe::scan();
    println!("acpu Tip5 microbench — {:?}", caps.chip);
    println!();

    let mut score = Score::vs("twenty-first");
    score.hdr("TIP5 (vs twenty_first 1.1)");

    let seed = 0xA1B2_C3D4_E5F6_0708u64;
    let initial = rand_state(seed);

    // ── 1. single permutation ──
    let acpu_perm_ns = best_of(
        || {
            let mut s = initial;
            for _ in 0..256 {
                tip5_permute(&mut s);
            }
            std::hint::black_box(&s);
        },
        400,
    );
    let acpu_per = acpu_perm_ns as f64 / 256.0;

    let tw_perm_ns = best_of(
        || {
            let mut t = Tip5 {
                state: initial.map(BFieldElement::from_raw_u64),
            };
            for _ in 0..256 {
                t.permutation();
            }
            std::hint::black_box(&t);
        },
        400,
    );
    let tw_per = tw_perm_ns as f64 / 256.0;

    score.row("permute (1×)", acpu_per as u64, tw_per as u64);

    // ── 2. batched permute, N=4 ──
    let acpu_batch4_ns = best_of(
        || {
            let mut s = [initial; 4];
            for _ in 0..256 {
                tip5_permute_batch::<4>(&mut s);
            }
            std::hint::black_box(&s);
        },
        400,
    );
    let acpu_batch4_per = acpu_batch4_ns as f64 / (256.0 * 4.0);
    score.row(
        "permute batch×4 (per perm)",
        acpu_batch4_per as u64,
        tw_per as u64,
    );

    // ── 3. batched permute, N=8 ──
    let acpu_batch8_ns = best_of(
        || {
            let mut s = [initial; 8];
            for _ in 0..128 {
                tip5_permute_batch::<8>(&mut s);
            }
            std::hint::black_box(&s);
        },
        400,
    );
    let acpu_batch8_per = acpu_batch8_ns as f64 / (128.0 * 8.0);
    score.row(
        "permute batch×8 (per perm)",
        acpu_batch8_per as u64,
        tw_per as u64,
    );

    // ── 3b. SME-scoped permute N=8 ──
    if acpu::probe::scan().has_sme {
        let sme_batch8_ns = best_of(
            || {
                let mut s = [initial; 8];
                for _ in 0..128 {
                    let _ = tip5_permute_sme::<8>(&mut s);
                }
                std::hint::black_box(&s);
            },
            400,
        );
        let sme_per = sme_batch8_ns as f64 / (128.0 * 8.0);
        score.row(
            "permute SME-scope×8 (per perm)",
            sme_per as u64,
            tw_per as u64,
        );

        // Also test N=16 — what the brief sketches as the natural batch.
        let sme_batch16_ns = best_of(
            || {
                let mut s = [initial; 16];
                for _ in 0..64 {
                    let _ = tip5_permute_sme::<16>(&mut s);
                }
                std::hint::black_box(&s);
            },
            400,
        );
        let sme16_per = sme_batch16_ns as f64 / (64.0 * 16.0);
        score.row(
            "permute SME-scope×16 (per perm)",
            sme16_per as u64,
            tw_per as u64,
        );
    } else {
        println!("skip: FEAT_SME not present — SME-scope rows omitted");
    }

    // ── 4. hash_pair ──
    let left = [initial[0], initial[1], initial[2], initial[3], initial[4]];
    let right = [initial[5], initial[6], initial[7], initial[8], initial[9]];
    let acpu_pair_ns = best_of(
        || {
            for _ in 0..256 {
                let d = tip5_hash_pair(left, right);
                std::hint::black_box(d);
            }
        },
        400,
    );
    let acpu_pair_per = acpu_pair_ns as f64 / 256.0;

    let l_d = Digest::new(left.map(BFieldElement::from_raw_u64));
    let r_d = Digest::new(right.map(BFieldElement::from_raw_u64));
    let tw_pair_ns = best_of(
        || {
            for _ in 0..256 {
                let d = Tip5::hash_pair(l_d, r_d);
                std::hint::black_box(d);
            }
        },
        400,
    );
    let tw_pair_per = tw_pair_ns as f64 / 256.0;
    score.row("hash_pair", acpu_pair_per as u64, tw_pair_per as u64);

    // ── 5. hash_varlen (100 elements ≈ 10 chunks) ──
    let bfe_in: Vec<BFieldElement> = (0..100u64).map(BFieldElement::new).collect();
    let raw_in: Vec<u64> = bfe_in.iter().map(|e| e.raw_u64()).collect();
    let acpu_var_ns = best_of(
        || {
            for _ in 0..32 {
                let d = tip5_hash_varlen(&raw_in);
                std::hint::black_box(d);
            }
        },
        400,
    );
    let acpu_var_per = acpu_var_ns as f64 / 32.0;

    let tw_var_ns = best_of(
        || {
            for _ in 0..32 {
                let d = Tip5::hash_varlen(&bfe_in);
                std::hint::black_box(d);
            }
        },
        400,
    );
    let tw_var_per = tw_var_ns as f64 / 32.0;
    score.row("hash_varlen[100]", acpu_var_per as u64, tw_var_per as u64);

    // ── derived rates ──
    println!();
    println!("derived rates");
    println!(
        "  acpu  permute single    : {:>7.1} ns  ({:.2} Mperm/s)",
        acpu_per,
        1.0e3 / acpu_per
    );
    println!(
        "  twenty-first single     : {:>7.1} ns  ({:.2} Mperm/s)",
        tw_per,
        1.0e3 / tw_per
    );
    println!(
        "  acpu  permute batch×8   : {:>7.1} ns/perm  ({:.2} Mperm/s)",
        acpu_batch8_per,
        1.0e3 / acpu_batch8_per
    );

    score.summary();
}
