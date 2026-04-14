//! ZK proving benchmark: Goldilocks field, Poseidon2, NTT, Merkle tree.
//! Complete benchmark for the hemera/nebu/acpu ZK stack.

#[path = "common.rs"]
mod common;
use common::*;

const N: usize = 4096;

fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(120));
        eprintln!("!!! 120s TIMEOUT !!!");
        std::process::exit(1);
    });

    let caps = acpu::probe::scan();
    println!("acpu ZK proving benchmark — {:?}", caps.chip);
    println!("Goldilocks p = 2^64 - 2^32 + 1");
    println!();

    let mut score = Score::new();
    let a: Vec<u64> = (0..N)
        .map(|i| (i as u64).wrapping_mul(0x9E3779B97F4A7C15) | 1)
        .collect();
    let b: Vec<u64> = (0..N)
        .map(|i| (i as u64).wrapping_mul(0x6C62272E07BB0142) | 1)
        .collect();
    let mut dst = vec![0u64; N];

    // ── 1. FIELD PRIMITIVES ──────────────────────────────────────────────

    score.hdr("FIELD PRIMITIVES (Goldilocks)");

    // mul chain (serial latency)
    let mul_ns = ns(|| {
        let mut acc = a[0];
        for &v in &b[..64] {
            acc = acpu::field::gl_mul(acc, v);
        }
        std::hint::black_box(acc);
    });
    println!(
        "  field_mul serial:       {:>5.1}ns/op",
        mul_ns as f64 / 64.0
    );

    // mul batch (throughput)
    let batch_ns = best_of(
        || {
            acpu::field::gl_mul_batch(&a, &b, &mut dst);
            std::hint::black_box(&dst);
        },
        200,
    );
    println!(
        "  field_mul batch:        {:>5.1}ns/op  ({:.0} Mmul/s)",
        batch_ns as f64 / N as f64,
        N as f64 / batch_ns as f64 * 1000.0
    );

    // add batch
    let add_ns = best_of(
        || {
            for i in 0..N {
                dst[i] = acpu::field::gl_add(a[i], b[i]);
            }
            std::hint::black_box(&dst);
        },
        200,
    );
    println!(
        "  field_add batch:        {:>5.1}ns/op",
        add_ns as f64 / N as f64
    );

    // sub batch
    let sub_ns = best_of(
        || {
            for i in 0..N {
                dst[i] = acpu::field::gl_sub(a[i], b[i]);
            }
            std::hint::black_box(&dst);
        },
        200,
    );
    println!(
        "  field_sub batch:        {:>5.1}ns/op",
        sub_ns as f64 / N as f64
    );

    // ── 2. S-BOX x^7 (full rounds) ──────────────────────────────────────

    score.hdr("S-BOX x^7 (full round, 16 elements)");

    let mut state16: [u64; 16] = core::array::from_fn(|i| a[i]);
    let sbox_ns = best_of(
        || {
            acpu::field::gl_pow7_x16(&mut state16);
            std::hint::black_box(&state16);
        },
        500,
    );
    println!(
        "  pow7_x16:              {:>5}ns  ({:.1}ns/element)",
        sbox_ns,
        sbox_ns as f64 / 16.0
    );

    // ── 3. FIELD INVERSION x^(-1) (partial S-box) ────────────────────────

    score.hdr("FIELD INVERSION x^(-1) (partial S-box)");

    let inv_ns = ns(|| {
        let mut x = a[0];
        for _ in 0..16 {
            x = acpu::field::gl_inv(x);
        }
        std::hint::black_box(x);
    });
    let inv_per = inv_ns as f64 / 16.0;
    println!(
        "  inv serial (chain 16): {:>5.0}ns/inv  (75 muls × {:.1}ns/mul)",
        inv_per,
        inv_per / 75.0
    );

    // ── 4. BATCH INVERSION (Montgomery trick) ────────────────────────────

    score.hdr("BATCH INVERSION (Montgomery trick)");

    let mut inv_out = vec![0u64; N];
    let batch_inv_ns = best_of(
        || {
            acpu::field::batch_inv(&a, &mut inv_out);
            std::hint::black_box(&inv_out);
        },
        50,
    );
    let per_inv = batch_inv_ns as f64 / N as f64;
    let vs_single = inv_per / per_inv;
    println!(
        "  batch_inv ({N}):       {:>5.1}ns/element  ({:.0}× vs single inv)",
        per_inv, vs_single
    );

    // ── 5. POSEIDON2 PERMUTATION ─────────────────────────────────────────

    score.hdr("POSEIDON2 PERMUTATION (t=16, hemera)");

    let rc: [u64; 144] = core::array::from_fn(|i| (i as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15));
    let diag: [u64; 16] = core::array::from_fn(|i| (i as u64 + 2).wrapping_mul(0x517CC1B727220A95));
    let mut state: [u64; 16] = core::array::from_fn(|i| i as u64 + 1);

    let perm_ns = best_of(
        || {
            state = core::array::from_fn(|i| i as u64 + 1);
            acpu::field::poseidon2_permute(&mut state, &rc, &diag);
            std::hint::black_box(&state);
        },
        200,
    );
    let hps = 1_000_000_000.0 / perm_ns as f64;

    // Theoretical: 16 inv × 75 muls × 3.2ns + 512 sbox7 muls × 1.1ns + ~500 adds
    // ≈ 3840 + 563 + 350 ≈ 4753ns theoretical floor
    let theory_ns = 4753.0;
    let efficiency = theory_ns / perm_ns as f64 * 100.0;
    println!(
        "  permute:               {:>5}ns  ({:.0}K hash/s, {:.0}% of {:.0}ns theoretical)",
        perm_ns,
        hps / 1000.0,
        efficiency,
        theory_ns
    );

    let mb_s = 56.0 / perm_ns as f64 * 1000.0;
    println!(
        "  hash throughput:       {:>5.1} MB/s  (56 bytes/perm)",
        mb_s
    );

    // Reference comparison: plonky3 standard Poseidon2 (t=12, x^7 partial S-box)
    // does ~50-100ns per permutation on x86 AVX2 — but uses x^7 (4 muls)
    // vs hemera x^(-1) (75 muls). Expected ratio: ~75/4 ≈ 19×.
    // Our {}ns for 75-mul inv is competitive given the algorithmic cost.
    println!("  vs plonky3 Poseidon2:  hemera uses x^(-1) partial S-box (75 muls)");
    println!("                         vs x^7 (4 muls) — different security tradeoff");

    // ── 6. MERKLE TREE ───────────────────────────────────────────────────

    score.hdr("MERKLE TREE (binary, Poseidon2)");

    for &n_leaves in &[64, 256, 1024, 4096] {
        let leaves: Vec<[u64; 4]> = (0..n_leaves)
            .map(|i| {
                [
                    i as u64 * 4 + 1,
                    i as u64 * 4 + 2,
                    i as u64 * 4 + 3,
                    i as u64 * 4 + 4,
                ]
            })
            .collect();

        let tree_ns = best_of(
            || {
                std::hint::black_box(acpu::field::merkle_root(&leaves, &rc, &diag));
            },
            if n_leaves <= 256 { 50 } else { 10 },
        );

        let n_hashes = n_leaves - 1; // binary tree has n-1 internal nodes
        let per_hash = tree_ns as f64 / n_hashes as f64;
        println!(
            "  {:<5} leaves ({:>4} hashes): {:>8}ns  ({:.0}ns/hash)",
            n_leaves, n_hashes, tree_ns, per_hash
        );
    }

    // ── 7. NTT (via nebu) ────────────────────────────────────────────────

    score.hdr("NTT (Goldilocks, nebu)");

    for &ntt_size in &[256, 1024, 4096, 16384, 65536] {
        let mut data: Vec<nebu::Goldilocks> = (0..ntt_size)
            .map(|i| nebu::Goldilocks::new((i as u64).wrapping_mul(0x9E3779B97F4A7C15)))
            .collect();

        // warmup
        nebu::ntt::ntt(&mut data);
        nebu::ntt::intt(&mut data);

        let iters = if ntt_size <= 1024 {
            50
        } else if ntt_size <= 16384 {
            20
        } else {
            5
        };

        let fwd_ns = best_of(
            || {
                nebu::ntt::ntt(&mut data);
                std::hint::black_box(&data);
            },
            iters,
        );

        let inv_ntt_ns = best_of(
            || {
                nebu::ntt::intt(&mut data);
                std::hint::black_box(&data);
            },
            iters,
        );

        let butterflies = (ntt_size / 2) * ((ntt_size as u32).trailing_zeros() as usize);
        let bf_rate = butterflies as f64 / fwd_ns as f64 * 1000.0;
        println!(
            "  NTT  2^{:<2} ({:>5}): {:>8}ns fwd  {:>8}ns inv  ({:.0}M butterfly/s)",
            (ntt_size as u32).trailing_zeros(),
            ntt_size,
            fwd_ns,
            inv_ntt_ns,
            bf_rate
        );
    }

    // ── 8. RECURSIVE VERIFIER ESTIMATE ───────────────────────────────────

    score.hdr("RECURSIVE PROOF ESTIMATES");

    println!("  permutation:            {:>5}ns", perm_ns);
    println!(
        "  1K hashes (verifier):   {:>5.1}ms",
        perm_ns as f64 * 1000.0 / 1e6
    );
    println!(
        "  10K hashes (prover):    {:>5.1}ms",
        perm_ns as f64 * 10000.0 / 1e6
    );

    // Merkle path: log2(n) hashes per verification
    for &tree_size in &[1024u64, 1_000_000, 1_000_000_000] {
        let depth = 64 - tree_size.leading_zeros();
        let path_ns = perm_ns as u64 * depth as u64;
        println!(
            "  Merkle path (n={:>10}): {:>5}ns  ({} hashes, depth {})",
            tree_size, path_ns, depth, depth
        );
    }

    println!();
    println!("=== end ===");
}
