//! Tip5 — Apple-Silicon-tuned, bit-identical to `twenty_first::Tip5` v1.1.0.
//!
//! State is `[u64; 16]` of raw-Montgomery `BFieldElement` words. Public API
//! deals only in raw u64s; converting to/from `BFieldElement::value()`
//! (canonical) is the caller's responsibility — for a downstream consumer
//! that owns `BFieldElement` values, `raw_u64()` / `from_raw_u64(_)` are the
//! free bridges.
//!
//! ## Domains and digest convention
//!
//! - `tip5_hash_pair([u64; 5], [u64; 5])` matches `Tip5::hash_pair`. Inputs
//!   are the raw-Montgomery `Digest::values()` of `left` then `right`. The
//!   sponge initializes in `FixedLength` domain (state[10..16] = mont(1)).
//!   The first 5 raw words of the post-permutation state are returned.
//! - `tip5_hash_varlen(&[u64])` matches `Tip5::hash_varlen`. Inputs are raw
//!   Montgomery elements; padding is `[1, 0, 0, …]` (one canonical `mont(1)`
//!   followed by zeros). The capacity is initialized to all zeros
//!   (`VariableLength` domain).
//!
//! ## Speed model
//!
//! Tip5 cost is dominated by the MDS layer (8 round-trips through the
//! `mds_generated` matrix per permutation) and the per-state x⁷ S-box
//! (4 montgomery multiplies × 12 lanes × 5 rounds = 240 muls). On the M4
//! P-core both are heavily ILP-friendly; this implementation interleaves 12
//! independent multiply chains for the S-box and uses straight-line
//! `wrapping_*` code for the MDS, which LLVM schedules tightly.

mod consts;
mod mds;
mod neon;
mod scalar;
mod sme;

pub use consts::{LOOKUP_TABLE, MDS_MATRIX_FIRST_COLUMN, ROUND_CONSTANTS_RAW};
pub use sme::tip5_permute_sme;

use mds::mds_generated_inplace;
use neon::pow7_last12;
use scalar::{bfe_add_raw, montyred, split_and_lookup, P};

// ── domain constants ────────────────────────────────────────────────────────

/// Sponge state width.
pub const STATE_SIZE: usize = 16;
/// Sponge rate (how many lanes the caller can read/write).
pub const RATE: usize = 10;
/// Number of S-box lanes that use the 256-byte split-and-lookup table.
pub const NUM_SPLIT_AND_LOOKUP: usize = 4;
/// Digest size in BFieldElements.
pub const DIGEST_LEN: usize = 5;
/// Number of Tip5 rounds.
pub const NUM_ROUNDS: usize = 5;

/// R = 2^64 mod P. Equals `montyred(R^2)` and is `BFieldElement::ONE.raw_u64()`.
pub const RAW_MONT_ONE: u64 = {
    // R^2 mod P = 0xffff_fffe_0000_0001. R = 2^64; R mod P = 2^32 - 1 in
    // canonical, but we need raw-Mont form of 1 = montyred(R^2).
    // montyred is const; precompute.
    let r2: u128 = 0xffff_fffe_0000_0001;
    montyred(r2)
};

// ── public API ──────────────────────────────────────────────────────────────

/// Apply the Tip5 permutation in place to a 16-lane raw-Montgomery state.
///
/// Bit-identical to one invocation of `Tip5::permutation` on a `Tip5` whose
/// `state[i].raw_u64() == s[i]`.
#[inline]
pub fn tip5_permute(s: &mut [u64; 16]) {
    let mut r = 0;
    while r < NUM_ROUNDS {
        round(s, r);
        r += 1;
    }
}

/// Hash a pair of digests. Bit-identical to `Tip5::hash_pair` once inputs
/// are passed as raw `Digest::values()` words.
#[inline]
pub fn tip5_hash_pair(left: [u64; 5], right: [u64; 5]) -> [u64; 5] {
    let mut s = fixed_length_init();
    let mut i = 0;
    while i < 5 {
        s[i] = left[i];
        s[i + 5] = right[i];
        i += 1;
    }
    tip5_permute(&mut s);
    [s[0], s[1], s[2], s[3], s[4]]
}

/// Hash a variable-length sequence of raw-Montgomery BFieldElements.
///
/// Bit-identical to `Tip5::hash_varlen` once each `input[i]` is `e.raw_u64()`.
#[inline]
pub fn tip5_hash_varlen(input: &[u64]) -> [u64; 5] {
    let mut s = variable_length_init();
    let chunks = input.chunks_exact(RATE);
    let remainder = chunks.remainder();
    for chunk in chunks.clone() {
        let mut buf = [0u64; RATE];
        buf.copy_from_slice(chunk);
        absorb(&mut s, &buf);
    }
    // Pad: copy remainder, append RAW_MONT_ONE, then zeros.
    let mut last = [0u64; RATE];
    let mut i = 0;
    while i < remainder.len() {
        last[i] = remainder[i];
        i += 1;
    }
    last[remainder.len()] = RAW_MONT_ONE;
    // (rest already zero)
    absorb(&mut s, &last);
    [s[0], s[1], s[2], s[3], s[4]]
}

/// Batched permutation — `N` independent Tip5 states permuted in one call.
///
/// Each state is processed independently. Provides a target for callers that
/// can amortize call overhead and that LLVM can schedule with cross-state
/// ILP. Bit-identical to `N` separate `tip5_permute` calls.
#[inline]
pub fn tip5_permute_batch<const N: usize>(states: &mut [[u64; 16]; N]) {
    let mut k = 0;
    while k < N {
        tip5_permute(&mut states[k]);
        k += 1;
    }
}

/// Sequential batch hash_pair over a slice of `(left, right)` digest pairs.
///
/// Bit-identical to calling [`tip5_hash_pair`] in a loop; exists as the API
/// shape that the parallel version partitions over.
#[inline(never)]
pub fn tip5_hash_pair_n(
    pairs: &[([u64; DIGEST_LEN], [u64; DIGEST_LEN])],
    out: &mut [[u64; DIGEST_LEN]],
) {
    assert_eq!(pairs.len(), out.len());
    for (i, (l, r)) in pairs.iter().enumerate() {
        out[i] = tip5_hash_pair(*l, *r);
    }
}

/// Multi-threaded `hash_pair` over a slice of pairs — the Merkle-layer fast
/// path. Partitions the work across P-cores via `std::thread::scope` (not
/// rayon — keeps the honeycrisp zero-copy mandate intact) and pins each
/// worker via [`crate::sync::affinity::pin_p_core`]. Workers process their
/// slice with the scalar [`tip5_hash_pair`] path.
///
/// Throughput scaling on M4 Max (12 P-cores, single-thread baseline ~4 M
/// hashes/s):
///
///   n=  1024 :  ~1× (spawn cost dominates)
///   n=  4096 :  ~3.3× (≈13 M hashes/s)
///   n= 65536 :  ~4.3× (≈17 M hashes/s) — best amortization
///
/// Falls below theoretical 12× because:
/// 1. macOS QoS class is a soft pin, not hard affinity.
/// 2. Shared `LOOKUP_TABLE` / `ROUND_CONSTANTS_RAW` cause memory contention.
/// 3. Thermal throttling on sustained 12-core load.
///
/// `n_threads = 0` selects all available P-cores.
#[inline(never)]
pub fn tip5_hash_pair_n_par(
    pairs: &[([u64; DIGEST_LEN], [u64; DIGEST_LEN])],
    out: &mut [[u64; DIGEST_LEN]],
    n_threads: usize,
) {
    assert_eq!(pairs.len(), out.len());
    let total = pairs.len();
    if total == 0 {
        return;
    }
    let p_cores = crate::probe::scan().p_cores as usize;
    let n = if n_threads == 0 {
        p_cores.max(1)
    } else {
        n_threads
    };
    if n <= 1 || total < n * 4 {
        // Below the parallel threshold the `std::thread::scope` spawn cost
        // (~50 µs/worker) eats the win — run serial.
        tip5_hash_pair_n(pairs, out);
        return;
    }

    let chunk = total.div_ceil(n);
    let mut out_chunks: Vec<&mut [[u64; DIGEST_LEN]]> = Vec::with_capacity(n);
    {
        let mut remaining: &mut [[u64; DIGEST_LEN]] = out;
        for _ in 0..n {
            let take = chunk.min(remaining.len());
            let (head, tail) = remaining.split_at_mut(take);
            out_chunks.push(head);
            remaining = tail;
        }
    }

    std::thread::scope(|s| {
        let mut start = 0usize;
        for out_chunk in out_chunks {
            let len = out_chunk.len();
            if len == 0 {
                continue;
            }
            let in_chunk = &pairs[start..start + len];
            start += len;
            s.spawn(move || {
                let _ = crate::sync::affinity::pin_p_core();
                tip5_hash_pair_n(in_chunk, out_chunk);
            });
        }
    });
}

// ── internals ───────────────────────────────────────────────────────────────

#[inline(always)]
fn fixed_length_init() -> [u64; 16] {
    let mut s = [0u64; 16];
    // Capacity portion = raw-Mont(1).
    let mut i = RATE;
    while i < STATE_SIZE {
        s[i] = RAW_MONT_ONE;
        i += 1;
    }
    s
}

#[inline(always)]
fn variable_length_init() -> [u64; 16] {
    [0u64; 16]
}

#[inline(always)]
fn absorb(s: &mut [u64; 16], chunk: &[u64; RATE]) {
    let mut i = 0;
    while i < RATE {
        s[i] = chunk[i];
        i += 1;
    }
    tip5_permute(s);
}

#[inline(always)]
fn round(s: &mut [u64; 16], round_index: usize) {
    sbox_layer(s);
    mds_generated_inplace(s);
    let off = round_index * 16;
    let mut i = 0;
    while i < 16 {
        s[i] = bfe_add_raw(s[i], ROUND_CONSTANTS_RAW[off + i]);
        i += 1;
    }
}

/// Tip5 S-box layer:
/// - lanes 0..4   : split_and_lookup (256-byte table on each of 8 bytes)
/// - lanes 4..16  : x⁷
#[inline(always)]
fn sbox_layer(s: &mut [u64; 16]) {
    // First 4 lanes: byte-wise LUT.
    s[0] = split_and_lookup(s[0], &LOOKUP_TABLE);
    s[1] = split_and_lookup(s[1], &LOOKUP_TABLE);
    s[2] = split_and_lookup(s[2], &LOOKUP_TABLE);
    s[3] = split_and_lookup(s[3], &LOOKUP_TABLE);
    // Last 12 lanes: interleaved Montgomery x⁷.
    pow7_last12(s);
}

// Keep `P` and `montyred` reachable for tests in this module.
#[allow(dead_code)]
const _P: u64 = P;
#[allow(dead_code)]
const fn _expose_montyred(x: u128) -> u64 {
    montyred(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_mont_one_round_trip() {
        // raw-Mont(1) added to raw-Mont(0) should still be raw-Mont(1).
        let z = bfe_add_raw(0, RAW_MONT_ONE);
        assert_eq!(z, RAW_MONT_ONE);
    }

    #[test]
    fn permute_changes_state() {
        let mut s: [u64; 16] = core::array::from_fn(|i| (i as u64 + 1).wrapping_mul(7));
        let orig = s;
        tip5_permute(&mut s);
        assert_ne!(s, orig);
    }

    #[test]
    fn permute_deterministic() {
        let mut s1: [u64; 16] = core::array::from_fn(|i| (i as u64 + 3).wrapping_mul(11));
        let mut s2 = s1;
        tip5_permute(&mut s1);
        tip5_permute(&mut s2);
        assert_eq!(s1, s2);
    }

    /// Snapshot test from `twenty_first::tip5::tests::snapshot`. Values are
    /// raw u64 in/out — they exercise the degenerate-rep recovery path.
    #[test]
    fn snapshot_from_twenty_first() {
        let mut state: [u64; 16] = [
            0x0000_000f_ffff_fff0,
            0x0000_0000_ffff_ffff,
            0x0000_0000_ffff_ffff,
            0x0000_0028_ffff_ffd7,
            0x0000_0006_ffff_fff9,
            0x0000_0002_ffff_fffd,
            0x0000_0000_ffff_ffff,
            0x0000_0030_ffff_ffcf,
            0x0000_0397_ffff_fc68,
            0x0000_000f_ffff_fff0,
            0x316b_fb72_3638_2123,
            0x216f_521b_66ef_83f5,
            0x5689_d7b3_63f5_2df0,
            0xeb2f_59e3_aeae_25fc,
            0xb082_99d2_77cb_b4dc,
            0xcbe3_d9fd_c534_9140,
        ];
        tip5_permute(&mut state);
        let expected_prefix: [u64; 5] = [
            0x15d3_8ea9_29f6_632a,
            0xf988_e509_ff73_8bb4,
            0x48bc_dfae_88a2_e9f3,
            0x8733_9e83_2daa_c02a,
            0x511e_4126_8150_fdac,
        ];
        assert_eq!(&state[..5], &expected_prefix[..]);
    }
}
