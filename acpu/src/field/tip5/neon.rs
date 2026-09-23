//! NEON-friendly x⁷ batch S-box for state[4..16].
//!
//! Apple M4 P-core has a 4-cycle multiply latency and 4-wide issue, so the
//! best schedule is: 12 independent chains, 4 steps each (sq, qu, sq·qu, x·).
//! All four steps for all 12 lanes run as straight-line code; the compiler
//! issues them with the OOO window absorbing the latency.
//!
//! This is the same pattern as `gl_mul_batch` but with Montgomery reduction
//! (`montyred`) instead of Goldilocks reduction, since Tip5 state values are
//! raw-Montgomery rather than canonical Goldilocks.

use super::scalar::montyred;

/// Apply x⁷ in place to state[4..16] (the 12 non-LUT lanes).
///
/// Four serial multiply layers, twelve interleaved chains. Each layer is a
/// 12-way independent set of `mul` / `umulh` pairs, fed to `montyred`.
#[inline(always)]
pub fn pow7_last12(state: &mut [u64; 16]) {
    // Layer 1: sq[i] = x[i]² (12 chains).
    let mut sq = [0u128; 12];
    let mut i = 0;
    while i < 12 {
        let x = state[i + 4] as u128;
        sq[i] = x * x;
        i += 1;
    }
    let mut sq_r = [0u64; 12];
    let mut i = 0;
    while i < 12 {
        sq_r[i] = montyred(sq[i]);
        i += 1;
    }

    // Layer 2: qu[i] = sq[i]² .
    let mut qu = [0u128; 12];
    let mut i = 0;
    while i < 12 {
        let s = sq_r[i] as u128;
        qu[i] = s * s;
        i += 1;
    }
    let mut qu_r = [0u64; 12];
    let mut i = 0;
    while i < 12 {
        qu_r[i] = montyred(qu[i]);
        i += 1;
    }

    // Layer 3: sq_qu[i] = sq[i] * qu[i] = x^6.
    let mut sq_qu = [0u128; 12];
    let mut i = 0;
    while i < 12 {
        sq_qu[i] = (sq_r[i] as u128) * (qu_r[i] as u128);
        i += 1;
    }
    let mut sq_qu_r = [0u64; 12];
    let mut i = 0;
    while i < 12 {
        sq_qu_r[i] = montyred(sq_qu[i]);
        i += 1;
    }

    // Layer 4: x⁷ = x * x⁶.
    let mut out = [0u128; 12];
    let mut i = 0;
    while i < 12 {
        out[i] = (state[i + 4] as u128) * (sq_qu_r[i] as u128);
        i += 1;
    }
    let mut i = 0;
    while i < 12 {
        state[i + 4] = montyred(out[i]);
        i += 1;
    }
}
