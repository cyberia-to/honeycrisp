//! Poseidon2 permutation for Goldilocks t=16 — fully inlined, zero call overhead.
//!
//! hemera 0.4 parameters (profile 15): R_F = 12 (6 + 6), R_P = 48, S-box x⁷ in
//! every round, external layer M_E = circ(2·M4, M4, M4, M4) with the Poseidon2
//! reference M4 = [[5,7,1,3],[4,6,1,1],[1,3,5,7],[1,1,4,6]], internal layer
//! M_I = 1 + diag(d). State stays in the register file for the entire
//! permutation.
//!
//! Round constants and the internal diagonal are caller-supplied so the kernel
//! stays dependency-free; `tests/hemera_differential.rs` pins it bit for bit
//! to `cyber-hemera` with hemera's own constants (receipt:
//! hemera/audit/profile-0.4-2026-10.md).

use super::goldilocks::*;

/// Full rounds, half before and half after the partial rounds.
pub const ROUNDS_F: usize = 12;
/// Partial rounds.
pub const ROUNDS_P: usize = 48;
/// Rounds in total.
pub const ROUNDS_TOTAL: usize = ROUNDS_F + ROUNDS_P; // 60
/// External round constants: one per state element per full round.
pub const NUM_EXTERNAL: usize = ROUNDS_F * 16; // 192
/// Round constants in total: external rows first, then one per partial round.
pub const NUM_CONSTANTS: usize = NUM_EXTERNAL + ROUNDS_P; // 240

/// Apply the hemera 0.4 Poseidon2 permutation in-place.
///
/// `state`: 16 Goldilocks field elements, canonical (`< P`). The output is
/// canonical.
/// `rc`: 240 round constants — 192 external (round-major, initial full
/// rounds then terminal full rounds) followed by 48 internal.
/// `diag`: the 16-element diagonal `d` of the internal matrix `1 + diag(d)`.
#[inline(never)]
pub fn poseidon2_permute(state: &mut [u64; 16], rc: &[u64; NUM_CONSTANTS], diag: &[u64; 16]) {
    let mut ci = 0usize;

    // Initial external layer (pre-mixing)
    mds_external(state);

    // 6 initial full rounds
    for _ in 0..ROUNDS_F / 2 {
        add_rc_full(state, rc, ci);
        ci += 16;
        sbox7_full(state);
        mds_external(state);
    }

    // 48 partial rounds: add_rc(state[0]) + state[0]^7 + internal layer
    for r in 0..ROUNDS_P {
        state[0] = gl_add(state[0], rc[NUM_EXTERNAL + r]);
        state[0] = gl_pow7(state[0]);
        mds_internal(state, diag);
    }

    // 6 terminal full rounds
    for _ in 0..ROUNDS_F / 2 {
        add_rc_full(state, rc, ci);
        ci += 16;
        sbox7_full(state);
        mds_external(state);
    }

    for s in state.iter_mut() {
        *s = canonical(*s);
    }
}

/// Reduce a representative below 2^64 into `[0, P)`.
#[inline(always)]
fn canonical(x: u64) -> u64 {
    if x >= P {
        x - P
    } else {
        x
    }
}

// ── round components ─────────────────────────────────────────────────────

#[inline(always)]
fn add_rc_full(state: &mut [u64; 16], rc: &[u64; NUM_CONSTANTS], offset: usize) {
    for i in 0..16 {
        state[i] = gl_add(state[i], rc[offset + i]);
    }
}

/// Full-round S-box: x^7 on all 16 elements.
/// Written as explicit steps to help LLVM interleave across elements.
#[inline(always)]
fn sbox7_full(state: &mut [u64; 16]) {
    // Step 1: x² for all 16
    let mut x2 = [0u64; 16];
    for i in 0..16 {
        x2[i] = gl_mul(state[i], state[i]);
    }
    // Step 2: x³ for all 16
    let mut x3 = [0u64; 16];
    for i in 0..16 {
        x3[i] = gl_mul(x2[i], state[i]);
    }
    // Step 3: x⁴ for all 16
    let mut x4 = [0u64; 16];
    for i in 0..16 {
        x4[i] = gl_mul(x2[i], x2[i]);
    }
    // Step 4: x⁷ = x³ × x⁴
    for i in 0..16 {
        state[i] = gl_mul(x3[i], x4[i]);
    }
}

/// External layer M_E = circ(2·M4, M4, M4, M4). No field multiplications —
/// only adds and doubles.
///
/// Applied as M4 on each 4-element chunk, then the column sums added back.
#[inline(always)]
fn mds_external(state: &mut [u64; 16]) {
    // Apply M4 to each 4-element chunk
    apply_m4(&mut state[0..4]);
    apply_m4(&mut state[4..8]);
    apply_m4(&mut state[8..12]);
    apply_m4(&mut state[12..16]);

    // Column sums: s[k] = state[k] + state[k+4] + state[k+8] + state[k+12]
    for k in 0..4 {
        let s = gl_add(
            gl_add(state[k], state[k + 4]),
            gl_add(state[k + 8], state[k + 12]),
        );
        state[k] = gl_add(state[k], s);
        state[k + 4] = gl_add(state[k + 4], s);
        state[k + 8] = gl_add(state[k + 8], s);
        state[k + 12] = gl_add(state[k + 12], s);
    }
}

/// The Poseidon2 reference M4 (paper Appendix B, Horizen Labs):
/// ```text
/// [5 7 1 3]
/// [4 6 1 1]
/// [1 3 5 7]
/// [1 1 4 6]
/// ```
/// Eight additions and four doublings — the paper's schedule, the same one
/// hemera's `field::apply_mat4` uses.
#[inline(always)]
fn apply_m4(x: &mut [u64]) {
    let t0 = gl_add(x[0], x[1]);
    let t1 = gl_add(x[2], x[3]);
    let t2 = gl_add(gl_add(x[1], x[1]), t1);
    let t3 = gl_add(gl_add(x[3], x[3]), t0);
    let t4 = gl_add(gl_add(gl_add(t1, t1), gl_add(t1, t1)), t3);
    let t5 = gl_add(gl_add(gl_add(t0, t0), gl_add(t0, t0)), t2);
    let t6 = gl_add(t3, t5);
    let t7 = gl_add(t2, t4);
    x[0] = t6;
    x[1] = t5;
    x[2] = t7;
    x[3] = t4;
}

/// Internal layer: M_I = 1 + diag(d), with 1 the all-ones matrix.
/// `state'[i] = d[i] * state[i] + sum(state)`.
#[inline(always)]
fn mds_internal(state: &mut [u64; 16], diag: &[u64; 16]) {
    // Sum all elements
    let mut sum = state[0];
    for s in &state[1..] {
        sum = gl_add(sum, *s);
    }
    // diag[i] * state[i] + sum
    for i in 0..16 {
        state[i] = gl_add(gl_mul(diag[i], state[i]), sum);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_rc() -> [u64; NUM_CONSTANTS] {
        core::array::from_fn(|i| canonical((i as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15)))
    }

    fn test_diag() -> [u64; 16] {
        core::array::from_fn(|i| canonical((i as u64 + 2).wrapping_mul(0x517CC1B727220A95)))
    }

    #[test]
    fn round_shape_is_hemera_0_4() {
        assert_eq!(ROUNDS_F, 12);
        assert_eq!(ROUNDS_P, 48);
        assert_eq!(ROUNDS_TOTAL, 60);
        assert_eq!(NUM_CONSTANTS, 240);
    }

    #[test]
    fn permute_changes_state() {
        let mut state: [u64; 16] = core::array::from_fn(|i| i as u64 + 1);
        let original = state;
        poseidon2_permute(&mut state, &test_rc(), &test_diag());
        assert_ne!(state, original);
    }

    #[test]
    fn permute_deterministic() {
        let rc = test_rc();
        let diag = test_diag();
        let mut s1: [u64; 16] = core::array::from_fn(|i| i as u64 + 1);
        let mut s2 = s1;
        poseidon2_permute(&mut s1, &rc, &diag);
        poseidon2_permute(&mut s2, &rc, &diag);
        assert_eq!(s1, s2);
    }

    #[test]
    fn permute_output_is_canonical() {
        let rc = test_rc();
        let diag = test_diag();
        let mut s: [u64; 16] = [P - 1; 16];
        poseidon2_permute(&mut s, &rc, &diag);
        assert!(s.iter().all(|&x| x < P));
    }

    #[test]
    fn permute_diffusion() {
        let rc = test_rc();
        let diag = test_diag();
        let mut s1: [u64; 16] = core::array::from_fn(|i| i as u64 + 100);
        let mut s2 = s1;
        s2[0] += 1;
        poseidon2_permute(&mut s1, &rc, &diag);
        poseidon2_permute(&mut s2, &rc, &diag);
        for i in 0..16 {
            assert_ne!(s1[i], s2[i], "element {} unchanged after input tweak", i);
        }
    }
}
