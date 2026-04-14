//! Poseidon2 permutation for Goldilocks t=16 — fully inlined, zero call overhead.
//!
//! Hemera parameters: R_F=8 (4+4), R_P=16, full S-box=x^7, partial S-box=x^(-1).
//! State stays in the register file for the entire permutation.

use super::goldilocks::*;

/// Apply the Hemera Poseidon2 permutation in-place.
///
/// `state`: 16 Goldilocks field elements.
/// `rc`: 144 round constants (128 external + 16 internal).
/// `diag`: 16 diagonal elements for internal MDS matrix.
#[inline(never)]
pub fn poseidon2_permute(state: &mut [u64; 16], rc: &[u64; 144], diag: &[u64; 16]) {
    let mut ci = 0usize;

    // Initial external MDS (pre-mixing)
    mds_external(state);

    // 4 initial full rounds
    for _ in 0..4 {
        add_rc_full(state, rc, ci);
        ci += 16;
        sbox7_full(state);
        mds_external(state);
    }

    // 16 partial rounds: add_rc(state[0]) + inv(state[0]) + internal_mds
    for r in 0..16 {
        state[0] = gl_add(state[0], rc[128 + r]);
        state[0] = gl_inv(state[0]);
        mds_internal(state, diag);
    }

    // 4 terminal full rounds
    for _ in 0..4 {
        add_rc_full(state, rc, ci);
        ci += 16;
        sbox7_full(state);
        mds_external(state);
    }
}

// ── round components ─────────────────────────────────────────────────────

#[inline(always)]
fn add_rc_full(state: &mut [u64; 16], rc: &[u64; 144], offset: usize) {
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

/// External MDS: circ-of-4×4. No field multiplications — only adds and doubles.
///
/// M4 = circ(2,3,1,1):
/// ```text
/// [2 3 1 1]
/// [1 2 3 1]
/// [1 1 2 3]
/// [3 1 1 2]
/// ```
/// Applied to 4 chunks of 4, then column sums added.
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

/// M4 = circ(2,3,1,1) — pure additions and doubles.
#[inline(always)]
fn apply_m4(x: &mut [u64]) {
    let t01 = gl_add(x[0], x[1]);
    let t23 = gl_add(x[2], x[3]);
    let t0123 = gl_add(t01, t23);
    let t01123 = gl_add(t0123, x[1]);
    let t01233 = gl_add(t0123, x[3]);
    let x0d = gl_add(x[0], x[0]); // double
    let x2d = gl_add(x[2], x[2]);
    x[3] = gl_add(t01233, x0d);
    x[1] = gl_add(t01123, x2d);
    x[0] = gl_add(t01123, t01);
    x[2] = gl_add(t01233, t23);
}

/// Internal MDS: M_I = 1 + diag(d).
/// `state'[i] = d[i] * state[i] + sum(state)`.
#[inline(always)]
fn mds_internal(state: &mut [u64; 16], diag: &[u64; 16]) {
    // Sum all elements
    let mut sum = state[0];
    for i in 1..16 {
        sum = gl_add(sum, state[i]);
    }
    // diag[i] * state[i] + sum
    for i in 0..16 {
        state[i] = gl_add(gl_mul(diag[i], state[i]), sum);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_rc() -> [u64; 144] {
        let mut rc = [0u64; 144];
        for i in 0..144 {
            rc[i] = (i as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15);
        }
        rc
    }

    fn test_diag() -> [u64; 16] {
        let mut d = [0u64; 16];
        for i in 0..16 {
            d[i] = (i as u64 + 2).wrapping_mul(0x517CC1B727220A95);
        }
        d
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
