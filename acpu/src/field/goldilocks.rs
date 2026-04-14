//! Goldilocks field (p = 2^64 - 2^32 + 1) — asm-optimized for aarch64.
//!
//! Single-element ops + batched interleaved variants.
//! The batch functions hide mul latency by interleaving independent chains.

/// The Goldilocks prime.
pub const P: u64 = 0xFFFF_FFFF_0000_0001;

/// ε = 2^32 - 1 = P.wrapping_neg(). The reduction constant.
pub const EPSILON: u64 = P.wrapping_neg();

// ── single-element operations ────────────────────────────────────────────

/// Field addition: (a + b) mod p.
#[inline]
pub fn gl_add(a: u64, b: u64) -> u64 {
    let (sum, c1) = a.overflowing_add(b);
    let (sum, c2) = sum.overflowing_add(u64::from(c1) * EPSILON);
    if c2 {
        sum + EPSILON
    } else {
        sum
    }
}

/// Field subtraction: (a - b) mod p.
#[inline]
pub fn gl_sub(a: u64, b: u64) -> u64 {
    let (diff, b1) = a.overflowing_sub(b);
    let (diff, b2) = diff.overflowing_sub(u64::from(b1) * EPSILON);
    if b2 {
        diff - EPSILON
    } else {
        diff
    }
}

/// Reduce a 128-bit product to Goldilocks.
/// Uses 2^64 ≡ ε (mod p).
#[inline]
pub fn gl_reduce128(lo: u64, hi: u64) -> u64 {
    let hi_hi = hi >> 32;
    let hi_lo = hi & EPSILON;
    let (mut t0, borrow) = lo.overflowing_sub(hi_hi);
    if borrow {
        t0 -= EPSILON;
    }
    let t1 = hi_lo * EPSILON;
    let (res, carry) = t0.overflowing_add(t1);
    res + EPSILON * u64::from(carry)
}

/// Field multiplication: (a * b) mod p.
#[inline]
pub fn gl_mul(a: u64, b: u64) -> u64 {
    #[cfg(target_arch = "aarch64")]
    {
        let lo: u64;
        let hi: u64;
        unsafe {
            core::arch::asm!(
                "mul {lo}, {a}, {b}",
                "umulh {hi}, {a}, {b}",
                a = in(reg) a,
                b = in(reg) b,
                lo = out(reg) lo,
                hi = out(reg) hi,
            );
        }
        gl_reduce128(lo, hi)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let wide = (a as u128) * (b as u128);
        gl_reduce128(wide as u64, (wide >> 64) as u64)
    }
}

/// x^7 (Poseidon2 full-round S-box). 4 multiplications.
#[inline]
pub fn gl_pow7(x: u64) -> u64 {
    let x2 = gl_mul(x, x);
    let x3 = gl_mul(x2, x);
    let x4 = gl_mul(x2, x2);
    gl_mul(x3, x4)
}

/// Multiplicative inverse: x^(p-2) mod p.
///
/// Optimized addition chain: 75 multiplications (vs 125 naive).
///
/// Decomposition: p-2 = (2^31-1)·2^33 + (2^32-1).
/// Build x^(2^31-1) in 39 muls, then both terms from it.
#[inline(never)]
pub fn gl_inv(x: u64) -> u64 {
    if x == 0 || x == P {
        return 0;
    }

    // Phase 1: build x^(2^31-1) using chain of 2^k-1 powers (39 muls)
    let x2 = gl_mul(x, x); // x^2
    let x3 = gl_mul(x2, x); // x^3
    let x4 = gl_mul(x2, x2); // x^4
    let x7 = gl_mul(x3, x4); // x^7

    // x^15 = x^(2^4-1)
    let x6 = gl_mul(x3, x3);
    let x12 = gl_mul(x6, x6);
    let x15 = gl_mul(x12, x3);

    // x^127 = x^(2^7-1)
    let x30 = gl_mul(x15, x15);
    let x60 = gl_mul(x30, x30);
    let x120 = gl_mul(x60, x60);
    let x127 = gl_mul(x120, x7);

    // x^255 = x^(2^8-1)
    let x254 = gl_mul(x127, x127);
    let x255 = gl_mul(x254, x);

    // x^(2^15-1): square x^255 seven times, mul by x^127
    let mut t = x255;
    for _ in 0..7 {
        t = gl_mul(t, t);
    }
    let x_2p15m1 = gl_mul(t, x127); // x^32767

    // x^(2^16-1)
    let x_2p16m2 = gl_mul(x_2p15m1, x_2p15m1);
    let x_2p16m1 = gl_mul(x_2p16m2, x); // x^65535

    // x^(2^31-1): square x^(2^16-1) fifteen times, mul by x^(2^15-1)
    t = x_2p16m1;
    for _ in 0..15 {
        t = gl_mul(t, t);
    }
    let x_2p31m1 = gl_mul(t, x_2p15m1); // x^(2^31-1)

    // Phase 2: x^(2^32-1) from x^(2^31-1)  (2 muls)
    let x_2p32m2 = gl_mul(x_2p31m1, x_2p31m1); // x^(2^32-2)
    let x_epsilon = gl_mul(x_2p32m2, x); // x^(2^32-1) = x^ε

    // Phase 3: x^((2^31-1)·2^33) from x^(2^31-1)  (33 squarings)
    t = x_2p31m1;
    for _ in 0..33 {
        t = gl_mul(t, t);
    }
    // t = x^((2^31-1)·2^33)

    // Phase 4: x^(p-2) = x^((2^31-1)·2^33 + 2^32-1)
    gl_mul(t, x_epsilon)
}

// ── batched operations (interleaved for latency hiding) ──────────────────

/// Batch field multiply: dst[i] = a[i] * b[i] mod p.
///
/// Hand-scheduled asm interleaves 4 independent mul+umulh chains
/// to hide the 4-cycle multiply latency on M1 P-cores.
pub fn gl_mul_batch(a: &[u64], b: &[u64], dst: &mut [u64]) {
    let len = a.len().min(b.len()).min(dst.len());
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        while i + 4 <= len {
            unsafe {
                let lo0: u64;
                let lo1: u64;
                let lo2: u64;
                let lo3: u64;
                let hi0: u64;
                let hi1: u64;
                let hi2: u64;
                let hi3: u64;
                core::arch::asm!(
                    // 4 independent mul chains — latency hidden by OOO
                    "mul {lo0}, {a0}, {b0}",
                    "mul {lo1}, {a1}, {b1}",
                    "mul {lo2}, {a2}, {b2}",
                    "mul {lo3}, {a3}, {b3}",
                    "umulh {hi0}, {a0}, {b0}",
                    "umulh {hi1}, {a1}, {b1}",
                    "umulh {hi2}, {a2}, {b2}",
                    "umulh {hi3}, {a3}, {b3}",
                    a0 = in(reg) a[i],
                    b0 = in(reg) b[i],
                    a1 = in(reg) a[i + 1],
                    b1 = in(reg) b[i + 1],
                    a2 = in(reg) a[i + 2],
                    b2 = in(reg) b[i + 2],
                    a3 = in(reg) a[i + 3],
                    b3 = in(reg) b[i + 3],
                    lo0 = out(reg) lo0,
                    lo1 = out(reg) lo1,
                    lo2 = out(reg) lo2,
                    lo3 = out(reg) lo3,
                    hi0 = out(reg) hi0,
                    hi1 = out(reg) hi1,
                    hi2 = out(reg) hi2,
                    hi3 = out(reg) hi3,
                );
                dst[i] = gl_reduce128(lo0, hi0);
                dst[i + 1] = gl_reduce128(lo1, hi1);
                dst[i + 2] = gl_reduce128(lo2, hi2);
                dst[i + 3] = gl_reduce128(lo3, hi3);
            }
            i += 4;
        }
    }

    while i < len {
        dst[i] = gl_mul(a[i], b[i]);
        i += 1;
    }
}

/// Interleaved x^7 for 16 elements (one full-round S-box layer).
/// 4 serial mul steps × 16 independent chains = optimal latency hiding.
pub fn gl_pow7_x16(state: &mut [u64; 16]) {
    // Step 1: x2[i] = x[i]^2
    let mut x2 = [0u64; 16];
    for i in 0..16 {
        x2[i] = gl_mul(state[i], state[i]);
    }
    // Step 2: x3[i] = x2[i] * x[i]
    let mut x3 = [0u64; 16];
    for i in 0..16 {
        x3[i] = gl_mul(x2[i], state[i]);
    }
    // Step 3: x4[i] = x2[i]^2
    let mut x4 = [0u64; 16];
    for i in 0..16 {
        x4[i] = gl_mul(x2[i], x2[i]);
    }
    // Step 4: out[i] = x3[i] * x4[i]
    for i in 0..16 {
        state[i] = gl_mul(x3[i], x4[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_sub() {
        assert_eq!(gl_add(1, 2), 3);
        assert_eq!(canonicalize(gl_add(P - 1, 1)), 0); // wrap
        assert_eq!(gl_sub(5, 3), 2);
        assert_eq!(canonicalize(gl_sub(0, 1)), P - 1); // underflow
    }

    #[test]
    fn test_mul() {
        assert_eq!(gl_mul(3, 7), 21);
        // (P-1)^2 mod P = 1
        assert_eq!(gl_reduce128_canonical(gl_mul(P - 1, P - 1)), 1);
    }

    #[test]
    fn test_pow7() {
        assert_eq!(gl_pow7(2), 128);
        assert_eq!(gl_pow7(0), 0);
        assert_eq!(gl_pow7(1), 1);
    }

    #[test]
    fn test_inv() {
        // x * inv(x) = 1
        let x = 42u64;
        let xi = gl_inv(x);
        assert_eq!(canonicalize(gl_mul(x, xi)), 1);

        // inv(1) = 1
        assert_eq!(canonicalize(gl_inv(1)), 1);

        // inv(0) = 0
        assert_eq!(gl_inv(0), 0);

        // inv(P-1) = P-1  (since (P-1)^2 = 1)
        assert_eq!(canonicalize(gl_inv(P - 1)), P - 1);
    }

    #[test]
    fn test_mul_batch() {
        let a = [3u64, 5, 7, 11, 13, 17, 19, 23];
        let b = [2u64, 3, 4, 5, 6, 7, 8, 9];
        let mut dst = [0u64; 8];
        gl_mul_batch(&a, &b, &mut dst);
        for i in 0..8 {
            assert_eq!(dst[i], gl_mul(a[i], b[i]));
        }
    }

    #[test]
    fn test_pow7_x16() {
        let mut state = [0u64; 16];
        for i in 0..16 {
            state[i] = (i as u64 + 1) * 3;
        }
        let mut expected = state;
        for i in 0..16 {
            expected[i] = gl_pow7(expected[i]);
        }
        gl_pow7_x16(&mut state);
        assert_eq!(state, expected);
    }

    fn canonicalize(x: u64) -> u64 {
        if x >= P {
            x - P
        } else {
            x
        }
    }

    fn gl_reduce128_canonical(x: u64) -> u64 {
        canonicalize(x)
    }
}
