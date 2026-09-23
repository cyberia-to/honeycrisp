//! Tip5 raw-Montgomery primitives — bit-identical to `twenty_first::BFieldElement`.
//!
//! The Tip5 state is stored as `[u64; 16]` of **raw** Montgomery values (i.e.
//! `BFieldElement::raw_u64()` outputs). Non-canonical reps (>= P) can appear
//! mid-permutation; twenty-first relies on that and so do we.
//!
//! All three primitives here are byte-for-byte transcriptions of twenty-first
//! 1.1.0. Do not "optimize" the algebra — the round-constant-add property
//! relies on `add` using `overflowing_sub(P - rhs)`.

/// Goldilocks prime.
pub const P: u64 = 0xffff_ffff_0000_0001;

/// Montgomery reduction: `montyred(x) = x * R^-1 mod P` where R = 2^64.
///
/// Output lies in `[0, P)`. Transcribed verbatim from
/// `twenty_first::BFieldElement::montyred`.
#[inline(always)]
pub const fn montyred(x: u128) -> u64 {
    let xl = x as u64;
    let xh = (x >> 64) as u64;
    let (a, e) = xl.overflowing_add(xl << 32);
    let b = a.wrapping_sub(a >> 32).wrapping_sub(e as u64);
    let (r, c) = xh.overflowing_sub(b);
    r.wrapping_sub((1 + !P) * c as u64)
}

/// Raw-Montgomery field add. Output is raw form, possibly degenerate (>= P)
/// when input is degenerate. Bit-identical to `BFieldElement::add`.
#[inline(always)]
pub const fn bfe_add_raw(a: u64, b: u64) -> u64 {
    // Compute a + b = a - (P - b).
    let (x1, c1) = a.overflowing_sub(P - b);
    if c1 {
        x1.wrapping_add(P)
    } else {
        x1
    }
}

/// Raw-Montgomery field multiply. Bit-identical to `BFieldElement::mul`.
///
/// Used by tests and helpers; the hot path multiplies inline through
/// `montyred` to keep the batch schedule visible to LLVM.
#[inline(always)]
#[cfg(test)]
pub const fn bfe_mul_raw(a: u64, b: u64) -> u64 {
    montyred((a as u128) * (b as u128))
}

/// Split-and-lookup S-box: replace each of the 8 little-endian bytes of `x`
/// with `LOOKUP_TABLE[byte]`. Bit-identical to
/// `Tip5::split_and_lookup`.
#[inline(always)]
pub fn split_and_lookup(x: u64, lut: &[u8; 256]) -> u64 {
    let mut bytes = x.to_le_bytes();
    // Unrolled — 8 byte lookups, all independent.
    bytes[0] = lut[bytes[0] as usize];
    bytes[1] = lut[bytes[1] as usize];
    bytes[2] = lut[bytes[2] as usize];
    bytes[3] = lut[bytes[3] as usize];
    bytes[4] = lut[bytes[4] as usize];
    bytes[5] = lut[bytes[5] as usize];
    bytes[6] = lut[bytes[6] as usize];
    bytes[7] = lut[bytes[7] as usize];
    u64::from_le_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Numerical sanity: montyred(0) == 0, montyred(R) == 1
    #[test]
    fn montyred_zero() {
        assert_eq!(montyred(0), 0);
    }

    #[test]
    fn bfe_add_below_p_is_canonical() {
        // 3 + 5 in Montgomery domain: a = mont(3), b = mont(5), result = mont(8).
        let mont_3 = montyred((3u128) * (0xffff_fffe_0000_0001u128));
        let mont_5 = montyred((5u128) * (0xffff_fffe_0000_0001u128));
        let mont_8 = montyred((8u128) * (0xffff_fffe_0000_0001u128));
        assert_eq!(bfe_add_raw(mont_3, mont_5), mont_8);
    }

    #[test]
    fn bfe_mul_one_is_identity() {
        // mont(1) = R mod P = montyred(R^2) — twenty-first's `new(1)`.
        const R2: u128 = 0xffff_fffe_0000_0001;
        let mont_1 = montyred(R2);
        let mont_42 = montyred(42u128 * R2);
        // 42 * 1 = 42 in field, both in raw Montgomery form.
        assert_eq!(bfe_mul_raw(mont_42, mont_1), mont_42);
    }
}
