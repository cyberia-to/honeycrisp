//! Bit-identity gate: `acpu::field::tip5::*` must match `twenty_first::Tip5`
//! on every input.
//!
//! `twenty-first` is admitted as a `dev-dependency` solely to drive this test.
//! Production code in `acpu` has no dependency on it.

use acpu::field::tip5::{tip5_hash_pair, tip5_hash_varlen, tip5_permute};
use twenty_first::prelude::BFieldElement;
use twenty_first::tip5::Tip5;

/// Splitmix64 — header-free deterministic PRNG (no external dep).
struct Smx(u64);
impl Smx {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

const P: u64 = 0xffff_ffff_0000_0001;

/// Sample a canonical Goldilocks field element via rejection — uniform in [0, p).
fn rand_field(rng: &mut Smx) -> u64 {
    loop {
        // Use ~63 bits of randomness to keep the rejection rate negligible.
        let v = rng.next() & 0x7fff_ffff_ffff_ffff;
        if v < P {
            return v;
        }
    }
}

fn twenty_first_permute(state: [u64; 16]) -> [u64; 16] {
    let mut t5 = Tip5 {
        state: state.map(BFieldElement::new),
    };
    t5.permutation();
    t5.state.map(|b| b.value())
}

fn twenty_first_hash_pair(left: [u64; 5], right: [u64; 5]) -> [u64; 5] {
    let l = twenty_first::prelude::Digest::new(left.map(BFieldElement::new));
    let r = twenty_first::prelude::Digest::new(right.map(BFieldElement::new));
    Tip5::hash_pair(l, r).values().map(|b| b.value())
}

fn twenty_first_hash_varlen(input: &[u64]) -> [u64; 5] {
    let buf: Vec<BFieldElement> = input.iter().copied().map(BFieldElement::new).collect();
    Tip5::hash_varlen(&buf).values().map(|b| b.value())
}

#[test]
fn permute_random_1000() {
    let mut rng = Smx::new(0xC0FFEE_BAD1DEA_u64);
    for trial in 0..1000 {
        let mut state = [0u64; 16];
        for s in &mut state {
            *s = rand_field(&mut rng);
        }
        let expected = twenty_first_permute(state);
        let mut got = state;
        tip5_permute(&mut got);
        assert_eq!(
            got, expected,
            "tip5_permute mismatch on trial {trial}; input={state:?}"
        );
    }
}

#[test]
fn permute_edge_inputs() {
    let cases: &[[u64; 16]] = &[
        [0; 16],
        [1; 16],
        [P - 1; 16],
        core::array::from_fn(|i| i as u64),
        core::array::from_fn(|i| P - 1 - i as u64),
    ];
    for (idx, state) in cases.iter().enumerate() {
        let expected = twenty_first_permute(*state);
        let mut got = *state;
        tip5_permute(&mut got);
        assert_eq!(got, expected, "permute_edge_inputs case {idx}");
    }
}

#[test]
fn hash_pair_random_1000() {
    let mut rng = Smx::new(0xDEAD_BEEF_FEED_FACEu64);
    for trial in 0..1000 {
        let mut left = [0u64; 5];
        let mut right = [0u64; 5];
        for x in &mut left {
            *x = rand_field(&mut rng);
        }
        for x in &mut right {
            *x = rand_field(&mut rng);
        }
        let expected = twenty_first_hash_pair(left, right);
        let got = tip5_hash_pair(left, right);
        assert_eq!(
            got, expected,
            "tip5_hash_pair mismatch on trial {trial}; left={left:?} right={right:?}"
        );
    }
}

#[test]
fn hash_varlen_padding_boundaries() {
    // Lengths that straddle the padding boundary in interesting ways:
    //   0       — single empty pad-only absorb
    //   1, 9    — single non-full chunk, padding at positions 1 / 9
    //   10      — one full chunk + one pad-only absorb (longest pad)
    //   11      — one full chunk + a one-element chunk + pad
    //   30      — three full chunks + pad-only
    //   100, 1000 — many full chunks
    let lens = [0usize, 1, 9, 10, 11, 30, 100, 1000];
    let mut rng = Smx::new(0x1234_5678_9ABC_DEF0u64);
    for &n in &lens {
        // Run several trials per length to catch any random-input divergence.
        for trial in 0..16 {
            let input: Vec<u64> = (0..n).map(|_| rand_field(&mut rng)).collect();
            let expected = twenty_first_hash_varlen(&input);
            let got = tip5_hash_varlen(&input);
            assert_eq!(
                got, expected,
                "tip5_hash_varlen mismatch at len={n} trial={trial}"
            );
        }
    }
}

#[test]
fn hash_varlen_random_1000() {
    let mut rng = Smx::new(0xBABE_F00D_C0DE_1337u64);
    for trial in 0..1000 {
        let n = (rng.next() % 200) as usize;
        let input: Vec<u64> = (0..n).map(|_| rand_field(&mut rng)).collect();
        let expected = twenty_first_hash_varlen(&input);
        let got = tip5_hash_varlen(&input);
        assert_eq!(got, expected, "trial {trial}, len={n}");
    }
}
