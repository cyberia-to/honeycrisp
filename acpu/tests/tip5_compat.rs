//! Bit-identity gate: `acpu::field::tip5::*` must equal `twenty_first::Tip5`.
//!
//! Mandatory. Failures here mean the implementation is incorrect and must
//! be fixed before any speed work can land.

use acpu::field::tip5::{tip5_hash_pair, tip5_hash_varlen, tip5_permute, tip5_permute_batch};
use rand::{Rng, RngCore, SeedableRng};
use twenty_first::prelude::{BFieldElement, Digest, Tip5};

// Number of random samples per fuzz block. Keep at least 1000 per the
// mission spec.
const N_SAMPLES: usize = 1024;

fn random_state(rng: &mut impl RngCore) -> [u64; 16] {
    let mut s = [0u64; 16];
    for v in s.iter_mut() {
        *v = rng.next_u64();
    }
    s
}

fn state_from_raw(raw: [u64; 16]) -> [BFieldElement; 16] {
    let mut out = [BFieldElement::default(); 16];
    for (o, r) in out.iter_mut().zip(raw.iter()) {
        *o = BFieldElement::from_raw_u64(*r);
    }
    out
}

fn state_to_raw(state: [BFieldElement; 16]) -> [u64; 16] {
    let mut out = [0u64; 16];
    for (o, e) in out.iter_mut().zip(state.iter()) {
        *o = e.raw_u64();
    }
    out
}

#[test]
fn permute_matches_twenty_first_1000_random_inputs() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x0C0F_FEE5_71F5);
    for _ in 0..N_SAMPLES {
        let raw = random_state(&mut rng);

        let mut ours = raw;
        tip5_permute(&mut ours);

        let mut theirs = Tip5 {
            state: state_from_raw(raw),
        };
        theirs.permutation();

        assert_eq!(ours, state_to_raw(theirs.state));
    }
}

#[test]
fn hash_pair_matches_twenty_first_1000_random_inputs() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x00DE_ADBE_EF42_u64);
    for _ in 0..N_SAMPLES {
        let mut left = [0u64; 5];
        let mut right = [0u64; 5];
        for v in left.iter_mut() {
            // Use canonical values that BFieldElement::new() will accept;
            // hash_pair is the production call path so canonical-only is the
            // realistic input distribution. Still exercises the round
            // function over all 80 round constants.
            *v = BFieldElement::new(rng.random::<u64>()).raw_u64();
        }
        for v in right.iter_mut() {
            *v = BFieldElement::new(rng.random::<u64>()).raw_u64();
        }

        let ours = tip5_hash_pair(left, right);

        let left_d = Digest::new([
            BFieldElement::from_raw_u64(left[0]),
            BFieldElement::from_raw_u64(left[1]),
            BFieldElement::from_raw_u64(left[2]),
            BFieldElement::from_raw_u64(left[3]),
            BFieldElement::from_raw_u64(left[4]),
        ]);
        let right_d = Digest::new([
            BFieldElement::from_raw_u64(right[0]),
            BFieldElement::from_raw_u64(right[1]),
            BFieldElement::from_raw_u64(right[2]),
            BFieldElement::from_raw_u64(right[3]),
            BFieldElement::from_raw_u64(right[4]),
        ]);
        let theirs = Tip5::hash_pair(left_d, right_d);

        let theirs_raw: [u64; 5] = [
            theirs.values()[0].raw_u64(),
            theirs.values()[1].raw_u64(),
            theirs.values()[2].raw_u64(),
            theirs.values()[3].raw_u64(),
            theirs.values()[4].raw_u64(),
        ];
        assert_eq!(ours, theirs_raw);
    }
}

#[test]
fn hash_varlen_matches_twenty_first_padding_boundaries() {
    // Lengths chosen to hit every padding case: empty, < RATE, == RATE,
    // RATE-1 (max remainder), RATE+1 (one full chunk + remainder of 1),
    // and large multi-chunk inputs including a multiple of RATE.
    let lengths = [0usize, 1, 9, 10, 11, 30, 100, 1000];

    let mut rng = rand::rngs::StdRng::seed_from_u64(0x00FE_EDFA_CE99_u64);

    for &len in &lengths {
        // Build canonical BFieldElement input.
        let bfe_in: Vec<BFieldElement> = (0..len)
            .map(|_| BFieldElement::new(rng.random::<u64>()))
            .collect();
        let raw_in: Vec<u64> = bfe_in.iter().map(|e| e.raw_u64()).collect();

        let ours = tip5_hash_varlen(&raw_in);

        let theirs = Tip5::hash_varlen(&bfe_in);
        let theirs_raw: [u64; 5] = [
            theirs.values()[0].raw_u64(),
            theirs.values()[1].raw_u64(),
            theirs.values()[2].raw_u64(),
            theirs.values()[3].raw_u64(),
            theirs.values()[4].raw_u64(),
        ];

        assert_eq!(
            ours, theirs_raw,
            "hash_varlen mismatch at input length {len}"
        );
    }
}

#[test]
fn permute_batch_matches_scalar_path() {
    const N: usize = 8;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x000B_ADF0_0D77_u64);

    let mut iter = 0;
    while iter < 128 {
        let mut originals = [[0u64; 16]; N];
        for s in originals.iter_mut() {
            *s = random_state(&mut rng);
        }

        // Batched path.
        let mut batched = originals;
        tip5_permute_batch::<N>(&mut batched);

        // Scalar path.
        let mut scalar = originals;
        for s in scalar.iter_mut() {
            tip5_permute(s);
        }
        // And cross-check against twenty-first.
        for k in 0..N {
            let mut t = Tip5 {
                state: state_from_raw(originals[k]),
            };
            t.permutation();
            assert_eq!(batched[k], state_to_raw(t.state));
            assert_eq!(scalar[k], state_to_raw(t.state));
        }
        iter += 1;
    }
}
