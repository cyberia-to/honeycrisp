//! acpu's Poseidon2 kernel against `cyber-hemera` 0.4, bit for bit.
//!
//! acpu keeps its own inlined permutation (register-resident, caller-supplied
//! constants) so the driver stays dependency-free. This test is the contract
//! that it is hemera's permutation: same round shape, same constants, same
//! output on edge and random states, and the same compression under
//! `hash_node` / `merkle_root` — with the one documented difference that
//! acpu's compression carries no tree flags in the capacity.

use acpu::field::permute::{poseidon2_permute, NUM_CONSTANTS, ROUNDS_F, ROUNDS_P};
use acpu::field::{hash_node, merkle_root};
use hemera::constants::ROUND_CONSTANTS_U64;
use hemera::field::{Goldilocks, MATRIX_DIAG_16};

const P: u64 = 0xFFFF_FFFF_0000_0001;

fn rc() -> [u64; NUM_CONSTANTS] {
    ROUND_CONSTANTS_U64
}

fn diag() -> [u64; 16] {
    MATRIX_DIAG_16.map(|d| d.as_canonical_u64())
}

/// hemera's permutation on canonical limbs.
fn reference(state: &[u64; 16]) -> [u64; 16] {
    let mut s = state.map(Goldilocks::new);
    hemera::permutation::permute(&mut s);
    s.map(|x| x.as_canonical_u64())
}

/// The bare compression `permute(left ‖ right ‖ 0⁸)[0..4]` through hemera.
fn reference_node(left: &[u64; 4], right: &[u64; 4]) -> [u64; 4] {
    let mut state = [0u64; 16];
    state[..4].copy_from_slice(left);
    state[4..8].copy_from_slice(right);
    let out = reference(&state);
    [out[0], out[1], out[2], out[3]]
}

/// xorshift64* — deterministic inputs, integers only.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn field(&mut self) -> u64 {
        self.next() % P
    }

    fn state(&mut self) -> [u64; 16] {
        core::array::from_fn(|_| self.field())
    }

    fn digest(&mut self) -> [u64; 4] {
        core::array::from_fn(|_| self.field())
    }
}

#[test]
fn round_shape_and_constants_are_hemera_0_4() {
    assert_eq!(ROUNDS_F, hemera::ROUNDS_F);
    assert_eq!(ROUNDS_P, hemera::ROUNDS_P);
    assert_eq!(NUM_CONSTANTS, hemera::constants::NUM_CONSTANTS);
    assert_eq!(ROUNDS_F + ROUNDS_P, hemera::ROUNDS_TOTAL);
    assert!(
        rc().iter().all(|&c| c < P),
        "hemera constants are canonical"
    );
    assert!(diag().iter().all(|&d| d < P));
}

#[test]
fn permutation_matches_hemera_on_edge_states() {
    let rc = rc();
    let diag = diag();
    let mut edges: Vec<[u64; 16]> = vec![[0; 16], [1; 16], [P - 1; 16]];
    for i in 0..16 {
        let mut unit = [0u64; 16];
        unit[i] = 1;
        edges.push(unit);
        let mut top = [0u64; 16];
        top[i] = P - 1;
        edges.push(top);
    }
    for state in edges {
        let mut ours = state;
        poseidon2_permute(&mut ours, &rc, &diag);
        assert_eq!(ours, reference(&state), "state {state:?}");
    }
}

#[test]
fn permutation_matches_hemera_on_random_states() {
    let rc = rc();
    let diag = diag();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut mismatches = 0u32;
    for _ in 0..100_000 {
        let state = rng.state();
        let mut ours = state;
        poseidon2_permute(&mut ours, &rc, &diag);
        if ours != reference(&state) {
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "mismatches over 100000 random states");
}

#[test]
fn permutation_output_is_canonical() {
    let rc = rc();
    let diag = diag();
    let mut rng = Rng(7);
    for _ in 0..10_000 {
        let mut state = rng.state();
        poseidon2_permute(&mut state, &rc, &diag);
        assert!(state.iter().all(|&x| x < P));
    }
}

#[test]
fn hash_node_matches_the_bare_permutation() {
    let rc = rc();
    let diag = diag();
    let mut rng = Rng(42);
    for _ in 0..10_000 {
        let (l, r) = (rng.digest(), rng.digest());
        assert_eq!(hash_node(&l, &r, &rc, &diag), reference_node(&l, &r));
    }
}

#[test]
fn hash_node_differs_from_hemera_tree_node_by_the_capacity_flags() {
    // hemera's tree mode sets flags in the capacity; acpu's compression leaves
    // the capacity at zero. Equal permutation, different sponge framing.
    let rc = rc();
    let diag = diag();
    let mut rng = Rng(99);
    let (l, r) = (rng.digest(), rng.digest());
    let to_hash = |d: &[u64; 4]| {
        let mut bytes = [0u8; 32];
        for (i, limb) in d.iter().enumerate() {
            bytes[i * 8..i * 8 + 8].copy_from_slice(&limb.to_le_bytes());
        }
        hemera::Hash::from_bytes(bytes)
    };
    let tree = hemera::tree::hash_node(&to_hash(&l), &to_hash(&r), false);
    let ours = to_hash(&hash_node(&l, &r, &rc, &diag));
    assert_ne!(tree.as_bytes(), ours.as_bytes());
}

#[test]
fn merkle_root_matches_a_reference_tree() {
    let rc = rc();
    let diag = diag();
    let mut rng = Rng(2026);
    let leaves: Vec<[u64; 4]> = (0..64).map(|_| rng.digest()).collect();
    let mut level = leaves.clone();
    while level.len() > 1 {
        level = level
            .chunks_exact(2)
            .map(|pair| reference_node(&pair[0], &pair[1]))
            .collect();
    }
    assert_eq!(merkle_root(&leaves, &rc, &diag), level[0]);
}
