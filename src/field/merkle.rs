//! Binary Merkle tree hashing using Poseidon2 permutation.
//!
//! hash_node: compress two 32-byte children into one 32-byte parent.
//! hash_leaves: build a complete binary Merkle tree from leaf data.

use super::goldilocks::*;
use super::permute::poseidon2_permute;

/// Hash two 32-byte children into a 32-byte parent.
///
/// Absorbs left (4 elements) and right (4 elements) into the rate portion
/// of the sponge state, permutes, and squeezes 4 elements.
#[inline]
pub fn hash_node(left: &[u64; 4], right: &[u64; 4], rc: &[u64; 144], diag: &[u64; 16]) -> [u64; 4] {
    let mut state = [0u64; 16];
    // Absorb: left into state[0..4], right into state[4..8]
    state[0] = left[0];
    state[1] = left[1];
    state[2] = left[2];
    state[3] = left[3];
    state[4] = right[0];
    state[5] = right[1];
    state[6] = right[2];
    state[7] = right[3];

    poseidon2_permute(&mut state, rc, diag);

    [state[0], state[1], state[2], state[3]]
}

/// Build a binary Merkle tree from `n` leaf hashes.
///
/// `leaves`: array of 4-element hashes (each leaf is 4 × u64 = 32 bytes).
/// Returns the root hash.
///
/// `n` must be a power of 2.
pub fn merkle_root(leaves: &[[u64; 4]], rc: &[u64; 144], diag: &[u64; 16]) -> [u64; 4] {
    let n = leaves.len();
    assert!(n > 0 && n.is_power_of_two());

    if n == 1 {
        return leaves[0];
    }

    // Build tree bottom-up in a temporary buffer
    let mut current: Vec<[u64; 4]> = leaves.to_vec();
    let mut size = n;

    while size > 1 {
        let half = size / 2;
        for i in 0..half {
            current[i] = hash_node(&current[2 * i], &current[2 * i + 1], rc, diag);
        }
        size = half;
    }

    current[0]
}

/// Batch inversion using Montgomery's trick.
///
/// Inverts `n` field elements with 1 inversion + 3(n-1) multiplications.
/// Zero elements produce zero in output.
pub fn batch_inv(input: &[u64], output: &mut [u64]) {
    let n = input.len().min(output.len());
    if n == 0 {
        return;
    }

    // Phase 1: prefix products
    output[0] = input[0];
    for i in 1..n {
        output[i] = if input[i] == 0 || input[i] == P {
            output[i - 1]
        } else {
            gl_mul(output[i - 1], input[i])
        };
    }

    // Phase 2: invert total product
    let mut inv_all = gl_inv(output[n - 1]);

    // Phase 3: propagate backward
    for i in (1..n).rev() {
        if input[i] == 0 || input[i] == P {
            output[i] = 0;
        } else {
            output[i] = gl_mul(inv_all, output[i - 1]);
            inv_all = gl_mul(inv_all, input[i]);
        }
    }
    output[0] = if input[0] == 0 || input[0] == P {
        0
    } else {
        inv_all
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_rc() -> [u64; 144] {
        core::array::from_fn(|i| (i as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15))
    }

    fn test_diag() -> [u64; 16] {
        core::array::from_fn(|i| (i as u64 + 2).wrapping_mul(0x517CC1B727220A95))
    }

    fn canonicalize(x: u64) -> u64 {
        if x >= P {
            x - P
        } else {
            x
        }
    }

    #[test]
    fn merkle_root_deterministic() {
        let rc = test_rc();
        let diag = test_diag();
        let leaves: Vec<[u64; 4]> = (0..8)
            .map(|i| [i * 4 + 1, i * 4 + 2, i * 4 + 3, i * 4 + 4])
            .collect();
        let r1 = merkle_root(&leaves, &rc, &diag);
        let r2 = merkle_root(&leaves, &rc, &diag);
        assert_eq!(r1, r2);
    }

    #[test]
    fn merkle_root_changes_with_input() {
        let rc = test_rc();
        let diag = test_diag();
        let mut leaves: Vec<[u64; 4]> = (0..4).map(|i| [i + 1, i + 2, i + 3, i + 4]).collect();
        let r1 = merkle_root(&leaves, &rc, &diag);
        leaves[0][0] += 1;
        let r2 = merkle_root(&leaves, &rc, &diag);
        assert_ne!(r1, r2);
    }

    #[test]
    fn batch_inv_correctness() {
        let input = [3u64, 7, 42, 100, 999];
        let mut output = [0u64; 5];
        batch_inv(&input, &mut output);
        for i in 0..5 {
            assert_eq!(
                canonicalize(gl_mul(input[i], output[i])),
                1,
                "inv failed at {}",
                i
            );
        }
    }

    #[test]
    fn batch_inv_handles_zero() {
        let input = [3u64, 0, 42];
        let mut output = [0u64; 3];
        batch_inv(&input, &mut output);
        assert_eq!(canonicalize(gl_mul(input[0], output[0])), 1);
        assert_eq!(output[1], 0);
        assert_eq!(canonicalize(gl_mul(input[2], output[2])), 1);
    }
}
