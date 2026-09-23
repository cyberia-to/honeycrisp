//! SME-scoped Tip5 permutation.
//!
//! The public surface is [`tip5_permute_sme`], a batched permutation that
//! runs N independent Tip5 states inside a single [`Stream`] scope.
//!
//! ## Status (2026-05-22)
//!
//! Phase 1 — API + correctness only. The current body opens a streaming-mode
//! context (so that downstream callers can rely on the lifetime contract and
//! the bit-identity gate is in place) and then permutes each state with the
//! existing scalar kernel. This guarantees the output is byte-identical to
//! N independent [`super::tip5_permute`] calls. Phase 2 (planned) replaces
//! the inner loop with batched SSVE arithmetic, which is the genuine
//! algorithmic win.
//!
//! ## Why a Stream is still useful at N=1
//!
//! Holding a live [`Stream`] is the contract under which any future SSVE /
//! SMOPA kernel can land here without changing callers. Stream construction
//! costs are ~30–80 cycles (SMSTART + ZA enable), which dominates only on
//! very small batches; for the trisha Merkle-layer use (thousands of pairs
//! per batch) this is far below the per-permutation cost.
//!
//! ## Why SMOPA itself does not (yet) accelerate Tip5
//!
//! Tip5's MDS layer is implemented as a fixed 16-input butterfly network
//! (`Tip5::generated_function`) of `wrapping_add` / `wrapping_sub` /
//! `wrapping_mul` on `u64`. The multiplies all have small signed
//! coefficients (|c| < 2^17) but the operand is full 64-bit. The natural
//! SME widening primitive on M4 (SMOPA INT16 → INT32) computes outer
//! products of pairs of INT16 lanes, which is structurally a matrix
//! multiply, not an element-wise wide-scalar broadcast. Mapping
//! `wrapping_mul(u64, i17_const)` onto SMOPA requires either reverting
//! `generated_function` to a direct 16×16 matrix-vector product (which
//! breaks raw-Montgomery bit-identity with the reference) or carrying out
//! a four-limb decomposition that costs more than the original scalar
//! multiply. The honest path is therefore to leave the algorithm as-is
//! and harvest the parallel-lane win via SSVE (8× u64 lanes per Z reg at
//! SVL=512), which is on the roadmap as phase 2.

use crate::streaming::Stream;

use super::tip5_permute;

/// Permute `N` independent Tip5 states inside one streaming-mode scope.
///
/// Bit-identical to calling [`tip5_permute`] `N` times. The streaming-mode
/// scope is opened once for the entire batch; per-permutation cost is the
/// scalar permute cost plus an amortized fraction of `SMSTART`/`SMSTOP`.
///
/// Returns an error only if the host lacks FEAT_SME; on every shipping
/// Apple SME implementation that path is taken cleanly.
///
/// # Example
///
/// ```no_run
/// use acpu::field::tip5::tip5_permute_sme;
///
/// let mut states = [[0u64; 16]; 8];
/// tip5_permute_sme::<8>(&mut states).expect("FEAT_SME required");
/// ```
#[inline]
pub fn tip5_permute_sme<const N: usize>(
    states: &mut [[u64; super::STATE_SIZE]; N],
) -> crate::Result<()> {
    let _stream = Stream::new()?;
    let mut k = 0;
    while k < N {
        tip5_permute(&mut states[k]);
        k += 1;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests — correctness only. Performance is covered by `bench/tip5.rs`.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::tip5::tip5_permute as scalar_permute;
    use crate::field::tip5::STATE_SIZE;

    fn rand_state(mut seed: u64) -> [u64; STATE_SIZE] {
        // xorshift* — local, dependency-free PRNG.
        let mut s = [0u64; STATE_SIZE];
        for v in s.iter_mut() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            *v = seed.wrapping_mul(0x2545_F491_4F6C_DD1D);
        }
        s
    }

    #[test]
    fn sme_permute_n1_bit_identical() {
        if !crate::probe::scan().has_sme {
            eprintln!("skip: FEAT_SME not present");
            return;
        }
        for seed in [1u64, 42, 0xDEAD_BEEF, 0xFEED_FACE] {
            let mut a = [rand_state(seed)];
            let mut b = a[0];
            tip5_permute_sme::<1>(&mut a).unwrap();
            scalar_permute(&mut b);
            assert_eq!(a[0], b, "seed {seed:x}");
        }
    }

    #[test]
    fn sme_permute_n8_bit_identical() {
        if !crate::probe::scan().has_sme {
            return;
        }
        const N: usize = 8;
        for seed in [1u64, 0xCAFE, 0xFFFF_0000_FFFF_0001] {
            let mut a: [[u64; 16]; N] =
                core::array::from_fn(|i| rand_state(seed.wrapping_add(i as u64)));
            let mut b = a;
            tip5_permute_sme::<N>(&mut a).unwrap();
            for slot in b.iter_mut() {
                scalar_permute(slot);
            }
            assert_eq!(a, b, "seed {seed:x}");
        }
    }
}
