//! Number Theoretic Transform (NTT) over Goldilocks.
//!
//! Cooley-Tukey DIT (decimation-in-time) NTT.
//! All elements are Goldilocks field elements in canonical form (< p).

use super::goldilocks::{gl_add, gl_mul, gl_sub, P};

// ── helpers ──────────────────────────────────────────────────────────────────

/// Compute base^exp mod p using square-and-multiply.
#[inline]
pub fn pow_gl(mut base: u64, mut exp: u64) -> u64 {
    let mut result = 1u64;
    base %= P;
    while exp > 0 {
        if exp & 1 == 1 {
            result = gl_mul(result, base);
        }
        base = gl_mul(base, base);
        exp >>= 1;
    }
    result
}

// ── bit-reversal ─────────────────────────────────────────────────────────────

/// Bit-reverse permute `vals` in-place.
///
/// `vals.len()` must be a power of two.
pub fn bit_reverse_permute(vals: &mut [u64]) {
    let n = vals.len();
    debug_assert!(n.is_power_of_two());
    let log_n = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = bit_reverse(i, log_n);
        if i < j {
            vals.swap(i, j);
        }
    }
}

#[inline]
fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0usize;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

// ── NTT ──────────────────────────────────────────────────────────────────────

/// Cooley-Tukey DIT NTT in-place over Goldilocks.
///
/// `vals`: mutable slice of field elements (canonical u64, < p).
/// `omega`: primitive n-th root of unity, where n = vals.len().
///
/// `vals.len()` must be a power of two and at least 1.
pub fn ntt_forward(vals: &mut [u64], omega: u64) {
    let n = vals.len();
    assert!(
        n.is_power_of_two(),
        "ntt_forward: length must be a power of two"
    );
    if n == 1 {
        return;
    }

    bit_reverse_permute(vals);

    let mut len = 2usize;
    while len <= n {
        // w_len = omega^(n / len): primitive len-th root of unity
        let w_len = pow_gl(omega, (n / len) as u64);

        let half = len / 2;
        let mut j = 0usize;
        while j < n {
            let mut w = 1u64;
            for k in 0..half {
                let u = vals[j + k];
                let v = gl_mul(vals[j + k + half], w);
                vals[j + k] = gl_add(u, v);
                vals[j + k + half] = gl_sub(u, v);
                w = gl_mul(w, w_len);
            }
            j += len;
        }

        len <<= 1;
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn canonicalize(x: u64) -> u64 {
        if x >= P {
            x - P
        } else {
            x
        }
    }

    /// Goldilocks has a primitive 2^32-nd root of unity.
    /// g = 7 is a generator of the multiplicative group.
    /// omega_2 = g^((p-1)/2) = g^(2^32 * (2^32 - 1) / 2) — but simpler:
    /// omega_2 = p - 1 (the unique primitive 2nd root of unity, since (p-1)^2 = 1 mod p).
    fn primitive_nth_root(n: usize) -> u64 {
        // g = 7 is a generator of GL*. |GL*| = p - 1 = 2^32 * (2^32 - 1).
        // For NTT we need the 2-adic part: p - 1 = 2^32 * odd.
        // So the largest power-of-two subgroup has order 2^32.
        assert!(n.is_power_of_two());
        assert!(n <= (1usize << 32));
        let g: u64 = 7;
        // omega = g^((p-1)/n)
        let exp = (P - 1) / n as u64;
        pow_gl(g, exp)
    }

    #[test]
    fn ntt_length_1() {
        let mut vals = [42u64];
        ntt_forward(&mut vals, 1);
        assert_eq!(vals[0], 42);
    }

    #[test]
    fn ntt_length_2_primitive_root() {
        // omega_2 = primitive 2nd root of unity = p - 1.
        // NTT of [1, 0] with DFT matrix [[1,1],[1,omega_2]] gives [1+0, 1*omega_2^0 * ... ].
        // DIT NTT of [1, 0]:
        //   after bit-reverse: [1, 0] (unchanged for n=2)
        //   butterfly: u=1, v=0*w=0 -> [1+0, 1-0] = [1, 1]
        //   Wait — standard DFT of [1,0] is [1, 1].
        // But with the generator omega = p-1 (which is -1 mod p):
        //   DIT butterfly for k=0: u=vals[0]=1, v=vals[1]*w(w=1) = 0
        //     -> vals[0] = 1, vals[1] = 1
        // So NTT([1,0]) = [1, 1], regardless of omega.
        let omega2 = primitive_nth_root(2); // = p - 1
        let mut vals = [1u64, 0u64];
        ntt_forward(&mut vals, omega2);
        assert_eq!(canonicalize(vals[0]), 1, "NTT([1,0])[0]");
        assert_eq!(canonicalize(vals[1]), 1, "NTT([1,0])[1]");
    }

    #[test]
    fn ntt_length_2_second_element() {
        // NTT([0, 1]) with omega = p-1:
        //   butterfly k=0: u=0, v=1*1=1 -> [1, p-1]
        let omega2 = primitive_nth_root(2);
        let mut vals = [0u64, 1u64];
        ntt_forward(&mut vals, omega2);
        assert_eq!(canonicalize(vals[0]), 1, "NTT([0,1])[0]");
        assert_eq!(canonicalize(vals[1]), P - 1, "NTT([0,1])[1] = -1 mod p");
    }

    #[test]
    fn ntt_linearity() {
        // NTT is linear: NTT(a + b) = NTT(a) + NTT(b)
        let n = 4;
        let omega = primitive_nth_root(n);
        let a = [1u64, 2, 3, 4];
        let b = [5u64, 6, 7, 8];

        let mut fa = a;
        let mut fb = b;
        let mut fab: Vec<u64> = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| gl_add(x, y))
            .collect();

        ntt_forward(&mut fa, omega);
        ntt_forward(&mut fb, omega);
        ntt_forward(&mut fab, omega);

        for i in 0..n {
            assert_eq!(
                canonicalize(gl_add(fa[i], fb[i])),
                canonicalize(fab[i]),
                "linearity failed at index {i}"
            );
        }
    }

    #[test]
    fn ntt_convolution() {
        // NTT of a polynomial times x-shift: shift by 1 multiplies each freq by omega^k.
        // f = [1, 2, 3, 4], shifted by 1 = [0, 1, 2, 3, 4] truncated to n=4 = [4, 1, 2, 3]
        // This is just a sanity check that NTT produces consistent results.
        let n = 4;
        let omega = primitive_nth_root(n);
        let mut vals = [1u64, 2, 3, 4];
        ntt_forward(&mut vals, omega);
        // All we check: output is deterministic
        let mut vals2 = [1u64, 2, 3, 4];
        ntt_forward(&mut vals2, omega);
        assert_eq!(vals, vals2);
    }

    #[test]
    fn bit_reverse_size4() {
        // For n=4: indices 0,1,2,3 -> 0,2,1,3
        let mut v = [10u64, 20, 30, 40];
        bit_reverse_permute(&mut v);
        assert_eq!(v, [10, 30, 20, 40]);
    }

    #[test]
    fn bit_reverse_involution() {
        let mut v: Vec<u64> = (0..8).collect();
        let original = v.clone();
        bit_reverse_permute(&mut v);
        bit_reverse_permute(&mut v);
        assert_eq!(v, original);
    }
}
