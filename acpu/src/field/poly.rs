//! Multilinear polynomial evaluation over Goldilocks.
//!
//! Evaluates a multilinear polynomial given by its evaluations on the boolean
//! hypercube at an arbitrary field point.

use super::goldilocks::{gl_add, gl_mul, gl_sub};

// ── multilinear evaluation ────────────────────────────────────────────────────

/// Evaluate a multilinear polynomial at a field point.
///
/// `evals`: evaluations of the polynomial on the boolean hypercube `{0,1}^k`,
///   ordered lexicographically. Length must be `2^k` for some k ≥ 0.
/// `point`: field element coordinates, length = k = log2(evals.len()).
///
/// Returns the field element `f(point[0], ..., point[k-1])`.
///
/// Algorithm: recursive linear interpolation (low-degree extension).
///
/// ```text
/// if len == 1: return evals[0]
/// mid = len / 2; x = point[0]
/// vl = multilinear_eval(&evals[..mid], &point[1..])
/// vr = multilinear_eval(&evals[mid..], &point[1..])
/// return vl + x * (vr - vl)
/// ```
pub fn multilinear_eval(evals: &[u64], point: &[u64]) -> u64 {
    let len = evals.len();
    debug_assert!(len.is_power_of_two());
    if len == 1 {
        return evals[0];
    }
    let mid = len / 2;
    let x = point[0];
    let vl = multilinear_eval(&evals[..mid], &point[1..]);
    let vr = multilinear_eval(&evals[mid..], &point[1..]);
    // vl + x * (vr - vl)
    gl_add(vl, gl_mul(x, gl_sub(vr, vl)))
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::goldilocks::{gl_inv, gl_mul, P};

    fn canonicalize(x: u64) -> u64 {
        if x >= P {
            x - P
        } else {
            x
        }
    }

    #[test]
    fn eval_length_1() {
        // Constant polynomial: always returns the single eval.
        assert_eq!(multilinear_eval(&[42u64], &[]), 42);
        assert_eq!(multilinear_eval(&[0u64], &[]), 0);
    }

    #[test]
    fn eval_at_zero_returns_first() {
        // f(0) on a 1-variable polynomial [a, b] = a.
        let a = 17u64;
        let b = 99u64;
        let result = multilinear_eval(&[a, b], &[0u64]);
        assert_eq!(result, a);
    }

    #[test]
    fn eval_at_one_returns_second() {
        // f(1) on a 1-variable polynomial [a, b] = b.
        let a = 17u64;
        let b = 99u64;
        let result = multilinear_eval(&[a, b], &[1u64]);
        assert_eq!(result, b);
    }

    #[test]
    fn eval_at_half_is_midpoint() {
        // f(1/2) = (a + b) / 2 mod p.
        let a = 10u64;
        let b = 20u64;
        let inv2 = gl_inv(2);
        let expected = canonicalize(gl_mul(gl_add(a, b), inv2));
        let result = canonicalize(multilinear_eval(&[a, b], &[inv2]));
        assert_eq!(result, expected);
    }

    #[test]
    fn eval_two_variables_corners() {
        // [a, b, c, d] represents f(x0, x1):
        //   f(0,0)=a, f(0,1)=b, f(1,0)=c, f(1,1)=d
        let (a, b, c, d) = (1u64, 2, 3, 4);
        let evals = [a, b, c, d];
        assert_eq!(multilinear_eval(&evals, &[0, 0]), a);
        assert_eq!(multilinear_eval(&evals, &[0, 1]), b);
        assert_eq!(multilinear_eval(&evals, &[1, 0]), c);
        assert_eq!(multilinear_eval(&evals, &[1, 1]), d);
    }

    #[test]
    fn eval_two_variables_arbitrary_point() {
        // f(x0, x1) = a*(1-x0)*(1-x1) + b*(1-x0)*x1 + c*x0*(1-x1) + d*x0*x1
        // Test at x0=2, x1=3:
        //   (1-2)=-1 mod p = p-1, (1-3)=-2 mod p = p-2
        //   f = a*(p-1)*(p-2) + b*(p-1)*3 + c*2*(p-2) + d*2*3
        let (a, b, c, d) = (1u64, 2, 3, 4);
        let evals = [a, b, c, d];
        let x0 = 2u64;
        let x1 = 3u64;
        let result = canonicalize(multilinear_eval(&evals, &[x0, x1]));

        // Reference: compute manually
        use crate::field::goldilocks::gl_sub;
        let one_minus_x0 = gl_sub(1, x0);
        let one_minus_x1 = gl_sub(1, x1);
        let t_a = gl_mul(gl_mul(a, one_minus_x0), one_minus_x1);
        let t_b = gl_mul(gl_mul(b, one_minus_x0), x1);
        let t_c = gl_mul(gl_mul(c, x0), one_minus_x1);
        let t_d = gl_mul(gl_mul(d, x0), x1);
        let expected = canonicalize(gl_add(gl_add(t_a, t_b), gl_add(t_c, t_d)));

        assert_eq!(result, expected);
    }

    #[test]
    fn eval_size_8_corners() {
        // Size-8 polynomial: each corner should return the corresponding eval.
        let evals: Vec<u64> = (0..8).map(|i| (i as u64 + 1) * 7).collect();
        let corners: &[[u64; 3]] = &[
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ];
        for (i, pt) in corners.iter().enumerate() {
            let result = canonicalize(multilinear_eval(&evals, pt));
            assert_eq!(result, evals[i], "corner {i} mismatch");
        }
    }
}
