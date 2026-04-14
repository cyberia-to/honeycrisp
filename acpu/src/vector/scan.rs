// ---------------------------------------------------------------------------
// Scan operations -- NEON prefix sum + matrix transpose
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// Exclusive prefix sum. `out[0] = 0`, `out[i] = x[0] + x[1] + ... + x[i-1]`.
/// Panics if `out.len() != x.len()`.
pub fn prefix_sum_f32(out: &mut [f32], x: &[f32]) {
    assert_eq!(out.len(), x.len(), "prefix_sum_f32: length mismatch");
    let len = x.len();
    if len == 0 {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        let mut i = 0;
        let mut carry: f32 = 0.0;
        unsafe {
            let px = x.as_ptr();
            let po = out.as_mut_ptr();
            let zero = vdupq_n_f32(0.0);

            // Process 4 elements at a time with NEON shift-and-add scan
            while i + 4 <= len {
                let v = vld1q_f32(px.add(i));

                // Local exclusive prefix sum within the 4-lane vector:
                // step 1: shift right by 1 lane, fill with 0
                let s1 = vextq_f32(zero, v, 3); // [0, v0, v1, v2]
                let v = vaddq_f32(v, s1); // [v0, v0+v1, v1+v2, v2+v3]
                                          // step 2: shift right by 2 lanes, fill with 0
                let s2 = vextq_f32(zero, v, 2); // [0, 0, v0, v0+v1]
                let v = vaddq_f32(v, s2); // [v0, v0+v1, v0+v1+v2, v0+v1+v2+v3]

                // Convert to exclusive: shift right by 1, insert 0
                let exc = vextq_f32(zero, v, 3); // [0, v0, v0+v1, v0+v1+v2]

                // Add carry from previous blocks
                let result = vaddq_f32(exc, vdupq_n_f32(carry));
                vst1q_f32(po.add(i), result);

                // New carry = carry + sum of all 4 elements (last lane of inclusive)
                carry += vgetq_lane_f32::<3>(v);
                i += 4;
            }

            // Scalar tail
            while i < len {
                *po.add(i) = carry;
                carry += *px.add(i);
                i += 1;
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut carry: f32 = 0.0;
        for i in 0..len {
            out[i] = carry;
            carry += x[i];
        }
    }
}

/// Matrix transpose. `src` is row-major `[rows x cols]`, `dst` is row-major
/// `[cols x rows]`. Panics if either slice is too short.
pub fn transpose_f32(dst: &mut [f32], src: &[f32], rows: usize, cols: usize) {
    assert!(
        src.len() >= rows * cols,
        "transpose_f32: src too short ({} < {})",
        src.len(),
        rows * cols
    );
    assert!(
        dst.len() >= rows * cols,
        "transpose_f32: dst too short ({} < {})",
        dst.len(),
        rows * cols
    );
    if rows == 0 || cols == 0 {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        let r4 = rows / 4;
        let c4 = cols / 4;
        let r_rem = rows % 4;
        let c_rem = cols % 4;

        unsafe {
            let ps = src.as_ptr();
            let pd = dst.as_mut_ptr();

            // 4x4 tiles with NEON transpose
            // Cache-blocking: process 64×64 super-tiles to keep both
            // src reads and dst writes within L1/L2 cache.
            const BLOCK: usize = 64;
            let rb = rows.div_ceil(BLOCK);
            let cb = cols.div_ceil(BLOCK);
            for br in 0..rb {
                let r_lo = br * BLOCK;
                let r_hi = (r_lo + BLOCK).min(r4 * 4);
                for bc in 0..cb {
                    let c_lo = bc * BLOCK;
                    let c_hi = (c_lo + BLOCK).min(c4 * 4);

                    for tr in (r_lo / 4)..(r_hi / 4) {
                        let r0 = tr * 4;
                        for tc in (c_lo / 4)..(c_hi / 4) {
                            let c0 = tc * 4;

                            // Load 4 rows of 4 elements each
                            let v0 = vld1q_f32(ps.add(r0 * cols + c0));
                            let v1 = vld1q_f32(ps.add((r0 + 1) * cols + c0));
                            let v2 = vld1q_f32(ps.add((r0 + 2) * cols + c0));
                            let v3 = vld1q_f32(ps.add((r0 + 3) * cols + c0));

                            // 4x4 transpose using trn + zip on f64 reinterpret
                            let t0 = vtrn1q_f32(v0, v1); // [a0,b0,a2,b2]
                            let t1 = vtrn2q_f32(v0, v1); // [a1,b1,a3,b3]
                            let t2 = vtrn1q_f32(v2, v3); // [c0,d0,c2,d2]
                            let t3 = vtrn2q_f32(v2, v3); // [c1,d1,c3,d3]

                            // Zip low/high halves via f64 reinterpret
                            let u0 = vreinterpretq_f32_f64(vzip1q_f64(
                                vreinterpretq_f64_f32(t0),
                                vreinterpretq_f64_f32(t2),
                            )); // row0: [a0,b0,c0,d0]
                            let u1 = vreinterpretq_f32_f64(vzip1q_f64(
                                vreinterpretq_f64_f32(t1),
                                vreinterpretq_f64_f32(t3),
                            )); // row1: [a1,b1,c1,d1]
                            let u2 = vreinterpretq_f32_f64(vzip2q_f64(
                                vreinterpretq_f64_f32(t0),
                                vreinterpretq_f64_f32(t2),
                            )); // row2: [a2,b2,c2,d2]
                            let u3 = vreinterpretq_f32_f64(vzip2q_f64(
                                vreinterpretq_f64_f32(t1),
                                vreinterpretq_f64_f32(t3),
                            )); // row3: [a3,b3,c3,d3]

                            // Store transposed: dst[col][row]
                            vst1q_f32(pd.add(c0 * rows + r0), u0);
                            vst1q_f32(pd.add((c0 + 1) * rows + r0), u1);
                            vst1q_f32(pd.add((c0 + 2) * rows + r0), u2);
                            vst1q_f32(pd.add((c0 + 3) * rows + r0), u3);
                        }
                    }
                } // bc
            } // br

            // Remainder: right strip (cols not multiple of 4)
            if c_rem > 0 {
                let c_start = c4 * 4;
                for r in 0..rows {
                    for c in c_start..cols {
                        *pd.add(c * rows + r) = *ps.add(r * cols + c);
                    }
                }
            }

            // Remainder: bottom strip (rows not multiple of 4), only tiled cols
            if r_rem > 0 {
                let r_start = r4 * 4;
                let c_end = c4 * 4;
                for r in r_start..rows {
                    for c in 0..c_end {
                        *pd.add(c * rows + r) = *ps.add(r * cols + c);
                    }
                }
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for r in 0..rows {
            for c in 0..cols {
                dst[c * rows + r] = src[r * cols + c];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_sum_basic() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut out = [0.0f32; 5];
        prefix_sum_f32(&mut out, &x);
        let expected = [0.0, 1.0, 3.0, 6.0, 10.0];
        for i in 0..5 {
            assert!(
                (out[i] - expected[i]).abs() < 1e-5,
                "mismatch at {}: {} != {}",
                i,
                out[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_prefix_sum_neon_block() {
        // Exactly 4 elements -- one full NEON block
        let x = [10.0, 20.0, 30.0, 40.0];
        let mut out = [0.0f32; 4];
        prefix_sum_f32(&mut out, &x);
        assert!((out[0] - 0.0).abs() < 1e-5);
        assert!((out[1] - 10.0).abs() < 1e-5);
        assert!((out[2] - 30.0).abs() < 1e-5);
        assert!((out[3] - 60.0).abs() < 1e-5);
    }

    #[test]
    fn test_prefix_sum_large() {
        let n = 1024;
        let x: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; n as usize];
        prefix_sum_f32(&mut out, &x);
        // out[i] = sum of 1..i = i*(i-1)/2 ... no, sum of x[0..i]
        let mut expected = 0.0f32;
        for i in 0..n as usize {
            assert!(
                (out[i] - expected).abs() < 1.0,
                "mismatch at {}: {} != {}",
                i,
                out[i],
                expected
            );
            expected += x[i];
        }
    }

    #[test]
    fn test_prefix_sum_empty() {
        let mut out: [f32; 0] = [];
        prefix_sum_f32(&mut out, &[]);
    }

    #[test]
    fn test_transpose_4x4() {
        #[rustfmt::skip]
        let src = [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let mut dst = [0.0f32; 16];
        transpose_f32(&mut dst, &src, 4, 4);
        #[rustfmt::skip]
        let expected = [
            1.0, 5.0, 9.0, 13.0,
            2.0, 6.0, 10.0, 14.0,
            3.0, 7.0, 11.0, 15.0,
            4.0, 8.0, 12.0, 16.0,
        ];
        for i in 0..16 {
            assert!(
                (dst[i] - expected[i]).abs() < 1e-5,
                "mismatch at {}: {} != {}",
                i,
                dst[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_transpose_non_square() {
        // 3 rows x 5 cols
        let src: Vec<f32> = (1..=15).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; 15];
        transpose_f32(&mut dst, &src, 3, 5);
        // dst should be 5 rows x 3 cols
        for r in 0..3 {
            for c in 0..5 {
                assert!(
                    (dst[c * 3 + r] - src[r * 5 + c]).abs() < 1e-5,
                    "mismatch at ({},{}): {} != {}",
                    r,
                    c,
                    dst[c * 3 + r],
                    src[r * 5 + c]
                );
            }
        }
    }

    #[test]
    fn test_transpose_large() {
        let rows = 17;
        let cols = 13;
        let src: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; rows * cols];
        transpose_f32(&mut dst, &src, rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                assert!(
                    (dst[c * rows + r] - src[r * cols + c]).abs() < 1e-5,
                    "mismatch at ({},{})",
                    r,
                    c
                );
            }
        }
    }

    #[test]
    fn test_transpose_empty() {
        let mut dst: [f32; 0] = [];
        transpose_f32(&mut dst, &[], 0, 0);
    }
}
