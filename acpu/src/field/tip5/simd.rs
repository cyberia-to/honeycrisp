//! SSVE-batched primitives for Tip5 on M4.
//!
//! All entry points open a [`crate::streaming::Stream`] for the duration
//! of the call and process 8 Goldilocks elements per SVE register lane
//! (SVL = 512 bits = 8 × u64). Caller-owned slices, bit-identical to the
//! scalar primitives in [`super`].

use crate::streaming::Stream;

/// Batched Goldilocks Montgomery multiply: `out[i] = raw_mul(a[i], b[i])`
/// for `i` in 0..8.
///
/// Inputs and output are Montgomery-raw u64 values (the same form held
/// inside Tip5's internal state). The function computes `MUL(a, b)`
/// (low 64 bits) and `UMULH(a, b)` (high 64 bits) in SVE, then runs
/// the same `montyred` reduction as the scalar path lane-by-lane.
///
/// Requires FEAT_SME. Returns `FeatureNotAvailable(Sme)` otherwise.
pub fn raw_mul_batch8(
    a: &[u64; 8],
    b: &[u64; 8],
    out: &mut [u64; 8],
) -> crate::Result<()> {
    if !crate::probe::scan().has_sme {
        return Err(crate::CpuError::FeatureNotAvailable(crate::Feature::Sme));
    }
    let stream = Stream::new()?;
    unsafe {
        raw_mul_batch8_asm(a.as_ptr(), b.as_ptr(), out.as_mut_ptr());
    }
    drop(stream);
    Ok(())
}

/// Streaming-amortized variant — caller already holds a live Stream.
/// Lets a hot loop pay the SMSTART/SMSTOP cost once across many calls.
///
/// # Safety
///
/// `_stream` proves streaming mode is live. Pointers must point at
/// 8 valid u64 each (a, b) and 8 writable u64 (out).
#[inline]
pub unsafe fn raw_mul_batch8_in_stream(
    _stream: &Stream,
    a: &[u64; 8],
    b: &[u64; 8],
    out: &mut [u64; 8],
) {
    raw_mul_batch8_asm(a.as_ptr(), b.as_ptr(), out.as_mut_ptr());
}

/// All-asm core. Loads 8 u64 each from a/b, runs MUL+UMULH+montyred in
/// SSVE, stores 8 u64 to out. Stream must already be live.
#[inline(never)]
unsafe fn raw_mul_batch8_asm(a: *const u64, b: *const u64, out: *mut u64) {
    core::arch::asm!(
        // P0 = all-true for .D lanes (8 lanes at SVL=64 bytes).
        ".word 0x25D8E3E0",                  // PTRUE P0.D, all
        // Pre-load reduction constants.
        ".word 0x05C203F0",                  // DUPM Z16.D, #0xFFFFFFFF
        ".word 0x25F8C031",                  // DUP  Z17.D, #1

        // Load Za = [x0], Zb = [x1]
        ".word 0xA5E0A000",                  // LD1D Z0.D, P0/Z, [x0]
        ".word 0xA5E0A021",                  // LD1D Z1.D, P0/Z, [x1]

        // Wide multiply: Z2 = lo(Za * Zb), Z3 = hi(Za * Zb)
        ".word 0x04E16002",                  // MUL   Z2.D, Z0.D, Z1.D
        ".word 0x04E16C03",                  // UMULH Z3.D, Z0.D, Z1.D

        // montyred:
        //   xl = Z2,  xh = Z3
        //   (a, e) = xl.overflowing_add(xl << 32)
        //   Z5 = xl << 32
        //   Z6 = xl + Z5  (= a, low 64)
        //   P1 = (xl > a)  (= e, carry-out predicate)
        ".word 0x04E09C45",                  // LSL Z5.D, Z2.D, #32
        ".word 0x04E50046",                  // ADD Z6.D, Z2.D, Z5.D
        ".word 0x24C60051",                  // CMPHI P1.D, P0/Z, Z2.D, Z6.D

        //   b = a - (a >> 32) - e
        //   Z7 = a >> 32
        //   Z8 = a - Z7  (Z6 - Z7)
        //   Z8 -= 1 under P1
        ".word 0x04E094C7",                  // LSR Z7.D, Z6.D, #32
        ".word 0x04E704C8",                  // SUB Z8.D, Z6.D, Z7.D
        ".word 0x04C10628",                  // SUB Z8.D, P1/M, Z8.D, Z17.D

        //   (r, c) = xh.overflowing_sub(b)
        //   Z9 = xh - b   (= r)
        //   P2 = (b > xh) (= c)
        ".word 0x04E80469",                  // SUB Z9.D, Z3.D, Z8.D
        ".word 0x24C30112",                  // CMPHI P2.D, P0/Z, Z8.D, Z3.D

        //   r -= 0xFFFFFFFF when c
        ".word 0x04C10A09",                  // SUB Z9.D, P2/M, Z9.D, Z16.D

        // Store result.
        ".word 0xE5E0E049",                  // ST1D Z9.D, P0, [x2]

        in("x0") a,
        in("x1") b,
        in("x2") out,
        options(nostack),
    );
}

// ---------------------------------------------------------------------------
// Tests — bit-identity against the scalar raw_mul path.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Replicate the scalar path's Montgomery reduce so the test can
    // generate expected outputs without exporting the private helpers.
    const P: u64 = 0xffff_ffff_0000_0001;
    fn montyred(x: u128) -> u64 {
        let xl = x as u64;
        let xh = (x >> 64) as u64;
        let (a, e) = xl.overflowing_add(xl << 32);
        let b = a.wrapping_sub(a >> 32).wrapping_sub(e as u64);
        let (r, c) = xh.overflowing_sub(b);
        r.wrapping_sub((1 + !P) * c as u64)
    }
    fn raw_mul_scalar(a: u64, b: u64) -> u64 {
        montyred((a as u128) * (b as u128))
    }

    fn check(a: [u64; 8], b: [u64; 8]) {
        if !crate::probe::scan().has_sme {
            return;
        }
        let mut sme = [0u64; 8];
        raw_mul_batch8(&a, &b, &mut sme).unwrap();
        for i in 0..8 {
            let expected = raw_mul_scalar(a[i], b[i]);
            assert_eq!(
                sme[i], expected,
                "lane {i}: a={:#x} b={:#x} sme={:#x} scalar={:#x}",
                a[i], b[i], sme[i], expected,
            );
        }
    }

    #[test]
    fn raw_mul_batch8_zero() {
        check([0u64; 8], [0u64; 8]);
    }

    #[test]
    fn raw_mul_batch8_one_one() {
        check([1u64; 8], [1u64; 8]);
    }

    #[test]
    fn raw_mul_batch8_small() {
        let a = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let b = [10u64, 20, 30, 40, 50, 60, 70, 80];
        check(a, b);
    }

    #[test]
    fn raw_mul_batch8_near_p() {
        // Exercise the borrow path (b > xh).
        let mut a = [0u64; 8];
        let mut b = [0u64; 8];
        for i in 0..8 {
            a[i] = P.wrapping_sub(i as u64 + 1);
            b[i] = P.wrapping_sub(2 * (i as u64) + 7);
        }
        check(a, b);
    }

    #[test]
    fn raw_mul_batch8_random_sweep() {
        // Splitmix64 — deterministic, no external dep.
        let mut s: u64 = 0xC0FFEE_BAD1DEA;
        let mut next = || {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        for _ in 0..200 {
            let mut a = [0u64; 8];
            let mut b = [0u64; 8];
            for i in 0..8 {
                a[i] = next();
                b[i] = next();
            }
            check(a, b);
        }
    }
}
