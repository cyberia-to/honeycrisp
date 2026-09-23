//! 16×16 INT32 SMOPA tile microkernel.
//!
//! Computes ZA0.S = (optionally += ) A_pack · B_pack for one 16×16 tile,
//! where the underlying operation is 2-way INT16 dot-product per ZA cell
//! (FEAT_SME2 SMOPA INT16 → INT32). On SVL=512 every Z register holds
//! 32 INT16 lanes, so per `kc` step the kernel consumes 32 INT16 from A
//! (16 rows × 2-way) and 32 INT16 from B (16 cols × 2-way), producing a
//! 16×16 INT32 partial accumulation in ZA0.S.
//!
//! ## Layout
//!
//! - `a_pack[p * 32 + (2*i + k)]` = `A[i, 2*p + k]` for i ∈ 0..16, k ∈ 0..2.
//!   Each `kc` step advances the 2-way K dimension by 2 columns.
//! - `b_pack[p * 32 + (2*j + k)]` = `B[2*p + k, j]` for j ∈ 0..16, k ∈ 0..2.
//! - `c[i * ldc + j]` ← (or +=) Σ_{p=0..kc, k=0..2} A[i, 2p+k] * B[2p+k, j]
//!   summed in INT32 with two's-complement wrap on overflow.
//!
//! This tile is the integer counterpart of `sme::tile::tile_16x16_f32`;
//! it exists for two reasons:
//!
//! 1. To serve as the bit-identity reference for INT16 SMOPA / UMOPA
//!    encoders — a numerical smoke test that runs SMOPA on real data and
//!    compares the result against a scalar matmul.
//! 2. To act as the building block any future caller (NTT, batched
//!    field-arithmetic) can compose without re-deriving the K-loop and
//!    MOVA-out plumbing.

use crate::streaming::Stream;

/// Compute one 16×16 INT32 tile of C with SMOPA over `kc * 2` K accumulations.
///
/// Each `kc` step folds two K columns into the ZA tile via one SMOPA op.
///
/// # Parameters
///
/// - `a_pack`: pointer to `kc * 32` INT16. Layout
///             `a_pack[p * 32 + 2*i + k] = A[i, 2*p + k]`.
/// - `b_pack`: pointer to `kc * 32` INT16. Layout
///             `b_pack[p * 32 + 2*j + k] = B[2*p + k, j]`.
/// - `c`:      pointer to the 16×16 destination tile (row-major INT32).
/// - `ldc`:    row stride of `c` in INT32 elements.
/// - `kc`:     number of outer-product steps (each consumes K=2 columns).
/// - `accumulate`: if true, C += A·B; if false, C = A·B.
///
/// # Safety
///
/// `_stream` proves the thread is in streaming mode with ZA enabled.
/// `a_pack` and `b_pack` must each have at least `kc * 32` valid INT16.
/// `c` must have at least `15 * ldc + 16` valid INT32 reachable from
/// its base. Caller must ensure FEAT_SME2 is present (Stream construction
/// gates on FEAT_SME but not SME2; the SMOPA opcode UNDEFs on chips that
/// lack SME2).
#[inline]
pub unsafe fn tile_16x16_i16_i32(
    _stream: &Stream,
    a_pack: *const i16,
    b_pack: *const i16,
    c: *mut i32,
    ldc: usize,
    kc: usize,
    accumulate: bool,
) {
    debug_assert!(kc > 0);
    let ldc_bytes = ldc * 4;
    if accumulate {
        tile_16x16_i16_accum(a_pack, b_pack, c, ldc_bytes, kc);
    } else {
        tile_16x16_i16_set(a_pack, b_pack, c, ldc_bytes, kc);
    }
}

#[inline(never)]
unsafe fn tile_16x16_i16_set(
    a_pack: *const i16,
    b_pack: *const i16,
    c: *mut i32,
    ldc_bytes: usize,
    kc: usize,
) {
    core::arch::asm!(
        ".word 0x2518E3E0",      // PTRUE P0.B, all (all 64 bytes; usable for .H + .S views)
        ".word 0x2518E3E1",      // PTRUE P1.B, all (32 INT16 lanes for the LD1H gates)
        ".word 0xC00800FF",      // ZERO {ZA}

        // K loop: x0 walks a_pack, x1 walks b_pack, x2 counts down kc.
        // Each iteration loads 32 INT16 (= 64 bytes) into Z0 / Z1
        // (predicated by P1/Z over the .B view) and applies one SMOPA
        // INT16→INT32 (predicated by P0/M over the .S accumulator view).
        "1:",
        ".word 0xA4A0A400",      // LD1H { Z0.H }, P1/Z, [x0]
        ".word 0xA4A0A421",      // LD1H { Z1.H }, P1/Z, [x1]
        "add x0, x0, #64",
        "add x1, x1, #64",
        ".word 0xA0810008",      // SMOPA ZA0.S, P0/M, P0/M, Z0.H, Z1.H
        "subs x2, x2, #1",
        "b.ne 1b",

        // Store 16 rows of ZA0.S into C (row-major INT32).
        "mov w12, #0",
        "2:",
        ".word 0xC0820002",      // MOVA Z2.S, P0/M, ZA0H.S[w12, 0]
        ".word 0xE540E062",      // ST1W Z2.S, P0, [x3]
        "add x3, x3, x4",
        "add w12, w12, #1",
        "cmp w12, #16",
        "b.ne 2b",

        inout("x0") a_pack => _,
        inout("x1") b_pack => _,
        inout("x2") kc => _,
        inout("x3") c => _,
        in("x4") ldc_bytes,
        out("x12") _,
        options(nostack),
    );
}

#[inline(never)]
unsafe fn tile_16x16_i16_accum(
    a_pack: *const i16,
    b_pack: *const i16,
    c: *mut i32,
    ldc_bytes: usize,
    kc: usize,
) {
    core::arch::asm!(
        ".word 0x2518E3E0",      // PTRUE P0.B, all (all 64 bytes; usable for .H + .S views)
        ".word 0x2518E3E1",      // PTRUE P1.B, all
        ".word 0xC00800FF",      // ZERO {ZA}

        "1:",
        ".word 0xA4A0A400",      // LD1H { Z0.H }, P1/Z, [x0]
        ".word 0xA4A0A421",      // LD1H { Z1.H }, P1/Z, [x1]
        "add x0, x0, #64",
        "add x1, x1, #64",
        ".word 0xA0810008",      // SMOPA ZA0.S, P0/M, P0/M, Z0.H, Z1.H
        "subs x2, x2, #1",
        "b.ne 1b",

        // Accumulate ZA0 into existing C row by row (SVE INT32 add).
        "mov w12, #0",
        "2:",
        ".word 0xA540A064",      // LD1W Z4.S, P0/Z, [x3]
        ".word 0xC0820002",      // MOVA Z2.S, P0/M, ZA0H.S[w12, 0]
        ".word 0x04A40044",      // ADD Z4.S, Z2.S, Z4.S — SVE INT32 add
        ".word 0xE540E064",      // ST1W Z4.S, P0, [x3]
        "add x3, x3, x4",
        "add w12, w12, #1",
        "cmp w12, #16",
        "b.ne 2b",

        inout("x0") a_pack => _,
        inout("x1") b_pack => _,
        inout("x2") kc => _,
        inout("x3") c => _,
        in("x4") ldc_bytes,
        out("x12") _,
        options(nostack),
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_matmul_i16_i32(a: &[i16], b: &[i16], c: &mut [i32], m: usize, n: usize, k: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut acc: i32 = 0;
                for p in 0..k {
                    acc = acc.wrapping_add((a[i * k + p] as i32) * (b[p * n + j] as i32));
                }
                c[i * n + j] = acc;
            }
        }
    }

    /// Pack A (m=16, k = 2 * kc) row-major into a_pack with layout
    /// `a_pack[p*32 + 2*i + k] = A[i, 2*p + k]`.
    fn pack_a(a: &[i16], kc: usize) -> Vec<i16> {
        let m = 16usize;
        let k_full = 2 * kc;
        let mut out = vec![0i16; kc * 32];
        for p in 0..kc {
            for i in 0..m {
                for kk in 0..2 {
                    out[p * 32 + 2 * i + kk] = a[i * k_full + 2 * p + kk];
                }
            }
        }
        out
    }

    /// Pack B (k = 2 * kc, n=16) row-major into b_pack with layout
    /// `b_pack[p*32 + 2*j + k] = B[2*p + k, j]`.
    fn pack_b(b: &[i16], kc: usize) -> Vec<i16> {
        let n = 16usize;
        let mut out = vec![0i16; kc * 32];
        for p in 0..kc {
            for j in 0..n {
                for kk in 0..2 {
                    out[p * 32 + 2 * j + kk] = b[(2 * p + kk) * n + j];
                }
            }
        }
        out
    }

    #[test]
    fn smopa_int16_tile_set_correct() {
        if !crate::probe::scan().has_sme2 {
            eprintln!("skip: FEAT_SME2 not present");
            return;
        }
        let m = 16usize;
        let n = 16usize;
        let kc = 4usize;
        let k = 2 * kc;
        // Small signed values to keep INT32 accumulation in range.
        let a: Vec<i16> = (0..m * k).map(|i| (((i as i32) % 11) - 5) as i16).collect();
        let b: Vec<i16> = (0..k * n).map(|i| (((i as i32) % 13) - 6) as i16).collect();
        let mut c_sme = vec![0i32; m * n];
        let mut c_ref = vec![0i32; m * n];

        let a_pack = pack_a(&a, kc);
        let b_pack = pack_b(&b, kc);

        let stream = Stream::new().unwrap();
        unsafe {
            tile_16x16_i16_i32(
                &stream,
                a_pack.as_ptr(),
                b_pack.as_ptr(),
                c_sme.as_mut_ptr(),
                n,
                kc,
                false,
            );
        }
        drop(stream);

        ref_matmul_i16_i32(&a, &b, &mut c_ref, m, n, k);

        for i in 0..m * n {
            assert_eq!(
                c_sme[i],
                c_ref[i],
                "tile mismatch at [{},{}]: sme={}, ref={}",
                i / n,
                i % n,
                c_sme[i],
                c_ref[i]
            );
        }
    }

    #[test]
    fn smopa_int16_tile_accumulate_correct() {
        if !crate::probe::scan().has_sme2 {
            return;
        }
        let m = 16usize;
        let n = 16usize;
        let kc = 2usize;
        let k = 2 * kc;
        let a: Vec<i16> = (0..m * k).map(|i| ((i as i32) - 8) as i16).collect();
        let b: Vec<i16> = (0..k * n).map(|i| ((i as i32) - 8) as i16).collect();
        let mut c_sme: Vec<i32> = (0..m * n).map(|i| (i as i32) * 7 - 100).collect();
        let mut c_ref: Vec<i32> = c_sme.clone();

        let a_pack = pack_a(&a, kc);
        let b_pack = pack_b(&b, kc);

        let stream = Stream::new().unwrap();
        unsafe {
            tile_16x16_i16_i32(
                &stream,
                a_pack.as_ptr(),
                b_pack.as_ptr(),
                c_sme.as_mut_ptr(),
                n,
                kc,
                true,
            );
        }
        drop(stream);

        // Reference: c_ref += a * b
        for i in 0..m {
            for j in 0..n {
                let mut acc: i32 = 0;
                for p in 0..k {
                    acc = acc.wrapping_add((a[i * k + p] as i32) * (b[p * n + j] as i32));
                }
                c_ref[i * n + j] = c_ref[i * n + j].wrapping_add(acc);
            }
        }

        for i in 0..m * n {
            assert_eq!(c_sme[i], c_ref[i], "accumulate mismatch at idx {i}");
        }
    }
}
