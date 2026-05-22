//! 16×16 fp32 SME tile microkernel.
//!
//! Computes ZA0.S = (optionally += ) A_pack × B_pack for one 16×16 tile,
//! where A_pack and B_pack are each laid out as `kc` SVL-wide vectors
//! (kc × 16 f32 each). Output is written into the caller's `c` slice
//! row by row (16 rows × 16 floats per row at stride `ldc`).
//!
//! The K loop, ZA-zero, and ZA-store are all in one inline-asm block
//! so LLVM cannot interleave NEON or normal-mode work between them
//! (which would be illegal — Z/P registers do not exist outside
//! streaming mode).

use crate::streaming::Stream;

/// Compute one 16×16 tile of C with FMOPA.S over `kc` accumulations.
///
/// # Parameters
///
/// - `a_pack`: pointer to `kc * 16` f32. Lane `i` of A column `p` is at
///             `a_pack[p * 16 + i]`. (column-major within each k step)
/// - `b_pack`: pointer to `kc * 16` f32. Lane `j` of B row `p` is at
///             `b_pack[p * 16 + j]`.
/// - `c`:      pointer to the 16×16 destination tile (row-major).
/// - `ldc`:    row stride of `c` in f32 elements.
/// - `kc`:     number of accumulations (≥ 1).
/// - `accumulate`: if true, C += A·B; if false, C = A·B.
///
/// # Safety
///
/// `_stream` proves the thread is in streaming mode with ZA enabled.
/// Pointers must satisfy normal aliasing rules for one tile each.
/// `a_pack` and `b_pack` must each have at least `kc * 16` valid f32.
/// `c` must have at least `15 * ldc + 16` valid f32 reachable from its base.
#[inline]
pub unsafe fn tile_16x16_f32(
    _stream: &Stream,
    a_pack: *const f32,
    b_pack: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    accumulate: bool,
) {
    debug_assert!(kc > 0);

    let ldc_bytes: usize = ldc * 4;

    if accumulate {
        tile_16x16_f32_accum(a_pack, b_pack, c, ldc_bytes, kc);
    } else {
        tile_16x16_f32_set(a_pack, b_pack, c, ldc_bytes, kc);
    }
}

/// Overwrite path: ZA0 = 0, accumulate K, store to C.
#[inline(never)]
unsafe fn tile_16x16_f32_set(
    a_pack: *const f32,
    b_pack: *const f32,
    c: *mut f32,
    ldc_bytes: usize,
    kc: usize,
) {
    core::arch::asm!(
        ".word 0x2598E3E0",      // PTRUE P0.S, all
        ".word 0xC00800FF",      // ZERO {ZA}

        // K loop: x0 advances over A pack, x1 over B pack, x2 counts down.
        "1:",
        ".word 0xA540A000",      // LD1W Z0.S, P0/Z, [x0]
        ".word 0xA540A021",      // LD1W Z1.S, P0/Z, [x1]
        "add x0, x0, #64",
        "add x1, x1, #64",
        ".word 0x80810000",      // FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z1.S
        "subs x2, x2, #1",
        "b.ne 1b",

        // Store 16 rows of ZA0.S into C: x3 walks the C pointer,
        // x4 holds ldc_bytes, w12 is the ZA row counter.
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

/// Accumulate path: ZA0 = 0, accumulate K, add to existing C and store.
#[inline(never)]
unsafe fn tile_16x16_f32_accum(
    a_pack: *const f32,
    b_pack: *const f32,
    c: *mut f32,
    ldc_bytes: usize,
    kc: usize,
) {
    core::arch::asm!(
        ".word 0x2598E3E0",      // PTRUE P0.S, all
        ".word 0xC00800FF",      // ZERO {ZA}

        "1:",
        ".word 0xA540A000",      // LD1W Z0.S, P0/Z, [x0]
        ".word 0xA540A021",      // LD1W Z1.S, P0/Z, [x1]
        "add x0, x0, #64",
        "add x1, x1, #64",
        ".word 0x80810000",      // FMOPA ZA0.S
        "subs x2, x2, #1",
        "b.ne 1b",

        // Accumulate ZA0 into existing C row by row.
        "mov w12, #0",
        "2:",
        ".word 0xA540A064",      // LD1W Z4.S, P0/Z, [x3]
        ".word 0xC0820002",      // MOVA Z2.S, P0/M, ZA0H.S[w12, 0]
        ".word 0x65840044",      // FADD Z4.S, Z2.S, Z4.S
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

    fn ref_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = acc;
            }
        }
    }

    /// A_pack[p * 16 + i] = A[i, p] (i.e. column p of A as 16 contiguous f32).
    /// B_pack[p * 16 + j] = B[p, j] (row p of B as 16 contiguous f32).
    /// The outer product Z(A_col) ⊗ Z(B_row) lands in ZA[i, j] = A[i,p] * B[p,j].
    fn pack_for_tile(
        a: &[f32],
        b: &[f32],
        k: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut a_pack = vec![0.0f32; k * 16];
        let mut b_pack = vec![0.0f32; k * 16];
        for p in 0..k {
            for i in 0..16 {
                a_pack[p * 16 + i] = a[i * k + p];
            }
            for j in 0..16 {
                b_pack[p * 16 + j] = b[p * 16 + j];
            }
        }
        (a_pack, b_pack)
    }

    #[test]
    fn tile_16x16_set_correct() {
        if !crate::probe::scan().has_sme {
            return;
        }
        let m = 16usize;
        let n = 16usize;
        let k = 8usize;
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32) * 0.1).collect();
        let mut c_sme = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        let (a_pack, b_pack) = pack_for_tile(&a, &b, k);

        let stream = Stream::new().unwrap();
        unsafe {
            tile_16x16_f32(
                &stream,
                a_pack.as_ptr(),
                b_pack.as_ptr(),
                c_sme.as_mut_ptr(),
                n,
                k,
                false,
            );
        }
        drop(stream);

        ref_matmul(&a, &b, &mut c_ref, m, n, k);

        let mut max_err = 0.0f32;
        for i in 0..m * n {
            let e = (c_sme[i] - c_ref[i]).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(
            max_err < 1e-4,
            "tile mismatch: max_err={max_err}, c_sme[0..4]={:?}, c_ref[0..4]={:?}",
            &c_sme[..4],
            &c_ref[..4]
        );
    }

    #[test]
    fn tile_16x16_accumulate_correct() {
        if !crate::probe::scan().has_sme {
            return;
        }
        let m = 16usize;
        let n = 16usize;
        let k = 4usize;
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 5) as f32) * 0.2).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32) * 0.2).collect();
        let mut c_sme: Vec<f32> = (0..m * n).map(|i| (i % 3) as f32).collect();
        let mut c_ref: Vec<f32> = c_sme.clone();

        let (a_pack, b_pack) = pack_for_tile(&a, &b, k);

        let stream = Stream::new().unwrap();
        unsafe {
            tile_16x16_f32(
                &stream,
                a_pack.as_ptr(),
                b_pack.as_ptr(),
                c_sme.as_mut_ptr(),
                n,
                k,
                true,
            );
        }
        drop(stream);

        // Reference: c_ref += a * b
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                c_ref[i * n + j] += acc;
            }
        }

        let mut max_err = 0.0f32;
        for i in 0..m * n {
            let e = (c_sme[i] - c_ref[i]).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(max_err < 1e-4, "accumulate mismatch: max_err={max_err}");
    }
}
