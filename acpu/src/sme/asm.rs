//! Raw SME instruction encoders.
//!
//! Encodings verified against `llvm-mc -mattr=+sme,+sme2 -show-encoding`.
//! Field layouts (least-significant bit first):
//!
//! | inst | bits 4:0 | bits 9:5 | bits 12:10 | bits 14:13 | bits 15:13 | bits 19:16 |
//! |------|----------|----------|-----------|-----------|------------|------------|
//! | FMOPA.S | ZAd[2:0] | Zn | Pn | — | Pm | Zm |
//! | MOVA Z from ZAh.S | Zd | — (offset slot) | Pg | W_base | tile (bit 7) | — |
//! | MOVA ZAh.S from Z | offset[1:0] (bits 1:0) ; Zn(9:5) | — | Pg | W_base | tile (bit 4) | — |
//!
//! The exact field positions for MOVA differ between Z←ZA and ZA←Z forms;
//! encode-side helpers below capture the correct shifts.
//!
//! ## Integer outer-product (SMOPA/UMOPA INT16)
//!
//! Two widths are exposed:
//!
//! - INT16 → INT32 (`smopa_int16_s`, `umopa_int16_s`): requires FEAT_SME2.
//!   Per ARM ARM C7.2.342 (SMOPA, signed integer 2-way) the operation is
//!   `ZA.S[i][j] += Zn.H[2i+k] * Zm.H[2j+k]` summed over k ∈ 0..1, i.e.
//!   a 2-way INT16 dot-product per ZA cell. ZA tile is 16×16 INT32 on
//!   SVL=512 (FEAT_SME requires SVL ≥ 128, but every shipping Apple SME
//!   implementation has SVL=512, so the tile geometry is fixed at 16×16).
//!   Encoding base = `0xA080_0008`. Variants: bit 24 = unsigned-on-Zn flag,
//!   bit 21 (which is part of the `0x81` constant) = unsigned-on-Zm flag.
//!
//! - INT16 → INT64 (`smopa_int16_d`, `umopa_int16_d`): requires FEAT_SME +
//!   FEAT_SME_I16I64 (M4 Max has both). Per ARM ARM C7.2.343 the operation
//!   is `ZA.D[i][j] += Zn.H[4i+k] * Zm.H[4j+k]` summed over k ∈ 0..3, i.e.
//!   a 4-way INT16 dot-product per ZA cell. ZA tile is 8×8 INT64 on
//!   SVL=512 (eight tiles ZA0.D..ZA7.D). Encoding base = `0xA0C0_0000`
//!   for SMOPA, `0xA1E0_0000` for UMOPA. Bit 3 = 0 selects .D form.
//!
//! Encodings reproduced for the test suite via `encoders_int16_assemble`
//! below.

/// FMOPA Zad.S += outer(Zn.S, Zm.S) gated by (Pn/M, Pm/M).
///
/// `ZAD` ∈ 0..3, `ZN` ∈ 0..31, `ZM` ∈ 0..15 (FMOPA restricts Zm to 4 bits),
/// `PN` ∈ 0..7, `PM` ∈ 0..7.
///
/// # Safety
///
/// Caller must hold a live [`crate::streaming::Stream`].
#[inline(always)]
pub unsafe fn fmopa_s<
    const ZAD: u32,
    const ZN: u32,
    const ZM: u32,
    const PN: u32,
    const PM: u32,
>() {
    const {
        assert!(ZAD < 4, "FMOPA.S ZAd must be 0..3");
        assert!(ZN < 32 && ZM < 16);
        assert!(PN < 8 && PM < 8);
    }
    core::arch::asm!(
        ".word {enc}",
        enc = const 0x80800000 | (ZM << 16) | (PM << 13) | (PN << 10) | (ZN << 5) | ZAD,
        options(nostack),
    );
}

/// MOVA Zd.S = ZAs.H.S\[W_base + offset\] gated by Pg/M.
///
/// Reads one horizontal slice of ZA tile `ZAS` into a Z register.
/// `OFFSET` ∈ 0..3, `ZAS` ∈ 0..3 (f32 tiles), `WBASE` ∈ 0..3
/// (corresponds to W12..W15 — W12 is the base register, +0..3 maps to
/// W12..W15). Pg is the predicate register Z lanes are gated by.
///
/// To address all 16 rows of a tile, set `WBASE=0`, OFFSET=0, and write
/// the desired row number into the actual W12 register before the call.
#[inline(always)]
pub unsafe fn mova_z_from_za_h_s<
    const ZD: u32,
    const PG: u32,
    const ZAS: u32,
    const WBASE: u32,
    const OFFSET: u32,
>() {
    const {
        assert!(ZD < 32 && PG < 8 && ZAS < 4 && WBASE < 4 && OFFSET < 4);
    }
    core::arch::asm!(
        ".word {enc}",
        enc = const 0xC0820000
            | ((ZAS & 0x3) << 7)
            | (WBASE << 13)
            | (PG << 10)
            | ((OFFSET & 0x3) << 5)
            | ZD,
        options(nostack),
    );
}

/// MOVA ZAd.H.S\[W_base + offset\] = Zn.S gated by Pg/M.
///
/// Writes one Z register into one horizontal slice of ZA tile `ZAD`.
#[inline(always)]
pub unsafe fn mova_za_h_from_z_s<
    const ZAD: u32,
    const WBASE: u32,
    const OFFSET: u32,
    const PG: u32,
    const ZN: u32,
>() {
    const {
        assert!(ZAD < 4 && WBASE < 4 && OFFSET < 4 && PG < 8 && ZN < 32);
    }
    core::arch::asm!(
        ".word {enc}",
        enc = const 0xC0800000
            | ((ZAD & 0x3) << 3)
            | (WBASE << 13)
            | (PG << 10)
            | (ZN << 5)
            | (OFFSET & 0x3),
        options(nostack),
    );
}

// ---------------------------------------------------------------------------
// Integer outer-product: SMOPA / UMOPA, INT16 operands
// ---------------------------------------------------------------------------

/// SMOPA `ZAd.S, Pn/M, Pm/M, Zn.H, Zm.H` — signed 2-way INT16→INT32 outer
/// product. Per ZA cell `(i, j)` accumulates `Zn.H[2i] * Zm.H[2j] +
/// Zn.H[2i+1] * Zm.H[2j+1]` into `ZA.S[i][j]`. Requires FEAT_SME2.
///
/// `ZAD` ∈ 0..3 (four INT32 tiles), `ZN`, `ZM` ∈ 0..31, `PN`, `PM` ∈ 0..7.
///
/// # Safety
///
/// Caller must hold a live [`crate::streaming::Stream`] and the host must
/// advertise `Feature::Sme2`.
#[inline(always)]
pub unsafe fn smopa_int16_s<
    const ZAD: u32,
    const ZN: u32,
    const ZM: u32,
    const PN: u32,
    const PM: u32,
>() {
    const {
        assert!(ZAD < 4, "SMOPA.S ZAd must be 0..3");
        assert!(ZN < 32 && ZM < 32);
        assert!(PN < 8 && PM < 8);
    }
    core::arch::asm!(
        ".word {enc}",
        enc = const 0xA080_0008 | (ZM << 16) | (PM << 13) | (PN << 10) | (ZN << 5) | ZAD,
        options(nostack),
    );
}

/// UMOPA `ZAd.S, Pn/M, Pm/M, Zn.H, Zm.H` — unsigned 2-way INT16→INT32 outer
/// product. Same shape as [`smopa_int16_s`] but treats both vectors as
/// unsigned. Requires FEAT_SME2.
#[inline(always)]
pub unsafe fn umopa_int16_s<
    const ZAD: u32,
    const ZN: u32,
    const ZM: u32,
    const PN: u32,
    const PM: u32,
>() {
    const {
        assert!(ZAD < 4, "UMOPA.S ZAd must be 0..3");
        assert!(ZN < 32 && ZM < 32);
        assert!(PN < 8 && PM < 8);
    }
    core::arch::asm!(
        ".word {enc}",
        enc = const 0xA180_0008 | (ZM << 16) | (PM << 13) | (PN << 10) | (ZN << 5) | ZAD,
        options(nostack),
    );
}

/// SMOPA `ZAd.D, Pn/M, Pm/M, Zn.H, Zm.H` — signed 4-way INT16→INT64 outer
/// product. Per ZA cell `(i, j)` accumulates the sum over k ∈ 0..3 of
/// `Zn.H[4i+k] * Zm.H[4j+k]` into `ZA.D[i][j]`. Requires FEAT_SME_I16I64.
///
/// `ZAD` ∈ 0..7 (eight INT64 tiles), `ZN`, `ZM` ∈ 0..31, `PN`, `PM` ∈ 0..7.
///
/// # Safety
///
/// Caller must hold a live [`crate::streaming::Stream`] and the host must
/// advertise `Feature::SmeI16I64`.
#[inline(always)]
pub unsafe fn smopa_int16_d<
    const ZAD: u32,
    const ZN: u32,
    const ZM: u32,
    const PN: u32,
    const PM: u32,
>() {
    const {
        assert!(ZAD < 8, "SMOPA.D ZAd must be 0..7");
        assert!(ZN < 32 && ZM < 32);
        assert!(PN < 8 && PM < 8);
    }
    core::arch::asm!(
        ".word {enc}",
        enc = const 0xA0C0_0000 | (ZM << 16) | (PM << 13) | (PN << 10) | (ZN << 5) | ZAD,
        options(nostack),
    );
}

/// UMOPA `ZAd.D, Pn/M, Pm/M, Zn.H, Zm.H` — unsigned 4-way INT16→INT64 outer
/// product. Same shape as [`smopa_int16_d`] but treats both vectors as
/// unsigned. Requires FEAT_SME_I16I64.
#[inline(always)]
pub unsafe fn umopa_int16_d<
    const ZAD: u32,
    const ZN: u32,
    const ZM: u32,
    const PN: u32,
    const PM: u32,
>() {
    const {
        assert!(ZAD < 8, "UMOPA.D ZAd must be 0..7");
        assert!(ZN < 32 && ZM < 32);
        assert!(PN < 8 && PM < 8);
    }
    core::arch::asm!(
        ".word {enc}",
        enc = const 0xA1E0_0000 | (ZM << 16) | (PM << 13) | (PN << 10) | (ZN << 5) | ZAD,
        options(nostack),
    );
}

// ---------------------------------------------------------------------------
// MOVA — extract a horizontal slice of an INT64 ZA tile into a Z register
// ---------------------------------------------------------------------------

/// MOVA `Zd.D, Pg/M, ZAs.H.D[W_base + offset]` — read one horizontal slice
/// of INT64 ZA tile `ZAS` into a Z register.
///
/// `ZD` ∈ 0..31, `PG` ∈ 0..7, `ZAS` ∈ 0..7 (INT64 tiles), `WBASE` ∈ 0..3
/// (W12..W15 — at most one register-relative offset slot is used), and
/// `OFFSET` ∈ 0..0 (only `[W12, 0]` is encoded here; rows are addressed by
/// writing the actual W12 register before the call).
///
/// Encoding base = `0xC0C2_0000`. `ZAS[2:0]` lives in bits 6..8.
#[inline(always)]
pub unsafe fn mova_z_from_za_h_d<const ZD: u32, const PG: u32, const ZAS: u32, const WBASE: u32>() {
    const {
        assert!(ZD < 32 && PG < 8 && ZAS < 8 && WBASE < 4);
    }
    core::arch::asm!(
        ".word {enc}",
        enc = const 0xC0C2_0000
            | ((ZAS & 0x7) << 6)
            | (WBASE << 13)
            | (PG << 10)
            | ZD,
        options(nostack),
    );
}

// ---------------------------------------------------------------------------
// Tests — touch each encoder so a typo trips the assembler immediately.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::Stream;

    #[test]
    fn encoders_int16_assemble() {
        if !crate::probe::scan().has_sme {
            eprintln!("skip: FEAT_SME not present");
            return;
        }
        let _s = Stream::new().unwrap();
        unsafe {
            // INT16 → INT32 (FEAT_SME2)
            smopa_int16_s::<0, 0, 1, 0, 0>();
            smopa_int16_s::<3, 31, 31, 7, 7>();
            umopa_int16_s::<0, 0, 1, 0, 0>();
            umopa_int16_s::<2, 5, 6, 1, 2>();

            // INT16 → INT64 (FEAT_SME_I16I64). Encoder still runs even if
            // FEAT_I16I64 is absent — the instruction will UNDEF, so guard
            // with a probe check before exercising on non-M4 silicon.
            if crate::probe::scan().has_sme_i16i64 {
                smopa_int16_d::<0, 0, 1, 0, 0>();
                smopa_int16_d::<7, 31, 31, 7, 7>();
                umopa_int16_d::<0, 0, 1, 0, 0>();
                mova_z_from_za_h_d::<0, 0, 0, 0>();
                mova_z_from_za_h_d::<31, 7, 7, 0>();
            }
        }
    }
}
