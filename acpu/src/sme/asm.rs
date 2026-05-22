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
