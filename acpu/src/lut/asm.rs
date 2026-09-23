//! SME2 LUTI2 / LUTI4 + ZT0 load/store encoders.
//!
//! ZT0 is a single SVL-wide table register introduced by SME2. LUTI2
//! indexes ZT0 by 2-bit lane indices, LUTI4 by 4-bit indices. With
//! SVL=64 bytes the table holds 4 sub-tables of 16 entries each for
//! the B variant (or 4 sub-tables of 4 S-entries for LUTI4.S).
//!
//! All instructions require a live [`crate::streaming::Stream`].
//! Encodings verified against `llvm-mc -mattr=+sme2 -show-encoding`.

/// `LDR ZT0, [Rn]` — load 64 bytes from \[Rn\] into the SME2 table register.
#[inline(always)]
pub unsafe fn ldr_zt0_x0() {
    // ldr zt0, [x0] = 0xE11F8000  (Rn in bits 9:5; Rn=0 here)
    core::arch::asm!(".word 0xE11F8000", options(nostack));
}

/// `STR ZT0, [Rn]` — store 64 bytes of ZT0 into \[Rn\].
#[inline(always)]
pub unsafe fn str_zt0_x0() {
    // str zt0, [x0] = 0xE13F8000
    core::arch::asm!(".word 0xE13F8000", options(nostack));
}

/// `ZERO {ZT0}` — clear ZT0.
#[inline(always)]
pub unsafe fn zero_zt0() {
    core::arch::asm!(".word 0xC0480001", options(nostack));
}

/// `LUTI4 Zd.B, ZT0, Zn[idx]` — for each byte lane i,
/// `Zd.B[i] = ZT0_sub[idx][ Zn.B[i] & 0xF ]` where `ZT0_sub[idx]` is
/// the `idx`-th 16-byte sub-table of ZT0.
///
/// IDX ∈ 0..3, Zd, Zn ∈ 0..31. Base encoding: `0xC0CA0000`. The lane
/// width is selected by bits 13:12 (`00` = .B). The `idx` field lives
/// in bits 15:14.
#[inline(always)]
pub unsafe fn luti4_b<const ZD: u32, const ZN: u32, const IDX: u32>() {
    const { assert!(ZD < 32 && ZN < 32 && IDX < 4) };
    core::arch::asm!(
        ".word {enc}",
        enc = const 0xC0CA0000 | (IDX << 14) | (ZN << 5) | ZD,
        options(nostack),
    );
}

/// `LUTI4 Zd.S, ZT0, Zn[idx]` — 32-bit-lane version. `IDX` ∈ 0..1
/// (table holds 2 sub-tables of 16 S-entries at SVL=64).
#[inline(always)]
pub unsafe fn luti4_s<const ZD: u32, const ZN: u32, const IDX: u32>() {
    const { assert!(ZD < 32 && ZN < 32 && IDX < 2) };
    core::arch::asm!(
        ".word {enc}",
        enc = const 0xC0CA2000 | (IDX << 14) | (ZN << 5) | ZD,
        options(nostack),
    );
}

/// `LUTI2 Zd.B, ZT0, Zn[idx]` — 2-bit-index byte lookup. `IDX` ∈ 0..7
/// (table holds 8 sub-tables of 4 B-entries at SVL=64).
#[inline(always)]
pub unsafe fn luti2_b<const ZD: u32, const ZN: u32, const IDX: u32>() {
    const { assert!(ZD < 32 && ZN < 32 && IDX < 8) };
    core::arch::asm!(
        ".word {enc}",
        enc = const 0xC0CC0000 | (IDX << 14) | (ZN << 5) | ZD,
        options(nostack),
    );
}
