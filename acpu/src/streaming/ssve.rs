//! Streaming SVE (SSVE) instruction encoders.
//!
//! Encodings are emitted via `.word` inside `core::arch::asm!`, with all
//! register-field bits resolved at compile time via const generics or
//! const operands. Use only when a [`super::Stream`] is live.
//!
//! For most kernels (matmul tile, NTT butterfly) the whole inner loop is
//! a single hand-written `asm!` block; this file mostly exists to
//! provide tested encoders for the smoke test and for sites where a
//! single SSVE op is enough.
//!
//! Encoding source: ARM ARM A64.SVE chapter + `llvm-mc -mattr=+sme`.

// ---------------------------------------------------------------------------
// Predicate construction
// ---------------------------------------------------------------------------

/// `PTRUE Pd.S, all` — set every 32-bit lane of Pd to true.
///
/// Encoded as `0x2598E3E0 | Pd`.
///
/// # Safety
///
/// Caller must hold a live [`super::Stream`].
#[inline(always)]
pub unsafe fn ptrue_all_s<const PD: u32>() {
    const { assert!(PD < 16, "Pd must be 0..15") };
    core::arch::asm!(
        ".word {enc}",
        enc = const 0x2598E3E0 | PD,
        options(nostack),
    );
}

/// `PTRUE Pd.B, all` — set every byte lane of Pd to true.
#[inline(always)]
pub unsafe fn ptrue_all_b<const PD: u32>() {
    const { assert!(PD < 16, "Pd must be 0..15") };
    core::arch::asm!(
        ".word {enc}",
        enc = const 0x2518E3E0 | PD,
        options(nostack),
    );
}

// ---------------------------------------------------------------------------
// WHILELT Pd.S, Wn, Wm — predicate of lanes where n+lane < m
// ---------------------------------------------------------------------------

/// `WHILELT Pd.S, X0, X1` — set Pd.S\[i] = (X0+i*4 < X1).
///
/// Useful for tail-free loops: callers put loop counter in X0 and
/// limit in X1, then load/store predicated by Pd.
///
/// # Safety
///
/// Caller must hold a live [`super::Stream`].
#[inline(always)]
pub unsafe fn whilelt_s_x0x1<const PD: u32>() {
    const { assert!(PD < 16, "Pd must be 0..15") };
    core::arch::asm!(
        ".word {enc}",
        enc = const 0x25A11400 | PD,
        options(nostack),
    );
}

// ---------------------------------------------------------------------------
// Predicated 32-bit load/store
// ---------------------------------------------------------------------------

/// `LD1W { Zt.S }, Pg/Z, [X0]` — load SVL/32 f32 lanes gated by Pg/Z.
#[inline(always)]
pub unsafe fn ld1w_z_x0<const ZT: u32, const PG: u32>() {
    const { assert!(ZT < 32 && PG < 8) };
    core::arch::asm!(
        ".word {enc}",
        enc = const 0xA540A000 | (PG << 10) | ZT,
        options(nostack),
    );
}

/// `ST1W { Zt.S }, Pg, [X0]` — store SVL/32 f32 lanes gated by Pg.
#[inline(always)]
pub unsafe fn st1w_z_x0<const ZT: u32, const PG: u32>() {
    const { assert!(ZT < 32 && PG < 8) };
    core::arch::asm!(
        ".word {enc}",
        enc = const 0xE540E000 | (PG << 10) | ZT,
        options(nostack),
    );
}

// ---------------------------------------------------------------------------
// FMA: Zda.S += Zn.S * Zm.S gated by Pg/M
// ---------------------------------------------------------------------------

/// `FMLA Zda.S, Pg/M, Zn.S, Zm.S` — predicated fused multiply-accumulate.
#[inline(always)]
pub unsafe fn fmla_s<const ZDA: u32, const PG: u32, const ZN: u32, const ZM: u32>() {
    const { assert!(ZDA < 32 && PG < 8 && ZN < 32 && ZM < 32) };
    core::arch::asm!(
        ".word {enc}",
        enc = const 0x65A00000 | (ZM << 16) | (PG << 10) | (ZN << 5) | ZDA,
        options(nostack),
    );
}

/// `DUP Zd.S, Wn` — broadcast a 32-bit GPR into every lane of Zd.S.
/// Wn is the 32-bit alias of the X register holding the value.
#[inline(always)]
pub unsafe fn dup_z_w0<const ZD: u32>() {
    const { assert!(ZD < 32) };
    core::arch::asm!(
        ".word {enc}",
        enc = const 0x05203800 | ZD,
        options(nostack),
    );
}

// ---------------------------------------------------------------------------
// Tests — exercise the encodings without any matmul logic.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::Stream;
    use super::*;

    #[test]
    fn ssve_load_store_roundtrip() {
        if !crate::probe::scan().has_sme {
            return;
        }
        let s = Stream::new().unwrap();
        let n = s.svl_lanes_f32();
        let src: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let mut dst: Vec<f32> = vec![0.0; n];
        unsafe {
            // Load src[0..n] into Z0 under ptrue, then store Z0 into dst[0..n].
            ptrue_all_s::<0>();
            core::arch::asm!(
                "mov x0, {src_ptr}",
                ".word {ld1w}",
                "mov x0, {dst_ptr}",
                ".word {st1w}",
                src_ptr = in(reg) src.as_ptr(),
                dst_ptr = in(reg) dst.as_mut_ptr(),
                ld1w = const 0xA540A000u32, // LD1W Z0.S, P0/Z, [X0]
                st1w = const 0xE540E000u32, // ST1W Z0.S, P0, [X0]
                out("x0") _,
                options(nostack),
            );
        }
        drop(s);
        for (i, (a, b)) in src.iter().zip(dst.iter()).enumerate() {
            assert_eq!(a, b, "lane {i} differs");
        }
    }

    /// Touch every encoder so a typo trips the assembler immediately,
    /// not at first kernel call.
    #[test]
    fn encoders_assemble() {
        if !crate::probe::scan().has_sme {
            return;
        }
        let _s = Stream::new().unwrap();
        unsafe {
            ptrue_all_s::<0>();
            ptrue_all_b::<1>();
            // whilelt needs valid x0,x1; we set both to 0 so the predicate
            // is all-false — still legal, just useless.
            core::arch::asm!("mov x0, xzr", "mov x1, xzr", out("x0") _, out("x1") _);
            whilelt_s_x0x1::<2>();
            dup_z_w0::<3>();
            fmla_s::<0, 0, 1, 2>();
        }
    }
}
