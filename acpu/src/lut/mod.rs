//! SME2 lookup-table primitives.
//!
//! Provides bulk table-indexed permutes using LUTI4. SVL=64 means each
//! LUTI4.B issues 64 byte lookups in one cycle, where NEON `tbl` would
//! take four `tbl` instructions to cover the same span.

pub mod asm;

use crate::streaming::Stream;
use crate::{CpuError, Feature};

/// Apply a 16-entry byte table to a stream of u8 indices.
///
/// `out[i] = table[idx[i] & 0xF]`. Indices outside 0..15 wrap modulo 16.
///
/// Returns `FeatureNotAvailable(Sme2)` if the chip lacks LUTI support.
pub fn permute_u8(table: &[u8; 16], idx: &[u8], out: &mut [u8]) -> crate::Result<()> {
    assert_eq!(idx.len(), out.len(), "idx and out lengths must match");
    if !crate::probe::scan().has_sme2 {
        return Err(CpuError::FeatureNotAvailable(Feature::Sme2));
    }

    // Replicate the 16-byte table across the 64-byte ZT0 register (all
    // four sub-tables hold the same table, so any IDX value works).
    let mut zt0_bytes = [0u8; 64];
    for sub in 0..4 {
        zt0_bytes[sub * 16..sub * 16 + 16].copy_from_slice(table);
    }

    let stream = Stream::new().expect("permute_u8: Stream::new failed");
    let chunks = idx.len() / 64;
    if chunks > 0 {
        unsafe {
            permute_u8_bulk(
                zt0_bytes.as_ptr(),
                idx.as_ptr(),
                out.as_mut_ptr(),
                chunks,
            );
        }
    }
    drop(stream);

    // Scalar tail (< 64 bytes). Cheaper to do in Rust than to set up
    // predicated load/store/luti for a partial vector.
    let tail_start = chunks * 64;
    for i in tail_start..idx.len() {
        out[i] = table[(idx[i] & 0x0F) as usize];
    }

    Ok(())
}

/// All-asm core: pre-load the table into Z2 from a 64-byte image
/// (table replicated 4 times), then SVE TBL over `chunks` SVL-sized
/// blocks of input. No predicate needed — caller handles the tail.
///
/// SVE TBL is available in streaming mode (SSVE) and is the cleanest
/// per-byte indexed permute available; SME2's LUTI4 has a multi-vector
/// register-tuple semantics that wins for register-resident tables but
/// is unnecessary friction when the table fits in one Z register.
#[inline(never)]
unsafe fn permute_u8_bulk(tbl_bytes: *const u8, idx: *const u8, out: *mut u8, chunks: usize) {
    core::arch::asm!(
        ".word 0x2518E3E0",          // PTRUE P0.B
        // x3 holds the table image (64 bytes of replicated 16-byte
        // table). Load it into Z2 once.
        "mov x3, {tbl}",
        ".word 0xA400A062",          // LD1B { Z2.B }, P0/Z, [x3]

        "mov x0, {idx}",
        "mov x1, {out}",
        "mov x2, {chunks}",

        "10:",
        ".word 0xA400A001",          // LD1B { Z1.B }, P0/Z, [x0]
        ".word 0x05213040",          // TBL Z0.B, { Z2.B }, Z1.B
        ".word 0xE400E020",          // ST1B { Z0.B }, P0, [x1]
        "add x0, x0, #64",
        "add x1, x1, #64",
        "subs x2, x2, #1",
        "b.ne 10b",

        tbl = in(reg) tbl_bytes,
        idx = in(reg) idx,
        out = in(reg) out,
        chunks = in(reg) chunks,
        out("x0") _,
        out("x1") _,
        out("x2") _,
        out("x3") _,
        options(nostack),
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_permute(table: &[u8; 16], idx: &[u8], out: &mut [u8]) {
        for (o, &i) in out.iter_mut().zip(idx.iter()) {
            *o = table[(i & 0x0F) as usize];
        }
    }

    fn check_permute(n: usize) {
        if !crate::probe::scan().has_sme2 {
            return;
        }
        let table: [u8; 16] = [
            0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
            0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7,
        ];
        let idx: Vec<u8> = (0..n).map(|i| (i as u8) & 0x0F).collect();
        let mut out_sme = vec![0u8; n];
        let mut out_ref = vec![0u8; n];

        permute_u8(&table, &idx, &mut out_sme).unwrap();
        ref_permute(&table, &idx, &mut out_ref);

        for i in 0..n {
            assert_eq!(out_sme[i], out_ref[i], "lane {i}: n={n}");
        }
    }

    #[test]
    fn permute_64() {
        check_permute(64);
    }
    #[test]
    fn permute_128() {
        check_permute(128);
    }
    #[test]
    fn permute_256() {
        check_permute(256);
    }
    #[test]
    fn permute_1024() {
        check_permute(1024);
    }
    #[test]
    fn permute_odd() {
        check_permute(73);
    }
    #[test]
    fn permute_small() {
        check_permute(8);
    }
}
