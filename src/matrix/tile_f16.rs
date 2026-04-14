//! AMX fp16 tile operations for HGEMM microkernels.
//!
//! FMA16: X[32 fp16] × Y[32 fp16] → Z[32×32 fp16] outer product.
//! 2048 FLOPs per instruction (4x of FMA32's 512).
//!
//! Z tile layout for fp16:
//!   64 rows × 64 bytes = 64 rows × 32 fp16 values
//!   Tile 0: rows {0, 2, 4, ..., 62} (32 rows)
//!   Tile 1: rows {1, 3, 5, ..., 63} (32 rows)
//!   Each tile: 32×32 fp16 matrix

use super::asm::{amx_op, OP_FMA16, OP_LDX, OP_LDY, OP_LDZ, OP_STZ};
use super::regs::{XRow, YRow};

// FMA16 operand encoding:
// Same layout as FMA32 but tile = bit 20 (0 or 1, not 0-3)
// bit 27: skip_z (no accumulate when set)

const fn fma16_first(xr: XRow, yr: YRow, tile: u8) -> u64 {
    let x_off = (xr.byte_offset() as u64) << 10;
    let y_off = yr.byte_offset() as u64;
    let z = (tile as u64 & 1) << 20;
    x_off | y_off | z | (1 << 27) // skip_z = no accumulate (overwrite)
}

const fn fma16_acc(xr: XRow, yr: YRow, tile: u8) -> u64 {
    let x_off = (xr.byte_offset() as u64) << 10;
    let y_off = yr.byte_offset() as u64;
    let z = (tile as u64 & 1) << 20;
    x_off | y_off | z // accumulate
}

// ---------------------------------------------------------------------------
// Z tile preload / store for fp16
// ---------------------------------------------------------------------------

/// Load C[32×32] fp16 into Z tile via LDZ.
///
/// # Safety
/// AMX must be active. `c` must point to 32 rows × `ldc` readable fp16 elements.
/// `ldc` is in fp16 elements (not bytes).
#[inline]
pub unsafe fn preload_c_f16(c: *const u16, ldc: usize, tile: u8) {
    for j in 0u8..32 {
        let z_row = j * 2 + (tile & 1);
        let c_addr = (c as *const u8).add(j as usize * ldc * 2);
        amx_op::<OP_LDZ>((c_addr as u64) | ((z_row as u64) << 56));
    }
}

/// Store Z tile to C[32×32] fp16 via STZ.
///
/// # Safety
/// AMX must be active. `c` must point to 32 rows × `ldc` writable fp16 elements.
#[inline]
pub unsafe fn store_c_f16(c: *mut u16, ldc: usize, tile: u8) {
    for j in 0u8..32 {
        let z_row = j * 2 + (tile & 1);
        let c_addr = (c as *mut u8).add(j as usize * ldc * 2);
        amx_op::<OP_STZ>((c_addr as u64) | ((z_row as u64) << 56));
    }
}

// ---------------------------------------------------------------------------
// 32×32 fp16 microkernel (single tile)
// ---------------------------------------------------------------------------

/// AMX 32×32 fp16 microkernel. Z tile 0. 2048 FLOPs/instruction.
///
/// a_panel: packed fp16 [k × 32] (each k-step = 64 bytes = 32 fp16)
/// b_panel: packed fp16 [k × 32] (each k-step = 64 bytes = 32 fp16)
///
/// Computes: Z[32×32] += sum_p( A[p] ⊗ B[p] )
///
/// # Safety
/// AMX must be active. Panels must be 64-byte aligned with `k*64` readable bytes.
#[inline]
pub unsafe fn microkernel_32x32_f16(a_panel: *const u8, b_panel: *const u8, k: usize) {
    let mut first = true;
    let mut p = 0usize;

    // 8-unrolled inner loop: 8 LDX + 8 LDY + 8 FMA16 = 24 ops, 16384 FLOPs
    while p + 8 <= k {
        for i in 0u8..8 {
            amx_op::<OP_LDX>((b_panel.add((p + i as usize) * 64) as u64) | ((i as u64) << 56));
            amx_op::<OP_LDY>((a_panel.add((p + i as usize) * 64) as u64) | ((i as u64) << 56));
        }
        if first {
            amx_op::<OP_FMA16>(fma16_first(
                XRow::new_unchecked(0),
                YRow::new_unchecked(0),
                0,
            ));
            first = false;
        } else {
            amx_op::<OP_FMA16>(fma16_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
        }
        for i in 1u8..8 {
            amx_op::<OP_FMA16>(fma16_acc(XRow::new_unchecked(i), YRow::new_unchecked(i), 0));
        }
        p += 8;
    }

    // Scalar tail
    while p < k {
        amx_op::<OP_LDX>(b_panel.add(p * 64) as u64);
        amx_op::<OP_LDY>(a_panel.add(p * 64) as u64);
        if first {
            amx_op::<OP_FMA16>(fma16_first(
                XRow::new_unchecked(0),
                YRow::new_unchecked(0),
                0,
            ));
            first = false;
        } else {
            amx_op::<OP_FMA16>(fma16_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
        }
        p += 1;
    }
}

/// Accumulate Z tile 0 (fp16 32×32) into f32 output buffer.
///
/// Reads Z via STZ into temp buffer, converts fp16→f32 via inline asm, adds to C.
///
/// # Safety
/// AMX must be active. `c` must point to 32 rows × `ldc` writable f32 elements.
#[inline]
pub unsafe fn accumulate_tile_f16_to_f32(c: *mut f32, ldc: usize) {
    let mut tmp = [0u16; 32 * 32];
    store_c_f16(tmp.as_mut_ptr(), 32, 0);

    for j in 0..32 {
        let dst = c.add(j * ldc);
        let src = tmp.as_ptr().add(j * 32);
        let mut i = 0;
        while i + 8 <= 32 {
            // fcvtl/fcvtl2: convert 8 fp16 → 2×4 f32, add to C
            core::arch::asm!(
                "ldr q0, [{src}]",          // 8 × fp16
                "fcvtl v1.4s, v0.4h",       // low 4 → f32
                "fcvtl2 v2.4s, v0.8h",      // high 4 → f32
                "ldp q3, q4, [{dst}]",      // 8 × f32 from C
                "fadd v3.4s, v3.4s, v1.4s",
                "fadd v4.4s, v4.4s, v2.4s",
                "stp q3, q4, [{dst}]",      // store back
                src = in(reg) src.add(i),
                dst = in(reg) dst.add(i),
                out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
            );
            i += 8;
        }
    }
}

/// Store Z tile 0 (fp16 32×32) directly to f32 buffer (overwrite, not accumulate).
///
/// # Safety
/// AMX must be active. `c` must point to 32 rows × `ldc` writable f32 elements.
#[inline]
pub unsafe fn store_tile_f16_to_f32(c: *mut f32, ldc: usize) {
    let mut tmp = [0u16; 32 * 32];
    store_c_f16(tmp.as_mut_ptr(), 32, 0);

    for j in 0..32 {
        let dst = c.add(j * ldc);
        let src = tmp.as_ptr().add(j * 32);
        let mut i = 0;
        while i + 8 <= 32 {
            core::arch::asm!(
                "ldr q0, [{src}]",
                "fcvtl v1.4s, v0.4h",
                "fcvtl2 v2.4s, v0.8h",
                "stp q1, q2, [{dst}]",
                src = in(reg) src.add(i),
                dst = in(reg) dst.add(i),
                out("v0") _, out("v1") _, out("v2") _,
            );
            i += 8;
        }
    }
}
