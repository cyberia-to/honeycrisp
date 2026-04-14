//! Safe(ish) AMX operations that require a live [`Matrix`].
//!
//! Every function here takes `&self` on `Matrix` so the borrow checker
//! guarantees the coprocessor is active for the duration of the call.

use super::asm::{
    self, OP_FMA16, OP_FMA32, OP_LDX, OP_LDY, OP_LDZ, OP_LDZI, OP_MAC16, OP_STX, OP_STY, OP_STZ,
};
use super::regs::{XRow, YRow, ZRow};
use super::Matrix;

// ---------------------------------------------------------------------------
// Operand packing helpers
// ---------------------------------------------------------------------------

/// Pack a pointer + row index into the operand format expected by
/// AMX load/store instructions: `ptr | (row << 56)`.
#[inline(always)]
fn load_store_operand(ptr: *const u8, row: u8) -> u64 {
    (ptr as u64) | ((row as u64) << 56)
}

// ---------------------------------------------------------------------------
// Load operations
// ---------------------------------------------------------------------------

impl Matrix {
    /// Load 64 bytes from `src` into X-register row `row`.
    ///
    /// # Safety
    ///
    /// `src` must point to at least 64 readable, 64-byte-aligned bytes.
    #[inline]
    pub unsafe fn ldx(&self, src: *const u8, row: XRow) {
        let operand = load_store_operand(src, row.index());
        asm::amx_op::<OP_LDX>(operand);
    }

    /// Load 64 bytes from `src` into Y-register row `row`.
    ///
    /// # Safety
    ///
    /// `src` must point to at least 64 readable, 64-byte-aligned bytes.
    #[inline]
    pub unsafe fn ldy(&self, src: *const u8, row: YRow) {
        let operand = load_store_operand(src, row.index());
        asm::amx_op::<OP_LDY>(operand);
    }

    /// Load 64 bytes from `src` into Z-register row `row`.
    ///
    /// # Safety
    ///
    /// `src` must point to at least 64 readable, 64-byte-aligned bytes.
    #[inline]
    pub unsafe fn ldz(&self, src: *const u8, row: ZRow) {
        let operand = load_store_operand(src, row.index());
        asm::amx_op::<OP_LDZ>(operand);
    }

    /// Load 64 bytes from `src` into Z-register row `row` (interleaved).
    ///
    /// # Safety
    ///
    /// `src` must point to at least 64 readable, 64-byte-aligned bytes.
    #[inline]
    pub unsafe fn ldzi(&self, src: *const u8, row: ZRow) {
        let operand = load_store_operand(src, row.index());
        asm::amx_op::<OP_LDZI>(operand);
    }

    // -----------------------------------------------------------------------
    // Store operations
    // -----------------------------------------------------------------------

    /// Store 64 bytes from X-register row `row` into `dst`.
    ///
    /// # Safety
    ///
    /// `dst` must point to at least 64 writable, 64-byte-aligned bytes.
    #[inline]
    pub unsafe fn stx(&self, dst: *mut u8, row: XRow) {
        let operand = load_store_operand(dst, row.index());
        asm::amx_op::<OP_STX>(operand);
    }

    /// Store 64 bytes from Y-register row `row` into `dst`.
    ///
    /// # Safety
    ///
    /// `dst` must point to at least 64 writable, 64-byte-aligned bytes.
    #[inline]
    pub unsafe fn sty(&self, dst: *mut u8, row: YRow) {
        let operand = load_store_operand(dst, row.index());
        asm::amx_op::<OP_STY>(operand);
    }

    /// Store 64 bytes from Z-register row `row` into `dst`.
    ///
    /// # Safety
    ///
    /// `dst` must point to at least 64 writable, 64-byte-aligned bytes.
    #[inline]
    pub unsafe fn stz(&self, dst: *mut u8, row: ZRow) {
        let operand = load_store_operand(dst, row.index());
        asm::amx_op::<OP_STZ>(operand);
    }

    // -----------------------------------------------------------------------
    // Compute operations
    // -----------------------------------------------------------------------

    /// Fused multiply-accumulate on f32 lanes.
    ///
    /// The `operand` word is a bit-packed configuration controlling
    /// which X row, Y row, and Z row participate, along with lane
    /// masks and accumulation flags. See corsix/amx for the encoding.
    ///
    /// # Safety
    ///
    /// `operand` must be a validly packed FMA32 operand word.
    #[inline]
    pub unsafe fn fma32(&self, operand: u64) {
        asm::amx_op::<OP_FMA32>(operand);
    }

    /// Fused multiply-accumulate on f16 lanes.
    ///
    /// # Safety
    ///
    /// `operand` must be a validly packed FMA16 operand word.
    #[inline]
    pub unsafe fn fma16(&self, operand: u64) {
        asm::amx_op::<OP_FMA16>(operand);
    }

    /// Multiply-accumulate on i16 lanes.
    ///
    /// # Safety
    ///
    /// `operand` must be a validly packed MAC16 operand word.
    #[inline]
    pub unsafe fn mac16(&self, operand: u64) {
        asm::amx_op::<OP_MAC16>(operand);
    }
}
