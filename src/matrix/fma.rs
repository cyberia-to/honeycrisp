//! Typed operand builder for AMX FMA instructions.
//!
//! AMX fma32/fma16/fmabf16 take a single 64-bit operand word that
//! controls which registers participate and how. This module provides
//! a safe builder so you never touch raw bit offsets.
//!
//! # Bit layout (fma32, matrix mode)
//!
//! ```text
//! 63    mode        0 = matrix (outer product), 1 = vector
//! 29    skip_x      1 = treat X as zero
//! 28    skip_y      1 = treat Y as zero
//! 27    skip_z      1 = treat Z as zero (no accumulate)
//! 25:20 z_row       starting Z row (for f32, only bits 1:0 matter = tile select)
//! 18:10 x_offset    byte offset into 512-byte circular X buffer
//! 8:0   y_offset    byte offset into 512-byte circular Y buffer
//! ```
//!
//! For f32 outer product: `z[j*4 + (z_row&3)][i] += x[i] * y[j]`
//! where i,j ∈ 0..16. Writes 16 Z rows spaced 4 apart.

use super::regs::{XRow, YRow};

/// Builder for AMX FMA operand words.
///
/// Defaults: x=X0, y=Y0, z_tile=0, accumulate=true, matrix mode.
#[derive(Copy, Clone, Debug)]
pub struct FmaOp {
    bits: u64,
}

impl FmaOp {
    /// New FMA operand: accumulate into Z tile 0 using X0 × Y0.
    #[inline]
    pub const fn new() -> Self {
        Self { bits: 0 }
    }

    /// Select X register as source.
    #[inline]
    pub const fn x(mut self, row: XRow) -> Self {
        // x_offset = row * 64 bytes, goes into bits 18:10
        self.bits = (self.bits & !0x7FC00) | ((row.byte_offset()) << 10);
        self
    }

    /// Select Y register as source.
    #[inline]
    pub const fn y(mut self, row: YRow) -> Self {
        // y_offset = row * 64 bytes, goes into bits 8:0
        self.bits = (self.bits & !0x1FF) | row.byte_offset();
        self
    }

    /// Select Z tile (0..=3 for f32). Each tile is an independent 16×16 matrix.
    #[inline]
    pub const fn z_tile(mut self, tile: u8) -> Self {
        let t = (tile & 3) as u64;
        self.bits = (self.bits & !(0x3F << 20)) | (t << 20);
        self
    }

    /// First iteration: set skip_z to ignore current Z contents (pure multiply).
    #[inline]
    pub const fn no_accumulate(mut self) -> Self {
        self.bits |= 1 << 27;
        self
    }

    /// Accumulate: z += x * y (default).
    #[inline]
    pub const fn accumulate(mut self) -> Self {
        self.bits &= !(1 << 27);
        self
    }

    /// Vector mode instead of matrix (outer product) mode.
    #[inline]
    pub const fn vector_mode(mut self) -> Self {
        self.bits |= 1 << 63;
        self
    }

    /// Extract the raw 64-bit operand word.
    #[inline]
    pub const fn build(self) -> u64 {
        self.bits
    }
}

impl Default for FmaOp {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors for common GEMM patterns
// ---------------------------------------------------------------------------

/// First rank-1 update: Z[tile] = X[xr] × Y[yr] (no accumulate).
#[inline]
pub const fn fma_first(xr: XRow, yr: YRow, tile: u8) -> u64 {
    FmaOp::new()
        .x(xr)
        .y(yr)
        .z_tile(tile)
        .no_accumulate()
        .build()
}

/// Subsequent rank-1 update: Z[tile] += X[xr] × Y[yr].
#[inline]
pub const fn fma_acc(xr: XRow, yr: YRow, tile: u8) -> u64 {
    FmaOp::new().x(xr).y(yr).z_tile(tile).build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operand_first_rank1() {
        let op = fma_first(
            unsafe { XRow::new_unchecked(0) },
            unsafe { YRow::new_unchecked(0) },
            0,
        );
        // skip_z bit (27) should be set
        assert_ne!(op & (1 << 27), 0);
        // x_offset and y_offset should be 0
        assert_eq!(op & 0x7FC00, 0); // x_offset bits 18:10
        assert_eq!(op & 0x1FF, 0); // y_offset bits 8:0
    }

    #[test]
    fn operand_acc_xn_yn() {
        let op = fma_acc(
            unsafe { XRow::new_unchecked(3) },
            unsafe { YRow::new_unchecked(5) },
            0,
        );
        // skip_z should NOT be set
        assert_eq!(op & (1 << 27), 0);
        // x_offset = 3*64 = 192, in bits 18:10 = 192 << 10
        assert_eq!((op >> 10) & 0x1FF, 192);
        // y_offset = 5*64 = 320, in bits 8:0
        assert_eq!(op & 0x1FF, 320);
    }

    #[test]
    fn operand_tile_select() {
        let op = FmaOp::new().z_tile(2).build();
        assert_eq!((op >> 20) & 0x3F, 2);
    }
}
