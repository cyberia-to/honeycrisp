//! Typed AMX register row indices.
//!
//! X and Y: 8 rows × 64 bytes each (512 bytes per bank).
//! Z: 64 rows × 64 bytes = 4096 bytes total. For f32, there are
//! 4 independent 16×16 tiles selected by `z_row & 3`.

use crate::CpuError;
use core::fmt;

// ---------------------------------------------------------------------------
// XRow (0..=7)
// ---------------------------------------------------------------------------

/// Index of an X-register row (0..=7).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct XRow(u8);

impl XRow {
    #[inline]
    pub fn new(index: u8) -> crate::Result<Self> {
        if index > 7 {
            Err(CpuError::AmxOpFailed(format!(
                "XRow index {index} out of range 0..=7"
            )))
        } else {
            Ok(Self(index))
        }
    }

    /// # Safety
    ///
    /// Caller must ensure `index <= 7`.
    #[inline]
    pub const unsafe fn new_unchecked(index: u8) -> Self {
        Self(index)
    }

    #[inline]
    pub const fn index(self) -> u8 {
        self.0
    }

    /// Byte offset into the 512-byte circular X buffer.
    #[inline]
    pub const fn byte_offset(self) -> u64 {
        (self.0 as u64) * 64
    }
}

impl fmt::Display for XRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "X{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// YRow (0..=7)
// ---------------------------------------------------------------------------

/// Index of a Y-register row (0..=7).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct YRow(u8);

impl YRow {
    #[inline]
    pub fn new(index: u8) -> crate::Result<Self> {
        if index > 7 {
            Err(CpuError::AmxOpFailed(format!(
                "YRow index {index} out of range 0..=7"
            )))
        } else {
            Ok(Self(index))
        }
    }

    /// # Safety
    ///
    /// Caller must ensure `index <= 7`.
    #[inline]
    pub const unsafe fn new_unchecked(index: u8) -> Self {
        Self(index)
    }

    #[inline]
    pub const fn index(self) -> u8 {
        self.0
    }

    /// Byte offset into the 512-byte circular Y buffer.
    #[inline]
    pub const fn byte_offset(self) -> u64 {
        (self.0 as u64) * 64
    }
}

impl fmt::Display for YRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Y{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// ZRow (0..=63)
// ---------------------------------------------------------------------------

/// Index of a Z-register row (0..=63).
///
/// The Z register file has 64 rows of 64 bytes = 4096 bytes total.
/// For f32 fma32 operations, there are 4 independent 16×16 tiles:
/// tile 0 uses rows {0,4,8,...,60}, tile 1 uses {1,5,9,...,61}, etc.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ZRow(u8);

impl ZRow {
    #[inline]
    pub fn new(index: u8) -> crate::Result<Self> {
        if index > 63 {
            Err(CpuError::AmxOpFailed(format!(
                "ZRow index {index} out of range 0..=63"
            )))
        } else {
            Ok(Self(index))
        }
    }

    /// # Safety
    ///
    /// Caller must ensure `index <= 63`.
    #[inline]
    pub const unsafe fn new_unchecked(index: u8) -> Self {
        Self(index)
    }

    #[inline]
    pub const fn index(self) -> u8 {
        self.0
    }

    /// Which of the 4 f32 tiles this row belongs to (0..=3).
    #[inline]
    pub const fn tile(self) -> u8 {
        self.0 & 3
    }
}

impl fmt::Display for ZRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Z{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Convenience const arrays
// ---------------------------------------------------------------------------

pub const ALL_X: [XRow; 8] = [
    XRow(0),
    XRow(1),
    XRow(2),
    XRow(3),
    XRow(4),
    XRow(5),
    XRow(6),
    XRow(7),
];

pub const ALL_Y: [YRow; 8] = [
    YRow(0),
    YRow(1),
    YRow(2),
    YRow(3),
    YRow(4),
    YRow(5),
    YRow(6),
    YRow(7),
];

/// All 8 Z-register rows, in order.
pub const ALL_Z: [ZRow; 8] = [
    ZRow(0),
    ZRow(1),
    ZRow(2),
    ZRow(3),
    ZRow(4),
    ZRow(5),
    ZRow(6),
    ZRow(7),
];
