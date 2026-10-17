//! Zero-copy memory driver for Apple Silicon.
//!
//! IOSurface-backed pinned buffers visible to every compute unit —
//! CPU, GPU, AMX, ANE — through one allocation.

#![no_std]

#[cfg(test)]
extern crate std;

pub mod block;
pub mod ffi;
pub mod grid;
pub mod layout;
pub mod tape;

pub use block::{
    Block, BlockPlan, ImageBlockPlan, NativeImageFormat, NativeImageLayout, NativePlaneLayout,
};
pub use grid::{Cell, Grid};
pub use layout::{Layout, Stat};
pub use tape::Tape;

#[derive(Debug)]
pub enum MemError {
    ZeroSize,
    SizeOverflow,
    BlockAlignmentInvalid,
    BlockPropertiesFailed,
    BlockCreateFailed,
    BlockLockFailed(i32),
    BlockExtentMismatch { expected: usize, actual: usize },
    BlockAddressInvalid,
    ImageFormatUnsupported(u32),
    ImageLayoutInvalid,
    BlockRetainFailed,
}

impl core::fmt::Display for MemError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MemError::ZeroSize => write!(f, "zero-size allocation"),
            MemError::SizeOverflow => write!(f, "allocation size exceeds integer or slice bounds"),
            MemError::BlockAlignmentInvalid => write!(f, "invalid IOSurface property alignment"),
            MemError::BlockPropertiesFailed => write!(f, "CoreFoundation property creation failed"),
            MemError::BlockCreateFailed => write!(f, "IOSurfaceCreate failed"),
            MemError::BlockLockFailed(kr) => write!(f, "IOSurfaceLock failed: {:#x}", kr),
            MemError::BlockExtentMismatch { expected, actual } => {
                write!(
                    f,
                    "IOSurface extent mismatch: expected {expected}, actual {actual}"
                )
            }
            MemError::BlockAddressInvalid => write!(f, "invalid IOSurface CPU mapping"),
            MemError::ImageFormatUnsupported(format) => {
                write!(f, "unsupported IOSurface image format {format:#010x}")
            }
            MemError::ImageLayoutInvalid => write!(f, "invalid IOSurface image plane layout"),
            MemError::BlockRetainFailed => write!(f, "IOSurface retain failed"),
        }
    }
}

impl core::error::Error for MemError {}
