//! Zero-copy memory driver: pinned buffers visible to every compute unit
//! through one allocation.
//!
//! Apple Silicon: IOSurface (CPU, Metal, AMX, ANE). Android/Linux: memfd —
//! same CPU semantics, fd-exportable for GPU import.

pub mod block;
#[cfg(target_vendor = "apple")]
pub mod ffi;
pub mod grid;
pub mod layout;
pub mod tape;

pub use block::Block;
pub use grid::{Cell, Grid};
pub use layout::{Layout, Stat};
pub use tape::Tape;

#[derive(Debug)]
pub enum MemError {
    ZeroSize,
    BlockCreateFailed,
    BlockLockFailed(i32),
}

impl std::fmt::Display for MemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemError::ZeroSize => write!(f, "zero-size allocation"),
            MemError::BlockCreateFailed => write!(f, "block allocation failed"),
            MemError::BlockLockFailed(kr) => write!(f, "block map/lock failed: {:#x}", kr),
        }
    }
}

impl std::error::Error for MemError {}
