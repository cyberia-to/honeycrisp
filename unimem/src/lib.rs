pub mod block;
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
            MemError::BlockCreateFailed => write!(f, "IOSurfaceCreate failed"),
            MemError::BlockLockFailed(kr) => write!(f, "IOSurfaceLock failed: {:#x}", kr),
        }
    }
}

impl std::error::Error for MemError {}
