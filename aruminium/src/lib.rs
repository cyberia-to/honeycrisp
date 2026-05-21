#![doc = include_str!("../README.md")]
#![allow(
    non_camel_case_types,
    non_upper_case_globals,
    non_snake_case,
    clippy::missing_transmute_annotations
)]

#[cfg(test)]
mod tests;

pub mod buffer;
pub mod command;
pub mod device;
pub mod dispatch;
pub mod encoder;
pub mod ffi;
pub mod pipeline;
pub mod render;
pub mod shader;
pub mod sync;
pub mod texture;

pub use buffer::Buffer;
pub use command::{Commands, Queue};
pub use device::Gpu;
pub use dispatch::{Batch, Dispatch, GpuFuture};
pub use encoder::{Copier, Encoder};
pub use pipeline::Pipeline;
pub use render::{
    ColorAttachmentDesc, ColorAttachmentSpec, LoadAction, PrimitiveType, RenderEncoder,
    RenderPassDescriptor, RenderPipeline, RenderPipelineSpec, StoreAction,
};
pub use shader::{Shader, ShaderLib};
pub use sync::{Event, Fence, SharedEvent};
pub use texture::Texture;

// Memory foundation — re-export for single-import convenience
pub use unimem::{Block, Layout, MemError, Tape};

// Re-export fp16 from acpu (single source of truth for numeric conversions)
pub use acpu::numeric::fp16::{f32_to_fp16, fp16_to_f32};
pub use acpu::{cast_f16_f32, cast_f32_f16};

/// Execute a closure inside an autorelease pool.
/// Use this to scope autoreleased ObjC objects (e.g. from `commands_unretained`).
/// The pool is drained even if the closure panics (unwind-safe).
#[inline]
pub fn autorelease_pool<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    struct PoolGuard(*mut std::ffi::c_void);
    impl Drop for PoolGuard {
        #[mutants::skip] // RAII pool drain — leak on skip, not testable
        fn drop(&mut self) {
            unsafe { ffi::objc_autoreleasePoolPop(self.0) }
        }
    }
    let pool = unsafe { ffi::objc_autoreleasePoolPush() };
    let _guard = PoolGuard(pool);
    f()
}

/// Errors returned by aruminium operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum GpuError {
    /// No Metal-capable GPU found on this system.
    DeviceNotFound,
    /// GPU buffer allocation failed.
    BufferCreationFailed(String),
    /// MSL shader compilation failed (includes compiler diagnostic).
    LibraryCompilationFailed(String),
    /// Named function not found in compiled shader library.
    FunctionNotFound(String),
    /// Compute pipeline creation failed.
    PipelineCreationFailed(String),
    /// Command buffer execution failed.
    CommandBufferError(String),
    /// Could not create a command encoder.
    EncoderCreationFailed,
    /// Could not create a command queue.
    QueueCreationFailed,
    /// Texture creation failed.
    TextureCreationFailed(String),
    /// Filesystem I/O error.
    Io(std::io::Error),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::DeviceNotFound => write!(f, "No Metal device found"),
            GpuError::BufferCreationFailed(msg) => write!(f, "Buffer creation failed: {}", msg),
            GpuError::LibraryCompilationFailed(msg) => {
                write!(f, "Shader compilation failed: {}", msg)
            }
            GpuError::FunctionNotFound(name) => write!(f, "Function not found: {}", name),
            GpuError::PipelineCreationFailed(msg) => {
                write!(f, "Pipeline creation failed: {}", msg)
            }
            GpuError::CommandBufferError(msg) => write!(f, "Command buffer error: {}", msg),
            GpuError::EncoderCreationFailed => write!(f, "Encoder creation failed"),
            GpuError::QueueCreationFailed => write!(f, "Command queue creation failed"),
            GpuError::TextureCreationFailed(msg) => {
                write!(f, "Texture creation failed: {}", msg)
            }
            GpuError::Io(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for GpuError {}

impl From<std::io::Error> for GpuError {
    fn from(e: std::io::Error) -> Self {
        GpuError::Io(e)
    }
}
