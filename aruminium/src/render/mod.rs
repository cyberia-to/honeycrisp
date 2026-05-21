//! Render submodule — raster path for aruminium.
//!
//! Mirrors the compute path:
//! - `RenderPipeline` ↔ `Pipeline`
//! - `RenderEncoder` ↔ `Encoder`
//! - `RenderPassDescriptor` configures the attachments encoded into.
//!
//! Phase 1: color attachments, basic draws, render-target textures.
//! Phase 2 (separate commit): depth/stencil, MSAA + resolve, vertex
//! descriptors, indexed draws, blend state, cull/winding.

pub mod encoder;
mod factory;
pub mod pass;
pub mod pipeline;

pub use encoder::{PrimitiveType, RenderEncoder};
pub use pass::{ColorAttachmentDesc, LoadAction, RenderPassDescriptor, StoreAction};
pub use pipeline::{ColorAttachmentSpec, RenderPipeline, RenderPipelineSpec};

#[cfg(test)]
mod tests;
