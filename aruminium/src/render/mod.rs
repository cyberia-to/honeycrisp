//! Render submodule — raster path for aruminium.
//!
//! Mirrors the compute path:
//! - `RenderPipeline` ↔ `Pipeline`
//! - `RenderEncoder` ↔ `Encoder`
//! - `RenderPassDescriptor` configures the attachments encoded into.
//!
//! Phase 1: color attachments, basic draws, render-target textures.
//! Phase 2: depth/stencil, MSAA + resolve, vertex descriptors, indexed
//!          draws, blend state, cull/winding.

pub mod encoder;
mod factory;
pub mod pass;
pub mod pipeline;
pub mod vertex;

pub use encoder::{IndexType, PrimitiveType, RenderEncoder};
pub use pass::{
    ColorAttachmentDesc, DepthAttachmentDesc, LoadAction, RenderPassDescriptor, StoreAction,
};
pub use pipeline::{
    BlendFactor, BlendOp, BlendState, ColorAttachmentSpec, CompareFunction, CullMode, DepthStencil,
    DepthStencilState, RenderPipeline, RenderPipelineSpec, Winding,
};
pub use vertex::{VertexAttribute, VertexBufferLayout, VertexDescriptor, VertexFormat, VertexStep};

#[cfg(test)]
mod tests;
