//! Render pipeline state — compiled vertex + fragment program + color formats.
//!
//! Mirrors `Pipeline` (the compute pipeline). A `RenderPipeline` wraps
//! `id<MTLRenderPipelineState>` and is built from a `RenderPipelineSpec`
//! via `Gpu::render_pipeline(...)` (see `render/factory.rs`).
//!
//! Phase 2 adds: depth/stencil format, MSAA sample count, vertex
//! descriptor, per-attachment blend state, depth-stencil state.

use crate::ffi::*;

/// A compiled render pipeline state. Wraps `id<MTLRenderPipelineState>`.
pub struct RenderPipeline {
    raw: ObjcId,
    color_attachments: usize,
    sample_count: u32,
}

impl RenderPipeline {
    pub(crate) fn from_raw(raw: ObjcId, color_attachments: usize, sample_count: u32) -> Self {
        RenderPipeline {
            raw,
            color_attachments,
            sample_count,
        }
    }

    pub fn color_attachments(&self) -> usize {
        self.color_attachments
    }

    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for RenderPipeline {
    #[mutants::skip]
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}

// ─────────────────────────────── pipeline spec ─────────────────────────────

/// Builder spec for `Gpu::render_pipeline`. Captures color attachment
/// formats + blend, depth/stencil formats, MSAA sample count, vertex
/// descriptor.
#[derive(Debug, Clone)]
pub struct RenderPipelineSpec {
    pub color_attachments: Vec<ColorAttachmentSpec>,
    pub depth_format: Option<NSUInteger>,
    pub stencil_format: Option<NSUInteger>,
    pub sample_count: u32,
    pub vertex_descriptor: Option<crate::render::vertex::VertexDescriptor>,
}

impl RenderPipelineSpec {
    /// Single color attachment, no depth, no MSAA.
    pub fn color(format: NSUInteger) -> Self {
        Self {
            color_attachments: vec![ColorAttachmentSpec::new(format)],
            depth_format: None,
            stencil_format: None,
            sample_count: 1,
            vertex_descriptor: None,
        }
    }

    /// Multiple color attachments.
    pub fn colors(formats: &[NSUInteger]) -> Self {
        Self {
            color_attachments: formats
                .iter()
                .copied()
                .map(ColorAttachmentSpec::new)
                .collect(),
            depth_format: None,
            stencil_format: None,
            sample_count: 1,
            vertex_descriptor: None,
        }
    }

    pub fn with_depth(mut self, format: NSUInteger) -> Self {
        self.depth_format = Some(format);
        self
    }

    pub fn with_stencil(mut self, format: NSUInteger) -> Self {
        self.stencil_format = Some(format);
        self
    }

    pub fn with_sample_count(mut self, n: u32) -> Self {
        self.sample_count = n;
        self
    }

    pub fn with_vertex_descriptor(mut self, desc: crate::render::vertex::VertexDescriptor) -> Self {
        self.vertex_descriptor = Some(desc);
        self
    }

    /// Replace the blend configuration on attachment `i`.
    pub fn with_blend(mut self, index: usize, blend: BlendState) -> Self {
        self.color_attachments[index].blend = Some(blend);
        self
    }
}

/// Per-attachment color spec: pixel format + optional blend state.
#[derive(Debug, Clone)]
pub struct ColorAttachmentSpec {
    pub format: NSUInteger,
    pub blend: Option<BlendState>,
    /// Write mask (bits: R=1, G=2, B=4, A=8). Default 0xF (all).
    pub write_mask: u32,
}

impl ColorAttachmentSpec {
    pub fn new(format: NSUInteger) -> Self {
        Self {
            format,
            blend: None,
            write_mask: 0xF,
        }
    }

    pub fn with_blend(mut self, blend: BlendState) -> Self {
        self.blend = Some(blend);
        self
    }

    pub fn with_write_mask(mut self, mask: u32) -> Self {
        self.write_mask = mask;
        self
    }
}

/// Blend state for one color attachment.
#[derive(Debug, Clone, Copy)]
pub struct BlendState {
    pub rgb_op: BlendOp,
    pub alpha_op: BlendOp,
    pub src_rgb: BlendFactor,
    pub dst_rgb: BlendFactor,
    pub src_alpha: BlendFactor,
    pub dst_alpha: BlendFactor,
}

impl BlendState {
    /// Standard alpha blending (source over).
    pub fn alpha_over() -> Self {
        Self {
            rgb_op: BlendOp::Add,
            alpha_op: BlendOp::Add,
            src_rgb: BlendFactor::SourceAlpha,
            dst_rgb: BlendFactor::OneMinusSourceAlpha,
            src_alpha: BlendFactor::One,
            dst_alpha: BlendFactor::OneMinusSourceAlpha,
        }
    }

    /// Additive blending.
    pub fn additive() -> Self {
        Self {
            rgb_op: BlendOp::Add,
            alpha_op: BlendOp::Add,
            src_rgb: BlendFactor::One,
            dst_rgb: BlendFactor::One,
            src_alpha: BlendFactor::One,
            dst_alpha: BlendFactor::One,
        }
    }
}

/// Blend factor (matches `MTLBlendFactor`).
#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub enum BlendFactor {
    Zero = MTLBlendFactorZero,
    One = MTLBlendFactorOne,
    SourceColor = MTLBlendFactorSourceColor,
    OneMinusSourceColor = MTLBlendFactorOneMinusSourceColor,
    SourceAlpha = MTLBlendFactorSourceAlpha,
    OneMinusSourceAlpha = MTLBlendFactorOneMinusSourceAlpha,
    DestinationAlpha = MTLBlendFactorDestinationAlpha,
    OneMinusDestinationAlpha = MTLBlendFactorOneMinusDestinationAlpha,
    DestinationColor = MTLBlendFactorDestinationColor,
    OneMinusDestinationColor = MTLBlendFactorOneMinusDestinationColor,
}

/// Blend operation (matches `MTLBlendOperation`).
#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub enum BlendOp {
    Add = MTLBlendOperationAdd,
    Subtract = MTLBlendOperationSubtract,
    ReverseSubtract = MTLBlendOperationReverseSubtract,
    Min = MTLBlendOperationMin,
    Max = MTLBlendOperationMax,
}

// ───────────────────────────── cull / winding / depth ─────────────────────

/// Triangle cull mode.
#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub enum CullMode {
    None = MTLCullModeNone,
    Front = MTLCullModeFront,
    Back = MTLCullModeBack,
}

/// Front-face winding order.
#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub enum Winding {
    Clockwise = MTLWindingClockwise,
    CounterClockwise = MTLWindingCounterClockwise,
}

/// Depth compare function.
#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub enum CompareFunction {
    Never = MTLCompareFunctionNever,
    Less = MTLCompareFunctionLess,
    Equal = MTLCompareFunctionEqual,
    LessEqual = MTLCompareFunctionLessEqual,
    Greater = MTLCompareFunctionGreater,
    NotEqual = MTLCompareFunctionNotEqual,
    GreaterEqual = MTLCompareFunctionGreaterEqual,
    Always = MTLCompareFunctionAlways,
}

/// Depth/stencil state configuration.
#[derive(Debug, Clone, Copy)]
pub struct DepthStencil {
    pub compare: CompareFunction,
    pub write_enabled: bool,
}

impl DepthStencil {
    /// Standard "draw if closer, write depth" state.
    pub fn less_write() -> Self {
        Self {
            compare: CompareFunction::Less,
            write_enabled: true,
        }
    }

    /// "Always pass, no depth write" — useful for overlays.
    pub fn always_no_write() -> Self {
        Self {
            compare: CompareFunction::Always,
            write_enabled: false,
        }
    }
}

/// A compiled depth/stencil state. Wraps `id<MTLDepthStencilState>`.
pub struct DepthStencilState {
    raw: ObjcId,
}

impl DepthStencilState {
    pub(crate) fn from_raw(raw: ObjcId) -> Self {
        Self { raw }
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for DepthStencilState {
    #[mutants::skip]
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}
