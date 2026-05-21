//! Render pipeline state — compiled vertex + fragment program + color formats.
//!
//! Mirrors `Pipeline` (the compute pipeline). A `RenderPipeline` wraps
//! `id<MTLRenderPipelineState>` and is built from a `RenderPipelineSpec`
//! via `Gpu::render_pipeline(...)` (see `render/factory.rs`).

use crate::ffi::*;

/// A compiled render pipeline state. Wraps `id<MTLRenderPipelineState>`.
pub struct RenderPipeline {
    raw: ObjcId,
    /// Number of color attachments configured.
    color_attachments: usize,
    /// Sample count (1 for non-MSAA).
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

    /// Number of color attachments this pipeline writes to.
    pub fn color_attachments(&self) -> usize {
        self.color_attachments
    }

    /// MSAA sample count (1 if not multisampled).
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for RenderPipeline {
    #[mutants::skip] // RAII release — tested by drop_stress_leak
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}

// ─────────────────────────────── pipeline spec ─────────────────────────────

/// Builder spec for `Gpu::render_pipeline`. Phase 1: pixel formats only.
///
/// Phase 2 will extend with depth, stencil, MSAA, blend, vertex descriptor.
#[derive(Debug, Clone)]
pub struct RenderPipelineSpec {
    /// Color attachments (one pixel format per slot). At least one required.
    pub color_attachments: Vec<ColorAttachmentSpec>,
}

impl RenderPipelineSpec {
    /// Single color attachment.
    pub fn color(format: NSUInteger) -> Self {
        Self {
            color_attachments: vec![ColorAttachmentSpec::new(format)],
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
        }
    }
}

/// Per-attachment color spec: pixel format. Phase 1 stub; Phase 2 adds blend.
#[derive(Debug, Clone)]
pub struct ColorAttachmentSpec {
    pub format: NSUInteger,
    /// Write mask (bits: R=1, G=2, B=4, A=8). Default 0xF (all).
    pub write_mask: u32,
}

impl ColorAttachmentSpec {
    pub fn new(format: NSUInteger) -> Self {
        Self {
            format,
            write_mask: 0xF,
        }
    }
}
