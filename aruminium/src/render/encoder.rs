//! Render command encoder — records draw calls into a command buffer.
//!
//! Mirrors `Encoder` (the compute encoder). Created via
//! `Commands::render_encoder(&pass_desc)`.

use crate::buffer::Buffer;
use crate::ffi::*;
use crate::render::pipeline::{CullMode, DepthStencilState, RenderPipeline, Winding};
use crate::texture::Texture;
use std::ffi::c_void;

/// Primitive topology (matches `MTLPrimitiveType`).
#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub enum PrimitiveType {
    Point = MTLPrimitiveTypePoint,
    Line = MTLPrimitiveTypeLine,
    LineStrip = MTLPrimitiveTypeLineStrip,
    Triangle = MTLPrimitiveTypeTriangle,
    TriangleStrip = MTLPrimitiveTypeTriangleStrip,
}

/// Index buffer element type (matches `MTLIndexType`).
#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub enum IndexType {
    UInt16 = MTLIndexTypeUInt16,
    UInt32 = MTLIndexTypeUInt32,
}

impl IndexType {
    pub fn size(self) -> usize {
        match self {
            IndexType::UInt16 => 2,
            IndexType::UInt32 => 4,
        }
    }
}

/// A render command encoder. Wraps `id<MTLRenderCommandEncoder>`.
pub struct RenderEncoder {
    raw: ObjcId,
    owned: bool,
    /// Whether `end()` has already been called. Tracks lifecycle so we don't
    /// re-issue endEncoding from Drop after explicit `end()`.
    ended: std::cell::Cell<bool>,
}

impl RenderEncoder {
    pub(crate) fn from_raw(raw: ObjcId, owned: bool) -> Self {
        RenderEncoder {
            raw,
            owned,
            ended: std::cell::Cell::new(false),
        }
    }

    /// Bind a render pipeline state.
    #[inline(always)]
    pub fn bind(&self, pipeline: &RenderPipeline) {
        unsafe { msg1_void(self.raw, SEL_setRenderPipelineState(), pipeline.as_raw()) };
    }

    /// Bind a vertex buffer at `index`.
    #[inline(always)]
    pub fn set_vertex_buffer(&self, index: u32, buffer: &Buffer, offset: usize) {
        unsafe {
            msg3_void(
                self.raw,
                SEL_setVertexBuffer_offset_atIndex(),
                buffer.as_raw(),
                offset,
                index as NSUInteger,
            )
        };
    }

    /// Bind a fragment buffer at `index`.
    #[inline(always)]
    pub fn set_fragment_buffer(&self, index: u32, buffer: &Buffer, offset: usize) {
        unsafe {
            msg3_void(
                self.raw,
                SEL_setFragmentBuffer_offset_atIndex(),
                buffer.as_raw(),
                offset,
                index as NSUInteger,
            )
        };
    }

    /// Push inline vertex constants at `index`.
    #[inline(always)]
    pub fn push_vertex(&self, data: &[u8], index: u32) {
        unsafe {
            msg_bytes_void(
                self.raw,
                SEL_setVertexBytes_length_atIndex(),
                data.as_ptr() as *const c_void,
                data.len(),
                index as NSUInteger,
            )
        };
    }

    /// Push inline fragment constants at `index`.
    #[inline(always)]
    pub fn push_fragment(&self, data: &[u8], index: u32) {
        unsafe {
            msg_bytes_void(
                self.raw,
                SEL_setFragmentBytes_length_atIndex(),
                data.as_ptr() as *const c_void,
                data.len(),
                index as NSUInteger,
            )
        };
    }

    /// Bind a texture at vertex stage.
    #[inline(always)]
    pub fn set_vertex_texture(&self, index: u32, texture: &Texture) {
        unsafe {
            type F = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, NSUInteger);
            let f: F = std::mem::transmute(objc_msgSend as *const c_void);
            f(
                self.raw,
                SEL_setVertexTexture_atIndex(),
                texture.as_raw(),
                index as NSUInteger,
            );
        }
    }

    /// Bind a texture at fragment stage.
    #[inline(always)]
    pub fn set_fragment_texture(&self, index: u32, texture: &Texture) {
        unsafe {
            type F = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, NSUInteger);
            let f: F = std::mem::transmute(objc_msgSend as *const c_void);
            f(
                self.raw,
                SEL_setFragmentTexture_atIndex(),
                texture.as_raw(),
                index as NSUInteger,
            );
        }
    }

    /// Set the viewport rect (in pixels, with depth range).
    #[inline]
    pub fn set_viewport(&self, x: f64, y: f64, w: f64, h: f64, near: f64, far: f64) {
        unsafe {
            let v = MTLViewport {
                origin_x: x,
                origin_y: y,
                width: w,
                height: h,
                znear: near,
                zfar: far,
            };
            type F = unsafe extern "C" fn(ObjcId, ObjcSel, MTLViewport);
            let f: F = std::mem::transmute(objc_msgSend as *const c_void);
            f(self.raw, SEL_setViewport(), v);
        }
    }

    /// Set the scissor rect (in pixels).
    #[inline]
    pub fn set_scissor(&self, x: u32, y: u32, w: u32, h: u32) {
        unsafe {
            let r = MTLScissorRect {
                x: x as NSUInteger,
                y: y as NSUInteger,
                width: w as NSUInteger,
                height: h as NSUInteger,
            };
            type F = unsafe extern "C" fn(ObjcId, ObjcSel, MTLScissorRect);
            let f: F = std::mem::transmute(objc_msgSend as *const c_void);
            f(self.raw, SEL_setScissorRect(), r);
        }
    }

    /// Set the cull mode.
    #[inline]
    pub fn set_cull_mode(&self, mode: CullMode) {
        unsafe { msg1_uint_void(self.raw, SEL_setCullMode(), mode as NSUInteger) };
    }

    /// Set front-face winding.
    #[inline]
    pub fn set_front_facing_winding(&self, winding: Winding) {
        unsafe { msg1_uint_void(self.raw, SEL_setFrontFacingWinding(), winding as NSUInteger) };
    }

    /// Bind a depth/stencil state.
    #[inline]
    pub fn set_depth_stencil_state(&self, state: &DepthStencilState) {
        unsafe { msg1_void(self.raw, SEL_setDepthStencilState(), state.as_raw()) };
    }

    /// Set depth bias (offset, slope-scale, clamp).
    #[inline]
    pub fn set_depth_bias(&self, bias: f32, slope_scale: f32, clamp: f32) {
        unsafe {
            type F = unsafe extern "C" fn(ObjcId, ObjcSel, f32, f32, f32);
            let f: F = std::mem::transmute(objc_msgSend as *const c_void);
            f(self.raw, SEL_setDepthBias(), bias, slope_scale, clamp);
        }
    }

    /// Issue a non-indexed draw.
    #[inline(always)]
    pub fn draw(&self, primitive: PrimitiveType, start: u32, count: u32) {
        unsafe {
            type F = unsafe extern "C" fn(ObjcId, ObjcSel, NSUInteger, NSUInteger, NSUInteger);
            let f: F = std::mem::transmute(objc_msgSend as *const c_void);
            f(
                self.raw,
                SEL_drawPrimitives_vertexStart_vertexCount(),
                primitive as NSUInteger,
                start as NSUInteger,
                count as NSUInteger,
            );
        }
    }

    /// Issue an instanced non-indexed draw.
    #[inline(always)]
    pub fn draw_instanced(&self, primitive: PrimitiveType, start: u32, count: u32, instances: u32) {
        unsafe {
            type F = unsafe extern "C" fn(
                ObjcId,
                ObjcSel,
                NSUInteger,
                NSUInteger,
                NSUInteger,
                NSUInteger,
            );
            let f: F = std::mem::transmute(objc_msgSend as *const c_void);
            f(
                self.raw,
                SEL_drawPrimitives_vertexStart_vertexCount_instanceCount(),
                primitive as NSUInteger,
                start as NSUInteger,
                count as NSUInteger,
                instances as NSUInteger,
            );
        }
    }

    /// Issue an indexed draw.
    #[inline(always)]
    pub fn draw_indexed(
        &self,
        primitive: PrimitiveType,
        index_count: u32,
        index_type: IndexType,
        index_buffer: &Buffer,
        index_buffer_offset: usize,
    ) {
        unsafe {
            type F = unsafe extern "C" fn(
                ObjcId,
                ObjcSel,
                NSUInteger,
                NSUInteger,
                NSUInteger,
                ObjcId,
                NSUInteger,
            );
            let f: F = std::mem::transmute(objc_msgSend as *const c_void);
            f(
                self.raw,
                SEL_drawIndexedPrimitives(),
                primitive as NSUInteger,
                index_count as NSUInteger,
                index_type as NSUInteger,
                index_buffer.as_raw(),
                index_buffer_offset,
            );
        }
    }

    /// Issue an instanced indexed draw.
    #[inline(always)]
    pub fn draw_indexed_instanced(
        &self,
        primitive: PrimitiveType,
        index_count: u32,
        index_type: IndexType,
        index_buffer: &Buffer,
        index_buffer_offset: usize,
        instances: u32,
    ) {
        unsafe {
            type F = unsafe extern "C" fn(
                ObjcId,
                ObjcSel,
                NSUInteger,
                NSUInteger,
                NSUInteger,
                ObjcId,
                NSUInteger,
                NSUInteger,
            );
            let f: F = std::mem::transmute(objc_msgSend as *const c_void);
            f(
                self.raw,
                SEL_drawIndexedPrimitives_instanced(),
                primitive as NSUInteger,
                index_count as NSUInteger,
                index_type as NSUInteger,
                index_buffer.as_raw(),
                index_buffer_offset,
                instances as NSUInteger,
            );
        }
    }

    /// Finish encoding. Must be called before submitting the command buffer.
    /// Consumes the encoder to enforce one-shot semantics.
    #[inline(always)]
    pub fn end(self) {
        self.ended.set(true);
        unsafe { msg0_void(self.raw, SEL_endEncoding()) };
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for RenderEncoder {
    #[mutants::skip] // RAII release — tested via drop_stress
    fn drop(&mut self) {
        if !self.ended.get() {
            // Caller dropped without calling end() — finish encoding to
            // keep Metal's command buffer in a valid state.
            unsafe { msg0_void(self.raw, SEL_endEncoding()) };
        }
        if self.owned {
            unsafe { release_nonnull(self.raw) };
        }
    }
}
