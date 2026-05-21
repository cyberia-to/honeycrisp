//! Render command encoder — records draw calls into a command buffer.
//!
//! Mirrors `Encoder` (the compute encoder). Created via
//! `Commands::render_encoder(&pass_desc)`.

use crate::buffer::Buffer;
use crate::ffi::*;
use crate::render::pipeline::RenderPipeline;
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
