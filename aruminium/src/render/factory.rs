//! Render-related `Gpu` factory methods.
//!
//! Kept in the render submodule so the compute-only `device.rs` stays
//! focused and within the 500-line per-file limit.

use crate::device::Gpu;
use crate::ffi::*;
use crate::render::pipeline::{RenderPipeline, RenderPipelineSpec};
use crate::shader::Shader;
use crate::texture::Texture;
use crate::GpuError;
use std::ffi::c_void;

impl Gpu {
    /// Create a 2D color texture usable as a render target.
    ///
    /// Usage flags: `RenderTarget | ShaderRead`. Storage mode `Private`.
    pub fn render_target(
        &self,
        width: u32,
        height: u32,
        format: NSUInteger,
    ) -> Result<Texture, GpuError> {
        if width == 0 || height == 0 {
            return Err(GpuError::TextureCreationFailed(
                "width/height must be > 0".into(),
            ));
        }
        unsafe {
            let cls = objc_getClass(c"MTLTextureDescriptor".as_ptr()) as ObjcId;
            let desc = msg0(cls, sel_registerName(c"new".as_ptr()));
            if desc.is_null() {
                return Err(GpuError::TextureCreationFailed(
                    "descriptor alloc failed".into(),
                ));
            }
            msg1_uint_void(desc, SEL_setTextureType(), MTLTextureType2D);
            msg1_uint_void(desc, SEL_setPixelFormat(), format);
            msg1_uint_void(desc, SEL_setWidth(), width as NSUInteger);
            msg1_uint_void(desc, SEL_setHeight(), height as NSUInteger);
            msg1_uint_void(
                desc,
                SEL_setUsage(),
                MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead,
            );
            msg1_uint_void(desc, SEL_setStorageMode(), 0x2); // Private
            let tex = self.texture(desc);
            release(desc);
            tex
        }
    }

    /// Create a render pipeline from a vertex + fragment function + spec.
    pub fn render_pipeline(
        &self,
        vertex: &Shader,
        fragment: &Shader,
        spec: &RenderPipelineSpec,
    ) -> Result<RenderPipeline, GpuError> {
        if spec.color_attachments.is_empty() {
            return Err(GpuError::PipelineCreationFailed(
                "at least one color attachment required".into(),
            ));
        }
        unsafe {
            let cls = objc_getClass(c"MTLRenderPipelineDescriptor".as_ptr()) as ObjcId;
            let desc = msg0(cls, sel_registerName(c"new".as_ptr()));
            if desc.is_null() {
                return Err(GpuError::PipelineCreationFailed(
                    "descriptor alloc failed".into(),
                ));
            }

            msg1_void(desc, SEL_setVertexFunction(), vertex.as_raw());
            msg1_void(desc, SEL_setFragmentFunction(), fragment.as_raw());

            let color_array = msg0(desc, SEL_colorAttachments());
            for (i, ca) in spec.color_attachments.iter().enumerate() {
                let slot =
                    msg1_uint_id(color_array, SEL_objectAtIndexedSubscript(), i as NSUInteger);
                msg1_uint_void(slot, SEL_setPixelFormat(), ca.format);
                msg1_uint_void(slot, SEL_setWriteMask(), ca.write_mask as NSUInteger);
                msg1_bool_void(slot, SEL_setBlendingEnabled(), false);
            }

            let mut error: ObjcId = std::ptr::null_mut();
            type F = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, *mut ObjcId) -> ObjcId;
            let f: F = std::mem::transmute(objc_msgSend as *const c_void);
            let raw = f(
                self.as_raw(),
                SEL_newRenderPipelineStateWithDescriptor_error(),
                desc,
                &mut error,
            );
            release(desc);
            if raw.is_null() {
                let msg = nserror_string(error).unwrap_or_else(|| "unknown error".into());
                return Err(GpuError::PipelineCreationFailed(msg));
            }
            Ok(RenderPipeline::from_raw(
                raw,
                spec.color_attachments.len(),
                1,
            ))
        }
    }
}
