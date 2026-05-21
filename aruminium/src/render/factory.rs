//! Render-related `Gpu` factory methods.
//!
//! Kept in the render submodule so the compute-only `device.rs` stays
//! focused and within the 500-line per-file limit.

use crate::device::Gpu;
use crate::ffi::*;
use crate::render::pipeline::{
    DepthStencil, DepthStencilState, RenderPipeline, RenderPipelineSpec,
};
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
        self.render_target_ms(width, height, format, 1)
    }

    /// Create a 2D color texture usable as a render target with the given
    /// MSAA sample count. `samples = 1` is the non-MSAA case.
    pub fn render_target_ms(
        &self,
        width: u32,
        height: u32,
        format: NSUInteger,
        samples: u32,
    ) -> Result<Texture, GpuError> {
        if width == 0 || height == 0 {
            return Err(GpuError::TextureCreationFailed(
                "width/height must be > 0".into(),
            ));
        }
        unsafe {
            let desc = build_texture_descriptor(
                format,
                width,
                height,
                MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead,
                samples,
            )?;
            let tex = self.texture(desc);
            release(desc);
            tex
        }
    }

    /// Create a 2D depth (or depth/stencil) texture usable as a depth attachment.
    pub fn depth_target(
        &self,
        width: u32,
        height: u32,
        format: NSUInteger,
    ) -> Result<Texture, GpuError> {
        self.depth_target_ms(width, height, format, 1)
    }

    /// Same as `depth_target` but multisampled.
    pub fn depth_target_ms(
        &self,
        width: u32,
        height: u32,
        format: NSUInteger,
        samples: u32,
    ) -> Result<Texture, GpuError> {
        if width == 0 || height == 0 {
            return Err(GpuError::TextureCreationFailed(
                "width/height must be > 0".into(),
            ));
        }
        unsafe {
            let desc = build_texture_descriptor(
                format,
                width,
                height,
                MTLTextureUsageRenderTarget,
                samples,
            )?;
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

            configure_color_attachments(desc, spec);

            if let Some(df) = spec.depth_format {
                msg1_uint_void(desc, SEL_setDepthAttachmentPixelFormat(), df);
            }
            if let Some(sf) = spec.stencil_format {
                msg1_uint_void(desc, SEL_setStencilAttachmentPixelFormat(), sf);
            }
            if spec.sample_count > 1 {
                msg1_uint_void(
                    desc,
                    SEL_setRasterSampleCount(),
                    spec.sample_count as NSUInteger,
                );
            }
            if let Some(vd) = &spec.vertex_descriptor {
                let vdesc = vd.build_objc();
                msg1_void(desc, SEL_setVertexDescriptor(), vdesc);
                release(vdesc);
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
                spec.sample_count,
            ))
        }
    }

    /// Build a depth/stencil state object from a `DepthStencil` spec.
    pub fn depth_stencil_state(&self, ds: DepthStencil) -> Result<DepthStencilState, GpuError> {
        unsafe {
            let cls = objc_getClass(c"MTLDepthStencilDescriptor".as_ptr()) as ObjcId;
            let desc = msg0(cls, sel_registerName(c"new".as_ptr()));
            if desc.is_null() {
                return Err(GpuError::PipelineCreationFailed(
                    "depth-stencil descriptor alloc failed".into(),
                ));
            }
            msg1_uint_void(
                desc,
                SEL_setDepthCompareFunction(),
                ds.compare as NSUInteger,
            );
            msg1_bool_void(desc, SEL_setDepthWriteEnabled(), ds.write_enabled);
            let raw = msg1_id_id(
                self.as_raw(),
                SEL_newDepthStencilStateWithDescriptor(),
                desc,
            );
            release(desc);
            if raw.is_null() {
                return Err(GpuError::PipelineCreationFailed(
                    "depth-stencil state creation failed".into(),
                ));
            }
            Ok(DepthStencilState::from_raw(raw))
        }
    }
}

unsafe fn build_texture_descriptor(
    format: NSUInteger,
    width: u32,
    height: u32,
    usage: NSUInteger,
    samples: u32,
) -> Result<ObjcId, GpuError> {
    let cls = objc_getClass(c"MTLTextureDescriptor".as_ptr()) as ObjcId;
    let desc = msg0(cls, sel_registerName(c"new".as_ptr()));
    if desc.is_null() {
        return Err(GpuError::TextureCreationFailed(
            "descriptor alloc failed".into(),
        ));
    }
    let tex_type = if samples > 1 {
        MTLTextureType2DMultisample
    } else {
        MTLTextureType2D
    };
    msg1_uint_void(desc, SEL_setTextureType(), tex_type);
    msg1_uint_void(desc, SEL_setPixelFormat(), format);
    msg1_uint_void(desc, SEL_setWidth(), width as NSUInteger);
    msg1_uint_void(desc, SEL_setHeight(), height as NSUInteger);
    msg1_uint_void(desc, SEL_setUsage(), usage);
    msg1_uint_void(desc, SEL_setStorageMode(), 0x2); // Private
    if samples > 1 {
        msg1_uint_void(desc, SEL_setSampleCount(), samples as NSUInteger);
    }
    Ok(desc)
}

unsafe fn configure_color_attachments(desc: ObjcId, spec: &RenderPipelineSpec) {
    let color_array = msg0(desc, SEL_colorAttachments());
    for (i, ca) in spec.color_attachments.iter().enumerate() {
        let slot = msg1_uint_id(color_array, SEL_objectAtIndexedSubscript(), i as NSUInteger);
        msg1_uint_void(slot, SEL_setPixelFormat(), ca.format);
        msg1_uint_void(slot, SEL_setWriteMask(), ca.write_mask as NSUInteger);
        if let Some(b) = &ca.blend {
            msg1_bool_void(slot, SEL_setBlendingEnabled(), true);
            msg1_uint_void(slot, SEL_setRgbBlendOperation(), b.rgb_op as NSUInteger);
            msg1_uint_void(slot, SEL_setAlphaBlendOperation(), b.alpha_op as NSUInteger);
            msg1_uint_void(slot, SEL_setSourceRGBBlendFactor(), b.src_rgb as NSUInteger);
            msg1_uint_void(
                slot,
                SEL_setDestinationRGBBlendFactor(),
                b.dst_rgb as NSUInteger,
            );
            msg1_uint_void(
                slot,
                SEL_setSourceAlphaBlendFactor(),
                b.src_alpha as NSUInteger,
            );
            msg1_uint_void(
                slot,
                SEL_setDestinationAlphaBlendFactor(),
                b.dst_alpha as NSUInteger,
            );
        } else {
            msg1_bool_void(slot, SEL_setBlendingEnabled(), false);
        }
    }
}
