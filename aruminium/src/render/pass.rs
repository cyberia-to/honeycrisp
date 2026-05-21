//! Render pass descriptor — what to render INTO.
//!
//! Wraps `MTLRenderPassDescriptor` and its color attachment sub-descriptors.
//! Phase 1: color attachments only. Phase 2 adds depth/stencil + resolve.

use crate::ffi::*;
use crate::texture::Texture;

/// Load action: what happens to attachment contents at pass start.
#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub enum LoadAction {
    DontCare = MTLLoadActionDontCare,
    Load = MTLLoadActionLoad,
    Clear = MTLLoadActionClear,
}

/// Store action: what happens to attachment contents at pass end.
#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub enum StoreAction {
    DontCare = MTLStoreActionDontCare,
    Store = MTLStoreActionStore,
    MultisampleResolve = MTLStoreActionMultisampleResolve,
    StoreAndMultisampleResolve = MTLStoreActionStoreAndMultisampleResolve,
}

/// Color attachment description for a single render pass slot.
pub struct ColorAttachmentDesc<'a> {
    pub texture: &'a Texture,
    pub load_action: LoadAction,
    pub store_action: StoreAction,
    pub clear_color: [f64; 4],
    /// Resolve target for MSAA → single-sample. None unless `store_action`
    /// is one of the multisample-resolve variants.
    pub resolve_texture: Option<&'a Texture>,
    pub level: u32,
    pub slice: u32,
}

impl<'a> ColorAttachmentDesc<'a> {
    /// Convenience: clear-then-store the given texture with a solid color.
    pub fn clear(texture: &'a Texture, clear: [f64; 4]) -> Self {
        Self {
            texture,
            load_action: LoadAction::Clear,
            store_action: StoreAction::Store,
            clear_color: clear,
            resolve_texture: None,
            level: 0,
            slice: 0,
        }
    }
}

/// Depth attachment description.
pub struct DepthAttachmentDesc<'a> {
    pub texture: &'a Texture,
    pub load_action: LoadAction,
    pub store_action: StoreAction,
    pub clear_depth: f64,
}

impl<'a> DepthAttachmentDesc<'a> {
    /// Convenience: clear depth to 1.0 (far plane), discard after pass.
    pub fn clear(texture: &'a Texture) -> Self {
        Self {
            texture,
            load_action: LoadAction::Clear,
            store_action: StoreAction::DontCare,
            clear_depth: 1.0,
        }
    }
}

/// A render pass descriptor. Wraps `id<MTLRenderPassDescriptor>` (autoreleased
/// by `+renderPassDescriptor` then retained by us).
pub struct RenderPassDescriptor {
    raw: ObjcId,
}

impl RenderPassDescriptor {
    /// Create an empty render pass descriptor. Add at least one color
    /// attachment before encoding.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        unsafe {
            let cls = objc_getClass(c"MTLRenderPassDescriptor".as_ptr()) as ObjcId;
            // +renderPassDescriptor returns an autoreleased instance — retain it.
            let raw = msg0(cls, SEL_renderPassDescriptor());
            retain(raw);
            Self { raw }
        }
    }

    /// Configure color attachment `index`.
    pub fn color_attachment(&mut self, index: usize, desc: ColorAttachmentDesc<'_>) {
        unsafe {
            let array = msg0(self.raw, SEL_colorAttachments());
            let slot = msg1_uint_id(array, SEL_objectAtIndexedSubscript(), index);

            msg1_void(slot, SEL_setTexture(), desc.texture.as_raw());
            msg1_uint_void(slot, SEL_setLoadAction(), desc.load_action as NSUInteger);
            msg1_uint_void(slot, SEL_setStoreAction(), desc.store_action as NSUInteger);
            msg1_uint_void(slot, SEL_setLevel(), desc.level as NSUInteger);
            msg1_uint_void(slot, SEL_setSlice(), desc.slice as NSUInteger);
            let clear = MTLClearColor {
                red: desc.clear_color[0],
                green: desc.clear_color[1],
                blue: desc.clear_color[2],
                alpha: desc.clear_color[3],
            };
            type F = unsafe extern "C" fn(ObjcId, ObjcSel, MTLClearColor);
            let f: F = std::mem::transmute(objc_msgSend as *const std::ffi::c_void);
            f(slot, SEL_setClearColor(), clear);

            if let Some(resolve) = desc.resolve_texture {
                msg1_void(slot, SEL_setResolveTexture(), resolve.as_raw());
            }
        }
    }

    /// Configure the depth attachment.
    pub fn depth_attachment(&mut self, desc: DepthAttachmentDesc<'_>) {
        unsafe {
            let slot = msg0(self.raw, SEL_depthAttachment());
            msg1_void(slot, SEL_setTexture(), desc.texture.as_raw());
            msg1_uint_void(slot, SEL_setLoadAction(), desc.load_action as NSUInteger);
            msg1_uint_void(slot, SEL_setStoreAction(), desc.store_action as NSUInteger);
            type F = unsafe extern "C" fn(ObjcId, ObjcSel, f64);
            let f: F = std::mem::transmute(objc_msgSend as *const std::ffi::c_void);
            f(slot, SEL_setClearDepth(), desc.clear_depth);
        }
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for RenderPassDescriptor {
    #[mutants::skip]
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}
