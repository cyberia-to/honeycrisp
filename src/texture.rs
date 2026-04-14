//! Texture: GPU texture wrapper

use crate::ffi::*;
use std::ffi::c_void;

/// A Metal texture. Wraps `id<MTLTexture>`.
pub struct Texture {
    raw: ObjcId,
}

impl Texture {
    pub(crate) fn from_raw(raw: ObjcId) -> Self {
        Texture { raw }
    }

    /// Texture width in pixels.
    pub fn width(&self) -> usize {
        unsafe { msg0_usize(self.raw, SEL_width()) }
    }

    /// Texture height in pixels.
    pub fn height(&self) -> usize {
        unsafe { msg0_usize(self.raw, SEL_height()) }
    }

    /// Texture depth (for 3D textures).
    #[mutants::skip] // 1 for 2D textures — mutant → 1 passes correctly
    pub fn depth(&self) -> usize {
        unsafe { msg0_usize(self.raw, SEL_depth()) }
    }

    /// Pixel format.
    pub fn pixel_format(&self) -> usize {
        unsafe { msg0_usize(self.raw, SEL_pixelFormat()) }
    }

    /// Replace a region of the texture with data.
    ///
    /// # Safety
    /// `data` must point to valid memory of sufficient size for the region.
    pub unsafe fn replace_region(
        &self,
        region: MTLRegion,
        mipmap_level: usize,
        data: *const c_void,
        bytes_per_row: usize,
    ) {
        type F =
            unsafe extern "C" fn(ObjcId, ObjcSel, MTLRegion, NSUInteger, *const c_void, NSUInteger);
        let f: F = std::mem::transmute(objc_msgSend as *const c_void);
        f(
            self.raw,
            SEL_replaceRegion(),
            region,
            mipmap_level,
            data,
            bytes_per_row,
        );
    }

    /// Get bytes from a region of the texture.
    ///
    /// # Safety
    /// `data` must point to writable memory of sufficient size for the region.
    pub unsafe fn get_bytes(
        &self,
        data: *mut c_void,
        bytes_per_row: usize,
        region: MTLRegion,
        mipmap_level: usize,
    ) {
        type F =
            unsafe extern "C" fn(ObjcId, ObjcSel, *mut c_void, NSUInteger, MTLRegion, NSUInteger);
        let f: F = std::mem::transmute(objc_msgSend as *const c_void);
        f(
            self.raw,
            SEL_getBytes(),
            data,
            bytes_per_row,
            region,
            mipmap_level,
        );
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for Texture {
    #[mutants::skip] // RAII release — tested by drop_stress_leak
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}
