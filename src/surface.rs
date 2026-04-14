//! IOSurface wrapper for ANE tensor I/O

#![allow(dead_code)]

use crate::ffi::*;
use crate::AneError;
use std::ffi::c_void;
use std::ptr;

/// A shared-memory tensor buffer backed by IOSurface.
/// Used to pass fp16 data to/from the Apple Neural Engine.
pub struct Buffer {
    raw: IOSurfaceRef,
    size: usize,
}

impl Buffer {
    /// Maximum surface size: 256 MB (ANE practical limit).
    const MAX_SURFACE_BYTES: usize = 256 * 1024 * 1024;

    /// Create an IOSurface of the given byte size.
    pub fn new(bytes: usize) -> Result<Self, AneError> {
        if bytes == 0 || bytes > Self::MAX_SURFACE_BYTES {
            return Err(AneError::SurfaceCreationFailed(format!(
                "{} bytes (must be 1..={})",
                bytes,
                Self::MAX_SURFACE_BYTES
            )));
        }
        unsafe {
            let dict = CFDictionaryCreateMutable(
                ptr::null(),
                0,
                &kCFTypeDictionaryKeyCallBacks as *const c_void,
                &kCFTypeDictionaryValueCallBacks as *const c_void,
            );
            CFDictionarySetValue(dict, cf_str("IOSurfaceWidth") as _, cf_num(bytes as i32));
            CFDictionarySetValue(dict, cf_str("IOSurfaceHeight") as _, cf_num(1));
            CFDictionarySetValue(dict, cf_str("IOSurfaceBytesPerElement") as _, cf_num(1));
            CFDictionarySetValue(
                dict,
                cf_str("IOSurfaceBytesPerRow") as _,
                cf_num(bytes as i32),
            );
            CFDictionarySetValue(
                dict,
                cf_str("IOSurfaceAllocSize") as _,
                cf_num(bytes as i32),
            );
            CFDictionarySetValue(dict, cf_str("IOSurfacePixelFormat") as _, cf_num(0));
            let raw = IOSurfaceCreate(dict);
            if raw.is_null() {
                return Err(AneError::SurfaceCreationFailed(format!("{} bytes", bytes)));
            }
            let size = IOSurfaceGetAllocSize(raw);
            Ok(Buffer { raw, size })
        }
    }

    /// Create with ANE tensor shape `[1, channels, 1, spatial]` in fp16.
    pub fn with_shape(channels: usize, spatial: usize) -> Result<Self, AneError> {
        Self::new(channels * spatial * 2)
    }

    /// Lock surface, call closure with mutable fp16 slice, unlock.
    pub fn write<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut [u16]) -> R,
    {
        unsafe {
            IOSurfaceLock(self.raw, 0, ptr::null_mut());
            let base = IOSurfaceGetBaseAddress(self.raw) as *mut u16;
            let len = self.size / 2;
            let slice = std::slice::from_raw_parts_mut(base, len);
            let result = f(slice);
            IOSurfaceUnlock(self.raw, 0, ptr::null_mut());
            result
        }
    }

    /// Lock surface (read-only), call closure with fp16 slice, unlock.
    pub fn read<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&[u16]) -> R,
    {
        unsafe {
            IOSurfaceLock(self.raw, 1, ptr::null_mut()); // kIOSurfaceLockReadOnly = 1
            let base = IOSurfaceGetBaseAddress(self.raw) as *const u16;
            let len = self.size / 2;
            let slice = std::slice::from_raw_parts(base, len);
            let result = f(slice);
            IOSurfaceUnlock(self.raw, 1, ptr::null_mut());
            result
        }
    }

    /// Get the raw IOSurfaceRef for passing to `Program::run_direct()`.
    pub fn as_raw(&self) -> IOSurfaceRef {
        self.raw
    }

    /// IOSurface ID.
    pub fn id(&self) -> u32 {
        unsafe { IOSurfaceGetID(self.raw) }
    }

    /// Allocation size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            CFRelease(self.raw as CFTypeRef);
        }
    }
}

// fp16 conversion functions: use acpu::{fp16_to_f32, f32_to_fp16, cast_f16_f32, cast_f32_f16}
// Re-exported from acpu via lib.rs — single source of truth.
