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

    /// Create an IOSurface with the ANEC interleaved geometry required by
    /// models compiled via `Program::compile_anec`.
    ///
    /// ANEC layout: channels are grouped in 4s; each group occupies a 64-byte
    /// row. Total rows = ceil(channels / 4), total bytes = rows × 64.
    /// This geometry must match the `InputRowStride` / `OutputRowStride` = 64
    /// and `Interleave` = 1 parameters in the ANEC IR net.plist.
    pub fn with_anec_channels(channels: usize) -> Result<Self, AneError> {
        if channels == 0 {
            return Err(AneError::SurfaceCreationFailed(
                "channels must be > 0".into(),
            ));
        }
        const STRIDE: usize = 64;
        let groups = channels.div_ceil(4);
        let total = groups * STRIDE;
        if total > Self::MAX_SURFACE_BYTES {
            return Err(AneError::SurfaceCreationFailed(format!(
                "{} bytes exceeds limit",
                total
            )));
        }
        unsafe {
            let dict = CFDictionaryCreateMutable(
                ptr::null(),
                0,
                &kCFTypeDictionaryKeyCallBacks as *const c_void,
                &kCFTypeDictionaryValueCallBacks as *const c_void,
            );
            CFDictionarySetValue(dict, cf_str("IOSurfaceWidth") as _, cf_num(STRIDE as i32));
            CFDictionarySetValue(dict, cf_str("IOSurfaceHeight") as _, cf_num(groups as i32));
            CFDictionarySetValue(dict, cf_str("IOSurfaceBytesPerElement") as _, cf_num(1));
            CFDictionarySetValue(
                dict,
                cf_str("IOSurfaceBytesPerRow") as _,
                cf_num(STRIDE as i32),
            );
            // Do NOT set IOSurfaceAllocSize: the kernel default (page-aligned, ≥ 16KB)
            // is required by the ANE hardware. Forcing alloc=groups*64 causes
            // ANEProgramProcessRequestDirect status=0x1d (inference error).
            let raw = IOSurfaceCreate(dict);
            if raw.is_null() {
                return Err(AneError::SurfaceCreationFailed(format!(
                    "{} channels ({} bytes)",
                    channels, total
                )));
            }
            let size = IOSurfaceGetAllocSize(raw);
            Ok(Buffer { raw, size })
        }
    }

    /// Byte stride between channel groups in an ANEC interleaved surface.
    pub const ANEC_GROUP_STRIDE: usize = 64;

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

    /// Lock surface (read-only), call closure with f32 slice, unlock.
    pub fn read_f32<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&[f32]) -> R,
    {
        unsafe {
            IOSurfaceLock(self.raw, 1, ptr::null_mut());
            let base = IOSurfaceGetBaseAddress(self.raw) as *const f32;
            let len = self.size / 4;
            let slice = std::slice::from_raw_parts(base, len);
            let result = f(slice);
            IOSurfaceUnlock(self.raw, 1, ptr::null_mut());
            result
        }
    }

    /// Lock surface (read-only), call closure with i16 slice, unlock.
    pub fn read_i16<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&[i16]) -> R,
    {
        unsafe {
            IOSurfaceLock(self.raw, 1, ptr::null_mut());
            let base = IOSurfaceGetBaseAddress(self.raw) as *const i16;
            let len = self.size / 2;
            let slice = std::slice::from_raw_parts(base, len);
            let result = f(slice);
            IOSurfaceUnlock(self.raw, 1, ptr::null_mut());
            result
        }
    }

    /// Lock surface (read-only), call closure with i8 slice, unlock.
    pub fn read_i8<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&[i8]) -> R,
    {
        unsafe {
            IOSurfaceLock(self.raw, 1, ptr::null_mut());
            let base = IOSurfaceGetBaseAddress(self.raw) as *const i8;
            let slice = std::slice::from_raw_parts(base, self.size);
            let result = f(slice);
            IOSurfaceUnlock(self.raw, 1, ptr::null_mut());
            result
        }
    }

    /// Lock surface (read-only), call closure with i32 slice, unlock.
    pub fn read_i32<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&[i32]) -> R,
    {
        unsafe {
            IOSurfaceLock(self.raw, 1, ptr::null_mut());
            let base = IOSurfaceGetBaseAddress(self.raw) as *const i32;
            let len = self.size / 4;
            let slice = std::slice::from_raw_parts(base, len);
            let result = f(slice);
            IOSurfaceUnlock(self.raw, 1, ptr::null_mut());
            result
        }
    }

    /// Lock surface (read-only), call closure with raw byte slice, unlock.
    pub fn read_bytes<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&[u8]) -> R,
    {
        unsafe {
            IOSurfaceLock(self.raw, 1, ptr::null_mut());
            let base = IOSurfaceGetBaseAddress(self.raw) as *const u8;
            let slice = std::slice::from_raw_parts(base, self.size);
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
