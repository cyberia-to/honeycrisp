use crate::ffi::*;
use crate::MemError;
use std::ffi::c_void;
use std::ptr::{self, NonNull};

/// Pinned shared memory block backed by IOSurface.
///
/// Visible to CPU, GPU (Metal wrap), AMX (CPU pointer), and ANE (private API).
/// Locked once at creation — address is stable for the entire lifetime.
pub struct Block {
    raw: IOSurfaceRef,
    va: NonNull<u8>,
    size: usize,
    id: u32,
}

// Immutable after creation. Lock held for lifetime. No mutable state.
unsafe impl Send for Block {}
unsafe impl Sync for Block {}

impl Block {
    /// Open a pinned memory block of `size` bytes.
    ///
    /// The block is locked immediately — `address()` is valid until drop.
    /// Allocation is lazy: kernel reserves address space, pages backed on first touch.
    pub fn open(size: usize) -> Result<Self, MemError> {
        if size == 0 {
            return Err(MemError::ZeroSize);
        }

        unsafe {
            let dict = CFDictionaryCreateMutable(
                ptr::null(),
                0,
                &kCFTypeDictionaryKeyCallBacks as *const c_void,
                &kCFTypeDictionaryValueCallBacks as *const c_void,
            );

            let sz = size as i64;
            CFDictionarySetValue(dict, cf_str("IOSurfaceWidth") as _, cf_i64(sz));
            CFDictionarySetValue(dict, cf_str("IOSurfaceHeight") as _, cf_i64(1));
            CFDictionarySetValue(dict, cf_str("IOSurfaceBytesPerElement") as _, cf_i64(1));
            CFDictionarySetValue(dict, cf_str("IOSurfaceBytesPerRow") as _, cf_i64(sz));
            CFDictionarySetValue(dict, cf_str("IOSurfaceAllocSize") as _, cf_i64(sz));
            CFDictionarySetValue(dict, cf_str("IOSurfacePixelFormat") as _, cf_i64(0));

            let raw = IOSurfaceCreate(dict);
            CFRelease(dict as CFTypeRef);

            if raw.is_null() {
                return Err(MemError::BlockCreateFailed);
            }

            let kr = IOSurfaceLock(raw, 0, ptr::null_mut());
            if kr != KERN_SUCCESS {
                CFRelease(raw as CFTypeRef);
                return Err(MemError::BlockLockFailed(kr));
            }

            let base = IOSurfaceGetBaseAddress(raw);
            let va = match NonNull::new(base as *mut u8) {
                Some(p) => p,
                None => {
                    IOSurfaceUnlock(raw, 0, ptr::null_mut());
                    CFRelease(raw as CFTypeRef);
                    return Err(MemError::BlockCreateFailed);
                }
            };

            let actual_size = IOSurfaceGetAllocSize(raw);
            let id = IOSurfaceGetID(raw);

            Ok(Block {
                raw,
                va,
                size: actual_size,
                id,
            })
        }
    }

    /// Memory address. Always valid (block is locked).
    #[inline(always)]
    pub fn address(&self) -> *mut u8 {
        self.va.as_ptr()
    }

    /// Size in bytes.
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Global IOSurface ID for cross-process sharing.
    #[inline(always)]
    pub fn id(&self) -> u32 {
        self.id
    }

    /// System handle for ANE (rane) and GPU (aruminium) integration.
    #[inline(always)]
    pub fn handle(&self) -> IOSurfaceRef {
        self.raw
    }

    // ── Typed slice accessors ──
    // Zero-cost views over the same physical memory.
    // Caller is responsible for ensuring the data is valid for the requested type.

    /// View as byte slice.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.va.as_ptr(), self.size) }
    }

    /// View as mutable byte slice.
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    pub fn as_bytes_mut(&self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.va.as_ptr(), self.size) }
    }

    /// View as f32 slice. Length = size / 4.
    #[inline(always)]
    pub fn as_f32(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.va.as_ptr() as *const f32, self.size / 4) }
    }

    /// View as mutable f32 slice. Length = size / 4.
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    pub fn as_f32_mut(&self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.va.as_ptr() as *mut f32, self.size / 4) }
    }

    /// View as u16 slice (fp16). Length = size / 2.
    #[inline(always)]
    pub fn as_u16(&self) -> &[u16] {
        unsafe { std::slice::from_raw_parts(self.va.as_ptr() as *const u16, self.size / 2) }
    }

    /// View as mutable u16 slice (fp16). Length = size / 2.
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    pub fn as_u16_mut(&self) -> &mut [u16] {
        unsafe { std::slice::from_raw_parts_mut(self.va.as_ptr() as *mut u16, self.size / 2) }
    }
}

impl Drop for Block {
    #[mutants::skip] // RAII release — kernel-level cleanup, not observable in tests
    fn drop(&mut self) {
        unsafe {
            IOSurfaceUnlock(self.raw, 0, ptr::null_mut());
            CFRelease(self.raw as CFTypeRef);
        }
    }
}
