//! Buffer: zero-copy shared GPU memory (replaces IOSurface for Metal)

use crate::ffi::*;
use std::ffi::c_void;

/// A Metal buffer with shared CPU/GPU memory. Wraps `id<MTLBuffer>`.
///
/// Created via `Gpu::buffer()`. Uses `MTLResourceStorageModeShared`
/// so CPU and GPU share the same physical memory — no copies, no lock/unlock.
pub struct Buffer {
    raw: ObjcId,
    size: usize,
    /// Cached contents pointer — valid for buffer lifetime with shared storage mode.
    ptr: *mut c_void,
}

impl Buffer {
    pub(crate) fn from_raw(raw: ObjcId, size: usize) -> Self {
        let ptr = unsafe { msg0_ptr(raw, SEL_contents()) };
        Buffer { raw, size, ptr }
    }

    /// Create from raw without querying contents (for Private storage mode).
    pub(crate) fn from_raw_private(raw: ObjcId, size: usize) -> Self {
        Buffer {
            raw,
            size,
            ptr: std::ptr::null_mut(),
        }
    }

    /// Whether this buffer has CPU-accessible memory (Shared mode).
    #[inline(always)]
    pub fn is_shared(&self) -> bool {
        !self.ptr.is_null()
    }

    /// Raw pointer to buffer contents. Cached — no ObjC call after first access.
    /// Restricted to crate — external code uses closure API (read, read_f32, etc.)
    /// to prevent pointer from outliving the buffer.
    #[inline(always)]
    #[allow(dead_code)]
    #[mutants::skip] // internal accessor, not exercised via public API
    pub(crate) fn contents(&self) -> *mut c_void {
        self.ptr
    }

    /// Read access to buffer data via closure.
    ///
    /// # Panics
    /// Panics if called on a private-mode buffer (not CPU-accessible).
    #[inline]
    #[track_caller]
    pub fn read<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&[u8]) -> R,
    {
        assert!(self.is_shared(), "read called on private buffer");
        let slice = unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.size) };
        f(slice)
    }

    /// Direct slice view into shared-mode buffer storage.
    ///
    /// Slice lifetime is tied to `&self`. Caller is responsible for not
    /// reading concurrently with in-flight GPU writes (typically: only call
    /// after a `wait` on the command buffer that wrote to this buffer).
    ///
    /// # Panics
    /// Panics if called on a private-mode buffer.
    #[inline]
    #[track_caller]
    pub fn as_bytes(&self) -> &[u8] {
        assert!(self.is_shared(), "as_bytes called on private buffer");
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.size) }
    }

    /// Write access to buffer data via closure.
    ///
    /// # Panics
    /// Panics if called on a private-mode buffer (not CPU-accessible).
    #[inline]
    #[track_caller]
    pub fn write<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut [u8]) -> R,
    {
        assert!(self.is_shared(), "write called on private buffer");
        let slice = unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.size) };
        f(slice)
    }

    /// Read access as f32 slice.
    ///
    /// Length = `size / 4` (trailing bytes not divisible by 4 are ignored).
    ///
    /// # Panics
    /// Panics if called on a private-mode buffer or if contents pointer is not 4-byte aligned.
    #[inline]
    #[track_caller]
    pub fn read_f32<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&[f32]) -> R,
    {
        assert!(self.is_shared(), "read_f32 called on private buffer");
        let ptr = self.ptr as *const f32;
        assert!(ptr.is_aligned(), "buffer contents not 4-byte aligned");
        let len = self.size / 4;
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        f(slice)
    }

    /// Write access as f32 slice.
    ///
    /// Length = `size / 4` (trailing bytes not divisible by 4 are ignored).
    ///
    /// # Panics
    /// Panics if called on a private-mode buffer or if contents pointer is not 4-byte aligned.
    #[inline]
    #[track_caller]
    pub fn write_f32<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut [f32]) -> R,
    {
        assert!(self.is_shared(), "write_f32 called on private buffer");
        let ptr = self.ptr as *mut f32;
        assert!(ptr.is_aligned(), "buffer contents not 4-byte aligned");
        let len = self.size / 4;
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
        f(slice)
    }

    /// Allocation size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for Buffer {
    #[mutants::skip] // RAII release — tested by drop_stress_leak
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}
