use crate::MemError;
use std::ffi::c_void;
use std::ptr::NonNull;

// AHardwareBuffer lives in libandroid.so, present on every device since API 26.
// Only the BLOB path is used here: a linear byte buffer, no format, no stride.
#[repr(C)]
#[derive(Default)]
struct AHardwareBufferDesc {
    width: u32,
    height: u32,
    layers: u32,
    format: u32,
    usage: u64,
    stride: u32,
    rfu0: u32,
    rfu1: u64,
}

const AHARDWAREBUFFER_FORMAT_BLOB: u32 = 0x21;
const USAGE_CPU_READ_OFTEN: u64 = 0x3;
const USAGE_CPU_WRITE_OFTEN: u64 = 0x30;
const USAGE_GPU_DATA_BUFFER: u64 = 1 << 24;

#[link(name = "android")]
unsafe extern "C" {
    fn AHardwareBuffer_allocate(
        desc: *const AHardwareBufferDesc,
        out: *mut *mut c_void,
    ) -> i32;
    fn AHardwareBuffer_release(buffer: *mut c_void);
    fn AHardwareBuffer_lock(
        buffer: *mut c_void,
        usage: u64,
        fence: i32,
        rect: *const c_void,
        out_virtual_address: *mut *mut c_void,
    ) -> i32;
    fn AHardwareBuffer_unlock(buffer: *mut c_void, fence: *mut i32) -> i32;
}

/// Pinned shared memory block backed by an AHardwareBuffer (BLOB).
///
/// Locked once at creation and held locked for the block's lifetime, so
/// `address()` is stable — the same contract the IOSurface backend gives.
/// `handle()` is what a Vulkan device imports.
pub struct Block {
    raw: NonNull<c_void>,
    va: NonNull<u8>,
    size: usize,
}

// Immutable after creation. Lock held for lifetime. No mutable state.
unsafe impl Send for Block {}
unsafe impl Sync for Block {}

impl Block {
    /// Open a pinned memory block of `size` bytes.
    pub fn open(size: usize) -> Result<Self, MemError> {
        if size == 0 {
            return Err(MemError::ZeroSize);
        }
        let desc = AHardwareBufferDesc {
            width: size as u32,
            height: 1,
            layers: 1,
            format: AHARDWAREBUFFER_FORMAT_BLOB,
            usage: USAGE_CPU_READ_OFTEN | USAGE_CPU_WRITE_OFTEN | USAGE_GPU_DATA_BUFFER,
            ..Default::default()
        };
        unsafe {
            let mut raw: *mut c_void = std::ptr::null_mut();
            if AHardwareBuffer_allocate(&desc, &mut raw) != 0 || raw.is_null() {
                return Err(MemError::BlockCreateFailed);
            }
            let mut va: *mut c_void = std::ptr::null_mut();
            let rc = AHardwareBuffer_lock(
                raw,
                USAGE_CPU_READ_OFTEN | USAGE_CPU_WRITE_OFTEN,
                -1,
                std::ptr::null(),
                &mut va,
            );
            if rc != 0 || va.is_null() {
                AHardwareBuffer_release(raw);
                return Err(MemError::BlockLockFailed(rc));
            }
            Ok(Block {
                raw: NonNull::new_unchecked(raw),
                va: NonNull::new_unchecked(va as *mut u8),
                size,
            })
        }
    }

    /// Stable base address, valid until drop.
    #[inline(always)]
    pub fn address(&self) -> *mut u8 {
        self.va.as_ptr()
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Importable extent — BLOB buffers are exactly the requested size.
    pub fn alloc_size(&self) -> usize {
        self.size
    }

    /// The `AHardwareBuffer*` this block owns. Borrowed: the importer must
    /// not release it.
    pub fn handle(&self) -> *mut c_void {
        self.raw.as_ptr()
    }

    /// Backend-neutral id (low bits of the handle on this backend).
    pub fn id(&self) -> u32 {
        self.raw.as_ptr() as usize as u32
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.va.as_ptr(), self.size) }
    }

    #[allow(clippy::mut_from_ref)]
    pub fn as_bytes_mut(&self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.va.as_ptr(), self.size) }
    }

    pub fn as_f32(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.va.as_ptr() as *const f32, self.size / 4) }
    }

    #[allow(clippy::mut_from_ref)]
    pub fn as_f32_mut(&self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.va.as_ptr() as *mut f32, self.size / 4) }
    }

    pub fn as_u16(&self) -> &[u16] {
        unsafe { std::slice::from_raw_parts(self.va.as_ptr() as *const u16, self.size / 2) }
    }

    #[allow(clippy::mut_from_ref)]
    pub fn as_u16_mut(&self) -> &mut [u16] {
        unsafe { std::slice::from_raw_parts_mut(self.va.as_ptr() as *mut u16, self.size / 2) }
    }
}

impl Drop for Block {
    fn drop(&mut self) {
        unsafe {
            let mut fence: i32 = -1;
            AHardwareBuffer_unlock(self.raw.as_ptr(), &mut fence);
            AHardwareBuffer_release(self.raw.as_ptr());
        }
    }
}
