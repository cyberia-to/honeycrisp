use crate::MemError;
use std::ptr::NonNull;

/// Pinned shared memory block backed by an anonymous memfd mapping.
///
/// The mapping is created once and lives for the block's lifetime —
/// `address()` is stable, exactly like the IOSurface backend. `fd()` is the
/// export handle for GPU import.
pub struct Block {
    fd: libc::c_int,
    va: NonNull<u8>,
    size: usize,
    alloc: usize,
}

// Immutable after creation. Mapping held for lifetime. No mutable state.
unsafe impl Send for Block {}
unsafe impl Sync for Block {}

impl Block {
    /// Open a pinned memory block of `size` bytes.
    ///
    /// Size is rounded up to the page size (16 KB pages on recent Android —
    /// the kernel decides, the round-trip through ftruncate+mmap follows it).
    pub fn open(size: usize) -> Result<Self, MemError> {
        if size == 0 {
            return Err(MemError::ZeroSize);
        }
        // Allocation rounds up to the largest page size in the fleet (16 KB
        // on recent Android kernels): Vulkan host-pointer import requires
        // the imported range to be alignment-multiple, and whole pages exist
        // anyway.
        let alloc = size.div_ceil(16384) * 16384;
        unsafe {
            let fd = libc::memfd_create(c"unimem".as_ptr(), libc::MFD_CLOEXEC);
            if fd < 0 {
                return Err(MemError::BlockCreateFailed);
            }
            if libc::ftruncate(fd, alloc as libc::off_t) != 0 {
                libc::close(fd);
                return Err(MemError::BlockCreateFailed);
            }
            let va = libc::mmap(
                std::ptr::null_mut(),
                alloc,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );
            if va == libc::MAP_FAILED {
                // Bionic names it __errno; std reads the right one everywhere.
                let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(-1);
                libc::close(fd);
                return Err(MemError::BlockLockFailed(errno));
            }
            Ok(Block { fd, va: NonNull::new_unchecked(va as *mut u8), size, alloc })
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

    /// Whole-page allocation backing this block (≥ `size()`, page-multiple).
    /// This is the importable extent for GPU host-pointer import.
    pub fn alloc_size(&self) -> usize {
        self.alloc
    }

    /// The export handle: a memfd file descriptor. The block keeps ownership;
    /// dup(2) it before handing it to an API that closes what it imports.
    pub fn fd(&self) -> i32 {
        self.fd
    }

    /// Backend-neutral id (the fd number on this backend).
    pub fn id(&self) -> u32 {
        self.fd as u32
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
            libc::munmap(self.va.as_ptr() as *mut libc::c_void, self.alloc);
            libc::close(self.fd);
        }
    }
}
