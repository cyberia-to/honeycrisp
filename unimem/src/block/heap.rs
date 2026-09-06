//! Pinned memory block backed by a plain aligned heap allocation.
//!
//! The fallback for platforms with no export handle story yet (Windows
//! today). Same contract as the other backends where it can be kept:
//! `address()` is stable for the block's lifetime, the whole extent is
//! page-aligned. What it cannot offer is an fd — `fd()` returns -1, and a
//! GPU import path on this platform must copy rather than alias. CPU
//! semantics (the part every caller relies on) are identical.

use crate::MemError;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ptr::NonNull;

/// The fleet-wide page size: allocations stay comparable across backends.
const PAGE: usize = 16 * 1024;

pub struct Block {
    va: NonNull<u8>,
    size: usize,
    alloc: usize,
}

// Immutable after creation. Allocation held for lifetime. No mutable state.
unsafe impl Send for Block {}
unsafe impl Sync for Block {}

impl Block {
    /// Open a pinned memory block of `size` bytes.
    pub fn open(size: usize) -> Result<Self, MemError> {
        if size == 0 {
            return Err(MemError::ZeroSize);
        }
        let alloc = size.div_ceil(PAGE) * PAGE;
        let layout = Layout::from_size_align(alloc, PAGE).map_err(|_| MemError::ZeroSize)?;
        // Zeroed like the mmap twin: a fresh block always reads as zeros.
        let va = unsafe { alloc_zeroed(layout) };
        match NonNull::new(va) {
            Some(va) => Ok(Block { va, size, alloc }),
            None => Err(MemError::BlockLockFailed(0)),
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
    pub fn alloc_size(&self) -> usize {
        self.alloc
    }

    /// No export handle on this backend: the heap has no fd to give.
    pub fn fd(&self) -> i32 {
        -1
    }

    /// Backend-neutral id (the address's low bits — unique while alive).
    pub fn id(&self) -> u32 {
        self.va.as_ptr() as usize as u32
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
            let layout = Layout::from_size_align_unchecked(self.alloc, PAGE);
            dealloc(self.va.as_ptr(), layout);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_reads_zeros_and_round_trips() {
        let b = Block::open(1000).expect("open");
        assert!(b.alloc_size() >= 1000);
        assert_eq!(b.alloc_size() % PAGE, 0);
        assert!(b.as_bytes().iter().all(|&x| x == 0));
        b.as_bytes_mut()[999] = 7;
        assert_eq!(b.as_bytes()[999], 7);
        assert_eq!(b.fd(), -1);
    }

    #[test]
    fn zero_size_refused() {
        assert!(Block::open(0).is_err());
    }
}
