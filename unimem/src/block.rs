use crate::ffi::*;
use crate::MemError;
use std::ptr::NonNull;

mod creation;
#[cfg(test)]
mod tests;

/// Checked numeric properties for one raw IOSurface row.
///
/// A plan carries numbers only and confers no reservation. Each `open` creates
/// an independent allocation. Callers account for `allocation_size()` plus their
/// own control/metadata before opening. CF/kernel bookkeeping and residency are
/// separate from that API-visible extent. A plan neither initializes contents
/// nor certifies immutable publication.
///
/// Fields can only be constructed through [`Block::plan`].
/// ```compile_fail,E0451
/// use unimem::BlockPlan;
/// let plan = BlockPlan { requested: 1, row_bytes: 1, allocation_size: 1 };
/// ```
/// There is no unchecked constructor.
/// ```compile_fail,E0599
/// use unimem::BlockPlan;
/// let plan = unsafe { BlockPlan::new_unchecked(1, 1, 1) };
/// ```
#[derive(Clone, Copy, Debug)]
pub struct BlockPlan {
    requested: usize,
    row_bytes: usize,
    allocation_size: usize,
}

impl BlockPlan {
    /// Logical byte width, before row padding.
    pub fn requested_size(&self) -> usize {
        self.requested
    }

    /// Row stride including native property alignment.
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// Exact API-visible backing extent required by this plan.
    pub fn allocation_size(&self) -> usize {
        self.allocation_size
    }

    /// Create and lock an independent surface using these exact properties.
    ///
    /// A returned extent differing from this plan is a creation error. Returned
    /// native failures release all temporary owners; opaque framework allocation
    /// failures have no universal recoverability guarantee.
    pub fn open(&self) -> Result<Block, MemError> {
        creation::open(self)
    }

    fn checked(
        requested: usize,
        row_alignment: usize,
        alloc_alignment: usize,
    ) -> Result<Self, MemError> {
        checked_size(requested)?;
        let row_bytes = aligned_size(requested, row_alignment)?;
        let allocation_size = aligned_size(row_bytes, alloc_alignment)?;
        Ok(Self {
            requested,
            row_bytes,
            allocation_size,
        })
    }

    fn check_native_alignment(&self, row: usize, allocation: usize) -> Result<(), MemError> {
        if row != self.row_bytes || allocation != self.allocation_size {
            return Err(MemError::BlockAlignmentInvalid);
        }
        Ok(())
    }
}

fn checked_size(size: usize) -> Result<i64, MemError> {
    if size == 0 {
        return Err(MemError::ZeroSize);
    }
    isize::try_from(size).map_err(|_| MemError::SizeOverflow)?;
    i64::try_from(size).map_err(|_| MemError::SizeOverflow)
}

fn aligned_size(size: usize, alignment: usize) -> Result<usize, MemError> {
    if alignment == 0 {
        return Err(MemError::BlockAlignmentInvalid);
    }
    let remainder = size % alignment;
    let aligned = if remainder == 0 {
        size
    } else {
        size.checked_add(alignment - remainder)
            .ok_or(MemError::SizeOverflow)?
    };
    checked_size(aligned)?;
    Ok(aligned)
}

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

// The owning mapping can move/share. CPU views require initialization and
// CPU/raw/device exclusion; the creation-time lock adds no per-access operation.
unsafe impl Send for Block {}
unsafe impl Sync for Block {}

impl Block {
    /// Check a raw row's native alignment and integer/slice bounds.
    ///
    /// Planning queries native properties without creating CF objects or backing
    /// storage. Logical width stays `size`; native row padding may increase the
    /// allocation. Callers reserve from the returned plan before opening it.
    pub fn plan(size: usize) -> Result<BlockPlan, MemError> {
        checked_size(size)?;
        // SAFETY: These SDK constants are borrowed CFStringRefs. Only checked,
        // representable numeric arguments reach the native alignment helper.
        unsafe {
            let plan = BlockPlan::checked(
                size,
                IOSurfaceGetPropertyAlignment(kIOSurfaceBytesPerRow),
                IOSurfaceGetPropertyAlignment(kIOSurfaceAllocSize),
            )?;
            plan.check_native_alignment(
                IOSurfaceAlignProperty(kIOSurfaceBytesPerRow, size),
                IOSurfaceAlignProperty(kIOSurfaceAllocSize, plan.row_bytes),
            )?;
            Ok(plan)
        }
    }

    /// Open a pinned memory block of at least `size` bytes.
    ///
    /// The block is locked immediately — `address()` is valid until drop.
    /// Allocation is lazy: kernel reserves address space, pages backed on first touch.
    pub fn open(size: usize) -> Result<Self, MemError> {
        Self::plan(size)?.open()
    }

    /// Memory address. Always valid (block is locked).
    #[inline(always)]
    pub fn address(&self) -> *mut u8 {
        self.va.as_ptr()
    }

    /// API-visible allocation size in bytes, including native row padding.
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
    // Zero-cost views over the owned mapping; creation does not initialize it.

    /// View all `size()` bytes, including native padding.
    ///
    /// # Safety
    /// Initialize the entire returned extent before forming this view, even to
    /// inspect its length or pointer. Exclude overlapping CPU/raw/device writes
    /// for the borrow's lifetime, including from other threads or retained imports.
    /// Establish prior device completion and visibility before CPU access.
    ///
    /// ```compile_fail,E0133
    /// fn view(block: &unimem::Block) { let _ = block.as_bytes(); }
    /// ```
    #[inline(always)]
    pub unsafe fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.va.as_ptr(), self.size) }
    }

    /// View all `size()` bytes mutably, including native padding.
    ///
    /// # Safety
    /// Initialize the entire returned extent before forming this view, even to
    /// inspect its length or pointer. Exclude every conflicting CPU/raw/device
    /// access for the borrow's lifetime, including other threads/retained imports.
    /// Establish prior device completion and visibility before CPU access.
    ///
    /// ```compile_fail,E0133
    /// fn view(block: &mut unimem::Block) { let _ = block.as_bytes_mut(); }
    /// ```
    /// ```compile_fail,E0596
    /// fn shared(block: &unimem::Block) { let _ = unsafe { block.as_bytes_mut() }; }
    /// ```
    /// ```compile_fail,E0499
    /// fn aliases(block: &mut unimem::Block) {
    ///     let a = unsafe { block.as_bytes_mut() };
    ///     let b = unsafe { block.as_bytes_mut() };
    ///     a[0] = b[0];
    /// }
    /// ```
    /// ```compile_fail,E0502
    /// fn aliases(block: &mut unimem::Block) {
    ///     let a = unsafe { block.as_bytes() };
    ///     let b = unsafe { block.as_bytes_mut() };
    ///     b[0] = a[0];
    /// }
    /// ```
    #[inline(always)]
    pub unsafe fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.va.as_ptr(), self.size) }
    }

    /// View as f32 slice. Length = size / 4.
    ///
    /// # Safety
    /// Initialize all `size()/4*4` returned bytes, including padding, before
    /// forming this view, even for length/pointer inspection. Exclude overlapping
    /// CPU/raw/device writes throughout the borrow, across threads and retained
    /// imports. Establish prior device completion and visibility before access.
    ///
    /// ```compile_fail,E0133
    /// fn view(block: &unimem::Block) { let _ = block.as_f32(); }
    /// ```
    #[inline(always)]
    pub unsafe fn as_f32(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.va.as_ptr() as *const f32, self.size / 4) }
    }

    /// View as mutable f32 slice. Length = size / 4.
    ///
    /// # Safety
    /// Initialize all `size()/4*4` returned bytes, including padding, before
    /// forming this view, even for length/pointer inspection. Exclude every
    /// conflicting CPU/raw/device access throughout the borrow, across threads
    /// and retained imports. Establish prior device completion and visibility.
    ///
    /// ```compile_fail,E0133
    /// fn view(block: &mut unimem::Block) { let _ = block.as_f32_mut(); }
    /// ```
    #[inline(always)]
    pub unsafe fn as_f32_mut(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.va.as_ptr() as *mut f32, self.size / 4) }
    }

    /// View as u16 slice (fp16). Length = size / 2.
    ///
    /// # Safety
    /// Initialize all `size()/2*2` returned bytes, including padding, before
    /// forming this view, even for length/pointer inspection. Exclude overlapping
    /// CPU/raw/device writes throughout the borrow, across threads and retained
    /// imports. Establish prior device completion and visibility before access.
    ///
    /// ```compile_fail,E0133
    /// fn view(block: &unimem::Block) { let _ = block.as_u16(); }
    /// ```
    #[inline(always)]
    pub unsafe fn as_u16(&self) -> &[u16] {
        unsafe { std::slice::from_raw_parts(self.va.as_ptr() as *const u16, self.size / 2) }
    }

    /// View as mutable u16 slice (fp16). Length = size / 2.
    ///
    /// # Safety
    /// Initialize all `size()/2*2` returned bytes, including padding, before
    /// forming this view, even for length/pointer inspection. Exclude every
    /// conflicting CPU/raw/device access throughout the borrow, across threads
    /// and retained imports. Establish prior device completion and visibility.
    ///
    /// ```compile_fail,E0133
    /// fn view(block: &mut unimem::Block) { let _ = block.as_u16_mut(); }
    /// ```
    #[inline(always)]
    pub unsafe fn as_u16_mut(&mut self) -> &mut [u16] {
        unsafe { std::slice::from_raw_parts_mut(self.va.as_ptr() as *mut u16, self.size / 2) }
    }
}

impl Drop for Block {
    fn drop(&mut self) {
        // SAFETY: Successful creation transfers one owned, locked surface here.
        unsafe { creation::unlock_and_release(self.raw) }
    }
}
