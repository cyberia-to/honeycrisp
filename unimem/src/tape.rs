use crate::block::Block;
use crate::MemError;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Tape allocator over a pinned memory block.
///
/// Like a Turing machine tape: head moves forward, writes sequentially.
/// `clear()` rewinds the head to the start. Pages stay pinned.
///
/// ~1ns take. Instant clear.
pub struct Tape {
    block: Block,
    head: AtomicUsize,
    total: usize,
}

// Block is Send+Sync, head is atomic.
unsafe impl Send for Tape {}
unsafe impl Sync for Tape {}

impl Tape {
    /// Start a new tape of `size` bytes.
    pub fn start(size: usize) -> Result<Self, MemError> {
        let block = Block::open(size)?;
        let total = block.size();
        Ok(Tape {
            block,
            head: AtomicUsize::new(0),
            total,
        })
    }

    /// Start a tape and touch one byte at a fixed 16 KiB stride.
    pub fn start_warm(size: usize) -> Result<Self, MemError> {
        let mut tape = Self::start(size)?;
        // SAFETY: Fresh unpublished owner; no takes, references or device aliases.
        unsafe { tape.warm() };
        Ok(tape)
    }

    /// Zero offsets 0,16384,... below total; preserve other bytes and the cursor.
    /// This fixed stride is independent of runtime VM page size and does not
    /// initialize the whole allocation.
    ///
    /// # Safety
    /// Exclude conflicting references, raw-pointer users and device access during
    /// these writes, including across threads or retained imports. No live typed
    /// value may be invalidated by the overwritten bytes. Establish prior device
    /// completion and visibility before writing.
    ///
    /// ```compile_fail,E0133
    /// fn warm(tape: &mut unimem::Tape) { tape.warm(); }
    /// ```
    /// ```compile_fail,E0596
    /// fn shared(tape: &unimem::Tape) { unsafe { tape.warm() }; }
    /// ```
    #[mutants::skip]
    pub unsafe fn warm(&mut self) {
        let ptr = self.block.address();
        let page = 16384; // Fixed warming stride, not a queried VM page size.
        let pages = self.total / page;
        unsafe {
            for i in 0..pages {
                std::ptr::write_volatile(ptr.add(i * page), 0);
            }
            if !self.total.is_multiple_of(page) {
                std::ptr::write_volatile(ptr.add(pages * page), 0);
            }
        }
    }

    /// Take `size` bytes with given alignment.
    ///
    /// Returns memory address, or None if tape is full.
    /// Lock-free: single compare_exchange loop. ~1ns.
    #[inline]
    pub fn take(&self, size: usize, align: usize) -> Option<*mut u8> {
        debug_assert!(align.is_power_of_two(), "alignment must be power of 2");
        if size == 0 {
            return None;
        }
        let mask = align - 1;
        loop {
            let current = self.head.load(Ordering::Relaxed);
            let aligned = (current + mask) & !mask;
            let new_head = aligned.checked_add(size)?;
            if new_head > self.total {
                return None;
            }
            if self
                .head
                .compare_exchange_weak(current, new_head, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return Some(unsafe { self.block.address().add(aligned) });
            }
        }
    }

    /// Take space for one value of type T.
    #[inline]
    pub fn take_one<T>(&self) -> Option<*mut T> {
        self.take(std::mem::size_of::<T>(), std::mem::align_of::<T>())
            .map(|p| p as *mut T)
    }

    /// Rewind tape to the start. All previous takes are invalidated.
    /// Instant. Pages stay pinned. Does NOT zero memory.
    #[inline]
    pub fn clear(&self) {
        self.head.store(0, Ordering::Release);
    }

    /// Bytes used.
    #[inline]
    pub fn used(&self) -> usize {
        self.head.load(Ordering::Relaxed)
    }

    /// Bytes free.
    #[inline]
    pub fn free(&self) -> usize {
        self.total.saturating_sub(self.used())
    }

    /// Total bytes.
    #[inline]
    pub fn total(&self) -> usize {
        self.total
    }

    /// Does this tape own the given address?
    #[inline]
    pub fn owns(&self, addr: *const u8) -> bool {
        let base = self.block.address() as usize;
        let a = addr as usize;
        a >= base && a < base + self.total
    }

    /// Access the backing block.
    #[inline]
    pub fn block(&self) -> &Block {
        &self.block
    }
}
