//! Data prefetch hints via ARM PRFM instructions.
//!
//! These are pure hints — the CPU is free to ignore them.  On Apple
//! Silicon the hardware prefetcher is very aggressive, so explicit
//! prefetching is only useful for irregular access patterns.

/// Prefetch `ptr` into L1 data cache for reading (PLDL1KEEP).
///
/// # Safety
///
/// `ptr` must be a valid, dereferenceable address (or the hint is ignored
/// by hardware, but the intent should still be a valid access).
#[inline(always)]
pub unsafe fn prefetch_l1(ptr: *const u8) {
    #[cfg(target_arch = "aarch64")]
    {
        core::arch::asm!(
            "prfm pldl1keep, [{ptr}]",
            ptr = in(reg) ptr,
            options(nostack, preserves_flags),
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = ptr;
    }
}

/// Prefetch `ptr` into L2 data cache for reading (PLDL2KEEP).
///
/// # Safety
///
/// See [`prefetch_l1`].
#[inline(always)]
pub unsafe fn prefetch_l2(ptr: *const u8) {
    #[cfg(target_arch = "aarch64")]
    {
        core::arch::asm!(
            "prfm pldl2keep, [{ptr}]",
            ptr = in(reg) ptr,
            options(nostack, preserves_flags),
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = ptr;
    }
}

/// Prefetch `ptr` into L1 data cache for writing (PSTL1KEEP).
///
/// # Safety
///
/// See [`prefetch_l1`].
#[inline(always)]
pub unsafe fn prefetch_l1_write(ptr: *mut u8) {
    #[cfg(target_arch = "aarch64")]
    {
        core::arch::asm!(
            "prfm pstl1keep, [{ptr}]",
            ptr = in(reg) ptr,
            options(nostack, preserves_flags),
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = ptr;
    }
}
