//! Synchronisation primitives and memory barriers for Apple Silicon.
//!
//! Provides lightweight wrappers around ARM barrier instructions,
//! core-affinity via macOS QoS classes, and data prefetch hints.

pub mod affinity;
pub mod prefetch;

// ---------------------------------------------------------------------------
// Memory barriers
// ---------------------------------------------------------------------------

/// Data Memory Barrier — Inner Shareable domain.
///
/// Ensures all explicit data memory accesses before the barrier are
/// observed before any explicit data memory accesses after the barrier,
/// within the inner-shareable domain (same cluster).
///
/// # Safety
///
/// Barrier instructions do not access memory themselves, but mis-placed
/// barriers can hide data races.  The caller must understand the intended
/// ordering guarantee.
#[inline(always)]
pub unsafe fn barrier() {
    #[cfg(target_arch = "aarch64")]
    {
        core::arch::asm!("dmb ish", options(nostack, preserves_flags));
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
    }
}

/// Data Synchronization Barrier — Inner Shareable domain.
///
/// Stronger than DMB: also waits for all cache-maintenance, TLB,
/// and branch-predictor maintenance operations to complete.
///
/// # Safety
///
/// See [`barrier`].
#[inline(always)]
pub unsafe fn fence() {
    #[cfg(target_arch = "aarch64")]
    {
        core::arch::asm!("dsb ish", options(nostack, preserves_flags));
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
    }
}

/// Instruction Synchronization Barrier.
///
/// Flushes the pipeline and ensures all subsequent instructions are
/// fetched from cache or memory after the barrier completes.
///
/// # Safety
///
/// See [`barrier`].
#[inline(always)]
pub unsafe fn isb() {
    #[cfg(target_arch = "aarch64")]
    {
        core::arch::asm!("isb", options(nostack, preserves_flags));
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // No meaningful equivalent on other architectures.
    }
}

/// Wait For Event.
///
/// Places the PE into a low-power state until an event is signalled
/// (via SEV, interrupt, or debug event).
///
/// # Safety
///
/// Must only be used in contexts where stalling the core is acceptable.
#[inline(always)]
pub unsafe fn wait() {
    #[cfg(target_arch = "aarch64")]
    {
        core::arch::asm!("wfe", options(nostack, preserves_flags));
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        std::thread::yield_now();
    }
}

/// Signal Event.
///
/// Sends an event to all PEs in the inner-shareable domain, waking
/// any core that is in WFE state.
///
/// # Safety
///
/// Benign in isolation, but incorrect pairing with [`wait`] can cause
/// live-locks or missed wake-ups.
#[inline(always)]
pub unsafe fn wake() {
    #[cfg(target_arch = "aarch64")]
    {
        core::arch::asm!("sev", options(nostack, preserves_flags));
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // No meaningful equivalent on other architectures.
    }
}
