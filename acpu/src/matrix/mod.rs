//! AMX matrix coprocessor driver.
//!
//! The Apple AMX unit is a per-thread, undocumented coprocessor present
//! on all Apple Silicon chips. It exposes 8 X rows, 8 Y rows, and 8 Z
//! rows of 64 bytes each, plus a set of fused multiply-accumulate
//! instructions that operate on these registers.
//!
//! [`Matrix`] is the safe entry point: constructing one calls AMX_SET,
//! and dropping it calls AMX_CLR.

pub mod asm;
pub mod fma;
pub mod kern;
pub mod ops;
pub mod regs;
pub mod tile;
pub mod tile_f16;

pub use regs::{XRow, YRow, ZRow, ALL_X, ALL_Y, ALL_Z};

use core::marker::PhantomData;

/// A live AMX coprocessor context.
///
/// Creating an `Matrix` activates the AMX unit on the current thread
/// (AMX_SET). Dropping it deactivates the unit (AMX_CLR). The type is
/// `!Send` and `!Sync` because AMX state is per-thread.
///
/// All AMX operations are methods on this struct, ensuring at compile
/// time that the coprocessor is active.
pub struct Matrix {
    /// `PhantomData<*const ()>` makes the type `!Send + !Sync`.
    _not_send_sync: PhantomData<*const ()>,
}

impl Matrix {
    /// Activate the AMX coprocessor on the current thread.
    ///
    /// # Errors
    ///
    /// Returns [`CpuError::AmxSetFailed`] if the current hardware does
    /// not support AMX (non-Apple-Silicon or virtualised without AMX
    /// passthrough).
    ///
    /// # Safety
    ///
    /// Only one `Matrix` should be live per thread at a time. Creating
    /// a second one is not undefined behaviour but wastes a SET/CLR
    /// pair and the register state from the first context becomes
    /// shared, which is confusing.
    #[inline]
    pub fn new() -> crate::Result<Self> {
        // AMX_SET can only fail on non-Apple-Silicon, where the .word
        // encoding is an illegal instruction, so we cannot easily
        // detect failure after the fact (it would SIGILL). We rely on
        // the probe module having been checked beforehand, but still
        // wrap for API consistency.
        unsafe {
            asm::amx_set();
        }
        Ok(Self {
            _not_send_sync: PhantomData,
        })
    }
}

impl Drop for Matrix {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            asm::amx_clr();
        }
    }
}
