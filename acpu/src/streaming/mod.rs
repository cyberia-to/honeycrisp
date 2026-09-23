//! Streaming SVE / SME context lifecycle.
//!
//! Constructing a [`Stream`] enters streaming mode (SMSTART) and enables
//! the ZA matrix accumulator. Dropping the [`Stream`] leaves streaming
//! mode (SMSTOP). The type is `!Send + !Sync` because PSTATE.SM and
//! PSTATE.ZA are per-thread.
//!
//! While a `Stream` is live, the SVE register file Z0–Z31 and predicate
//! registers P0–P15 become addressable, with the streaming vector length
//! (SVL) reported by [`Stream::svl_bytes`].
//!
//! `Stream` and [`crate::matrix::Matrix`] (AMX) share the same physical
//! coprocessor block. Holding both live on the same thread is not
//! undefined behaviour but interleaves writes to a single hardware state
//! and gives nonsense results. Callers should hold exactly one of the
//! two per thread at any time.

pub mod asm;
pub mod kern;
pub mod ssve;

use crate::{CpuError, Feature};
use core::marker::PhantomData;

/// A live streaming-mode context.
///
/// Construction calls `SMSTART` (both SM and ZA bits) and zeroes ZA.
/// Drop calls `SMSTOP` (both bits).
pub struct Stream {
    _not_send_sync: PhantomData<*const ()>,
}

impl Stream {
    /// Enter streaming mode on the current thread and enable ZA.
    ///
    /// Returns [`CpuError::FeatureNotAvailable`] if FEAT_SME is absent.
    pub fn new() -> crate::Result<Self> {
        if !crate::probe::scan().has_sme {
            return Err(CpuError::FeatureNotAvailable(Feature::Sme));
        }
        unsafe {
            asm::smstart();
            asm::zero_za();
        }
        Ok(Self {
            _not_send_sync: PhantomData,
        })
    }

    /// Streaming vector length in bytes (SVL / 8).
    ///
    /// On every shipping Apple SME implementation this is 64
    /// (SVL = 512 bits). The value is read from hardware via RDSVL
    /// each call; the cost is one instruction.
    #[inline]
    pub fn svl_bytes(&self) -> usize {
        unsafe { asm::rdsvl_bytes() as usize }
    }

    /// Streaming vector length in 32-bit lanes.
    #[inline]
    pub fn svl_lanes_f32(&self) -> usize {
        self.svl_bytes() / 4
    }

    /// Streaming vector length in 64-bit lanes.
    #[inline]
    pub fn svl_lanes_f64(&self) -> usize {
        self.svl_bytes() / 8
    }

    /// Re-zero the entire ZA accumulator without leaving streaming mode.
    #[inline]
    pub fn zero_za(&self) {
        unsafe { asm::zero_za() }
    }
}

impl Drop for Stream {
    #[inline]
    fn drop(&mut self) {
        unsafe { asm::smstop() }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_lifecycle() {
        if !crate::probe::scan().has_sme {
            eprintln!("skip: FEAT_SME not present");
            return;
        }
        let s = Stream::new().expect("Stream::new failed on SME host");
        let svl = s.svl_bytes();
        assert_eq!(
            svl, 64,
            "expected SVL=64 bytes on Apple SME implementations, got {svl}"
        );
        drop(s);
    }

    #[test]
    fn stream_double_open_serial() {
        // Open, close, open again — must work.
        if !crate::probe::scan().has_sme {
            return;
        }
        for _ in 0..3 {
            let s = Stream::new().unwrap();
            assert_eq!(s.svl_bytes(), 64);
        }
    }

    #[test]
    fn stream_errors_when_absent() {
        // If we are NOT on an SME host, constructing must fail cleanly.
        if crate::probe::scan().has_sme {
            return;
        }
        let r = Stream::new();
        assert!(matches!(
            r,
            Err(crate::CpuError::FeatureNotAvailable(Feature::Sme))
        ));
    }
}
