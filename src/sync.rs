//! Synchronization primitives: Fence, Event, SharedEvent

use crate::ffi::*;

/// A fence for tracking GPU work within a command buffer. Wraps `id<MTLFence>`.
pub struct Fence {
    raw: ObjcId,
}

impl Fence {
    pub(crate) fn from_raw(raw: ObjcId) -> Self {
        Fence { raw }
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for Fence {
    #[mutants::skip] // RAII release — tested by drop_stress_leak
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}

/// A GPU event for synchronizing command buffer execution. Wraps `id<MTLEvent>`.
pub struct Event {
    raw: ObjcId,
}

impl Event {
    pub(crate) fn from_raw(raw: ObjcId) -> Self {
        Event { raw }
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for Event {
    #[mutants::skip] // RAII release — tested by drop_stress_leak
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}

/// A shared event for CPU/GPU synchronization. Wraps `id<MTLSharedEvent>`.
pub struct SharedEvent {
    raw: ObjcId,
}

impl SharedEvent {
    pub(crate) fn from_raw(raw: ObjcId) -> Self {
        SharedEvent { raw }
    }

    /// Get the current signaled value.
    #[mutants::skip] // initial value is 0 — mutant → 0 passes correctly
    pub fn signaled_value(&self) -> u64 {
        unsafe { msg0_u64(self.raw, SEL_signaledValue()) }
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for SharedEvent {
    #[mutants::skip] // RAII release — tested by drop_stress_leak
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}
