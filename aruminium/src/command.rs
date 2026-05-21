//! Queue + Commands

use crate::encoder::{Copier, Encoder};
use crate::ffi::*;
use crate::GpuError;

#[cold]
#[inline(never)]
fn err_cmd_buffer() -> GpuError {
    GpuError::CommandBufferError("creation failed".into())
}

#[cold]
#[inline(never)]
fn err_encoder() -> GpuError {
    GpuError::EncoderCreationFailed
}

/// A command queue for submitting work to the GPU. Wraps `id<MTLCommandQueue>`.
pub struct Queue {
    raw: ObjcId,
}

impl Queue {
    pub(crate) fn from_raw(raw: ObjcId) -> Self {
        Queue { raw }
    }

    /// Wrap an existing `id<MTLCommandQueue>` (e.g. from wgpu).
    ///
    /// Increments the queue's retain count; releases on drop. The caller
    /// keeps independent ownership — aruminium will not double-free.
    /// Use this together with [`crate::Gpu::from_raw`] to share one
    /// `MTLDevice` + `MTLCommandQueue` pair between aruminium and another
    /// Metal client (e.g. wgpu) so submissions are serialized on a single
    /// GPU stream and `MTLSharedEvent`s can sync across engines.
    ///
    /// # Safety
    /// - `queue` must be a valid `id<MTLCommandQueue>` pointer.
    /// - The queue must remain valid for at least the lifetime of the
    ///   returned `Queue` (caller's retain count must not drop to zero
    ///   before our `release` runs).
    pub unsafe fn from_raw_shared(queue: ObjcId) -> Self {
        unsafe { retain(queue) };
        Self::from_raw(queue)
    }

    /// Create a new command buffer (retained).
    /// Uses ARC fast-retain to skip autorelease+retain round-trip.
    #[inline(always)]
    pub fn commands(&self) -> Result<Commands, GpuError> {
        let raw = unsafe { msg0_retained(self.raw, SEL_commandBuffer()) };
        if raw.is_null() {
            return Err(err_cmd_buffer());
        }
        Ok(Commands { raw, owned: true })
    }

    /// Create a command buffer without retain/release overhead.
    ///
    /// # Safety
    /// The returned buffer is autoreleased. Caller must ensure an autorelease pool
    /// is active and that the buffer is not used after the pool is drained.
    /// Use `autorelease_pool()` to create a pool scope.
    #[inline]
    pub unsafe fn commands_unretained(&self) -> Result<Commands, GpuError> {
        let raw = msg0(self.raw, SEL_commandBuffer());
        if raw.is_null() {
            return Err(err_cmd_buffer());
        }
        Ok(Commands { raw, owned: false })
    }

    /// Create a command buffer that does NOT retain referenced resources.
    /// Faster than `commands()` — Metal skips retain/release on all
    /// buffers, textures, and pipeline states used in the command buffer.
    ///
    /// # Safety
    /// Caller must ensure all resources referenced by encoded commands
    /// remain alive until the command buffer completes.
    #[inline(always)]
    pub unsafe fn commands_fast(&self) -> Result<Commands, GpuError> {
        let raw = msg0(self.raw, SEL_commandBufferWithUnretainedReferences());
        if raw.is_null() {
            return Err(err_cmd_buffer());
        }
        objc_retain(raw);
        Ok(Commands { raw, owned: true })
    }

    /// Zero-overhead command buffer: unretained references, no null check, no Result.
    /// Uses `commandBufferWithUnretainedReferences` — Metal skips internal
    /// retain/release on all encoded resources.
    ///
    /// # Safety
    /// All resources referenced by encoded commands must remain alive until completion.
    #[inline(always)]
    pub unsafe fn commands_unchecked(&self) -> Commands {
        let raw = msg0_retained(self.raw, SEL_commandBufferWithUnretainedReferences());
        Commands { raw, owned: true }
    }

    /// Fastest possible command buffer: no retain, no release, no null check.
    /// Object is autoreleased — caller MUST be inside an `autorelease_pool`.
    /// Uses `commandBufferWithUnretainedReferences` for Metal-side savings too.
    ///
    /// # Safety
    /// Must be inside `autorelease_pool`. Object invalid after pool drains.
    /// All encoded resources must outlive the command buffer.
    #[inline(always)]
    pub unsafe fn commands_autoreleased(&self) -> Commands {
        let raw = msg0(self.raw, SEL_commandBufferWithUnretainedReferences());
        Commands { raw, owned: false }
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for Queue {
    #[mutants::skip] // RAII release — tested by drop_stress_leak
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}

/// A command buffer for encoding GPU commands. Wraps `id<MTLCommandBuffer>`.
pub struct Commands {
    raw: ObjcId,
    owned: bool,
}

impl Commands {
    /// Create a compute command encoder (retained).
    /// Uses ARC fast-retain to skip autorelease+retain round-trip.
    #[inline(always)]
    pub fn encoder(&self) -> Result<Encoder, GpuError> {
        let raw = unsafe { msg0_retained(self.raw, SEL_computeCommandEncoder()) };
        if raw.is_null() {
            return Err(err_encoder());
        }
        Ok(Encoder::from_raw(raw, true))
    }

    /// Create a compute encoder without retain/release overhead.
    ///
    /// # Safety
    /// Same as `commands_unretained`.
    #[inline]
    pub unsafe fn encoder_unretained(&self) -> Result<Encoder, GpuError> {
        let raw = msg0(self.raw, SEL_computeCommandEncoder());
        if raw.is_null() {
            return Err(err_encoder());
        }
        Ok(Encoder::from_raw(raw, false))
    }

    /// Zero-overhead compute encoder: no null check, no Result.
    ///
    /// # Safety
    /// Command buffer must be in a valid state for encoding.
    #[inline(always)]
    pub unsafe fn encoder_unchecked(&self) -> Encoder {
        let raw = msg0_retained(self.raw, SEL_computeCommandEncoder());
        Encoder::from_raw(raw, true)
    }

    /// Fastest encoder: no retain, no release, no null check.
    /// Autoreleased — valid only within current autorelease pool.
    ///
    /// # Safety
    /// Must be inside `autorelease_pool`. Command buffer must be valid.
    #[inline(always)]
    pub unsafe fn encoder_autoreleased(&self) -> Encoder {
        let raw = msg0(self.raw, SEL_computeCommandEncoder());
        Encoder::from_raw(raw, false)
    }

    /// Create a blit command encoder.
    #[inline]
    pub fn copier(&self) -> Result<Copier, GpuError> {
        let raw = unsafe { msg0(self.raw, SEL_blitCommandEncoder()) };
        if raw.is_null() {
            return Err(err_encoder());
        }
        unsafe { retain(raw) };
        Ok(Copier::from_raw(raw))
    }

    /// Submit the command buffer for execution.
    #[inline(always)]
    pub fn submit(&self) {
        unsafe { msg0_void(self.raw, SEL_commit()) };
    }

    /// Block until the command buffer completes.
    #[inline(always)]
    pub fn wait(&self) {
        unsafe { msg0_void(self.raw, SEL_waitUntilCompleted()) };
    }

    /// Command buffer status (see `STATUS_*` constants).
    pub fn status(&self) -> u64 {
        unsafe { msg0_u64(self.raw, SEL_status()) }
    }

    pub const STATUS_NOT_ENQUEUED: u64 = 0;
    pub const STATUS_ENQUEUED: u64 = 1;
    pub const STATUS_COMMITTED: u64 = 2;
    pub const STATUS_SCHEDULED: u64 = 3;
    pub const STATUS_COMPLETED: u64 = 4;
    pub const STATUS_ERROR: u64 = 5;

    /// Get error description if the command buffer failed.
    #[mutants::skip] // None on success — mutant → None passes correctly
    pub fn error(&self) -> Option<String> {
        let err = unsafe { msg0(self.raw, SEL_error()) };
        nserror_string(err)
    }

    /// GPU start time in seconds (absolute, from device boot).
    /// Only valid after command buffer completes.
    pub fn gpu_start_time(&self) -> f64 {
        unsafe {
            let sel = SEL_GPUStartTime();
            let f: extern "C" fn(ObjcId, ObjcSel) -> f64 =
                std::mem::transmute(objc_msgSend as *const ());
            f(self.raw, sel)
        }
    }

    /// GPU end time in seconds (absolute, from device boot).
    /// Only valid after command buffer completes.
    pub fn gpu_end_time(&self) -> f64 {
        unsafe {
            let sel = SEL_GPUEndTime();
            let f: extern "C" fn(ObjcId, ObjcSel) -> f64 =
                std::mem::transmute(objc_msgSend as *const ());
            f(self.raw, sel)
        }
    }

    /// GPU execution time in seconds. Call after wait_until_completed().
    pub fn gpu_time(&self) -> f64 {
        self.gpu_end_time() - self.gpu_start_time()
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for Commands {
    #[mutants::skip] // RAII release — tested by drop_stress_leak
    fn drop(&mut self) {
        if self.owned {
            unsafe { release_nonnull(self.raw) };
        }
    }
}
