//! Dispatch: pre-resolved IMP dispatch engine for hot loops
//!
//! Resolves all ObjC method implementations at construction time.
//! Hot path calls go through direct function pointers — no objc_msgSend,
//! no selector lookup, no method cache check.

use crate::buffer::Buffer;
use crate::command::Queue;
use crate::ffi::*;
use crate::pipeline::Pipeline;
use std::ffi::c_void;

// IMP type aliases for the hot path
type ImpId = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
type ImpVoid = unsafe extern "C" fn(ObjcId, ObjcSel);
type ImpObj = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId);
type ImpBuf = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, NSUInteger, NSUInteger);
type ImpBytes = unsafe extern "C" fn(ObjcId, ObjcSel, *const c_void, NSUInteger, NSUInteger);
type ImpDisp = unsafe extern "C" fn(ObjcId, ObjcSel, MTLSize, MTLSize);

/// # Safety
/// T must be a function pointer type with the same size as `*const c_void`.
unsafe fn resolve_imp<T>(cls: ObjcClass, sel: ObjcSel) -> T {
    assert!(
        std::mem::size_of::<T>() == std::mem::size_of::<*const c_void>(),
        "resolve_imp: T must be pointer-sized"
    );
    std::mem::transmute_copy(&class_getMethodImplementation(cls, sel))
}

/// Pre-resolved compute dispatch engine.
///
/// Resolves all ObjC method implementations once at construction.
/// Every dispatch call goes through a direct function pointer — bypasses
/// objc_msgSend entirely. For inference loops doing thousands of dispatches,
/// this eliminates the selector lookup overhead.
///
/// Uses `commandBufferWithUnretainedReferences` — Metal skips internal
/// retain/release on all resources, saving ~2μs per command buffer.
pub struct Dispatch {
    queue: ObjcId,
    // Selectors (passed as IMP argument on every call)
    sel_cmd_buf: ObjcSel,
    sel_encoder: ObjcSel,
    sel_set_pipe: ObjcSel,
    sel_set_buf: ObjcSel,
    sel_set_bytes: ObjcSel,
    sel_dispatch: ObjcSel,
    sel_dispatch_groups: ObjcSel,
    sel_end: ObjcSel,
    sel_commit: ObjcSel,
    sel_wait: ObjcSel,
    // Pre-resolved IMPs — all resolved eagerly in new()
    imp_cmd_buf: ImpId,
    imp_encoder: ImpId,
    imp_commit: ImpVoid,
    imp_wait: ImpVoid,
    imp_set_pipe: ImpObj,
    imp_set_buf: ImpBuf,
    imp_set_bytes: ImpBytes,
    imp_dispatch: ImpDisp,
    imp_dispatch_groups: ImpDisp,
    imp_end: ImpVoid,
}

impl Dispatch {
    /// Create a new dispatcher from a command queue.
    /// Resolves all ObjC method implementations eagerly.
    /// Retains the queue — safe to drop the original `Queue`.
    pub fn new(queue: &Queue) -> Self {
        let q = queue.as_raw();
        unsafe { objc_retain(q) };

        // Selectors
        // Use regular commandBuffer — ARC fast-retain skips autorelease round-trip.
        // commandBufferWithUnretainedReferences has different Metal-internal path
        // that can be slower with ARC retain pattern.
        let sel_cmd_buf = SEL_commandBuffer();
        let sel_encoder = SEL_computeCommandEncoder();
        let sel_set_pipe = SEL_setComputePipelineState();
        let sel_set_buf = SEL_setBuffer_offset_atIndex();
        let sel_set_bytes = SEL_setBytes_length_atIndex();
        let sel_dispatch = SEL_dispatchThreads();
        let sel_dispatch_groups = SEL_dispatchThreadgroups();
        let sel_end = SEL_endEncoding();
        let sel_commit = SEL_commit();
        let sel_wait = SEL_waitUntilCompleted();

        unsafe {
            let q_cls = object_getClass(q);
            let imp_cmd_buf: ImpId = resolve_imp(q_cls, sel_cmd_buf);

            // Create a temporary command buffer to get its class
            let cmd = imp_cmd_buf(q, sel_cmd_buf);
            objc_retain(cmd);
            let cmd_cls = object_getClass(cmd);

            let imp_encoder: ImpId = resolve_imp(cmd_cls, sel_encoder);
            let imp_commit: ImpVoid = resolve_imp(cmd_cls, sel_commit);
            let imp_wait: ImpVoid = resolve_imp(cmd_cls, sel_wait);

            // Create a temporary encoder to get its class
            let enc = imp_encoder(cmd, sel_encoder);
            objc_retain(enc);
            let enc_cls = object_getClass(enc);

            let imp_set_pipe: ImpObj = resolve_imp(enc_cls, sel_set_pipe);
            let imp_set_buf: ImpBuf = resolve_imp(enc_cls, sel_set_buf);
            let imp_set_bytes: ImpBytes = resolve_imp(enc_cls, sel_set_bytes);
            let imp_dispatch: ImpDisp = resolve_imp(enc_cls, sel_dispatch);
            let imp_dispatch_groups: ImpDisp = resolve_imp(enc_cls, sel_dispatch_groups);
            let imp_end: ImpVoid = resolve_imp(enc_cls, sel_end);

            // Cleanup temp objects
            imp_end(enc, sel_end);
            imp_commit(cmd, sel_commit);
            imp_wait(cmd, sel_wait);
            objc_release(enc);
            objc_release(cmd);

            Dispatch {
                queue: q,
                sel_cmd_buf,
                sel_encoder,
                sel_set_pipe,
                sel_set_buf,
                sel_set_bytes,
                sel_dispatch,
                sel_dispatch_groups,
                sel_end,
                sel_commit,
                sel_wait,
                imp_cmd_buf,
                imp_encoder,
                imp_commit,
                imp_wait,
                imp_set_pipe,
                imp_set_buf,
                imp_set_bytes,
                imp_dispatch,
                imp_dispatch_groups,
                imp_end,
            }
        }
    }

    /// Dispatch a single compute operation. Creates a command buffer,
    /// encodes, commits, and waits for completion.
    ///
    /// Hybrid path: msgSend + ARC fast-retain for cmd/encoder creation,
    /// pre-resolved IMP for all encoding operations.
    ///
    /// # Safety
    /// All buffers and pipeline must remain alive until completion.
    #[inline(always)]
    pub unsafe fn dispatch(
        &self,
        pipeline: &Pipeline,
        buffers: &[(&Buffer, usize, usize)],
        grid: (usize, usize, usize),
        group: (usize, usize, usize),
    ) {
        // ARC fast-retain: runtime skips autorelease+retain round-trip
        let cmd = msg0_retained(self.queue, self.sel_cmd_buf);
        let enc = msg0_retained(cmd, self.sel_encoder);

        (self.imp_set_pipe)(enc, self.sel_set_pipe, pipeline.as_raw());
        for &(buf, offset, index) in buffers {
            (self.imp_set_buf)(enc, self.sel_set_buf, buf.as_raw(), offset, index);
        }
        let g = MTLSize {
            width: grid.0,
            height: grid.1,
            depth: grid.2,
        };
        let t = MTLSize {
            width: group.0,
            height: group.1,
            depth: group.2,
        };
        (self.imp_dispatch)(enc, self.sel_dispatch, g, t);
        (self.imp_end)(enc, self.sel_end);
        (self.imp_commit)(cmd, self.sel_commit);
        (self.imp_wait)(cmd, self.sel_wait);

        objc_release(enc);
        objc_release(cmd);
    }

    /// Dispatch with inline bytes (e.g. uniforms).
    ///
    /// # Safety
    /// All buffers and pipeline must remain alive until completion.
    #[inline(always)]
    pub unsafe fn dispatch_with_bytes(
        &self,
        pipeline: &Pipeline,
        buffers: &[(&Buffer, usize, usize)],
        bytes: &[u8],
        bytes_index: usize,
        grid: (usize, usize, usize),
        group: (usize, usize, usize),
    ) {
        let cmd = msg0_retained(self.queue, self.sel_cmd_buf);
        let enc = msg0_retained(cmd, self.sel_encoder);

        (self.imp_set_pipe)(enc, self.sel_set_pipe, pipeline.as_raw());
        for &(buf, offset, index) in buffers {
            (self.imp_set_buf)(enc, self.sel_set_buf, buf.as_raw(), offset, index);
        }
        (self.imp_set_bytes)(
            enc,
            self.sel_set_bytes,
            bytes.as_ptr() as *const c_void,
            bytes.len(),
            bytes_index,
        );
        let g = MTLSize {
            width: grid.0,
            height: grid.1,
            depth: grid.2,
        };
        let t = MTLSize {
            width: group.0,
            height: group.1,
            depth: group.2,
        };
        (self.imp_dispatch)(enc, self.sel_dispatch, g, t);
        (self.imp_end)(enc, self.sel_end);
        (self.imp_commit)(cmd, self.sel_commit);
        (self.imp_wait)(cmd, self.sel_wait);

        objc_release(enc);
        objc_release(cmd);
    }

    /// Encode multiple dispatches into a single command buffer.
    /// Much more efficient than individual dispatch() calls for inference.
    ///
    /// # Safety
    /// All resources must remain alive until completion.
    #[inline(always)]
    pub unsafe fn batch<F>(&self, encode: F)
    where
        F: FnOnce(&Batch),
    {
        let cmd = msg0_retained(self.queue, self.sel_cmd_buf);
        let enc = msg0_retained(cmd, self.sel_encoder);

        let batch = Batch {
            enc,
            imp_set_pipe: self.imp_set_pipe,
            imp_set_buf: self.imp_set_buf,
            imp_set_bytes: self.imp_set_bytes,
            imp_dispatch: self.imp_dispatch,
            imp_dispatch_groups: self.imp_dispatch_groups,
            sel_set_pipe: self.sel_set_pipe,
            sel_set_buf: self.sel_set_buf,
            sel_set_bytes: self.sel_set_bytes,
            sel_dispatch: self.sel_dispatch,
            sel_dispatch_groups: self.sel_dispatch_groups,
        };

        encode(&batch);

        (self.imp_end)(enc, self.sel_end);
        (self.imp_commit)(cmd, self.sel_commit);
        (self.imp_wait)(cmd, self.sel_wait);

        objc_release(enc);
        objc_release(cmd);
    }

    /// Like `batch` but without autorelease pool management.
    /// Caller MUST be inside an `autorelease_pool` scope.
    /// Use this for multi-batch inference loops where the caller manages
    /// a single pool for the entire pass.
    ///
    /// # Safety
    /// Must be inside `autorelease_pool`. All resources must remain alive.
    #[inline(always)]
    pub unsafe fn batch_raw<F>(&self, encode: F)
    where
        F: FnOnce(&Batch),
    {
        let cmd = (self.imp_cmd_buf)(self.queue, self.sel_cmd_buf);
        assert!(!cmd.is_null(), "command buffer creation returned null");
        let enc = (self.imp_encoder)(cmd, self.sel_encoder);
        assert!(!enc.is_null(), "compute encoder creation returned null");

        let batch = Batch {
            enc,
            imp_set_pipe: self.imp_set_pipe,
            imp_set_buf: self.imp_set_buf,
            imp_set_bytes: self.imp_set_bytes,
            imp_dispatch: self.imp_dispatch,
            imp_dispatch_groups: self.imp_dispatch_groups,
            sel_set_pipe: self.sel_set_pipe,
            sel_set_buf: self.sel_set_buf,
            sel_set_bytes: self.sel_set_bytes,
            sel_dispatch: self.sel_dispatch,
            sel_dispatch_groups: self.sel_dispatch_groups,
        };

        encode(&batch);

        (self.imp_end)(enc, self.sel_end);
        (self.imp_commit)(cmd, self.sel_commit);
        (self.imp_wait)(cmd, self.sel_wait);
    }

    /// Pipelined dispatch: encode + commit, return handle for deferred wait.
    /// Overlap GPU execution of batch N with CPU encoding of batch N+1.
    ///
    /// Usage:
    /// ```ignore
    /// let mut prev = None;
    /// for pass in passes {
    ///     let handle = disp.batch_async(|batch| { ... });
    ///     if let Some(h) = prev { h.wait(); }
    ///     prev = Some(handle);
    /// }
    /// if let Some(h) = prev { h.wait(); }
    /// ```
    ///
    /// # Safety
    /// All resources must remain alive until `GpuFuture::wait()` is called.
    #[inline(always)]
    pub unsafe fn batch_async<F>(&self, encode: F) -> GpuFuture
    where
        F: FnOnce(&Batch),
    {
        let cmd = msg0_retained(self.queue, self.sel_cmd_buf);
        let enc = msg0_retained(cmd, self.sel_encoder);

        let batch = Batch {
            enc,
            imp_set_pipe: self.imp_set_pipe,
            imp_set_buf: self.imp_set_buf,
            imp_set_bytes: self.imp_set_bytes,
            imp_dispatch: self.imp_dispatch,
            imp_dispatch_groups: self.imp_dispatch_groups,
            sel_set_pipe: self.sel_set_pipe,
            sel_set_buf: self.sel_set_buf,
            sel_set_bytes: self.sel_set_bytes,
            sel_dispatch: self.sel_dispatch,
            sel_dispatch_groups: self.sel_dispatch_groups,
        };

        encode(&batch);

        (self.imp_end)(enc, self.sel_end);
        (self.imp_commit)(cmd, self.sel_commit);
        // Do NOT wait — return handle for deferred wait

        objc_release(enc);
        GpuFuture {
            cmd,
            sel_wait: self.sel_wait,
        }
    }
}

impl Drop for Dispatch {
    #[mutants::skip] // RAII release — tested by drop_stress_leak
    fn drop(&mut self) {
        unsafe { objc_release(self.queue) };
    }
}

/// Handle for a committed but not yet completed GPU command buffer.
/// Call `wait()` to block until execution finishes.
pub struct GpuFuture {
    cmd: ObjcId,
    sel_wait: ObjcSel,
}

impl GpuFuture {
    /// Block until the GPU finishes executing this command buffer.
    #[inline(always)]
    #[mutants::skip] // Drop impl also waits — mutant → () still completes
    pub fn wait(self) {
        let cmd = self.cmd;
        let sel = self.sel_wait;
        std::mem::forget(self); // disarm Drop first — leak on panic, not double-free
        unsafe {
            msg0_void(cmd, sel);
            objc_release(cmd);
        }
    }
}

impl Drop for GpuFuture {
    #[mutants::skip] // RAII release — tested by drop_stress_leak
    fn drop(&mut self) {
        // If not waited, wait + release to prevent leak
        unsafe {
            msg0_void(self.cmd, self.sel_wait);
            objc_release(self.cmd);
        }
    }
}

/// A batch encoder for encoding multiple dispatches into one command buffer.
pub struct Batch {
    enc: ObjcId,
    imp_set_pipe: ImpObj,
    imp_set_buf: ImpBuf,
    imp_set_bytes: ImpBytes,
    imp_dispatch: ImpDisp,
    imp_dispatch_groups: ImpDisp,
    sel_set_pipe: ObjcSel,
    sel_set_buf: ObjcSel,
    sel_set_bytes: ObjcSel,
    sel_dispatch: ObjcSel,
    sel_dispatch_groups: ObjcSel,
}

impl Batch {
    #[inline(always)]
    pub fn bind(&self, pipeline: &Pipeline) {
        unsafe { (self.imp_set_pipe)(self.enc, self.sel_set_pipe, pipeline.as_raw()) };
    }

    #[inline(always)]
    pub fn bind_buffer(&self, buffer: &Buffer, offset: usize, index: usize) {
        unsafe { (self.imp_set_buf)(self.enc, self.sel_set_buf, buffer.as_raw(), offset, index) };
    }

    #[inline(always)]
    pub fn push(&self, data: &[u8], index: usize) {
        unsafe {
            (self.imp_set_bytes)(
                self.enc,
                self.sel_set_bytes,
                data.as_ptr() as *const c_void,
                data.len(),
                index,
            )
        };
    }

    #[inline(always)]
    pub fn launch(&self, grid: (usize, usize, usize), group: (usize, usize, usize)) {
        let g = MTLSize {
            width: grid.0,
            height: grid.1,
            depth: grid.2,
        };
        let t = MTLSize {
            width: group.0,
            height: group.1,
            depth: group.2,
        };
        unsafe { (self.imp_dispatch)(self.enc, self.sel_dispatch, g, t) };
    }

    #[inline(always)]
    pub fn launch_groups(&self, groups: (usize, usize, usize), threads: (usize, usize, usize)) {
        let g = MTLSize {
            width: groups.0,
            height: groups.1,
            depth: groups.2,
        };
        let t = MTLSize {
            width: threads.0,
            height: threads.1,
            depth: threads.2,
        };
        unsafe { (self.imp_dispatch_groups)(self.enc, self.sel_dispatch_groups, g, t) };
    }
}
