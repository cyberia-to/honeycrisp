//! Encoder + Copier

use crate::buffer::Buffer;
use crate::ffi::*;
use crate::pipeline::Pipeline;
use std::ffi::c_void;

/// A compute command encoder. Wraps `id<MTLComputeCommandEncoder>`.
pub struct Encoder {
    raw: ObjcId,
    owned: bool,
}

impl Encoder {
    pub(crate) fn from_raw(raw: ObjcId, owned: bool) -> Self {
        Encoder { raw, owned }
    }

    /// Bind a compute pipeline state.
    #[inline(always)]
    pub fn bind(&self, pipeline: &Pipeline) {
        unsafe { msg1_void(self.raw, SEL_setComputePipelineState(), pipeline.as_raw()) };
    }

    /// Bind a buffer at an index.
    #[inline(always)]
    pub fn bind_buffer(&self, buffer: &Buffer, offset: usize, index: usize) {
        unsafe {
            msg3_void(
                self.raw,
                SEL_setBuffer_offset_atIndex(),
                buffer.as_raw(),
                offset,
                index,
            )
        };
    }

    /// Push inline bytes at an index.
    #[inline(always)]
    pub fn push(&self, data: &[u8], index: usize) {
        unsafe {
            msg_bytes_void(
                self.raw,
                SEL_setBytes_length_atIndex(),
                data.as_ptr() as *const c_void,
                data.len(),
                index,
            )
        };
    }

    /// Launch threads with automatic threadgroup sizing.
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
        unsafe { msg_dispatch_void(self.raw, SEL_dispatchThreads(), g, t) };
    }

    /// Dispatch threadgroups with explicit group count.
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
        unsafe { msg_dispatch_void(self.raw, SEL_dispatchThreadgroups(), g, t) };
    }

    /// Finish encoding. Must be called before submitting the command buffer.
    #[inline(always)]
    pub fn finish(&self) {
        unsafe { msg0_void(self.raw, SEL_endEncoding()) };
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for Encoder {
    #[mutants::skip] // RAII release — tested by drop_stress_leak
    fn drop(&mut self) {
        if self.owned {
            unsafe { release_nonnull(self.raw) };
        }
    }
}

/// A blit command encoder. Wraps `id<MTLBlitCommandEncoder>`.
pub struct Copier {
    raw: ObjcId,
}

impl Copier {
    pub(crate) fn from_raw(raw: ObjcId) -> Self {
        Copier { raw }
    }

    /// Copy from one buffer to another.
    pub fn copy(
        &self,
        src: &Buffer,
        src_offset: usize,
        dst: &Buffer,
        dst_offset: usize,
        size: usize,
    ) {
        unsafe {
            type F = unsafe extern "C" fn(
                ObjcId,
                ObjcSel,
                ObjcId,
                NSUInteger,
                ObjcId,
                NSUInteger,
                NSUInteger,
            );
            let f: F = std::mem::transmute(objc_msgSend as *const c_void);
            f(
                self.raw,
                SEL_copyFromBuffer(),
                src.as_raw(),
                src_offset,
                dst.as_raw(),
                dst_offset,
                size,
            );
        }
    }

    /// End encoding.
    #[inline(always)]
    pub fn finish(&self) {
        unsafe { msg0_void(self.raw, SEL_endEncoding()) };
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for Copier {
    #[mutants::skip] // RAII release — tested by drop_stress_leak
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}
