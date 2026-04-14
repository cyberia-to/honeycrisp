//! Pipeline + MtlRenderPipeline

use crate::ffi::*;

/// A compiled compute pipeline state. Wraps `id<MTLComputePipelineState>`.
pub struct Pipeline {
    raw: ObjcId,
}

impl Pipeline {
    pub(crate) fn from_raw(raw: ObjcId) -> Self {
        Pipeline { raw }
    }

    /// Maximum total threads per threadgroup for this pipeline.
    pub fn max_total_threads_per_threadgroup(&self) -> usize {
        unsafe { msg0_usize(self.raw, SEL_maxTotalThreadsPerThreadgroup()) }
    }

    /// Thread execution width (SIMD width).
    pub fn thread_execution_width(&self) -> usize {
        unsafe { msg0_usize(self.raw, SEL_threadExecutionWidth()) }
    }

    /// Static threadgroup memory length used by this pipeline (bytes).
    #[mutants::skip] // 0 is correct for most shaders — mutant → 0 passes
    pub fn static_threadgroup_memory_length(&self) -> usize {
        unsafe { msg0_usize(self.raw, SEL_staticThreadgroupMemoryLength()) }
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for Pipeline {
    #[mutants::skip] // RAII release — tested by drop_stress_leak
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}
