//! Gpu: Metal GPU device discovery and factory methods

use crate::buffer::Buffer;
use crate::command::Queue;
use crate::ffi::*;
use crate::pipeline::Pipeline;
use crate::shader::{Shader, ShaderLib};
use crate::texture::Texture;
use crate::GpuError;
use std::ffi::c_void;

/// A Metal GPU device. Wraps `id<MTLDevice>`.
pub struct Gpu {
    raw: ObjcId,
}

impl Gpu {
    /// Get the system default Metal device.
    ///
    /// Falls back to first device from MTLCopyAllDevices if
    /// MTLCreateSystemDefaultDevice returns nil (headless/no display session).
    #[mutants::skip] // fallback path makes delete-! equivalent
    pub fn open() -> Result<Self, GpuError> {
        let raw = unsafe { MTLCreateSystemDefaultDevice() };
        if !raw.is_null() {
            return Ok(Gpu { raw });
        }
        // Fallback: MTLCopyAllDevices works even without active display session
        let devices = Self::all()?;
        devices.into_iter().next().ok_or(GpuError::DeviceNotFound)
    }

    /// Get all available Metal devices.
    pub fn all() -> Result<Vec<Self>, GpuError> {
        let arr = unsafe { MTLCopyAllDevices() };
        if arr.is_null() {
            return Err(GpuError::DeviceNotFound);
        }
        let count = unsafe { msg0_usize(arr, SEL_count()) };
        let mut devices = Vec::with_capacity(count);
        for i in 0..count {
            let dev = unsafe { msg1::<NSUInteger>(arr, SEL_objectAtIndex(), i) };
            if !dev.is_null() {
                unsafe { retain(dev) };
                devices.push(Gpu { raw: dev });
            }
        }
        unsafe { release(arr) };
        Ok(devices)
    }

    /// Device name (e.g. "Apple M1 Pro").
    pub fn name(&self) -> String {
        unsafe {
            let ns = msg0(self.raw, SEL_name());
            nsstring_to_rust(ns).unwrap_or_else(|| "unknown".into())
        }
    }

    /// Whether the device has unified memory (shared CPU/GPU).
    #[mutants::skip] // always true on Apple Silicon — mutant → true passes correctly
    pub fn has_unified_memory(&self) -> bool {
        unsafe { msg0_bool(self.raw, SEL_hasUnifiedMemory()) }
    }

    /// Maximum buffer length in bytes.
    pub fn max_buffer_length(&self) -> usize {
        unsafe { msg0_usize(self.raw, SEL_maxBufferLength()) }
    }

    /// Maximum threads per threadgroup.
    pub fn max_threads_per_threadgroup(&self) -> MTLSize {
        unsafe { msg0_mtlsize(self.raw, SEL_maxThreadsPerThreadgroup()) }
    }

    /// Recommended max working set size in bytes.
    pub fn recommended_max_working_set_size(&self) -> u64 {
        unsafe { msg0_u64(self.raw, SEL_recommendedMaxWorkingSetSize()) }
    }

    /// Create a new command queue.
    pub fn new_command_queue(&self) -> Result<Queue, GpuError> {
        let raw = unsafe { msg0(self.raw, SEL_newCommandQueue()) };
        if raw.is_null() {
            return Err(GpuError::QueueCreationFailed);
        }
        Ok(Queue::from_raw(raw))
    }

    /// Create a shared-mode buffer of the given byte size.
    ///
    /// `size` must be > 0. Use `buffer_with_data` for initialized buffers.
    pub fn buffer(&self, size: usize) -> Result<Buffer, GpuError> {
        if size == 0 {
            return Err(GpuError::BufferCreationFailed("size must be > 0".into()));
        }
        let raw = unsafe {
            msg2::<NSUInteger, NSUInteger>(
                self.raw,
                SEL_newBufferWithLength_options(),
                size,
                MTLResourceStorageModeShared,
            )
        };
        if raw.is_null() {
            return Err(GpuError::BufferCreationFailed(format!("{} bytes", size)));
        }
        Ok(Buffer::from_raw(raw, size))
    }

    /// Create a GPU-private buffer. Not accessible from CPU — use blit encoder to copy.
    /// Private storage gives Metal full control over memory placement and caching,
    /// which can yield higher GPU bandwidth for inter-layer inference buffers.
    pub fn buffer_private(&self, size: usize) -> Result<Buffer, GpuError> {
        if size == 0 {
            return Err(GpuError::BufferCreationFailed("size must be > 0".into()));
        }
        let raw = unsafe {
            msg2::<NSUInteger, NSUInteger>(
                self.raw,
                SEL_newBufferWithLength_options(),
                size,
                MTLResourceStorageModePrivate,
            )
        };
        if raw.is_null() {
            return Err(GpuError::BufferCreationFailed(format!("{} bytes", size)));
        }
        Ok(Buffer::from_raw_private(raw, size))
    }

    /// Wrap an existing pointer as a Metal buffer — zero copy.
    ///
    /// The pointer must be page-aligned and the memory must stay alive
    /// for the lifetime of the returned buffer. Metal reads/writes
    /// directly to the provided memory — no allocation, no copy.
    ///
    /// # Safety
    /// - `ptr` must be page-aligned (4096 or 16384 on Apple Silicon)
    /// - `size` must be page-aligned
    /// - Memory at `ptr..ptr+size` must remain valid while the buffer exists
    /// - Caller must not free the memory while GPU commands are in flight
    pub unsafe fn buffer_wrap(
        &self,
        ptr: *mut std::ffi::c_void,
        size: usize,
    ) -> Result<Buffer, GpuError> {
        if ptr.is_null() || size == 0 {
            return Err(GpuError::BufferCreationFailed(
                "null ptr or zero size".into(),
            ));
        }
        // deallocator = nil — we manage the memory ourselves
        let raw = msg4::<*mut c_void, NSUInteger, NSUInteger, *const c_void>(
            self.raw,
            SEL_newBufferWithBytesNoCopy_length_options_deallocator(),
            ptr,
            size,
            MTLResourceStorageModeShared,
            std::ptr::null(), // nil deallocator block
        );
        if raw.is_null() {
            return Err(GpuError::BufferCreationFailed(format!(
                "newBufferWithBytesNoCopy failed: ptr={:?} size={}",
                ptr, size
            )));
        }
        Ok(Buffer::from_raw(raw, size))
    }

    /// Wrap a unimem::Block as a Metal buffer — zero copy.
    ///
    /// The MTLBuffer shares the Block's physical memory (IOSurface-backed).
    /// Block must outlive the returned Buffer.
    pub fn wrap(&self, block: &unimem::Block) -> Result<Buffer, GpuError> {
        unsafe { self.buffer_wrap(block.address() as *mut std::ffi::c_void, block.size()) }
    }

    /// Create a shared-mode buffer initialized with data.
    pub fn buffer_with_data(&self, data: &[u8]) -> Result<Buffer, GpuError> {
        let raw = unsafe {
            msg3::<*const c_void, NSUInteger, NSUInteger>(
                self.raw,
                SEL_newBufferWithBytes_length_options(),
                data.as_ptr() as *const c_void,
                data.len(),
                MTLResourceStorageModeShared,
            )
        };
        if raw.is_null() {
            return Err(GpuError::BufferCreationFailed(format!(
                "{} bytes",
                data.len()
            )));
        }
        Ok(Buffer::from_raw(raw, data.len()))
    }

    /// Compile Metal Shading Language source into a library.
    pub fn compile(&self, source: &str) -> Result<ShaderLib, GpuError> {
        let ns_source = nsstring(source);
        let mut error: ObjcId = std::ptr::null_mut();
        let raw = unsafe {
            type F = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, *mut ObjcId) -> ObjcId;
            let f: F = std::mem::transmute(objc_msgSend as *const c_void);
            let r = f(
                self.raw,
                SEL_newLibraryWithSource_options_error(),
                ns_source,
                std::ptr::null_mut(),
                &mut error,
            );
            CFRelease(ns_source as CFTypeRef);
            r
        };
        if raw.is_null() {
            let msg = nserror_string(error).unwrap_or_else(|| "unknown error".into());
            return Err(GpuError::LibraryCompilationFailed(msg));
        }
        Ok(ShaderLib::from_raw(raw))
    }

    /// Create a compute pipeline from a function.
    pub fn pipeline(&self, function: &Shader) -> Result<Pipeline, GpuError> {
        let mut error: ObjcId = std::ptr::null_mut();
        let raw = unsafe {
            type F = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, *mut ObjcId) -> ObjcId;
            let f: F = std::mem::transmute(objc_msgSend as *const c_void);
            f(
                self.raw,
                SEL_newComputePipelineStateWithFunction_error(),
                function.as_raw(),
                &mut error,
            )
        };
        if raw.is_null() {
            let msg = nserror_string(error).unwrap_or_else(|| "unknown error".into());
            return Err(GpuError::PipelineCreationFailed(msg));
        }
        Ok(Pipeline::from_raw(raw))
    }

    /// Create a texture with the given descriptor.
    ///
    /// # Safety
    /// `desc` must be a valid MTLTextureDescriptor object.
    pub unsafe fn texture(&self, desc: ObjcId) -> Result<Texture, GpuError> {
        let raw = msg1::<ObjcId>(self.raw, SEL_newTextureWithDescriptor(), desc);
        if raw.is_null() {
            return Err(GpuError::TextureCreationFailed(
                "descriptor rejected".into(),
            ));
        }
        Ok(Texture::from_raw(raw))
    }

    /// Create a new fence.
    pub fn fence(&self) -> Result<crate::sync::Fence, GpuError> {
        let raw = unsafe { msg0(self.raw, SEL_newFence()) };
        if raw.is_null() {
            return Err(GpuError::CommandBufferError("fence creation failed".into()));
        }
        Ok(crate::sync::Fence::from_raw(raw))
    }

    /// Create a new event.
    pub fn event(&self) -> Result<crate::sync::Event, GpuError> {
        let raw = unsafe { msg0(self.raw, SEL_newEvent()) };
        if raw.is_null() {
            return Err(GpuError::CommandBufferError("event creation failed".into()));
        }
        Ok(crate::sync::Event::from_raw(raw))
    }

    /// Create a new shared event.
    pub fn shared_event(&self) -> Result<crate::sync::SharedEvent, GpuError> {
        let raw = unsafe { msg0(self.raw, SEL_newSharedEvent()) };
        if raw.is_null() {
            return Err(GpuError::CommandBufferError(
                "shared event creation failed".into(),
            ));
        }
        Ok(crate::sync::SharedEvent::from_raw(raw))
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for Gpu {
    #[mutants::skip] // RAII release — tested by drop_stress_leak
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}
