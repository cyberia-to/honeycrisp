//! Cached ObjC selectors
//!
//! sel_registerName is idempotent: same C string -> same pointer.
//! We cache the result via AtomicPtr with Relaxed ordering — on ARM64 this
//! compiles to a plain `ldr` (no memory barrier). The race on first init is
//! benign: sel_registerName always returns the same value for the same string.

use super::{sel_registerName, ObjcSel};

macro_rules! cached_sel {
    ($name:ident, $lit:expr) => {
        #[inline(always)]
        pub fn $name() -> ObjcSel {
            use std::sync::atomic::{AtomicPtr, Ordering};
            static CACHE: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());
            let p = CACHE.load(Ordering::Relaxed);
            if !p.is_null() {
                return p as ObjcSel;
            }
            let s = unsafe { sel_registerName($lit.as_ptr()) };
            CACHE.store(s as *mut (), Ordering::Relaxed);
            s
        }
    };
}

// Hot path selectors — called on every command buffer dispatch
cached_sel!(SEL_commandBuffer, c"commandBuffer");
cached_sel!(SEL_memoryBarrierWithScope, c"memoryBarrierWithScope:");
cached_sel!(SEL_computeCommandEncoder, c"computeCommandEncoder");
cached_sel!(SEL_setComputePipelineState, c"setComputePipelineState:");
cached_sel!(SEL_setBuffer_offset_atIndex, c"setBuffer:offset:atIndex:");
cached_sel!(SEL_setBytes_length_atIndex, c"setBytes:length:atIndex:");
cached_sel!(
    SEL_dispatchThreads,
    c"dispatchThreads:threadsPerThreadgroup:"
);
cached_sel!(
    SEL_dispatchThreadgroups,
    c"dispatchThreadgroups:threadsPerThreadgroup:"
);
cached_sel!(SEL_endEncoding, c"endEncoding");
cached_sel!(SEL_commit, c"commit");
cached_sel!(SEL_waitUntilCompleted, c"waitUntilCompleted");
cached_sel!(SEL_contents, c"contents");
cached_sel!(SEL_status, c"status");
cached_sel!(SEL_error, c"error");

// Fast command buffer (unretained references)
cached_sel!(
    SEL_commandBufferWithUnretainedReferences,
    c"commandBufferWithUnretainedReferences"
);

// Device selectors
cached_sel!(SEL_name, c"name");
cached_sel!(SEL_newCommandQueue, c"newCommandQueue");
cached_sel!(
    SEL_newBufferWithLength_options,
    c"newBufferWithLength:options:"
);
cached_sel!(
    SEL_newBufferWithBytes_length_options,
    c"newBufferWithBytes:length:options:"
);
cached_sel!(
    SEL_newBufferWithBytesNoCopy_length_options_deallocator,
    c"newBufferWithBytesNoCopy:length:options:deallocator:"
);
cached_sel!(
    SEL_newLibraryWithSource_options_error,
    c"newLibraryWithSource:options:error:"
);
cached_sel!(
    SEL_newComputePipelineStateWithFunction_error,
    c"newComputePipelineStateWithFunction:error:"
);
cached_sel!(SEL_hasUnifiedMemory, c"hasUnifiedMemory");
cached_sel!(SEL_maxBufferLength, c"maxBufferLength");
cached_sel!(SEL_maxThreadsPerThreadgroup, c"maxThreadsPerThreadgroup");
cached_sel!(
    SEL_recommendedMaxWorkingSetSize,
    c"recommendedMaxWorkingSetSize"
);
cached_sel!(SEL_newTextureWithDescriptor, c"newTextureWithDescriptor:");
cached_sel!(SEL_newFence, c"newFence");
cached_sel!(SEL_newEvent, c"newEvent");
cached_sel!(SEL_newSharedEvent, c"newSharedEvent");

// Library/function selectors
cached_sel!(SEL_newFunctionWithName, c"newFunctionWithName:");
cached_sel!(SEL_functionNames, c"functionNames");

// Pipeline selectors
cached_sel!(
    SEL_maxTotalThreadsPerThreadgroup,
    c"maxTotalThreadsPerThreadgroup"
);
cached_sel!(SEL_threadExecutionWidth, c"threadExecutionWidth");
cached_sel!(
    SEL_staticThreadgroupMemoryLength,
    c"staticThreadgroupMemoryLength"
);

// Command buffer timing selectors
cached_sel!(SEL_GPUStartTime, c"GPUStartTime");
cached_sel!(SEL_GPUEndTime, c"GPUEndTime");

// Collection selectors
cached_sel!(SEL_count, c"count");
cached_sel!(SEL_objectAtIndex, c"objectAtIndex:");

// Blit encoder
cached_sel!(SEL_blitCommandEncoder, c"blitCommandEncoder");
cached_sel!(
    SEL_copyFromBuffer,
    c"copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:"
);

// Texture
cached_sel!(SEL_width, c"width");
cached_sel!(SEL_height, c"height");
cached_sel!(SEL_depth, c"depth");
cached_sel!(SEL_pixelFormat, c"pixelFormat");
cached_sel!(
    SEL_replaceRegion,
    c"replaceRegion:mipmapLevel:withBytes:bytesPerRow:"
);
cached_sel!(
    SEL_getBytes,
    c"getBytes:bytesPerRow:fromRegion:mipmapLevel:"
);

// Sync
cached_sel!(SEL_signaledValue, c"signaledValue");

// NSString / NSError
cached_sel!(SEL_localizedDescription, c"localizedDescription");
cached_sel!(SEL_UTF8String, c"UTF8String");

// ── Render pipeline ──

cached_sel!(
    SEL_newRenderPipelineStateWithDescriptor_error,
    c"newRenderPipelineStateWithDescriptor:error:"
);
cached_sel!(SEL_setVertexFunction, c"setVertexFunction:");
cached_sel!(SEL_setFragmentFunction, c"setFragmentFunction:");
cached_sel!(SEL_colorAttachments, c"colorAttachments");
cached_sel!(SEL_objectAtIndexedSubscript, c"objectAtIndexedSubscript:");
cached_sel!(SEL_setPixelFormat, c"setPixelFormat:");
cached_sel!(
    SEL_setDepthAttachmentPixelFormat,
    c"setDepthAttachmentPixelFormat:"
);
cached_sel!(
    SEL_setStencilAttachmentPixelFormat,
    c"setStencilAttachmentPixelFormat:"
);
cached_sel!(SEL_setSampleCount, c"setSampleCount:");
cached_sel!(SEL_setRasterSampleCount, c"setRasterSampleCount:");
cached_sel!(SEL_setVertexDescriptor, c"setVertexDescriptor:");

// ── Render pass descriptor + attachments ──

cached_sel!(SEL_renderPassDescriptor, c"renderPassDescriptor");
cached_sel!(SEL_setTexture, c"setTexture:");
cached_sel!(SEL_setLoadAction, c"setLoadAction:");
cached_sel!(SEL_setStoreAction, c"setStoreAction:");
cached_sel!(SEL_setClearColor, c"setClearColor:");
cached_sel!(SEL_setClearDepth, c"setClearDepth:");
cached_sel!(SEL_setLevel, c"setLevel:");
cached_sel!(SEL_setSlice, c"setSlice:");
cached_sel!(SEL_setResolveTexture, c"setResolveTexture:");
cached_sel!(SEL_depthAttachment, c"depthAttachment");

// ── Render command encoder ──

cached_sel!(
    SEL_renderCommandEncoderWithDescriptor,
    c"renderCommandEncoderWithDescriptor:"
);
cached_sel!(SEL_setRenderPipelineState, c"setRenderPipelineState:");
cached_sel!(
    SEL_setVertexBuffer_offset_atIndex,
    c"setVertexBuffer:offset:atIndex:"
);
cached_sel!(
    SEL_setFragmentBuffer_offset_atIndex,
    c"setFragmentBuffer:offset:atIndex:"
);
cached_sel!(
    SEL_setVertexBytes_length_atIndex,
    c"setVertexBytes:length:atIndex:"
);
cached_sel!(
    SEL_setFragmentBytes_length_atIndex,
    c"setFragmentBytes:length:atIndex:"
);
cached_sel!(SEL_setVertexTexture_atIndex, c"setVertexTexture:atIndex:");
cached_sel!(
    SEL_setFragmentTexture_atIndex,
    c"setFragmentTexture:atIndex:"
);
cached_sel!(SEL_setViewport, c"setViewport:");
cached_sel!(SEL_setScissorRect, c"setScissorRect:");
cached_sel!(
    SEL_drawPrimitives_vertexStart_vertexCount,
    c"drawPrimitives:vertexStart:vertexCount:"
);
cached_sel!(
    SEL_drawPrimitives_vertexStart_vertexCount_instanceCount,
    c"drawPrimitives:vertexStart:vertexCount:instanceCount:"
);
cached_sel!(
    SEL_drawIndexedPrimitives,
    c"drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:"
);
cached_sel!(
    SEL_drawIndexedPrimitives_instanced,
    c"drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:"
);
cached_sel!(SEL_setCullMode, c"setCullMode:");
cached_sel!(SEL_setFrontFacingWinding, c"setFrontFacingWinding:");
cached_sel!(SEL_setDepthStencilState, c"setDepthStencilState:");
cached_sel!(SEL_setDepthBias, c"setDepthBias:slopeScale:clamp:");

// ── Depth stencil state ──

cached_sel!(
    SEL_newDepthStencilStateWithDescriptor,
    c"newDepthStencilStateWithDescriptor:"
);
cached_sel!(SEL_setDepthCompareFunction, c"setDepthCompareFunction:");
cached_sel!(SEL_setDepthWriteEnabled, c"setDepthWriteEnabled:");

// ── Vertex descriptor ──

cached_sel!(SEL_attributes, c"attributes");
cached_sel!(SEL_layouts, c"layouts");
cached_sel!(SEL_setFormat, c"setFormat:");
cached_sel!(SEL_setOffset, c"setOffset:");
cached_sel!(SEL_setBufferIndex, c"setBufferIndex:");
cached_sel!(SEL_setStride, c"setStride:");
cached_sel!(SEL_setStepFunction, c"setStepFunction:");
cached_sel!(SEL_setStepRate, c"setStepRate:");

// ── Render pipeline color attachment blend ──

cached_sel!(SEL_setRgbBlendOperation, c"setRgbBlendOperation:");
cached_sel!(SEL_setAlphaBlendOperation, c"setAlphaBlendOperation:");
cached_sel!(SEL_setSourceRGBBlendFactor, c"setSourceRGBBlendFactor:");
cached_sel!(SEL_setSourceAlphaBlendFactor, c"setSourceAlphaBlendFactor:");
cached_sel!(
    SEL_setDestinationRGBBlendFactor,
    c"setDestinationRGBBlendFactor:"
);
cached_sel!(
    SEL_setDestinationAlphaBlendFactor,
    c"setDestinationAlphaBlendFactor:"
);

// ── Texture descriptor extras (render target) ──

cached_sel!(SEL_setTextureType, c"setTextureType:");
cached_sel!(SEL_setWidth, c"setWidth:");
cached_sel!(SEL_setHeight, c"setHeight:");
cached_sel!(SEL_setUsage, c"setUsage:");
cached_sel!(SEL_setStorageMode, c"setStorageMode:");

// ── Render pipeline color attachment write/blend toggle ──

cached_sel!(SEL_setBlendingEnabled, c"setBlendingEnabled:");
cached_sel!(SEL_setWriteMask, c"setWriteMask:");
