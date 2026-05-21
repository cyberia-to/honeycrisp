# aruminium — API specification

pure Rust driver for Apple Metal GPU. direct objc_msgSend FFI,
zero external dependencies, only macOS system frameworks.

## concepts

| concept | what it is |
|---------|-----------|
| device | a Metal GPU — discovered at runtime, owns all GPU resources |
| buffer | CPU/GPU memory region — shared (zero-copy) or private (GPU-only) |
| library | compiled shader code — one or more functions from MSL source |
| function | a single shader entry point — vertex, fragment, or kernel |
| pipeline | a compiled GPU state object — binds function + config for dispatch |
| queue | a serial command submission channel to the GPU |
| command buffer | a batch of encoded GPU commands — submitted atomically |
| encoder | records commands into a command buffer — compute or blit |
| dispatcher | pre-resolved IMP dispatch engine for inference hot loops |
| texture | GPU image data — 2D/3D, region read/write |
| fence | GPU work tracking within a single command buffer |
| event | synchronization between command buffers |
| shared event | CPU/GPU synchronization with signaled counter |

## lifecycle

```
source  ->  compile  ->  pipeline  ->  encode  ->  commit  ->  complete
  MSL      MTLLibrary   pipeline    encoder    cmdBuf      GPU done
```

## device

| method | signature | semantics |
|--------|-----------|-----------|
| open | `() -> Result<Gpu>` | get default Metal GPU |
| all | `() -> Result<Vec<Gpu>>` | enumerate all Metal GPUs |
| name | `(&self) -> String` | device name (e.g. "Apple M1 Pro") |
| has_unified_memory | `(&self) -> bool` | shared CPU/GPU memory architecture |
| max_buffer_length | `(&self) -> usize` | max buffer allocation in bytes |
| max_threads_per_threadgroup | `(&self) -> MTLSize` | max threads per threadgroup |
| recommended_max_working_set_size | `(&self) -> u64` | recommended GPU memory budget |
| new_command_queue | `(&self) -> Result<Queue>` | create command queue |
| buffer | `(&self, bytes) -> Result<Buffer>` | allocate shared buffer (CPU+GPU) |
| buffer_private | `(&self, bytes) -> Result<Buffer>` | allocate private buffer (GPU-only) |
| buffer_with_data | `(&self, &[u8]) -> Result<Buffer>` | shared buffer with initial data |
| compile | `(&self, &str) -> Result<ShaderLib>` | compile MSL source |
| pipeline | `(&self, &Shader) -> Result<Pipeline>` | create compute pipeline |
| texture | `(&self, desc) -> Result<Texture>` | create texture from descriptor (unsafe) |
| fence | `(&self) -> Fence` | create fence |
| event | `(&self) -> Event` | create event |
| shared_event | `(&self) -> SharedEvent` | create shared event |

### apple mapping

| method | ObjC |
|--------|------|
| open | MTLCreateSystemDefaultDevice() |
| all | MTLCopyAllDevices() |
| name | [device name] |
| has_unified_memory | [device hasUnifiedMemory] |
| max_buffer_length | [device maxBufferLength] |
| max_threads_per_threadgroup | [device maxThreadsPerThreadgroup] |
| recommended_max_working_set_size | [device recommendedMaxWorkingSetSize] |
| new_command_queue | [device newCommandQueue] |
| buffer | [device newBufferWithLength:options:] (StorageModeShared) |
| buffer_private | [device newBufferWithLength:options:] (StorageModePrivate) |
| buffer_with_data | [device newBufferWithBytes:length:options:] |
| compile | [device newLibraryWithSource:options:error:] |
| pipeline | [device newComputePipelineStateWithFunction:error:] |
| texture | [device newTextureWithDescriptor:] |
| fence | [device newFence] |
| event | [device newEvent] |
| shared_event | [device newSharedEvent] |

## buffer

CPU/GPU memory region. two storage modes:

- **shared** (default) — zero-copy, CPU and GPU share physical memory.
  no lock/unlock needed. contents pointer cached at creation.
- **private** — GPU-only, higher bandwidth for inter-kernel buffers.
  CPU cannot read/write. use blit encoder to copy data in/out.

| method | signature | semantics |
|--------|-----------|-----------|
| is_shared | `(&self) -> bool` | true if CPU-accessible (shared mode) |
| contents | `(&self) -> *mut c_void` | raw pointer to shared memory (cached) |
| read | `(&self, \|&[u8]\|)` | read access via closure |
| write | `(&self, \|&mut [u8]\|)` | write access via closure |
| read_f32 | `(&self, \|&[f32]\|)` | typed read as f32 |
| write_f32 | `(&self, \|&mut [f32]\|)` | typed write as f32 |
| size | `(&self) -> usize` | allocation in bytes |
| drop | automatic | [buffer release] |

### apple mapping

| method | ObjC |
|--------|------|
| contents | [buffer contents] |
| size | construction parameter |
| drop | objc_release |

## library

compiled shader code from MSL source text.

| method | signature | semantics |
|--------|-----------|-----------|
| function | `(&self, &str) -> Result<Shader>` | get function by name |
| function_names | `(&self) -> Vec<String>` | list all function names |

### apple mapping

| method | ObjC |
|--------|------|
| function | [library newFunctionWithName:] |
| function_names | [library functionNames] |

## function

a single shader entry point extracted from a library.

| method | signature | semantics |
|--------|-----------|-----------|
| name | `(&self) -> String` | function name |

## compute pipeline

compiled GPU state — function + hardware config.

| method | signature | semantics |
|--------|-----------|-----------|
| max_total_threads_per_threadgroup | `(&self) -> usize` | max threads per threadgroup for this pipeline |
| thread_execution_width | `(&self) -> usize` | SIMD width (32 on Apple GPU) |
| static_threadgroup_memory_length | `(&self) -> usize` | threadgroup memory used by pipeline (bytes) |

### apple mapping

| method | ObjC |
|--------|------|
| max_total_threads_per_threadgroup | [pipeline maxTotalThreadsPerThreadgroup] |
| thread_execution_width | [pipeline threadExecutionWidth] |
| static_threadgroup_memory_length | [pipeline staticThreadgroupMemoryLength] |

## command queue

| method | signature | semantics |
|--------|-----------|-----------|
| commands | `(&self) -> Result<Commands>` | retained, ARC fast-retain |
| commands_unretained | `unsafe (&self) -> Result<Commands>` | autoreleased, no retain overhead |
| commands_fast | `unsafe (&self) -> Result<Commands>` | unretained references — Metal skips resource retain/release |
| commands_unchecked | `unsafe (&self) -> Commands` | unretained refs, no null check |
| commands_autoreleased | `unsafe (&self) -> Commands` | fastest — must be in autorelease_pool |

overhead hierarchy (low to high):

```
commands_autoreleased   — zero overhead, requires pool
commands_unchecked      — no null check, unretained refs
commands_fast           — unretained refs, null checked
commands_unretained     — autoreleased, null checked
commands                — retained, safe, standard
```

### apple mapping

| method | ObjC |
|--------|------|
| commands | [queue commandBuffer] + objc_retainAutoreleasedReturnValue |
| commands_unretained | [queue commandBuffer] (no retain) |
| commands_fast | [queue commandBufferWithUnretainedReferences] + retain |
| commands_unchecked | [queue commandBufferWithUnretainedReferences] + ARC fast-retain |
| commands_autoreleased | [queue commandBufferWithUnretainedReferences] (no retain) |

## command buffer

| method | signature | semantics |
|--------|-----------|-----------|
| encoder | `(&self) -> Result<Encoder>` | retained compute encoder |
| encoder_unretained | `unsafe (&self) -> Result<Encoder>` | autoreleased |
| encoder_unchecked | `unsafe (&self) -> Encoder` | no null check, retained |
| encoder_autoreleased | `unsafe (&self) -> Encoder` | fastest, requires pool |
| copier | `(&self) -> Result<Copier>` | blit encoder |
| submit | `(&self)` | submit for GPU execution |
| wait | `(&self)` | block until GPU done |
| status | `(&self) -> u64` | execution status code |
| error | `(&self) -> Option<String>` | error description if failed |
| gpu_start_time | `(&self) -> f64` | GPU start time (seconds since boot) |
| gpu_end_time | `(&self) -> f64` | GPU end time (seconds since boot) |
| gpu_time | `(&self) -> f64` | GPU execution duration (end - start) |

### apple mapping

| method | ObjC |
|--------|------|
| encoder | [cmdBuf computeCommandEncoder] + ARC fast-retain |
| copier | [cmdBuf blitCommandEncoder] |
| submit | [cmdBuf commit] |
| wait | [cmdBuf waitUntilCompleted] |
| status | [cmdBuf status] |
| error | [cmdBuf error] |
| gpu_start_time | [cmdBuf GPUStartTime] |
| gpu_end_time | [cmdBuf GPUEndTime] |

## compute encoder

| method | signature | semantics |
|--------|-----------|-----------|
| bind | `(&self, &Pipeline)` | bind compute pipeline |
| bind_buffer | `(&self, &Buffer, offset, index)` | bind buffer at index |
| push | `(&self, &[u8], index)` | inline constant data |
| launch | `(&self, grid, group)` | dispatch with auto non-uniform grid handling |
| launch_groups | `(&self, groups, threads)` | dispatch with explicit group count |
| finish | `(&self)` | finish encoding |

### apple mapping

| method | ObjC |
|--------|------|
| bind | [encoder setComputePipelineState:] |
| bind_buffer | [encoder setBuffer:offset:atIndex:] |
| push | [encoder setBytes:length:atIndex:] |
| launch | [encoder dispatchThreads:threadsPerThreadgroup:] |
| launch_groups | [encoder dispatchThreadgroups:threadsPerThreadgroup:] |
| finish | [encoder endEncoding] |

## blit encoder

| method | signature | semantics |
|--------|-----------|-----------|
| copy | `(&self, src, src_off, dst, dst_off, size)` | GPU buffer-to-buffer copy |
| finish | `(&self)` | finish encoding |

### apple mapping

| method | ObjC |
|--------|------|
| copy | [encoder copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:] |
| finish | [encoder endEncoding] |

## compute dispatcher

pre-resolved IMP dispatch engine for inference hot loops.
resolves all ObjC method implementations at construction — every
dispatch call goes through direct function pointers, bypassing
objc_msgSend entirely.

| method | signature | semantics |
|--------|-----------|-----------|
| new | `(&Queue) -> Self` | resolve all IMPs eagerly |
| dispatch | `unsafe (&self, pipeline, buffers, grid, group)` | single dispatch: encode + commit + wait |
| dispatch_with_bytes | `unsafe (&self, pipeline, buffers, bytes, index, grid, group)` | single dispatch with inline constants |
| batch | `unsafe (&self, \|&Batch\|)` | multiple dispatches in one command buffer |
| batch_raw | `unsafe (&self, \|&Batch\|)` | batch without autorelease management (caller manages pool) |
| batch_async | `unsafe (&self, \|&Batch\|) -> GpuFuture` | encode + commit, return handle for deferred wait |

### batch

provided to batch closures. same IMP-resolved hot path.

| method | signature | semantics |
|--------|-----------|-----------|
| bind | `(&self, &Pipeline)` | bind pipeline |
| bind_buffer | `(&self, &Buffer, offset, index)` | bind buffer |
| push | `(&self, &[u8], index)` | inline constants |
| launch | `(&self, grid, group)` | dispatch |
| launch_groups | `(&self, groups, threads)` | dispatch with explicit groups |

### gpu future

handle for committed but not yet completed command buffer.

| method | signature | semantics |
|--------|-----------|-----------|
| wait | `(self)` | block until GPU finishes, release command buffer |
| drop | automatic | if not waited, waits + releases (prevents leak) |

### pipelining pattern

```rust
let mut prev = None;
for pass in passes {
    let handle = disp.batch_async(|batch| { ... });
    if let Some(h) = prev { h.wait(); }
    prev = Some(handle);
}
if let Some(h) = prev { h.wait(); }
```

overlap GPU execution of batch N with CPU encoding of batch N+1.

## texture

GPU image data. wraps `id<MTLTexture>`.

| method | signature | semantics |
|--------|-----------|-----------|
| width | `(&self) -> usize` | width in pixels |
| height | `(&self) -> usize` | height in pixels |
| depth | `(&self) -> usize` | depth (3D textures) |
| pixel_format | `(&self) -> usize` | MTLPixelFormat value |
| replace_region | `unsafe (&self, region, mipmap, data, bytes_per_row)` | write data to region |
| get_bytes | `unsafe (&self, data, bytes_per_row, region, mipmap)` | read data from region |

### apple mapping

| method | ObjC |
|--------|------|
| width | [texture width] |
| height | [texture height] |
| depth | [texture depth] |
| pixel_format | [texture pixelFormat] |
| replace_region | [texture replaceRegion:mipmapLevel:withBytes:bytesPerRow:] |
| get_bytes | [texture getBytes:bytesPerRow:fromRegion:mipmapLevel:] |

## synchronization

### fence

GPU work tracking within a single command buffer.

| method | signature | semantics |
|--------|-----------|-----------|
| as_raw | `(&self) -> ObjcId` | raw pointer for encoder fence ops |

### event

synchronization between command buffers on same device.

| method | signature | semantics |
|--------|-----------|-----------|
| as_raw | `(&self) -> ObjcId` | raw pointer for command buffer signal/wait |

### shared event

CPU/GPU synchronization with monotonic counter.

| method | signature | semantics |
|--------|-----------|-----------|
| signaled_value | `(&self) -> u64` | current signaled counter value |
| as_raw | `(&self) -> ObjcId` | raw pointer |

## conversion

fp16<->f32 via inline NEON assembly (aarch64) with software fallback.

| function | signature | semantics |
|----------|-----------|-----------|
| fp16_to_f32 | `(u16) -> f32` | single half -> single precision |
| f32_to_fp16 | `(f32) -> u16` | single -> half precision |
| cast_f16_f32 | `(&mut [f32], &[u16])` | bulk half -> single (32/iter, 4x unrolled NEON) |
| cast_f32_f16 | `(&mut [u16], &[f32])` | bulk single -> half (32/iter, 4x unrolled NEON) |

tail: 8/iter NEON, then scalar fallback.

## autorelease pool

```rust
autorelease_pool(|| {
    // autoreleased ObjC objects valid here
})
```

required when using unretained/autoreleased command buffer and encoder variants.

## errors

```
DeviceNotFound              no Metal GPU available
BufferCreationFailed(String) buffer allocation failed
LibraryCompilationFailed(String) MSL compilation error
FunctionNotFound(String)    shader function not in library
PipelineCreationFailed(String) pipeline creation error
CommandBufferError(String)  command buffer execution error
EncoderCreationFailed       encoder creation failed
QueueCreationFailed         command queue creation failed
TextureCreationFailed(String) texture creation failed
Io(io::Error)               filesystem error
```

## execution model

- one pipeline = one compiled shader function
- command buffers submitted atomically via commit
- GPU executes command buffers in order per queue
- multiple queues enable concurrent GPU work
- shared buffers need no synchronization between command buffer boundaries
- private buffers need blit encoder for CPU data transfer
- dispatch_threads handles non-uniform grids automatically
- dispatch_threadgroups requires manual grid division
- Dispatch bypasses objc_msgSend for hot-loop performance

## driver stack

```
aruminium crate (objc_msgSend FFI + IMP resolution)
  -> Metal.framework (linked at build time)
    -> GPU driver
      -> GPU hardware
```

Metal.framework is public. linked via `#[link(name = "Metal", kind = "framework")]`.
core path: objc_msgSend with transmuted function pointers.
hot path: pre-resolved IMP via class_getMethodImplementation.

## render pipeline (phase 1)

raster path: vertex + fragment shaders, render-target textures, render pass
descriptors, render encoders. mirrors the compute path 1:1.

```
shaders -> RenderPipeline -> RenderPassDescriptor -> RenderEncoder
                                                      -> draw
                                                      -> end
```

### render pipeline

`RenderPipeline` wraps `id<MTLRenderPipelineState>`.

| method | signature | semantics |
|--------|-----------|-----------|
| color_attachments | `(&self) -> usize` | number of color attachments |
| sample_count | `(&self) -> u32` | MSAA sample count (1 = none, set by phase 2) |
| as_raw | `(&self) -> ObjcId` | raw MTLRenderPipelineState |

`RenderPipelineSpec` — builder for `Gpu::render_pipeline`. phase 1 fields:
`color_attachments: Vec<ColorAttachmentSpec>`.

`ColorAttachmentSpec` — per-slot pixel format + write mask.

### render-target texture

| Gpu method | signature | semantics |
|-----------|-----------|-----------|
| render_target | `(&self, w, h, format) -> Result<Texture>` | color render target, usage `RenderTarget \| ShaderRead`, storage Private |
| render_pipeline | `(&self, &Shader, &Shader, &RenderPipelineSpec) -> Result<RenderPipeline>` | compile vertex+fragment into pipeline state |

### render pass descriptor

`RenderPassDescriptor` wraps `id<MTLRenderPassDescriptor>` — what to render INTO.

| method | signature | semantics |
|--------|-----------|-----------|
| new | `() -> Self` | empty descriptor (retained) |
| color_attachment | `(&mut self, index, ColorAttachmentDesc)` | configure color slot |
| as_raw | `(&self) -> ObjcId` | raw descriptor |

`ColorAttachmentDesc` — `{ texture, load_action, store_action, clear_color, level, slice }`.
`LoadAction` = DontCare | Load | Clear. `StoreAction` = DontCare | Store | MultisampleResolve | StoreAndMultisampleResolve.

### render encoder

| Commands method | signature | semantics |
|----------------|-----------|-----------|
| render_encoder | `(&self, &RenderPassDescriptor) -> Result<RenderEncoder>` | create encoder for pass |

| RenderEncoder method | signature | semantics |
|----------------------|-----------|-----------|
| bind | `(&self, &RenderPipeline)` | bind pipeline state |
| set_vertex_buffer | `(&self, index, &Buffer, offset)` | vertex stage buffer |
| set_fragment_buffer | `(&self, index, &Buffer, offset)` | fragment stage buffer |
| push_vertex | `(&self, &[u8], index)` | inline vertex bytes |
| push_fragment | `(&self, &[u8], index)` | inline fragment bytes |
| set_vertex_texture | `(&self, index, &Texture)` | vertex stage texture |
| set_fragment_texture | `(&self, index, &Texture)` | fragment stage texture |
| set_viewport | `(&self, x, y, w, h, near, far)` | viewport rect (f64) |
| set_scissor | `(&self, x, y, w, h)` | scissor rect (u32) |
| draw | `(&self, PrimitiveType, start, count)` | non-indexed draw |
| end | `(self)` | finish encoding (consumes) |

`PrimitiveType` = Point | Line | LineStrip | Triangle | TriangleStrip.

### apple mapping (phase 1)

| method | ObjC |
|--------|------|
| Gpu::render_target | [device newTextureWithDescriptor:] (usage RenderTarget) |
| Gpu::render_pipeline | [device newRenderPipelineStateWithDescriptor:error:] |
| Commands::render_encoder | [cmdBuf renderCommandEncoderWithDescriptor:] |
| RenderEncoder::bind | [encoder setRenderPipelineState:] |
| RenderEncoder::set_vertex_buffer | [encoder setVertexBuffer:offset:atIndex:] |
| RenderEncoder::set_fragment_buffer | [encoder setFragmentBuffer:offset:atIndex:] |
| RenderEncoder::set_vertex_texture | [encoder setVertexTexture:atIndex:] |
| RenderEncoder::set_fragment_texture | [encoder setFragmentTexture:atIndex:] |
| RenderEncoder::set_viewport | [encoder setViewport:] |
| RenderEncoder::set_scissor | [encoder setScissorRect:] |
| RenderEncoder::draw | [encoder drawPrimitives:vertexStart:vertexCount:] |
| RenderEncoder::end | [encoder endEncoding] |

## render pipeline (phase 2)

production-grade raster: depth/stencil, MSAA + resolve, vertex
descriptors, indexed/instanced draws, blend state, cull mode + winding,
depth bias.

### RenderPipelineSpec — phase 2 fields

| field | type | semantics |
|-------|------|-----------|
| depth_format | `Option<MTLPixelFormat>` | depth attachment format (None = none) |
| stencil_format | `Option<MTLPixelFormat>` | stencil attachment format |
| sample_count | `u32` | MSAA sample count (1 = none) |
| vertex_descriptor | `Option<VertexDescriptor>` | typed vertex input layout |

builders: `with_depth(fmt)`, `with_stencil(fmt)`, `with_sample_count(n)`,
`with_vertex_descriptor(vd)`, `with_blend(idx, blend)`.

### ColorAttachmentSpec — phase 2 fields

| field | type | semantics |
|-------|------|-----------|
| blend | `Option<BlendState>` | per-attachment blend |
| write_mask | `u32` | RGBA write mask (R=1, G=2, B=4, A=8) |

`BlendState { rgb_op, alpha_op, src_rgb, dst_rgb, src_alpha, dst_alpha }`.
constructors: `BlendState::alpha_over()`, `BlendState::additive()`.

`BlendOp` = Add | Subtract | ReverseSubtract | Min | Max.
`BlendFactor` = Zero | One | SourceColor | OneMinusSourceColor | ... .

### Depth / stencil

`DepthStencil { compare: CompareFunction, write_enabled: bool }`.
constructors: `DepthStencil::less_write()`, `DepthStencil::always_no_write()`.

`CompareFunction` = Never | Less | Equal | LessEqual | Greater | NotEqual
| GreaterEqual | Always.

| Gpu method | signature | semantics |
|-----------|-----------|-----------|
| depth_stencil_state | `(&self, DepthStencil) -> Result<DepthStencilState>` | compile depth/stencil state |
| depth_target | `(&self, w, h, fmt) -> Result<Texture>` | depth render target |
| depth_target_ms | `(&self, w, h, fmt, samples) -> Result<Texture>` | multisampled depth |
| render_target_ms | `(&self, w, h, fmt, samples) -> Result<Texture>` | multisampled color |

### RenderPassDescriptor — phase 2

| method | signature | semantics |
|--------|-----------|-----------|
| depth_attachment | `(&mut self, DepthAttachmentDesc)` | bind depth attachment |

`DepthAttachmentDesc { texture, load_action, store_action, clear_depth }`.
constructor: `DepthAttachmentDesc::clear(&tex)` — load=Clear, store=DontCare, clear_depth=1.0.

`ColorAttachmentDesc::resolve_texture: Option<&Texture>` — MSAA resolve target.

### RenderEncoder — phase 2

| method | signature | semantics |
|--------|-----------|-----------|
| set_cull_mode | `(&self, CullMode)` | None / Front / Back |
| set_front_facing_winding | `(&self, Winding)` | Clockwise / CounterClockwise |
| set_depth_stencil_state | `(&self, &DepthStencilState)` | bind depth/stencil state |
| set_depth_bias | `(&self, bias, slope_scale, clamp)` | depth offset (shadow maps) |
| draw_instanced | `(&self, PrimitiveType, start, count, instances)` | non-indexed instanced |
| draw_indexed | `(&self, PrimitiveType, index_count, IndexType, &Buffer, offset)` | indexed draw |
| draw_indexed_instanced | `(&self, PrimitiveType, index_count, IndexType, &Buffer, offset, instances)` | indexed instanced |

`IndexType` = UInt16 | UInt32 (with `.size()` accessor).

### Vertex descriptor

`VertexDescriptor` — typed vertex input layout.

```rust
VertexDescriptor::new()
    .with_attribute(VertexAttribute { shader_location, format, offset, buffer_index })
    .with_layout(VertexBufferLayout { buffer_index, stride, step, step_rate })
```

`VertexFormat` = Float | Float2 | Float3 | Float4 | Half2 | Half4 |
UChar4Normalized | UInt | Int.
`VertexStep` = Constant | PerVertex | PerInstance.

### apple mapping (phase 2)

| method | ObjC |
|--------|------|
| Gpu::depth_target | [device newTextureWithDescriptor:] (usage RenderTarget, depth format) |
| Gpu::depth_stencil_state | [device newDepthStencilStateWithDescriptor:] |
| RenderPipelineSpec::with_depth | [pipelineDesc setDepthAttachmentPixelFormat:] |
| RenderPipelineSpec::with_sample_count | [pipelineDesc setRasterSampleCount:] |
| RenderPipelineSpec::with_vertex_descriptor | [pipelineDesc setVertexDescriptor:] |
| BlendState | [colorAttachment setBlendingEnabled: + set\*BlendFactor: + set\*BlendOperation:] |
| ColorAttachmentDesc::resolve_texture | [colorAttachment setResolveTexture:] |
| RenderPassDescriptor::depth_attachment | [passDesc depthAttachment] + setTexture/setLoadAction/setStoreAction/setClearDepth: |
| RenderEncoder::set_cull_mode | [encoder setCullMode:] |
| RenderEncoder::set_front_facing_winding | [encoder setFrontFacingWinding:] |
| RenderEncoder::set_depth_stencil_state | [encoder setDepthStencilState:] |
| RenderEncoder::set_depth_bias | [encoder setDepthBias:slopeScale:clamp:] |
| RenderEncoder::draw_indexed | [encoder drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:] |
| RenderEncoder::draw_indexed_instanced | [encoder drawIndexedPrimitives:...:instanceCount:] |
| RenderEncoder::draw_instanced | [encoder drawPrimitives:vertexStart:vertexCount:instanceCount:] |

phase 3+ (deferred): argument buffers, tile shaders, mesh shaders,
acceleration structures.
