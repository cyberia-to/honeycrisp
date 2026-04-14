# explanations

how aruminium works and why it's built this way.

## why direct FFI

Metal.framework is an ObjC API. the standard Rust approach is objc2-metal
(safe wrappers over ObjC runtime). aruminium bypasses this entirely:

```
objc2-metal:   Rust -> objc2 runtime -> ObjC protocol wrappers -> Metal.framework
aruminium:     Rust -> objc_msgSend (transmuted fn pointers) -> Metal.framework
```

one less layer. no ObjC protocol conformance checks, no dynamic dispatch
tables, no wrapper allocations. the cost is manual memory management
(retain/release) and unsafe transmutes for every call.

## objc_msgSend dispatch

every Metal API call is an ObjC message send. the C function `objc_msgSend`
takes a target object, a selector (method name), and arguments. the return
type varies — so we transmute `objc_msgSend` to typed function pointers:

```rust
type F = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
let f: F = std::mem::transmute(objc_msgSend as *const c_void);
f(device, sel_newCommandQueue)
```

this is the same mechanism the ObjC compiler uses. safe as long as the
type signature matches the actual method.

## selector caching

`sel_registerName("commandBuffer")` resolves a C string to a selector
pointer. idempotent — same string always returns same pointer. we cache
the result in an `AtomicPtr` with `Relaxed` ordering:

- ARM64: compiles to plain `ldr` (no memory barrier)
- first call: resolves + stores
- subsequent calls: single atomic load
- race on first init is benign (same value written by all threads)

50+ selectors are cached this way. hot-path cost: one pointer load.

## three dispatch tiers

### tier 1: objc_msgSend (standard path)

every API call goes through `objc_msgSend` with cached selectors.
safe wrappers with null checks and Result return types. used by
examples, probe, and general-purpose code.

### tier 2: ARC fast-retain

`objc_retainAutoreleasedReturnValue` called immediately after
`objc_msgSend`. the ObjC runtime recognizes this pattern and skips
the autorelease+retain round-trip. saves ~100ns per command buffer
creation. used by default `commands()` and `encoder()`.

### tier 3: pre-resolved IMP (Dispatch)

`class_getMethodImplementation` resolves a selector to a direct
function pointer (IMP) at construction time. subsequent calls bypass
`objc_msgSend` entirely — no selector lookup, no method cache check,
no dispatch overhead:

```
objc_msgSend path:  selector -> method cache -> IMP -> call
IMP path:           IMP -> call
```

saves ~50ns per call. for inference loops doing 600+ dispatches per
token, this adds up.

## memory management

Metal objects follow ObjC retain/release semantics:

- `newXxx` methods return **retained** objects (caller owns)
- `commandBuffer` returns **autoreleased** (runtime owns temporarily)
- `objc_retain` / `objc_release` are direct C calls (not msg_send)

aruminium tracks ownership with `owned: bool` on command buffers
and encoders. `Drop` only calls `release` on owned objects.

five command buffer variants trade safety for speed:

| variant | retain | null check | resource refs |
|---------|--------|------------|---------------|
| command_buffer | ARC fast | yes | retained |
| command_buffer_unretained | none | yes | retained |
| command_buffer_fast | explicit | yes | unretained |
| command_buffer_unchecked | ARC fast | no | unretained |
| command_buffer_autoreleased | none | no | unretained |

"unretained references" means Metal skips retain/release on all
buffers, textures, and pipelines used in the command buffer.
caller must ensure resources outlive the command buffer.

## shared vs private buffers

**shared** (`MTLResourceStorageModeShared`): CPU and GPU share the
same physical memory. zero-copy — write from CPU, read on GPU,
no synchronization needed between command buffer boundaries.
the `contents()` pointer is cached at creation time.

**private** (`MTLResourceStorageModePrivate`): GPU-only memory.
Metal has full control over placement and caching. higher bandwidth
for inter-kernel buffers. CPU access requires blit encoder copy.

## fp16 conversion

half-precision floats (fp16) are the common weight format for inference.
conversion uses inline ARM64 NEON assembly:

- single value: `fcvt` instruction (h->s or s->h)
- bulk (32/iter): 4x unrolled `fcvtl`/`fcvtn` with `ldp`/`stp`
- tail: 8/iter NEON, then scalar fallback

throughput: 58-72 GB/s on M1 Pro (memory bandwidth limited).

software fallback for non-aarch64 platforms handles IEEE 754
half-precision format manually (sign, exponent, mantissa bit shifting).

## MTLSize on ARM64

`MTLSize` is a 24-byte struct (3x `usize`). on ARM64, structs up to
4 registers are passed in registers, not on the stack. the transmute
from Rust struct to C calling convention works because both use the
same ABI. this is verified by the dispatch examples — if the ABI
didn't match, threadgroup dimensions would be garbage.

## batch encoding

the single most important optimization for inference: encoding
multiple dispatches into one command buffer.

without batching: N dispatches = N command buffers = N round-trips
to the GPU driver. each round-trip costs ~20us.

with batching: N dispatches = 1 command buffer = 1 round-trip.
pipeline switches happen inside the same encoder — different
`setComputePipelineState` calls within one `computeCommandEncoder`.

`dispatch_batch_async` adds pipelining: commit batch N, start
encoding batch N+1 on CPU while GPU executes batch N. overlap
hides encoding latency completely.

## relationship to cyb-llm

```
cyb-llm          inference runtime (graphs, models, scheduling)
cyb-llm/backend  Metal backend (MSL shaders, jet dispatch, frame allocator)
aruminium        this crate (device, buffer, pipeline, dispatch)
```

aruminium does not know about models, operations, or shaders.
it provides the Metal.framework API surface that the backend needs.
one-way dependency: cyb-llm depends on aruminium, not the reverse.
