# unimem: Zero-Copy Memory Driver for Apple Silicon

## Goal

Single pinned buffer visible to CPU, GPU, AMX, and ANE — zero copies between pipeline stages. The memory layer for inference on unified memory.

```
file I/O → IOSurface → CPU / AMX / ANE / GPU → result
               ↑
      pinned, shared, never copied
```

v1 adds NVMe DMA via DEXT — full zero-copy from disk to compute.

```
NVMe DMA → IOSurface(IOVA) → AMX → ANE → result
                 ↑
        never moves, never copies
```

---

## Why this exists

Every inference framework on Apple Silicon leaks performance through copies:

- `malloc` gives virtual addresses — hardware can't share them directly
- Metal buffers copy on sync between CPU/GPU
- CoreML copies between pipeline stages internally
- Model loading: NVMe → kernel buf → userspace buf → framework buf

The M1/M2/M3/M4 unified memory makes this unnecessary. CPU, GPU, AMX, ANE share the same physical DRAM. One buffer, visible to everyone. Nobody has built this properly yet.

---

## Architecture: v0 and v1

### v0 — IOSurface-backed (implement now)

IOSurface provides pinned, kernel-managed shared memory accessible to CPU, GPU, AMX, and ANE from userspace. Proven by rane (ANE crate). No special signing, no kernel components.

```
                    IOSurface (pinned, DART-registered)
                   /     |      \         \
                CPU    AMX     GPU       ANE
              (VA)   (VA)   (Metal)  (private API)
```

One copy on model load (file I/O → IOSurface). Zero copies during inference.

### v1 — DEXT + IOSurface (after Apple Developer enrollment)

DriverKit extension reads DART IOVA from IOSurface-backed memory via `CreateMemoryDescriptorFromClient` + `IODMACommand::PrepareForDMA`. NVMe uses IOVA for direct DMA. Full zero-copy from disk to compute.

```
NVMe ──DMA(IOVA)──→ IOSurface ──→ CPU/AMX/ANE/GPU
                         ↑
               DEXT reads IOVA via IODMACommand
```

---

## Crate structure (v0)

```
unimem/
  src/
    lib.rs          ← public API, re-exports, MemError
    ffi.rs          ← raw FFI: IOSurface, CoreFoundation symbols + types
    block.rs        ← Block: pinned IOSurface, locked at creation
    tape.rs         ← Tape: Turing tape bump allocator
    grid.rs         ← Grid/Cell: fixed-size cell grid over Tape
    layout.rs       ← Layout: three-tape inference layout
```

---

## Layer 1 — IOSurface allocation (`block`)

### Purpose

Allocate pinned shared memory via IOSurface.framework. Buffer is visible to CPU, GPU (via Metal wrap), AMX (via CPU pointer), and ANE (via private API). Pages are wired — never swapped, never moved.

### Mechanism

Raw FFI to IOSurface.framework and CoreFoundation. Same pattern as rane — no objc2, no wrapper crates.

IOSurface FFI functions:
- `IOSurfaceCreate(properties)` → IOSurfaceRef
- `IOSurfaceLock(block, options, seed)` → lock for CPU access
- `IOSurfaceUnlock(block, options, seed)` → release lock
- `IOSurfaceGetBaseAddress(block)` → virtual pointer
- `IOSurfaceGetAllocSize(block)` → allocation size
- `IOSurfaceGetID(block)` → global block ID (for sharing)

CoreFoundation FFI functions:
- `CFDictionaryCreate(alloc, keys, values, count, keyCallbacks, valueCallbacks)` → immutable dict
- `CFNumberCreate(alloc, type, valuePtr)` → wrap int as CFNumber
- `CFRelease(ref)` → release any CF object

IOSurface property keys (extern CFStringRef constants):
- `kIOSurfaceWidth` → size (total bytes as width)
- `kIOSurfaceHeight` → 1 (single row)
- `kIOSurfaceBytesPerElement` → 1 (raw bytes)
- `kIOSurfaceBytesPerRow` → size rounded up to its native property alignment
- `kIOSurfaceAllocSize` → row stride rounded up to allocation-property alignment
- `kIOSurfacePixelFormat` → 0 (no pixel format, raw buffer)

`Block::plan(size)` returns an immutable `BlockPlan` with requested size, row
stride and allocation extent. All are checked for integer and Rust slice
representability; alignment arithmetic must agree with `IOSurfaceAlignProperty`.
The native row alignment follows `IOSurfaceRef.h`'s requirement for manually
calculated BytesPerRow. Width remains the requested byte count; padding increases
the returned allocation size when necessary.

A plan carries numbers only and confers no reservation. `BlockPlan::open(&self)`
uses those exact properties; `Block::open(size)` delegates through a plan.
Callers reserve the API-visible backing extent and their own control/metadata
before creation. CF/kernel bookkeeping and physical residency are separate.
`IOSurfaceGetAllocSize` must equal the planned extent; mismatch fails creation.
Planning neither initializes contents nor certifies immutable publication.

### Lock model

Block is locked once at creation time and stays locked until drop. This makes VA stable for the entire lifetime — Tape can hand out raw pointers without per-alloc lock/unlock overhead.

- `open()`: IOSurfaceCreate → IOSurfaceLock → cache VA → ready
- `drop()`: IOSurfaceUnlock → CFRelease

No public lock/unlock API. Lock is internal implementation detail.

### Data: Block

| Field | Type | Description |
|-------|------|-------------|
| ref | IOSurfaceRef | kernel block handle (CFRelease on drop) |
| va | non-null pointer | base virtual address (stable — locked at creation) |
| size | usize | allocation size in bytes |
| id | u32 | global IOSurface ID |

### Thread safety

Block is Send + Sync as an owning mapping. Its six byte/u16/f32 views are unsafe;
mutable views require `&mut Block`. Initialize the entire returned extent, including
padding, before forming any view, even for len/pointer inspection. Byte views
cover `size()`; typed views cover `floor(size()/sizeof(T))*sizeof(T)` bytes.
Exclude overlapping CPU/raw/device writes for shared borrows and all conflicting
access for mutable borrows, across threads and retained imports. Establish prior
device completion and visibility before CPU access. The creation-time lock lasts
until drop; Block performs no lock/unlock or synchronization around typed access.

### Operations

| Operation | Input | Output | Latency | Notes |
|-----------|-------|--------|---------|-------|
| open | size (bytes) | Block or error | ~20us | creates, locks, caches VA |
| address | — | raw pointer | inline, 0ns | returns cached VA (always valid) |
| size | — | usize | inline, 0ns | returns cached size |
| id | — | u32 | inline, 0ns | for sharing with DEXT (v1) |
| handle | — | IOSurfaceRef | inline, 0ns | raw handle for ANE (rane) / GPU (Metal) integration |
| drop | — | — | ~5us | IOSurfaceUnlock + CFRelease |

Historical experiment claims and their provenance limits are preserved in the
[access audit](../../audit/2026-10-10-unimem-block-access/README.md#historical-iosurface-claims).

### Invariants

- Allocation is lazy — kernel reserves VA space, pages backed on first touch
- First-touch page fault: ~1000-1200ns per 16KB page
- Raw Metal import requires a per-mapping proof of runtime VM-page alignment and
  complete actual-extent containment within one VM region.
- Block is Send + Sync; typed CPU access obeys the unsafe borrow contract above.
- Drop sequence: IOSurfaceUnlock → CFRelease — no leaks
- open(0) returns error (ZeroSize)
- Size/alignment failures and returned native null/lock errors return `MemError`.
  Opaque framework allocation failures have no universal recoverability guarantee.
- Owned temporary CF references release on every path; borrowed constant keys
  remain borrowed. A successful lock is unlocked before the surface is released.
- Successful Block mappings are nonnull, typed-view aligned and slice-representable.

---

## Layer 2 — Bump allocator (`tape`)

### Purpose

Fast sub-allocation over a single IOSurface. Zero syscalls after init. Clear in O(1). Block stays pinned between clears.

### Mechanism

Atomic bump pointer over Block. Allocation = compare-exchange loop with alignment rounding. Clear = single atomic store.

### Data: Tape

| Field | Type | Description |
|-------|------|-------------|
| block | Block | backing IOSurface |
| cursor | atomic usize | current allocation offset |
| capacity | usize | total tape size (= block.size) |

### Thread safety

Tape is Send + Sync with an atomic cursor. Raw takes retain no borrow or
initialization proof. Unsafe `warm(&mut self)` requires exclusive access to its
writes; safe `start_warm` operates on a fresh, unpublished Tape.

### Alloc algorithm

```
fn take(size, align) -> Option<*mut u8>:
    loop:
        current = cursor.load(Relaxed)
        aligned = (current + align - 1) & !(align - 1)
        new_cursor = aligned + size
        if new_cursor > capacity:
            return None
        if cursor.compare_exchange_weak(current, new_cursor, Relaxed, Relaxed).ok:
            return block.address() + aligned
```

compare_exchange chosen over fetch_add because:
- No wasted space on overshoot (fetch_add advances cursor even on failure)
- Retry loop cost is negligible — contention on bump allocator is rare
- Bounds check happens before cursor advances, not after

### Operations

| Operation | Input | Output | Latency | Notes |
|-----------|-------|--------|---------|-------|
| start | capacity (bytes) | Tape or error | ~20us | creates IOSurface |
| take | size, alignment | pointer or none | < 5ns | compare_exchange loop |
| take_one | type T | typed pointer or none | < 5ns | uses size_of/align_of T |
| clear | — | — | < 10ns | store 0 to cursor |
| used | — | usize | inline | cursor.load |
| free | — | usize | inline | capacity - used |
| owns | pointer | bool | inline | ptr within [base, base+capacity) |

### Invariants

- Take is lock-free — compare_exchange_weak, no mutex, no syscall
- Alignment: must be power of 2. Minimum 1, recommended 64 for AMX SIMD
- Warm writes one zero byte at offsets 0,16384,… below total, independently of
  runtime VM page size. It preserves the cursor and leaves other bytes unchanged.
  Callers exclude conflicting CPU/raw/device access and live values invalidated
  by those writes; warming does not initialize the whole allocation.
- start(0) returns error (ZeroSize)
- no_std compatible core logic (only depends on atomic ops + pointer arithmetic)
- Clear does NOT zero memory — caller responsible if needed
- Concurrent take is safe (atomic). Concurrent take + clear is NOT safe — caller must ensure no take is in-flight during clear (e.g. single-threaded clear between inference passes)

---

## Layer 3 — Fixed-size tensor grid (`grid`)

### Purpose

Pre-allocated grid for inference tensors. Take and give with zero allocator overhead.

### Mechanism

Tape-backed array of fixed-size cells. Lock-free queue (crossbeam SegQueue) tracks free cell indices. Grid owns the Tape.

### Sizing

Tape capacity = CELL_SIZE * CELLS. CELL_SIZE must be a multiple of 64 (AMX alignment). This guarantees every cell is 64-byte aligned with no wasted padding.

Example: 32 cells of 4MB each → Tape of 128MB → one IOSurface of 128MB.

### Data: Grid

| Parameter | Description |
|-----------|-------------|
| CELL_SIZE | size of each cell in bytes (compile-time, must be multiple of 64) |
| CELLS | number of cells (compile-time) |

### Data: Cell

Cell borrows the Grid (`Cell<'a>` with lifetime tied to `&'a Grid`). This prevents use-after-free at compile time with zero runtime overhead — no Arc, no refcount.

| Field | Type | Description |
|-------|------|-------------|
| ptr | raw pointer | virtual address of cell data |
| id | usize | cell index (for give back to grid) |
| _grid | PhantomData<&'a Grid> | lifetime tie — Cell cannot outlive Grid |

### Operations

| Operation | Input | Output | Latency | Notes |
|-----------|-------|--------|---------|-------|
| new | — | Grid or error | ~20us | creates tape (CELL_SIZE * CELLS), fills free queue |
| take | — | Cell or none | ~10ns | pop from lock-free queue |
| give | Cell | — | ~10ns | push index back to queue |
| free | — | usize | ~10ns | free queue length |
| total | — | usize | inline | CELLS (compile-time) |

### Invariants

- Cell count fixed at compile time — no runtime resize
- Take returns none when grid full — no panic, no block
- Cell lifetime tied to Grid — compile-time use-after-free prevention
- Given cells immediately reusable
- Grid::drop is safe only when all Cells are given (enforced by lifetime — compiler prevents outstanding borrows)

---

## Layer 4 — Hardware integration (`dma`)

### Purpose

v0: no trait. Block and Tape expose address() and handle() directly. Hardware crates (rane, aruminium) consume IOSurfaceRef or raw pointers.

v1: add DmaBuffer trait with iova_segments() when DEXT is available.

### v0 integration points

| Consumer | Gets | How |
|----------|------|-----|
| CPU code | raw pointer | tape.take() or cell.ptr |
| AMX | raw pointer | same as CPU (AMX uses CPU instructions) |
| ANE (rane) | IOSurfaceRef | block.handle() → pass to _ANEIOSurfaceObject |
| GPU (Metal) | IOSurfaceRef | block.handle() → MTLTexture from IOSurface |

No trait needed — direct method calls. Trait appears in v1 when DEXT adds IOVA capability.

---

## Performance targets (v0)

| Operation | Target | Baseline (mimalloc) |
|-----------|--------|---------------------|
| Single take (tape) | < 5ns | ~100ns |
| Sequential write 1GB | > 23 GB/s (volatile) | ~20 GB/s |
| Sequential write 1GB (NEON) | > 60 GB/s | ~50 GB/s |
| Tape clear 1GB | < 10ns | ~5ms (free loop) |
| IOSurface alloc 256MB | < 25us | N/A |
| Grid take/give | < 10ns | N/A |

---

## Error model

| Error | Layer | Cause |
|-------|-------|-------|
| ZeroSize | block, tape | open(0) / start(0) — zero-size allocation requested |
| SizeOverflow | block | requested or aligned extent exceeds integer/slice bounds |
| BlockAlignmentInvalid | block | zero native alignment or alignment calculation disagreement |
| BlockPropertiesFailed | block | CF number or immutable dictionary creation returned null |
| BlockCreateFailed | block | IOSurfaceCreate returned null (OOM or system limit) |
| BlockLockFailed | block | IOSurfaceLock returned non-zero |
| BlockExtentMismatch { expected, actual } | block | native extent differs from the planned allocation |
| BlockAddressInvalid | block | null, misaligned or wrapping CPU mapping |
| InvalidAlignment | tape | alignment not power of 2 |
| tape full | tape | take: not enough space remaining |
| grid full | grid | all cells in use |

Tape.take and Grid.take return Option (None = exhausted) — not Result. These are hot-path operations, Option is lighter.

Block::open and Grid::new return Result<T, MemError> — these are cold-path init operations.

All errors non-panicking. No unwrap, no expect in library code.

---

## Dependencies

| Dependency | Purpose | Layer |
|------------|---------|-------|
| IOSurface.framework | pinned shared buffers | block |
| CoreFoundation | CFDictionary for IOSurface properties | block |
| crossbeam | lock-free queue for grid | grid |
| criterion | benchmarking | benches |

No objc2. No Metal. No Hypervisor. No DriverKit (v0).

---

## Build

```
cargo build
cargo test
cargo bench
```

No special signing required for v0. No entitlements. No SIP changes.

---

## Non-goals (v0)

- No PA/IOVA visibility — that's v1 with DEXT
- No NVMe DMA — that's v1
- No safe API at this layer — unsafe foundation
- No async — synchronous polling
- No allocator trait impl — not a general purpose allocator

---

## Implementation order (v0)

1. `ffi.rs` — IOSurface + CoreFoundation FFI declarations (extern C, link attrs, type aliases)
2. `block.rs` — Block struct: open (create + lock + cache VA), address, handle, drop (unlock + CFRelease)
3. `tape.rs` — Tape struct: start (creates Block), take (compare_exchange), clear, used/free/owns
4. Benchmarks — prove tape take < 5ns, Block throughput matches iosurface_probe experiment
5. `grid.rs` — Grid struct: new (creates Tape), take/give with SegQueue, Cell with lifetime
6. Integration test — roundtrip: take → write → read → verify
7. ANE test — create Block, pass handle() to rane, verify shared access

---

## v1 roadmap: full zero-copy via DEXT

### Prerequisites

- Apple Developer Program enrollment ($99/year)
- DriverKit entitlement provisioning from Apple
- SIP reduced security + `systemextensionsctl developer on` for testing

### Architecture

DEXT (CybMemDriver) acts as bridge between IOSurface and NVMe:

1. Client creates IOSurface, locks it, sends VA + length to DEXT
2. DEXT calls `CreateMemoryDescriptorFromClient(VA, length)` → IOMemoryDescriptor
3. DEXT calls `IODMACommand::PrepareForDMA(descriptor)` → DART IOVA segments (up to 32)
4. Client receives IOVA list, passes to NVMe for direct DMA

### Key API (verified in DriverKit SDK 24.2)

```
IOUserClient::CreateMemoryDescriptorFromClient(
    uint64_t options,
    uint32_t segmentsCount,
    const IOAddressSegment segments[32],
    IOMemoryDescriptor ** memory
)

IODMACommand::PrepareForDMA(
    uint64_t options,
    IOMemoryDescriptor * memory,
    uint64_t offset,
    uint64_t length,
    uint64_t * flags,
    uint32_t * segmentsCount,
    IOAddressSegment segments[32]
)
```

### DART IOMMU makes scatter/gather transparent

On Apple Silicon, PrepareForDMA returns DART IOVAs, not raw physical addresses. DART maps scattered physical pages into contiguous IOVA ranges for each device. NVMe sees contiguous address space even if physical RAM is fragmented.

```
Physical RAM:    [page7] [page3] [page19] [page42]   scattered
                    ↓        ↓        ↓        ↓
DART IOVA:       [0x1000] [0x2000] [0x3000] [0x4000]  contiguous for device
```

`kIOMemoryPhysicallyContiguous` is kernel-only — DriverKit cannot request it. But DART makes it unnecessary for DMA.

### What we verified (experiments)

| Finding | Source |
|---------|--------|
| Hypervisor: hv_vm_map does NOT improve host access latency (~10ns unchanged) | experiments/hyp_probe |
| Hypervisor: minimum 16KB pages on Apple Silicon | experiments/hyp_probe |
| Hypervisor: GPU/ANE cannot see guest IPA — useless for hardware sharing | audit |
| IOMallocContiguous: kernel-only, not callable from userspace | audit |
| IOMemoryDescriptorGetPhysicalAddress: kernel-only C++ method, not a C function | audit |
| DriverKit kIOMemoryPhysicallyContiguous: not available, kernel-only flag | research |
| CreateMemoryDescriptorFromClient: exists in DriverKit SDK 24.2, takes client VA segments | verified in headers |
| IODMACommand::PrepareForDMA: returns up to 32 DART IOVA segments | verified in headers |
| DEXT requires Apple Developer Program ($99/yr) — no self-signing workaround | research |

### DEXT experiment code (prepared, needs signing)

```
experiments/
  dext_iosurface_pa/       ← Path 2: IOSurface → DEXT reads IOVA
    dext/CybMemDriver.iig
    dext/CybMemDriver.cpp  ← CreateMemoryDescriptorFromClient + IODMACommand
    client/src/main.rs     ← Rust client: create IOSurface, call DEXT, print IOVAs
  dext_contiguous_alloc/   ← Path 1: DEXT alloc (limited — no contiguous flag)
```

### v1 open questions

1. Can DEXT read IOVA from IOSurface that was created by another process (the client)?
2. What is PrepareForDMA latency for 256MB IOSurface?
3. How many IOVA segments does a 1GB IOSurface produce? (max 32 per call — may need multiple calls)
4. Can NVMe controller be addressed via DEXT, or does it need a separate NVMe DEXT?

---

## Relationship to the cyber stack

```
unimem        ← this crate: IOSurface + tape + grid
  ↓
rane          ← ANE inference (uses IOSurface for buffers)
aruminium     ← AMX/GPU ops via Metal (can wrap IOSurface)
  ↓
cyb-runtime   ← orchestrates the full pipeline
  ↓
tru           ← runs tri-kernel on the cybergraph
```
