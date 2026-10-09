# unimem: API Reference

Extracted from the actual implementation in `src/`.

---

## lib.rs

```rust
pub mod ffi;
pub mod block;
pub mod tape;
pub mod grid;

pub use block::{Block, BlockPlan};
pub use tape::Tape;
pub use grid::{Grid, Cell};

#[derive(Debug)]
pub enum MemError {
    ZeroSize,
    SizeOverflow,
    BlockAlignmentInvalid,
    BlockPropertiesFailed,
    BlockCreateFailed,
    BlockLockFailed(i32),
    BlockExtentMismatch { expected: usize, actual: usize },
    BlockAddressInvalid,
}
```

---

## ffi.rs

IOSurface + CoreFoundation raw FFI. No objc2, no wrappers.

```rust
// Types
pub type IOSurfaceRef = *mut c_void;
pub type CFTypeRef = *const c_void;
pub type CFStringRef = *const c_void;
pub type CFDictionaryRef = *const c_void;
pub type CFMutableDictionaryRef = *mut c_void;
pub type CFIndex = isize;
pub type CFNumberType = CFIndex;
pub type kern_return_t = i32;

// IOSurface.framework
extern "C" {
    pub fn IOSurfaceCreate(properties: CFDictionaryRef) -> IOSurfaceRef;
    pub fn IOSurfaceLock(block: IOSurfaceRef, options: u32, seed: *mut u32) -> kern_return_t;
    pub fn IOSurfaceUnlock(block: IOSurfaceRef, options: u32, seed: *mut u32) -> kern_return_t;
    pub fn IOSurfaceGetBaseAddress(block: IOSurfaceRef) -> *mut c_void;
    pub fn IOSurfaceGetAllocSize(block: IOSurfaceRef) -> usize;
    pub fn IOSurfaceGetID(block: IOSurfaceRef) -> u32;
    pub fn IOSurfaceGetPropertyAlignment(property: CFStringRef) -> usize;
    pub fn IOSurfaceAlignProperty(property: CFStringRef, value: usize) -> usize;
}

// CoreFoundation
extern "C" {
    pub fn CFDictionaryCreate(...) -> CFDictionaryRef;
    pub fn CFDictionaryCreateMutable(...) -> CFMutableDictionaryRef; // capacity: CFIndex
    pub fn CFDictionarySetValue(dict, key, value);
    pub fn CFNumberCreate(allocator, theType: CFNumberType, valuePtr) -> *const c_void;
    pub fn CFRelease(cf: CFTypeRef);
    pub static kCFTypeDictionaryKeyCallBacks: c_void;
    pub static kCFTypeDictionaryValueCallBacks: c_void;
}

// Block uses borrowed kIOSurface* keys and owned temporary CF numbers/dictionary.
```

---

## block.rs

Pinned shared memory. Locked at creation, VA stable for lifetime.

```rust
pub struct Block {
    raw: IOSurfaceRef,    // CFRelease on drop
    va: NonNull<u8>,      // stable — locked at creation
    size: usize,
    id: u32,
}

// Send + Sync — immutable after creation

impl Block {
    /// Check native alignment and integer/slice bounds without creating a surface.
    pub fn plan(size: usize) -> Result<BlockPlan, MemError>;

    /// Create pinned IOSurface. Locked immediately.
    /// ~20µs, size-independent. Errors on size=0.
    pub fn open(size: usize) -> Result<Self, MemError>;

    /// Base VA. Always valid. Inline.
    pub fn address(&self) -> *mut u8;

    /// Allocation size in bytes.
    pub fn size(&self) -> usize;

    /// Global IOSurface ID.
    pub fn id(&self) -> u32;

    /// Raw IOSurfaceRef for ANE (rane) / GPU (Metal).
    pub fn handle(&self) -> IOSurfaceRef;
}

#[derive(Clone, Copy, Debug)]
pub struct BlockPlan { /* private checked numeric fields */ }

impl BlockPlan {
    pub fn requested_size(&self) -> usize;
    pub fn row_bytes(&self) -> usize;
    pub fn allocation_size(&self) -> usize;
    pub fn open(&self) -> Result<Block, MemError>;
}

impl Drop for Block {
    fn drop(&mut self) {
        // IOSurfaceUnlock → CFRelease
    }
}
```

IOSurface properties set at creation:

| Key | Value |
|-----|-------|
| IOSurfaceWidth | size |
| IOSurfaceHeight | 1 |
| IOSurfaceBytesPerElement | 1 |
| IOSurfaceBytesPerRow | size aligned to the native row requirement |
| IOSurfaceAllocSize | row stride aligned to the native allocation requirement |
| IOSurfacePixelFormat | 0 |

The plan stores these numeric properties and grants no reservation or initialized
contents. Repeated `open` calls create independent allocations. The actual native
extent must equal the plan or creation fails; base mapping validation precedes
publication. Returned native failures release owned temporaries and any surface,
unlocking first only when a lock succeeded. Hidden framework allocation failures
are outside the returned-error guarantee.

---

## tape.rs

Bump allocator. ~1ns take. Reset in 0.3ns.

```rust
pub struct Tape {
    block: Block,
    cursor: AtomicUsize,
    total: usize,
}

// Send + Sync — atomic cursor, immutable block

impl Tape {
    /// Create tape backed by IOSurface.
    /// ~20µs (block creation cost).
    pub fn start(total: usize) -> Result<Self, MemError>;

    /// Allocate bytes. ~1ns. Lock-free compare_exchange loop.
    /// Returns None if exhausted.
    pub fn take(&self, size: usize, align: usize) -> Option<*mut u8>;

    /// Typed allocation.
    pub fn take_one<T>(&self) -> Option<*mut T>;

    /// Reset to zero. 0.3ns. Pages stay pinned.
    pub fn clear(&self);

    /// Bytes used.
    pub fn used(&self) -> usize;

    /// Bytes free.
    pub fn free(&self) -> usize;

    /// Total total.
    pub fn total(&self) -> usize;

    /// Pointer within this tape?
    pub fn owns(&self, ptr: *const u8) -> bool;

    /// Access backing Block.
    pub fn block(&self) -> &Block;
}
```

Alloc algorithm:
```rust
loop {
    current = cursor.load(Relaxed);
    aligned = (current + align - 1) & !(align - 1);
    new_cursor = aligned + size;
    if new_cursor > total { return None; }
    if cursor.compare_exchange_weak(current, new_cursor, Relaxed, Relaxed).is_ok() {
        return Some(block.address() + aligned);
    }
}
```

---

## grid.rs

Fixed-size tensor grid. ~15ns take+give cycle.

```rust
pub struct Grid<const CELL_SIZE: usize, const CELLS: usize> {
    tape: Tape,
    free: SegQueue<usize>,  // crossbeam lock-free queue
}

pub struct Cell<'a> {
    ptr: *mut u8,
    index: usize,
    _grid: PhantomData<&'a ()>,  // cannot outlive Grid
}

impl<const CELL_SIZE: usize, const CELLS: usize> Grid<CELL_SIZE, CELLS> {
    /// Create grid. CELL_SIZE must be multiple of 64.
    pub fn new() -> Result<Self, MemError>;

    /// Take cell. ~10ns. None if exhausted.
    pub fn take(&self) -> Option<Cell<'_>>;

    /// Give cell back. ~10ns.
    pub fn give(&self, cell: Cell<'_>);

    /// Free cells count.
    pub fn free(&self) -> usize;

    /// Total cells (compile-time).
    pub const fn total(&self) -> usize;

    /// Backing tape.
    pub fn tape(&self) -> &Tape;
}

impl<'a> Cell<'a> {
    pub fn address(&self) -> *mut u8;
    pub unsafe fn bytes(&mut self, len: usize) -> &mut [u8];
    pub fn index(&self) -> usize;
}
```

---

## Usage examples

### Basic tape allocation
```rust
use unimem::Tape;

let tape = Tape::start(64 * 1024 * 1024)?; // 64MB pinned

let buf = tape.take(4096, 64).unwrap();
unsafe { buf.write_bytes(0, 4096); }

let tensor = tape.take(1 << 20, 64).unwrap(); // 1MB, 64-byte aligned
unsafe {
    let slice = std::slice::from_raw_parts_mut(tensor as *mut f32, 256 * 1024);
    slice[0] = 1.0;
}

tape.clear(); // instant — all allocations invalidated
```

### Tensor grid for inference
```rust
use unimem::Grid;

// 32 cells of 4MB each = 128MB IOSurface
let grid: Grid<{ 4 * 1024 * 1024 }, 32> = Grid::new()?;

let mut cell = grid.take().unwrap();
unsafe {
    let data = cell.bytes(4 * 1024 * 1024);
    data[0] = 0xFF;
}
grid.give(cell);
```

### ANE integration via rane
```rust
use unimem::Block;

let block = Block::open(input_size)?;

// Write input data
unsafe {
    let ptr = block.address() as *mut u16;
    // fill fp16 tensor...
}

// Pass to ANE — same physical memory, zero copy
let iosurface_ref = block.handle();
// rane uses IOSurfaceRef directly via _ANEIOSurfaceObject
```

---

## Benchmark results

All numbers measured on Apple Silicon M1. Volatile u64, single thread.

| Operation | unimem | malloc | Vec | Box | mmap | mmap+mlock |
|---|---|---|---|---|---|---|
| take 64 B | 1.3 ns | 16 ns | 17 ns | 15 ns | - | - |
| take 4 KB | 0.9 ns | 18 ns | 20 ns | 83 ns | 464 ns | - |
| take 1 MB | 0.9 ns | 23 ns | 22 ns | - | 461 ns | - |
| free all | 0.3 ns | ~5 ms | ~5 ms | ~5 ms | ~5 ms | ~5 ms |
| grid cycle 4 KB | 15 ns | 19 ns | 20 ns | - | 928 ns | - |
| init 16 MB (lazy) | 23 us | - | - | - | 0.47 us | - |
| init 16 MB (warm) | 1.4 ms | 6.7 us | - | - | - | 1.6 ms |
| write 64 MB | 23.2 GB/s | 23.1 GB/s | 23.4 GB/s | 23.4 GB/s | 22.8 GB/s | 22.7 GB/s |
| read 64 MB | 22.5 GB/s | 21.6 GB/s | 22.8 GB/s | 22.8 GB/s | 22.0 GB/s | 21.2 GB/s |
| pinned | yes | no | no | no | no | yes |
| HW shared | CPU+GPU+AMX+ANE | CPU | CPU | CPU | CPU | CPU |

Init breakdown (16 MB, one-time cost):
- unimem lazy (23 us): IOSurface kernel object + DART registration + lock. Pages not backed.
- unimem warm (1.4 ms): same + walk all 1024 pages (16KB each), trigger page faults.
- malloc touch (6.7 us): malloc reuses cached pages from system allocator — no real faults.
- mmap+mlock+touch (1.6 ms): mmap + wire + page faults. Same cost as unimem warm.

malloc looks faster on init because the system allocator caches freed pages. On first-ever allocation (cold process), malloc+touch would be ~1.5ms too.

Bandwidth identical across all methods — DRAM bottleneck (~23 GB/s volatile u64).

unimem wins on: take speed (15-25x), dealloc speed (16Mx), and hardware sharing (only method where CPU+GPU+AMX+ANE see one buffer without copies).
