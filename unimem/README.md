# unimem

unified memory for Apple Silicon. IOSurface-backed pinned buffers visible to every compute unit — CPU, GPU, AMX, ANE — through one allocation.

Apple Silicon shares DRAM across all devices, but every framework still copies: NVMe → kernel → userspace → framework → device. 

unimem eliminates the middle: allocate once via IOSurface, get a stable virtual address pinned for the buffer's lifetime, pass the same physical pages to NEON, AMX, Metal, and ANE. zero copies. zero translations.

experimental. API unstable.

```rust,ignore
let block = unimem::Block::open(16 * 1024 * 1024)?;
let ptr = block.as_f32_mut();  // stable VA, visible to all devices
block.handle()                 // IOSurfaceRef for ANE/GPU
```

requires macOS + Apple Silicon.

## what's inside

four layers, each built on the one below:

| layer | type | what it does | latency |
|-------|------|-------------|---------|
| Block | pinned buffer | IOSurface-backed shared memory, locked at creation | ~20 us create |
| Tape | bump allocator | compare-exchange cursor over a Block, instant rewind | ~1 ns take, ~0.3 ns clear |
| Grid | tensor pool | fixed-size cells with lock-free queue (crossbeam) | ~10 ns take/give |
| Layout | inference layout | three tapes — weights, scratch, history — matching inference lifecycle | clear_pass / clear_talk |

## why not Vec / malloc / mmap

all achieve the same ~23 GB/s bandwidth (DRAM-bound). the difference:

- Vec/malloc: not pinned, not IOSurface, invisible to GPU and ANE. requires explicit copy to each device
- mmap: pinned but no IOSurface handle. GPU and ANE cannot address it
- Block: pinned + IOSurface. one allocation, every device reads/writes the same physical pages. `block.handle()` passes directly to aruminium and rane

Tape is 15–25x faster than malloc for allocation. Grid take/give cycle is ~15 ns. neither touches the kernel after initial Block creation.

## api

```rust,ignore
// pinned shared buffer
Block::open(bytes) -> Result<Block>
block.as_ptr() -> *mut u8
block.as_f32() / as_f32_mut() / as_u16() / as_u16_mut()
block.handle() -> IOSurfaceRef  // for GPU/ANE
block.id() -> u32               // for cross-process sharing

// bump allocator over a Block
Tape::start(bytes) -> Result<Tape>
tape.take(size, align) -> Option<*mut u8>   // ~1 ns
tape.clear()                                 // ~0.3 ns, rewind cursor
tape.block() -> &Block

// fixed-size tensor pool
Grid::<CELL_SIZE, CELLS>::new() -> Result<Grid>
grid.take() -> Option<Cell<'_>>    // lifetime-tied to Grid
grid.give(cell)

// three-tape inference layout
Layout::new(weights, scratch, history) -> Result<Layout>
layout.weights() -> &Tape   // load once
layout.scratch() -> &Tape   // clear per token
layout.history() -> &Tape   // clear per conversation
layout.clear_pass()          // scratch only
layout.clear_talk()          // scratch + history
```

## build

```bash
cargo build --release
cargo test
cargo bench          # alloc + bandwidth benchmarks
cargo run --example pipeline   # transformer layer: unimem vs standard alloc
```

## internals

raw FFI to IOSurface.framework and CoreFoundation. no objc2, no wrappers. IOSurface is locked once at creation (IOSurfaceLock), unlocked at drop. Apple Silicon uses 16 KB kernel pages. Block is Send + Sync (immutable after creation). Tape is Send + Sync (atomic cursor).

## license

cyber license: don't trust. don't fear. don't beg.
