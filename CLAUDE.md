# Claude Code Instructions

## project: unimem

pure Rust memory driver for Apple Silicon. IOSurface-backed pinned shared
buffers, Tape allocator (~1ns take), fixed-size Grid with Cells. zero-copy
sharing between CPU, GPU, AMX, and ANE.

## role in the stack

unimem is a hardware memory driver. it allocates and manages buffers.
it does NOT run compute, compile shaders, build graphs, or schedule ops.

```
unimem       memory: IOSurface, Block, Tape, Grid
acpu         driver: CPU/AMX compute (NEON, AMX inline asm)
aruminium    driver: Metal GPU compute (shaders, pipelines)
rane         driver: ANE hardware (MIL compile, dispatch)
  ↑ drivers — raw hardware access, no model knowledge
──────────────────────────────────────────────────────
  ↓ runtimes — model graphs, scheduling, inference logic
cyb/llm      runtime: graph IR, jets, scheduling, model loading
```

all inference logic (attention blocks, transformer layers, model loading,
op scheduling, graph optimization) belongs in the runtime layer
(https://github.com/cyberia-to/cyb), not in the drivers.

drivers expose raw capabilities. runtimes compose them.

## architecture

```
src/
  lib.rs          public API: Block, Tape, Grid, Cell, Layout, Stat, MemError
  ffi.rs          IOSurface + CoreFoundation raw FFI
  block.rs        Block: pinned IOSurface, locked at creation
  tape.rs         Tape: Turing tape bump allocator (~1ns take, 0.3ns clear)
  grid.rs         Grid/Cell: fixed-size cell grid over Tape
  layout.rs       Layout: three-tape inference layout (weights/scratch/history)
```

## source of truth

`specs/` is canonical. if specs/ and code disagree, resolve
in specs/ first, then propagate to code.

## build & verify

```bash
cargo build
cargo test
cargo bench
cargo run --example pipeline --release
```

no special signing. no entitlements. no SIP changes.

## key gotchas

- IOSurface locked once at creation, unlocked at drop. address stable for lifetime.
- Tape take is compare_exchange loop, not fetch_add (no space waste on overshoot).
- Apple Silicon uses 16KB kernel pages, not 4KB.
- Block is Send+Sync (immutable after creation). Tape is Send+Sync (atomic head).
- Grid Cell has lifetime tied to Grid — compile-time use-after-free prevention.
- IOSurfaceRef from block.handle() is directly compatible with rane and aruminium.

## sibling drivers

- acpu (https://github.com/cyberia-to/acpu) — CPU/AMX: matmul_f32, softmax, normalize, rotate, silu
- aruminium (https://github.com/cyberia-to/aruminium) — Metal GPU: shaders, buffers, compute
- rane (https://github.com/cyberia-to/rane) — ANE: MIL compile, load, run

## coding conventions

- raw FFI to IOSurface.framework and CoreFoundation. no objc2, no wrapper crates.
- `cargo fmt` enforced. clippy clean.
- unsafe confined to ffi.rs and Block/Tape internals.

## git workflow

- atomic commits — one logical change per commit
- conventional prefixes: feat:, fix:, refactor:, docs:, test:, chore:
- commit by default after completing a change

## license

cyber license: don't trust. don't fear. don't beg.
