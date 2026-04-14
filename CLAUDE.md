# Claude Code Instructions

## auditor mindset

the project is supervised by an engineer with 30 years of experience.
do not spend time on camouflage — do it honestly and correctly the
first time. one time correctly is cheaper than five times beautifully.

## honesty

never fake results. if a system produces nothing — show nothing.
a dash is more honest than a copied number. never substitute
appearance of progress for actual progress. never generate placeholder
data to fill a gap.

## literal interpretation

when the user says something, they mean it literally. do not
reinterpret. if unsure, ask once. do not guess and iterate.

## chain of verification

for non-trivial decisions affecting correctness:
1. initial answer
2. 3-5 verification questions that would expose errors
3. answer each independently
4. revised answer incorporating corrections

skip for trivial tasks.

## build & verify

```bash
cargo fmt --all                           # format
cargo clippy --workspace -- -W warnings   # lint
cargo build --release --workspace         # build all
cargo run --example vecadd                # verify Metal access
```

every commit: format clean, clippy clean, builds, examples run.

## project: aruminium

pure Rust driver for Apple Metal GPU. compile shaders, create pipelines,
dispatch compute and render work on GPU hardware. zero external
dependencies in the core crate — only macOS system frameworks.

## role in the stack

aruminium is a hardware GPU driver. it compiles shaders and dispatches GPU work.
it does NOT allocate shared memory, run CPU math, build graphs, or schedule ops.

```
cyb-mem      memory: IOSurface, arena, pool
acpu         driver: CPU/AMX compute (NEON, AMX inline asm)
aruminium    driver: Metal GPU compute (shaders, pipelines)  ← this crate
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

## sibling drivers

- cyb-mem (https://github.com/cyberia-to/unimem) — memory: IOSurface, arena, zero-copy buffers
- acpu (https://github.com/cyberia-to/acpu) — CPU/AMX: sgemm, softmax, rmsnorm, rope, silu
- rane (https://github.com/cyberia-to/rane) — ANE: MIL compile, load, run

## architecture

cargo workspace with two members:

- `aruminium` (root crate) — library + metal_probe binary + examples.
  zero external dependencies. links only system frameworks via FFI.
- `benches/` (crate `metal-benches`) — CLI binaries (bench, compare).
  heavy dependencies isolated here.

```
src/                  core library (zero deps)
  lib.rs              public API: Gpu, Buffer, GpuError
  ffi/                Metal.framework, CoreFoundation, libobjc FFI
    mod.rs            types, constants, extern links, string helpers
    selectors.rs      cached ObjC selectors and classes
    trampoline.rs     typed objc_msgSend trampolines
  device.rs           Gpu: discovery, properties, factory methods
  buffer.rs           Buffer: zero-copy shared GPU memory
  fp16.rs             fp16<->f32 conversion (NEON + software fallback)
  command.rs          Queue, Commands
  dispatch.rs         Dispatch, Batch, GpuFuture
  encoder.rs          Encoder, Copier
  shader.rs           ShaderLib, Shader: MSL compilation
  pipeline.rs         Pipeline
  sync.rs             Fence, Event, SharedEvent
  texture.rs          Texture
  probe.rs            metal_probe binary — 5-level capability probe
examples/             runnable demos (vecadd, matmul)
benches/              separate crate with heavy deps (bench, compare)
docs/                 documentation (tutorial, guide, explanations)
specs/                specification (source of truth)
  README.md           API spec — concepts, lifecycle, apple mapping
```

## source of truth

`specs/` is canonical. if specs/ and code disagree, resolve
in specs/ first, then propagate to code.

## pipeline contract

```
MSL source → Metal.framework (linked)
  → MTLLibrary (compile → GPU bytecode)
    → MTLComputePipelineState
      → Command Buffer → Encoder → Dispatch
        → GPU hardware
```

MTLBuffer with StorageModeShared is the zero-copy CPU↔GPU data path.
no lock/unlock needed for shared mode.

## key gotchas

- Metal access requires macOS with Metal-capable GPU. examples fail on Linux.
- Metal.framework is public — linked at compile time, not dlopen.
- all ObjC protocol methods go through objc_msgSend transmuted to typed fn pointers.
- MTLSize struct (24 bytes) passed in registers on ARM64 — verify transmute works.
- newXxx methods return retained objects. commandBuffer is autoreleased — retain it.
- MTLBuffer.contents() returns raw pointer — valid for buffer lifetime.
- never write naive loops for matmul on CPU — use Accelerate.framework.

## do not touch

without explicit discussion:
- Cargo.toml dependency versions
- src/ffi.rs FFI signatures (must match system frameworks)
- specs/ structure
- LICENSE

## quality

file size limit: 500 lines per source file. split into submodules
if exceeded.

every commit:
- type check / lint — zero warnings
- builds clean
- examples run

## coding conventions

- no ObjC, no Swift, no headers. FFI through libobjc + Metal.framework link.
- MTLBuffer for zero-copy CPU↔GPU data. inline NEON asm for fp16.
- `cargo fmt` enforced. clippy clean.

## git workflow

- atomic commits — one logical change per commit
- conventional prefixes: feat:, fix:, refactor:, docs:, test:, chore:
- commit by default after completing a change

## license

cyber license: don't trust. don't fear. don't beg.
