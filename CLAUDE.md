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

known failure mode: the user says "show real numbers" and the agent
reformats display labels instead of showing actual data. this is the
masquerading instinct — optimizing for "looks correct" instead of
"is correct."

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
cargo test --workspace                    # all tests
```

every commit: format clean, clippy clean, builds, tests pass.

## project: honeycrisp

cargo workspace containing the hardware driver layer for Apple Silicon.
four crates, each a driver for one compute unit:

```
unimem       memory: IOSurface, Block, Tape, Grid, zero-copy sharing
acpu         CPU/AMX: NEON vector, AMX matrix, numeric, sync, PMU
aruminium    GPU: Metal shaders, pipelines, buffers, compute dispatch
rane         ANE: MIL compile, neural engine load/run
```

## role in the stack

honeycrisp crates are hardware drivers. they expose raw capabilities.
they do NOT allocate shared memory across devices, build model graphs,
or schedule ops.

```
honeycrisp/
  unimem       memory: IOSurface, arena, pool
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

## dependency graph

```
unimem          (base — no internal deps, IOSurface FFI only)
  ↑
acpu            (depends on unimem)
  ↑
rane            (depends on unimem + acpu)
aruminium       (depends on unimem + acpu)
```

nebu (https://github.com/cyberia-to/nebu) is a separate crate that
optionally depends on acpu for optimized field arithmetic.

## crate details

### unimem — memory driver
- IOSurface-backed pinned shared buffers
- Tape allocator (~1ns take), Grid with Cells
- zero-copy sharing between CPU, GPU, AMX, ANE
- raw FFI to IOSurface.framework and CoreFoundation

### acpu — CPU compute driver
- AMX matrix coprocessor (.word encoded inline asm)
- NEON vector engine (core::arch::aarch64)
- numeric extensions: FP16, BF16, DotProd, I8MM, FCMA
- sync primitives: LSE atomics, core affinity, memory barriers
- PMU performance counters (dlopen libkperf)
- zero external dependencies

### aruminium — GPU compute driver
- Metal.framework shaders, pipelines, buffers
- compute and render dispatch
- FFI through libobjc + Metal.framework link

### rane — ANE driver
- MIL program builder → ANE bytecode
- compile/load/run/unload via objc_msgSend
- IOSurface for zero-copy CPU↔ANE data

## source of truth

each crate's `specs/` is canonical. if specs/ and code disagree,
resolve in specs/ first, then propagate to code.

## key gotchas

- AMX instructions are undocumented. encoded via `.word` in inline asm.
- AMX context is per-thread. each thread needs its own AmxCtx.
- NEON registers v8–v15 are callee-saved. inline asm must respect this.
- IOSurface locked once at creation, unlocked at drop.
- Metal.framework is public — linked at compile time, not dlopen.
- ANE access requires Apple Silicon Mac.
- all public functions in acpu operate on caller-owned slices.
- target: aarch64-apple-darwin only. not cross-platform.

## do not touch

without explicit discussion:
- Cargo.toml dependency versions
- FFI signatures (must match system frameworks)
- specs/ structure
- LICENSE
- AMX .word encodings (must match corsix/amx documentation)

## quality

file size limit: 500 lines per source file. split into submodules
if exceeded.

every commit:
- type check / lint — zero warnings
- builds clean
- tests pass

## coding conventions

- no external dependencies in core crates (except crossbeam-queue in unimem).
- inline asm for AMX (.word encoding), core::arch::aarch64 for NEON.
- FFI through libobjc + dlopen (rane, unimem) or Metal.framework link (aruminium).
- `cargo fmt` enforced (max_width = 100). clippy clean.
- unsafe code confined to asm.rs, ffi.rs, and ops.rs.

## git workflow

- atomic commits — one logical change per commit
- conventional prefixes: feat:, fix:, refactor:, docs:, test:, chore:
- commit by default after completing a change

## shell: nushell

use `nu -c '...'` or `nu script.nu` for scripting.
reserve bash for git commands and system tools only.

## writing style

state what something is directly. never define by negation.

## license

cyber license: don't trust. don't fear. don't beg.
