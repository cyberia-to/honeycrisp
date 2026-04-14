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
cargo run --example matmul                # verify AMX access
```

every commit: format clean, clippy clean, builds, examples run.

## project: acpu

pure Rust driver for Apple Silicon CPU compute. direct access to
AMX matrix coprocessor, NEON vector engine, numeric extensions
(FP16, BF16, DotProd, I8MM, FCMA, RDM), sync primitives (LSE
atomics, core affinity, memory barriers), and PMU performance
counters. zero external dependencies — only inline assembly,
sysctl, and dlopen for libkperf.

## role in the stack

acpu is a hardware compute driver. it runs math on CPU/AMX/NEON.
it does NOT allocate memory, compile shaders, build graphs, or schedule ops.

```
cyb-mem      memory: IOSurface, arena, pool
acpu         driver: CPU/AMX compute (NEON, AMX inline asm)  ← this crate
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

## sibling drivers

- cyb-mem (https://github.com/cyberia-to/unimem) — memory: IOSurface, arena, zero-copy buffers
- aruminium (https://github.com/cyberia-to/aruminium) — Metal GPU: shaders, buffers, compute
- rane (https://github.com/cyberia-to/rane) — ANE: MIL compile, load, run

## architecture

single crate, six organs:

```
src/
  lib.rs              pub API re-exports, RamxError
  probe.rs            Chip, Caps, Feature, detect()
  matrix/             AMX coprocessor
    mod.rs            AmxCtx lifecycle (set/clr)
    ops.rs            load/store, fma32/fma16/fmabf16/mac16
    regs.rs           XRow, YRow, ZRow typed wrappers
    asm.rs            raw .word encoding macros
  vector/             NEON compute kernels
    mod.rs            dispatch
    math.rs           exp, log, tanh, sigmoid, gelu, silu
    reduce.rs         sum, max, min, dot, norm_l2
    softmax.rs        softmax, rmsnorm
    rope.rs           rotary positional embedding
  numeric/            precision and format extensions
    fp16.rs           FP16 arithmetic + bulk convert
    bf16.rs           BF16 ops + bulk convert
    quant.rs          DotProd, I8MM, quantize/dequantize
    complex.rs        FCMA complex mul-acc
  sync/               concurrency primitives
    mod.rs            barriers, wfe/sev
    affinity.rs       pin_p_core, pin_e_core
    prefetch.rs       PRFM wrappers
  pulse/              performance counters
    mod.rs            PulseCtx, Counter, Snapshot
    ffi.rs            dlopen libkperf, kpc_* symbols
  gemm.rs             sgemm, hgemm, bgemm, qgemm (auto-dispatch)
  convert.rs          bulk conversion re-exports
  probe/
    main.rs           acpu_probe binary
examples/
  matmul.rs           AMX matrix multiply demo
specs/
  README.md           API specification (source of truth)
```

## source of truth

`specs/` is canonical. if specs/ and code disagree, resolve
in specs/ first, then propagate to code.

## key gotchas

- AMX instructions are undocumented. encoded via `.word` in inline asm.
- AMX context is per-thread. each thread needs its own AmxCtx.
- AMX set/clr must bracket all AMX operations.
- NEON registers v8–v15 are callee-saved. inline asm must respect this.
- PMU access requires dlopen of libkperf.dylib (same pattern as rane).
- core affinity uses QoS classes, not hard pinning.
- all public functions operate on caller-owned slices. acpu allocates nothing.
- target: aarch64-apple-darwin only. not cross-platform.

## do not touch

without explicit discussion:
- Cargo.toml dependency versions
- specs/ structure
- LICENSE
- AMX .word encodings (must match corsix/amx documentation)

## quality

file size limit: 500 lines per source file. split into submodules
if exceeded.

every commit:
- type check / lint — zero warnings
- builds clean
- examples run

## coding conventions

- no external dependencies. no C compiler. no frameworks.
- inline asm for AMX (.word encoding), core::arch::aarch64 for NEON.
- dlopen only for libkperf.dylib (PMU).
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
