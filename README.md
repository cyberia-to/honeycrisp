# honeycrisp

bare-metal Rust drivers for every compute unit on Apple Silicon. experimental, API unstable.

Apple gives you Accelerate — a black box that picks algorithms for you, hides the hardware, and decides what's fast enough. honeycrisp gives you the hardware itself. every NEON lane, every AMX tile, every Metal dispatch, every ANE program — yours to control, yours to schedule, yours to push past what the framework authors thought you'd need.

the focus is workloads Apple never optimized Accelerate for: LLM inference, zero-knowledge proving, and real-time rendering. hand-written NEON and AMX assembly eliminates the abstraction tax — 1.2–10× faster than Accelerate across elementwise, SGEMM, crypto, and media workloads. full benchmark table below.

honeycrisp is also the most complete open-source documentation of Apple Silicon's undocumented hardware. AMX instruction encoding, ANE MIL bytecode format, IOSurface internals, PMU counter access — everything Apple ships without docs, reverse-engineered and captured in Rust code and specs/ files. if you want to understand what the chip actually does, start here.

## the hardware

Apple Silicon has three compute units sharing unified memory. each speaks a different protocol: NEON intrinsics and `.word`-encoded AMX instructions for CPU, Metal framework for GPU, MIL bytecode for ANE. four crates — a shared memory foundation and one driver per compute unit.

## architecture

```
unimem          memory: IOSurface, arena, pool (no internal deps)
  ↑
acpu            driver: CPU/AMX compute (NEON, AMX inline asm)
  ↑               ↑
rane            aruminium       (both depend on unimem + acpu)
ANE hardware    Metal GPU
  ↑ drivers — raw hardware access, no model knowledge
────────────────────────────────────────────────────────
  ↓ runtimes — model graphs, scheduling, inference
cyb/llm         runtime: graph IR, jets, scheduling, model loading
```

drivers expose raw capabilities. runtimes compose them.

## build

```bash
cargo build --release --workspace
cargo test --workspace
```

requires macOS on Apple Silicon (aarch64-apple-darwin).

## benchmark

```bash
cargo run --release -p acpu --example bench_summary
```

80 operations across 16 categories, compared against Apple Accelerate, CommonCrypto, and scalar baselines. representative results on M1 Pro (8P+2E):

| category | acpu | vs | highlight |
|----------|------|----|-----------|
| elementwise f32 | exp, log, tanh, sigmoid, gelu, silu | Apple vvexpf/vvlogf/vvtanhf | 1.25–1.55× faster |
| reductions f32 | sum, dot, length, max, min | Apple vDSP/cblas | parity to 1.88× |
| SGEMM f32 | 32×32 → 4096×4096 | Apple cblas_sgemm | 1.01–10× (small sizes dominate) |
| AI inference | FFN 4K, llama FFN, attention, softmax | Apple cblas + vDSP chain | pipeline ops 1.1–1.4× |
| media | blend, clamp, RGB↔YUV, histogram, resize | Apple vDSP, scalar | 1.3–8× at 1080p |
| crypto | SHA-256, AES-128, PMULL | CommonCrypto | SHA 7×, PMULL 70×+ |
| ZK Goldilocks | field mul, inv, Poseidon2, NTT | nebu pure Rust | 1.1–2× |
| memory BW | STREAM copy/scale/add/triad | M1 Pro reference | parity (95+ GB/s copy) |

full table: `cargo run --release -p acpu --example bench_summary`

## crates

| crate | what | crates.io |
|-------|------|-----------|
| [unimem](unimem/) | zero-copy memory — IOSurface pinned buffers, Tape bump allocator, Grid tensor pool | [![crates.io](https://img.shields.io/crates/v/unimem.svg)](https://crates.io/crates/unimem) |
| [acpu](acpu/) | CPU/AMX compute — NEON vector, AMX matrix, crypto, ZK field arithmetic, PMU | [![crates.io](https://img.shields.io/crates/v/acpu.svg)](https://crates.io/crates/acpu) |
| [aruminium](aruminium/) | Metal GPU — shader compile, pipeline, compute dispatch, pre-resolved IMP | [![crates.io](https://img.shields.io/crates/v/aruminium.svg)](https://crates.io/crates/aruminium) |
| [rane](rane/) | Apple Neural Engine — MIL compile, SRAM load, hardware dispatch | [![crates.io](https://img.shields.io/crates/v/rane.svg)](https://crates.io/crates/rane) |

```rust,ignore
// unimem — one allocation, every device sees it
let block = unimem::Block::open(n * 4)?;

// acpu — AMX matrix multiply, NEON softmax
acpu::matmul_f32(a.as_f32(), b.as_f32(), block.as_f32_mut(), m, n, k);
acpu::vector::softmax(block.as_f32_mut());

// aruminium — Metal GPU compute
let gpu = aruminium::Gpu::open()?;
let buf = gpu.wrap(&block)?;  // zero-copy: MTLBuffer over same physical pages

// rane — ANE hardware dispatch
let program = rane::mil::matmul(64, 64, 64);
let mut model = rane::Program::compile(&program, &[])?;
model.load()?;
model.run(&input, &output)?;
```

## license

cyber license: don't trust. don't fear. don't beg.
