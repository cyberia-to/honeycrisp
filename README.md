# acpu

the lightest cpu.

pure Rust driver for Apple Silicon CPU compute. direct access to AMX
matrix coprocessor, NEON vector engine, numeric extensions (fp16, bf16,
i8mm), sync primitives, and PMU counters. zero external dependencies —
only inline assembly and system calls.

```rust,ignore
use acpu::Block;

let a = Block::open(m * k * 4)?;
let b = Block::open(k * n * 4)?;
let c = Block::open(m * n * 4)?;

// fill a, b...
acpu::matmul_f32(a.as_f32(), b.as_f32(), c.as_f32_mut(), m, n, k);
```

requires macOS + Apple Silicon (M1/M2/M3/M4).

## zero-copy memory

all compute functions work directly with `unimem::Block` — IOSurface-backed
pinned memory shared across CPU, GPU (aruminium), and ANE (rane):

```rust,ignore
use acpu::Block;

let block = Block::open(n * 4)?;
acpu::matmul_f32(a.as_f32(), b.as_f32(), block.as_f32_mut(), m, n, k);
acpu::vector::softmax(block.as_f32_mut());
```

same physical memory, zero copies. Block is re-exported from unimem
for single-import convenience.

## what's inside

| module | what it does |
|--------|-------------|
| `gemm` | sgemm/hgemm/bgemm/qgemm — AMX 16x16 microkernel, GEBP cache blocking, multi-core |
| `vector` | softmax, rmsnorm, exp, log, tanh, sigmoid, gelu, silu, rope |
| `numeric` | fp16, bf16, i8 — bulk NEON conversion, dot product, quantize/dequantize |
| `matrix` | raw AMX coprocessor — load/store, fma32/fma16/fmabf16, tile ops |
| `sync` | memory barriers (dmb/dsb/isb), P-core/E-core affinity, prefetch |
| `pulse` | PMU performance counters via libkperf.dylib |
| `probe` | chip detection, feature flags, core counts |

## numbers

M1 Pro:

```text
sgemm 1024x1024:     0.25 ms  (8.6 GFLOPS/core)
hgemm 1024x1024:     0.18 ms
fp16→f32 16M:         13 GB/s
f32→fp16 16M:         38 GB/s
softmax 4096:         0.8 us
```

## api

```rust,ignore
// matrix multiply (AMX-accelerated)
acpu::matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m, n, k)
acpu::matmul_f16(a: &[u16], b: &[u16], c: &mut [u16], m, n, k)

// vector math (NEON-accelerated, in-place)
acpu::vector::softmax(x: &mut [f32])
acpu::vector::normalize(out: &mut [f32], x: &[f32], w: &[f32], eps: f32)
acpu::vector::exp(x: &mut [f32])
acpu::vector::sigmoid(x: &mut [f32])
acpu::vector::silu(x: &mut [f32])
acpu::vector::gelu(x: &mut [f32])
acpu::vector::rotate(out: &mut [f32], x: &[f32], freqs: &[f32], pos: usize)

// reductions
acpu::vector::sum(x: &[f32]) -> f32
acpu::vector::max(x: &[f32]) -> f32
acpu::vector::dot(a: &[f32], b: &[f32]) -> f32

// type conversion (NEON bulk)
acpu::cast_f32_f16(dst: &mut [u16], src: &[f32])
acpu::cast_f16_f32(dst: &mut [f32], src: &[u16])
acpu::cast_f32_bf16(dst: &mut [u16], src: &[f32])

// memory (re-exported from unimem)
acpu::Block::open(bytes) -> Result<Block>
block.as_f32() / as_f32_mut()
block.as_u16() / as_u16_mut()
```

## build

```bash
cargo build --release
cargo run --example matmul
cargo run --bin acpu_probe
cargo run --release --example bench_sgemm
```

## license

cyber license: don't trust. don't fear. don't beg.
