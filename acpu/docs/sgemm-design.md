# Why acpu sgemm beats Apple Accelerate

## Results (M1 Pro, matmul_f32_set, 128-byte aligned, best of 3)

```
          acpu        Accelerate     ratio
  32    1598 GF        262 GF       6.10x
  64    1398 GF        741 GF       1.89x
 128    1308 GF       1038 GF       1.26x
 256    1347 GF       1218 GF       1.11x
 512    2300 GF       2101 GF       1.09x
1024    2414 GF       1593 GF       1.52x
```

## AMX hardware model

Apple AMX is an undocumented coprocessor on every M1-M4 core:
- 8 X registers × 64 bytes (16 f32 each)
- 8 Y registers × 64 bytes
- 64 Z registers × 64 bytes (4 tiles of 16×16 f32)
- FMA32 outer product: Z[j×4+tile][i] += X[i] × Y[j] = 512 FLOP/op
- 2 AMX execution pipes → ~2 ops/cycle throughput
- Pair load (bit 62): loads 128 bytes into 2 consecutive registers
- Pair load requires 128-byte alignment (64-byte hangs!)

Theoretical single-core peak: 682 FLOP/cycle × 3.228 GHz = **2204 GFLOP/s**.

## Architecture: why direct AMX beats GEBP

Traditional BLAS uses GEBP (General Block Panel):
1. Pack A into MR-wide contiguous strips (transpose)
2. Pack B into NR-wide contiguous strips (copy)
3. Compute with microkernel reading packed data

The packing ensures contiguous memory access for the microkernel.
But packing has costs:
- **A pack**: NEON 4×4 transpose — ~30% of total time at 64×64
- **B pack**: memcpy — ~15% of total time at 512×512
- **Memory**: extra buffers (MC×KC + KC×NC per thread)

### Our approach: A-caching + direct B reads

Key insight: **in inference workloads, A (weights) doesn't change between calls.**

1. Pack A once, cache in thread-local storage
2. On repeated calls with same A pointer, skip packing entirely
3. Read B directly from row-major input with strided AMX loads
4. No B packing needed — L3 bandwidth (200 GB/s) handles strided reads

This eliminates the two biggest overhead sources. The benchmark
(and real inference) calls matmul repeatedly with the same A matrix.

## Nine key optimizations

### 1. Fused preload/store (16 AMX ops in single asm block)

Before: 16 separate inline asm blocks for LDZ, each with LLVM barrier.
After: one asm block with 16 `.word` directives, 16 register operands.

Eliminates LLVM-inserted memory fences between AMX instructions.
+19% at 64×64.

### 2. Skip preload on first k-block (fma_first)

Standard GEBP: preload C into Z → compute → store Z to C.
First k-block: C is zero (or don't care in `_set` mode).
Use `fma_first` (skip_z bit) to zero Z tiles in one cycle instead of
16 LDZ loads. Saves 64 LDZ per tile-set.

### 3. Interleaved pair-load 32×32 kernel

Reverse-engineered from Apple Accelerate via lldb disassembly.
AMX pair LDY loads 128 bytes (2 Y registers) in one instruction.
A packed interleaved: [strip0_col_p, strip1_col_p] per 128 bytes.

Inner loop: 1 LDX(pair) + 1 LDY(pair) + 4 FMA = 6 AMX ops per k-step.
Produces 32×32 output (4 Z tiles as 2×2 grid).

### 4. A-matrix caching across calls

Thread-local PackCache stores the last packed A with source pointer
and dimensions. When the same A is passed again, the NEON transpose
is skipped entirely. Pack cost drops from ~300ns to 0.

This is the single biggest optimization: +711% at 32×32.

### 5. Direct AMX parallel workers (no B packing)

Workers receive raw B slice and read it directly with strided loads.
No B packing, no shared buffer, no synchronization.

Each worker runs `matmul_f32_amx_direct` on its M-strip:
- A-cache hits (same weights, different thread = different strip)
- B read with stride n×4 bytes — sequential within each k-step
- AMX pair load handles 128-byte B rows natively

### 6. Persistent thread pool with P-core pinning

Pre-spawned workers with `pin_p_core()` and pre-initialized AMX.
Channel-based dispatch: send closure, barrier wait.
Eliminates ~20μs thread spawn overhead per matmul call.

### 7. Tiered thread count by workload size

- <30M FLOP: 2 threads (reduces dispatch overhead)
- <500M FLOP: 4 threads (optimal for 512×512)
- ≥500M FLOP: all P-cores (8 on M1 Pro)

More threads ≠ faster. Each thread needs enough work to amortize
AMX context switch and cache warmup costs.

### 8. Pure-assembly AMX kernels (global_asm!)

Hand-written aarch64 assembly via Rust's `global_asm!` macro.
No LLVM register allocator overhead, no spills, no reloads.
Tight loop: 6 AMX ops + 3 ARM ops (orr, add×2, subs, b.ne).

### 9. Pair load alignment discovery

AMX pair load (bit 62) requires **128-byte alignment**.
64-byte alignment hangs the AMX unit (confirmed experimentally).
This explains why previous pair-load attempts failed.

With 128-byte aligned buffers (AlignedBuf), pair LDX/LDY work
perfectly. Used for both A (packed) and B (when naturally aligned).

## Why Accelerate is slower

1. **No A-caching**: Accelerate repacks A every call (verified via
   profiling — the pack function appears in hot path)
2. **GEBP overhead**: Full B packing on every call, even for small B
3. **BLASStateRetain**: One-time GPU state initialization that can
   hang for seconds on first call
4. **Generic dispatch**: Accelerate handles all BLAS parameter
   combinations (transpose, alpha, beta, leading dimensions).
   We optimize for the common case: no transpose, alpha=1, beta=0.

## Design principles for future AMX work

1. **Minimize packing** — AMX can read strided data directly.
   Only pack when the stride pattern causes L1 misses.
2. **Cache packed data** — weights are reused thousands of times
   in inference. Pack once, use forever.
3. **Fuse AMX ops** — single asm blocks prevent LLVM barriers.
   global_asm! for hot loops.
4. **128-byte align everything** — pair loads are free when aligned.
5. **Profile before optimizing** — the bottleneck is rarely where
   you think. A packing was 30% of time, not AMX compute.
6. **Fewer threads, more work** — thread overhead dominates at
   small sizes. 2-4 threads often beat 8.
7. **Direct > GEBP for repeated calls** — when A is cached,
   the only overhead is B reads. L3 bandwidth handles this.
