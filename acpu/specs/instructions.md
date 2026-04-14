# Apple Silicon CPU Instruction Reference

Complete catalog of compute instructions accessible from acpu on
Apple Silicon M1–M4. Each section maps hardware features to the
acpu module that exposes them.

Detected on this machine: Apple M1 Max, AMX v1, 8P+2E cores,
L1=128B line, L2=4MB. FP16, DotProd, FCMA, RDM, LSE, LRCPC: yes.
BF16, I8MM: no (M1, appears in M2+).

---

## 1. AMX — Matrix Coprocessor

Undocumented per-thread coprocessor. 23 valid opcodes (0-22).
Full reference: [specs/amx.md](amx.md).

**Module:** `acpu::matrix`

### Register file
- X: 8 × 64 B = 512 B (input rows for FMA)
- Y: 8 × 64 B = 512 B (input columns for FMA)
- Z: 64 × 64 B = 4096 B (accumulator, 4 independent 16×16 f32 tiles)

### Instructions (opcodes 0-22)

| Op | Name | Description | FLOPS |
|----|------|-------------|-------|
| 0 | LDX | load 64 B → X row | — |
| 1 | LDY | load 64 B → Y row | — |
| 2 | STX | store X row → 64 B | — |
| 3 | STY | store Y row → 64 B | — |
| 4 | LDZ | load 64 B → Z row | — |
| 5 | STZ | store Z row → 64 B | — |
| 6 | LDZI | load interleaved → Z | — |
| 7 | STZI | store interleaved ← Z | — |
| 8 | EXTRX | extract Z slice → X | — |
| 9 | EXTRY | extract Z slice → Y | — |
| 10 | FMA64 | f64 outer product 8×8, Z stride 8 | 128 |
| 11 | FMS64 | f64 outer subtract 8×8 | 128 |
| 12 | **FMA32** | **f32 outer product 16×16, Z stride 4** | **512** |
| 13 | FMS32 | f32 outer subtract 16×16 | 512 |
| 14 | MAC16 | i16 multiply-accumulate 32×32, Z stride 2 | 2048 |
| 15 | FMA16 | f16 outer product 32×32, Z stride 2 | 2048 |
| 16 | FMS16 | f16 outer subtract 32×32 | 2048 |
| 17 | SET/CLR | activate/deactivate AMX (reg 0=set, 1=clr) | — |
| 18 | ??? | Z write, 1 row. possible i32 reduced product | ? |
| 19 | ??? | Z write, 1 row. values are sums (dot product?) | ? |
| 20 | ??? | Z write, 16 rows stride 2. i32 matrix variant | ? |
| 21 | ??? | Z write, 16 rows stride 2. f32 matrix variant | ? |
| 22 | ??? | X write. extract variant | — |

### Encoding
```
.word (0x00201000 + (opcode << 5) + register_encoding)
```
Operand: 64-bit value in GPR. Bit layout per opcode in amx.md.

---

## 2. NEON — 128-bit SIMD

ARM Advanced SIMD. 32 registers × 128 bits. 4-wide f32 or 2-wide f64.

**Module:** `acpu::vector` (math, reduce, softmax, rope)

### Arithmetic (f32 × 4)

| Intrinsic | Op | Throughput | Use in acpu |
|-----------|-----|-----------|-------------|
| `vfmaq_f32` | fused multiply-add | 1/cycle | gemm accumulate, math |
| `vaddq_f32` | add | 1/cycle | accumulate_tile, reduce |
| `vsubq_f32` | subtract | 1/cycle | — |
| `vmulq_f32` | multiply | 1/cycle | math kernels |
| `vdivq_f32` | divide | ~10 cycles | softmax |
| `vsqrtq_f32` | square root | ~10 cycles | normalize |
| `vmaxq_f32` / `vminq_f32` | max/min | 1/cycle | reduce |
| `vabsq_f32` | absolute | 1/cycle | — |
| `vnegq_f32` | negate | 1/cycle | — |
| `vrecpeq_f32` + `vrecpsq_f32` | reciprocal estimate | 2 cycles | softmax fast path |
| `vrsqrteq_f32` + `vrsqrtsq_f32` | rsqrt estimate | 2 cycles | normalize fast path |

### Load / Store

| Intrinsic | Description |
|-----------|-------------|
| `vld1q_f32` | load 4 f32 (unaligned OK) |
| `vst1q_f32` | store 4 f32 |
| `vld1q_f32_x2` / `x4` | load 8/16 f32 |
| `vdupq_n_f32` | broadcast scalar → 4 lanes |
| `vdupq_laneq_f32` | broadcast one lane → 4 |

### Shuffle / Transpose

| Intrinsic | Description | Use in acpu |
|-----------|-------------|-------------|
| `vzip1q_f32` | interleave low halves | pack_a NEON 4×4 transpose |
| `vzip2q_f32` | interleave high halves | pack_a NEON 4×4 transpose |
| `vtrn1q_f32` | transpose even elements | — |
| `vtrn2q_f32` | transpose odd elements | — |
| `vextq_f32` | extract/rotate | rotate |
| `vrev64q_f32` | reverse within 64-bit | rotate |

### Reduction (horizontal)

| Intrinsic | Description |
|-----------|-------------|
| `vaddvq_f32` | sum 4 lanes → scalar |
| `vmaxvq_f32` | max of 4 lanes |
| `vminvq_f32` | min of 4 lanes |
| `vpaddq_f32` | pairwise add |

### Comparison

| Intrinsic | Description |
|-----------|-------------|
| `vcgtq_f32` | greater than → mask |
| `vcgeq_f32` | greater or equal |
| `vceqq_f32` | equal |
| `vbslq_f32` | bitwise select (blend) |

---

## 3. FP16 — Half-Precision (FEAT_FP16)

16-bit IEEE 754 float. Present on all M-series.

**Module:** `acpu::numeric::fp16`

| Intrinsic | Description |
|-----------|-------------|
| `vcvt_f32_f16` | convert 4 f16 → 4 f32 |
| `vcvt_f16_f32` | convert 4 f32 → 4 f16 |
| `vfmaq_f16` | f16 fused multiply-add (8-wide) |
| `vaddq_f16` | f16 add (8-wide) |

AMX FMA16 (opcode 15): 32×32 f16 outer product, 2048 FLOPS/instruction.

---

## 4. BF16 — BFloat16 (FEAT_BF16, M2+)

16-bit brain float. NOT available on M1. Available M2+.

| Intrinsic | Description |
|-----------|-------------|
| `vcvtq_low_bf16_f32` | convert f32 → bf16 |
| `vbfmmlaq_f32` | bf16 matrix multiply-accumulate 2×4 → f32 |
| `vbfdotq_f32` | bf16 dot product |
| `vbfmlalbq_f32` | bf16 widening multiply-add (low) |
| `vbfmlaltq_f32` | bf16 widening multiply-add (high) |

**Module:** `acpu::numeric::bf16` (software convert on M1, native on M2+)

---

## 5. DotProd — Integer Dot Product (FEAT_DotProd)

4-element i8/u8 dot product accumulated into i32. Present on all M-series.

**Module:** `acpu::numeric::quant`

| Intrinsic | Description | FLOPS (per call) |
|-----------|-------------|------------------|
| `vdotq_s32` | dot(i8×4) → i32, 4 accumulators | 32 int ops |
| `vdotq_u32` | unsigned variant | 32 int ops |
| `vusdotq_s32` | mixed signed/unsigned | 32 int ops |

---

## 6. I8MM — Int8 Matrix Multiply (FEAT_I8MM, M2+)

8×8 int8 matrix multiply → int32. NOT available on M1.

| Intrinsic | Description |
|-----------|-------------|
| `vmmlaq_s32` | signed i8 matmul 2×8 × 8×2 → 2×2 i32 |
| `vmmlaq_u32` | unsigned variant |
| `vusmmlaq_s32` | mixed signed/unsigned |

AMX opcode 14 (MAC16): 32×32 i16 multiply-accumulate.
AMX opcodes 20-21: possibly i32/mixed matrix ops (undocumented).

---

## 7. FCMA — Complex Multiply-Accumulate (FEAT_FCMA)

Fused complex number operations. Present on all M-series.

**Module:** `acpu::numeric::complex`

| Intrinsic | Description |
|-----------|-------------|
| `vcmlaq_f32` | complex multiply-accumulate (rotation 0°) |
| `vcmlaq_rot90_f32` | complex mul-acc rotated 90° |
| `vcmlaq_rot180_f32` | complex mul-acc rotated 180° |
| `vcmlaq_rot270_f32` | complex mul-acc rotated 270° |

Computes `(a + bi) × (c + di)` in 2 instructions (0° + 90°).

---

## 8. RDM — Rounding Doubling Multiply (FEAT_RDM)

Fixed-point multiply with rounding. Present on all M-series.

| Intrinsic | Description |
|-----------|-------------|
| `vqrdmulhq_s16` | rounding doubling multiply high (i16) |
| `vqrdmlahq_s16` | rounding doubling multiply-add high |
| `vqrdmlshq_s16` | rounding doubling multiply-sub high |
| `vqrdmulhq_s32` | i32 variant |

Useful for fixed-point quantized inference.

---

## 9. LSE — Large System Extensions (FEAT_LSE)

Hardware atomics. Present on all M-series. Essential for lock-free threading.

**Module:** `acpu::sync`

| Instruction | Intrinsic / asm | Description |
|-------------|----------------|-------------|
| LDADD | `core::sync::atomic` | atomic fetch-add |
| LDSET | — | atomic fetch-or |
| LDCLR | — | atomic fetch-and-not |
| CAS | — | compare-and-swap |
| SWP | — | atomic swap |

All available as `Ordering::Relaxed/Acquire/Release/SeqCst` via
`std::sync::atomic::AtomicUsize` etc.

---

## 10. Barriers and Events

**Module:** `acpu::sync`

| Instruction | Function | Description |
|-------------|---------|-------------|
| DMB ISH | `barrier()` | data memory barrier (inner shareable) |
| DSB ISH | `fence()` | data synchronization barrier |
| ISB | `isb()` | instruction synchronization barrier |
| WFE | `wait()` | wait for event (low-power sleep) |
| SEV | `wake()` | signal event (wake all cores in cluster) |

WFE/SEV pair: lightweight cross-core signaling without mutex overhead.

---

## 11. Prefetch (PRFM)

**Module:** `acpu::sync::prefetch`

| Instruction | Function | Description |
|-------------|---------|-------------|
| PRFM PLDL1KEEP | `prefetch_l1(ptr)` | prefetch for read into L1 |
| PRFM PLDL2KEEP | `prefetch_l2(ptr)` | prefetch for read into L2 |
| PRFM PSTL1KEEP | `prefetch_l1_write(ptr)` | prefetch for write into L1 |

Software prefetch hints. Hardware prefetcher is aggressive on Apple Silicon;
explicit prefetch most useful for non-sequential access patterns (packed panels).

---

## 12. Core Affinity (QoS)

**Module:** `acpu::sync::affinity`

| Function | QoS class | Effect |
|----------|-----------|--------|
| `pin_p_core()` | USER_INTERACTIVE (0x21) | schedule on P-cores |
| `pin_e_core()` | BACKGROUND (0x09) | schedule on E-cores |
| `pin_any()` | DEFAULT (0x15) | no preference |

Uses `pthread_set_qos_class_self_np()`. Soft hint, not hard pinning.

---

## 13. PMU — Performance Monitoring Unit

**Module:** `acpu::pulse`

Access via dlopen of `libkperf.dylib` (private framework).
Requires SIP-related permissions.

| Counter | Description |
|---------|-------------|
| Cycles | CPU cycles |
| Instructions | retired instructions |
| BranchMiss | branch mispredictions |
| L1DMiss | L1 data cache misses |
| L1IMiss | L1 instruction cache misses |

---

## 14. LRCPC — Load-Acquire RCpc (FEAT_LRCPC)

Weaker acquire semantics for better performance on weakly-ordered loads.

| Instruction | Description |
|-------------|-------------|
| LDAPR | load-acquire with release consistency (cheaper than LDAR) |

Available via `core::sync::atomic::Ordering::Acquire` with LLVM optimization.

---

## Feature Matrix by Chip

| Feature | M1 | M2 | M3 | M4 |
|---------|----|----|----|----|
| AMX v1 | YES | YES | — | — |
| AMX v2 | — | — | YES | YES |
| NEON | YES | YES | YES | YES |
| FP16 | YES | YES | YES | YES |
| BF16 | **NO** | YES | YES | YES |
| DotProd | YES | YES | YES | YES |
| I8MM | **NO** | YES | YES | YES |
| FCMA | YES | YES | YES | YES |
| RDM | YES | YES | YES | YES |
| LSE | YES | YES | YES | YES |
| LRCPC | YES | YES | YES | YES |

AMX v2 (M3+): adds opcodes 8-9 (EXTRX/EXTRY), possibly opcodes 18-22.
BF16 + I8MM appear in M2. All other features present since M1.
