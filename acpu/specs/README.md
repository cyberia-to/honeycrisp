# acpu — API specification

pure Rust driver for Apple Silicon CPU compute. direct access to
every useful compute unit in M1–M4: matrix coprocessor, vector
engine, numeric extensions, atomics, memory system, performance
counters. zero external dependencies — only inline assembly and
system calls.

## organs

six groups. each maps to a compute unit or system resource in silicon.

| organ | silicon | what it does |
|-------|---------|-------------|
| **probe** | sysctl, MRS | detect chip, enumerate capabilities |
| **matrix** | AMX coprocessor | 512-bit matrix fma, undocumented Apple hardware |
| **vector** | NEON (AdvSIMD) | 128-bit SIMD, 32 registers, the workhorse |
| **numeric** | FP16, BF16, DotProd, I8MM, FCMA, RDM | precision and format extensions inside NEON |
| **sync** | LSE, LRCPC, barriers | atomics, memory ordering, core affinity |
| **pulse** | PMU (Apple kpc) | cycle counters, cache misses, branch stats |

## concepts

| concept | what it is |
|---------|-----------|
| chip | identified Apple Silicon die — M1, M1 Pro, M2, M3, M4, etc. |
| features | runtime capability flags — what extensions this chip has |
| matrix | AMX context — set/clr bracket, owns matrix register file |
| xrow, yrow, zrow | typed AMX register handles (8 × 512-bit each bank) |
| kernel | a compute operation dispatched to the best available unit |
| lane | NEON vector register (128-bit, v0–v31) |

---

# probe — chip detection

runtime identification of Apple Silicon chip and its capabilities.
all detection is zero-cost after first call (cached in static).

## chip

| variant | CPU | AMX ver | extensions |
|---------|-----|---------|-----------|
| M1 | Firestorm/Icestorm | 1 | NEON, FP16, DotProd, FCMA, RDM, LSE |
| M1 Pro/Max/Ultra | Firestorm/Icestorm | 1 | same as M1 |
| M2 | Avalanche/Blizzard | 2 | + BF16, I8MM, AMX bf16 ops |
| M2 Pro/Max/Ultra | Avalanche/Blizzard | 2 | same as M2 |
| M3 | Everest/Sawtooth | 3 | + AMX ldx/ldy, matint |
| M3 Pro/Max/Ultra | Everest/Sawtooth | 3 | same as M3 |
| M4 | Dorada/Brava | 4 | + AMX extrh/extrv, vecfp/vecint |
| M4 Pro/Max | Dorada/Brava | 4 | same as M4 |

## features

| field | type | semantics |
|-------|------|-----------|
| chip | Chip | enum variant |
| amx_ver | u8 | 1–4, matches chip generation |
| has_fp16 | bool | FEAT_FP16 — always true M1+ |
| has_bf16 | bool | FEAT_BF16 — M2+ |
| has_dotprod | bool | FEAT_DotProd — always true M1+ |
| has_i8mm | bool | FEAT_I8MM — M2+ |
| has_fcma | bool | FEAT_FCMA — always true M1+ |
| has_rdm | bool | FEAT_RDM — always true M1+ |
| has_lse | bool | FEAT_LSE — always true M1+ |
| has_lrcpc | bool | FEAT_LRCPC — always true M1+ |
| p_cores | u8 | performance core count |
| e_cores | u8 | efficiency core count |
| l1_line | usize | L1 cache line size (bytes) |
| l2_size | usize | L2 cache size per cluster (bytes) |

| method | signature | semantics |
|--------|-----------|-----------|
| scan | `() -> &'static Features` | detect once, return cached reference |
| chip | `() -> Chip` | shortcut for scan().chip |
| has | `(Feature) -> bool` | query single feature flag |

### system mapping

| field | source |
|-------|--------|
| chip | `sysctl hw.optional.arm.FEAT_*` + `machdep.cpu.brand_string` |
| core counts | `sysctl hw.perflevel0.physicalcpu` / `hw.perflevel1.physicalcpu` |
| cache | `sysctl hw.perflevel0.l1dcachesize` / `hw.perflevel0.l2cachesize` |
| features | `sysctl hw.optional.arm.FEAT_FP16` etc. or `MRS ID_AA64ISAR0_EL1` |

---

# matrix — AMX coprocessor

Apple Matrix coprocessor. undocumented. three register banks:
X (8 × 512-bit), Y (8 × 512-bit), Z (8 × 512-bit).
total: 4608 bytes of matrix register state.

AMX instructions live in reserved ARM encoding space at
`0x201000 + (opcode << 5) + operand`. emitted via `.word` in
inline assembly.

## context lifecycle

| method | signature | semantics |
|--------|-----------|-----------|
| new | `() -> Matrix` | AMX_SET — enable coprocessor, zero registers |
| drop | automatic | AMX_CLR — disable coprocessor |

AMX_SET/AMX_CLR are bracketing instructions. all AMX operations
must occur between set and clr. context is per-thread.

### encoding

```
AMX_SET: NOP; NOP; NOP; .word (0x201000 + (17 << 5) + 0)
AMX_CLR: NOP; NOP; NOP; .word (0x201000 + (17 << 5) + 1)
```

## register model

| type | bank | count | width | total |
|------|------|-------|-------|-------|
| XRow | X | 8 | 512 bits (64 bytes) | 512 bytes |
| YRow | Y | 8 | 512 bits (64 bytes) | 512 bytes |
| ZRow | Z | 8 | 512 bits (64 bytes) | 512 bytes |

registers are typed wrappers: `XRow(0)` .. `XRow(7)`.

Z rows are accumulators — fma/mac results land here.

## load / store

| method | signature | semantics |
|--------|-----------|-----------|
| ldx | `(&self, row: XRow, ptr: *const u8)` | load 64 bytes into X row |
| ldy | `(&self, row: YRow, ptr: *const u8)` | load 64 bytes into Y row |
| stx | `(&self, row: XRow, ptr: *mut u8)` | store 64 bytes from X row |
| sty | `(&self, row: YRow, ptr: *mut u8)` | store 64 bytes from Y row |
| stz | `(&self, row: ZRow, ptr: *mut u8)` | store 64 bytes from Z row |
| ldzi | `(&self, row: ZRow, ptr: *const u8)` | load into Z row (interleaved) |

### encoding

operand GPR holds: `ptr | (row_index << 56)` for load/store.

| op | opcode |
|----|--------|
| ldx | 0 |
| ldy | 1 |
| stx | 2 |
| sty | 3 |
| ldz | 4 |
| stz | 5 |
| ldzi | 6 |
| stzi | 7 |

## compute

| method | signature | semantics |
|--------|-----------|-----------|
| fma32 | `(&self, x: XRow, y: YRow, z: ZRow)` | Z += X × Y (fp32, outer product) |
| fma16 | `(&self, x: XRow, y: YRow, z: ZRow)` | Z += X × Y (fp16 inputs, fp32 accum) |
| fmabf16 | `(&self, x: XRow, y: YRow, z: ZRow)` | Z += X × Y (bf16 inputs, fp32 accum) — M2+ |
| mac16 | `(&self, x: XRow, y: YRow, z: ZRow)` | Z += X × Y (int16, int32 accum) |
| matint | `(&self, x: XRow, y: YRow, z: ZRow)` | integer matrix op — M3+ |
| vecfp | `(&self, x: XRow, z: ZRow)` | vector fp op — M4+ |
| vecint | `(&self, x: XRow, z: ZRow)` | vector int op — M4+ |
| extrh | `(&self, z: ZRow, ptr: *mut u8)` | extract horizontal from Z — M4+ |
| extrv | `(&self, z: ZRow, ptr: *mut u8)` | extract vertical from Z — M4+ |

### encoding (compute)

operand GPR holds bit-packed config:

```
fma32:  bits[4:0]  = x_offset
        bits[8:6]  = y_row_select (0–7)
        bits[19:10] = z_row
        bit[27]    = accumulate mode (1 = +=, 0 = =)
opcode: 10 (fma32), 11 (fma16), 12 (fmabf16), 13 (mac16)
M3+:    14 (matint)
M4+:    15 (vecfp), 16 (vecint), 8 (extrh), 9 (extrv)
```

exact bit layout per opcode documented in corsix/amx Instructions.md.

## fma geometry

```
X row:  64 bytes = 16 × fp32  or  32 × fp16
Y row:  64 bytes = 16 × fp32  or  32 × fp16
Z row:  64 bytes = 16 × fp32

fma32:  Z[16×16] += outer_product(X[16], Y[16])
fma16:  Z[16×16] += outer_product(X[32→16], Y[32→16])  (fp16 in, fp32 accum)
```

one fma32 = 256 fp32 multiply-accumulates per instruction.

---

# vector — NEON (AdvSIMD)

ARM Advanced SIMD. 32 registers × 128 bits. documented, stable,
available on all AArch64. accessed via `core::arch::aarch64`
intrinsics or inline assembly.

acpu exposes NEON through typed kernel functions, not raw
intrinsics. the user calls `acpu::exp(slice)`, not
`vfmaq_f32(a, b, c)`.

## register file

```
v0–v31:  128-bit SIMD registers
  as f32:  4 lanes  (float32x4_t)
  as f16:  8 lanes  (float16x8_t)
  as f64:  2 lanes  (float64x2_t)
  as i32:  4 lanes  (int32x4_t)
  as i16:  8 lanes  (int16x8_t)
  as i8:  16 lanes  (int8x16_t)
  as u8:  16 lanes  (uint8x16_t)
```

## operations used

not a full NEON reference — only the instruction classes acpu uses.

| class | instructions | what for |
|-------|-------------|----------|
| arithmetic | fadd, fmul, fmla, fmls, fneg, fabs | vector math kernels |
| compare | fcmeq, fcmgt, fcmge | branchless select |
| convert | fcvt (fp16↔fp32), fcvtn, fcvtl | precision conversion |
| load/store | ld1, st1, ld1q (4-reg), st1q | bulk data movement |
| shuffle | tbl, trn1, trn2, zip1, zip2, uzp1, uzp2 | transpose, interleave |
| reduce | faddp, fmaxnmv, fminnmv | horizontal sum, min, max |
| bitwise | and, orr, eor, bsl, bif, bit | mask ops, branchless logic |
| shift | shl, sshr, ushr, ssra | fixed-point, quantization |
| reciprocal | frecpe, frecps, frsqrte, frsqrts | fast 1/x, 1/√x (Newton steps) |

---

# numeric — precision and format extensions

extensions that operate within the NEON register file but add
specialized data formats. each gated by a feature flag in caps.

## fp16 (FEAT_FP16) — M1+

native half-precision arithmetic in NEON. not just conversion —
actual compute in 16-bit.

| operation | instructions | semantics |
|-----------|-------------|-----------|
| arithmetic | fadd(h), fmul(h), fmla(h) | fp16 vector math |
| convert | fcvt h↔s, fcvtn, fcvtl | fp16 ↔ fp32 bulk |
| compare | fcmeq(h), fcmgt(h) | fp16 comparison |

### bulk conversion (NEON vectorized)

| function | signature | semantics |
|----------|-----------|-----------|
| cast_f16_f32 | `(&mut [f32], &[u16])` | bulk fp16→fp32, 32/iter (4× unrolled) |
| cast_f32_f16 | `(&mut [u16], &[f32])` | bulk fp32→fp16, 32/iter (4× unrolled) |
| fp16_to_f32 | `(u16) -> f32` | scalar, single NEON fcvt |
| f32_to_fp16 | `(f32) -> u16` | scalar, single NEON fcvt |

## bf16 (FEAT_BF16) — M2+

brain float: 8-bit exponent, 7-bit mantissa. same range as fp32,
less precision. preferred for training.

| operation | instructions | semantics |
|-----------|-------------|-----------|
| dot | bfdot | bf16 dot product → fp32 |
| matmul | bfmmla | bf16 2×4 × 4×2 → fp32 2×2 |
| convert | bfcvt, bfcvt2 | fp32 → bf16 (truncate) |

### bulk conversion

| function | signature | semantics |
|----------|-----------|-----------|
| cast_f32_bf16 | `(&mut [u16], &[f32])` | bulk fp32→bf16 via bfcvt |
| cast_bf16_f32 | `(&mut [f32], &[u16])` | bulk bf16→fp32 (shift left 16) |

## dotprod (FEAT_DotProd) — M1+

INT8 dot product. four int8 × int8 multiplies accumulated into
one int32. the foundation of quantized inference.

| operation | instructions | semantics |
|-----------|-------------|-----------|
| signed | sdot | 4 × (i8 × i8) → i32, accumulated |
| unsigned | udot | 4 × (u8 × u8) → u32, accumulated |
| mixed | usdot | 4 × (u8 × i8) → i32 — M2+ only |

## i8mm (FEAT_I8MM) — M2+

INT8 matrix multiply. 2×8 × 8×2 → 2×2 int32. eight times the
throughput of scalar int8 multiply.

| operation | instructions | semantics |
|-----------|-------------|-----------|
| signed | smmla | i8[2×8] × i8[8×2] → i32[2×2] |
| unsigned | ummla | u8[2×8] × u8[8×2] → u32[2×2] |
| mixed | usmmla | u8[2×8] × i8[8×2] → i32[2×2] |

## fcma (FEAT_FCMA) — M1+

complex number arithmetic in NEON registers. pairs of floats
treated as (real, imag).

| operation | instructions | semantics |
|-----------|-------------|-----------|
| rotate | fcadd | complex add with 90° or 270° rotation |
| mul-acc | fcmla | complex multiply-accumulate (0°/90°/180°/270°) |

used for: FFT, complex-valued attention, signal processing.

## rdm (FEAT_RDM) — M1+

rounding doubling multiply. fixed-point DSP without overflow.

| operation | instructions | semantics |
|-----------|-------------|-----------|
| mul-add | sqrdmlah | sat(a + round(b×c >> shift)) |
| mul-sub | sqrdmlsh | sat(a - round(b×c >> shift)) |

used for: fixed-point quantization, audio DSP.

---

# sync — atomics, ordering, affinity

concurrency primitives for parallel compute across P-cores and
E-cores. not for general concurrent programming — specifically
for multi-threaded GEMM, producer-consumer pipelines, and
work-stealing.

## lse atomics (FEAT_LSE) — M1+

hardware atomic operations. single instruction, no LL/SC loop.
essential for lock-free thread pool in parallel GEMM.

| operation | instructions | semantics |
|-----------|-------------|-----------|
| fetch-add | ldadd, ldaddal | atomic add, return old value |
| compare-swap | cas, casal | atomic CAS, single instruction |
| swap | swp, swpal | atomic exchange |
| clear | ldclr | atomic bit clear |
| set | ldset | atomic bit set |

### Rust mapping

LSE atomics are used automatically by LLVM on AArch64 when
`-Ctarget-feature=+lse` is set (default on macOS). `std::sync::atomic`
operations compile to single LSE instructions. acpu does not wrap
these — Rust already does the right thing.

acpu exposes LSE through the sync module only where the standard
library is insufficient (e.g. custom fence patterns, spin-wait
with WFE).

## lrcpc (FEAT_LRCPC) — M1+

load-acquire with weaker ordering. faster than full acquire
barrier for producer-consumer patterns.

| operation | instructions | semantics |
|-----------|-------------|-----------|
| load-acquire | ldapr | load with acquire semantics, weaker than ldar |

used between pack-thread and compute-thread in parallel GEMM.

## memory barriers

| function | instruction | semantics |
|----------|-----------|-----------|
| barrier | DMB ISH | data memory barrier, inner shareable |
| fence | DSB ISH | data sync barrier, inner shareable |
| isb | ISB | instruction sync barrier |
| wait | WFE | wait for event (low-power spin) |
| wake | SEV | signal event (wake spinning core) |

wait/wake pair: spin-waiting threads use WFE to sleep until
producer calls SEV. saves power and thermal headroom during
parallel GEMM synchronization.

## core affinity

pin threads to performance or efficiency cores.

| function | signature | semantics |
|----------|-----------|-----------|
| pin_p_core | `()` | pin current thread to P-core cluster |
| pin_e_core | `()` | pin current thread to E-core cluster |
| pin_any | `()` | remove pinning, allow migration |

### system mapping

| function | system call |
|----------|------------|
| pin_p_core | `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE)` |
| pin_e_core | `pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND)` |
| pin_any | `pthread_set_qos_class_self_np(QOS_CLASS_DEFAULT)` |

note: macOS QoS classes influence core scheduling but do not
guarantee hard affinity. USER_INTERACTIVE strongly prefers P-cores.
BACKGROUND strongly prefers E-cores. for inference, pin all compute
threads to P-cores.

## prefetch

| function | signature | semantics |
|----------|-----------|-----------|
| prefetch_l1 | `(ptr: *const u8)` | PRFM PLDL1KEEP — prefetch into L1 |
| prefetch_l2 | `(ptr: *const u8)` | PRFM PLDL2KEEP — prefetch into L2 |
| prefetch_l1_write | `(ptr: *mut u8)` | PRFM PSTL1KEEP — prefetch for write |

used in GEMM packing loops to hide memory latency.

---

# pulse — performance counters

Apple Performance Monitoring Unit. undocumented kpc_* API in
`/usr/lib/libkperf.dylib`. gives cycle-accurate measurements
without Instruments or dtrace.

accessed via dlopen at runtime (same pattern as rane uses for
AppleNeuralEngine.framework).

## counters

| counter | what it counts |
|---------|---------------|
| cycles | CPU cycles (fixed counter 0) |
| instructions | instructions retired (fixed counter 1) |
| branches | branches retired |
| branch_misses | branch mispredictions |
| l1d_misses | L1 data cache misses |
| l1i_misses | L1 instruction cache misses |
| l2_misses | L2 cache misses |

## context

| method | signature | semantics |
|--------|-----------|-----------|
| new | `(counters: &[Counter]) -> Result<Counters>` | configure PMU via kpc_set_config |
| start | `(&mut self)` | kpc_set_counting + kpc_set_thread_counting |
| read | `(&self) -> Snapshot` | kpc_get_thread_counters64 |
| stop | `(&mut self)` | disable counting |
| elapsed | `(&self, a: &Snapshot, b: &Snapshot) -> Counts` | delta between two snapshots |

### system mapping

| method | symbol (libkperf.dylib) |
|--------|------------------------|
| configure | kpc_set_config |
| start | kpc_set_counting(KPC_CLASS_FIXED \| KPC_CLASS_CONFIGURABLE) |
| read | kpc_get_thread_counters64 |
| stop | kpc_set_counting(0) |

### usage pattern

```rust
let mut pulse = Counters::new(&[Counter::Cycles, Counter::L1dMisses])?;
pulse.start();
let a = pulse.read();
// ... compute ...
let b = pulse.read();
let counts = pulse.elapsed(&a, &b);
println!("cycles: {}, L1 misses: {}", counts.cycles, counts.l1d_misses);
pulse.stop();
```

---

# kernels — high-level compute operations

safe, auto-dispatching operations. each kernel picks the fastest
available path based on caps:

```
AMX  →  NEON+extension  →  NEON scalar  →  fallback
```

dispatch is resolved at first call and cached.

## gemm

| function | signature | semantics |
|----------|-----------|-----------|
| matmul_f32 | `(a, b, c, m, n, k)` | C[m×n] += A[m×k] × B[k×n], fp32 |
| matmul_f16 | `(a, b, c, m, n, k)` | C[m×n] += A[m×k] × B[k×n], fp16 in, fp32 accum |
| matmul_bf16 | `(a, b, c, m, n, k)` | C[m×n] += A[m×k] × B[k×n], bf16 in, fp32 accum — M2+ |
| matmul_i8 | `(a, b, c, m, n, k, scale, zero)` | int8 quantized matmul → fp32 |

dispatch:
- matmul_f32: AMX fma32 → NEON fmla
- matmul_f16: AMX fma16 → NEON FP16 fmla
- matmul_bf16: AMX fmabf16 → NEON bfmmla → NEON bfdot
- matmul_i8: NEON I8MM smmla → NEON DotProd sdot → scalar

## math (elementwise, vectorized)

all operate in-place on fp32 slices. NEON vectorized, 4-wide
minimum, tail-masked.

| function | signature | semantics |
|----------|-----------|-----------|
| exp | `(&mut [f32])` | e^x, polynomial approximation |
| log | `(&mut [f32])` | ln(x) |
| tanh | `(&mut [f32])` | tanh(x) |
| sigmoid | `(&mut [f32])` | 1/(1+e^-x) |
| gelu | `(&mut [f32])` | 0.5x(1+tanh(√(2/π)(x+0.044715x³))) |
| silu | `(&mut [f32])` | x × sigmoid(x) |
| softmax | `(&mut [f32])` | exp(x)/Σexp(x) |
| normalize | `(out, x, weight, eps)` | x × weight / √(mean(x²)+ε) |
| rotate | `(out, x, freqs, pos)` | rotary positional embedding |

## convert (bulk, vectorized)

| function | signature | semantics |
|----------|-----------|-----------|
| cast_f16_f32 | `(&mut [f32], &[u16])` | fp16 → fp32, NEON 32/iter |
| cast_f32_f16 | `(&mut [u16], &[f32])` | fp32 → fp16, NEON 32/iter |
| cast_bf16_f32 | `(&mut [f32], &[u16])` | bf16 → fp32, shift |
| cast_f32_bf16 | `(&mut [u16], &[f32])` | fp32 → bf16, NEON bfcvt |
| cast_f32_i8 | `(&mut [i8], &[f32], scale)` | quantize fp32 → int8 |
| cast_i8_f32 | `(&mut [f32], &[i8], scale, zero)` | dequantize int8 → fp32 |

## reduce

| function | signature | semantics |
|----------|-----------|-----------|
| sum | `(&[f32]) -> f32` | NEON pairwise add |
| max | `(&[f32]) -> f32` | NEON fmaxnmv |
| min | `(&[f32]) -> f32` | NEON fminnmv |
| dot | `(&[f32], &[f32]) -> f32` | NEON fmla + reduce |
| length | `(&[f32]) -> f32` | √Σx² |

---

# errors

```
ChipNotSupported         not Apple Silicon
AmxSetFailed             AMX_SET instruction failed
AmxOpFailed(String)      AMX operation error
PmuNotAvailable          libkperf.dylib not found or kpc denied
PmuConfigFailed(String)  counter configuration rejected
FeatureNotAvailable(Feature)  required extension absent on this chip
AffinityFailed(String)   QoS class change failed
```

---

# execution model

- AMX is per-thread. each thread needs its own Matrix
- AMX set/clr are cheap (~1 cycle). open a context per GEMM call, not per thread lifetime
- NEON registers are callee-saved (v8–v15). inline asm must respect this
- all kernels are synchronous. no async dispatch model (this is CPU, not GPU)
- parallel GEMM: partition M dimension across threads, each thread gets own Matrix
- sync between threads: WFE/SEV + LSE atomics (no mutexes in hot path)
- memory: all buffers are caller-owned slices. acpu allocates nothing on heap
- all public functions are `#[inline]` or dispatch through cached function pointers

## driver stack

```
acpu crate
  → inline asm (.word for AMX, intrinsics for NEON)
  → sysctl (chip detection)
  → libkperf.dylib dlopen (PMU counters)
  → pthread (core affinity)
  → no frameworks, no ObjC, no C compiler
```

---

# module map

```
src/
  lib.rs              pub API re-exports, CpuError
  probe.rs            Chip, Features, Feature, scan()
  matrix/
    mod.rs            Matrix lifecycle (set/clr)
    ops.rs            load/store, fma32/fma16/fmabf16/mac16
    regs.rs           XRow, YRow, ZRow typed wrappers
    asm.rs            raw .word encoding macros
  vector/
    mod.rs            NEON kernel dispatch
    math.rs           exp, log, tanh, sigmoid, gelu, silu
    reduce.rs         sum, max, min, dot, length
    softmax.rs        softmax, normalize
    rope.rs           rotary positional embedding
  numeric/
    fp16.rs           FP16 arithmetic + bulk convert
    bf16.rs           BF16 ops + bulk convert
    quant.rs          DotProd, I8MM, quantize/dequantize
    complex.rs        FCMA complex mul-acc
  sync/
    mod.rs            barriers, wait/wake
    affinity.rs       pin_p_core, pin_e_core
    prefetch.rs       PRFM wrappers
  pulse/
    mod.rs            Counters, Counter, Snapshot
    ffi.rs            dlopen libkperf, kpc_* symbols
  gemm.rs             matmul_f32, matmul_f16, matmul_bf16, matmul_i8 (auto-dispatch)
  convert.rs          cast_* bulk conversion (re-exports from numeric)
  probe/
    main.rs           acpu_probe binary — exercise every organ
```

file limit: 500 lines per source file. split if exceeded.

---

# license

cyber license: don't trust. don't fear. don't beg.
