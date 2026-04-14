# ultimate CPU benchmark design

reference-quality benchmark for Apple Silicon CPU compute.
every category must compare against standard tools or Apple frameworks.

## what we have vs what's missing

### HAVE (bench/ today)
- memory latency (pointer chase) — compares with known L1/L2/L3/DRAM values
- STREAM bandwidth (copy/scale/add/triad) — matches STREAM methodology
- prefetch impact
- P/E core comparison, cross-core latency
- elementwise f32 (exp/log/tanh/sigmoid/gelu/silu) vs Apple Accelerate
- reductions f32 (sum/dot/length/max/min) vs Apple Accelerate
- compound f32 (softmax/normalize) vs Apple Accelerate
- sgemm f32 spectrum vs Apple cblas_sgemm
- mixed-precision gemm (fp16/bf16/i8) — acpu only, no Apple comparison
- numeric conversions (f32↔f16/bf16/i8)
- complex multiply-accumulate
- RoPE
- AMX utilization (bandwidth + compute)

### MISSING — must add

#### 1. integer compute
- NEON integer throughput: i8/i16/i32/i64 add, mul, shift, bitwise
- scalar integer: div, mod (no SIMD, latency-bound)
- comparison: clang/gcc auto-vectorized loop as baseline
- i8mm: INT8 matrix multiply (FEAT_I8MM, M1+)
  - acpu has `cast_f32_i8` but no native i8 SIMD compute bench

#### 2. integer SIMD
- NEON i8×16, i16×8, i32×4 lane throughput
- saturating arithmetic (sqadd, uqadd)
- widening multiply (smull, umull)
- dot product (sdot/udot — FEAT_DotProd)
- comparison with Apple's vDSP integer functions

#### 3. crypto/hash extensions
- AES-NI equivalent: AESE/AESD/AESMC (FEAT_AES)
- SHA: SHA256H, SHA256SU0 (FEAT_SHA256)
- throughput in GB/s — compare with OpenSSL `openssl speed aes-128-cbc`

#### 4. branch prediction
- predictable loop (100% taken)
- random branch (50/50) — measure mispredict penalty
- indirect branch (vtable dispatch) — measure BTB

#### 5. instruction throughput / IPC
- scalar integer IPC (add chain)
- NEON IPC (independent FMA chain)
- mixed scalar+NEON IPC
- compare with theoretical peak (8-wide decode on M1 P-core)

#### 6. atomic operations
- atomic increment throughput (single core)
- atomic CAS throughput (single core)
- contended atomic (multi-core) — scaling curve
- LSE atomics (FEAT_LSE) vs LL/SC comparison

#### 7. memory system detail
- cache line size verification (stride test)
- TLB reach and miss penalty
- false sharing penalty (adjacent cache line writes from 2 cores)
- memory-level parallelism (how many outstanding loads)

#### 8. syscall / OS overhead
- mach_absolute_time() call overhead
- thread_switch() overhead
- mmap/munmap cycle
- context switch latency

#### 9. standard comparisons
current comparisons:
- Apple Accelerate (cblas_sgemm, vDSP_*, vvexpf)

should add:
- OpenBLAS sgemm (if installed via brew)
- Rust nalgebra/ndarray sgemm (pure Rust baseline)
- memcpy from libc (already have as baseline)
- OpenSSL speed (for crypto)
- system `sysctl` values for reference

#### 10. hang detection (from test_hang.rs)
- every bench file gets a timeout watchdog thread
- common.rs should provide `pub fn watchdog(seconds: u64)`
- kills process with diagnostic if benchmark hangs
- already in bench_all.rs pattern, needs to be in common.rs

## file plan

new bench files (all under 300 lines):

| file | category | comparison |
|------|----------|------------|
| bench/integer.rs | i8/i16/i32/i64 NEON + scalar | clang baseline |
| bench/crypto.rs | AES/SHA throughput | OpenSSL speed |
| bench/branch.rs | branch prediction, IPC | theoretical peak |
| bench/atomics.rs | atomic ops, LSE, contention | single vs multi |
| bench/tlb.rs | TLB, false sharing, MLP | hardware specs |
| bench/syscall.rs | timer, mmap, ctx switch | lmbench |

update existing:
| file | change |
|------|--------|
| bench/common.rs | add watchdog(), standardize timeout |
| bench/sgemm.rs | add OpenBLAS comparison if available |

## priority

1. integer.rs — biggest gap, most visible missing category
2. common.rs watchdog — safety, prevents CI hangs
3. crypto.rs — easy to implement, impressive numbers
4. branch.rs + IPC — reveals microarchitecture
5. atomics.rs — important for concurrent workloads
6. tlb.rs + syscall.rs — advanced, lower priority

## comparison standards

| our bench | standard tool | what to compare |
|-----------|--------------|-----------------|
| STREAM bandwidth | STREAM benchmark | GB/s per kernel |
| sgemm GFLOPS | LINPACK / HPL | peak GFLOPS |
| memory latency | lmbench lat_mem_rd | ns per level |
| crypto GB/s | openssl speed | AES/SHA throughput |
| IPC | Geekbench single-core | instructions/cycle |
| atomic throughput | lmbench lat_sig | ops/sec |
