# benchmark design: workload-driven analysis

five target workloads. for each: what CPU operations matter,
what the driver should expose, and what the bench should measure.

## 0. ZK proving (HIGHEST PRIORITY)

the foundational workload. every proof in the cyber stack runs through
nebu (Goldilocks field) → hemera (Poseidon2 hash) → zheng (STARK prover).
optimizing field arithmetic has multiplicative impact on everything above.

### the Goldilocks field

p = 2^64 - 2^32 + 1. every element is one u64. the reduction identity
`2^64 ≡ 2^32 - 1 (mod p)` makes modular arithmetic cheap.

### hemera Poseidon2 structure (t=16, Goldilocks)

```
permutation(state[16]):
  initial MDS
  4 full rounds:     add_rc(16) + x^7 all 16 + external_mds
  16 partial rounds: add_rc(1)  + x^(-1) on state[0] + internal_mds
  4 full rounds:     add_rc(16) + x^7 all 16 + external_mds
```

per permutation: ~2768 field multiplications + ~500 additions.
x^(-1) inversion is THE bottleneck (72% of muls).

### critical operations — what acpu must accelerate

| operation | calls/perm | current (hemera) | target |
|-----------|-----------|------------------|--------|
| field mul (a×b mod p) | ~2768 | scalar Rust, ~5ns | asm interleaved, <1.5ns |
| inv (x^(-1), partial S-box) | 16 | ~125 muls each | addition chain, ~70 muls |
| sbox7 (x^7, full rounds) | 8×16 = 128 | 4 muls each | interleaved x16 |
| external MDS (circ 4×4) | 9 | adds+doubles | already optimal |
| internal MDS (diag+sum) | 16 | 16 field muls | asm interleaved |
| field add/sub | ~500 | scalar, ~1ns | already fast |
| NTT butterfly | N/2×log2N | scalar (nebu) | asm batched |
| batch inverse | 3 muls/elem | scalar (nebu) | pipelined |

### hemera parameters (from hemera/rs/src/permutation.rs)

```
field:  Goldilocks p = 2^64 - 2^32 + 1
width:  t = 16 (rate=8, capacity=8)
rounds: 4 full + 16 partial + 4 full = 24 total
full S-box:    x^7 (4 muls, applied to all 16 elements)
partial S-box: x^(-1) (field inversion, ~125 muls, applied to state[0] only)
external MDS:  circ-of-4×4 (adds+doubles only, no field muls)
internal MDS:  diag(d) + ones (16 field muls + 16 adds)
```

### exact operation count per permutation

source: hemera/rs/src/permutation.rs + field.rs

```
full round (×8):    16 add_rc + 16 sbox7(=64 mul) + MDS(=adds only)
partial round (×16): 1 add_rc + 1 inv(=~125 mul) + internal(=16 mul + 16 add)
```

| component | per round | ×rounds | total muls | total adds |
|-----------|-----------|---------|------------|------------|
| sbox7 (full) | 16×4=64 | ×8 | 512 | - |
| inv (partial) | ~125 | ×16 | **2000** | - |
| internal_linear mul | 16 | ×16 | 256 | 256 |
| external MDS | - | ×9 | 0 | ~100 |
| add_rc (full) | - | ×8 | - | 128 |
| add_rc (partial) | - | ×16 | - | 16 |
| **TOTAL** | | | **~2768 muls** | **~500 adds** |

**the inversion dominates.** 16 partial rounds × ~125 muls = 2000 muls.
full-round S-boxes: 512 muls. internal linear: 256 muls.

at 5ns/mul (current scalar): **~13,840ns per permutation = 72K hashes/sec.**

### optimization targets

**1. field inversion (~125 muls → ~70 muls)**
hemera's inv() uses naive square-and-multiply scanning all 63 bits.
p-2 = 0xFFFFFFFEFFFFFFFF has Hamming weight 63 (all bits except bit 32).
optimized addition chain for Goldilocks p-2 can reduce to ~70 muls
using intermediate powers: x^(2^k - 1) chains.

saves: 16 × 55 = 880 muls per permutation → **~1888 total.**

**2. interleaved field mul asm (5ns → 1.5ns)**
hand-scheduled mul+umulh with 4+ independent chains.
M1 P-core: mul is 1-cycle throughput, 4-cycle latency.

1888 muls × 1.5ns = **2832ns per permutation = 353K hashes/sec.**

**3. inv_goldilocks addition chain (partial S-box, 72% of total cost)**
the 16 partial-round inversions are serial (each depends on previous
state through matmul_internal). cannot parallelize across rounds.
CAN optimize the addition chain within each inversion (125→70 muls).
this is the single highest-impact optimization: saves 880 muls/perm.

**4. sbox7_x16 in full rounds (minor, 19% of total)**
16 independent x^7 chains × 4 muls each = 64 muls per full round.
with 16 independent chains hiding 4-cycle latency:
each of 4 steps takes ~4 cycles = 16 cycles for all 16 S-boxes.
8 full rounds × 64 = 512 muls. less critical than inv optimization.

**5. internal matmul with asm**
16 field muls (diag × state) + sum — all 16 muls are independent.
with interleaved asm: 16 muls in ~8 cycles.

### revised operation budget with all optimizations

| component | current muls | optimized muls | time at 1.5ns/mul |
|-----------|-------------|----------------|-------------------|
| sbox7 ×8 | 512 | 512 | 768ns |
| inv ×16 | 2000 | 1120 (70/inv) | 1680ns |
| internal ×16 | 256 | 256 | 384ns |
| adds | - | - | ~150ns |
| **TOTAL** | **2768** | **1888** | **~2982ns** |

realistic target: **~3000ns per permutation = 333K hashes/sec.**
aggressive target with full asm: **~2000ns = 500K hashes/sec.**

### what to optimize in nebu (not acpu)

field arithmetic is nebu's domain. the NEON/asm kernels live in nebu.
acpu provides the benchmark harness via bench/zk.rs.

**optimization 1: reduce128 in assembly**

current: Rust u128 multiply compiles to `mul` + `umulh` + 6 scalar ops.
problem: LLVM doesn't interleave independent multiplies.

hand-written asm for 4 field muls interleaved:
```asm
// 4 independent a×b pairs in x0-x7 (a0,b0,a1,b1,a2,b2,a3,b3)
mul   x8,  x0, x1    // lo0     cycle 0
mul   x10, x2, x3    // lo1     cycle 1
mul   x12, x4, x5    // lo2     cycle 2
mul   x14, x6, x7    // lo3     cycle 3
umulh x9,  x0, x1    // hi0     cycle 3 (latency hidden by 3 other muls)
umulh x11, x2, x3    // hi1
umulh x13, x4, x5    // hi2
umulh x15, x6, x7    // hi3
// now reduce hi:lo → Goldilocks for all 4
// ~4 ops per reduction × 4 = 16 ops, mostly parallel
```

M1 P-core: `mul` is 1-cycle throughput, 4-cycle latency.
4 independent chains → **4 field muls in ~8 cycles = 2ns each.**
with 8 chains (full S-box layer): **8 field muls in ~12 cycles.**

**optimization 2: inv_goldilocks optimized addition chain**

current inv() scans all 63 bits of p-2 = 0xFFFFFFFEFFFFFFFF.
62 squarings + 62 multiplications (all bits set except bit 32) = ~124 muls.
optimized addition chain using Goldilocks structure:
  x^(2^k - 1) chains reduce to ~70 muls.
16 inversions × 55 saved = **880 fewer muls per permutation.**
this is THE critical optimization — inv is 72% of total cost.

**optimization 3: external_linear as pure adds**

circ(2,1,...,1): `new[i] = state[i] + sum`.
sum 8 elements (7 adds) + 8 adds to broadcast = 15 adds.
at ~0.3ns per add: **~5ns total, dominated by dependency chain.**

**optimization 4: internal_linear with shift-muls**

diag = [2, 3, 5, 9, 17, 33, 65, 129] = [1+2^k for k=0..7].
`d_i * x_i = x_i + x_i << k` — replace mul with add+shift!
8 additions instead of 8 field muls. saves 176 muls per permutation.

revised total: **520 - 176 = 344 muls + ~500 adds.**
at 2ns/mul: **688ns per permutation = 1.45M hashes/sec.**

**optimization 5: NTT with cached twiddles**

current nebu NTT recomputes ω_m each stage (expensive exp).
precompute_twiddles() exists but isn't used in hot path.
fix: use precomputed table in ntt/intt.

**optimization 6: complete poseidon2 in assembly**

the entire state (8×u64 = 64 bytes) fits in 8 scalar registers.
round constants are 86 values loaded from memory.
the permutation can run entirely in registers with no memory traffic.

### where the kernels live

acpu: hardware-specific asm kernels (goldilocks_mul_batch, inv_goldilocks,
      reduce128_asm, NTT butterfly batch). acpu is the hardware driver —
      it owns architecture-specific optimizations (NEON, AMX, inline asm).
nebu: portable Goldilocks abstraction. calls acpu for aarch64 fast paths,
      falls back to scalar Rust on other platforms.
hemera: Poseidon2 permutation. calls nebu field ops (which call acpu).
acpu bench/zk.rs: benchmark harness measuring the full stack.

### bench additions for ZK

| bench | metric | comparison |
|-------|--------|------------|
| field_mul throughput | Gop/s, ns/op | plonky3 Goldilocks |
| inv throughput | ns/inv | plonky3 |
| poseidon2 permutation | ns/perm, hashes/s | plonky3, circom |
| NTT (2^16, 2^20, 2^24) | ms, butterflies/s | plonky3 |
| batch inverse (1K, 64K) | ns/element | plonky3 |
| full hash (56 bytes) | ns/hash, MB/s | SHA-256, BLAKE3 |

### architecture note

nebu is the field library. acpu is the hardware driver.
the question: where do the NEON/asm kernels live?

option A: acpu exposes `goldilocks_mul_batch`, nebu calls acpu.
option B: nebu gets its own NEON/asm in rs/arch/aarch64.rs.

answer: **option B**. field arithmetic is nebu's domain.
acpu provides general-purpose NEON primitives (exp, dot, etc.).
nebu provides Goldilocks-specific NEON/asm kernels.
BUT: acpu's bench suite should include Goldilocks benchmarks
as a representative ZK workload, calling nebu directly.

## 1. AI / inference

CPU compute in an integrated pipeline: CPU handles attention/FFN
while ANE runs other layers and GPU runs decode. CPU is NOT a fallback —
it is a parallel worker in the heterogeneous pipeline. acpu must be
fast enough that CPU-assigned layers don't bottleneck the pipeline.

### what we HAVE
- sgemm f32 (AMX), matmul fp16/bf16/i8 (convert path)
- dot_i8 (SDOT), sad_u8, absmax_i8, scale_acc_i16
- softmax, RMSnorm, RoPE, exp/gelu/silu

### what's MISSING
| operation | why | effort |
|-----------|-----|--------|
| i8 GEMM via SDOT | 4× faster quantized matmul | 2 sessions |
| i4 dequant | GGUF Q4_0/Q4_1 format (llama) | 1 session |
| group quant helpers | per-group scale extraction | 0.5 session |
| gather/scatter | MoE routing, sparse attention | 1 session |
| fused attention kernel | Q×K^T → softmax → ×V in one pass | 2 sessions |

### bench: bench/ai.rs
- i8 GEMM throughput vs f32 GEMM vs Apple Accelerate
- quantize + matmul pipeline overhead
- KV-cache append bandwidth
- fused attention latency (if implemented)

## 2. blockchain / crypto

three chains matter: Bitcoin (SHA-256), Ethereum (Keccak-256, secp256k1),
Monero (RandomX — full CPU workload including AES, integer, branch, memory).
plus the cyber stack itself (Poseidon2 over Goldilocks, covered in ZK section).

### what's MISSING
| operation | chain | instruction | effort |
|-----------|-------|------------|--------|
| SHA-256 round | Bitcoin | SHA256H/SHA256SU0 | 0.5 session |
| Keccak-256 (SHA-3) | Ethereum | bitwise + rotate | 1 session |
| secp256k1 mul | Ethereum | 256-bit modular arith | 2 sessions |
| AES round | Monero/RandomX | AESE/AESD/AESMC | 0.5 session |
| SHA-512 round | ed25519 | SHA512H/SHA512SU0 | 0.5 session |
| PMULL (GCM/CRC) | TLS, storage | PMULL/PMULL2 | 0.5 session |
| popcount | various | CNT | trivial |
| constant-time compare | signatures | NEON vceqq | trivial |
| random memory access | Monero/RandomX | cache-latency bound | 0.5 session |

### bench: bench/crypto.rs
- SHA-256 throughput (GB/s) vs `openssl speed sha256`
- Keccak-256 throughput vs keccak crate
- AES single-block throughput vs `openssl speed aes-128-ecb`
- PMULL throughput
- full-membership proof benchmark (Poseidon2 Merkle path)

## 3. rendering

### what's MISSING
| operation | effort |
|-----------|--------|
| rsqrt (FRSQRTE+Newton) | trivial |
| reciprocal (FRECPE+Newton) | trivial |
| clamp, lerp | trivial |
| 4×4 matmul (NEON) | 0.5 session |
| cross product | trivial |
| transpose 4×4 | trivial |

### bench: bench/render.rs
- 4×4 matmul throughput
- rsqrt accuracy+throughput
- lerp/clamp throughput

## 4. media

### what's MISSING
| operation | effort |
|-----------|--------|
| alpha blend u8 | 0.5 session |
| table lookup (TBL) | trivial |
| FIR filter | 0.5 session |

### bench: bench/media.rs
- alpha blend Mpixels/s
- FIR throughput samples/s

---

## implementation priority

| # | what | where | workload | sessions | impact |
|---|------|-------|----------|----------|--------|
| 1 | Goldilocks field mul asm | acpu | ZK | 2 | HIGHEST — 2768 muls/perm |
| 2 | inv addition chain (125→70 muls) | acpu | ZK | 1 | saves 880 muls/perm |
| 3 | inv_goldilocks addition chain | acpu | ZK | 1 | 72% of permutation cost |
| 4 | Poseidon2 permute asm | acpu+hemera | ZK | 1 | full pipeline in registers |
| 5 | NTT butterfly batch | acpu+nebu | ZK | 1 | STARK proving |
| 6 | bench/zk.rs | acpu bench | ZK | 0.5 | measure all above |
| 7 | SHA-256/Keccak-256 | acpu | blockchain | 1 | Bitcoin+Ethereum |
| 8 | AES round | acpu | blockchain | 0.5 | Monero/RandomX |
| 9 | i8 GEMM via SDOT | acpu | AI | 2 | quantized pipeline |
| 10 | i4 dequant | acpu | AI | 1 | GGUF Q4 format |
| 11 | rsqrt/clamp/lerp/4×4 | acpu | rendering | 0.5 | f32 toolkit |
| 12 | alpha blend/TBL/FIR | acpu | media | 1 | media toolkit |

## bench file plan

| file | workload | location |
|------|----------|----------|
| bench/zk.rs | Goldilocks mul, inv, Poseidon2 permute, NTT | acpu (calls nebu+hemera) |
| bench/ai.rs | i8 GEMM, quant pipeline | acpu |
| bench/crypto.rs | AES, SHA-256, PMULL | acpu |
| bench/render.rs | 4×4 matmul, rsqrt, lerp | acpu |
| bench/media.rs | alpha blend, FIR, TBL | acpu |
