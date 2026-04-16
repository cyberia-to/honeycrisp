# optimization roadmap

what to build next, prioritized by impact on the three target workloads:
LLM inference, zero-knowledge proving, real-time rendering.

## tier 1 — high impact, unblocks downstream

| # | what | crate | workload | sessions | why |
|---|------|-------|----------|----------|-----|
| 1 | exp asm (the only Accelerate loss) | acpu | all | 1 | 0.90× — only category where Apple wins. requires hand asm, LLVM cannot emit optimal code |
| 2 | NTT butterfly batch asm | nebu+acpu | ZK | 1 | STARK proving bottleneck. current NTT is scalar. interleaved butterfly pairs in asm |
| 3 | i8 GEMM native SDOT path | acpu | inference | 2 | current i8 matmul dequantizes to f32. native SDOT accumulation is 4× throughput |
| 4 | sgemm parallel B-packing | acpu | inference | 1 | 4096×4096 at 46% of ceiling. single-threaded B-pack is the bottleneck |
| 5 | fused attention kernel | acpu | inference | 2 | Q×K^T → softmax → ×V in one pass. eliminates intermediate materialization |

## tier 2 — meaningful gains, moderate effort

| # | what | crate | workload | sessions | why |
|---|------|-------|----------|----------|-----|
| 6 | i4 dequant (GGUF Q4_0/Q4_1) | acpu | inference | 1 | llama.cpp format. needed for quantized model loading |
| 7 | Poseidon2 full asm permutation | acpu+hemera | ZK | 1 | entire state (8×u64) fits in registers. no memory traffic |
| 8 | field mul interleaved asm (4-chain) | nebu | ZK | 1 | 4 independent mul+umulh chains hiding 4-cycle latency. 2ns/mul vs 5ns |
| 9 | Keccak-256 | acpu | crypto | 1 | Ethereum hash. bitwise+rotate, no special instructions |
| 10 | RoPE NEON sin/cos polynomial | acpu | inference | 0.5 | current: scalar sin/cos. 6334ns → ~3000ns with NEON polynomial |
| 11 | bf16 runtime detection + NEON fallback | acpu | inference | 0.5 | f32→bf16 at 833ns (3.2× memcpy). NEON bit-manip without FEAT_BF16 |

## tier 3 — polish, completeness

| # | what | crate | workload | sessions | why |
|---|------|-------|----------|----------|-----|
| 12 | alpha blend u8 | acpu | media | 0.5 | only missing media op |
| 13 | gather/scatter | acpu | inference | 1 | MoE routing, sparse attention |
| 14 | secp256k1 mul | acpu | crypto | 2 | Ethereum signatures. 256-bit modular arithmetic |
| 15 | group quant helpers | acpu | inference | 0.5 | per-group scale extraction for quantized models |
| 16 | sgemm KC tuning for 4096 | acpu | inference | 0.5 | KC=256 to reduce TLB misses at large sizes |

## tier 4 — microarch exploration (low priority)

| # | what | sessions | why |
|---|------|----------|-----|
| 17 | branch prediction bench | 0.5 | mispredict penalty, BTB behavior |
| 18 | IPC measurement (scalar, NEON, mixed) | 0.5 | verify against 8-wide decode |
| 19 | atomic contention scaling | 0.5 | LSE vs LL/SC, multi-core curves |
| 20 | TLB reach + false sharing | 0.5 | memory system characterization |
| 21 | syscall overhead (mach_absolute_time, mmap) | 0.5 | OS overhead baseline |

## what's done (shipped in v0.2.0)

already implemented — do not re-plan:

- inv addition chain: 75-mul (was 125). shipped in nebu
- SHA-256, AES-128, PMULL: acpu/src/crypto/
- rsqrt, recip, clamp, lerp, cross3: acpu/src/vector/render.rs
- RGB↔YUV, histogram, resize: acpu/src/vector/media.rs
- integer ops (sum_i32, max_i32, dot_i8, sad_u8, absmax_i8): acpu/src/vector/integer.rs
- integer fused (sad_u8, ssd_i32, scale_acc_i16, sum_abs_i8): acpu/src/vector/integer_fused.rs
- Goldilocks field (gl_mul, gl_inv, gl_pow7, batch_inv, poseidon2_permute, merkle_root): acpu/src/field/
- RoPE: acpu/src/vector/rope.rs (NEON, but sin/cos still scalar)
- softmax: 2-pass (not 3-pass). online max+exp fused
- GEMM: f32 (AMX), f16/bf16/i8 (convert→f32 path)
- prefix_sum, transpose: acpu/src/vector/scan.rs
- complex multiply: FCMLA vectorized, 3.3 Ge/s
- 11 benchmark modules covering all categories

## session estimate

| tier | sessions | what you get |
|------|----------|-------------|
| 1 | 7 | zero Accelerate losses, native i8 GEMM, fused attention, NTT asm |
| 1+2 | 13 | + Poseidon2 full asm, Keccak, fast RoPE/bf16, field mul asm |
| 1+2+3 | 18 | + alpha blend, gather/scatter, secp256k1, quant helpers |
| all | 21 | + microarch exploration suite |
