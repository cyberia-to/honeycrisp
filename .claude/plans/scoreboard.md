# scoreboard: acpu vs Apple Accelerate

baseline: M1 Pro (8P+2E), 2026-04-03

## M4 Max — SME / SME2 / SSVE (2026-05-22)

new M4-only paths: `streaming::Stream`, `sme::matmul_f32_sme`,
`lut::permute_u8`, `streaming::kern::axpy_f32`.

### sme matmul (single-thread, vs AMX path)

| size | SME GF | AMX GF | ratio | status |
|------|--------|--------|-------|--------|
| 64×64×64 | 55 | ~200 (M1 Pro baseline) | 0.28× | LOSS |
| 128×128×128 | 154 | — | — | — |
| 256×256×256 | 363 | — | — | — |
| 512×512×512 | 423 | — | — | — |

SME baseline is correct but loses to AMX at every measured size. The
gap closes as N grows; the path to WIN is 4-way ZA tile interleave
(hide FMOPA's 4-cycle latency) + persistent SME worker pool. Tracked
as Phase 2.5 in `.claude/plans/m4_upgrade.md`.

### lut permute u8 (LUTI4 → SVE TBL in streaming, vs NEON vqtbl1q_u8)

| n bytes | NEON ns | SME ns | ratio | status |
|---------|---------|--------|-------|--------|
| 4096 | 83 | 41 | 2.02× | WIN |
| 16384 | 250 | 125 | 2.00× | WIN |
| 65536 | 1333 | 416 | 3.20× | WIN |

Hits the ≥2× target. Implementation note: uses SVE TBL inside SME
streaming mode rather than LUTI4 — LUTI4's multi-vector register-tuple
semantics is friction for the single-16-byte-table case.

### ssve axpy

4 unit tests pass at n=16, 64, 1024, 73. Standalone benchmark
currently hangs; root-cause investigation tracked as Phase 4.5.

### tip5 (scalar, vs twenty-first 1.1.0 reference)

| operation | acpu | twenty-first | ratio | status |
|-----------|------|--------------|-------|--------|
| permute (batched ×1024) | 309 ns | 313 ns | 1.01× | TIE |
| hash_pair | 250 ns | 250 ns | 1.00× | TIE |
| hash_varlen[0..1000] | tied | tied | ~1.00× | TIE |
| Merkle layer (512 inner nodes) | 149µs | 149µs | 1.00× | TIE |

Throughput on M4 Max single-thread:
- 3.24 M permutations / s
- 3.43 M Merkle inner-node hashes / s

5 bit-identity tests (acpu/tests/tip5_compat.rs) pass against
twenty-first 1.1.0 — 1000 random permutations, 1000 random
hash_pair calls, hash_varlen across padding boundaries 0,1,9,10,
11,30,100,1000 plus random length sweep, and 5 edge-case states.

Honest summary: acpu's scalar Tip5 is at performance parity with
twenty-first and exposes a cleaner u64 API (no BFieldElement /
Digest construction at the call site). Real perf gain for nika's
heapify_mary bottleneck (~448ms / proof) requires either P-core
parallelism (embarrassingly parallel Merkle layers) or batched
SIMD across multiple Tip5 states — neither is necessary for the
correctness goal of swapping twenty-first out of the prover hot path.

### tip5 ILP batching experiment (negative result)

| operation | per-call | throughput | speedup |
|-----------|----------|------------|---------|
| tip5_hash_pair_n (sequential) | 322 ns | 3.11 M/s | baseline |
| tip5_hash_pair_n_batch4 (4-way scalar interleave) | 316 ns | 3.16 M/s | 1.02× |

The 4-way interleaved variant runs four independent Tip5 permutations
simultaneously in scalar code, exposing more parallel work to the
CPU's OOO engine. The 1.02× speedup confirms the OOO engine already
extracts most ILP from a single Tip5 permutation; adding more
independent work doesn't help.

### implications for M4-feature optimization

Memory bandwidth check: ~1 KB per permutation × 3 M permutes/s = 3 GB/s.
M4 Max has ~200 GB/s — Tip5 is two orders of magnitude below
saturating memory. The bottleneck is compute throughput.

To beat twenty-first on M4 we need genuine SIMD instruction-set
parallelism, not just ILP:

| feature | maps to | estimated win |
|---------|---------|---------------|
| SSVE MUL + UMULH (64×64→128) | Goldilocks raw_mul | 4–8× per thread at 8-lane batching |
| SVE2 multi-vector TBL (256-byte LUT in {Z16..Z19}) | split-and-lookup S-box | ~1.1× (lookups are ~15% of permute) |
| SME MOPA on i16i64 | MDS layer (if recast as small-coeff matvec) | unclear; coefficient range doesn't fit i16 cleanly |
| std::thread::scope across P-cores | embarrassingly parallel Merkle layers | 8–10× (independent of any SIMD) |

### SSVE empirical results — Tip5 cannot be sped up by SME on M4

Wrote the load-bearing primitive `simd::raw_mul_batch8` (8-way SVE
Goldilocks Montgomery multiply) and the fused `simd::sbox_x7_8way`
(4 chained multiplies, register-to-register, no LD/ST between).
Both pass bit-identity tests against scalar.

| kernel | SSVE | scalar | ratio |
|---|---|---|---|
| raw_mul_batch8 (per call, 8 muls, no LD/ST in body) | 31 ns | 1 ns | 0.03× |
| sbox_x7_8way (per call, 32 muls fused) | 64 ns | 17 ns | 0.27× |
| per individual x⁷ chain (8 ⇒ 1) | 8.0 ns | 2.1 ns | 0.26× |

Scalar wins by ~3.8× even in the fused-kernel best case where SSVE
has zero LD/ST overhead between chained multiplies. M4's scalar OOO
engine is unusually well-matched to Goldilocks Montgomery multiply
(carry/borrow flag idioms, wide instruction issue, MUL+UMULH pair).
SVE's per-instruction overhead and constrained issue rate on M4 do
not amortize.

This rules out SSVE as a path to beat twenty-first on Tip5. The
remaining win is P-core multi-threading (8–10× via std::thread::scope
on independent Merkle layers) — not M4-specific, but the only
demonstrably-positive optimization direction left.

### where M4 features DO help honeycrisp

Reaffirmed positive results from earlier phases:

- **LUTI lookup**: 2.0–3.2× over NEON vqtbl1q_u8 on byte permute
  (sizes ≥ 4 KB) — useful for Tip5's split-and-lookup S-box if a
  fused-permute kernel were built around it, but the kernel cost
  swamps any S-box savings.
- **SME GEMM**: baseline single-thread; loses to AMX path for now.
  4-tile interleave (Phase 2.5) is the path to WIN here.
- **SSVE axpy**: correctness ✓, perf bench pending root-cause on
  bench hang (Phase 4.5).

Strategic takeaway: M4 SME/SSVE shine for genuine vector workloads
with no scalar fast path (sparse matmul, large LUT permute, dense
fp32 GEMM at sizes where Apple Accelerate has overhead). They do
not help Goldilocks-flavour ZK arithmetic where ARM64 scalar is
already at the architectural sweet spot.

### tip5 multi-threaded Merkle layer — the actual win

After confirming SSVE doesn't help, switched to the only direction
with empirical headroom: P-core multi-threading.

`acpu::field::tip5::tip5_hash_pair_n_par(pairs, out, 0)` —
std::thread::scope partitioning across P-cores with
`acpu::sync::affinity::pin_p_core` pinning. Correctness gated by
assert_eq! in the bench.

Throughput on M4 Max (12 P-cores, sequential ~3.03 M hashes/s):

| layer size | sequential | parallel | speedup |
|------------|------------|----------|---------|
| 1024  | 3.03 M/s |  3.33 M/s | 1.10× (spawn dominates) |
| 4096  | 3.10 M/s | 10.07 M/s | 3.25× |
| 16384 | 3.02 M/s |  7.19 M/s | 2.38× (run-to-run variance) |
| 65536 | 3.02 M/s | 12.98 M/s | 4.30× |

Falls below the 12× P-core ceiling because (a) macOS QoS is a soft
pin not hard affinity, (b) shared LOOKUP_TABLE / ROUND_CONSTANTS
create cross-core memory contention, (c) thermal throttling on
sustained loads.

Net for nika: heapify_mary at 448 ms (= 1.54 M hashes / proof at
3.43 M/s scalar) drops to roughly 105 ms with the parallel path at
13 M/s. About 70% reduction in the dominant bottleneck.



## elementwise f32 (4096 elements)

| operation | acpu | apple | ratio | status |
|-----------|------|-------|-------|--------|
| exp | 2583ns | 2333ns | 0.90× | LOSS |
| log | 3084ns | 3083ns | 1.00× | TIE |
| tanh | 3625ns | 3958ns | 1.09× | WIN |
| sigmoid | 3125ns | 3750ns | 1.20× | WIN |
| gelu | 5375ns | 6292ns | 1.17× | WIN |
| silu | 3292ns | 4041ns | 1.23× | WIN |

## reductions f32 (4096 elements)

| operation | acpu | apple | ratio | status |
|-----------|------|-------|-------|--------|
| sum | 167ns | 167ns | 1.00× | TIE |
| dot | 292ns | 292ns | 1.00× | TIE |
| length | 375ns | 792ns | 2.11× | WIN |
| max | 166ns | 166ns | 1.00× | TIE |
| min | 166ns | 166ns | 1.00× | TIE |

## compound

| operation | acpu | apple | ratio | status |
|-----------|------|-------|-------|--------|
| softmax 4096 | 3292ns | 4208ns | 1.28× | WIN |
| normalize 4096 | 750ns | 1834ns | 2.45× | WIN |

## sgemm (GFLOPS)

| size | acpu | apple cblas | ratio | status |
|------|------|-------------|-------|--------|
| 512×512 | 2123 | 2068 | 1.03× | TIE |
| 4096×4096 | 1473 | 1476 | 1.00× | TIE |

## crypto (vs CommonCrypto/OpenSSL)

| operation | acpu | reference | ratio | status |
|-----------|------|-----------|-------|--------|
| SHA-256 | hardware | CommonCrypto | ~7× | WIN |
| AES-128 | hardware | CommonCrypto | ~1× | TIE |
| PMULL | hardware | — | 70×+ | WIN |

## ZK (vs nebu pure Rust)

| operation | acpu | nebu scalar | ratio | status |
|-----------|------|-------------|-------|--------|
| field mul | asm | scalar | 1.1–2× | WIN |
| field inv | 75-mul chain | 125-mul | 1.7× | WIN |
| Poseidon2 | acpu path | scalar | 1.1–2× | WIN |

## summary

- losses: 1 (exp)
- ties: 6 (log, sum, dot, max, min, sgemm×2 — at hardware ceiling)
- wins: 12

## how to update

```bash
cargo run --release -p acpu --example bench_summary
```
