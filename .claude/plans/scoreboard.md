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
