# scoreboard: acpu vs Apple Accelerate

baseline: M1 Pro (8P+2E), 2026-04-03

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
