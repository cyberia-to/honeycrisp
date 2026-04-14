# beat Apple Accelerate in every category

baseline: bench_all 2026-04-03, M1Pro 8P+2E

## current scoreboard

| category | operation | acpu | apple | ratio | status |
|----------|-----------|------|-------|-------|--------|
| elementwise | exp | 2583ns | 2333ns | 0.90× | LOSS |
| elementwise | log | 3084ns | 3083ns | 1.00× | TIE |
| elementwise | tanh | 3625ns | 3958ns | 1.09× | win |
| elementwise | sigmoid | 3125ns | 3750ns | 1.20× | win |
| elementwise | gelu | 5375ns | 6292ns | 1.17× | win |
| elementwise | silu | 3292ns | 4041ns | 1.23× | win |
| reduction | sum | 167ns | 167ns | 1.00× | TIE |
| reduction | dot | 292ns | 292ns | 1.00× | TIE |
| reduction | length | 375ns | 792ns | 2.11× | win |
| reduction | max | 166ns | 166ns | 1.00× | TIE |
| reduction | min | 166ns | 166ns | 1.00× | TIE |
| compound | softmax | 4333ns | 4208ns | 0.97× | TIE |
| compound | normalize | 1208ns | 1834ns | 1.52× | win |
| sgemm | 512×512 | 2123GF | 2068GF | 1.03× | TIE |
| sgemm | 4096×4096 | 1473GF | 1476GF | 1.00× | TIE |

losses: 1 (exp)
ties: 8 (log, sum, dot, max, min, softmax, sgemm-512, sgemm-4096)
wins: 6

## progress after sessions 1+3+5 (2026-04-03)

completed:
- exp: Estrin polynomial (depth-3), vrndnq+vfmsq, 5-term minimax.
  still 2583ns vs 2333ns (0.89×). LLVM ceiling — Apple uses hand-tuned asm.
- exp_to/log_to: out-of-place variants (match vvexpf API). benchmark fixed.
- all math: 16-wide unroll with interleaved x4 variants
- softmax: 4333ns → 3292ns (24% faster, now 0.96× = tie)
- normalize: 1208ns → 750ns (38% faster)
- complex_mul_acc: FCMLA vectorized, 2.3 → 3.3 Ge/s (43%)
- bench_full: Apple Accelerate comparison columns added

remaining:
- exp: only path is hand-written assembly (session 4 scope)
- reductions (sum/dot/max/min): at L1 bandwidth limit, cannot improve
- sgemm 512/4096: parallel B-packing, KC tuning (session 4)
- bf16 f32→bf16: runtime detection or NEON vectorized bit-manip
- RoPE: NEON sin/cos polynomial

## non-Apple benchmarks needing improvement

| operation | current | target | notes |
|-----------|---------|--------|-------|
| complex_mul_acc | 2.3 Ge/s (scalar) | 8+ Ge/s | FCMLA missing entirely |
| bf16 round-trip | 20.2 GB/s | 40+ GB/s | bfcvtn gated behind feature flag |
| RoPE 4096 | 6334ns | 3000ns | scalar sin/cos bottleneck |
| AMX utilization | 37% across board | 60%+ | current microbenchmark artifact? |
| f32→bf16 | 833ns (3.21× memcpy) | 400ns | scalar fallback if no bf16 ISA |

---

## plan: 6 sessions, priority order

### session 1: exp (the only loss) + log (tie → win)

**exp (target: <2100ns, from 2583ns)**
- current: Cody-Waite + 6-term Horner, 4-wide NEON
- problem: 4-wide is half the throughput it could be. Apple uses 8-wide vvexpf
- fix: unroll to 16-wide (4 accumulators × 4 f32), same polynomial
- secondary: evaluate Schraudolph fast-exp approximation for cases where ULP precision is relaxed
- verify: max ULP error vs libm stays within 1 ULP

**log (target: <2800ns, from 3084ns)**
- current: bit decomposition + 7-term Taylor, 4-wide
- fix: unroll to 16-wide, same as exp
- secondary: consider Cephes minimax polynomial (fewer terms, same accuracy)

estimated: 1 session (6 pomodoros)

### session 2: reductions (5 ties → 5 wins)

**sum, max, min (target: <140ns each, from 166-167ns)**
- current: 8-accumulator, 32-wide main loop — already excellent
- problem: at 4096 elements this is 128 iterations of 32-wide. already memory-bound
- analysis: 4096 × 4 bytes = 16KB. L1 bandwidth ~200 GB/s. theoretical floor = 16KB / 200 GB/s = 80ns
- fix 1: try 64-wide (16 accumulators) to saturate 2 NEON load pipes per cycle
- fix 2: add software prefetch for next cache line (PRFM PLDL1KEEP)
- verify: benchmark at multiple sizes to confirm not measurement noise

**dot (target: <250ns, from 292ns)**
- same approach: widen to 64-wide, prefetch
- theoretical: 2 × 16KB loads = 32KB, floor = 160ns. currently 1.8× floor — room exists

**length (already 2.11× win, skip)**

estimated: 1 session

### session 3: softmax (tie → win)

**softmax (target: <3500ns, from 4333ns)**
- current: 3-pass (find max, exp+sum, divide), 16-wide
- problem: 3 passes = 3× memory traffic over 4096 floats
- fix 1: fuse pass 1+2: online softmax (Milakov-Gimelshein). single pass computes running max + running sum of exp, second pass divides. 2 passes instead of 3
- fix 2: widen to 32-wide in fused pass
- fix 3: use the improved exp from session 1 inside softmax
- verify: numerical accuracy (online softmax can accumulate differently)

estimated: 1 session

### session 4: sgemm large (2 ties → 2 wins)

**sgemm 512 (target: 2300GF, from 2123GF)**
- current: 2123GF = 66% of 3200GF BW ceiling
- analysis: 512×512 = 1MB per matrix. B fits in L2. 4-thread territory
- fix 1: tune thread_cap for 512: currently uses 4 threads (sz 384-768). try 6
- fix 2: NC blocking — currently NC=512 for this size. try NC=256 to keep B-strip in L1
- fix 3: parallel B-packing (currently single-threaded) — at 512 this is ~1MB pack

**sgemm 4096 (target: 1550GF, from 1473GF)**
- current: 1473GF = 46% of 3200GF ceiling
- analysis: 4096×4096 = 64MB per matrix. heavily memory-bound
- bottleneck: B-packing is single-threaded (main thread packs all B, workers wait)
- fix 1: parallel B-packing across workers during barrier sync
- fix 2: KC tuning — currently KC=512 fixed. at 4096, try KC=256 to reduce TLB misses
- fix 3: double-buffer B-packs: pack next KC strip while computing current
- fix 4: explicit PRFM in gebp_kernel inner loop for next B-strip

estimated: 2 sessions

### session 5: complex_mul_acc + bf16 + RoPE

**complex_mul_acc (target: 8 Ge/s, from 2.3 Ge/s)**
- current: pure scalar loop. no NEON at all
- fix: implement FCMLA (ARMv8.3 FEAT_FCMA, available on all M1+)
  ```asm
  fcmla v_acc.4s, v_a.4s, v_b.4s, #0    // real part
  fcmla v_acc.4s, v_a.4s, v_b.4s, #90   // imaginary part
  ```
- 2 instructions per 2 complex pairs = 4× throughput over scalar
- unroll 4× for pipeline saturation

**bf16 f32→bf16 (target: <400ns, from 833ns)**
- current: gated behind compile-time `target_feature = "bf16"`
- fix: runtime detection via `mrs x0, ID_AA64ISAR1_EL1` (BF16 field bits[47:44])
- fallback: NEON bit-manipulation with round-to-nearest-even (shift + bias add)
  already exists as f32_to_bf16_rne — vectorize it with NEON (ushr + add + ushr)

**RoPE (target: 3000ns, from 6334ns)**
- current: scalar sin/cos per pair — the bottleneck
- fix 1: NEON polynomial sin/cos (same Cody-Waite + Horner as exp, different coefficients)
- fix 2: process 4 frequency pairs per iteration instead of 2
- fix 3: eliminate array→vld1q stack bounce — use vsetq_lane_f32 directly

estimated: 1 session

### session 6: verify + sweep

- rerun bench_all, confirm all categories are wins
- run bench_full (sgemm spectrum) to confirm no regressions
- clippy clean, fmt clean
- update specs/ if any API semantics changed

---

## verification dimensions per fix

| dimension | method |
|-----------|--------|
| correctness | existing tests + new edge-case tests per kernel |
| performance | bench_all before/after, 3 runs median |
| accuracy | max ULP error vs libm for math kernels |
| regression | full bench_all sweep after each session |

## priority if time-constrained

1. exp (only loss — reputation)
2. softmax (most impactful for inference workloads)
3. sgemm 4096 (parallel B-packing — architectural improvement)
4. complex_mul_acc (3× improvement, easy win)
5. reductions (small absolute gains but establishes dominance)
6. log, bf16, RoPE (polish)
