# sme — Scalable Matrix Extension (M4+)

source-of-truth specification for the `streaming`, `sme`, and `lut` organs
of acpu. These three organs all live behind the FEAT_SME gate and share
the same physical matrix coprocessor block as the legacy AMX unit. Per
thread, AMX and SME are mutually exclusive — only one `Matrix` or
`Stream` should be live at a time.

## concepts

| concept | what it is |
|---------|-----------|
| Stream | live streaming-mode context. Owns PSTATE.SM and PSTATE.ZA |
| SVL | streaming vector length in bits. Apple M4 family: 512 (64 bytes) |
| Z | SVE register file (Z0–Z31), each SVL bits wide, only addressable in streaming mode |
| P | SVE predicate register file (P0–P15), each SVL/8 bits, gates each lane of a Z reg |
| ZA | the matrix accumulator. SVL×SVL bits total. Sliced by element type into "tiles" |
| ZA tile | a SVL-wide × SVL-wide / type-bits-per-element slice (e.g. ZA0.S … ZA3.S for f32) |
| MOPA | the outer-product family that writes Z⊗Z into a ZA tile |
| MOVA | the move family that copies Z↔ZA in horizontal or vertical slices |
| ZT0 | SME2 lookup-table register, SVL bits wide, holds a LUTI table |
| LUTI2 | 2-bit-indexed lookup into ZT0, fills a Z register |
| LUTI4 | 4-bit-indexed lookup into ZT0, fills a Z register |

## chip support matrix

| chip | FEAT_SME | FEAT_SME2 | F64F64 | I16I64 | SVL |
|------|----------|-----------|--------|--------|-----|
| M1, M2, M3 (all variants) | no | no | no | no | — |
| M4, M4 Pro, M4 Max | yes | yes | yes | yes | 64 B |

All paths in `streaming`, `sme`, and `lut` MUST be gated by
`probe::scan().has_sme` before construction. Constructing a `Stream` on
a chip without FEAT_SME will SIGILL on `SMSTART`.

---

# streaming — Stream lifecycle and SSVE register handles

## context lifecycle

| method | signature | semantics |
|--------|-----------|-----------|
| new | `() -> Result<Stream>` | SMSTART (sets PSTATE.SM and PSTATE.ZA), zeroes ZA |
| drop | automatic | SMSTOP (clears both PSTATE bits) |
| svl_bytes | `(&self) -> usize` | RDSVL — 64 on M4 family |
| zero_za | `(&self)` | re-zero ZA without exiting streaming mode |

`Stream` is `!Send + !Sync` (PSTATE is per-thread). Creating a second
`Stream` on the same thread without dropping the first is undefined
behaviour at the spec level — both contexts would write the same PSTATE
bits, but the second `Drop` would clear streaming mode while callers
think it is live.

### encoding

```
SMSTART (full): 0xD503477F   # PSTATE.SM=1, PSTATE.ZA=1
SMSTOP (full):  0xD503467F   # PSTATE.SM=0, PSTATE.ZA=0
RDSVL Xd,#imm:  0x04BF5800 + (imm6 << 5) + Xd
ZERO {za}:      0xC00800FF   # zero all ZA tiles
```

## SSVE register handles

Inside `Stream`, SVE registers Z0–Z31 and predicates P0–P15 become
addressable. acpu wraps the most useful predicated ops as typed
methods. Each method is one `.word`-encoded instruction in inline asm
because stable Rust 1.95 LLVM does not emit SME mnemonics without
target-feature support that the host crate cannot enable.

The register handles are zero-sized type tags. The borrow checker
ensures the Stream outlives them.

| type | what it names |
|------|---------------|
| Zr | one of Z0..Z31 |
| Pr | one of P0..P15 |

### essential ops (Phase 1 surface)

| method | encoding pattern | semantics |
|--------|-----------------|-----------|
| ptrue_all_s | `0x2598E3E0 \| Pd` | P[d] = all-ones for 32-bit lanes |
| ptrue_all_b | `0x2518E3E0 \| Pd` | P[d] = all-ones for byte lanes |
| whilelt_s | `0x25A01400 \| (Rm << 16) \| (Rn << 5) \| Pd` | P[d].s = lane < Rm-Rn |
| ld1w | `0xA5404000 \| (Pg << 10) \| (Rn << 5) \| Zt` | Z[t].s = load 32-bit lanes from [Rn] masked by P[g]/z |
| st1w | `0xE540E000 \| (Pg << 10) \| (Rn << 5) \| Zt` | store Z[t].s to [Rn] masked by P[g] |
| fmla_s | `0x65A00000 \| (Zm << 16) \| (Pg << 10) \| (Zn << 5) \| Zda` | Z[da].s += Z[n].s * Z[m].s gated by P[g] |
| fmul_s | `0x65800800 \| (Zm << 16) \| (Zn << 5) \| Zd` | unpredicated Z[d].s = Z[n].s * Z[m].s |
| fadd_s | `0x65800000 \| (Zm << 16) \| (Zn << 5) \| Zd` | unpredicated Z[d].s = Z[n].s + Z[m].s |
| dup_x_s | `0x05203800 \| (Rn << 5) \| Zd` | Z[d].s = broadcast(Rn[31:0]) |

These cover loop bodies that need predicated load/store, an FMA, and a
tail-mask predicate. The rest of the SSVE ISA is added on demand as
later phases need it.

---

# sme — outer-product matmul

## ZA tiles

The 64×64-byte ZA accumulator partitions into per-element-type tiles:

| tile name | element | tile count | tile shape (lanes) |
|-----------|---------|-----------|--------------------|
| ZA0.S — ZA3.S | f32 / s32 | 4 | 16 × 16 |
| ZA0.D — ZA7.D | f64 / s64 | 8 | 8 × 8 |
| ZA0.H — ZA1.H | f16 / s16 | 2 | 32 × 32 (M4 lacks f16f16) |
| ZA0.B | i8 / s8 | 1 | 64 × 64 |

acpu exposes one outer-product instruction per supported type. Each is
a wide-accumulating MOPA that consumes two Z registers and one
predicate per dimension.

| method | encoding pattern | semantics |
|--------|-----------------|-----------|
| fmopa_s | `0x80800000 \| (Zm << 16) \| (Pm << 13) \| (Pn << 10) \| (Zn << 5) \| ZAd` | ZA[d].s += outer(Z[n].s, Z[m].s) |
| bfmopa_s | `0x81800000 \| (Zm << 16) \| (Pm << 13) \| (Pn << 10) \| (Zn << 5) \| ZAd` | ZA[d].s += outer(bf16 widen Z[n].h, Z[m].h) |
| smopa_s | `0xA0800000 \| (Zm << 16) \| (Pm << 13) \| (Pn << 10) \| (Zn << 5) \| ZAd` | ZA[d].s += outer(i8 widen Z[n].b, Z[m].b) (4-way) |
| fmopa_d | `0x80C00000 \| (Zm << 16) \| (Pm << 13) \| (Pn << 10) \| (Zn << 5) \| ZAd` | ZA[d].d += outer(Z[n].d, Z[m].d) (FEAT_SME_F64F64) |
| smopa_d | `0xA0C00000 \| (Zm << 16) \| (Pm << 13) \| (Pn << 10) \| (Zn << 5) \| ZAd` | ZA[d].d += outer(i16 widen Z[n].h, Z[m].h) (FEAT_SME_I16I64) |

ZAd index range:
- `_s`: 0..=3
- `_d`: 0..=7

## MOVA: stream lanes between Z and ZA

For f32, the row of ZA0.S indexed by `w12+ofs` is one SVL-wide vector
of 16 f32. `mova_z_from_za_h_s` reads that row into a Z register;
`mova_za_h_from_z_s` writes a Z register into that row.

| method | encoding pattern | semantics |
|--------|-----------------|-----------|
| mova_za_h_from_z_s | `0xC0800000 \| (ZAd << 3) \| (Pg << 10) \| (Wi << 16) \| (ofs) \| Zn` | ZA tile[d].horiz[w+ofs] = Z[n].s gated by P[g] |
| mova_z_from_za_h_s | `0xC0820000 \| (Zd) \| (Pg << 10) \| (ZAs << 3) \| ofs` | Z[d].s = ZA tile[s].horiz[w+ofs] gated by P[g] |

## public matmul entry points

| function | signature | semantics |
|----------|-----------|-----------|
| matmul_f32_sme | `(a, b, c, m, n, k)` | C[m×n] += A[m×k] × B[k×n], fp32, ZA tile microkernel + cache blocking |
| matmul_f32_sme_set | same | C[m×n] = A × B (overwrite) |
| matmul_f64_sme | same with f64 slices | requires FEAT_SME_F64F64 |
| matmul_bf16_sme | bf16 inputs → fp32 C | always available on FEAT_SME |
| matmul_i8_sme | i8 inputs → i32 C | always available on FEAT_SME |

Dispatch rule for the unified `acpu::matmul_f32` (added in Phase 2):

```
if has_sme and size in fast_sme_range:   matmul_f32_sme
elif has_amx (M3+):                       matmul_f32_amx  (existing)
else:                                     matmul_f32_neon (existing)
```

`fast_sme_range` is set empirically by `bench/sme.rs`; provisional
range is 64 ≤ m,n,k ≤ 1024 (where AMX's 16×16 GEBP loop has more
per-call overhead than SME's predicated tail).

---

# lut — SME2 LUTI lookup (Phase 3)

## ZT0 load

ZT0 is a single SVL-wide table register. Loaded via `LDR ZT0, [Rn]`.

| method | encoding | semantics |
|--------|----------|-----------|
| ldr_zt0 | `0xE11F8000 \| Rn` | ZT0 = load(SVL bytes from [Rn]) |
| str_zt0 | `0xE13F8000 \| Rn` | store ZT0 to [Rn] |
| zero_zt0 | `0xC0480001` | ZT0 = 0 |

## lookup ops

| method | encoding pattern | semantics |
|--------|-----------------|-----------|
| luti2_s | `0xC0CC2000 \| (idx2 << 10) \| (Zn << 5) \| Zd` | Z[d].s = ZT0[Z[n].s gather, 2-bit lane index = idx2] |
| luti4_s | `0xC0CA2000 \| (idx1 << 10) \| (Zn << 5) \| Zd` | Z[d].s = ZT0[Z[n].s gather, 4-bit lane index = idx1] |

Public API:

| function | signature | semantics |
|----------|-----------|-----------|
| permute_u8_sme | `(table: &[u8;256], idx: &[u8], out: &mut [u8])` | Sbox-style 8→8 permute, LUTI2 |
| permute_u32_sme | `(table: &[u32;16], idx: &[u8], out: &mut [u32])` | 4-bit indexed gather, LUTI4 |
| gather_u32_sme | `(table: &[u32], idx: &[u32], out: &mut [u32])` | indirect gather over a large table; chunks into SVL-sized ZT0 loads |

---

# error mapping

| error | when |
|-------|------|
| FeatureNotAvailable(Sme) | constructing Stream on non-M4 host |
| FeatureNotAvailable(Sme2) | calling any `lut::*` on non-SME2 host |
| FeatureNotAvailable(SmeF64F64) | calling `matmul_f64_sme` without F64F64 |
| FeatureNotAvailable(SmeI16I64) | calling `matmul_i16_sme` without I16I64 |

---

# implementation files

```
src/
  streaming/
    mod.rs            Stream struct, lifecycle, svl_bytes, zero_za
    asm.rs            raw .word encodings for SMSTART, SMSTOP, RDSVL, ZERO ZA
    ssve.rs           predicated SSVE ops (ld1w, st1w, ptrue, whilelt, fmla, ...)
  sme/
    mod.rs            public matmul_*_sme entry points
    asm.rs            FMOPA / BFMOPA / SMOPA / MOVA encodings
    tile.rs           16×16 f32 ZA tile microkernel (load A row, load B row, FMOPA loop, store C)
    gemm.rs           cache-blocked matmul_f32_sme (mirrors matmul_f32_amx_single)
  lut/
    mod.rs            permute_*_sme, gather_*_sme
    asm.rs            LDR/STR ZT0, LUTI2, LUTI4 encodings
examples/
  sme_smoke.rs        Stream::new(); print svl_bytes; verify == 64
  sme_matmul.rs       small matmul_f32_sme demo + correctness vs scalar
  lut_permute.rs      8-bit permute demo via LUTI2
bench/
  sme.rs              SME GEMM spectrum vs Accelerate vs AMX
  lut.rs              LUTI2/LUTI4 vs NEON TBL
  ssve.rs             gold_mul + tip5 via SSVE vs existing NEON paths
```

500-line file budget per CLAUDE.md applies.

---

# license

cyber license: don't trust. don't fear. don't beg.
