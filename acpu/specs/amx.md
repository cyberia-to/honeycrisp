# Apple AMX Instruction Set Reference

Reverse-engineered on Apple Silicon (M-series). All opcodes verified on real hardware.

## Encoding

Every AMX instruction is a single `.word`:
```
.word (0x00201000 + (opcode << 5) + register_encoding)
```

The `register_encoding` converts LLVM's register numbering to the 5-bit hardware
encoding using the BCD trick: `0{reg} - ((0{reg} >> 4) * 6)`.

Operand is a 64-bit value in the general-purpose register. The hardware reads this
register and interprets the bits according to the opcode.

## Register File

| Register | Count | Size each | Total | Access |
|----------|-------|-----------|-------|--------|
| X | 8 rows | 64 bytes | 512 B | LDX/STX |
| Y | 8 rows | 64 bytes | 512 B | LDY/STY |
| Z | 64 rows | 64 bytes | 4096 B | LDZ/STZ/LDZI/STZI |

Z has 4 independent 16×16 f32 tiles (selected by `z_row & 3`):
- Tile 0: Z rows {0, 4, 8, ..., 60}
- Tile 1: Z rows {1, 5, 9, ..., 61}
- Tile 2: Z rows {2, 6, 10, ..., 62}
- Tile 3: Z rows {3, 7, 11, ..., 63}

## Lifecycle

- `AMX_SET` (opcode 17, operand register = 0): activate AMX on this thread
- `AMX_CLR` (opcode 17, operand register = 1): deactivate AMX on this thread
- AMX state is per-thread. Each thread needs its own SET/CLR bracket.
- SET/CLR are cheap (~1 cycle). Z/X/Y contents persist across SET/CLR pairs.

## Complete Opcode Map

Verified on hardware. Opcodes 23-31 cause SIGILL.

### Load / Store (opcodes 0-7)

| Op | Name | Operand | Effect |
|----|------|---------|--------|
| 0 | LDX | `ptr \| (row << 56)` | Load 64 bytes from `ptr` into X[row] (row: 0-7) |
| 1 | LDY | `ptr \| (row << 56)` | Load 64 bytes from `ptr` into Y[row] (row: 0-7) |
| 2 | STX | `ptr \| (row << 56)` | Store 64 bytes from X[row] to `ptr` |
| 3 | STY | `ptr \| (row << 56)` | Store 64 bytes from Y[row] to `ptr` |
| 4 | LDZ | `ptr \| (row << 56)` | Load 64 bytes from `ptr` into Z[row] (row: 0-63) |
| 5 | STZ | `ptr \| (row << 56)` | Store 64 bytes from Z[row] to `ptr` |
| 6 | LDZI | `ptr \| (row << 56)` | Load into Z[row] with interleaved layout |
| 7 | STZI | `ptr \| (row << 56)` | Store from Z[row] with interleaved layout |

Row field: bits 58:56 for X/Y (3 bits, 0-7), bits 61:56 for Z (6 bits, 0-63).

### Extract (opcodes 8-9)

| Op | Name | Effect |
|----|------|--------|
| 8 | EXTRX | Extract from Z into X (horizontal slice) |
| 9 | EXTRY | Extract from Z into Y (vertical slice) |

Operand format: selects which Z row/column to extract.

### FMA — Fused Multiply-Accumulate (opcodes 10-16)

All FMA instructions write to Z. X and Y are read-only inputs.

| Op | Name | Data type | Z stride | Elements/row | Outer product size |
|----|------|-----------|----------|-------------|-------------------|
| 10 | FMA64 | f64 | 8 | 8 | 8×8 |
| 11 | FMS64 | f64 (negate) | 8 | 8 | 8×8 |
| 12 | **FMA32** | **f32** | **4** | **16** | **16×16** |
| 13 | FMS32 | f32 (negate) | 4 | 16 | 16×16 |
| 14 | MAC16 | i16 | 2 | 32 | 32×32 |
| 15 | FMA16 | f16 | 2 | 32 | 32×32 |
| 16 | FMS16 | f16 (negate) | 2 | 32 | 32×32 |

FMS = fused multiply-subtract (Z -= X × Y instead of Z += X × Y).

#### FMA32 Operand Bit Layout (opcode 12)

Verified on hardware via single-bit probing:

```
Bit 63:    vector mode (0 = matrix/outer product, 1 = vector/element-wise)
Bit 62:    no visible effect
Bits 61-60: KILL output (Z empty when set)
Bits 59-38: no visible effect (default behavior)
Bit 37:    partial output (1 Z row only)
Bits 36-34: KILL output (Z empty when set)
Bit 33:    switches to stride-8 mode (f64-like Z addressing)
Bit 32:    switches to stride-8 mode
Bits 31-30: no visible effect
Bit 29:    skip_x (treat X as zero — no visible effect with Y present)
Bit 28:    skip_y (treat Y as zero — Z[0][0] = 1.0 with default X)
Bit 27:    skip_z (no accumulate: Z = X×Y instead of Z += X×Y)
Bits 26-22: no visible effect
Bit 21:    Z tile select bit 1 (tile 2: first row at Z[2])
Bit 20:    Z tile select bit 0 (tile 1: first row at Z[1])
Bits 19-16: no visible effect (X offset high bits)
Bits 15-12: X offset — selects X row via byte offset
Bits 11-10: X offset low bits
Bits 9-6:  Y offset — selects Y row via byte offset
Bits 5-2:  Y offset — changes output value (Y element selection)
Bits 1-0:  Y offset low bits
```

Functional fields:
- **Y offset (bits 9:0)**: byte offset into 512-byte circular Y buffer. Y row N = N×64 bytes.
- **X offset (bits 19:10)**: byte offset into 512-byte circular X buffer. X row N = N×64 bytes.
- **Z tile (bits 21:20)**: selects Z tile 0-3.
- **skip_z (bit 27)**: first iteration flag — Z = X×Y (ignores current Z).
- **skip_y (bit 28)**: Y treated as zero.
- **skip_x (bit 29)**: X treated as zero.
- **vector mode (bit 63)**: element-wise instead of outer product.

#### FMA32 Operation (matrix mode, bit 63 = 0)

```
for j in 0..16:
    for i in 0..16:
        Z[(j * 4 + tile) % 64][i] += X[x_offset/4 + i] * Y[y_offset/4 + j]
```

One fma32 instruction = 512 FLOPS (16 × 16 × 2).

### Unknown Opcodes (18-22)

Valid on this hardware. Not yet fully characterized.

| Op | Z changed | Z rows | Z stride | Hypothesis |
|----|-----------|--------|----------|------------|
| 18 | YES | 1 | - | i32 dot product or reduced mac |
| 19 | YES | 1 | - | f32 dot product (values are sums) |
| 20 | YES | 16 | 2 | i32 matrix mode (like mac16 variant) |
| 21 | YES | 16 | 2 | f32 matrix with fp16-like accumulation |
| 22 | X changed | 0 | - | X extract variant (like extrx) |

Opcodes 18-19 produce single-row Z output (possible dot product / reduction).
Opcodes 20-21 produce 16-row output with stride 2 (possible mixed-precision).
Opcode 22 modifies X registers (possible extract with different addressing).

Opcodes 23-31: SIGILL (illegal instruction) on this chip.

## Performance Characteristics

- LDX/LDY: ~1-2 cycles from L1
- FMA32: ~2-3 cycles throughput (pipelined)
- STZ: ~1-2 cycles
- AMX load and FMA do NOT overlap well (interleaving hurts performance)
- AMX SET/CLR: ~1 cycle each
- Prefetch (PRFM) before AMX loads improves throughput ~5-10%
- 4 Z tiles allow register blocking: one Y load can serve 4 FMA instructions

## Optimal GEMM Pattern

1. Pack A into MR=16 wide column-major strips (Y loading)
2. Pack B into NR=16 wide row-major strips (X loading)
3. For each K batch of 8: load 8 Y + 8 X, issue 8 FMA32
4. Use all 4 Z tiles for 16×64 computation (each Y load → 4 FMAs)
5. STZ result to memory, NEON vaddq for C accumulation
6. L1 constraint: KC × (MR + NR) × 4 ≤ L1D (64KB for Apple Silicon)
7. L2 constraint: MC × KC × 4 ≤ L2 (4MB per core)
