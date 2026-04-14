//! AMX f32 tile operations for GEMM microkernels.
//!
//! Two microkernels:
//! - `microkernel_16x16`: single tile, Z tile 0
//! - `microkernel_16x32`: double-wide, Z tiles 0+1, 2× compute per A load

use super::asm::{amx_op, OP_FMA32, OP_LDX, OP_LDY, OP_LDZ, OP_STZ};
use super::fma::{fma_acc, fma_first};
use super::regs::{XRow, YRow};

// ---------------------------------------------------------------------------
// C preload / direct store — eliminates AMX→CPU sync overhead
// ---------------------------------------------------------------------------

/// Load C[16×16] into Z tile via LDZ. No CPU involvement in data path.
///
/// # Safety
/// AMX must be active. `c` must point to `16 * ldc` readable f32 elements.
#[inline]
pub unsafe fn preload_c(c: *const f32, ldc: usize, tile: u8) {
    let stride = ldc * 4; // bytes per row
    let base = c as *const u8;
    let t = tile as u64;
    // Precompute all 16 operands and issue in single asm block.
    let o0 = (base as u64) | (t << 56);
    let o1 = (base.add(stride) as u64) | ((4 + t) << 56);
    let o2 = (base.add(stride * 2) as u64) | ((8 + t) << 56);
    let o3 = (base.add(stride * 3) as u64) | ((12 + t) << 56);
    let o4 = (base.add(stride * 4) as u64) | ((16 + t) << 56);
    let o5 = (base.add(stride * 5) as u64) | ((20 + t) << 56);
    let o6 = (base.add(stride * 6) as u64) | ((24 + t) << 56);
    let o7 = (base.add(stride * 7) as u64) | ((28 + t) << 56);
    let o8 = (base.add(stride * 8) as u64) | ((32 + t) << 56);
    let o9 = (base.add(stride * 9) as u64) | ((36 + t) << 56);
    let oa = (base.add(stride * 10) as u64) | ((40 + t) << 56);
    let ob = (base.add(stride * 11) as u64) | ((44 + t) << 56);
    let oc = (base.add(stride * 12) as u64) | ((48 + t) << 56);
    let od = (base.add(stride * 13) as u64) | ((52 + t) << 56);
    let oe = (base.add(stride * 14) as u64) | ((56 + t) << 56);
    let of = (base.add(stride * 15) as u64) | ((60 + t) << 56);
    // 16 LDZ in one asm block — no LLVM barriers between them.
    core::arch::asm!(
        ".word (0x00201000 + (4 << 5) + 0{o0} - ((0{o0} >> 4) * 6))",
        ".word (0x00201000 + (4 << 5) + 0{o1} - ((0{o1} >> 4) * 6))",
        ".word (0x00201000 + (4 << 5) + 0{o2} - ((0{o2} >> 4) * 6))",
        ".word (0x00201000 + (4 << 5) + 0{o3} - ((0{o3} >> 4) * 6))",
        ".word (0x00201000 + (4 << 5) + 0{o4} - ((0{o4} >> 4) * 6))",
        ".word (0x00201000 + (4 << 5) + 0{o5} - ((0{o5} >> 4) * 6))",
        ".word (0x00201000 + (4 << 5) + 0{o6} - ((0{o6} >> 4) * 6))",
        ".word (0x00201000 + (4 << 5) + 0{o7} - ((0{o7} >> 4) * 6))",
        ".word (0x00201000 + (4 << 5) + 0{o8} - ((0{o8} >> 4) * 6))",
        ".word (0x00201000 + (4 << 5) + 0{o9} - ((0{o9} >> 4) * 6))",
        ".word (0x00201000 + (4 << 5) + 0{oa} - ((0{oa} >> 4) * 6))",
        ".word (0x00201000 + (4 << 5) + 0{ob} - ((0{ob} >> 4) * 6))",
        ".word (0x00201000 + (4 << 5) + 0{oc} - ((0{oc} >> 4) * 6))",
        ".word (0x00201000 + (4 << 5) + 0{od} - ((0{od} >> 4) * 6))",
        ".word (0x00201000 + (4 << 5) + 0{oe} - ((0{oe} >> 4) * 6))",
        ".word (0x00201000 + (4 << 5) + 0{of} - ((0{of} >> 4) * 6))",
        o0 = in(reg) o0, o1 = in(reg) o1, o2 = in(reg) o2, o3 = in(reg) o3,
        o4 = in(reg) o4, o5 = in(reg) o5, o6 = in(reg) o6, o7 = in(reg) o7,
        o8 = in(reg) o8, o9 = in(reg) o9, oa = in(reg) oa, ob = in(reg) ob,
        oc = in(reg) oc, od = in(reg) od, oe = in(reg) oe, of = in(reg) of,
        options(nostack),
    );
}

/// Store Z tile directly to C[16×16] via STZ. CPU never reads the data.
///
/// # Safety
/// AMX must be active. `c` must point to `16 * ldc` writable f32 elements.
#[inline]
pub unsafe fn store_c(c: *mut f32, ldc: usize, tile: u8) {
    let stride = ldc * 4;
    let base = c as *mut u8;
    let t = tile as u64;
    let o0 = (base as u64) | (t << 56);
    let o1 = (base.add(stride) as u64) | ((4 + t) << 56);
    let o2 = (base.add(stride * 2) as u64) | ((8 + t) << 56);
    let o3 = (base.add(stride * 3) as u64) | ((12 + t) << 56);
    let o4 = (base.add(stride * 4) as u64) | ((16 + t) << 56);
    let o5 = (base.add(stride * 5) as u64) | ((20 + t) << 56);
    let o6 = (base.add(stride * 6) as u64) | ((24 + t) << 56);
    let o7 = (base.add(stride * 7) as u64) | ((28 + t) << 56);
    let o8 = (base.add(stride * 8) as u64) | ((32 + t) << 56);
    let o9 = (base.add(stride * 9) as u64) | ((36 + t) << 56);
    let oa = (base.add(stride * 10) as u64) | ((40 + t) << 56);
    let ob = (base.add(stride * 11) as u64) | ((44 + t) << 56);
    let oc = (base.add(stride * 12) as u64) | ((48 + t) << 56);
    let od = (base.add(stride * 13) as u64) | ((52 + t) << 56);
    let oe = (base.add(stride * 14) as u64) | ((56 + t) << 56);
    let of = (base.add(stride * 15) as u64) | ((60 + t) << 56);
    core::arch::asm!(
        ".word (0x00201000 + (5 << 5) + 0{o0} - ((0{o0} >> 4) * 6))",
        ".word (0x00201000 + (5 << 5) + 0{o1} - ((0{o1} >> 4) * 6))",
        ".word (0x00201000 + (5 << 5) + 0{o2} - ((0{o2} >> 4) * 6))",
        ".word (0x00201000 + (5 << 5) + 0{o3} - ((0{o3} >> 4) * 6))",
        ".word (0x00201000 + (5 << 5) + 0{o4} - ((0{o4} >> 4) * 6))",
        ".word (0x00201000 + (5 << 5) + 0{o5} - ((0{o5} >> 4) * 6))",
        ".word (0x00201000 + (5 << 5) + 0{o6} - ((0{o6} >> 4) * 6))",
        ".word (0x00201000 + (5 << 5) + 0{o7} - ((0{o7} >> 4) * 6))",
        ".word (0x00201000 + (5 << 5) + 0{o8} - ((0{o8} >> 4) * 6))",
        ".word (0x00201000 + (5 << 5) + 0{o9} - ((0{o9} >> 4) * 6))",
        ".word (0x00201000 + (5 << 5) + 0{oa} - ((0{oa} >> 4) * 6))",
        ".word (0x00201000 + (5 << 5) + 0{ob} - ((0{ob} >> 4) * 6))",
        ".word (0x00201000 + (5 << 5) + 0{oc} - ((0{oc} >> 4) * 6))",
        ".word (0x00201000 + (5 << 5) + 0{od} - ((0{od} >> 4) * 6))",
        ".word (0x00201000 + (5 << 5) + 0{oe} - ((0{oe} >> 4) * 6))",
        ".word (0x00201000 + (5 << 5) + 0{of} - ((0{of} >> 4) * 6))",
        o0 = in(reg) o0, o1 = in(reg) o1, o2 = in(reg) o2, o3 = in(reg) o3,
        o4 = in(reg) o4, o5 = in(reg) o5, o6 = in(reg) o6, o7 = in(reg) o7,
        o8 = in(reg) o8, o9 = in(reg) o9, oa = in(reg) oa, ob = in(reg) ob,
        oc = in(reg) oc, od = in(reg) od, oe = in(reg) oe, of = in(reg) of,
        options(nostack),
    );
}

// ---------------------------------------------------------------------------
// 16×16 microkernel (single tile)
// ---------------------------------------------------------------------------

/// AMX 16×16 f32 microkernel. Z tile 0.
///
/// # Safety
/// AMX must be active. Panels must be 64-byte aligned with `k*64` readable bytes.
#[inline]
pub unsafe fn microkernel_16x16(a_panel: *const u8, b_panel: *const u8, k: usize) {
    let mut first = true;
    let mut p = 0usize;

    while p + 8 <= k {
        // Prefetch next batch while loading current.
        if p + 16 <= k {
            for i in (0..8).step_by(4) {
                crate::sync::prefetch::prefetch_l1(b_panel.add((p + 8 + i) * 64));
                crate::sync::prefetch::prefetch_l1(a_panel.add((p + 8 + i) * 64));
            }
        }
        for i in 0u8..8 {
            amx_op::<OP_LDX>((b_panel.add((p + i as usize) * 64) as u64) | ((i as u64) << 56));
            amx_op::<OP_LDY>((a_panel.add((p + i as usize) * 64) as u64) | ((i as u64) << 56));
        }
        if first {
            amx_op::<OP_FMA32>(fma_first(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
            first = false;
        } else {
            amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
        }
        for i in 1u8..8 {
            amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(i), YRow::new_unchecked(i), 0));
        }
        p += 8;
    }

    while p < k {
        amx_op::<OP_LDX>(b_panel.add(p * 64) as u64);
        amx_op::<OP_LDY>(a_panel.add(p * 64) as u64);
        if first {
            amx_op::<OP_FMA32>(fma_first(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
            first = false;
        } else {
            amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
        }
        p += 1;
    }
}

// ---------------------------------------------------------------------------
// 16×16 microkernel — accumulate-only (Z preloaded with C)
// ---------------------------------------------------------------------------

/// AMX 16×16 microkernel that accumulates into existing Z tile 0.
/// Use after `preload_c` to compute C += A × B without AMX→CPU sync.
///
/// # Safety
/// AMX must be active. Z tile must be preloaded. Panels: 64-byte aligned, `k*64` bytes.
#[inline]
pub unsafe fn microkernel_16x16_acc(a_panel: *const u8, b_panel: *const u8, k: usize, bs: usize) {
    let mut p = 0usize;

    while p + 8 <= k {
        if p + 16 <= k {
            for i in (0..8).step_by(4) {
                crate::sync::prefetch::prefetch_l1(b_panel.add((p + 8 + i) * bs));
                crate::sync::prefetch::prefetch_l1(a_panel.add((p + 8 + i) * 64));
            }
        }
        for i in 0u8..8 {
            amx_op::<OP_LDX>((b_panel.add((p + i as usize) * bs) as u64) | ((i as u64) << 56));
            amx_op::<OP_LDY>((a_panel.add((p + i as usize) * 64) as u64) | ((i as u64) << 56));
        }
        for i in 0u8..8 {
            amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(i), YRow::new_unchecked(i), 0));
        }
        p += 8;
    }

    while p < k {
        amx_op::<OP_LDX>(b_panel.add(p * bs) as u64);
        amx_op::<OP_LDY>(a_panel.add(p * 64) as u64);
        amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
        p += 1;
    }
}

// ---------------------------------------------------------------------------
// 16×32 microkernel (double-wide: 2 tiles side by side in N)
// ---------------------------------------------------------------------------

/// AMX 16×32 f32 microkernel. Uses Z tiles 0 and 1.
///
/// Computes C[16×32] += A[16×K] × B[K×32] using 2 tiles:
/// - Tile 0: C[0..16, 0..16]   = A × B_left
/// - Tile 1: C[0..16, 16..32]  = A × B_right
///
/// # Safety
/// AMX must be active. All panels must be 64-byte aligned with `k*64` readable bytes.
#[inline]
pub unsafe fn microkernel_16x32(
    a_panel: *const u8,
    b_left: *const u8,
    b_right: *const u8,
    k: usize,
) {
    let mut first_t0 = true;
    let mut first_t1 = true;
    let mut p = 0usize;

    // Batch of 4: 4Y + 4X(left) + 4X(right) = 4Y + 8X = all registers used.
    while p + 4 <= k {
        // Prefetch next batch.
        if p + 8 <= k {
            crate::sync::prefetch::prefetch_l1(a_panel.add((p + 4) * 64));
            crate::sync::prefetch::prefetch_l1(b_left.add((p + 4) * 64));
            crate::sync::prefetch::prefetch_l1(b_right.add((p + 4) * 64));
        }

        for i in 0u8..4 {
            amx_op::<OP_LDY>((a_panel.add((p + i as usize) * 64) as u64) | ((i as u64) << 56));
        }
        for i in 0u8..4 {
            amx_op::<OP_LDX>((b_left.add((p + i as usize) * 64) as u64) | ((i as u64) << 56));
        }

        // FMA tile 0
        if first_t0 {
            amx_op::<OP_FMA32>(fma_first(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
            first_t0 = false;
        } else {
            amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
        }
        for i in 1u8..4 {
            amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(i), YRow::new_unchecked(i), 0));
        }

        // Load B_right into X[4..7]
        for i in 0u8..4 {
            amx_op::<OP_LDX>(
                (b_right.add((p + i as usize) * 64) as u64) | (((4 + i) as u64) << 56),
            );
        }

        // FMA tile 1 (reuses Y[0..3])
        if first_t1 {
            amx_op::<OP_FMA32>(fma_first(XRow::new_unchecked(4), YRow::new_unchecked(0), 1));
            first_t1 = false;
        } else {
            amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(4), YRow::new_unchecked(0), 1));
        }
        for i in 1u8..4 {
            amx_op::<OP_FMA32>(fma_acc(
                XRow::new_unchecked(4 + i),
                YRow::new_unchecked(i),
                1,
            ));
        }

        p += 4;
    }

    // Remainder: one at a time.
    while p < k {
        amx_op::<OP_LDY>(a_panel.add(p * 64) as u64);
        amx_op::<OP_LDX>(b_left.add(p * 64) as u64);

        if first_t0 {
            amx_op::<OP_FMA32>(fma_first(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
            first_t0 = false;
        } else {
            amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
        }

        amx_op::<OP_LDX>((b_right.add(p * 64) as u64) | (1u64 << 56));
        if first_t1 {
            amx_op::<OP_FMA32>(fma_first(XRow::new_unchecked(1), YRow::new_unchecked(0), 1));
            first_t1 = false;
        } else {
            amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(1), YRow::new_unchecked(0), 1));
        }

        p += 1;
    }
}

// ---------------------------------------------------------------------------
// Fused AMX batch: single asm block eliminates LLVM fences between ops
// ---------------------------------------------------------------------------

/// 18 AMX ops in ONE asm block: 2 LDY + 8 LDX + 8 FMA.
/// No LLVM-inserted fences between instructions.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn fused_batch_16x64(
    a0: u64,
    a1: u64,
    bx0: u64,
    bx1: u64,
    bx2: u64,
    bx3: u64,
    bx4: u64,
    bx5: u64,
    bx6: u64,
    bx7: u64,
    f0: u64,
    f1: u64,
    f2: u64,
    f3: u64,
    g0: u64,
    g1: u64,
    g2: u64,
    g3: u64,
) {
    // Interleaved: overlap loads with FMA for AMX pipeline utilization.
    // op 1 = LDY, op 0 = LDX, op 12 = FMA32
    core::arch::asm!(
        // Load Y[0], start loading X[0..1]
        ".word (0x00201000 + (1 << 5) + 0{a0} - ((0{a0} >> 4) * 6))",
        ".word (0x00201000 + (0 << 5) + 0{bx0} - ((0{bx0} >> 4) * 6))",
        ".word (0x00201000 + (0 << 5) + 0{bx1} - ((0{bx1} >> 4) * 6))",
        // FMA tile 0 while X[2] loads
        ".word (0x00201000 + (0 << 5) + 0{bx2} - ((0{bx2} >> 4) * 6))",
        ".word (0x00201000 + (12 << 5) + 0{f0} - ((0{f0} >> 4) * 6))",
        // FMA tile 1 while X[3] loads
        ".word (0x00201000 + (0 << 5) + 0{bx3} - ((0{bx3} >> 4) * 6))",
        ".word (0x00201000 + (12 << 5) + 0{f1} - ((0{f1} >> 4) * 6))",
        // FMA tiles 2-3, load Y[1]
        ".word (0x00201000 + (12 << 5) + 0{f2} - ((0{f2} >> 4) * 6))",
        ".word (0x00201000 + (1 << 5) + 0{a1} - ((0{a1} >> 4) * 6))",
        ".word (0x00201000 + (12 << 5) + 0{f3} - ((0{f3} >> 4) * 6))",
        // Load X[4..5], FMA tile 0 with Y[1]
        ".word (0x00201000 + (0 << 5) + 0{bx4} - ((0{bx4} >> 4) * 6))",
        ".word (0x00201000 + (0 << 5) + 0{bx5} - ((0{bx5} >> 4) * 6))",
        ".word (0x00201000 + (12 << 5) + 0{g0} - ((0{g0} >> 4) * 6))",
        // Load X[6..7], FMA tiles 1-3 with Y[1]
        ".word (0x00201000 + (0 << 5) + 0{bx6} - ((0{bx6} >> 4) * 6))",
        ".word (0x00201000 + (12 << 5) + 0{g1} - ((0{g1} >> 4) * 6))",
        ".word (0x00201000 + (0 << 5) + 0{bx7} - ((0{bx7} >> 4) * 6))",
        ".word (0x00201000 + (12 << 5) + 0{g2} - ((0{g2} >> 4) * 6))",
        ".word (0x00201000 + (12 << 5) + 0{g3} - ((0{g3} >> 4) * 6))",
        a0 = in(reg) a0,
        a1 = in(reg) a1,
        bx0 = in(reg) bx0,
        bx1 = in(reg) bx1,
        bx2 = in(reg) bx2,
        bx3 = in(reg) bx3,
        bx4 = in(reg) bx4,
        bx5 = in(reg) bx5,
        bx6 = in(reg) bx6,
        bx7 = in(reg) bx7,
        f0 = in(reg) f0,
        f1 = in(reg) f1,
        f2 = in(reg) f2,
        f3 = in(reg) f3,
        g0 = in(reg) g0,
        g1 = in(reg) g1,
        g2 = in(reg) g2,
        g3 = in(reg) g3,
        options(nostack),
    );
}

// ---------------------------------------------------------------------------
// 16×64 microkernel (quad-wide: all 4 Z tiles in N direction)
// ---------------------------------------------------------------------------

/// AMX 16×64 f32 microkernel. Uses all 4 Z tiles.
///
/// Each Y load (A column) serves 4 FMAs — maximum register reuse.
/// Batch of 2: 2Y + 8X + 8 FMA = 18 instructions for 4096 FLOPS.
///
/// # Safety
/// AMX must be active. All panels must be 64-byte aligned with `k*64` readable bytes.
#[inline]
pub unsafe fn microkernel_16x64(
    a_panel: *const u8,
    b0: *const u8,
    b1: *const u8,
    b2: *const u8,
    b3: *const u8,
    k: usize,
) {
    let mut p = 0usize;

    // First batch: use skip_z (bit 27) to clear all 4 tiles.
    if p + 2 <= k {
        let a0 = a_panel.add(p * 64) as u64;
        let a1 = (a_panel.add((p + 1) * 64) as u64) | (1u64 << 56);
        let bx0 = b0.add(p * 64) as u64;
        let bx1 = (b1.add(p * 64) as u64) | (1u64 << 56);
        let bx2 = (b2.add(p * 64) as u64) | (2u64 << 56);
        let bx3 = (b3.add(p * 64) as u64) | (3u64 << 56);
        let bx4 = (b0.add((p + 1) * 64) as u64) | (4u64 << 56);
        let bx5 = (b1.add((p + 1) * 64) as u64) | (5u64 << 56);
        let bx6 = (b2.add((p + 1) * 64) as u64) | (6u64 << 56);
        let bx7 = (b3.add((p + 1) * 64) as u64) | (7u64 << 56);

        // fma_first operands (skip_z=1) for tiles 0-3.
        let f0 = fma_first(XRow::new_unchecked(0), YRow::new_unchecked(0), 0);
        let f1 = fma_first(XRow::new_unchecked(1), YRow::new_unchecked(0), 1);
        let f2 = fma_first(XRow::new_unchecked(2), YRow::new_unchecked(0), 2);
        let f3 = fma_first(XRow::new_unchecked(3), YRow::new_unchecked(0), 3);
        let g0 = fma_acc(XRow::new_unchecked(4), YRow::new_unchecked(1), 0);
        let g1 = fma_acc(XRow::new_unchecked(5), YRow::new_unchecked(1), 1);
        let g2 = fma_acc(XRow::new_unchecked(6), YRow::new_unchecked(1), 2);
        let g3 = fma_acc(XRow::new_unchecked(7), YRow::new_unchecked(1), 3);

        // Single asm block: 18 AMX ops, no LLVM fences between them.
        fused_batch_16x64(
            a0, a1, bx0, bx1, bx2, bx3, bx4, bx5, bx6, bx7, f0, f1, f2, f3, g0, g1, g2, g3,
        );
        p += 2;
    }

    // Steady-state: accumulate (no skip_z).
    while p + 2 <= k {
        if p + 4 <= k {
            crate::sync::prefetch::prefetch_l1(a_panel.add((p + 2) * 64));
            crate::sync::prefetch::prefetch_l1(b0.add((p + 2) * 64));
        }

        let a0 = a_panel.add(p * 64) as u64;
        let a1 = (a_panel.add((p + 1) * 64) as u64) | (1u64 << 56);
        let bx0 = b0.add(p * 64) as u64;
        let bx1 = (b1.add(p * 64) as u64) | (1u64 << 56);
        let bx2 = (b2.add(p * 64) as u64) | (2u64 << 56);
        let bx3 = (b3.add(p * 64) as u64) | (3u64 << 56);
        let bx4 = (b0.add((p + 1) * 64) as u64) | (4u64 << 56);
        let bx5 = (b1.add((p + 1) * 64) as u64) | (5u64 << 56);
        let bx6 = (b2.add((p + 1) * 64) as u64) | (6u64 << 56);
        let bx7 = (b3.add((p + 1) * 64) as u64) | (7u64 << 56);

        let f0 = fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0);
        let f1 = fma_acc(XRow::new_unchecked(1), YRow::new_unchecked(0), 1);
        let f2 = fma_acc(XRow::new_unchecked(2), YRow::new_unchecked(0), 2);
        let f3 = fma_acc(XRow::new_unchecked(3), YRow::new_unchecked(0), 3);
        let g0 = fma_acc(XRow::new_unchecked(4), YRow::new_unchecked(1), 0);
        let g1 = fma_acc(XRow::new_unchecked(5), YRow::new_unchecked(1), 1);
        let g2 = fma_acc(XRow::new_unchecked(6), YRow::new_unchecked(1), 2);
        let g3 = fma_acc(XRow::new_unchecked(7), YRow::new_unchecked(1), 3);

        fused_batch_16x64(
            a0, a1, bx0, bx1, bx2, bx3, bx4, bx5, bx6, bx7, f0, f1, f2, f3, g0, g1, g2, g3,
        );
        p += 2;
    }

    // Remainder.
    if p < k {
        amx_op::<OP_LDY>(a_panel.add(p * 64) as u64);
        amx_op::<OP_LDX>(b0.add(p * 64) as u64);
        amx_op::<OP_LDX>((b1.add(p * 64) as u64) | (1u64 << 56));
        amx_op::<OP_LDX>((b2.add(p * 64) as u64) | (2u64 << 56));
        amx_op::<OP_LDX>((b3.add(p * 64) as u64) | (3u64 << 56));

        amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
        amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(1), YRow::new_unchecked(0), 1));
        amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(2), YRow::new_unchecked(0), 2));
        amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(3), YRow::new_unchecked(0), 3));
    }
}

// ---------------------------------------------------------------------------
// 16×64 microkernel — strided B, zeros Z (no preload needed)
// ---------------------------------------------------------------------------

/// AMX 16×64 microkernel with strided B. Zeros Z via fma_first, no preload_c needed.
/// Use with accumulate_tile to add result to C via NEON (avoids 64 LDZ per call).
///
/// # Safety
/// AMX must be active. A: 64-byte stride. B panels: `bs`-byte stride.
#[inline]
pub unsafe fn microkernel_16x64_strided(
    a_panel: *const u8,
    b0: *const u8,
    b1: *const u8,
    b2: *const u8,
    b3: *const u8,
    k: usize,
    bs: usize,
) {
    let f0_first = fma_first(XRow::new_unchecked(0), YRow::new_unchecked(0), 0);
    let f1_first = fma_first(XRow::new_unchecked(1), YRow::new_unchecked(0), 1);
    let f2_first = fma_first(XRow::new_unchecked(2), YRow::new_unchecked(0), 2);
    let f3_first = fma_first(XRow::new_unchecked(3), YRow::new_unchecked(0), 3);
    let f0 = fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0);
    let f1 = fma_acc(XRow::new_unchecked(1), YRow::new_unchecked(0), 1);
    let f2 = fma_acc(XRow::new_unchecked(2), YRow::new_unchecked(0), 2);
    let f3 = fma_acc(XRow::new_unchecked(3), YRow::new_unchecked(0), 3);

    // First k-step: fma_first zeros Z tiles.
    if k > 0 {
        amx_op::<OP_LDY>(a_panel as u64);
        amx_op::<OP_LDX>(b0 as u64);
        amx_op::<OP_LDX>((b1 as u64) | (1u64 << 56));
        amx_op::<OP_LDX>((b2 as u64) | (2u64 << 56));
        amx_op::<OP_LDX>((b3 as u64) | (3u64 << 56));
        amx_op::<OP_FMA32>(f0_first);
        amx_op::<OP_FMA32>(f1_first);
        amx_op::<OP_FMA32>(f2_first);
        amx_op::<OP_FMA32>(f3_first);
    }

    let mut p = 1usize;
    while p < k {
        amx_op::<OP_LDY>(a_panel.add(p * 64) as u64);
        amx_op::<OP_LDX>(b0.add(p * bs) as u64);
        amx_op::<OP_LDX>((b1.add(p * bs) as u64) | (1u64 << 56));
        amx_op::<OP_LDX>((b2.add(p * bs) as u64) | (2u64 << 56));
        amx_op::<OP_LDX>((b3.add(p * bs) as u64) | (3u64 << 56));
        amx_op::<OP_FMA32>(f0);
        amx_op::<OP_FMA32>(f1);
        amx_op::<OP_FMA32>(f2);
        amx_op::<OP_FMA32>(f3);
        p += 1;
    }
}

// ---------------------------------------------------------------------------
// 16×64 microkernel — accumulate-only (Z preloaded with C)
// ---------------------------------------------------------------------------

/// AMX 16×64 microkernel that accumulates into existing Z tiles 0-3.
/// `bs` = B stride in bytes (64 for packed NR-wide strips, n*4 for direct row-major).
///
/// # Safety
/// AMX must be active. Z tiles must be preloaded. A panel: 64-byte stride.
/// B panels: `bs`-byte stride, each row = 64 readable bytes.
#[inline]
pub unsafe fn microkernel_16x64_acc(
    a_panel: *const u8,
    b0: *const u8,
    b1: *const u8,
    b2: *const u8,
    b3: *const u8,
    k: usize,
    bs: usize,
) {
    let f0 = fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0);
    let f1 = fma_acc(XRow::new_unchecked(1), YRow::new_unchecked(0), 1);
    let f2 = fma_acc(XRow::new_unchecked(2), YRow::new_unchecked(0), 2);
    let f3 = fma_acc(XRow::new_unchecked(3), YRow::new_unchecked(0), 3);

    let mut p = 0usize;
    while p < k {
        let ay = a_panel.add(p * 64) as u64;
        let bx0 = b0.add(p * bs) as u64;
        let bx1 = (b1.add(p * bs) as u64) | (1u64 << 56);
        let bx2 = (b2.add(p * bs) as u64) | (2u64 << 56);
        let bx3 = (b3.add(p * bs) as u64) | (3u64 << 56);

        // 9 AMX ops in ONE asm block: no LLVM barriers between them.
        core::arch::asm!(
            ".word (0x00201000 + (1 << 5) + 0{ay} - ((0{ay} >> 4) * 6))",
            ".word (0x00201000 + (0 << 5) + 0{bx0} - ((0{bx0} >> 4) * 6))",
            ".word (0x00201000 + (0 << 5) + 0{bx1} - ((0{bx1} >> 4) * 6))",
            ".word (0x00201000 + (0 << 5) + 0{bx2} - ((0{bx2} >> 4) * 6))",
            ".word (0x00201000 + (0 << 5) + 0{bx3} - ((0{bx3} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{f0} - ((0{f0} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{f1} - ((0{f1} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{f2} - ((0{f2} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{f3} - ((0{f3} >> 4) * 6))",
            ay = in(reg) ay,
            bx0 = in(reg) bx0,
            bx1 = in(reg) bx1,
            bx2 = in(reg) bx2,
            bx3 = in(reg) bx3,
            f0 = in(reg) f0,
            f1 = in(reg) f1,
            f2 = in(reg) f2,
            f3 = in(reg) f3,
            options(nostack),
        );
        p += 1;
    }
}

// ---------------------------------------------------------------------------
// 16×32 microkernel — accumulate-only (Z preloaded with C)
// ---------------------------------------------------------------------------

/// AMX 16×32 microkernel that accumulates into existing Z tiles 0-1.
/// Use after `preload_c` for both tiles.
///
/// # Safety
/// AMX must be active. Z tiles must be preloaded. All panels: 64-byte aligned, `k*64` bytes.
#[inline]
pub unsafe fn microkernel_16x32_acc(
    a_panel: *const u8,
    b_left: *const u8,
    b_right: *const u8,
    k: usize,
    bs: usize,
) {
    let mut p = 0usize;

    while p + 4 <= k {
        if p + 8 <= k {
            crate::sync::prefetch::prefetch_l1(a_panel.add((p + 4) * 64));
            crate::sync::prefetch::prefetch_l1(b_left.add((p + 4) * bs));
            crate::sync::prefetch::prefetch_l1(b_right.add((p + 4) * bs));
        }

        for i in 0u8..4 {
            amx_op::<OP_LDY>((a_panel.add((p + i as usize) * 64) as u64) | ((i as u64) << 56));
        }
        for i in 0u8..4 {
            amx_op::<OP_LDX>((b_left.add((p + i as usize) * bs) as u64) | ((i as u64) << 56));
        }
        for i in 0u8..4 {
            amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(i), YRow::new_unchecked(i), 0));
        }

        for i in 0u8..4 {
            amx_op::<OP_LDX>(
                (b_right.add((p + i as usize) * bs) as u64) | (((4 + i) as u64) << 56),
            );
        }
        for i in 0u8..4 {
            amx_op::<OP_FMA32>(fma_acc(
                XRow::new_unchecked(4 + i),
                YRow::new_unchecked(i),
                1,
            ));
        }

        p += 4;
    }

    while p < k {
        amx_op::<OP_LDY>(a_panel.add(p * 64) as u64);
        amx_op::<OP_LDX>(b_left.add(p * bs) as u64);
        amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));

        amx_op::<OP_LDX>((b_right.add(p * bs) as u64) | (1u64 << 56));
        amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(1), YRow::new_unchecked(0), 1));

        p += 1;
    }
}

// ---------------------------------------------------------------------------
// Tile store / accumulate
// ---------------------------------------------------------------------------

/// Store the 16×16 result from Z tile 0 into a row-major f32 buffer.
///
/// # Safety
/// AMX must be active. `dst` must have 1024 writable bytes, 64-byte aligned.
#[inline]
pub unsafe fn store_tile_16x16(dst: *mut u8) {
    for j in 0u8..16 {
        let z_row = j * 4;
        amx_op::<OP_STZ>((dst.add(j as usize * 64) as u64) | ((z_row as u64) << 56));
    }
}

/// Add Z tile contents into existing C buffer. NEON-vectorized.
///
/// # Safety
/// AMX must be active. `c` must point to valid f32 data with stride `ldc`.
#[inline]
pub unsafe fn accumulate_tile(c: *mut f32, ldc: usize, tile: u8) {
    #[repr(align(64))]
    struct A64([f32; 16]);

    let mut zbuf = A64([0f32; 16]);
    let z_ptr = zbuf.0.as_mut_ptr() as *mut u8;

    for j in 0u8..16 {
        let z_row = j * 4 + tile;
        amx_op::<OP_STZ>((z_ptr as u64) | ((z_row as u64) << 56));

        let c_row = c.add(j as usize * ldc);

        #[cfg(target_arch = "aarch64")]
        {
            use core::arch::aarch64::{vaddq_f32, vld1q_f32, vst1q_f32};
            for q in 0..4usize {
                let existing = vld1q_f32(c_row.add(q * 4));
                let z_val = vld1q_f32(zbuf.0.as_ptr().add(q * 4));
                vst1q_f32(c_row.add(q * 4), vaddq_f32(existing, z_val));
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for i in 0..16 {
                *c_row.add(i) += zbuf.0[i];
            }
        }
    }
}

/// Backward-compatible wrapper for tile 0.
///
/// # Safety
/// AMX must be active. `c` must point to valid f32 data with stride `ldc`.
#[inline]
pub unsafe fn accumulate_tile_16x16(c: *mut f32, ldc: usize) {
    accumulate_tile(c, ldc, 0);
}

// ---------------------------------------------------------------------------
// 32×32 GEBP microkernel — individual loads, packed B (stride=64)
// ---------------------------------------------------------------------------

/// AMX 32×32 for GEBP path. Individual LDY + LDX (no pair load).
/// 2 LDY + 2 LDX + 4 FMA = 8 ops/k-step (vs 9 in 16×64).
///
/// # Safety
/// AMX active. Z tiles preloaded. A: 64-byte stride. B: 64-byte stride (packed).
#[inline]
pub unsafe fn microkernel_32x32_gebp(
    a0: *const u8,
    a1: *const u8,
    b0: *const u8,
    b1: *const u8,
    k: usize,
) {
    let f00 = fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0);
    let f10 = fma_acc(XRow::new_unchecked(1), YRow::new_unchecked(0), 1);
    let f01 = fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(1), 2);
    let f11 = fma_acc(XRow::new_unchecked(1), YRow::new_unchecked(1), 3);

    // Unrolled by 2: 16 AMX ops per 2 k-steps in single asm block.
    // Interleave loads/FMA for pipeline utilization.
    let g00 = fma_acc(XRow::new_unchecked(2), YRow::new_unchecked(2), 0);
    let g10 = fma_acc(XRow::new_unchecked(3), YRow::new_unchecked(2), 1);
    let g01 = fma_acc(XRow::new_unchecked(2), YRow::new_unchecked(3), 2);
    let g11 = fma_acc(XRow::new_unchecked(3), YRow::new_unchecked(3), 3);

    let mut p = 0usize;
    while p + 2 <= k {
        let ay0 = a0.add(p * 64) as u64;
        let ay1 = (a1.add(p * 64) as u64) | (1u64 << 56);
        let bx0_v = b0.add(p * 64) as u64;
        let bx1_v = (b1.add(p * 64) as u64) | (1u64 << 56);
        let ay2 = (a0.add((p + 1) * 64) as u64) | (2u64 << 56);
        let ay3 = (a1.add((p + 1) * 64) as u64) | (3u64 << 56);
        let bx2_v = (b0.add((p + 1) * 64) as u64) | (2u64 << 56);
        let bx3_v = (b1.add((p + 1) * 64) as u64) | (3u64 << 56);
        core::arch::asm!(
            // k-step p: load A[0..1], B[0..1], FMA
            ".word (0x00201000 + (1 << 5) + 0{ay0} - ((0{ay0} >> 4) * 6))",
            ".word (0x00201000 + (0 << 5) + 0{bx0} - ((0{bx0} >> 4) * 6))",
            ".word (0x00201000 + (1 << 5) + 0{ay1} - ((0{ay1} >> 4) * 6))",
            ".word (0x00201000 + (0 << 5) + 0{bx1} - ((0{bx1} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{f00} - ((0{f00} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{f10} - ((0{f10} >> 4) * 6))",
            // k-step p+1: load A[2..3], B[2..3] while FMA continues
            ".word (0x00201000 + (1 << 5) + 0{ay2} - ((0{ay2} >> 4) * 6))",
            ".word (0x00201000 + (0 << 5) + 0{bx2} - ((0{bx2} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{f01} - ((0{f01} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{f11} - ((0{f11} >> 4) * 6))",
            ".word (0x00201000 + (1 << 5) + 0{ay3} - ((0{ay3} >> 4) * 6))",
            ".word (0x00201000 + (0 << 5) + 0{bx3} - ((0{bx3} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{g00} - ((0{g00} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{g10} - ((0{g10} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{g01} - ((0{g01} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{g11} - ((0{g11} >> 4) * 6))",
            ay0 = in(reg) ay0,
            ay1 = in(reg) ay1,
            bx0 = in(reg) bx0_v,
            bx1 = in(reg) bx1_v,
            ay2 = in(reg) ay2,
            ay3 = in(reg) ay3,
            bx2 = in(reg) bx2_v,
            bx3 = in(reg) bx3_v,
            f00 = in(reg) f00,
            f10 = in(reg) f10,
            f01 = in(reg) f01,
            f11 = in(reg) f11,
            g00 = in(reg) g00,
            g10 = in(reg) g10,
            g01 = in(reg) g01,
            g11 = in(reg) g11,
            options(nostack),
        );
        p += 2;
    }
    // Remainder: single k-step.
    if p < k {
        amx_op::<OP_LDY>(a0.add(p * 64) as u64);
        amx_op::<OP_LDY>((a1.add(p * 64) as u64) | (1u64 << 56));
        amx_op::<OP_LDX>(b0.add(p * 64) as u64);
        amx_op::<OP_LDX>((b1.add(p * 64) as u64) | (1u64 << 56));
        amx_op::<OP_FMA32>(f00);
        amx_op::<OP_FMA32>(f10);
        amx_op::<OP_FMA32>(f01);
        amx_op::<OP_FMA32>(f11);
    }
}

// ---------------------------------------------------------------------------
// 32×32 microkernel — pair-load, accumulate-only (for small path)
// ---------------------------------------------------------------------------

/// AMX 32×32 pair-load microkernel (reverse-engineered from Accelerate).
///
/// 2×2 Z tile layout. AMX pair loads (bit 62): 128 bytes → 2 registers.
/// **6 ops per k-step**: 1 LDX(pair) + 1 LDY(pair) + 4 FMA.
///
/// `a_pair`: interleaved A pack — for column p: strip0[p] at offset p*128,
///   strip1[p] at p*128+64. Must be 128-byte aligned.
/// `b0`: B row-major (columns 0:32 contiguous). Must be 64-byte aligned.
///   Pair LDX loads cols 0:16 → X[0], cols 16:32 → X[1].
/// `bs`: B row stride in bytes.
///
/// # Safety
/// AMX active. Z tiles preloaded. Data must be properly aligned.
#[inline]
pub unsafe fn microkernel_32x32_acc(
    a_pair: *const u8,
    _a1: *const u8,
    b0: *const u8,
    _b1: *const u8,
    k: usize,
    bs: usize,
) {
    const PAIR: u64 = 1u64 << 62;

    let f00 = fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0);
    let f10 = fma_acc(XRow::new_unchecked(1), YRow::new_unchecked(0), 1);
    let f01 = fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(1), 2);
    let f11 = fma_acc(XRow::new_unchecked(1), YRow::new_unchecked(1), 3);

    // 6 AMX ops per k-step: 1 LDX(pair) + 1 LDY(pair) + 4 FMA.
    // B must be 128-byte aligned, 128 bytes per k-step (32 floats: cols 0:16 + 16:32).
    // A must be 128-byte aligned, 128 bytes per k-step (32 floats: strip0 + strip1).
    let mut p = 0usize;
    while p < k {
        let bx = (b0.add(p * bs) as u64) | PAIR;
        let ay = (a_pair.add(p * 128) as u64) | PAIR;
        core::arch::asm!(
            ".word (0x00201000 + (0 << 5) + 0{bx} - ((0{bx} >> 4) * 6))",
            ".word (0x00201000 + (1 << 5) + 0{ay} - ((0{ay} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{f00} - ((0{f00} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{f10} - ((0{f10} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{f01} - ((0{f01} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{f11} - ((0{f11} >> 4) * 6))",
            bx = in(reg) bx,
            ay = in(reg) ay,
            f00 = in(reg) f00,
            f10 = in(reg) f10,
            f01 = in(reg) f01,
            f11 = in(reg) f11,
            options(nostack),
        );
        p += 1;
    }
}

/// AMX 32×32 pair-load microkernel — first k-block (zeros Z via fma_first).
/// No preload_c needed. Saves 64 LDZ ops per tile-set.
///
/// # Safety
/// AMX active. Z tiles NOT preloaded (will be zeroed by fma_first).
#[inline]
pub unsafe fn microkernel_32x32_first(a_pair: *const u8, b0: *const u8, k: usize, bs: usize) {
    const PAIR: u64 = 1u64 << 62;

    let f00_first = fma_first(XRow::new_unchecked(0), YRow::new_unchecked(0), 0);
    let f10_first = fma_first(XRow::new_unchecked(1), YRow::new_unchecked(0), 1);
    let f01_first = fma_first(XRow::new_unchecked(0), YRow::new_unchecked(1), 2);
    let f11_first = fma_first(XRow::new_unchecked(1), YRow::new_unchecked(1), 3);
    let f00 = fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0);
    let f10 = fma_acc(XRow::new_unchecked(1), YRow::new_unchecked(0), 1);
    let f01 = fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(1), 2);
    let f11 = fma_acc(XRow::new_unchecked(1), YRow::new_unchecked(1), 3);

    // First k-step: fma_first zeros all 4 Z tiles.
    if k > 0 {
        let bx = (b0 as u64) | PAIR;
        let ay = (a_pair as u64) | PAIR;
        core::arch::asm!(
            ".word (0x00201000 + (0 << 5) + 0{bx} - ((0{bx} >> 4) * 6))",
            ".word (0x00201000 + (1 << 5) + 0{ay} - ((0{ay} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{f00} - ((0{f00} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{f10} - ((0{f10} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{f01} - ((0{f01} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{f11} - ((0{f11} >> 4) * 6))",
            bx = in(reg) bx,
            ay = in(reg) ay,
            f00 = in(reg) f00_first,
            f10 = in(reg) f10_first,
            f01 = in(reg) f01_first,
            f11 = in(reg) f11_first,
            options(nostack),
        );
    }
    // Remaining k-steps: fma_acc.
    let mut p = 1usize;
    while p < k {
        let bx = (b0.add(p * bs) as u64) | PAIR;
        let ay = (a_pair.add(p * 128) as u64) | PAIR;
        core::arch::asm!(
            ".word (0x00201000 + (0 << 5) + 0{bx} - ((0{bx} >> 4) * 6))",
            ".word (0x00201000 + (1 << 5) + 0{ay} - ((0{ay} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{f00} - ((0{f00} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{f10} - ((0{f10} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{f01} - ((0{f01} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{f11} - ((0{f11} >> 4) * 6))",
            bx = in(reg) bx, ay = in(reg) ay,
            f00 = in(reg) f00, f10 = in(reg) f10,
            f01 = in(reg) f01, f11 = in(reg) f11,
            options(nostack),
        );
        p += 1;
    }
}

/// AMX 16×16 microkernel — first k-block (zeros Z tile 0 via fma_first).
///
/// # Safety
/// AMX active. Z tile NOT preloaded. A: 64-byte stride. B: `bs`-byte stride.
#[inline]
pub unsafe fn microkernel_16x16_first(a_panel: *const u8, b_panel: *const u8, k: usize, bs: usize) {
    if k > 0 {
        amx_op::<OP_LDX>(b_panel as u64);
        amx_op::<OP_LDY>(a_panel as u64);
        amx_op::<OP_FMA32>(fma_first(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
    }
    let f = fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0);
    let mut p = 1usize;
    while p < k {
        amx_op::<OP_LDX>(b_panel.add(p * bs) as u64);
        amx_op::<OP_LDY>(a_panel.add(p * 64) as u64);
        amx_op::<OP_FMA32>(f);
        p += 1;
    }
}

/// AMX 32×32 pair-load microkernel WITHOUT pair LDX (for unaligned B).
/// 7 ops/k-step: 1 LDY(pair) + 2 LDX + 4 FMA.
///
/// # Safety
/// AMX active. Z tiles preloaded. A: 128-byte aligned interleaved pack.
/// B: 64-byte aligned, `bs`-byte stride.
#[inline]
pub unsafe fn microkernel_32x32_acc_nopairx(a_pair: *const u8, b0: *const u8, k: usize, bs: usize) {
    const PAIR: u64 = 1u64 << 62;

    let f00 = fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0);
    let f10 = fma_acc(XRow::new_unchecked(1), YRow::new_unchecked(0), 1);
    let f01 = fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(1), 2);
    let f11 = fma_acc(XRow::new_unchecked(1), YRow::new_unchecked(1), 3);

    let mut p = 0usize;
    while p < k {
        let ay = (a_pair.add(p * 128) as u64) | PAIR;
        let bx0 = b0.add(p * bs) as u64;
        let bx1 = (b0.add(p * bs + 64) as u64) | (1u64 << 56);
        core::arch::asm!(
            ".word (0x00201000 + (1 << 5) + 0{ay} - ((0{ay} >> 4) * 6))",
            ".word (0x00201000 + (0 << 5) + 0{bx0} - ((0{bx0} >> 4) * 6))",
            ".word (0x00201000 + (0 << 5) + 0{bx1} - ((0{bx1} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{f00} - ((0{f00} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{f10} - ((0{f10} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{f01} - ((0{f01} >> 4) * 6))",
            ".word (0x00201000 + (12 << 5) + 0{f11} - ((0{f11} >> 4) * 6))",
            ay = in(reg) ay,
            bx0 = in(reg) bx0,
            bx1 = in(reg) bx1,
            f00 = in(reg) f00,
            f10 = in(reg) f10,
            f01 = in(reg) f01,
            f11 = in(reg) f11,
            options(nostack),
        );
        p += 1;
    }
}
