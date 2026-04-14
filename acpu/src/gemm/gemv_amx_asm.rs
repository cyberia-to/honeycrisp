//! Pure assembly AMX GEMV — zero per-instruction overhead.
//!
//! _acpu_gemv_amx(c, a_bcast, b, n, k, n_tiles):
//! All AMX instructions as raw .word, no Rust amx_op wrapper.
//! K-blocked by 8: Y0-Y7 preloaded, pair LDX, vector FMA.

#[cfg(target_arch = "aarch64")]
core::arch::global_asm!(
    ".text",
    ".global _acpu_gemv_amx",
    ".p2align 4",
    // x0 = c ptr (output, n f32)
    // x1 = a_bcast ptr (k × 64 bytes, each A[ki] broadcast to 16 f32, 128B aligned)
    // x2 = b ptr (k × n f32, row-major)
    // x3 = n (columns)
    // x4 = k (rows)
    // x5 = n_tiles (n / 16)
    "_acpu_gemv_amx:",
    "stp x29, x30, [sp, #-96]!",
    "stp x19, x20, [sp, #16]",
    "stp x21, x22, [sp, #32]",
    "stp x23, x24, [sp, #48]",
    "stp x25, x26, [sp, #64]",
    "stp x27, x28, [sp, #80]",
    "mov x29, sp",
    // Constants
    "lsl x6, x3, #2",               // n_bytes = n * 4 (B row stride)
    "mov x19, x0",                  // c_base
    "mov x20, x1",                  // a_bcast_base
    "mov x21, x2",                  // b_base
    "mov x22, x5",                  // n_tiles
    "lsr x23, x4, #3",              // k8 = k / 8
    "and x24, x4, #7",              // k_rem = k % 8
    "lsr x25, x5, #1",              // pair_tiles = n_tiles / 2
    "mov x26, #0x4000000000000000", // PAIR bit for LDX
    // Encode: OP_LDX=0, OP_LDY=1, OP_STZ=5, OP_FMA32=12
    // .word = 0x201000 + (op << 5) + reg
    // LDX  = 0x201000 + (0  << 5) + reg = 0x201000 + reg
    // LDY  = 0x201000 + (1  << 5) + reg = 0x201020 + reg
    // STZ  = 0x201000 + (5  << 5) + reg = 0x2010A0 + reg
    // FMA32= 0x201000 + (12 << 5) + reg = 0x201180 + reg

    // First K-step flag in x27 (1=first, 0=accumulate)
    "mov x27, #1",
    // ── K-block loop (8 K-steps per block) ──
    "cbz x23, 50f", // skip if k8 == 0
    "mov x28, #0",  // kb counter
    ".p2align 4",
    "10:",
    // x7 = a_bcast + kb*8*64
    "add x7, x20, x28, lsl #9", // kb * 512 = kb << 9
    // Preload Y0-Y7. Row bits in 62:56. Use movz to build row constant.
    "mov x8, x7",
    ".word 0x00201028", // LDY x8 → Y0
    "add x8, x7, #64",
    "movz x9, #1, lsl #48",
    "lsl x9, x9, #8", // x9 = 1 << 56
    "orr x8, x8, x9",
    ".word 0x00201028", // LDY x8 → Y1
    "add x8, x7, #128",
    "add x9, x9, x9", // x9 = 2 << 56
    "orr x8, x8, x9",
    ".word 0x00201028", // LDY x8 → Y2
    "movz x9, #3, lsl #48",
    "lsl x9, x9, #8", // x9 = 3 << 56
    "add x8, x7, #192",
    "orr x8, x8, x9",
    ".word 0x00201028", // LDY x8 → Y3
    "movz x9, #4, lsl #48",
    "lsl x9, x9, #8",
    "add x8, x7, #256",
    "orr x8, x8, x9",
    ".word 0x00201028", // LDY x8 → Y4
    "movz x9, #5, lsl #48",
    "lsl x9, x9, #8",
    "add x8, x7, #320",
    "orr x8, x8, x9",
    ".word 0x00201028", // LDY x8 → Y5
    "movz x9, #6, lsl #48",
    "lsl x9, x9, #8",
    "add x8, x7, #384",
    "orr x8, x8, x9",
    ".word 0x00201028", // LDY x8 → Y6
    "movz x9, #7, lsl #48",
    "lsl x9, x9, #8",
    "add x8, x7, #448",
    "orr x8, x8, x9",
    ".word 0x00201028", // LDY x8 → Y7
    // Now process 8 sub-K steps
    // x9 = b_row base = b_base + (kb*8) * n * 4
    "lsl x9, x28, #3", // kb * 8
    "mul x9, x9, x6",  // * n_bytes
    "add x9, x21, x9", // b_base + offset
    "mov x10, #0",     // sub_k counter
    "20:",
    // For this sub_k: sweep all tiles
    // x11 = b_row for this k = x9 + sub_k * n_bytes
    "mul x11, x10, x6",
    "add x11, x9, x11",
    // Build FMA operand base:
    // vector mode (bit 63) | y_offset = sub_k * 64 (bits 8:0) | x_offset = 0 (bits 18:10)
    // For first_k: add skip_z (bit 27)
    "lsl x12, x10, #6",                  // y_offset = sub_k * 64
    "orr x12, x12, #0x8000000000000000", // vector mode bit 63
    // x13 = fma_operand with skip_z (first)
    "orr x13, x12, #0x0000000008000000", // skip_z bit 27
    // Pair tile loop
    "mov x14, #0",  // pair_tile counter
    "cbz x25, 35f", // skip pairs if none
    "30:",
    // LDX pair: load B[ki, pt*32..pt*32+32] into X0+X1
    "lsl x15, x14, #7",  // pt * 128 bytes
    "add x15, x11, x15", // b_row + offset
    "orr x15, x15, x26", // add PAIR bit
    ".word 0x0020100F",  // LDX x15
    // FMA for tile0: Z[pt*2] += X0 * Y[sub_k]
    // z_row = pt*2, bits 25:20
    "lsl x16, x14, #1",  // pt * 2 = tile0
    "lsl x16, x16, #20", // shift to bits 25:20
    "cbz x27, 31f",      // branch on first_k
    "orr x15, x13, x16", // fma_first + z_row
    ".word 0x0020118F",  // FMA32 x15
    // tile1: z_row = pt*2+1, x_offset = 64 (X1)
    "add x16, x16, #0x100000", // z_row + 1 << 20
    "orr x16, x16, #0x10000",  // x_offset = 64 << 10 = 0x10000
    "orr x15, x13, x16",
    ".word 0x0020118F", // FMA32 x15
    "b 32f",
    "31:",
    "orr x15, x12, x16", // fma_acc + z_row
    ".word 0x0020118F",  // FMA32 x15
    "add x16, x16, #0x100000",
    "orr x16, x16, #0x10000",
    "orr x15, x12, x16",
    ".word 0x0020118F", // FMA32 x15
    "32:",
    "mov x27, #0", // first_k = false
    "add x14, x14, #1",
    "cmp x14, x25",
    "b.lt 30b",
    "35:",
    // Odd tile remainder (if n_tiles is odd)
    "tst x22, #1",
    "b.eq 36f",
    "lsl x15, x25, #7", // pair_tiles * 128
    "add x15, x11, x15",
    ".word 0x0020100F",  // LDX x15 (single, no pair)
    "lsl x16, x25, #21", // z_row = pair_tiles*2 << 20... actually n_tiles-1
    "sub x16, x22, #1",
    "lsl x16, x16, #20",
    "cbz x27, 37f",
    "orr x15, x13, x16",
    ".word 0x0020118F",
    "mov x27, #0",
    "b 36f",
    "37:",
    "orr x15, x12, x16",
    ".word 0x0020118F",
    "36:",
    "add x10, x10, #1",
    "cmp x10, #8",
    "b.lt 20b",
    "add x28, x28, #1",
    "cmp x28, x23",
    "b.lt 10b",
    "50:",
    // TODO: K remainder (skip for now, handled by caller)

    // Store Z rows → C
    // STZ: operand = addr | (z_row << 56)
    "mov x14, #0",
    "cbz x22, 60f",
    "55:",
    "lsl x15, x14, #6",  // tile * 64 bytes offset in C
    "add x15, x19, x15", // c_base + offset
    "lsl x16, x14, #56", // z_row << 56
    "orr x15, x15, x16",
    ".word 0x002010AF", // STZ x15
    "add x14, x14, #1",
    "cmp x14, x22",
    "b.lt 55b",
    "60:",
    "ldp x27, x28, [sp, #80]",
    "ldp x25, x26, [sp, #64]",
    "ldp x23, x24, [sp, #48]",
    "ldp x21, x22, [sp, #32]",
    "ldp x19, x20, [sp, #16]",
    "ldp x29, x30, [sp], #96",
    "ret",
);

#[cfg(target_arch = "aarch64")]
extern "C" {
    fn acpu_gemv_amx(
        c: *mut f32,
        a_bcast: *const u8,
        b: *const f32,
        n: usize,
        k: usize,
        n_tiles: usize,
    );
}

/// Pure-asm AMX GEMV. Pre-broadcasts A, then calls asm kernel.
#[cfg(target_arch = "aarch64")]
pub fn gemv_pure_asm(a: &[f32], b: &[f32], c: &mut [f32], n: usize, k: usize) {
    debug_assert!(k % 8 == 0, "gemv_pure_asm requires k divisible by 8");

    crate::gemm::ensure_amx();

    let n_tiles = n / 16;
    let layout = std::alloc::Layout::from_size_align(k * 64, 128).unwrap();
    let a_ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    for ki in 0..k {
        let val = a[ki];
        let dst = unsafe { std::slice::from_raw_parts_mut(a_ptr.add(ki * 64) as *mut f32, 16) };
        for s in dst.iter_mut() {
            *s = val;
        }
    }

    unsafe {
        acpu_gemv_amx(c.as_mut_ptr(), a_ptr, b.as_ptr(), n, k, n_tiles);
        std::alloc::dealloc(a_ptr, layout);
    }
}
