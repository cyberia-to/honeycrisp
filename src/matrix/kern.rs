//! Pure assembly AMX kernels via global_asm!.
//!
//! These bypass LLVM's register allocator entirely, eliminating
//! spills, reloads, and unnecessary memory barriers between AMX ops.

#[cfg(target_arch = "aarch64")]
core::arch::global_asm!(
    // 32×32 first-k kernel: zeros Z via fma_first, then fma_acc.
    // C ABI: x0=a_pair, x1=b_ptr, x2=k, x3=bs
    // 6 AMX ops/k-step: LDX(pair) + LDY(pair) + 4×FMA32
    ".global _acpu_kern_32x32_first",
    ".p2align 4",
    "_acpu_kern_32x32_first:",
    "cbz x2, 2f",
    // PAIR bit
    "mov x12, #0x4000000000000000",
    // fma_first operands (skip_z bit 27 set)
    "mov w8, #0x8000000", // X[0]×Y[0]→tile0, skip_z
    "mov w9, #0x8110000", // X[1]×Y[0]→tile1, skip_z
    "movz w10, #0x0040",  // X[0]×Y[1]→tile2, skip_z
    "movk w10, #0x0820, lsl #16",
    "movz w11, #0x0040", // X[1]×Y[1]→tile3, skip_z
    "movk w11, #0x0831, lsl #16",
    // First k-step
    "orr x13, x1, x12",
    "orr x14, x0, x12",
    ".word 0x0020100D", // LDX x13
    ".word 0x0020102E", // LDY x14
    ".word 0x00201188", // FMA32 x8
    ".word 0x00201189", // FMA32 x9
    ".word 0x0020118A", // FMA32 x10
    ".word 0x0020118B", // FMA32 x11
    // Switch to fma_acc (no skip_z)
    "mov w8, #0",
    "mov w9, #0x110000",
    "movz w10, #0x0040",
    "movk w10, #0x0020, lsl #16",
    "movz w11, #0x0040",
    "movk w11, #0x0031, lsl #16",
    "sub x2, x2, #1",
    "add x0, x0, #128",
    "add x1, x1, x3",
    "cbz x2, 2f",
    // Steady-state loop
    ".p2align 4",
    "1:",
    "orr x13, x1, x12",
    "orr x14, x0, x12",
    ".word 0x0020100D", // LDX x13
    ".word 0x0020102E", // LDY x14
    ".word 0x00201188", // FMA32 x8
    ".word 0x00201189", // FMA32 x9
    ".word 0x0020118A", // FMA32 x10
    ".word 0x0020118B", // FMA32 x11
    "add x0, x0, #128",
    "add x1, x1, x3",
    "subs x2, x2, #1",
    "b.ne 1b",
    "2:",
    "ret",
    // 32×32 acc kernel: Z tiles must be preloaded.
    // C ABI: x0=a_pair, x1=b_ptr, x2=k, x3=bs
    ".global _acpu_kern_32x32_acc",
    ".p2align 4",
    "_acpu_kern_32x32_acc:",
    "cbz x2, 2f",
    "mov x12, #0x4000000000000000",
    "mov w8, #0",
    "mov w9, #0x110000",
    "movz w10, #0x0040",
    "movk w10, #0x0020, lsl #16",
    "movz w11, #0x0040",
    "movk w11, #0x0031, lsl #16",
    ".p2align 4",
    "1:",
    "orr x13, x1, x12",
    "orr x14, x0, x12",
    ".word 0x0020100D",
    ".word 0x0020102E",
    ".word 0x00201188",
    ".word 0x00201189",
    ".word 0x0020118A",
    ".word 0x0020118B",
    "add x0, x0, #128",
    "add x1, x1, x3",
    "subs x2, x2, #1",
    "b.ne 1b",
    "2:",
    "ret",
);

// FFI declarations for the assembly kernels.
unsafe extern "C" {
    fn acpu_kern_32x32_first(a_pair: *const u8, b_ptr: *const u8, k: usize, bs: usize);
    fn acpu_kern_32x32_acc(a_pair: *const u8, b_ptr: *const u8, k: usize, bs: usize);
}

/// Safe wrapper for the pure-asm first-k kernel.
///
/// # Safety
/// AMX active. a_pair and b_ptr must be 128-byte aligned with valid data.
#[inline]
pub unsafe fn kern_32x32_first(a_pair: *const u8, b_ptr: *const u8, k: usize, bs: usize) {
    acpu_kern_32x32_first(a_pair, b_ptr, k, bs);
}

/// Safe wrapper for the pure-asm acc kernel.
///
/// # Safety
/// AMX active. Z tiles preloaded. a_pair and b_ptr must be 128-byte aligned.
#[inline]
pub unsafe fn kern_32x32_acc(a_pair: *const u8, b_ptr: *const u8, k: usize, bs: usize) {
    acpu_kern_32x32_acc(a_pair, b_ptr, k, bs);
}
