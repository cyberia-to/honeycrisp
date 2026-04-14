//! Hand-written assembly exp kernel — bypasses LLVM register allocator
//! for optimal NEON pipeline scheduling.
//!
//! C ABI: _acpu_exp_f32(dst: *mut f32, src: *const f32, n: usize)
//! Processes 16 f32 per iteration. Caller handles remainder.

#[cfg(target_arch = "aarch64")]
core::arch::global_asm!(
    // Constants (will be loaded from literal pool)
    ".section __DATA,__const",
    ".p2align 4",
    "_acpu_exp_consts:",
    ".float 88.37626",   // EXP_HI
    ".float -87.33655",  // EXP_LO
    ".float 1.4426950",  // 1/ln2
    ".float 0.69314718", // ln2
    ".float 1.0",        // c0 = 1
    ".float 1.0",        // c1 = 1
    ".float 0.5000001",  // c2
    ".float 0.1666656",  // c3
    ".float 0.0416756",  // c4
    ".float 0.0083716",  // c5
    ".float 0.0",        // padding
    ".float 0.0",        // padding
    ".text",
    ".global _acpu_exp_f32",
    ".p2align 4",
    // x0 = dst, x1 = src, x2 = n (count of f32, must be multiple of 16)
    "_acpu_exp_f32:",
    // Save callee-saved NEON regs v8-v15
    "stp d8, d9, [sp, #-64]!",
    "stp d10, d11, [sp, #16]",
    "stp d12, d13, [sp, #32]",
    "stp d14, d15, [sp, #48]",
    // Load constants into v24-v31 (callee-saved area, we saved them)
    // Actually v8-v15 are callee-saved, v16-v31 are caller-saved.
    // Use v24-v29 for constants (caller-saved, no need to save).
    "adrp x3, _acpu_exp_consts@PAGE",
    "add x3, x3, _acpu_exp_consts@PAGEOFF",
    "ld1r {{v24.4s}}, [x3]", // v24 = EXP_HI
    "add x3, x3, #4",
    "ld1r {{v25.4s}}, [x3]", // v25 = EXP_LO
    "add x3, x3, #4",
    "ld1r {{v26.4s}}, [x3]", // v26 = 1/ln2
    "add x3, x3, #4",
    "ld1r {{v27.4s}}, [x3]", // v27 = ln2
    "add x3, x3, #4",
    // c0..c5
    "ld1r {{v28.4s}}, [x3]", // v28 = c0 = 1.0
    "add x3, x3, #4",
    // c1 = 1.0, same as c0
    "add x3, x3, #4",
    "ld1r {{v29.4s}}, [x3]", // v29 = c2
    "add x3, x3, #4",
    "ld1r {{v30.4s}}, [x3]", // v30 = c3
    "add x3, x3, #4",
    "ld1r {{v8.4s}}, [x3]", // v8 = c4  (callee-saved, we saved it)
    "add x3, x3, #4",
    "ld1r {{v9.4s}}, [x3]", // v9 = c5  (callee-saved, we saved it)
    // v10 = 127 as int for 2^n
    "movi v10.4s, #127",
    // Loop: process 16 floats per iteration
    "cmp x2, #16",
    "b.lt 2f",
    ".p2align 4",
    "1:",
    // Load 16 floats
    "ldp q0, q1, [x1]",
    "ldp q2, q3, [x1, #32]",
    // Clamp: x = max(min(x, HI), LO)
    "fmin v0.4s, v0.4s, v24.4s",
    "fmin v1.4s, v1.4s, v24.4s",
    "fmin v2.4s, v2.4s, v24.4s",
    "fmin v3.4s, v3.4s, v24.4s",
    "fmax v0.4s, v0.4s, v25.4s",
    "fmax v1.4s, v1.4s, v25.4s",
    "fmax v2.4s, v2.4s, v25.4s",
    "fmax v3.4s, v3.4s, v25.4s",
    // n = round(x / ln2)
    "fmul v4.4s, v0.4s, v26.4s",
    "fmul v5.4s, v1.4s, v26.4s",
    "fmul v6.4s, v2.4s, v26.4s",
    "fmul v7.4s, v3.4s, v26.4s",
    "frintn v4.4s, v4.4s",
    "frintn v5.4s, v5.4s",
    "frintn v6.4s, v6.4s",
    "frintn v7.4s, v7.4s",
    // r = x - n * ln2  (via FMS: r = x - n*ln2)
    "fmls v0.4s, v4.4s, v27.4s",
    "fmls v1.4s, v5.4s, v27.4s",
    "fmls v2.4s, v6.4s, v27.4s",
    "fmls v3.4s, v7.4s, v27.4s",
    // Now v0-v3 = r, v4-v7 = n

    // Estrin's scheme: p(r) = (c0 + c1*r) + r²*(c2 + c3*r) + r⁴*(c4 + c5*r)
    // r²
    "fmul v11.4s, v0.4s, v0.4s",
    "fmul v12.4s, v1.4s, v1.4s",
    "fmul v13.4s, v2.4s, v2.4s",
    "fmul v14.4s, v3.4s, v3.4s",
    // p01 = c0 + c1*r = 1 + r  (c0=c1=1, so p01 = 1+r = c0 + r)
    // Use fadd since c1=1: p01 = 1 + r
    "fadd v15.4s, v28.4s, v0.4s",
    "fadd v16.4s, v28.4s, v1.4s",
    "fadd v17.4s, v28.4s, v2.4s",
    "fadd v18.4s, v28.4s, v3.4s",
    // p23 = c2 + c3*r
    "mov v19.16b, v29.16b",
    "mov v20.16b, v29.16b",
    "mov v21.16b, v29.16b",
    "mov v22.16b, v29.16b",
    "fmla v19.4s, v30.4s, v0.4s",
    "fmla v20.4s, v30.4s, v1.4s",
    "fmla v21.4s, v30.4s, v2.4s",
    "fmla v22.4s, v30.4s, v3.4s",
    // p45 = c4 + c5*r
    "mov v23.16b, v8.16b",
    // reuse v24-like scratch — but v24 holds EXP_HI! Use v31.
    "mov v31.16b, v8.16b",
    // Actually we need 4 copies. Use p45_0=v23, p45_1=v31, p45_2=scratch, p45_3=scratch
    // Problem: running out of registers. Let me restructure.
    // Do p45 in-place using v8 copies:
    "fmul v23.4s, v9.4s, v0.4s",  // c5*r0
    "fadd v23.4s, v23.4s, v8.4s", // c4 + c5*r0
    "fmul v31.4s, v9.4s, v1.4s",
    "fadd v31.4s, v31.4s, v8.4s",
    // r⁴
    "fmul v0.4s, v11.4s, v11.4s", // r⁴_0 (reuse v0, r no longer needed)
    "fmul v1.4s, v12.4s, v12.4s", // r⁴_1
    // Combine: p = p01 + p23*r² + p45*r⁴
    "fmla v15.4s, v19.4s, v11.4s", // p01_0 += p23_0 * r²_0
    "fmla v16.4s, v20.4s, v12.4s", // p01_1 += p23_1 * r²_1
    "fmla v15.4s, v23.4s, v0.4s",  // += p45_0 * r⁴_0
    "fmla v16.4s, v31.4s, v1.4s",  // += p45_1 * r⁴_1
    // Now do p45 for lanes 2,3
    "fmul v23.4s, v9.4s, v2.4s",
    "fadd v23.4s, v23.4s, v8.4s",
    "fmul v31.4s, v9.4s, v3.4s",
    "fadd v31.4s, v31.4s, v8.4s",
    "fmul v2.4s, v13.4s, v13.4s", // r⁴_2
    "fmul v3.4s, v14.4s, v14.4s", // r⁴_3
    "fmla v17.4s, v21.4s, v13.4s",
    "fmla v18.4s, v22.4s, v14.4s",
    "fmla v17.4s, v23.4s, v2.4s",
    "fmla v18.4s, v31.4s, v3.4s",
    // 2^n: convert n to int, add 127, shift left 23
    "fcvtns v4.4s, v4.4s",
    "fcvtns v5.4s, v5.4s",
    "fcvtns v6.4s, v6.4s",
    "fcvtns v7.4s, v7.4s",
    "add v4.4s, v4.4s, v10.4s",
    "add v5.4s, v5.4s, v10.4s",
    "add v6.4s, v6.4s, v10.4s",
    "add v7.4s, v7.4s, v10.4s",
    "shl v4.4s, v4.4s, #23",
    "shl v5.4s, v5.4s, #23",
    "shl v6.4s, v6.4s, #23",
    "shl v7.4s, v7.4s, #23",
    // result = p * 2^n
    "fmul v15.4s, v15.4s, v4.4s",
    "fmul v16.4s, v16.4s, v5.4s",
    "fmul v17.4s, v17.4s, v6.4s",
    "fmul v18.4s, v18.4s, v7.4s",
    // Store 16 floats
    "stp q15, q16, [x0]",
    "stp q17, q18, [x0, #32]",
    // Advance pointers
    "add x0, x0, #64",
    "add x1, x1, #64",
    "sub x2, x2, #16",
    "cmp x2, #16",
    "b.ge 1b",
    "2:",
    // Restore callee-saved NEON regs
    "ldp d14, d15, [sp, #48]",
    "ldp d12, d13, [sp, #32]",
    "ldp d10, d11, [sp, #16]",
    "ldp d8, d9, [sp], #64",
    "ret",
);

#[cfg(target_arch = "aarch64")]
extern "C" {
    /// Hand-scheduled NEON exp kernel.
    /// Processes floor(n/16)*16 elements from src to dst.
    /// Caller must handle remainder.
    fn acpu_exp_f32(dst: *mut f32, src: *const f32, n: usize);
}

/// Process `n` elements (must be multiple of 16) using hand-written asm.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn exp_asm(dst: *mut f32, src: *const f32, n: usize) {
    acpu_exp_f32(dst, src, n);
}
