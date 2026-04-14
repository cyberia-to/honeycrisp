//! Raw AMX instruction encoding via inline assembly.
//!
//! Apple AMX instructions are encoded as:
//!   .word (0x201000 + (opcode << 5) + operand_register)
//!
//! The operand register trick from yvt/amx-rs converts the LLVM
//! register number into the hardware encoding expected by the `.word`
//! directive without needing a lookup table.

/// Emit a single AMX instruction.
///
/// `OP` is the 5-bit opcode, `operand` is the u64 value loaded into
/// a general-purpose register that the coprocessor reads.
///
/// # Safety
///
/// The caller must ensure the AMX coprocessor has been activated
/// (AMX_SET) before invoking any operation, and that `operand` is
/// a correctly packed operand word for the given opcode.
#[inline(always)]
pub unsafe fn amx_op<const OP: u32>(operand: u64) {
    core::arch::asm!(
        ".word (0x00201000 + ({op} << 5) + 0{operand} - ((0{operand} >> 4) * 6))",
        op = const OP,
        operand = in(reg) operand,
        options(nostack),
    );
}

/// Emit AMX_SET — activate the AMX coprocessor on this thread.
///
/// # Safety
///
/// Must be called exactly once per thread before any AMX operation.
/// Pair with [`amx_clr`].
#[inline(always)]
pub unsafe fn amx_set() {
    core::arch::asm!(
        "nop",
        "nop",
        "nop",
        ".word (0x00201000 + (17 << 5) + 0)",
        options(nostack),
    );
}

/// Emit AMX_CLR — deactivate the AMX coprocessor on this thread.
///
/// # Safety
///
/// Must be called after all AMX work is done. Pair with [`amx_set`].
#[inline(always)]
pub unsafe fn amx_clr() {
    core::arch::asm!(
        "nop",
        "nop",
        "nop",
        ".word (0x00201000 + (17 << 5) + 1)",
        options(nostack),
    );
}

// ---------------------------------------------------------------------------
// Opcode constants — used as const-generic arguments to `amx_op`.
// ---------------------------------------------------------------------------

/// Load 64 bytes into an X register row.
pub const OP_LDX: u32 = 0;
/// Load 64 bytes into a Y register row.
pub const OP_LDY: u32 = 1;
/// Store 64 bytes from an X register row.
pub const OP_STX: u32 = 2;
/// Store 64 bytes from a Y register row.
pub const OP_STY: u32 = 3;
/// Load 64 bytes into a Z register row.
pub const OP_LDZ: u32 = 4;
/// Store 64 bytes from a Z register row.
pub const OP_STZ: u32 = 5;
/// Load 64 bytes into a Z register row (interleaved).
pub const OP_LDZI: u32 = 6;
/// Store 64 bytes from a Z register row (interleaved).
pub const OP_STZI: u32 = 7;

/// Extract horizontal slice (M4+).
pub const OP_EXTRH: u32 = 8;
/// Extract vertical slice (M4+).
pub const OP_EXTRV: u32 = 9;

/// Fused multiply-accumulate, f64.
pub const OP_FMA64: u32 = 10;
/// Fused multiply-subtract, f64.
pub const OP_FMS64: u32 = 11;
/// Fused multiply-accumulate, f32.
pub const OP_FMA32: u32 = 12;
/// Fused multiply-subtract, f32.
pub const OP_FMS32: u32 = 13;
/// Multiply-accumulate, i16.
pub const OP_MAC16: u32 = 14;
/// Fused multiply-accumulate, f16.
pub const OP_FMA16: u32 = 15;
/// Fused multiply-subtract, f16.
pub const OP_FMS16: u32 = 16;
