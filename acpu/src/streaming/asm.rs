//! Raw SME / SSVE instruction encoding via inline assembly.
//!
//! Stable Rust 1.95 does not let crates enable +sme target features, so
//! every SME / SSVE instruction is emitted via `.word` in inline asm —
//! the same trick used for the undocumented AMX path.
//!
//! Encodings verified against `llvm-mc -mattr=+sme,+sme2 -show-encoding`.

// ---------------------------------------------------------------------------
// Streaming-mode lifecycle
// ---------------------------------------------------------------------------

/// SMSTART (full): set PSTATE.SM = 1 and PSTATE.ZA = 1.
///
/// # Safety
///
/// FEAT_SME must be present on the running CPU. On chips without SME,
/// this is an illegal instruction and will SIGILL. Probe via
/// [`crate::probe::scan`] before calling.
#[inline(always)]
pub unsafe fn smstart() {
    core::arch::asm!(".word 0xD503477F", options(nostack));
}

/// SMSTOP (full): clear PSTATE.SM and PSTATE.ZA.
///
/// # Safety
///
/// Must follow an SMSTART on the same thread. Calling SMSTOP outside
/// streaming mode is a no-op on PSTATE but still a privileged form of
/// MSR; on FEAT_SME chips it is always legal.
#[inline(always)]
pub unsafe fn smstop() {
    core::arch::asm!(".word 0xD503467F", options(nostack));
}

/// SMSTART SM: enter streaming SVE mode without enabling ZA.
#[inline(always)]
pub unsafe fn smstart_sm() {
    core::arch::asm!(".word 0xD503437F", options(nostack));
}

/// SMSTOP SM: leave streaming SVE mode.
#[inline(always)]
pub unsafe fn smstop_sm() {
    core::arch::asm!(".word 0xD503427F", options(nostack));
}

/// SMSTART ZA: enable the ZA matrix accumulator.
#[inline(always)]
pub unsafe fn smstart_za() {
    core::arch::asm!(".word 0xD503457F", options(nostack));
}

/// SMSTOP ZA: disable the ZA matrix accumulator.
#[inline(always)]
pub unsafe fn smstop_za() {
    core::arch::asm!(".word 0xD503447F", options(nostack));
}

// ---------------------------------------------------------------------------
// RDSVL — read streaming vector length
// ---------------------------------------------------------------------------

/// `RDSVL X0, #1` — returns SVL in bytes via X0.
///
/// Encoded as `.word 0x04BF5820`. Only valid inside streaming mode.
///
/// # Safety
///
/// Caller must already be inside streaming mode (i.e. a live `Stream`).
/// Clobbers X0; the value flows out through inline asm's `out("x0")`.
#[inline(always)]
pub unsafe fn rdsvl_bytes() -> u64 {
    let svl: u64;
    core::arch::asm!(
        ".word 0x04BF5820",
        out("x0") svl,
        options(nostack, pure, readonly),
    );
    svl
}

// ---------------------------------------------------------------------------
// ZERO {ZA}
// ---------------------------------------------------------------------------

/// `ZERO {ZA}` — clear the entire ZA accumulator.
///
/// Encoded as `.word 0xC00800FF`. PSTATE.ZA must be 1.
///
/// # Safety
///
/// Caller must be inside streaming mode with ZA enabled.
#[inline(always)]
pub unsafe fn zero_za() {
    core::arch::asm!(".word 0xC00800FF", options(nostack));
}
