//! SSVE numerical kernels.
//!
//! Wider-vector versions of common f32 vector operations using
//! predicated SSVE inside SME streaming mode. SVL=64 bytes (16 f32
//! lanes per Z register) means each FMLA does 16 ops vs NEON's 4.
//!
//! All entry points open + close their own [`super::Stream`].
//! Callers that already hold a Stream should use the `*_in_stream`
//! variants to avoid the SMSTART/SMSTOP cost (a few hundred picoseconds
//! per call).

use super::Stream;
use crate::{CpuError, Feature};

/// `y[i] += a * x[i]` over the entire slice.
///
/// Requires FEAT_SME. On chips without SME, returns
/// `FeatureNotAvailable(Sme)`.
pub fn axpy_f32(a: f32, x: &[f32], y: &mut [f32]) -> crate::Result<()> {
    assert_eq!(x.len(), y.len());
    if !crate::probe::scan().has_sme {
        return Err(CpuError::FeatureNotAvailable(Feature::Sme));
    }
    let stream = Stream::new()?;
    axpy_f32_in_stream(&stream, a, x, y);
    drop(stream);
    Ok(())
}

/// Streaming-mode-aware AXPY: caller already holds the Stream.
pub fn axpy_f32_in_stream(_stream: &Stream, a: f32, x: &[f32], y: &mut [f32]) {
    let n = x.len();
    if n == 0 {
        return;
    }
    let chunks = n / 16;
    if chunks > 0 {
        unsafe {
            axpy_bulk_asm(a, x.as_ptr(), y.as_mut_ptr(), chunks);
        }
    }
    let tail = chunks * 16;
    for i in tail..n {
        y[i] += a * x[i];
    }
}

#[inline(never)]
unsafe fn axpy_bulk_asm(a: f32, x: *const f32, y: *mut f32, chunks: usize) {
    // Put `a` on the stack so we can broadcast-load it with LD1RW. The
    // alternative — pinning to W4 and using `DUP Z2.S, W4` — depended on
    // LLVM honoring `in("w4")` even when the asm body only references
    // w4 through `.word`-encoded instructions; that turned out to be
    // unreliable.
    let a_local: [f32; 1] = [a];
    core::arch::asm!(
        ".word 0x2598E3E0",          // PTRUE P0.S, all
        // Broadcast a from [x3] into Z2.S.
        "mov x3, {ap}",
        ".word 0x8540C062",          // LD1RW {Z2.S}, P0/Z, [x3]

        "mov x0, {x}",
        "mov x1, {y}",
        "mov x2, {n}",

        "10:",
        ".word 0xA540A000",          // LD1W Z0.S, P0/Z, [x0]   — x slice
        ".word 0xA540A021",          // LD1W Z1.S, P0/Z, [x1]   — y slice
        ".word 0x65A20001",          // FMLA Z1.S, P0/M, Z0.S, Z2.S  — y += x * a
        ".word 0xE540E021",          // ST1W Z1.S, P0, [x1]
        "add x0, x0, #64",
        "add x1, x1, #64",
        "subs x2, x2, #1",
        "b.ne 10b",

        ap = in(reg) a_local.as_ptr(),
        x = in(reg) x,
        y = in(reg) y,
        n = in(reg) chunks,
        out("x0") _,
        out("x1") _,
        out("x2") _,
        out("x3") _,
        // Streaming-mode SSVE clobbers Z/P state. Declare the lower-128
        // V registers we overlap (Z0/Z1/Z2) as clobbered so LLVM doesn't
        // assume their values survive the asm block.
        out("v0") _,
        out("v1") _,
        out("v2") _,
        options(nostack),
    );
}

// dot_f32 was tried as a Phase 4 surface but the FMLA-accumulate +
// FADDV-reduce path returned 0 in our test environment — likely an
// encoding mismatch on the reduction or a stream-mode subtlety yet to
// isolate. Deferred to Phase 4.5; axpy is sufficient demonstration of
// the wider-vector SSVE path.

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_axpy(a: f32, x: &[f32], y: &mut [f32]) {
        for (xi, yi) in x.iter().zip(y.iter_mut()) {
            *yi += a * *xi;
        }
    }

    fn check_axpy(n: usize) {
        if !crate::probe::scan().has_sme {
            return;
        }
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let mut y_sme: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let mut y_ref = y_sme.clone();
        let a = 1.25f32;

        axpy_f32(a, &x, &mut y_sme).unwrap();
        ref_axpy(a, &x, &mut y_ref);

        let mut max_err = 0.0f32;
        for i in 0..n {
            let e = (y_sme[i] - y_ref[i]).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(max_err < 1e-4, "axpy n={n}: max_err={max_err}");
    }

    #[test]
    fn axpy_16() {
        check_axpy(16);
    }
    #[test]
    fn axpy_64() {
        check_axpy(64);
    }
    #[test]
    fn axpy_1024() {
        check_axpy(1024);
    }
    #[test]
    fn axpy_odd() {
        check_axpy(73);
    }
}
