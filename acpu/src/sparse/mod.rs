//! Sparse operations: CSR matvec and Chebyshev heat-kernel approximation.
//!
//! CSR matvec: y = A·x where A is sparse CSR. For the bostrom graph (avg
//! degree ~1.85) most rows are 1–3 entries — AMX offers no advantage here.
//! NEON prefetch hides gather latency from the scattered x[col_idx[j]] loads.

pub mod chebyshev;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// y += A·x  (accumulate into y).
///
/// A in CSR format: row i spans col_idx[row_ptr[i]..row_ptr[i+1]]
/// with corresponding weights values[...].
pub fn csr_matvec(row_ptr: &[u32], col_idx: &[u32], values: &[f32], x: &[f32], y: &mut [f32]) {
    let n = row_ptr.len().saturating_sub(1);
    debug_assert_eq!(y.len(), n);
    debug_assert_eq!(col_idx.len(), values.len());
    inner(row_ptr, col_idx, values, x, y, n, false);
}

/// y = A·x  (set; zeroes y first).
pub fn csr_matvec_set(row_ptr: &[u32], col_idx: &[u32], values: &[f32], x: &[f32], y: &mut [f32]) {
    let n = row_ptr.len().saturating_sub(1);
    debug_assert_eq!(y.len(), n);
    debug_assert_eq!(col_idx.len(), values.len());
    y.fill(0.0);
    inner(row_ptr, col_idx, values, x, y, n, true);
}

// ── inner dispatch ────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
fn inner(
    row_ptr: &[u32],
    col_idx: &[u32],
    values: &[f32],
    x: &[f32],
    y: &mut [f32],
    n: usize,
    _set: bool,
) {
    // Software prefetch distance: fetch 8 entries ahead.
    const PF: usize = 8;

    unsafe {
        for i in 0..n {
            let s = row_ptr[i] as usize;
            let e = row_ptr[i + 1] as usize;
            let len = e - s;
            if len == 0 {
                continue;
            }

            let cols = &col_idx[s..e];
            let vals = &values[s..e];
            let mut acc = 0.0f32;

            // Prefetch the x entries we'll need for this row.
            for pf in 0..len.min(PF) {
                let ptr = x.as_ptr().add(cols[pf] as usize);
                // PRFM PLDL1KEEP — bring into L1 dcache.
                core::arch::asm!(
                    "prfm pldl1keep, [{ptr}]",
                    ptr = in(reg) ptr,
                    options(nostack, readonly)
                );
            }

            let mut j = 0usize;
            // 4-wide NEON FMA when row is long enough to amortise setup.
            while j + 4 <= len {
                // Prefetch 8 ahead of current j.
                if j + 4 + PF < len {
                    for pf in 0..4 {
                        let ptr = x.as_ptr().add(cols[j + 4 + pf] as usize);
                        core::arch::asm!(
                            "prfm pldl1keep, [{ptr}]",
                            ptr = in(reg) ptr,
                            options(nostack, readonly)
                        );
                    }
                }

                // Gather 4 x values (no hardware gather on ARM; scalar loads).
                let xarr = [
                    *x.get_unchecked(cols[j] as usize),
                    *x.get_unchecked(cols[j + 1] as usize),
                    *x.get_unchecked(cols[j + 2] as usize),
                    *x.get_unchecked(cols[j + 3] as usize),
                ];
                let xv = vld1q_f32(xarr.as_ptr());
                let wv = vld1q_f32(vals.as_ptr().add(j));
                acc += vaddvq_f32(vmulq_f32(xv, wv));
                j += 4;
            }
            // Scalar tail.
            while j < len {
                acc += vals[j] * *x.get_unchecked(cols[j] as usize);
                j += 1;
            }

            *y.get_unchecked_mut(i) += acc;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn inner(
    row_ptr: &[u32],
    col_idx: &[u32],
    values: &[f32],
    x: &[f32],
    y: &mut [f32],
    n: usize,
    _set: bool,
) {
    for i in 0..n {
        let s = row_ptr[i] as usize;
        let e = row_ptr[i + 1] as usize;
        let mut acc = 0.0f32;
        for j in s..e {
            acc += values[j] * x[col_idx[j] as usize];
        }
        y[i] += acc;
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build CSR for a simple 4-node undirected path: 0-1-2-3.
    fn path4() -> (Vec<u32>, Vec<u32>, Vec<f32>) {
        // row_ptr, col_idx, values
        let row_ptr = vec![0, 1, 3, 5, 6];
        let col_idx = vec![1, 0, 2, 1, 3, 2];
        let values = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        (row_ptr, col_idx, values)
    }

    #[test]
    fn matvec_path4_accumulate() {
        let (rp, ci, v) = path4();
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let mut y = [0.0f32; 4];
        csr_matvec(&rp, &ci, &v, &x, &mut y);
        // y[0] = x[1] = 2
        // y[1] = x[0]+x[2] = 4
        // y[2] = x[1]+x[3] = 6
        // y[3] = x[2] = 3
        assert!((y[0] - 2.0).abs() < 1e-5, "y[0]={}", y[0]);
        assert!((y[1] - 4.0).abs() < 1e-5, "y[1]={}", y[1]);
        assert!((y[2] - 6.0).abs() < 1e-5, "y[2]={}", y[2]);
        assert!((y[3] - 3.0).abs() < 1e-5, "y[3]={}", y[3]);
    }

    #[test]
    fn matvec_set_zeroes_first() {
        let (rp, ci, v) = path4();
        let x = [1.0f32; 4];
        let mut y = [99.0f32; 4]; // pre-filled with garbage
        csr_matvec_set(&rp, &ci, &v, &x, &mut y);
        // Every node in a path has degree ≤ 2 and x=1, so y[i] = degree[i].
        assert!((y[0] - 1.0).abs() < 1e-5);
        assert!((y[1] - 2.0).abs() < 1e-5);
        assert!((y[2] - 2.0).abs() < 1e-5);
        assert!((y[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn matvec_long_row() {
        // Star graph: node 0 connected to 1..16 (tests NEON 4-wide path).
        let n = 17usize;
        let mut row_ptr = vec![0u32; n + 1];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        // Node 0: neighbors 1..16
        for j in 1..n {
            col_idx.push(j as u32);
            values.push(1.0f32);
        }
        row_ptr[1] = (n - 1) as u32;
        // Nodes 1..16: single neighbor = 0
        for i in 1..n {
            col_idx.push(0u32);
            values.push(1.0f32);
            row_ptr[i + 1] = row_ptr[i] + 1;
        }
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut y = vec![0.0f32; n];
        csr_matvec(&row_ptr, &col_idx, &values, &x, &mut y);
        // y[0] = 1+2+...+16 = 136
        let expected0: f32 = (1..n).map(|i| i as f32).sum();
        assert!(
            (y[0] - expected0).abs() < 1e-2,
            "y[0]={} expected={}",
            y[0],
            expected0
        );
        // y[i] = x[0] = 0 for i in 1..n
        for i in 1..n {
            assert!(y[i].abs() < 1e-5, "y[{i}]={}", y[i]);
        }
    }
}
