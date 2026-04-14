//! Shared benchmark infrastructure.

#![allow(dead_code)]

use std::time::Instant;

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

pub fn med(t: &mut [u64]) -> u64 {
    t.sort_unstable();
    let n = t.len();
    if n == 0 {
        return 0;
    }
    if n % 2 == 1 {
        t[n / 2]
    } else {
        (t[n / 2 - 1] + t[n / 2]) / 2
    }
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

pub fn ns<F: FnMut()>(mut f: F) -> u64 {
    f();
    let dl = std::time::Instant::now() + std::time::Duration::from_secs(2);
    let mut t = Vec::with_capacity(200);
    for _ in 0..200 {
        if std::time::Instant::now() > dl {
            break;
        }
        let s = std::time::Instant::now();
        f();
        t.push(s.elapsed().as_nanos() as u64);
    }
    if t.is_empty() {
        return u64::MAX;
    }
    med(&mut t)
}

pub fn best_of<F: FnMut()>(mut f: F, iters: usize) -> u64 {
    let mut best = u64::MAX;
    for _ in 0..iters {
        let s = Instant::now();
        f();
        let elapsed = s.elapsed().as_nanos() as u64;
        if elapsed < best {
            best = elapsed;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Scoreboard
// ---------------------------------------------------------------------------

pub struct Score {
    pub wins: u32,
    pub ties: u32,
    pub total: u32,
    pub col_b: &'static str,
}

impl Score {
    pub fn new() -> Self {
        Self {
            wins: 0,
            ties: 0,
            total: 0,
            col_b: "apple",
        }
    }

    /// Create with custom comparison column name.
    pub fn vs(baseline: &'static str) -> Self {
        Self {
            wins: 0,
            ties: 0,
            total: 0,
            col_b: baseline,
        }
    }

    pub fn hdr(&self, section: &str) {
        println!();
        println!("--- {} ---", section);
        println!(
            "{:<28} {:>12} {:>12} {:>8}",
            "operation", "acpu", self.col_b, "speedup"
        );
    }

    pub fn row(&mut self, op: &str, acpu_ns: u64, apple_ns: u64) {
        self.total += 1;
        let ratio = apple_ns as f64 / acpu_ns as f64;
        let marker = if ratio > 1.05 {
            self.wins += 1;
            " \u{2190}"
        } else if ratio >= 0.95 {
            self.ties += 1;
            "  \u{2248}"
        } else {
            ""
        };
        println!(
            "{:<28} {:>10} ns {:>10} ns {:>7.2}x{}",
            op, acpu_ns, apple_ns, ratio, marker
        );
    }

    pub fn row_gf(&mut self, op: &str, acpu_gf: f64, apple_gf: f64) {
        self.total += 1;
        let ratio = acpu_gf / apple_gf;
        let marker = if ratio > 1.05 {
            self.wins += 1;
            " \u{2190}"
        } else if ratio >= 0.95 {
            self.ties += 1;
            "  \u{2248}"
        } else {
            ""
        };
        println!(
            "{:<28} {:>8.1} GF {:>8.1} GF {:>7.2}x{}",
            op, acpu_gf, apple_gf, ratio, marker
        );
    }

    pub fn summary(&self) {
        let losses = self.total - self.wins - self.ties;
        println!();
        println!("=== scoreboard ===");
        println!(
            "wins {}, ties {}, losses {}, total {}",
            self.wins, self.ties, losses, self.total
        );
    }
}

// ---------------------------------------------------------------------------
// Apple Accelerate FFI
// ---------------------------------------------------------------------------

pub type CblasOrder = i32;
pub type CblasTranspose = i32;
pub const CBLAS_ROW_MAJOR: CblasOrder = 101;
pub const CBLAS_NO_TRANS: CblasTranspose = 111;

extern "C" {
    // BLAS
    pub fn cblas_sgemm(
        order: CblasOrder,
        transa: CblasTranspose,
        transb: CblasTranspose,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
    pub fn cblas_sdot(n: i32, x: *const f32, incx: i32, y: *const f32, incy: i32) -> f32;
    pub fn cblas_snrm2(n: i32, x: *const f32, incx: i32) -> f32;

    // vForce
    pub fn vvexpf(result: *mut f32, src: *const f32, n: *const i32);
    pub fn vvlogf(result: *mut f32, src: *const f32, n: *const i32);
    pub fn vvtanhf(result: *mut f32, src: *const f32, n: *const i32);

    // vDSP
    pub fn vDSP_sve(a: *const f32, stride: i32, result: *mut f32, n: u64);
    pub fn vDSP_svesq(a: *const f32, stride: i32, result: *mut f32, n: u64);
    pub fn vDSP_maxv(a: *const f32, stride: i32, result: *mut f32, n: u64);
    pub fn vDSP_minv(a: *const f32, stride: i32, result: *mut f32, n: u64);
    pub fn vDSP_vneg(a: *const f32, stride_a: i32, c: *mut f32, stride_c: i32, n: u64);
    pub fn vDSP_vadd(
        a: *const f32,
        stride_a: i32,
        b: *const f32,
        stride_b: i32,
        c: *mut f32,
        stride_c: i32,
        n: u64,
    );
    pub fn vDSP_vmul(
        a: *const f32,
        stride_a: i32,
        b: *const f32,
        stride_b: i32,
        c: *mut f32,
        stride_c: i32,
        n: u64,
    );
    pub fn vDSP_vsadd(
        a: *const f32,
        stride_a: i32,
        b: *const f32,
        c: *mut f32,
        stride_c: i32,
        n: u64,
    );
    pub fn vDSP_vsmul(
        a: *const f32,
        stride_a: i32,
        b: *const f32,
        c: *mut f32,
        stride_c: i32,
        n: u64,
    );
    pub fn vDSP_vsdiv(
        a: *const f32,
        stride_a: i32,
        b: *const f32,
        c: *mut f32,
        stride_c: i32,
        n: u64,
    );
    pub fn vDSP_svdiv(
        a: *const f32,
        b: *const f32,
        stride_b: i32,
        c: *mut f32,
        stride_c: i32,
        n: u64,
    );
}
