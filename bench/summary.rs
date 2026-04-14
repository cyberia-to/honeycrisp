//! Unified benchmark summary — ONE table with all results.
//! Runs key operations from each category, compares against Apple/scalar/reference.
//! IMPORTANT: All Apple Accelerate BLAS calls run FIRST to avoid thread pool deadlock.

#[path = "common.rs"]
mod common;
use common::*;
use std::time::Instant;

// ── Pure Rust Goldilocks (nebu cross-platform baseline, no acpu asm) ──
mod gl_pure {
    pub const P: u64 = 0xFFFF_FFFF_0000_0001;
    const EPS: u64 = 0xFFFF_FFFF;

    #[inline]
    fn reduce128(x: u128) -> u64 {
        let lo = x as u64;
        let hi = (x >> 64) as u64;
        let (mut t0, borrow) = lo.overflowing_sub(hi >> 32);
        if borrow {
            t0 = t0.wrapping_sub(EPS);
        }
        let t1 = (hi & EPS).wrapping_mul(EPS);
        let (res, carry) = t0.overflowing_add(t1);
        let mut v = res.wrapping_add(EPS * carry as u64);
        if v >= P {
            v = v.wrapping_sub(P);
        }
        v
    }
    #[inline]
    pub fn mul(a: u64, b: u64) -> u64 {
        reduce128(a as u128 * b as u128)
    }
    #[inline]
    pub fn add(a: u64, b: u64) -> u64 {
        let (sum, over) = a.overflowing_add(b);
        let (mut sum, over) = sum.overflowing_add(over as u64 * EPS);
        if over {
            sum = sum.wrapping_add(EPS);
        }
        if sum >= P {
            sum = sum.wrapping_sub(P);
        }
        sum
    }
    pub fn inv(x: u64) -> u64 {
        let mut t = x;
        for i in (0..=62).rev() {
            t = mul(t, t);
            if i != 32 {
                t = mul(t, x);
            }
        }
        t
    }
    pub fn pow7(x: u64) -> u64 {
        let x2 = mul(x, x);
        let x3 = mul(x2, x);
        let x4 = mul(x2, x2);
        mul(x3, x4)
    }
    pub fn mul_batch(a: &[u64], b: &[u64], dst: &mut [u64]) {
        for i in 0..a.len().min(b.len()).min(dst.len()) {
            dst[i] = mul(a[i], b[i]);
        }
    }
    fn exp_mod(mut base: u64, mut e: u64) -> u64 {
        let mut result = 1u64;
        while e > 0 {
            if e & 1 == 1 {
                result = mul(result, base);
            }
            base = mul(base, base);
            e >>= 1;
        }
        result
    }
    /// Pure Rust NTT (same Cooley-Tukey as nebu, but using gl_pure mul/add).
    pub fn ntt_pure(a: &mut [u64]) {
        let n = a.len();
        let k = n.trailing_zeros();
        // bit-reverse permute
        for i in 0..n {
            let mut j = 0usize;
            for b in 0..k {
                j |= ((i >> b as usize) & 1) << (k - 1 - b) as usize;
            }
            if i < j {
                a.swap(i, j);
            }
        }
        for s in 0..k {
            let m = 1usize << (s + 1);
            let omega_m = exp_mod(7, (P - 1) / m as u64);
            let half_m = m / 2;
            let mut j = 0;
            while j < n {
                let mut w = 1u64;
                for i in 0..half_m {
                    let t = mul(w, a[j + i + half_m]);
                    a[j + i + half_m] = add(a[j + i], P - t); // sub
                    a[j + i] = add(a[j + i], t);
                    w = mul(w, omega_m);
                }
                j += m;
            }
        }
    }
    pub fn batch_inv(input: &[u64], output: &mut [u64]) {
        let n = input.len();
        if n == 0 {
            return;
        }
        let mut partials = vec![0u64; n];
        partials[0] = input[0];
        for i in 1..n {
            partials[i] = mul(partials[i - 1], input[i]);
        }
        let mut acc = inv(partials[n - 1]);
        for i in (1..n).rev() {
            output[i] = mul(acc, partials[i - 1]);
            acc = mul(acc, input[i]);
        }
        output[0] = acc;
    }
}

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        o: i32,
        ta: i32,
        tb: i32,
        m: i32,
        n: i32,
        k: i32,
        a: f32,
        ap: *const f32,
        lda: i32,
        bp: *const f32,
        ldb: i32,
        b: f32,
        cp: *mut f32,
        ldc: i32,
    );
    fn cblas_sdot(n: i32, x: *const f32, ix: i32, y: *const f32, iy: i32) -> f32;
    fn cblas_snrm2(n: i32, x: *const f32, ix: i32) -> f32;
    fn vvexpf(y: *mut f32, x: *const f32, n: *const i32);
    fn vvlogf(y: *mut f32, x: *const f32, n: *const i32);
    fn vvtanhf(y: *mut f32, x: *const f32, n: *const i32);
    fn vvrsqrtf(y: *mut f32, x: *const f32, n: *const i32);
    fn vvrecf(y: *mut f32, x: *const f32, n: *const i32);
    fn vDSP_sve(a: *const f32, ia: i64, c: *mut f32, n: u64);
    fn vDSP_svesq(a: *const f32, ia: i64, c: *mut f32, n: u64);
    fn vDSP_maxv(a: *const f32, ia: i64, c: *mut f32, n: u64);
    fn vDSP_minv(a: *const f32, ia: i64, c: *mut f32, n: u64);
    fn vDSP_vneg(a: *const f32, ia: i64, c: *mut f32, ic: i64, n: u64);
    fn vDSP_vadd(a: *const f32, ia: i64, b: *const f32, ib: i64, c: *mut f32, ic: i64, n: u64);
    fn vDSP_vmul(a: *const f32, ia: i64, b: *const f32, ib: i64, c: *mut f32, ic: i64, n: u64);
    fn vDSP_vsub(b: *const f32, ib: i64, a: *const f32, ia: i64, c: *mut f32, ic: i64, n: u64);
    fn vDSP_vsadd(a: *const f32, ia: i64, b: *const f32, c: *mut f32, ic: i64, n: u64);
    fn vDSP_vsmul(a: *const f32, ia: i64, b: *const f32, c: *mut f32, ic: i64, n: u64);
    fn vDSP_vsdiv(a: *const f32, ia: i64, b: *const f32, c: *mut f32, ic: i64, n: u64);
    fn vDSP_svdiv(a: *const f32, b: *const f32, ib: i64, c: *mut f32, ic: i64, n: u64);
    fn vDSP_vclip(
        a: *const f32,
        ia: i64,
        lo: *const f32,
        hi: *const f32,
        c: *mut f32,
        ic: i64,
        n: u64,
    );
    fn vDSP_vintb(
        a: *const f32,
        ia: i64,
        b: *const f32,
        ib: i64,
        t: *const f32,
        c: *mut f32,
        ic: i64,
        n: u64,
    );
    fn CC_SHA256(data: *const u8, len: u32, md: *mut u8) -> *mut u8;
    fn CCCrypt(
        op: u32,
        alg: u32,
        opt: u32,
        key: *const u8,
        kl: usize,
        iv: *const u8,
        di: *const u8,
        dl: usize,
        d_o: *mut u8,
        da: usize,
        dm: *mut usize,
    ) -> i32;
}

struct Row {
    group: &'static str,
    op: String,
    unit: &'static str,
    acpu: f64,
    baseline: String,
    baseline_val: f64,
}

fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(600));
        eprintln!("TIMEOUT 600s");
        std::process::exit(1);
    });

    let caps = acpu::probe::scan();
    let mut rows: Vec<Row> = Vec::new();
    let n = 4096usize;
    let nn = n as i32;
    let nu = n as u64;

    // ── methodology: pin P-core for all measurements ──
    let _ = acpu::sync::affinity::pin_p_core();
    // thermal warmup: light work to stabilize clocks
    let mut warmup_buf = vec![0f32; 4096];
    for _ in 0..1000 {
        for i in 0..4096 {
            warmup_buf[i] = (warmup_buf[i] + 1.0).sqrt();
        }
    }
    std::hint::black_box(&warmup_buf);

    /// Brief pause to let thermals settle after heavy compute.
    fn cool() {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    // ── data ──
    let src: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) / n as f32).collect();
    let pos: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 20.0).collect();
    let bv: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.7 + 0.3).sin()).collect();
    let dv: Vec<f32> = (0..n).map(|i| ((i as f32) * 1.3).cos()).collect();
    let wt: Vec<f32> = (0..n)
        .map(|i| ((i % 13) as f32 * 0.1).abs() + 0.1)
        .collect();
    let mut out = vec![0f32; n];
    let mut out2 = vec![0f32; n];

    // ╔══════════════════════════════════════════════════════════════════════╗
    // ║  PHASE 1: Apple Accelerate BLAS calls FIRST (avoid deadlock)       ║
    // ╚══════════════════════════════════════════════════════════════════════╝

    // SGEMM Apple baselines
    let sgemm_sizes: &[(usize, &str)] = &[
        (32, "32×32"),
        (64, "64×64"),
        (128, "128×128"),
        (256, "256×256"),
        (512, "512×512"),
        (1024, "1024×1024"),
        (2048, "2048×2048"),
        (4096, "4096×4096"),
    ];
    let mut apple_sgemm: Vec<u64> = Vec::new();
    for &(sz, _) in sgemm_sizes {
        let a = vec![0.1f32; sz * sz];
        let b = vec![0.1f32; sz * sz];
        let mut c = vec![0f32; sz * sz];
        let iters = if sz >= 4096 {
            2
        } else if sz >= 2048 {
            3
        } else if sz >= 512 {
            10
        } else if sz >= 128 {
            50
        } else {
            200
        };
        for _ in 0..3 {
            unsafe {
                cblas_sgemm(
                    101,
                    111,
                    111,
                    sz as i32,
                    sz as i32,
                    sz as i32,
                    1.0,
                    a.as_ptr(),
                    sz as i32,
                    b.as_ptr(),
                    sz as i32,
                    0.0,
                    c.as_mut_ptr(),
                    sz as i32,
                );
            }
        }
        apple_sgemm.push(best_of(
            || unsafe {
                cblas_sgemm(
                    101,
                    111,
                    111,
                    sz as i32,
                    sz as i32,
                    sz as i32,
                    1.0,
                    a.as_ptr(),
                    sz as i32,
                    b.as_ptr(),
                    sz as i32,
                    0.0,
                    c.as_mut_ptr(),
                    sz as i32,
                );
                std::hint::black_box(&c);
            },
            iters,
        ));
    }

    // AI inference SGEMM Apple baselines (non-square)
    let ai_sizes: &[(usize, usize, usize, &str)] = &[
        (4096, 4096, 4096, "FFN 4K×4K"),
        (4096, 11008, 4096, "llama FFN up 4K×11K"),
        (32, 4096, 128, "attn QK^T (32 heads)"),
    ];
    let mut apple_ai: Vec<u64> = Vec::new();
    for &(m, ai_n, k, _) in ai_sizes {
        let a = vec![0.1f32; m * k];
        let b = vec![0.1f32; k * ai_n];
        let mut c = vec![0f32; m * ai_n];
        let iters = if m * ai_n * k > 100_000_000 { 3 } else { 20 };
        for _ in 0..2 {
            unsafe {
                cblas_sgemm(
                    101,
                    111,
                    111,
                    m as i32,
                    ai_n as i32,
                    k as i32,
                    1.0,
                    a.as_ptr(),
                    k as i32,
                    b.as_ptr(),
                    ai_n as i32,
                    0.0,
                    c.as_mut_ptr(),
                    ai_n as i32,
                );
            }
        }
        apple_ai.push(best_of(
            || unsafe {
                cblas_sgemm(
                    101,
                    111,
                    111,
                    m as i32,
                    ai_n as i32,
                    k as i32,
                    1.0,
                    a.as_ptr(),
                    k as i32,
                    b.as_ptr(),
                    ai_n as i32,
                    0.0,
                    c.as_mut_ptr(),
                    ai_n as i32,
                );
                std::hint::black_box(&c);
            },
            iters,
        ));
    }

    // Matvec Apple baselines
    let mv_sizes: &[(usize, usize, &str)] = &[
        (512, 512, "1×512×512"),
        (1024, 1024, "1×1024×1024"),
        (4096, 4096, "1×4096×4096"),
        (4096, 11008, "1×4096×11008"),
    ];
    let mut apple_mv: Vec<u64> = Vec::new();
    for &(k, nn_mv, _) in mv_sizes {
        let a = vec![0.1f32; k];
        let b = vec![0.1f32; k * nn_mv];
        let mut c = vec![0f32; nn_mv];
        for _ in 0..3 {
            unsafe {
                cblas_sgemm(
                    101,
                    111,
                    111,
                    1,
                    nn_mv as i32,
                    k as i32,
                    1.0,
                    a.as_ptr(),
                    k as i32,
                    b.as_ptr(),
                    nn_mv as i32,
                    0.0,
                    c.as_mut_ptr(),
                    nn_mv as i32,
                );
            }
        }
        apple_mv.push(best_of(
            || unsafe {
                cblas_sgemm(
                    101,
                    111,
                    111,
                    1,
                    nn_mv as i32,
                    k as i32,
                    1.0,
                    a.as_ptr(),
                    k as i32,
                    b.as_ptr(),
                    nn_mv as i32,
                    0.0,
                    c.as_mut_ptr(),
                    nn_mv as i32,
                );
                std::hint::black_box(&c);
            },
            50,
        ));
    }

    // ╔══════════════════════════════════════════════════════════════════════╗
    // ║  PHASE 2: Elementwise, Reductions, Compound (vs Apple vForce/vDSP) ║
    // ╚══════════════════════════════════════════════════════════════════════╝

    let g = "Elementwise f32";

    // exp
    let t_a = ns(|| {
        acpu::vector::math::exp_to(&pos, &mut out);
        std::hint::black_box(&out);
    });
    let t_b = ns(|| unsafe {
        vvexpf(out2.as_mut_ptr(), pos.as_ptr(), &nn);
        std::hint::black_box(&out2);
    });
    rows.push(Row {
        group: g,
        op: "exp 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "Apple vvexpf".into(),
        baseline_val: t_b as f64,
    });

    // log
    let t_a = ns(|| {
        acpu::vector::math::log_to(&src, &mut out);
        std::hint::black_box(&out);
    });
    let t_b = ns(|| unsafe {
        vvlogf(out2.as_mut_ptr(), src.as_ptr(), &nn);
        std::hint::black_box(&out2);
    });
    rows.push(Row {
        group: g,
        op: "log 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "Apple vvlogf".into(),
        baseline_val: t_b as f64,
    });

    // tanh
    let t_a = ns(|| {
        out.copy_from_slice(&pos);
        acpu::vector::math::tanh(&mut out);
        std::hint::black_box(&out);
    });
    let t_b = ns(|| unsafe {
        vvtanhf(out2.as_mut_ptr(), pos.as_ptr(), &nn);
        std::hint::black_box(&out2);
    });
    rows.push(Row {
        group: g,
        op: "tanh 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "Apple vvtanhf".into(),
        baseline_val: t_b as f64,
    });

    // sigmoid
    let mut neg = vec![0f32; n];
    let mut expn = vec![0f32; n];
    let t_a = ns(|| {
        out.copy_from_slice(&pos);
        acpu::vector::math::sigmoid(&mut out);
        std::hint::black_box(&out);
    });
    let one = 1.0f32;
    let t_b = ns(|| unsafe {
        vDSP_vneg(pos.as_ptr(), 1, neg.as_mut_ptr(), 1, nu);
        vvexpf(expn.as_mut_ptr(), neg.as_ptr(), &nn);
        vDSP_vsadd(expn.as_ptr(), 1, &one, expn.as_mut_ptr(), 1, nu);
        vDSP_svdiv(&one, expn.as_ptr(), 1, out2.as_mut_ptr(), 1, nu);
        std::hint::black_box(&out2);
    });
    rows.push(Row {
        group: g,
        op: "sigmoid 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "Apple chain".into(),
        baseline_val: t_b as f64,
    });

    // gelu
    let mut tmp = vec![0f32; n];
    let t_a = ns(|| {
        out.copy_from_slice(&pos);
        acpu::vector::math::gelu(&mut out);
        std::hint::black_box(&out);
    });
    let half = 0.5f32;
    let t_b = ns(|| unsafe {
        vDSP_vmul(pos.as_ptr(), 1, pos.as_ptr(), 1, tmp.as_mut_ptr(), 1, nu);
        vDSP_vmul(tmp.as_ptr(), 1, pos.as_ptr(), 1, tmp.as_mut_ptr(), 1, nu);
        let c = 0.044715f32;
        vDSP_vsmul(tmp.as_ptr(), 1, &c, tmp.as_mut_ptr(), 1, nu);
        vDSP_vadd(pos.as_ptr(), 1, tmp.as_ptr(), 1, tmp.as_mut_ptr(), 1, nu);
        let s2pi = 0.7978845608f32;
        vDSP_vsmul(tmp.as_ptr(), 1, &s2pi, tmp.as_mut_ptr(), 1, nu);
        vvtanhf(out2.as_mut_ptr(), tmp.as_ptr(), &nn);
        vDSP_vsadd(out2.as_ptr(), 1, &one, out2.as_mut_ptr(), 1, nu);
        vDSP_vsmul(pos.as_ptr(), 1, &half, tmp.as_mut_ptr(), 1, nu);
        vDSP_vmul(tmp.as_ptr(), 1, out2.as_ptr(), 1, out2.as_mut_ptr(), 1, nu);
        std::hint::black_box(&out2);
    });
    rows.push(Row {
        group: g,
        op: "gelu 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "Apple chain".into(),
        baseline_val: t_b as f64,
    });

    // silu
    let mut sig_buf = vec![0f32; n];
    let t_a = ns(|| {
        out.copy_from_slice(&pos);
        acpu::vector::math::silu(&mut out);
        std::hint::black_box(&out);
    });
    let t_b = ns(|| unsafe {
        vDSP_vneg(pos.as_ptr(), 1, neg.as_mut_ptr(), 1, nu);
        vvexpf(expn.as_mut_ptr(), neg.as_ptr(), &nn);
        vDSP_vsadd(expn.as_ptr(), 1, &one, expn.as_mut_ptr(), 1, nu);
        vDSP_svdiv(&one, expn.as_ptr(), 1, sig_buf.as_mut_ptr(), 1, nu);
        vDSP_vmul(
            pos.as_ptr(),
            1,
            sig_buf.as_ptr(),
            1,
            out2.as_mut_ptr(),
            1,
            nu,
        );
        std::hint::black_box(&out2);
    });
    rows.push(Row {
        group: g,
        op: "silu 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "Apple chain".into(),
        baseline_val: t_b as f64,
    });

    // ── Reductions ──
    let g = "Reductions f32";
    let mut r = 0f32;

    let t_a = ns(|| {
        std::hint::black_box(acpu::vector::reduce::sum(&src));
    });
    let t_b = ns(|| unsafe {
        vDSP_sve(src.as_ptr(), 1, &mut r, nu);
        std::hint::black_box(r);
    });
    rows.push(Row {
        group: g,
        op: "sum 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "Apple sve".into(),
        baseline_val: t_b as f64,
    });

    let t_a = ns(|| {
        std::hint::black_box(acpu::vector::reduce::dot(&bv, &dv));
    });
    let t_b = ns(|| unsafe {
        std::hint::black_box(cblas_sdot(nn, bv.as_ptr(), 1, dv.as_ptr(), 1));
    });
    rows.push(Row {
        group: g,
        op: "dot 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "Apple sdot".into(),
        baseline_val: t_b as f64,
    });

    let t_a = ns(|| {
        std::hint::black_box(acpu::vector::reduce::length(&src));
    });
    let t_b = ns(|| unsafe {
        std::hint::black_box(cblas_snrm2(nn, src.as_ptr(), 1));
    });
    rows.push(Row {
        group: g,
        op: "length (L2) 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "Apple snrm2".into(),
        baseline_val: t_b as f64,
    });

    let t_a = ns(|| {
        std::hint::black_box(acpu::vector::reduce::max(&src));
    });
    let t_b = ns(|| unsafe {
        vDSP_maxv(src.as_ptr(), 1, &mut r, nu);
        std::hint::black_box(r);
    });
    rows.push(Row {
        group: g,
        op: "max 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "Apple maxv".into(),
        baseline_val: t_b as f64,
    });

    let t_a = ns(|| {
        std::hint::black_box(acpu::vector::reduce::min(&src));
    });
    let t_b = ns(|| unsafe {
        vDSP_minv(src.as_ptr(), 1, &mut r, nu);
        std::hint::black_box(r);
    });
    rows.push(Row {
        group: g,
        op: "min 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "Apple minv".into(),
        baseline_val: t_b as f64,
    });

    // ── Compound ──
    let g = "Compound f32";

    let t_a = ns(|| {
        out.copy_from_slice(&pos);
        acpu::vector::softmax(&mut out);
        std::hint::black_box(&out);
    });
    let mut mx = 0f32;
    let mut s = 0f32;
    let t_b = ns(|| unsafe {
        vDSP_maxv(pos.as_ptr(), 1, &mut mx, nu);
        let neg_mx = -mx;
        vDSP_vsadd(pos.as_ptr(), 1, &neg_mx, out2.as_mut_ptr(), 1, nu);
        vvexpf(out2.as_mut_ptr(), out2.as_ptr(), &nn);
        vDSP_sve(out2.as_ptr(), 1, &mut s, nu);
        vDSP_vsdiv(out2.as_ptr(), 1, &s, out2.as_mut_ptr(), 1, nu);
        std::hint::black_box(&out2);
    });
    rows.push(Row {
        group: g,
        op: "softmax 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "Apple chain".into(),
        baseline_val: t_b as f64,
    });

    let t_a = ns(|| {
        acpu::vector::normalize(&mut out, &src, &wt, 1e-5);
        std::hint::black_box(&out);
    });
    let mut ss = 0f32;
    let t_b = ns(|| unsafe {
        vDSP_svesq(src.as_ptr(), 1, &mut ss, nu);
        let inv = 1.0 / (ss / n as f32 + 1e-5f32).sqrt();
        vDSP_vsmul(src.as_ptr(), 1, &inv, out2.as_mut_ptr(), 1, nu);
        vDSP_vmul(out2.as_ptr(), 1, wt.as_ptr(), 1, out2.as_mut_ptr(), 1, nu);
        std::hint::black_box(&out2);
    });
    rows.push(Row {
        group: g,
        op: "rmsnorm 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "Apple chain".into(),
        baseline_val: t_b as f64,
    });

    // ── Rendering ──
    let g = "Rendering f32";
    let pos_r: Vec<f32> = (1..=n).map(|i| i as f32 * 0.01 + 0.1).collect();

    let t_a = ns(|| {
        acpu::vector::render::rsqrt_to(&pos_r, &mut out);
        std::hint::black_box(&out);
    });
    let t_b = ns(|| unsafe {
        vvrsqrtf(out2.as_mut_ptr(), pos_r.as_ptr(), &nn);
        std::hint::black_box(&out2);
    });
    rows.push(Row {
        group: g,
        op: "rsqrt 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "Apple vvrsqrtf".into(),
        baseline_val: t_b as f64,
    });

    let t_a = ns(|| {
        acpu::vector::render::recip_to(&pos_r, &mut out);
        std::hint::black_box(&out);
    });
    let t_b = ns(|| unsafe {
        vvrecf(out2.as_mut_ptr(), pos_r.as_ptr(), &nn);
        std::hint::black_box(&out2);
    });
    rows.push(Row {
        group: g,
        op: "recip 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "Apple vvrecf".into(),
        baseline_val: t_b as f64,
    });

    let lo = -100f32;
    let hi = 100f32;
    let t_a = ns(|| {
        acpu::vector::render::clamp_to(&pos, &mut out, lo, hi);
        std::hint::black_box(&out);
    });
    let t_b = ns(|| unsafe {
        vDSP_vclip(pos.as_ptr(), 1, &lo, &hi, out2.as_mut_ptr(), 1, nu);
        std::hint::black_box(&out2);
    });
    rows.push(Row {
        group: g,
        op: "clamp 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "Apple vclip".into(),
        baseline_val: t_b as f64,
    });

    let bv2: Vec<f32> = (0..n).map(|i| i as f32 * 0.3).collect();
    let t_lerp = 0.5f32;
    let t_a = ns(|| {
        acpu::vector::render::lerp(&mut out, &src, &bv2, t_lerp);
        std::hint::black_box(&out);
    });
    let t_b = ns(|| unsafe {
        vDSP_vintb(
            src.as_ptr(),
            1,
            bv2.as_ptr(),
            1,
            &t_lerp,
            out2.as_mut_ptr(),
            1,
            nu,
        );
        std::hint::black_box(&out2);
    });
    rows.push(Row {
        group: g,
        op: "lerp 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "Apple vintb".into(),
        baseline_val: t_b as f64,
    });

    // ╔══════════════════════════════════════════════════════════════════════╗
    // ║  Integer SIMD (vs real-world alternatives)                        ║
    // ╚══════════════════════════════════════════════════════════════════════╝
    let g = "Integer SIMD";
    let i8a: Vec<i8> = (0..n).map(|i| ((i * 7) % 255) as i8).collect();
    let i8b: Vec<i8> = (0..n).map(|i| ((i * 13) % 255) as i8).collect();
    let i32a: Vec<i32> = (0..n).map(|i| i as i32 * 7 - 2000).collect();
    let i32b: Vec<i32> = (0..n).map(|i| i as i32 * 3 + 500).collect();
    let i16a: Vec<i16> = (0..n).map(|i| i as i16 * 5 - 2000).collect();
    let i16b: Vec<i16> = (0..n).map(|i| i as i16 * 3 + 1000).collect();
    let u8a: Vec<u8> = (0..n).map(|i| (i % 200) as u8).collect();
    let u8b: Vec<u8> = (0..n).map(|i| ((i * 3 + 50) % 200) as u8).collect();

    // dot_i8 SDOT vs "don't quantize, use f32 cblas_sdot"
    let f32a: Vec<f32> = i8a.iter().map(|&x| x as f32).collect();
    let f32b: Vec<f32> = i8b.iter().map(|&x| x as f32).collect();
    let t_a = best_of(
        || {
            std::hint::black_box(acpu::vector::integer::dot_i8(&i8a, &i8b));
        },
        500,
    );
    let t_b = ns(|| unsafe {
        std::hint::black_box(cblas_sdot(nn, f32a.as_ptr(), 1, f32b.as_ptr(), 1));
    });
    rows.push(Row {
        group: g,
        op: "dot_i8 SDOT 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "f32 sdot".into(),
        baseline_val: t_b as f64,
    });

    // sad_u8 — video ME, no Apple equiv → vs auto-vectorized loop
    let t_a = best_of(
        || {
            std::hint::black_box(acpu::vector::integer_fused::sad_u8(&u8a, &u8b));
        },
        500,
    );
    let t_b = best_of(
        || {
            let s: u64 = u8a
                .iter()
                .zip(&u8b)
                .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u64)
                .sum();
            std::hint::black_box(s);
        },
        500,
    );
    rows.push(Row {
        group: g,
        op: "sad_u8 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "auto-vec".into(),
        baseline_val: t_b as f64,
    });

    // sum_i32 vs cast-to-f32 + vDSP_sve (the "just use float" approach)
    let i32_as_f32: Vec<f32> = i32a.iter().map(|&x| x as f32).collect();
    let t_a = ns(|| {
        std::hint::black_box(acpu::vector::integer::sum_i32(&i32a));
    });
    let mut rf = 0f32;
    let t_b = ns(|| unsafe {
        vDSP_sve(i32_as_f32.as_ptr(), 1, &mut rf, nu);
        std::hint::black_box(rf);
    });
    rows.push(Row {
        group: g,
        op: "sum_i32 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "f32 vDSP_sve".into(),
        baseline_val: t_b as f64,
    });

    // macc_i16 vs cast-to-f32 + vDSP_vmul + vDSP_vadd
    let i16a_f32: Vec<f32> = i16a.iter().map(|&x| x as f32).collect();
    let i16b_f32: Vec<f32> = i16b.iter().map(|&x| x as f32).collect();
    let mut acc_f32 = vec![0f32; n];
    let mut tmp_f32 = vec![0f32; n];
    let mut acc32 = vec![0i32; n];
    let t_a = ns(|| {
        acc32.fill(0);
        acpu::vector::integer::macc_i16(&mut acc32, &i16a, &i16b);
        std::hint::black_box(&acc32);
    });
    let t_b = ns(|| unsafe {
        vDSP_vmul(
            i16a_f32.as_ptr(),
            1,
            i16b_f32.as_ptr(),
            1,
            tmp_f32.as_mut_ptr(),
            1,
            nu,
        );
        vDSP_vadd(
            acc_f32.as_ptr(),
            1,
            tmp_f32.as_ptr(),
            1,
            acc_f32.as_mut_ptr(),
            1,
            nu,
        );
        std::hint::black_box(&acc_f32);
    });
    rows.push(Row {
        group: g,
        op: "macc_i16 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "f32 vDSP".into(),
        baseline_val: t_b as f64,
    });

    // absmax_i8 — quant calibration, vs f32 max(abs())
    let i8_as_f32: Vec<f32> = i8a.iter().map(|&x| (x as f32).abs()).collect();
    let t_a = ns(|| {
        std::hint::black_box(acpu::vector::integer::absmax_i8(&i8a));
    });
    let mut rmx = 0f32;
    let t_b = ns(|| unsafe {
        vDSP_maxv(i8_as_f32.as_ptr(), 1, &mut rmx, nu);
        std::hint::black_box(rmx);
    });
    rows.push(Row {
        group: g,
        op: "absmax_i8 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "f32 vDSP_maxv".into(),
        baseline_val: t_b as f64,
    });

    // ssd_i32 — L2 distance, vs f32 cblas_snrm2 on difference
    let diff_f32: Vec<f32> = i32a
        .iter()
        .zip(&i32b)
        .map(|(&a, &b)| (a - b) as f32)
        .collect();
    let t_a = ns(|| {
        std::hint::black_box(acpu::vector::integer_fused::ssd_i32(&i32a, &i32b));
    });
    let t_b = ns(|| unsafe {
        let r = cblas_snrm2(nn, diff_f32.as_ptr(), 1);
        std::hint::black_box(r * r);
    });
    rows.push(Row {
        group: g,
        op: "ssd_i32 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "f32 snrm2²".into(),
        baseline_val: t_b as f64,
    });

    // ╔══════════════════════════════════════════════════════════════════════╗
    // ║  Compute Primitives                                                ║
    // ╚══════════════════════════════════════════════════════════════════════╝
    let g = "Compute Primitives";

    // prefix_sum (exclusive scan) — vs scalar running sum
    let scan_src: Vec<f32> = (0..n).map(|i| (i % 100) as f32 * 0.01).collect();
    let mut scan_out = vec![0f32; n];
    let mut scan_out2 = vec![0f32; n];
    let t_a = ns(|| {
        acpu::vector::prefix_sum_f32(&mut scan_out, &scan_src);
        std::hint::black_box(&scan_out);
    });
    let t_b = ns(|| {
        let mut acc = 0f32;
        for i in 0..n {
            scan_out2[i] = acc;
            acc += scan_src[i];
        }
        std::hint::black_box(&scan_out2);
    });
    rows.push(Row {
        group: g,
        op: "prefix_sum 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "scalar".into(),
        baseline_val: t_b as f64,
    });

    // transpose 256×256 — vs vDSP_mtrans
    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        fn vDSP_mtrans(a: *const f32, ia: i64, c: *mut f32, ic: i64, m: u64, n: u64);
    }
    let tr_sz = 256usize;
    let tr_src: Vec<f32> = (0..tr_sz * tr_sz).map(|i| i as f32 * 0.001).collect();
    let mut tr_dst = vec![0f32; tr_sz * tr_sz];
    let mut tr_dst2 = vec![0f32; tr_sz * tr_sz];
    let t_a = ns(|| {
        acpu::vector::transpose_f32(&mut tr_dst, &tr_src, tr_sz, tr_sz);
        std::hint::black_box(&tr_dst);
    });
    let t_b = ns(|| unsafe {
        vDSP_mtrans(
            tr_src.as_ptr(),
            1,
            tr_dst2.as_mut_ptr(),
            1,
            tr_sz as u64,
            tr_sz as u64,
        );
        std::hint::black_box(&tr_dst2);
    });
    rows.push(Row {
        group: g,
        op: "transpose 256×256".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "Apple mtrans".into(),
        baseline_val: t_b as f64,
    });

    // transpose 1024×1024
    let tr_sz2 = 1024usize;
    let tr_src2: Vec<f32> = (0..tr_sz2 * tr_sz2).map(|i| i as f32 * 0.001).collect();
    let mut tr_dst3 = vec![0f32; tr_sz2 * tr_sz2];
    let mut tr_dst4 = vec![0f32; tr_sz2 * tr_sz2];
    let t_a = best_of(
        || {
            acpu::vector::transpose_f32(&mut tr_dst3, &tr_src2, tr_sz2, tr_sz2);
            std::hint::black_box(&tr_dst3);
        },
        20,
    );
    let t_b = best_of(
        || unsafe {
            vDSP_mtrans(
                tr_src2.as_ptr(),
                1,
                tr_dst4.as_mut_ptr(),
                1,
                tr_sz2 as u64,
                tr_sz2 as u64,
            );
            std::hint::black_box(&tr_dst4);
        },
        20,
    );
    rows.push(Row {
        group: g,
        op: "transpose 1024×1024".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "Apple mtrans".into(),
        baseline_val: t_b as f64,
    });

    // ╔══════════════════════════════════════════════════════════════════════╗
    // ║  Numeric: conversions, complex, rope                               ║
    // ╚══════════════════════════════════════════════════════════════════════╝
    let g = "Numeric";

    // complex mul-acc vs Apple vDSP chain
    let ca: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.1).collect();
    let cb: Vec<f32> = (0..n).map(|i| (i % 11) as f32 * 0.1).collect();
    let mut cacc = vec![0f32; n];
    let t_a = ns(|| {
        cacc.fill(0.0);
        acpu::numeric::complex::complex_mul_acc(&mut cacc, &ca, &cb);
        std::hint::black_box(&cacc);
    });
    let half_n = n / 2;
    let a_re: Vec<f32> = (0..half_n).map(|i| ca[2 * i]).collect();
    let a_im: Vec<f32> = (0..half_n).map(|i| ca[2 * i + 1]).collect();
    let b_re: Vec<f32> = (0..half_n).map(|i| cb[2 * i]).collect();
    let b_im: Vec<f32> = (0..half_n).map(|i| cb[2 * i + 1]).collect();
    let mut t1 = vec![0f32; half_n];
    let mut t2 = vec![0f32; half_n];
    let mut c_re = vec![0f32; half_n];
    let mut c_im = vec![0f32; half_n];
    let hn = half_n as u64;
    let t_b = ns(|| unsafe {
        vDSP_vmul(a_re.as_ptr(), 1, b_re.as_ptr(), 1, t1.as_mut_ptr(), 1, hn);
        vDSP_vmul(a_im.as_ptr(), 1, b_im.as_ptr(), 1, t2.as_mut_ptr(), 1, hn);
        vDSP_vsub(t2.as_ptr(), 1, t1.as_ptr(), 1, c_re.as_mut_ptr(), 1, hn);
        vDSP_vmul(a_re.as_ptr(), 1, b_im.as_ptr(), 1, t1.as_mut_ptr(), 1, hn);
        vDSP_vmul(a_im.as_ptr(), 1, b_re.as_ptr(), 1, t2.as_mut_ptr(), 1, hn);
        vDSP_vadd(t1.as_ptr(), 1, t2.as_ptr(), 1, c_im.as_mut_ptr(), 1, hn);
        std::hint::black_box(&c_re);
    });
    rows.push(Row {
        group: g,
        op: "complex_mul_acc 2048".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "Apple chain".into(),
        baseline_val: t_b as f64,
    });

    // f32→f16 (NEON fcvtn vs scalar)
    let conv_src: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    let mut f16_buf = vec![0u16; n];
    let mut f32_dst = vec![0f32; n];
    let t_a = best_of(
        || {
            acpu::cast_f32_f16(&mut f16_buf, &conv_src);
            std::hint::black_box(&f16_buf);
        },
        200,
    );
    let t_b = best_of(
        || {
            for i in 0..n {
                f16_buf[i] = acpu::numeric::fp16::f32_to_fp16(conv_src[i]);
            }
            std::hint::black_box(&f16_buf);
        },
        200,
    );
    rows.push(Row {
        group: g,
        op: "f32→f16 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "scalar fp16".into(),
        baseline_val: t_b as f64,
    });

    // f16→f32 (NEON fcvtl vs scalar)
    acpu::cast_f32_f16(&mut f16_buf, &conv_src);
    let t_a = best_of(
        || {
            acpu::cast_f16_f32(&mut f32_dst, &f16_buf);
            std::hint::black_box(&f32_dst);
        },
        200,
    );
    let t_b = best_of(
        || {
            for i in 0..n {
                f32_dst[i] = acpu::numeric::fp16::fp16_to_f32(f16_buf[i]);
            }
            std::hint::black_box(&f32_dst);
        },
        200,
    );
    rows.push(Row {
        group: g,
        op: "f16→f32 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "scalar fp16".into(),
        baseline_val: t_b as f64,
    });

    // bf16 round-trip (vs scalar bit-shift)
    let mut bf16_buf = vec![0u16; n];
    let t_a = ns(|| {
        acpu::cast_f32_bf16(&mut bf16_buf, &conv_src);
        acpu::cast_bf16_f32(&mut f32_dst, &bf16_buf);
        std::hint::black_box(&f32_dst);
    });
    let t_b = ns(|| {
        for i in 0..n {
            bf16_buf[i] = (conv_src[i].to_bits() >> 16) as u16;
        }
        for i in 0..n {
            f32_dst[i] = f32::from_bits((bf16_buf[i] as u32) << 16);
        }
        std::hint::black_box(&f32_dst);
    });
    rows.push(Row {
        group: g,
        op: "bf16 round-trip 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "scalar shift".into(),
        baseline_val: t_b as f64,
    });

    // i8 quant round-trip (vs scalar)
    let mut i8_buf = vec![0i8; n];
    let t_a = ns(|| {
        acpu::cast_f32_i8(&mut i8_buf, &conv_src, 0.1);
        acpu::cast_i8_f32(&mut f32_dst, &i8_buf, 0.1, 0);
        std::hint::black_box(&f32_dst);
    });
    let t_b = ns(|| {
        for i in 0..n {
            i8_buf[i] = (conv_src[i] / 0.1).round().max(-128.0).min(127.0) as i8;
        }
        for i in 0..n {
            f32_dst[i] = i8_buf[i] as f32 * 0.1;
        }
        std::hint::black_box(&f32_dst);
    });
    rows.push(Row {
        group: g,
        op: "i8 quant rt 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "scalar".into(),
        baseline_val: t_b as f64,
    });

    // ╔══════════════════════════════════════════════════════════════════════╗
    // ║  PHASE 3: SGEMM + Matvec (acpu, after Apple already measured)      ║
    // ╚══════════════════════════════════════════════════════════════════════╝
    let g = "SGEMM f32";
    for (idx, &(sz, label)) in sgemm_sizes.iter().enumerate() {
        let a = vec![0.1f32; sz * sz];
        let b = vec![0.1f32; sz * sz];
        let mut c = vec![0f32; sz * sz];
        acpu::matmul_f32_set(&a, &b, &mut c, sz, sz, sz);
        let iters = if sz >= 4096 {
            2
        } else if sz >= 2048 {
            3
        } else if sz >= 512 {
            10
        } else if sz >= 128 {
            50
        } else {
            200
        };
        let t = best_of(
            || {
                acpu::matmul_f32_set(&a, &b, &mut c, sz, sz, sz);
                std::hint::black_box(&c);
            },
            iters,
        );
        let flops = 2.0 * (sz as f64).powi(3);
        let acpu_gf = flops / t as f64;
        let apple_gf = flops / apple_sgemm[idx] as f64;
        rows.push(Row {
            group: g,
            op: label.into(),
            unit: "GF",
            acpu: acpu_gf,
            baseline: "Apple cblas".into(),
            baseline_val: apple_gf,
        });
    }

    // Mixed-precision GEMM — effective GF/GB (GFLOPS per GB of input read)
    // fp16/bf16 inputs are 2× denser than f32, i8 is 4× denser
    let g = "Mixed GEMM 256";
    let msz: usize = 256;
    let mlen = msz * msz;
    let mflops = 2.0 * (msz as f64).powi(3);
    let ma = vec![0.1f32; mlen];
    let mb = vec![0.2f32; mlen];
    let input_gb_f32 = 2.0 * mlen as f64 * 4.0 / 1e9; // 2 matrices × 256² × 4 bytes
    let input_gb_f16 = 2.0 * mlen as f64 * 2.0 / 1e9;
    let input_gb_i8 = 2.0 * mlen as f64 * 1.0 / 1e9;
    // f32 baseline
    let mut mc = vec![0f32; mlen];
    for _ in 0..3 {
        acpu::matmul_f32_set(&ma, &mb, &mut mc, msz, msz, msz);
    }
    let f32_base = best_of(
        || {
            acpu::matmul_f32_set(&ma, &mb, &mut mc, msz, msz, msz);
            std::hint::black_box(&mc);
        },
        50,
    );
    let f32_gfgb = (mflops / f32_base as f64) / input_gb_f32;
    {
        let mut a16 = vec![0u16; mlen];
        let mut b16 = vec![0u16; mlen];
        acpu::cast_f32_f16(&mut a16, &ma);
        acpu::cast_f32_f16(&mut b16, &mb);
        let mut c = vec![0f32; mlen];
        for _ in 0..3 {
            acpu::matmul_f16(&a16, &b16, &mut c, msz, msz, msz);
        }
        let t = best_of(
            || {
                acpu::matmul_f16(&a16, &b16, &mut c, msz, msz, msz);
                std::hint::black_box(&c);
            },
            50,
        );
        let gfgb = (mflops / t as f64) / input_gb_f16;
        rows.push(Row {
            group: g,
            op: "hgemm fp16".into(),
            unit: "GF/GB",
            acpu: gfgb,
            baseline: "sgemm f32".into(),
            baseline_val: f32_gfgb,
        });
    }
    {
        let mut a16 = vec![0u16; mlen];
        let mut b16 = vec![0u16; mlen];
        acpu::cast_f32_bf16(&mut a16, &ma);
        acpu::cast_f32_bf16(&mut b16, &mb);
        let mut c = vec![0f32; mlen];
        for _ in 0..3 {
            acpu::matmul_bf16(&a16, &b16, &mut c, msz, msz, msz);
        }
        let t = best_of(
            || {
                acpu::matmul_bf16(&a16, &b16, &mut c, msz, msz, msz);
                std::hint::black_box(&c);
            },
            50,
        );
        let gfgb = (mflops / t as f64) / input_gb_f16;
        rows.push(Row {
            group: g,
            op: "bgemm bf16".into(),
            unit: "GF/GB",
            acpu: gfgb,
            baseline: "sgemm f32".into(),
            baseline_val: f32_gfgb,
        });
    }
    {
        let mut a8 = vec![0i8; mlen];
        let mut b8 = vec![0i8; mlen];
        acpu::cast_f32_i8(&mut a8, &ma, 127.0);
        acpu::cast_f32_i8(&mut b8, &mb, 127.0);
        let mut c = vec![0f32; mlen];
        for _ in 0..3 {
            acpu::matmul_i8(&a8, &b8, &mut c, msz, msz, msz, 1.0 / 127.0, 0);
        }
        let t = best_of(
            || {
                acpu::matmul_i8(&a8, &b8, &mut c, msz, msz, msz, 1.0 / 127.0, 0);
                std::hint::black_box(&c);
            },
            50,
        );
        let gfgb = (mflops / t as f64) / input_gb_i8;
        rows.push(Row {
            group: g,
            op: "qgemm i8".into(),
            unit: "GF/GB",
            acpu: gfgb,
            baseline: "sgemm f32".into(),
            baseline_val: f32_gfgb,
        });
    }

    // Matvec
    let g = "Matvec (m=1)";
    for (idx, &(k, nn_mv, label)) in mv_sizes.iter().enumerate() {
        let a = vec![0.1f32; k];
        let b = vec![0.1f32; k * nn_mv];
        let mut c = vec![0f32; nn_mv];
        acpu::gemm::matvec::matvec_f32_set(&a, &b, &mut c, nn_mv, k);
        let t = best_of(
            || {
                acpu::gemm::matvec::matvec_f32_set(&a, &b, &mut c, nn_mv, k);
                std::hint::black_box(&c);
            },
            50,
        );
        let flops = 2.0 * k as f64 * nn_mv as f64;
        let acpu_gf = flops / t as f64;
        let apple_gf = flops / apple_mv[idx] as f64;
        rows.push(Row {
            group: g,
            op: label.into(),
            unit: "GF",
            acpu: acpu_gf,
            baseline: "Apple cblas".into(),
            baseline_val: apple_gf,
        });
    }

    // ╔══════════════════════════════════════════════════════════════════════╗
    cool(); // thermal settle after SGEMM
            // ║  AI Inference (transformer shapes, vs Apple cblas)                 ║
            // ╚══════════════════════════════════════════════════════════════════════╝
    let g = "AI Inference";
    for (idx, &(m, ai_n, k, label)) in ai_sizes.iter().enumerate() {
        let a = vec![0.1f32; m * k];
        let b = vec![0.1f32; k * ai_n];
        let mut c = vec![0f32; m * ai_n];
        acpu::matmul_f32_set(&a, &b, &mut c, m, ai_n, k);
        let iters = if m * ai_n * k > 100_000_000 { 3 } else { 20 };
        let t = best_of(
            || {
                acpu::matmul_f32_set(&a, &b, &mut c, m, ai_n, k);
                std::hint::black_box(&c);
            },
            iters,
        );
        let flops = 2.0 * m as f64 * ai_n as f64 * k as f64;
        let acpu_gf = flops / t as f64;
        let apple_gf = flops / apple_ai[idx] as f64;
        rows.push(Row {
            group: g,
            op: label.into(),
            unit: "GF",
            acpu: acpu_gf,
            baseline: "Apple cblas".into(),
            baseline_val: apple_gf,
        });
    }

    // AI pipeline ops (vs Apple Accelerate chains)
    let ai_x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 20.0).collect();
    let ai_w: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.1 + 0.1).collect();
    let ai_freqs: Vec<f32> = (0..n / 2)
        .map(|i| 1.0 / 10000f32.powf(2.0 * i as f32 / n as f32))
        .collect();
    let mut ai_out = vec![0f32; n];
    let mut ai_out2 = vec![0f32; n];

    // rmsnorm vs Apple chain (svesq + scale + mul)
    let t_a = ns(|| {
        acpu::vector::normalize(&mut ai_out, &ai_x, &ai_w, 1e-5);
        std::hint::black_box(&ai_out);
    });
    let mut ai_ss = 0f32;
    let t_b = ns(|| unsafe {
        vDSP_svesq(ai_x.as_ptr(), 1, &mut ai_ss, nu);
        let inv = 1.0 / (ai_ss / n as f32 + 1e-5f32).sqrt();
        vDSP_vsmul(ai_x.as_ptr(), 1, &inv, ai_out2.as_mut_ptr(), 1, nu);
        vDSP_vmul(
            ai_out2.as_ptr(),
            1,
            ai_w.as_ptr(),
            1,
            ai_out2.as_mut_ptr(),
            1,
            nu,
        );
        std::hint::black_box(&ai_out2);
    });
    rows.push(Row {
        group: g,
        op: "rmsnorm 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "Apple chain".into(),
        baseline_val: t_b as f64,
    });

    // rope — no Apple equivalent, vs scalar
    let t_a = ns(|| {
        acpu::vector::rotate(&mut ai_out, &ai_x, &ai_freqs, 42);
        std::hint::black_box(&ai_out);
    });
    let t_b = ns(|| {
        let dim = n;
        for i in 0..dim / 2 {
            let freq = ai_freqs[i];
            let (sin_v, cos_v) = (freq * 42.0 as f32).sin_cos();
            ai_out2[2 * i] = ai_x[2 * i] * cos_v - ai_x[2 * i + 1] * sin_v;
            ai_out2[2 * i + 1] = ai_x[2 * i] * sin_v + ai_x[2 * i + 1] * cos_v;
        }
        std::hint::black_box(&ai_out2);
    });
    rows.push(Row {
        group: g,
        op: "rope 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "scalar sincos".into(),
        baseline_val: t_b as f64,
    });

    // softmax at attention seq lengths (vs Apple chain: max + exp + sum + div)
    for &seq in &[128, 512, 2048, 8192] {
        let attn_src: Vec<f32> = (0..seq).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let mut attn = attn_src.clone();
        let mut attn_apple = vec![0f32; seq];
        let seq_i = seq as i32;
        let seq_u = seq as u64;
        let t_a = best_of(
            || {
                attn.copy_from_slice(&attn_src);
                acpu::vector::softmax(&mut attn);
                std::hint::black_box(&attn);
            },
            200,
        );
        let mut amx = 0f32;
        let mut asm = 0f32;
        let t_b = best_of(
            || unsafe {
                vDSP_maxv(attn_src.as_ptr(), 1, &mut amx, seq_u);
                let neg_mx = -amx;
                vDSP_vsadd(
                    attn_src.as_ptr(),
                    1,
                    &neg_mx,
                    attn_apple.as_mut_ptr(),
                    1,
                    seq_u,
                );
                vvexpf(attn_apple.as_mut_ptr(), attn_apple.as_ptr(), &seq_i);
                vDSP_sve(attn_apple.as_ptr(), 1, &mut asm, seq_u);
                vDSP_vsdiv(
                    attn_apple.as_ptr(),
                    1,
                    &asm,
                    attn_apple.as_mut_ptr(),
                    1,
                    seq_u,
                );
                std::hint::black_box(&attn_apple);
            },
            200,
        );
        rows.push(Row {
            group: g,
            op: format!("softmax seq={}", seq),
            unit: "ns",
            acpu: t_a as f64,
            baseline: "Apple chain".into(),
            baseline_val: t_b as f64,
        });
    }

    // Full attention pipeline: Q×K^T → scale → softmax → ×V
    // Llama-2 single head: seq=512, d=128
    {
        let seq = 512usize;
        let d = 128usize;
        let q = vec![0.01f32; seq * d];
        let k = vec![0.01f32; seq * d];
        let v = vec![0.01f32; seq * d];
        let mut kt = vec![0f32; d * seq]; // K transposed
        let mut scores = vec![0f32; seq * seq];
        let mut attn_out = vec![0f32; seq * d];
        let scale = 1.0 / (d as f32).sqrt();

        // acpu pipeline
        let t_a = best_of(
            || {
                // transpose K
                for i in 0..seq {
                    for j in 0..d {
                        kt[j * seq + i] = k[i * d + j];
                    }
                }
                // QK^T
                acpu::matmul_f32_set(&q, &kt, &mut scores, seq, seq, d);
                // scale
                for s in scores.iter_mut() {
                    *s *= scale;
                }
                // softmax per row
                for row in 0..seq {
                    acpu::vector::softmax(&mut scores[row * seq..(row + 1) * seq]);
                }
                // scores × V
                acpu::matmul_f32_set(&scores, &v, &mut attn_out, seq, d, seq);
                std::hint::black_box(&attn_out);
            },
            5,
        );

        // Scalar pipeline (no cblas — would deadlock after acpu thread pool)
        let t_b = best_of(
            || {
                for i in 0..seq {
                    for j in 0..d {
                        kt[j * seq + i] = k[i * d + j];
                    }
                }
                // naive matmul QK^T
                for i in 0..seq {
                    for j in 0..seq {
                        let mut s = 0f32;
                        for kk in 0..d {
                            s += q[i * d + kk] * kt[kk * seq + j];
                        }
                        scores[i * seq + j] = s;
                    }
                }
                for s in scores.iter_mut() {
                    *s *= scale;
                }
                for row in 0..seq {
                    let sl = &mut scores[row * seq..(row + 1) * seq];
                    let mut mx = f32::NEG_INFINITY;
                    for &v in sl.iter() {
                        if v > mx {
                            mx = v;
                        }
                    }
                    let mut sum = 0f32;
                    for v in sl.iter_mut() {
                        *v = (*v - mx).exp();
                        sum += *v;
                    }
                    for v in sl.iter_mut() {
                        *v /= sum;
                    }
                }
                // naive matmul scores×V
                for i in 0..seq {
                    for j in 0..d {
                        let mut s = 0f32;
                        for kk in 0..seq {
                            s += scores[i * seq + kk] * v[kk * d + j];
                        }
                        attn_out[i * d + j] = s;
                    }
                }
                std::hint::black_box(&attn_out);
            },
            2,
        );
        rows.push(Row {
            group: g,
            op: "attn 512×128 full".into(),
            unit: "ns",
            acpu: t_a as f64,
            baseline: "scalar".into(),
            baseline_val: t_b as f64,
        });
    }

    // ╔══════════════════════════════════════════════════════════════════════╗
    cool(); // thermal settle after AI inference
            // ║  Media (image/video pipeline sizes)                                ║
            // ╚══════════════════════════════════════════════════════════════════════╝
    let g = "Media";
    let px = 1920 * 1080usize; // 1080p = 2073600 pixels
    let mpx = px as f64 / 1e6; // for Mpix/s conversion

    // alpha blend (lerp) — 1080p vs Apple vDSP_vintb
    let frame_a: Vec<f32> = (0..px).map(|i| (i % 256) as f32 / 255.0).collect();
    let frame_b: Vec<f32> = (0..px).map(|i| ((i * 7) % 256) as f32 / 255.0).collect();
    let mut frame_out = vec![0f32; px];
    let mut frame_out2 = vec![0f32; px];
    let blend_t = 0.5f32;
    let pxu = px as u64;
    let t_a = best_of(
        || {
            acpu::vector::render::lerp(&mut frame_out, &frame_a, &frame_b, blend_t);
            std::hint::black_box(&frame_out);
        },
        20,
    );
    let t_b = best_of(
        || unsafe {
            vDSP_vintb(
                frame_a.as_ptr(),
                1,
                frame_b.as_ptr(),
                1,
                &blend_t,
                frame_out2.as_mut_ptr(),
                1,
                pxu,
            );
            std::hint::black_box(&frame_out2);
        },
        20,
    );
    rows.push(Row {
        group: g,
        op: "blend 1080p".into(),
        unit: "Mp/s",
        acpu: mpx / (t_a as f64 / 1e9),
        baseline: "Apple vDSP".into(),
        baseline_val: mpx / (t_b as f64 / 1e9),
    });

    // tone map (clamp) — 1080p vs Apple vDSP_vclip
    let lo_c = 0.0f32;
    let hi_c = 1.0f32;
    let t_a = best_of(
        || {
            acpu::vector::render::clamp_to(&frame_a, &mut frame_out, lo_c, hi_c);
            std::hint::black_box(&frame_out);
        },
        20,
    );
    let t_b = best_of(
        || unsafe {
            vDSP_vclip(
                frame_a.as_ptr(),
                1,
                &lo_c,
                &hi_c,
                frame_out2.as_mut_ptr(),
                1,
                pxu,
            );
            std::hint::black_box(&frame_out2);
        },
        20,
    );
    rows.push(Row {
        group: g,
        op: "clamp 1080p".into(),
        unit: "Mp/s",
        acpu: mpx / (t_a as f64 / 1e9),
        baseline: "Apple vDSP".into(),
        baseline_val: mpx / (t_b as f64 / 1e9),
    });

    // HDR convert f32→f16 — 1080p vs scalar
    let mut f16_frame = vec![0u16; px];
    let t_a = best_of(
        || {
            acpu::cast_f32_f16(&mut f16_frame, &frame_a);
            std::hint::black_box(&f16_frame);
        },
        20,
    );
    let t_b = best_of(
        || {
            for i in 0..px {
                f16_frame[i] = acpu::numeric::fp16::f32_to_fp16(frame_a[i]);
            }
            std::hint::black_box(&f16_frame);
        },
        5,
    );
    rows.push(Row {
        group: g,
        op: "f32→f16 1080p".into(),
        unit: "Mp/s",
        acpu: mpx / (t_a as f64 / 1e9),
        baseline: "scalar fp16".into(),
        baseline_val: mpx / (t_b as f64 / 1e9),
    });

    // motion estimation SAD — 64K block (256×256)
    let me_sz = 256 * 256usize;
    let me_mpx = me_sz as f64 / 1e6;
    let me_a: Vec<u8> = (0..me_sz).map(|i| (i % 200) as u8).collect();
    let me_b: Vec<u8> = (0..me_sz).map(|i| ((i * 3 + 50) % 200) as u8).collect();
    let t_a = best_of(
        || {
            std::hint::black_box(acpu::vector::integer_fused::sad_u8(&me_a, &me_b));
        },
        200,
    );
    let t_b = best_of(
        || {
            let s: u64 = me_a
                .iter()
                .zip(&me_b)
                .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u64)
                .sum();
            std::hint::black_box(s);
        },
        200,
    );
    rows.push(Row {
        group: g,
        op: "SAD 256×256 (ME)".into(),
        unit: "Mp/s",
        acpu: me_mpx / (t_a as f64 / 1e9),
        baseline: "auto-vec".into(),
        baseline_val: me_mpx / (t_b as f64 / 1e9),
    });

    // rsqrt for lighting — 1080p vs Apple vvrsqrtf
    let pos_frame: Vec<f32> = (0..px).map(|i| (i as f32 + 1.0) * 0.001).collect();
    let mut rsqrt_out = vec![0f32; px];
    let mut rsqrt_out2 = vec![0f32; px];
    let pxi = px as i32;
    let t_a = best_of(
        || {
            acpu::vector::render::rsqrt_to(&pos_frame, &mut rsqrt_out);
            std::hint::black_box(&rsqrt_out);
        },
        20,
    );
    let t_b = best_of(
        || unsafe {
            vvrsqrtf(rsqrt_out2.as_mut_ptr(), pos_frame.as_ptr(), &pxi);
            std::hint::black_box(&rsqrt_out2);
        },
        20,
    );
    rows.push(Row {
        group: g,
        op: "rsqrt 1080p".into(),
        unit: "Mp/s",
        acpu: mpx / (t_a as f64 / 1e9),
        baseline: "Apple vvrsqrtf".into(),
        baseline_val: mpx / (t_b as f64 / 1e9),
    });

    // RGB→YUV 1080p — vs scalar BT.601
    let rgb_data: Vec<u8> = (0..px * 3).map(|i| (i % 256) as u8).collect();
    let mut yuv_data = vec![0u8; px * 3];
    let mut yuv_data2 = vec![0u8; px * 3];
    let t_a = best_of(
        || {
            acpu::vector::rgb_to_yuv(&mut yuv_data, &rgb_data);
            std::hint::black_box(&yuv_data);
        },
        20,
    );
    let t_b = best_of(
        || {
            for i in 0..px {
                let off = i * 3;
                let (r, g, b) = (
                    rgb_data[off] as i32,
                    rgb_data[off + 1] as i32,
                    rgb_data[off + 2] as i32,
                );
                yuv_data2[off] = (16 + ((66 * r + 129 * g + 25 * b + 128) >> 8)) as u8;
                yuv_data2[off + 1] = (128 + ((-38 * r - 74 * g + 112 * b + 128) >> 8)) as u8;
                yuv_data2[off + 2] = (128 + ((112 * r - 94 * g - 18 * b + 128) >> 8)) as u8;
            }
            std::hint::black_box(&yuv_data2);
        },
        10,
    );
    rows.push(Row {
        group: g,
        op: "RGB→YUV 1080p".into(),
        unit: "Mp/s",
        acpu: mpx / (t_a as f64 / 1e9),
        baseline: "scalar BT601".into(),
        baseline_val: mpx / (t_b as f64 / 1e9),
    });

    // YUV→RGB 1080p — vs scalar
    acpu::vector::rgb_to_yuv(&mut yuv_data, &rgb_data);
    let mut rgb_out = vec![0u8; px * 3];
    let mut rgb_out2 = vec![0u8; px * 3];
    let t_a = best_of(
        || {
            acpu::vector::yuv_to_rgb(&mut rgb_out, &yuv_data);
            std::hint::black_box(&rgb_out);
        },
        20,
    );
    let t_b = best_of(
        || {
            for i in 0..px {
                let off = i * 3;
                let (y, u, v) = (
                    yuv_data[off] as i32,
                    yuv_data[off + 1] as i32,
                    yuv_data[off + 2] as i32,
                );
                let c = 298 * (y - 16);
                let d = u - 128;
                let e = v - 128;
                rgb_out2[off] = ((c + 409 * e + 128) >> 8).clamp(0, 255) as u8;
                rgb_out2[off + 1] = ((c - 100 * d - 208 * e + 128) >> 8).clamp(0, 255) as u8;
                rgb_out2[off + 2] = ((c + 516 * d + 128) >> 8).clamp(0, 255) as u8;
            }
            std::hint::black_box(&rgb_out2);
        },
        10,
    );
    rows.push(Row {
        group: g,
        op: "YUV→RGB 1080p".into(),
        unit: "Mp/s",
        acpu: mpx / (t_a as f64 / 1e9),
        baseline: "scalar BT601".into(),
        baseline_val: mpx / (t_b as f64 / 1e9),
    });

    // histogram 1080p — vs scalar
    let hist_data: Vec<u8> = (0..px).map(|i| (i % 256) as u8).collect();
    let mut hist = [0u32; 256];
    let mut hist2 = [0u32; 256];
    let t_a = best_of(
        || {
            acpu::vector::histogram_u8(&mut hist, &hist_data);
            std::hint::black_box(&hist);
        },
        50,
    );
    let t_b = best_of(
        || {
            hist2 = [0u32; 256];
            for &v in &hist_data {
                hist2[v as usize] += 1;
            }
            std::hint::black_box(&hist2);
        },
        50,
    );
    rows.push(Row {
        group: g,
        op: "histogram 1080p".into(),
        unit: "Mp/s",
        acpu: mpx / (t_a as f64 / 1e9),
        baseline: "scalar".into(),
        baseline_val: mpx / (t_b as f64 / 1e9),
    });

    // bilinear resize 1080p→540p — vs scalar
    let src_w = 1920usize;
    let src_h = 1080usize;
    let dst_w = 960usize;
    let dst_h = 540usize;
    let resize_src: Vec<f32> = (0..src_w * src_h)
        .map(|i| (i % 1000) as f32 * 0.001)
        .collect();
    let mut resize_dst = vec![0f32; dst_w * dst_h];
    let mut resize_dst2 = vec![0f32; dst_w * dst_h];
    let dst_mpx = (dst_w * dst_h) as f64 / 1e6;
    let t_a = best_of(
        || {
            acpu::vector::resize_bilinear_f32(
                &mut resize_dst,
                &resize_src,
                src_w,
                src_h,
                dst_w,
                dst_h,
            );
            std::hint::black_box(&resize_dst);
        },
        10,
    );
    let t_b = best_of(
        || {
            for dy in 0..dst_h {
                for dx in 0..dst_w {
                    let sx = dx as f32 * (src_w - 1) as f32 / (dst_w - 1) as f32;
                    let sy = dy as f32 * (src_h - 1) as f32 / (dst_h - 1) as f32;
                    let x0 = sx as usize;
                    let y0 = sy as usize;
                    let x1 = (x0 + 1).min(src_w - 1);
                    let y1 = (y0 + 1).min(src_h - 1);
                    let fx = sx - x0 as f32;
                    let fy = sy - y0 as f32;
                    let v = (1.0 - fx) * (1.0 - fy) * resize_src[y0 * src_w + x0]
                        + fx * (1.0 - fy) * resize_src[y0 * src_w + x1]
                        + (1.0 - fx) * fy * resize_src[y1 * src_w + x0]
                        + fx * fy * resize_src[y1 * src_w + x1];
                    resize_dst2[dy * dst_w + dx] = v;
                }
            }
            std::hint::black_box(&resize_dst2);
        },
        2,
    );
    rows.push(Row {
        group: g,
        op: "resize 1080→540p".into(),
        unit: "Mp/s",
        acpu: dst_mpx / (t_a as f64 / 1e9),
        baseline: "scalar bilin".into(),
        baseline_val: dst_mpx / (t_b as f64 / 1e9),
    });

    // ╔══════════════════════════════════════════════════════════════════════╗
    // ║  Crypto (vs CommonCrypto)                                          ║
    // ╚══════════════════════════════════════════════════════════════════════╝
    let g = "Crypto";

    // SHA-256
    let block = [0xAAu8; 64];
    let iters_sha = 10000u64;
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let t_a = best_of(
        || {
            for _ in 0..iters_sha {
                acpu::crypto::sha256_compress(&mut state, &block);
            }
            std::hint::black_box(&state);
        },
        5,
    );
    let mut dig = [0u8; 32];
    let t_b = best_of(
        || {
            for _ in 0..iters_sha {
                unsafe {
                    CC_SHA256(block.as_ptr(), 64, dig.as_mut_ptr());
                }
            }
            std::hint::black_box(&dig);
        },
        5,
    );
    let a_gbs = iters_sha as f64 * 64.0 / t_a as f64;
    let b_gbs = iters_sha as f64 * 64.0 / t_b as f64;
    rows.push(Row {
        group: g,
        op: "SHA-256 10K×64B".into(),
        unit: "GB/s",
        acpu: a_gbs,
        baseline: "CC_SHA256".into(),
        baseline_val: b_gbs,
    });

    // AES-128
    let aes_key: [u8; 16] = [
        0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f,
        0x3c,
    ];
    let round_keys = aes128_expand(&aes_key);
    let aes_count = 4096usize;
    let aes_len = aes_count * 16;
    let mut aes_blocks = vec![[0x42u8; 16]; aes_count];
    let t_a = best_of(
        || {
            acpu::crypto::aes_encrypt_blocks(&mut aes_blocks, &round_keys);
            std::hint::black_box(&aes_blocks);
        },
        50,
    );
    let plain = vec![0x42u8; aes_len];
    let mut cipher = vec![0u8; aes_len];
    let mut out_len: usize = 0;
    let t_b = best_of(
        || unsafe {
            CCCrypt(
                0,
                0,
                2,
                aes_key.as_ptr(),
                16,
                std::ptr::null(),
                plain.as_ptr(),
                aes_len,
                cipher.as_mut_ptr(),
                aes_len,
                &mut out_len,
            );
            std::hint::black_box(&cipher);
        },
        50,
    );
    let a_aes = aes_len as f64 / t_a as f64;
    let b_aes = aes_len as f64 / t_b as f64;
    rows.push(Row {
        group: g,
        op: "AES-128 4096blk".into(),
        unit: "GB/s",
        acpu: a_aes,
        baseline: "CCCrypt".into(),
        baseline_val: b_aes,
    });

    // PMULL (hardware vs scalar carry-less multiply)
    let pmull_iters = 1_000_000u64;
    let mut pmull_a = 0xDEADBEEFCAFEBABEu64;
    let pmull_b = 0x1234567890ABCDEFu64;
    let t_a = best_of(
        || {
            let mut acc = 0u128;
            let mut pa = pmull_a;
            for _ in 0..pmull_iters {
                acc = acc.wrapping_add(acpu::crypto::pmull_64(pa, pmull_b));
                pa = pa.wrapping_add(1);
            }
            std::hint::black_box(acc);
        },
        5,
    );
    let t_b = best_of(
        || {
            let mut acc = 0u128;
            let mut pa = pmull_a;
            for _ in 0..pmull_iters {
                // scalar carry-less multiply
                let mut r = 0u128;
                let b128 = pmull_b as u128;
                for bit in 0..64 {
                    if (pa >> bit) & 1 == 1 {
                        r ^= b128 << bit;
                    }
                }
                acc = acc.wrapping_add(r);
                pa = pa.wrapping_add(1);
            }
            std::hint::black_box(acc);
        },
        2,
    );
    let a_gbs_p = pmull_iters as f64 * 16.0 / t_a as f64;
    let b_gbs_p = pmull_iters as f64 * 16.0 / t_b as f64;
    rows.push(Row {
        group: g,
        op: "PMULL 1M×64b".into(),
        unit: "GB/s",
        acpu: a_gbs_p,
        baseline: "scalar".into(),
        baseline_val: b_gbs_p,
    });

    // ╔══════════════════════════════════════════════════════════════════════╗
    // ║  ZK Goldilocks (acpu asm vs nebu pure Rust cross-platform)          ║
    // ╚══════════════════════════════════════════════════════════════════════╝
    let g = "ZK Goldilocks";
    let ga: Vec<u64> = (0..n)
        .map(|i| (i as u64).wrapping_mul(0x9E3779B97F4A7C15) | 1)
        .collect();
    let gb: Vec<u64> = (0..n)
        .map(|i| (i as u64).wrapping_mul(0x6C62272E07BB0142) | 1)
        .collect();
    let mut gd = vec![0u64; n];
    let mut gd2 = vec![0u64; n];

    // field_mul batch: acpu interleaved vs nebu pure Rust
    let t_a = best_of(
        || {
            acpu::field::gl_mul_batch(&ga, &gb, &mut gd);
            std::hint::black_box(&gd);
        },
        200,
    );
    let t_b = best_of(
        || {
            gl_pure::mul_batch(&ga, &gb, &mut gd2);
            std::hint::black_box(&gd2);
        },
        200,
    );
    rows.push(Row {
        group: g,
        op: "field_mul batch 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "nebu pure".into(),
        baseline_val: t_b as f64,
    });

    // field_inv chain: acpu vs nebu pure Rust
    let t_a = ns(|| {
        let mut x = ga[0];
        for _ in 0..16 {
            x = acpu::field::gl_inv(x);
        }
        std::hint::black_box(x);
    });
    let t_b = ns(|| {
        let mut x = ga[0];
        for _ in 0..16 {
            x = gl_pure::inv(x);
        }
        std::hint::black_box(x);
    });
    rows.push(Row {
        group: g,
        op: "field_inv (chain 16)".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "nebu pure".into(),
        baseline_val: t_b as f64,
    });

    // pow7 x16: acpu batched vs nebu pure scalar
    let mut state16: [u64; 16] = core::array::from_fn(|i| ga[i]);
    let mut state16b: [u64; 16] = core::array::from_fn(|i| ga[i]);
    let t_a = ns(|| {
        acpu::field::gl_pow7_x16(&mut state16);
        std::hint::black_box(&state16);
    });
    let t_b = ns(|| {
        for i in 0..16 {
            state16b[i] = gl_pure::pow7(ga[i]);
        }
        std::hint::black_box(&state16b);
    });
    rows.push(Row {
        group: g,
        op: "pow7_x16 (S-box)".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "nebu pure".into(),
        baseline_val: t_b as f64,
    });

    // batch_inv: acpu vs nebu pure Rust Montgomery
    let mut inv_out = vec![0u64; n];
    let mut inv_out2 = vec![0u64; n];
    let t_a = best_of(
        || {
            acpu::field::batch_inv(&ga, &mut inv_out);
            std::hint::black_box(&inv_out);
        },
        50,
    );
    let t_b = best_of(
        || {
            gl_pure::batch_inv(&ga, &mut inv_out2);
            std::hint::black_box(&inv_out2);
        },
        50,
    );
    rows.push(Row {
        group: g,
        op: "batch_inv 4096".into(),
        unit: "ns",
        acpu: t_a as f64,
        baseline: "nebu pure".into(),
        baseline_val: t_b as f64,
    });

    // Poseidon2 permute: acpu vs theoretical floor
    let rc: [u64; 144] = core::array::from_fn(|i| (i as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15));
    let diag: [u64; 16] = core::array::from_fn(|i| (i as u64 + 2).wrapping_mul(0x517CC1B727220A95));
    let mut st: [u64; 16] = core::array::from_fn(|i| i as u64 + 1);
    let t_p = best_of(
        || {
            st = core::array::from_fn(|i| i as u64 + 1);
            acpu::field::poseidon2_permute(&mut st, &rc, &diag);
            std::hint::black_box(&st);
        },
        200,
    );
    rows.push(Row {
        group: g,
        op: "Poseidon2 permute".into(),
        unit: "ns",
        acpu: t_p as f64,
        baseline: "theoretical".into(),
        baseline_val: 4753.0,
    });

    // NTT: nebu+acpu vs pure Rust NTT (same algorithm, different field arithmetic)
    for &(ntt_sz, label) in &[
        (4096usize, "NTT 2^12 fwd"),
        (16384, "NTT 2^14 fwd"),
        (65536, "NTT 2^16 fwd"),
    ] {
        let raw: Vec<u64> = (0..ntt_sz)
            .map(|i| (i as u64).wrapping_mul(0x9E3779B97F4A7C15) % gl_pure::P)
            .collect();
        let mut ntt_data: Vec<nebu::Goldilocks> =
            raw.iter().map(|&v| nebu::Goldilocks::new(v)).collect();
        nebu::ntt::ntt(&mut ntt_data); // warmup
        let iters = if ntt_sz <= 4096 {
            20
        } else if ntt_sz <= 16384 {
            10
        } else {
            3
        };
        let t_a = best_of(
            || {
                ntt_data = raw.iter().map(|&v| nebu::Goldilocks::new(v)).collect();
                nebu::ntt::ntt(&mut ntt_data);
                std::hint::black_box(&ntt_data);
            },
            iters,
        );
        // pure Rust NTT baseline (same O(n log n), but gl_pure mul/add)
        let mut pure_data = raw.clone();
        gl_pure::ntt_pure(&mut pure_data); // warmup
        let pure_iters = if ntt_sz <= 4096 {
            10
        } else if ntt_sz <= 16384 {
            3
        } else {
            1
        };
        let t_b = best_of(
            || {
                pure_data = raw.clone();
                gl_pure::ntt_pure(&mut pure_data);
                std::hint::black_box(&pure_data);
            },
            pure_iters,
        );
        rows.push(Row {
            group: g,
            op: label.into(),
            unit: "μs",
            acpu: t_a as f64 / 1e3,
            baseline: "pure Rust".into(),
            baseline_val: t_b as f64 / 1e3,
        });
    }

    // ╔══════════════════════════════════════════════════════════════════════╗
    // ║  Memory BW                                                         ║
    // ╚══════════════════════════════════════════════════════════════════════╝
    let g = "Memory BW";
    let sn = 4 * 1024 * 1024usize;
    let sa: Vec<f32> = (0..sn).map(|i| (i % 1000) as f32 * 0.001).collect();
    let sb: Vec<f32> = (0..sn).map(|i| (i % 997) as f32 * 0.001).collect();
    let mut sc = vec![0f32; sn];
    let mut sd = vec![0f32; sn];
    let bytes = sn as f64 * 4.0;
    // already pinned to P-core at start

    // warmup
    for _ in 0..5 {
        sc.copy_from_slice(&sa);
    }

    // copy (read + write = 2× bytes)
    let t = best_of(
        || {
            sc.copy_from_slice(&sa);
            std::hint::black_box(&sc);
        },
        50,
    );
    rows.push(Row {
        group: g,
        op: "STREAM copy 16MB".into(),
        unit: "GB/s",
        acpu: 2.0 * bytes / t as f64,
        baseline: "ref M1Pro".into(),
        baseline_val: 96.0,
    });

    // scale (read + write = 2× bytes) — NEON + STNP
    let scalar = 3.14159f32;
    for _ in 0..10 {
        neon_stream_scale(&sa, &mut sc, scalar);
    }
    let t = best_of(
        || {
            neon_stream_scale(&sa, &mut sc, scalar);
            std::hint::black_box(&sc);
        },
        50,
    );
    rows.push(Row {
        group: g,
        op: "STREAM scale 16MB".into(),
        unit: "GB/s",
        acpu: 2.0 * bytes / t as f64,
        baseline: "ref M1Pro".into(),
        baseline_val: 64.0,
    });

    // add (2 reads + 1 write = 3× bytes) — NEON + STNP
    for _ in 0..10 {
        neon_stream_add(&sa, &sb, &mut sc);
    }
    let t = best_of(
        || {
            neon_stream_add(&sa, &sb, &mut sc);
            std::hint::black_box(&sc);
        },
        50,
    );
    rows.push(Row {
        group: g,
        op: "STREAM add 16MB".into(),
        unit: "GB/s",
        acpu: 3.0 * bytes / t as f64,
        baseline: "ref M1Pro".into(),
        baseline_val: 80.0,
    });

    // triad (2 reads + 1 write = 3× bytes) — NEON + STNP
    for _ in 0..10 {
        neon_stream_triad(&sa, &sb, &mut sd, scalar);
    }
    let t = best_of(
        || {
            neon_stream_triad(&sa, &sb, &mut sd, scalar);
            std::hint::black_box(&sd);
        },
        50,
    );
    rows.push(Row {
        group: g,
        op: "STREAM triad 16MB".into(),
        unit: "GB/s",
        acpu: 3.0 * bytes / t as f64,
        baseline: "ref M1Pro".into(),
        baseline_val: 80.0,
    });

    // ══════════ PRINT TABLE ══════════
    let _ = acpu::sync::affinity::pin_any();
    print_table(&caps, &rows);
}

// ── NEON STREAM kernels with STNP (non-temporal stores) ──

fn neon_stream_scale(a: &[f32], c: &mut [f32], scalar: f32) {
    let n = a.len();
    unsafe {
        use core::arch::aarch64::*;
        let sv = vdupq_n_f32(scalar);
        let pa = a.as_ptr();
        let pc = c.as_mut_ptr() as *mut u8;
        let mut i = 0;
        while i + 32 <= n {
            core::arch::asm!("prfm pldl1strm, [{pa}, #512]", pa = in(reg) pa.add(i));
            let r0 = vmulq_f32(sv, vld1q_f32(pa.add(i)));
            let r1 = vmulq_f32(sv, vld1q_f32(pa.add(i + 4)));
            let r2 = vmulq_f32(sv, vld1q_f32(pa.add(i + 8)));
            let r3 = vmulq_f32(sv, vld1q_f32(pa.add(i + 12)));
            let r4 = vmulq_f32(sv, vld1q_f32(pa.add(i + 16)));
            let r5 = vmulq_f32(sv, vld1q_f32(pa.add(i + 20)));
            let r6 = vmulq_f32(sv, vld1q_f32(pa.add(i + 24)));
            let r7 = vmulq_f32(sv, vld1q_f32(pa.add(i + 28)));
            core::arch::asm!(
                "stnp q0, q1, [{p}]", "stnp q2, q3, [{p}, #32]",
                "stnp q4, q5, [{p}, #64]", "stnp q6, q7, [{p}, #96]",
                p = in(reg) pc.add(i * 4),
                in("v0") r0, in("v1") r1, in("v2") r2, in("v3") r3,
                in("v4") r4, in("v5") r5, in("v6") r6, in("v7") r7,
            );
            i += 32;
        }
        while i + 4 <= n {
            vst1q_f32(
                pc.add(i * 4) as *mut f32,
                vmulq_f32(sv, vld1q_f32(pa.add(i))),
            );
            i += 4;
        }
    }
}

fn neon_stream_add(a: &[f32], b: &[f32], c: &mut [f32]) {
    let n = a.len();
    unsafe {
        use core::arch::aarch64::*;
        let pa = a.as_ptr();
        let pb = b.as_ptr();
        let pc = c.as_mut_ptr() as *mut u8;
        let mut i = 0;
        while i + 32 <= n {
            core::arch::asm!("prfm pldl1strm, [{pa}, #512]", "prfm pldl1strm, [{pb}, #512]",
                pa = in(reg) pa.add(i), pb = in(reg) pb.add(i));
            let r0 = vaddq_f32(vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            let r1 = vaddq_f32(vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
            let r2 = vaddq_f32(vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
            let r3 = vaddq_f32(vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
            let r4 = vaddq_f32(vld1q_f32(pa.add(i + 16)), vld1q_f32(pb.add(i + 16)));
            let r5 = vaddq_f32(vld1q_f32(pa.add(i + 20)), vld1q_f32(pb.add(i + 20)));
            let r6 = vaddq_f32(vld1q_f32(pa.add(i + 24)), vld1q_f32(pb.add(i + 24)));
            let r7 = vaddq_f32(vld1q_f32(pa.add(i + 28)), vld1q_f32(pb.add(i + 28)));
            core::arch::asm!(
                "stnp q0, q1, [{p}]", "stnp q2, q3, [{p}, #32]",
                "stnp q4, q5, [{p}, #64]", "stnp q6, q7, [{p}, #96]",
                p = in(reg) pc.add(i * 4),
                in("v0") r0, in("v1") r1, in("v2") r2, in("v3") r3,
                in("v4") r4, in("v5") r5, in("v6") r6, in("v7") r7,
            );
            i += 32;
        }
        while i + 4 <= n {
            vst1q_f32(
                pc.add(i * 4) as *mut f32,
                vaddq_f32(vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i))),
            );
            i += 4;
        }
    }
}

fn neon_stream_triad(a: &[f32], b: &[f32], d: &mut [f32], scalar: f32) {
    let n = a.len();
    unsafe {
        use core::arch::aarch64::*;
        let sv = vdupq_n_f32(scalar);
        let pa = a.as_ptr();
        let pb = b.as_ptr();
        let pd = d.as_mut_ptr() as *mut u8;
        let mut i = 0;
        while i + 32 <= n {
            core::arch::asm!("prfm pldl1strm, [{pa}, #512]", "prfm pldl1strm, [{pb}, #512]",
                pa = in(reg) pa.add(i), pb = in(reg) pb.add(i));
            let r0 = vfmaq_f32(vld1q_f32(pa.add(i)), sv, vld1q_f32(pb.add(i)));
            let r1 = vfmaq_f32(vld1q_f32(pa.add(i + 4)), sv, vld1q_f32(pb.add(i + 4)));
            let r2 = vfmaq_f32(vld1q_f32(pa.add(i + 8)), sv, vld1q_f32(pb.add(i + 8)));
            let r3 = vfmaq_f32(vld1q_f32(pa.add(i + 12)), sv, vld1q_f32(pb.add(i + 12)));
            let r4 = vfmaq_f32(vld1q_f32(pa.add(i + 16)), sv, vld1q_f32(pb.add(i + 16)));
            let r5 = vfmaq_f32(vld1q_f32(pa.add(i + 20)), sv, vld1q_f32(pb.add(i + 20)));
            let r6 = vfmaq_f32(vld1q_f32(pa.add(i + 24)), sv, vld1q_f32(pb.add(i + 24)));
            let r7 = vfmaq_f32(vld1q_f32(pa.add(i + 28)), sv, vld1q_f32(pb.add(i + 28)));
            core::arch::asm!(
                "stnp q0, q1, [{p}]", "stnp q2, q3, [{p}, #32]",
                "stnp q4, q5, [{p}, #64]", "stnp q6, q7, [{p}, #96]",
                p = in(reg) pd.add(i * 4),
                in("v0") r0, in("v1") r1, in("v2") r2, in("v3") r3,
                in("v4") r4, in("v5") r5, in("v6") r6, in("v7") r7,
            );
            i += 32;
        }
        while i + 4 <= n {
            vst1q_f32(
                pd.add(i * 4) as *mut f32,
                vfmaq_f32(vld1q_f32(pa.add(i)), sv, vld1q_f32(pb.add(i))),
            );
            i += 4;
        }
    }
}

fn fmt_val(v: f64) -> String {
    if v >= 1_000_000.0 {
        format!("{:.0}", v)
    } else if v >= 10_000.0 {
        format!("{:.0}", v)
    } else if v >= 100.0 {
        format!("{:.1}", v)
    } else if v >= 1.0 {
        format!("{:.1}", v)
    } else {
        format!("{:.2}", v)
    }
}

fn fmt_ratio(r: f64) -> String {
    if r >= 100.0 {
        format!("{:.0}×", r)
    } else if r >= 10.0 {
        format!("{:.1}×", r)
    } else {
        format!("{:.2}×", r)
    }
}

fn print_table(caps: &acpu::Features, rows: &[Row]) {
    // column widths: op=24  unit=4  acpu=11  vs=16  base=11  ratio=7  mark=2
    let w = 87;
    println!();
    println!("╔{:═<w$}╗", "");
    let chip = format!("{:?}", caps.chip);
    let title = format!(
        "  acpu benchmark summary — {} ({}P+{}E)   ",
        chip, caps.p_cores, caps.e_cores
    );
    println!("║{:<w$}║", title);
    println!("╠{:═<w$}╣", "");
    println!(
        "║ {:<24} {:<4} {:>11} {:>16} {:>11} {:>7} {:>4}   ║",
        "operation", "unit", "acpu", "vs", "baseline", "ratio", ""
    );
    println!("╠{:═<w$}╣", "");

    let mut prev_group = "";
    let mut wins = 0u32;
    let mut ties = 0u32;
    let mut losses = 0u32;
    let mut na_count = 0u32;
    for r in rows {
        if r.group != prev_group {
            if !prev_group.is_empty() {
                println!("║{:─<w$}║", "");
            }
            println!("║ {:<pad$} ║", r.group, pad = w - 2);
            prev_group = r.group;
        }
        let ratio = if r.baseline_val > 0.0 {
            if r.unit == "GF" || r.unit == "GB/s" || r.unit == "GF/GB" || r.unit == "Mp/s" {
                r.acpu / r.baseline_val
            } else {
                r.baseline_val / r.acpu
            }
        } else {
            0.0
        };

        let mark = if r.baseline_val == 0.0 {
            na_count += 1;
            "—"
        } else if ratio >= 1.05 {
            wins += 1;
            "←"
        } else if ratio >= 0.95 {
            ties += 1;
            "≈"
        } else {
            losses += 1;
            ""
        };

        let acpu_s = fmt_val(r.acpu);
        if r.baseline_val > 0.0 {
            let base_s = fmt_val(r.baseline_val);
            let ratio_s = fmt_ratio(ratio);
            println!(
                "║   {:<24} {:<4} {:>11} {:>16} {:>11} {:>7} {:>2}   ║",
                r.op, r.unit, acpu_s, r.baseline, base_s, ratio_s, mark
            );
        } else {
            println!(
                "║   {:<24} {:<4} {:>11} {:>16} {:>11} {:>7} {:>2}   ║",
                r.op, r.unit, acpu_s, "—", "—", "—", mark
            );
        }
    }

    let total = wins + ties + losses;
    println!("╠{:═<w$}╣", "");
    let summary = format!(
        "  TOTAL: {} wins, {} ties, {} losses ({} compared, {} absolute)",
        wins, ties, losses, total, na_count
    );
    println!("║{:<w$}║", summary);
    println!("║{:<w$}║", "  ← = acpu faster, ≈ = parity (±5%)");
    println!("╚{:═<w$}╝", "");
}

// AES-128 key expansion
fn aes128_expand(key: &[u8; 16]) -> Vec<[u8; 16]> {
    let rcon: [u8; 10] = [1, 2, 4, 8, 16, 32, 64, 128, 0x1b, 0x36];
    #[rustfmt::skip]
    const S: [u8; 256] = [
        0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
        0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
        0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
        0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
        0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
        0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
        0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
        0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
        0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
        0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
        0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
        0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
        0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
        0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
        0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
        0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
    ];
    fn sw(w: u32) -> u32 {
        let b = w.to_be_bytes();
        u32::from_be_bytes([
            S[b[0] as usize],
            S[b[1] as usize],
            S[b[2] as usize],
            S[b[3] as usize],
        ])
    }
    let mut w = vec![0u32; 44];
    for i in 0..4 {
        w[i] = u32::from_be_bytes([key[4 * i], key[4 * i + 1], key[4 * i + 2], key[4 * i + 3]]);
    }
    for i in 4..44 {
        let mut t = w[i - 1];
        if i % 4 == 0 {
            t = sw(t.rotate_left(8)) ^ ((rcon[i / 4 - 1] as u32) << 24);
        }
        w[i] = w[i - 4] ^ t;
    }
    (0..11)
        .map(|r| {
            let mut rk = [0u8; 16];
            for j in 0..4 {
                rk[4 * j..4 * j + 4].copy_from_slice(&w[4 * r + j].to_be_bytes());
            }
            rk
        })
        .collect()
}
