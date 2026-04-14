//! Elementwise operations: acpu vs Apple Accelerate (4096 f32).
#[path = "common.rs"]
mod common;
use common::*;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {}

fn main() {
    let n: usize = 4096;
    let ni = n as i32;
    let nu = n as u64;

    // --- setup vectors ---
    let src: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) / n as f32).collect(); // (0,1]
    let pos: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 20.0).collect(); // [-20,+20]
    let b: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.7 + 0.3).sin()).collect();
    let d: Vec<f32> = (0..n).map(|i| ((i as f32) * 1.3).cos()).collect();
    let d2: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.5 + 1.0).sin().abs() + 0.1)
        .collect();

    let mut score = Score::new();

    // =====================================================================
    // ELEMENTWISE
    // =====================================================================
    score.hdr("ELEMENTWISE (4096 f32)");

    // --- exp ---
    {
        let mut dst = vec![0.0f32; n];
        let t_acpu = ns(|| acpu::vector::math::exp_to(&pos, &mut dst));
        let mut out = vec![0.0f32; n];
        let t_apple = ns(|| unsafe { vvexpf(out.as_mut_ptr(), pos.as_ptr(), &ni) });
        score.row("exp", t_acpu, t_apple);
    }

    // --- log ---
    {
        let mut dst = vec![0.0f32; n];
        let t_acpu = ns(|| acpu::vector::math::log_to(&src, &mut dst));
        let mut out = vec![0.0f32; n];
        let t_apple = ns(|| unsafe { vvlogf(out.as_mut_ptr(), src.as_ptr(), &ni) });
        score.row("log", t_acpu, t_apple);
    }

    // --- tanh ---
    {
        let mut buf = pos.clone();
        let t_acpu = ns(|| {
            buf.copy_from_slice(&pos);
            acpu::vector::math::tanh(&mut buf);
        });
        let mut out = vec![0.0f32; n];
        let t_apple = ns(|| unsafe { vvtanhf(out.as_mut_ptr(), pos.as_ptr(), &ni) });
        score.row("tanh", t_acpu, t_apple);
    }

    // --- sigmoid ---
    {
        let mut buf = pos.clone();
        let t_acpu = ns(|| {
            buf.copy_from_slice(&pos);
            acpu::vector::math::sigmoid(&mut buf);
        });
        // Apple: sigmoid(x) = 1 / (1 + exp(-x))
        let mut neg = vec![0.0f32; n];
        let mut expn = vec![0.0f32; n];
        let one = 1.0f32;
        let mut out = vec![0.0f32; n];
        let t_apple = ns(|| unsafe {
            vDSP_vneg(pos.as_ptr(), 1, neg.as_mut_ptr(), 1, nu);
            vvexpf(expn.as_mut_ptr(), neg.as_ptr(), &ni);
            vDSP_vsadd(expn.as_ptr(), 1, &one, expn.as_mut_ptr(), 1, nu);
            vDSP_svdiv(&one, expn.as_ptr(), 1, out.as_mut_ptr(), 1, nu);
        });
        score.row("sigmoid", t_acpu, t_apple);
    }

    // --- gelu ---
    {
        let mut buf = pos.clone();
        let t_acpu = ns(|| {
            buf.copy_from_slice(&pos);
            acpu::vector::math::gelu(&mut buf);
        });
        // Apple: gelu(x) ~ 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
        // approximate with: tanh path
        let mut tmp = vec![0.0f32; n];
        let mut out = vec![0.0f32; n];
        let half = 0.5f32;
        let t_apple = ns(|| unsafe {
            // x^3 -> tmp
            vDSP_vmul(pos.as_ptr(), 1, pos.as_ptr(), 1, tmp.as_mut_ptr(), 1, nu);
            vDSP_vmul(tmp.as_ptr(), 1, pos.as_ptr(), 1, tmp.as_mut_ptr(), 1, nu);
            // 0.044715 * x^3
            let c = 0.044715f32;
            vDSP_vsmul(tmp.as_ptr(), 1, &c, tmp.as_mut_ptr(), 1, nu);
            // x + 0.044715*x^3
            vDSP_vadd(pos.as_ptr(), 1, tmp.as_ptr(), 1, tmp.as_mut_ptr(), 1, nu);
            // sqrt(2/pi) * (...)
            let s2pi = 0.7978845608f32;
            vDSP_vsmul(tmp.as_ptr(), 1, &s2pi, tmp.as_mut_ptr(), 1, nu);
            // tanh
            vvtanhf(out.as_mut_ptr(), tmp.as_ptr(), &ni);
            // 1 + tanh(...)
            let one = 1.0f32;
            vDSP_vsadd(out.as_ptr(), 1, &one, out.as_mut_ptr(), 1, nu);
            // 0.5 * x
            vDSP_vsmul(pos.as_ptr(), 1, &half, tmp.as_mut_ptr(), 1, nu);
            // result = 0.5*x * (1+tanh(...))
            vDSP_vmul(tmp.as_ptr(), 1, out.as_ptr(), 1, out.as_mut_ptr(), 1, nu);
        });
        score.row("gelu", t_acpu, t_apple);
    }

    // --- silu ---
    {
        let mut buf = pos.clone();
        let t_acpu = ns(|| {
            buf.copy_from_slice(&pos);
            acpu::vector::math::silu(&mut buf);
        });
        // Apple: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        let mut neg = vec![0.0f32; n];
        let mut expn = vec![0.0f32; n];
        let one = 1.0f32;
        let mut sig = vec![0.0f32; n];
        let mut out = vec![0.0f32; n];
        let t_apple = ns(|| unsafe {
            vDSP_vneg(pos.as_ptr(), 1, neg.as_mut_ptr(), 1, nu);
            vvexpf(expn.as_mut_ptr(), neg.as_ptr(), &ni);
            vDSP_vsadd(expn.as_ptr(), 1, &one, expn.as_mut_ptr(), 1, nu);
            vDSP_svdiv(&one, expn.as_ptr(), 1, sig.as_mut_ptr(), 1, nu);
            vDSP_vmul(pos.as_ptr(), 1, sig.as_ptr(), 1, out.as_mut_ptr(), 1, nu);
        });
        score.row("silu", t_acpu, t_apple);
    }

    // =====================================================================
    // REDUCTIONS
    // =====================================================================
    score.hdr("REDUCTIONS (4096 f32)");

    // --- sum ---
    {
        let t_acpu = ns(|| {
            std::hint::black_box(acpu::vector::reduce::sum(&src));
        });
        let mut result = 0.0f32;
        let t_apple = ns(|| unsafe {
            vDSP_sve(src.as_ptr(), 1, &mut result, nu);
            std::hint::black_box(result);
        });
        score.row("sum", t_acpu, t_apple);
    }

    // --- dot ---
    {
        let t_acpu = ns(|| {
            std::hint::black_box(acpu::vector::reduce::dot(&b, &d));
        });
        let t_apple = ns(|| unsafe {
            std::hint::black_box(cblas_sdot(ni, b.as_ptr(), 1, d.as_ptr(), 1));
        });
        score.row("dot", t_acpu, t_apple);
    }

    // --- length (L2 norm) ---
    {
        let t_acpu = ns(|| {
            std::hint::black_box(acpu::vector::reduce::length(&src));
        });
        let t_apple = ns(|| unsafe {
            std::hint::black_box(cblas_snrm2(ni, src.as_ptr(), 1));
        });
        score.row("length", t_acpu, t_apple);
    }

    // --- max ---
    {
        let t_acpu = ns(|| {
            std::hint::black_box(acpu::vector::reduce::max(&src));
        });
        let mut result = 0.0f32;
        let t_apple = ns(|| unsafe {
            vDSP_maxv(src.as_ptr(), 1, &mut result, nu);
            std::hint::black_box(result);
        });
        score.row("max", t_acpu, t_apple);
    }

    // --- min ---
    {
        let t_acpu = ns(|| {
            std::hint::black_box(acpu::vector::reduce::min(&src));
        });
        let mut result = 0.0f32;
        let t_apple = ns(|| unsafe {
            vDSP_minv(src.as_ptr(), 1, &mut result, nu);
            std::hint::black_box(result);
        });
        score.row("min", t_acpu, t_apple);
    }

    // =====================================================================
    // COMPOUND
    // =====================================================================
    score.hdr("COMPOUND (4096 f32)");

    // --- softmax ---
    {
        let mut buf = pos.clone();
        let t_acpu = ns(|| {
            buf.copy_from_slice(&pos);
            acpu::vector::softmax::softmax(&mut buf);
        });
        let mut out = vec![0.0f32; n];
        let mut mx = 0.0f32;
        let mut s = 0.0f32;
        let t_apple = ns(|| unsafe {
            // max
            vDSP_maxv(pos.as_ptr(), 1, &mut mx, nu);
            // subtract max
            let neg_mx = -mx;
            vDSP_vsadd(pos.as_ptr(), 1, &neg_mx, out.as_mut_ptr(), 1, nu);
            // exp
            vvexpf(out.as_mut_ptr(), out.as_ptr(), &ni);
            // sum
            vDSP_sve(out.as_ptr(), 1, &mut s, nu);
            // divide
            vDSP_vsdiv(out.as_ptr(), 1, &s, out.as_mut_ptr(), 1, nu);
        });
        score.row("softmax", t_acpu, t_apple);
    }

    // --- normalize (RMS norm) ---
    {
        let weight = d2.clone();
        let mut out_acpu = vec![0.0f32; n];
        let t_acpu = ns(|| {
            acpu::vector::softmax::normalize(&mut out_acpu, &src, &weight, 1e-5);
        });
        let mut out_apple = vec![0.0f32; n];
        let mut ss = 0.0f32;
        let t_apple = ns(|| unsafe {
            // sum of squares
            vDSP_svesq(src.as_ptr(), 1, &mut ss, nu);
            // mean + eps -> inv_rms
            let inv_rms = 1.0 / (ss / n as f32 + 1e-5f32).sqrt();
            // scale by inv_rms
            vDSP_vsmul(src.as_ptr(), 1, &inv_rms, out_apple.as_mut_ptr(), 1, nu);
            // multiply by weight
            vDSP_vmul(
                out_apple.as_ptr(),
                1,
                weight.as_ptr(),
                1,
                out_apple.as_mut_ptr(),
                1,
                nu,
            );
        });
        score.row("normalize", t_acpu, t_apple);
    }

    // =====================================================================
    score.summary();
}
