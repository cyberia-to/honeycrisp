// ---------------------------------------------------------------------------
// Rendering-focused NEON operations -- rsqrt, recip, clamp, lerp, cross3
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

// ---------------------------------------------------------------------------
// rsqrt: fast inverse square root via FRSQRTE + 1 Newton-Raphson step
// ---------------------------------------------------------------------------

/// In-place fast inverse square root: x[i] = 1/sqrt(x[i]).
/// Uses NEON FRSQRTE + one Newton-Raphson refinement step.
pub fn rsqrt(x: &mut [f32]) {
    let len = x.len();

    #[cfg(target_arch = "aarch64")]
    {
        let mut i = 0;
        unsafe {
            let p = x.as_mut_ptr();
            while i + 16 <= len {
                let v0 = vld1q_f32(p.add(i));
                let v1 = vld1q_f32(p.add(i + 4));
                let v2 = vld1q_f32(p.add(i + 8));
                let v3 = vld1q_f32(p.add(i + 12));
                let e0 = vrsqrteq_f32(v0);
                let e1 = vrsqrteq_f32(v1);
                let e2 = vrsqrteq_f32(v2);
                let e3 = vrsqrteq_f32(v3);
                let r0 = vmulq_f32(e0, vrsqrtsq_f32(vmulq_f32(v0, e0), e0));
                let r1 = vmulq_f32(e1, vrsqrtsq_f32(vmulq_f32(v1, e1), e1));
                let r2 = vmulq_f32(e2, vrsqrtsq_f32(vmulq_f32(v2, e2), e2));
                let r3 = vmulq_f32(e3, vrsqrtsq_f32(vmulq_f32(v3, e3), e3));
                vst1q_f32(p.add(i), r0);
                vst1q_f32(p.add(i + 4), r1);
                vst1q_f32(p.add(i + 8), r2);
                vst1q_f32(p.add(i + 12), r3);
                i += 16;
            }
            while i + 4 <= len {
                let v = vld1q_f32(p.add(i));
                let e = vrsqrteq_f32(v);
                let r = vmulq_f32(e, vrsqrtsq_f32(vmulq_f32(v, e), e));
                vst1q_f32(p.add(i), r);
                i += 4;
            }
        }
        while i < len {
            x[i] = 1.0 / x[i].sqrt();
            i += 1;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    for v in x.iter_mut() {
        *v = 1.0 / v.sqrt();
    }
}

/// Fast inverse square root into separate buffer: dst[i] = 1/sqrt(src[i]).
/// Panics if lengths differ.
pub fn rsqrt_to(src: &[f32], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len(), "rsqrt_to: length mismatch");
    let len = src.len();

    #[cfg(target_arch = "aarch64")]
    {
        let mut i = 0;
        unsafe {
            let ps = src.as_ptr();
            let pd = dst.as_mut_ptr();
            while i + 16 <= len {
                let v0 = vld1q_f32(ps.add(i));
                let v1 = vld1q_f32(ps.add(i + 4));
                let v2 = vld1q_f32(ps.add(i + 8));
                let v3 = vld1q_f32(ps.add(i + 12));
                let e0 = vrsqrteq_f32(v0);
                let e1 = vrsqrteq_f32(v1);
                let e2 = vrsqrteq_f32(v2);
                let e3 = vrsqrteq_f32(v3);
                let r0 = vmulq_f32(e0, vrsqrtsq_f32(vmulq_f32(v0, e0), e0));
                let r1 = vmulq_f32(e1, vrsqrtsq_f32(vmulq_f32(v1, e1), e1));
                let r2 = vmulq_f32(e2, vrsqrtsq_f32(vmulq_f32(v2, e2), e2));
                let r3 = vmulq_f32(e3, vrsqrtsq_f32(vmulq_f32(v3, e3), e3));
                vst1q_f32(pd.add(i), r0);
                vst1q_f32(pd.add(i + 4), r1);
                vst1q_f32(pd.add(i + 8), r2);
                vst1q_f32(pd.add(i + 12), r3);
                i += 16;
            }
            while i + 4 <= len {
                let v = vld1q_f32(ps.add(i));
                let e = vrsqrteq_f32(v);
                let r = vmulq_f32(e, vrsqrtsq_f32(vmulq_f32(v, e), e));
                vst1q_f32(pd.add(i), r);
                i += 4;
            }
        }
        while i < len {
            dst[i] = 1.0 / src[i].sqrt();
            i += 1;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = 1.0 / s.sqrt();
    }
}

// ---------------------------------------------------------------------------
// recip: fast reciprocal via FRECPE + 1 Newton-Raphson step
// ---------------------------------------------------------------------------

/// In-place fast reciprocal: x[i] = 1/x[i].
/// Uses NEON FRECPE + one Newton-Raphson refinement step.
pub fn recip(x: &mut [f32]) {
    let len = x.len();

    #[cfg(target_arch = "aarch64")]
    {
        let mut i = 0;
        unsafe {
            let p = x.as_mut_ptr();
            while i + 16 <= len {
                let v0 = vld1q_f32(p.add(i));
                let v1 = vld1q_f32(p.add(i + 4));
                let v2 = vld1q_f32(p.add(i + 8));
                let v3 = vld1q_f32(p.add(i + 12));
                let e0 = vrecpeq_f32(v0);
                let e1 = vrecpeq_f32(v1);
                let e2 = vrecpeq_f32(v2);
                let e3 = vrecpeq_f32(v3);
                let r0 = vmulq_f32(e0, vrecpsq_f32(v0, e0));
                let r1 = vmulq_f32(e1, vrecpsq_f32(v1, e1));
                let r2 = vmulq_f32(e2, vrecpsq_f32(v2, e2));
                let r3 = vmulq_f32(e3, vrecpsq_f32(v3, e3));
                vst1q_f32(p.add(i), r0);
                vst1q_f32(p.add(i + 4), r1);
                vst1q_f32(p.add(i + 8), r2);
                vst1q_f32(p.add(i + 12), r3);
                i += 16;
            }
            while i + 4 <= len {
                let v = vld1q_f32(p.add(i));
                let e = vrecpeq_f32(v);
                let r = vmulq_f32(e, vrecpsq_f32(v, e));
                vst1q_f32(p.add(i), r);
                i += 4;
            }
        }
        while i < len {
            x[i] = 1.0 / x[i];
            i += 1;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    for v in x.iter_mut() {
        *v = 1.0 / *v;
    }
}

/// Reciprocal 1/x: src → dst (out-of-place, matches Apple vvrecf API).
pub fn recip_to(src: &[f32], dst: &mut [f32]) {
    let len = src.len().min(dst.len());
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let ps = src.as_ptr();
            let pd = dst.as_mut_ptr();
            while i + 16 <= len {
                let v0 = vld1q_f32(ps.add(i));
                let v1 = vld1q_f32(ps.add(i + 4));
                let v2 = vld1q_f32(ps.add(i + 8));
                let v3 = vld1q_f32(ps.add(i + 12));
                let e0 = vrecpeq_f32(v0);
                let e1 = vrecpeq_f32(v1);
                let e2 = vrecpeq_f32(v2);
                let e3 = vrecpeq_f32(v3);
                vst1q_f32(pd.add(i), vmulq_f32(e0, vrecpsq_f32(v0, e0)));
                vst1q_f32(pd.add(i + 4), vmulq_f32(e1, vrecpsq_f32(v1, e1)));
                vst1q_f32(pd.add(i + 8), vmulq_f32(e2, vrecpsq_f32(v2, e2)));
                vst1q_f32(pd.add(i + 12), vmulq_f32(e3, vrecpsq_f32(v3, e3)));
                i += 16;
            }
        }
    }

    while i < len {
        dst[i] = 1.0 / src[i];
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// clamp: clamp to [lo, hi] range
// ---------------------------------------------------------------------------

/// Clamp every element to [lo, hi] in-place.
/// Uses NEON FMAX + FMIN instructions.
pub fn clamp(x: &mut [f32], lo: f32, hi: f32) {
    let len = x.len();

    #[cfg(target_arch = "aarch64")]
    {
        let mut i = 0;
        unsafe {
            let p = x.as_mut_ptr();
            let vlo = vdupq_n_f32(lo);
            let vhi = vdupq_n_f32(hi);
            while i + 16 <= len {
                let r0 = vminq_f32(vmaxq_f32(vld1q_f32(p.add(i)), vlo), vhi);
                let r1 = vminq_f32(vmaxq_f32(vld1q_f32(p.add(i + 4)), vlo), vhi);
                let r2 = vminq_f32(vmaxq_f32(vld1q_f32(p.add(i + 8)), vlo), vhi);
                let r3 = vminq_f32(vmaxq_f32(vld1q_f32(p.add(i + 12)), vlo), vhi);
                vst1q_f32(p.add(i), r0);
                vst1q_f32(p.add(i + 4), r1);
                vst1q_f32(p.add(i + 8), r2);
                vst1q_f32(p.add(i + 12), r3);
                i += 16;
            }
            while i + 4 <= len {
                let r = vminq_f32(vmaxq_f32(vld1q_f32(p.add(i)), vlo), vhi);
                vst1q_f32(p.add(i), r);
                i += 4;
            }
        }
        while i < len {
            x[i] = x[i].max(lo).min(hi);
            i += 1;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    for v in x.iter_mut() {
        *v = v.max(lo).min(hi);
    }
}

/// Clamp src → dst (out-of-place, matches Apple vDSP_vclip API).
pub fn clamp_to(src: &[f32], dst: &mut [f32], lo: f32, hi: f32) {
    let len = src.len().min(dst.len());
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let ps = src.as_ptr();
            let pd = dst.as_mut_ptr();
            let vlo = vdupq_n_f32(lo);
            let vhi = vdupq_n_f32(hi);
            while i + 16 <= len {
                vst1q_f32(
                    pd.add(i),
                    vminq_f32(vmaxq_f32(vld1q_f32(ps.add(i)), vlo), vhi),
                );
                vst1q_f32(
                    pd.add(i + 4),
                    vminq_f32(vmaxq_f32(vld1q_f32(ps.add(i + 4)), vlo), vhi),
                );
                vst1q_f32(
                    pd.add(i + 8),
                    vminq_f32(vmaxq_f32(vld1q_f32(ps.add(i + 8)), vlo), vhi),
                );
                vst1q_f32(
                    pd.add(i + 12),
                    vminq_f32(vmaxq_f32(vld1q_f32(ps.add(i + 12)), vlo), vhi),
                );
                i += 16;
            }
        }
    }

    while i < len {
        dst[i] = src[i].max(lo).min(hi);
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// lerp: linear interpolation dst = a + t*(b - a)
// ---------------------------------------------------------------------------

/// Linear interpolation: dst[i] = a[i] + t * (b[i] - a[i]).
/// Panics if lengths differ.
pub fn lerp(dst: &mut [f32], a: &[f32], b: &[f32], t: f32) {
    assert_eq!(a.len(), b.len(), "lerp: a/b length mismatch");
    assert_eq!(a.len(), dst.len(), "lerp: dst length mismatch");
    let len = a.len();

    #[cfg(target_arch = "aarch64")]
    {
        let mut i = 0;
        unsafe {
            let pa = a.as_ptr();
            let pb = b.as_ptr();
            let pd = dst.as_mut_ptr();
            let vt = vdupq_n_f32(t);
            while i + 16 <= len {
                let a0 = vld1q_f32(pa.add(i));
                let a1 = vld1q_f32(pa.add(i + 4));
                let a2 = vld1q_f32(pa.add(i + 8));
                let a3 = vld1q_f32(pa.add(i + 12));
                let d0 = vsubq_f32(vld1q_f32(pb.add(i)), a0);
                let d1 = vsubq_f32(vld1q_f32(pb.add(i + 4)), a1);
                let d2 = vsubq_f32(vld1q_f32(pb.add(i + 8)), a2);
                let d3 = vsubq_f32(vld1q_f32(pb.add(i + 12)), a3);
                vst1q_f32(pd.add(i), vfmaq_f32(a0, d0, vt));
                vst1q_f32(pd.add(i + 4), vfmaq_f32(a1, d1, vt));
                vst1q_f32(pd.add(i + 8), vfmaq_f32(a2, d2, vt));
                vst1q_f32(pd.add(i + 12), vfmaq_f32(a3, d3, vt));
                i += 16;
            }
            while i + 4 <= len {
                let va = vld1q_f32(pa.add(i));
                let vd = vsubq_f32(vld1q_f32(pb.add(i)), va);
                vst1q_f32(pd.add(i), vfmaq_f32(va, vd, vt));
                i += 4;
            }
        }
        while i < len {
            dst[i] = a[i] + t * (b[i] - a[i]);
            i += 1;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    for ((d, &av), &bv) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
        *d = av + t * (bv - av);
    }
}

// ---------------------------------------------------------------------------
// cross3: cross product of 3D vectors
// ---------------------------------------------------------------------------

/// Cross product of two 3D vectors.
#[inline]
pub fn cross3(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsqrt_accuracy() {
        let src: Vec<f32> = (1..=128).map(|i| i as f32).collect();
        let mut buf = src.clone();
        rsqrt(&mut buf);
        for (i, &v) in buf.iter().enumerate() {
            let exact = 1.0 / (src[i].sqrt());
            let rel = (v - exact).abs() / exact;
            assert!(
                rel < 0.001,
                "rsqrt[{}]: got {}, want {}, rel {}",
                i,
                v,
                exact,
                rel
            );
        }
    }

    #[test]
    fn test_rsqrt_to() {
        let src: Vec<f32> = (1..=64).map(|i| i as f32 * 0.5).collect();
        let mut dst = vec![0.0f32; src.len()];
        rsqrt_to(&src, &mut dst);
        for (i, &v) in dst.iter().enumerate() {
            let exact = 1.0 / src[i].sqrt();
            let rel = (v - exact).abs() / exact;
            assert!(rel < 0.001, "rsqrt_to[{}]: rel err {}", i, rel);
        }
    }

    #[test]
    fn test_recip_accuracy() {
        let orig: Vec<f32> = (1..=128).map(|i| i as f32 * 0.1 + 0.5).collect();
        let mut buf = orig.clone();
        recip(&mut buf);
        for (i, &v) in buf.iter().enumerate() {
            let exact = 1.0 / orig[i];
            let rel = (v - exact).abs() / exact.abs();
            assert!(
                rel < 0.001,
                "recip[{}]: got {}, want {}, rel {}",
                i,
                v,
                exact,
                rel
            );
        }
    }

    #[test]
    fn test_clamp_bounds() {
        let mut x: Vec<f32> = (-10..=10).map(|i| i as f32).collect();
        clamp(&mut x, -3.0, 5.0);
        for &v in &x {
            assert!(v >= -3.0, "clamp below lo: {}", v);
            assert!(v <= 5.0, "clamp above hi: {}", v);
        }
        assert_eq!(x[0], -3.0); // -10 clamped to -3
        assert_eq!(x[20], 5.0); // 10 clamped to 5
        assert_eq!(x[10], 0.0); // 0 untouched
    }

    #[test]
    fn test_lerp_endpoints() {
        let n = 64;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32) * 10.0).collect();
        let mut dst = vec![0.0f32; n];

        // t=0 -> a
        lerp(&mut dst, &a, &b, 0.0);
        for i in 0..n {
            assert!((dst[i] - a[i]).abs() < 1e-5, "lerp t=0 [{}]", i);
        }

        // t=1 -> b
        lerp(&mut dst, &a, &b, 1.0);
        for i in 0..n {
            assert!((dst[i] - b[i]).abs() < 1e-5, "lerp t=1 [{}]", i);
        }

        // t=0.5 -> midpoint
        lerp(&mut dst, &a, &b, 0.5);
        for i in 0..n {
            let mid = (a[i] + b[i]) * 0.5;
            assert!((dst[i] - mid).abs() < 1e-4, "lerp t=0.5 [{}]", i);
        }
    }

    #[test]
    fn test_cross3_known() {
        // x cross y = z
        let r = cross3(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        assert!((r[0]).abs() < 1e-6);
        assert!((r[1]).abs() < 1e-6);
        assert!((r[2] - 1.0).abs() < 1e-6);

        // y cross x = -z
        let r = cross3(&[0.0, 1.0, 0.0], &[1.0, 0.0, 0.0]);
        assert!((r[2] + 1.0).abs() < 1e-6);

        // parallel vectors -> zero
        let r = cross3(&[2.0, 4.0, 6.0], &[1.0, 2.0, 3.0]);
        assert!((r[0]).abs() < 1e-5);
        assert!((r[1]).abs() < 1e-5);
        assert!((r[2]).abs() < 1e-5);

        // known non-trivial
        let r = cross3(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert!((r[0] - (-3.0)).abs() < 1e-5);
        assert!((r[1] - 6.0).abs() < 1e-5);
        assert!((r[2] - (-3.0)).abs() < 1e-5);
    }
}
