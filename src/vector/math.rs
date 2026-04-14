// ---------------------------------------------------------------------------
// Elementwise math functions -- NEON fast-path + scalar fallback
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

// ---- constants ------------------------------------------------------------

const LN2: f32 = core::f32::consts::LN_2; // 0.6931472
const LN2_INV: f32 = 1.0 / LN2; // 1.442695
const LN2_HI: f32 = 0.693_145_75; // high part for Cody-Waite
const LN2_LO: f32 = 1.428_606_8e-6; // low part

// Polynomial coefficients for exp(r) on [-ln2/2, ln2/2] (Cephes / Remez)
const EXP_P0: f32 = 1.0;
const EXP_P1: f32 = 1.0;
const EXP_P2: f32 = 0.5; // 1/2!
const EXP_P3: f32 = 0.166_666_7; // 1/3!
const EXP_P4: f32 = 0.041_666_67; // 1/4!
const EXP_P5: f32 = 0.008_333_34; // 1/5!
const EXP_P6: f32 = 0.001_388_89; // 1/6!

// 5-term minimax polynomial for NEON exp (Remez-optimized, max err < 2 ULP)
const EXP_C5: f32 = 0.008_371_6;
const EXP_C4: f32 = 0.041_675_6;
const EXP_C3: f32 = 0.166_665_6;
const EXP_C2: f32 = 0.500_000_1;

const EXP_HI: f32 = 88.376_26;
const EXP_LO: f32 = -87.336_55;

// GELU constant  sqrt(2/pi)
const SQRT_2_OVER_PI: f32 = 0.797_884_6;
const GELU_COEFF: f32 = 0.044_715;

// ---- scalar helpers -------------------------------------------------------

#[inline(always)]
pub(crate) fn exp_scalar(x: f32) -> f32 {
    let x = x.clamp(EXP_LO, EXP_HI);
    let n = (x * LN2_INV + 0.5).floor();
    let r = x - n * LN2_HI - n * LN2_LO;
    // Horner form: p = P0 + r*(P1 + r*(P2 + r*(P3 + r*(P4 + r*(P5 + r*P6)))))
    let mut p = EXP_P6;
    p = p * r + EXP_P5;
    p = p * r + EXP_P4;
    p = p * r + EXP_P3;
    p = p * r + EXP_P2;
    p = p * r + EXP_P1;
    p = p * r + EXP_P0;
    // multiply by 2^n via bit manipulation
    let ni = n as i32;
    let bits = ((ni + 127) as u32) << 23;
    let pow2n = f32::from_bits(bits);
    p * pow2n
}

#[inline(always)]
fn log_scalar(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }
    // Decompose x = 2^e * m, where m in [1, 2)
    let bits = x.to_bits();
    let e = ((bits >> 23) & 0xFF) as i32 - 127;
    let m_bits = (bits & 0x007F_FFFF) | 0x3F80_0000; // mantissa in [1.0, 2.0)
    let m = f32::from_bits(m_bits);
    let exp_adj = e as f32;
    // ln(1+f) via Taylor: f - f²/2 + f³/3 - f⁴/4 + ...
    let f = m - 1.0;
    let f2 = f * f;
    let f3 = f2 * f;
    let f4 = f2 * f2;
    let f5 = f4 * f;
    let f6 = f4 * f2;
    let f7 = f4 * f3;
    let r = f - f2 * 0.5 + f3 * (1.0 / 3.0) - f4 * 0.25 + f5 * 0.2 - f6 * (1.0 / 6.0)
        + f7 * (1.0 / 7.0);
    exp_adj * LN2 + r
}

#[inline(always)]
fn tanh_scalar(x: f32) -> f32 {
    if x.abs() > 9.0 {
        return x.signum();
    }
    let e2x = exp_scalar(2.0 * x);
    (e2x - 1.0) / (e2x + 1.0)
}

#[inline(always)]
fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + exp_scalar(-x))
}

// ---- NEON helpers ---------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn exp_neon(x: float32x4_t) -> float32x4_t {
    let x = vmaxq_f32(vminq_f32(x, vdupq_n_f32(EXP_HI)), vdupq_n_f32(EXP_LO));

    // n = round(x / ln2)  — vrndnq is round-to-nearest, saves 1 op vs floor+0.5
    let n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(LN2_INV)));

    // r = x - n*ln2 via FMA (single-step, sufficient for f32)
    let r = vfmsq_f32(x, n, vdupq_n_f32(LN2));

    // Estrin's scheme: depth-3 polynomial
    let r2 = vmulq_f32(r, r);
    let r4 = vmulq_f32(r2, r2);
    let p01 = vfmaq_f32(vdupq_n_f32(EXP_P0), vdupq_n_f32(EXP_P1), r);
    let p23 = vfmaq_f32(vdupq_n_f32(EXP_C2), vdupq_n_f32(EXP_C3), r);
    let p45 = vfmaq_f32(vdupq_n_f32(EXP_C4), vdupq_n_f32(EXP_C5), r);
    let p = vfmaq_f32(vfmaq_f32(p01, p23, r2), p45, r4);

    // 2^n via integer add to exponent field
    let ni = vcvtq_s32_f32(n);
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(ni, vdupq_n_s32(127))));
    vmulq_f32(p, pow2n)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn log_neon(x: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);
    let ln2_v = vdupq_n_f32(LN2);

    // Decompose x = 2^e * m, where m in [1, 2)
    let bits = vreinterpretq_s32_f32(x);
    let e_raw = vsubq_s32(vshrq_n_s32::<23>(bits), vdupq_n_s32(127));
    let m_bits = vorrq_s32(
        vandq_s32(bits, vdupq_n_s32(0x007F_FFFFu32 as i32)),
        vdupq_n_s32(0x3F80_0000u32 as i32),
    );
    let m = vreinterpretq_f32_s32(m_bits);

    // ln(1+f) via Horner form: f*(c1 + f*(c2 + f*(c3 + f*(c4 + f*(c5 + f*(c6 + f*c7))))))
    // Coefficients: c1=1, c2=-1/2, c3=1/3, c4=-1/4, c5=1/5, c6=-1/6, c7=1/7
    let f = vsubq_f32(m, one);
    let mut p = vdupq_n_f32(1.0 / 7.0);
    p = vfmaq_f32(vdupq_n_f32(-1.0 / 6.0), p, f);
    p = vfmaq_f32(vdupq_n_f32(0.2), p, f);
    p = vfmaq_f32(vdupq_n_f32(-0.25), p, f);
    p = vfmaq_f32(vdupq_n_f32(1.0 / 3.0), p, f);
    p = vfmaq_f32(vdupq_n_f32(-0.5), p, f);
    p = vfmaq_f32(one, p, f);
    let r = vmulq_f32(p, f);

    let ef = vcvtq_f32_s32(e_raw);
    vfmaq_f32(r, ef, ln2_v)
}

// ---- interleaved 16-wide kernels ------------------------------------------

/// 16-wide exp: manually interleave 4 independent FMA chains so the OOO core
/// overlaps dependent operations across lanes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn exp_neon_x4(
    x0: float32x4_t,
    x1: float32x4_t,
    x2: float32x4_t,
    x3: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
    let lo = vdupq_n_f32(EXP_LO);
    let hi = vdupq_n_f32(EXP_HI);
    let x0 = vmaxq_f32(vminq_f32(x0, hi), lo);
    let x1 = vmaxq_f32(vminq_f32(x1, hi), lo);
    let x2 = vmaxq_f32(vminq_f32(x2, hi), lo);
    let x3 = vmaxq_f32(vminq_f32(x3, hi), lo);

    let inv_ln2 = vdupq_n_f32(LN2_INV);
    let n0 = vrndnq_f32(vmulq_f32(x0, inv_ln2));
    let n1 = vrndnq_f32(vmulq_f32(x1, inv_ln2));
    let n2 = vrndnq_f32(vmulq_f32(x2, inv_ln2));
    let n3 = vrndnq_f32(vmulq_f32(x3, inv_ln2));

    let ln2 = vdupq_n_f32(LN2);
    let r0 = vfmsq_f32(x0, n0, ln2);
    let r1 = vfmsq_f32(x1, n1, ln2);
    let r2 = vfmsq_f32(x2, n2, ln2);
    let r3 = vfmsq_f32(x3, n3, ln2);

    // Estrin's scheme — depth 3 instead of 5, interleaved
    let c5 = vdupq_n_f32(EXP_C5);
    let c4 = vdupq_n_f32(EXP_C4);
    let c3 = vdupq_n_f32(EXP_C3);
    let c2 = vdupq_n_f32(EXP_C2);
    let c1 = vdupq_n_f32(EXP_P1);
    let c0 = vdupq_n_f32(EXP_P0);

    let r2_0 = vmulq_f32(r0, r0);
    let r2_1 = vmulq_f32(r1, r1);
    let r2_2 = vmulq_f32(r2, r2);
    let r2_3 = vmulq_f32(r3, r3);
    let r4_0 = vmulq_f32(r2_0, r2_0);
    let r4_1 = vmulq_f32(r2_1, r2_1);
    let r4_2 = vmulq_f32(r2_2, r2_2);
    let r4_3 = vmulq_f32(r2_3, r2_3);

    let p01_0 = vfmaq_f32(c0, c1, r0);
    let p01_1 = vfmaq_f32(c0, c1, r1);
    let p01_2 = vfmaq_f32(c0, c1, r2);
    let p01_3 = vfmaq_f32(c0, c1, r3);
    let p23_0 = vfmaq_f32(c2, c3, r0);
    let p23_1 = vfmaq_f32(c2, c3, r1);
    let p23_2 = vfmaq_f32(c2, c3, r2);
    let p23_3 = vfmaq_f32(c2, c3, r3);
    let p45_0 = vfmaq_f32(c4, c5, r0);
    let p45_1 = vfmaq_f32(c4, c5, r1);
    let p45_2 = vfmaq_f32(c4, c5, r2);
    let p45_3 = vfmaq_f32(c4, c5, r3);

    let p0 = vfmaq_f32(vfmaq_f32(p01_0, p23_0, r2_0), p45_0, r4_0);
    let p1 = vfmaq_f32(vfmaq_f32(p01_1, p23_1, r2_1), p45_1, r4_1);
    let p2 = vfmaq_f32(vfmaq_f32(p01_2, p23_2, r2_2), p45_2, r4_2);
    let p3 = vfmaq_f32(vfmaq_f32(p01_3, p23_3, r2_3), p45_3, r4_3);

    let bias = vdupq_n_s32(127);
    let pow0 = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(n0), bias)));
    let pow1 = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(n1), bias)));
    let pow2 = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(n2), bias)));
    let pow3 = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(n3), bias)));

    (
        vmulq_f32(p0, pow0),
        vmulq_f32(p1, pow1),
        vmulq_f32(p2, pow2),
        vmulq_f32(p3, pow3),
    )
}

/// 16-wide log: manually interleaved Horner chains.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn log_neon_x4(
    x0: float32x4_t,
    x1: float32x4_t,
    x2: float32x4_t,
    x3: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
    let one = vdupq_n_f32(1.0);
    let ln2_v = vdupq_n_f32(LN2);
    let mask = vdupq_n_s32(0x007F_FFFFu32 as i32);
    let base = vdupq_n_s32(0x3F80_0000u32 as i32);
    let bias = vdupq_n_s32(127);

    let b0 = vreinterpretq_s32_f32(x0);
    let b1 = vreinterpretq_s32_f32(x1);
    let b2 = vreinterpretq_s32_f32(x2);
    let b3 = vreinterpretq_s32_f32(x3);

    let e0 = vsubq_s32(vshrq_n_s32::<23>(b0), bias);
    let e1 = vsubq_s32(vshrq_n_s32::<23>(b1), bias);
    let e2 = vsubq_s32(vshrq_n_s32::<23>(b2), bias);
    let e3 = vsubq_s32(vshrq_n_s32::<23>(b3), bias);

    let f0 = vsubq_f32(
        vreinterpretq_f32_s32(vorrq_s32(vandq_s32(b0, mask), base)),
        one,
    );
    let f1 = vsubq_f32(
        vreinterpretq_f32_s32(vorrq_s32(vandq_s32(b1, mask), base)),
        one,
    );
    let f2 = vsubq_f32(
        vreinterpretq_f32_s32(vorrq_s32(vandq_s32(b2, mask), base)),
        one,
    );
    let f3 = vsubq_f32(
        vreinterpretq_f32_s32(vorrq_s32(vandq_s32(b3, mask), base)),
        one,
    );

    // Horner: p = c7*f + c6, then p = p*f + c5, ...
    let c7 = vdupq_n_f32(1.0 / 7.0);
    let c6 = vdupq_n_f32(-1.0 / 6.0);
    let c5 = vdupq_n_f32(0.2);
    let c4 = vdupq_n_f32(-0.25);
    let c3 = vdupq_n_f32(1.0 / 3.0);
    let c2 = vdupq_n_f32(-0.5);

    let mut p0 = c7;
    let mut p1 = c7;
    let mut p2 = c7;
    let mut p3 = c7;
    p0 = vfmaq_f32(c6, p0, f0);
    p1 = vfmaq_f32(c6, p1, f1);
    p2 = vfmaq_f32(c6, p2, f2);
    p3 = vfmaq_f32(c6, p3, f3);
    p0 = vfmaq_f32(c5, p0, f0);
    p1 = vfmaq_f32(c5, p1, f1);
    p2 = vfmaq_f32(c5, p2, f2);
    p3 = vfmaq_f32(c5, p3, f3);
    p0 = vfmaq_f32(c4, p0, f0);
    p1 = vfmaq_f32(c4, p1, f1);
    p2 = vfmaq_f32(c4, p2, f2);
    p3 = vfmaq_f32(c4, p3, f3);
    p0 = vfmaq_f32(c3, p0, f0);
    p1 = vfmaq_f32(c3, p1, f1);
    p2 = vfmaq_f32(c3, p2, f2);
    p3 = vfmaq_f32(c3, p3, f3);
    p0 = vfmaq_f32(c2, p0, f0);
    p1 = vfmaq_f32(c2, p1, f1);
    p2 = vfmaq_f32(c2, p2, f2);
    p3 = vfmaq_f32(c2, p3, f3);
    p0 = vfmaq_f32(one, p0, f0);
    p1 = vfmaq_f32(one, p1, f1);
    p2 = vfmaq_f32(one, p2, f2);
    p3 = vfmaq_f32(one, p3, f3);

    let r0 = vmulq_f32(p0, f0);
    let r1 = vmulq_f32(p1, f1);
    let r2 = vmulq_f32(p2, f2);
    let r3 = vmulq_f32(p3, f3);

    (
        vfmaq_f32(r0, vcvtq_f32_s32(e0), ln2_v),
        vfmaq_f32(r1, vcvtq_f32_s32(e1), ln2_v),
        vfmaq_f32(r2, vcvtq_f32_s32(e2), ln2_v),
        vfmaq_f32(r3, vcvtq_f32_s32(e3), ln2_v),
    )
}

// ---- public functions -----------------------------------------------------

/// Elementwise e^x in-place.
pub fn exp(x: &mut [f32]) {
    let len = x.len();
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let ptr = x.as_mut_ptr();
            while i + 16 <= len {
                let (r0, r1, r2, r3) = exp_neon_x4(
                    vld1q_f32(ptr.add(i)),
                    vld1q_f32(ptr.add(i + 4)),
                    vld1q_f32(ptr.add(i + 8)),
                    vld1q_f32(ptr.add(i + 12)),
                );
                vst1q_f32(ptr.add(i), r0);
                vst1q_f32(ptr.add(i + 4), r1);
                vst1q_f32(ptr.add(i + 8), r2);
                vst1q_f32(ptr.add(i + 12), r3);
                i += 16;
            }
            while i + 4 <= len {
                vst1q_f32(ptr.add(i), exp_neon(vld1q_f32(ptr.add(i))));
                i += 4;
            }
        }
    }

    // scalar tail / fallback
    while i < len {
        x[i] = exp_scalar(x[i]);
        i += 1;
    }
}

/// Elementwise e^x, reading from `src` and writing to `dst`.
pub fn exp_to(src: &[f32], dst: &mut [f32]) {
    let len = src.len().min(dst.len());
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let inp = src.as_ptr();
            let out = dst.as_mut_ptr();
            // Use hand-written asm kernel for bulk
            let bulk = len & !15;
            if bulk > 0 {
                super::exp_kern::exp_asm(out, inp, bulk);
                i = bulk;
            }
            while i + 4 <= len {
                vst1q_f32(out.add(i), exp_neon(vld1q_f32(inp.add(i))));
                i += 4;
            }
        }
    }

    while i < len {
        dst[i] = exp_scalar(src[i]);
        i += 1;
    }
}

/// Elementwise ln(x), reading from `src` and writing to `dst`.
pub fn log_to(src: &[f32], dst: &mut [f32]) {
    let len = src.len().min(dst.len());
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let inp = src.as_ptr();
            let out = dst.as_mut_ptr();
            while i + 16 <= len {
                let (r0, r1, r2, r3) = log_neon_x4(
                    vld1q_f32(inp.add(i)),
                    vld1q_f32(inp.add(i + 4)),
                    vld1q_f32(inp.add(i + 8)),
                    vld1q_f32(inp.add(i + 12)),
                );
                vst1q_f32(out.add(i), r0);
                vst1q_f32(out.add(i + 4), r1);
                vst1q_f32(out.add(i + 8), r2);
                vst1q_f32(out.add(i + 12), r3);
                i += 16;
            }
            while i + 4 <= len {
                vst1q_f32(out.add(i), log_neon(vld1q_f32(inp.add(i))));
                i += 4;
            }
        }
    }

    while i < len {
        dst[i] = log_scalar(src[i]);
        i += 1;
    }
}

/// Elementwise ln(x) in-place.
pub fn log(x: &mut [f32]) {
    let len = x.len();
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let ptr = x.as_mut_ptr();
            while i + 16 <= len {
                let (r0, r1, r2, r3) = log_neon_x4(
                    vld1q_f32(ptr.add(i)),
                    vld1q_f32(ptr.add(i + 4)),
                    vld1q_f32(ptr.add(i + 8)),
                    vld1q_f32(ptr.add(i + 12)),
                );
                vst1q_f32(ptr.add(i), r0);
                vst1q_f32(ptr.add(i + 4), r1);
                vst1q_f32(ptr.add(i + 8), r2);
                vst1q_f32(ptr.add(i + 12), r3);
                i += 16;
            }
            while i + 4 <= len {
                vst1q_f32(ptr.add(i), log_neon(vld1q_f32(ptr.add(i))));
                i += 4;
            }
        }
    }

    while i < len {
        x[i] = log_scalar(x[i]);
        i += 1;
    }
}

/// Elementwise tanh in-place.
pub fn tanh(x: &mut [f32]) {
    let len = x.len();
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let two = vdupq_n_f32(2.0);
            let one = vdupq_n_f32(1.0);
            let clamp_pos = vdupq_n_f32(9.0);
            let clamp_neg = vdupq_n_f32(-9.0);
            let ptr = x.as_mut_ptr();
            while i + 16 <= len {
                let v0 = vmaxq_f32(vminq_f32(vld1q_f32(ptr.add(i)), clamp_pos), clamp_neg);
                let v1 = vmaxq_f32(vminq_f32(vld1q_f32(ptr.add(i + 4)), clamp_pos), clamp_neg);
                let v2 = vmaxq_f32(vminq_f32(vld1q_f32(ptr.add(i + 8)), clamp_pos), clamp_neg);
                let v3 = vmaxq_f32(vminq_f32(vld1q_f32(ptr.add(i + 12)), clamp_pos), clamp_neg);
                let e0 = exp_neon(vmulq_f32(two, v0));
                let e1 = exp_neon(vmulq_f32(two, v1));
                let e2 = exp_neon(vmulq_f32(two, v2));
                let e3 = exp_neon(vmulq_f32(two, v3));
                vst1q_f32(
                    ptr.add(i),
                    vdivq_f32(vsubq_f32(e0, one), vaddq_f32(e0, one)),
                );
                vst1q_f32(
                    ptr.add(i + 4),
                    vdivq_f32(vsubq_f32(e1, one), vaddq_f32(e1, one)),
                );
                vst1q_f32(
                    ptr.add(i + 8),
                    vdivq_f32(vsubq_f32(e2, one), vaddq_f32(e2, one)),
                );
                vst1q_f32(
                    ptr.add(i + 12),
                    vdivq_f32(vsubq_f32(e3, one), vaddq_f32(e3, one)),
                );
                i += 16;
            }
            while i + 4 <= len {
                let v = vmaxq_f32(vminq_f32(vld1q_f32(ptr.add(i)), clamp_pos), clamp_neg);
                let e2x = exp_neon(vmulq_f32(two, v));
                vst1q_f32(
                    ptr.add(i),
                    vdivq_f32(vsubq_f32(e2x, one), vaddq_f32(e2x, one)),
                );
                i += 4;
            }
        }
    }

    while i < len {
        x[i] = tanh_scalar(x[i]);
        i += 1;
    }
}

/// Elementwise sigmoid 1/(1+e^-x) in-place.
pub fn sigmoid(x: &mut [f32]) {
    let len = x.len();
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let one = vdupq_n_f32(1.0);
            let ptr = x.as_mut_ptr();
            while i + 16 <= len {
                let e0 = exp_neon(vnegq_f32(vld1q_f32(ptr.add(i))));
                let e1 = exp_neon(vnegq_f32(vld1q_f32(ptr.add(i + 4))));
                let e2 = exp_neon(vnegq_f32(vld1q_f32(ptr.add(i + 8))));
                let e3 = exp_neon(vnegq_f32(vld1q_f32(ptr.add(i + 12))));
                vst1q_f32(ptr.add(i), vdivq_f32(one, vaddq_f32(one, e0)));
                vst1q_f32(ptr.add(i + 4), vdivq_f32(one, vaddq_f32(one, e1)));
                vst1q_f32(ptr.add(i + 8), vdivq_f32(one, vaddq_f32(one, e2)));
                vst1q_f32(ptr.add(i + 12), vdivq_f32(one, vaddq_f32(one, e3)));
                i += 16;
            }
            while i + 4 <= len {
                let e = exp_neon(vnegq_f32(vld1q_f32(ptr.add(i))));
                vst1q_f32(ptr.add(i), vdivq_f32(one, vaddq_f32(one, e)));
                i += 4;
            }
        }
    }

    while i < len {
        x[i] = sigmoid_scalar(x[i]);
        i += 1;
    }
}

/// Elementwise GELU in-place.
/// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(x: &mut [f32]) {
    let len = x.len();
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let half = vdupq_n_f32(0.5);
            let one = vdupq_n_f32(1.0);
            let two = vdupq_n_f32(2.0);
            let coeff = vdupq_n_f32(GELU_COEFF);
            let s2pi = vdupq_n_f32(SQRT_2_OVER_PI);
            let clamp_pos = vdupq_n_f32(9.0);
            let clamp_neg = vdupq_n_f32(-9.0);
            let ptr = x.as_mut_ptr();

            while i + 16 <= len {
                // Load and compute inner for all 4 lanes
                let v0 = vld1q_f32(ptr.add(i));
                let v1 = vld1q_f32(ptr.add(i + 4));
                let v2 = vld1q_f32(ptr.add(i + 8));
                let v3 = vld1q_f32(ptr.add(i + 12));
                let i0 = vmaxq_f32(
                    vminq_f32(
                        vmulq_f32(s2pi, vfmaq_f32(v0, coeff, vmulq_f32(vmulq_f32(v0, v0), v0))),
                        clamp_pos,
                    ),
                    clamp_neg,
                );
                let i1 = vmaxq_f32(
                    vminq_f32(
                        vmulq_f32(s2pi, vfmaq_f32(v1, coeff, vmulq_f32(vmulq_f32(v1, v1), v1))),
                        clamp_pos,
                    ),
                    clamp_neg,
                );
                let i2 = vmaxq_f32(
                    vminq_f32(
                        vmulq_f32(s2pi, vfmaq_f32(v2, coeff, vmulq_f32(vmulq_f32(v2, v2), v2))),
                        clamp_pos,
                    ),
                    clamp_neg,
                );
                let i3 = vmaxq_f32(
                    vminq_f32(
                        vmulq_f32(s2pi, vfmaq_f32(v3, coeff, vmulq_f32(vmulq_f32(v3, v3), v3))),
                        clamp_pos,
                    ),
                    clamp_neg,
                );
                // exp(2*inner) for all lanes
                let e0 = exp_neon(vmulq_f32(two, i0));
                let e1 = exp_neon(vmulq_f32(two, i1));
                let e2 = exp_neon(vmulq_f32(two, i2));
                let e3 = exp_neon(vmulq_f32(two, i3));
                // tanh + gelu
                let t0 = vdivq_f32(vsubq_f32(e0, one), vaddq_f32(e0, one));
                let t1 = vdivq_f32(vsubq_f32(e1, one), vaddq_f32(e1, one));
                let t2 = vdivq_f32(vsubq_f32(e2, one), vaddq_f32(e2, one));
                let t3 = vdivq_f32(vsubq_f32(e3, one), vaddq_f32(e3, one));
                vst1q_f32(
                    ptr.add(i),
                    vmulq_f32(half, vmulq_f32(v0, vaddq_f32(one, t0))),
                );
                vst1q_f32(
                    ptr.add(i + 4),
                    vmulq_f32(half, vmulq_f32(v1, vaddq_f32(one, t1))),
                );
                vst1q_f32(
                    ptr.add(i + 8),
                    vmulq_f32(half, vmulq_f32(v2, vaddq_f32(one, t2))),
                );
                vst1q_f32(
                    ptr.add(i + 12),
                    vmulq_f32(half, vmulq_f32(v3, vaddq_f32(one, t3))),
                );
                i += 16;
            }
            while i + 4 <= len {
                let v = vld1q_f32(ptr.add(i));
                let x3 = vmulq_f32(vmulq_f32(v, v), v);
                let inner = vmulq_f32(s2pi, vfmaq_f32(v, coeff, x3));
                let inner_c = vmaxq_f32(vminq_f32(inner, clamp_pos), clamp_neg);
                let e2 = exp_neon(vmulq_f32(two, inner_c));
                let th = vdivq_f32(vsubq_f32(e2, one), vaddq_f32(e2, one));
                vst1q_f32(
                    ptr.add(i),
                    vmulq_f32(half, vmulq_f32(v, vaddq_f32(one, th))),
                );
                i += 4;
            }
        }
    }

    while i < len {
        let xi = x[i];
        let inner = SQRT_2_OVER_PI * (xi + GELU_COEFF * xi * xi * xi);
        x[i] = 0.5 * xi * (1.0 + tanh_scalar(inner));
        i += 1;
    }
}

/// Elementwise SiLU (x * sigmoid(x)) in-place.
pub fn silu(x: &mut [f32]) {
    let len = x.len();
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let one = vdupq_n_f32(1.0);
            let ptr = x.as_mut_ptr();
            while i + 16 <= len {
                let v0 = vld1q_f32(ptr.add(i));
                let v1 = vld1q_f32(ptr.add(i + 4));
                let v2 = vld1q_f32(ptr.add(i + 8));
                let v3 = vld1q_f32(ptr.add(i + 12));
                let e0 = exp_neon(vnegq_f32(v0));
                let e1 = exp_neon(vnegq_f32(v1));
                let e2 = exp_neon(vnegq_f32(v2));
                let e3 = exp_neon(vnegq_f32(v3));
                vst1q_f32(
                    ptr.add(i),
                    vmulq_f32(v0, vdivq_f32(one, vaddq_f32(one, e0))),
                );
                vst1q_f32(
                    ptr.add(i + 4),
                    vmulq_f32(v1, vdivq_f32(one, vaddq_f32(one, e1))),
                );
                vst1q_f32(
                    ptr.add(i + 8),
                    vmulq_f32(v2, vdivq_f32(one, vaddq_f32(one, e2))),
                );
                vst1q_f32(
                    ptr.add(i + 12),
                    vmulq_f32(v3, vdivq_f32(one, vaddq_f32(one, e3))),
                );
                i += 16;
            }
            while i + 4 <= len {
                let v = vld1q_f32(ptr.add(i));
                let e = exp_neon(vnegq_f32(v));
                vst1q_f32(ptr.add(i), vmulq_f32(v, vdivq_f32(one, vaddq_f32(one, e))));
                i += 4;
            }
        }
    }

    while i < len {
        let xi = x[i];
        x[i] = xi * sigmoid_scalar(xi);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_exp_basic() {
        let mut v = vec![0.0, 1.0, -1.0, 2.0];
        exp(&mut v);
        assert!(approx_eq(v[0], 1.0, 1e-5));
        assert!(approx_eq(v[1], std::f32::consts::E, 1e-4));
        assert!(approx_eq(v[2], 1.0 / std::f32::consts::E, 1e-4));
    }

    #[test]
    fn test_sigmoid_bounds() {
        let mut v = vec![-10.0, 0.0, 10.0];
        sigmoid(&mut v);
        assert!(v[0] < 0.01);
        assert!(approx_eq(v[1], 0.5, 1e-5));
        assert!(v[2] > 0.99);
    }

    #[test]
    fn test_tanh_bounds() {
        let mut v = vec![-20.0, 0.0, 20.0];
        tanh(&mut v);
        assert!(approx_eq(v[0], -1.0, 1e-5));
        assert!(approx_eq(v[1], 0.0, 1e-5));
        assert!(approx_eq(v[2], 1.0, 1e-5));
    }

    #[test]
    fn test_log_basic() {
        let mut v = vec![1.0, std::f32::consts::E, 0.5];
        log(&mut v);
        assert!(approx_eq(v[0], 0.0, 1e-5));
        assert!(approx_eq(v[1], 1.0, 1e-3));
        assert!(approx_eq(v[2], -(2.0f32.ln()), 1e-3));
    }

    #[test]
    fn test_gelu_zero() {
        let mut v = vec![0.0];
        gelu(&mut v);
        assert!(approx_eq(v[0], 0.0, 1e-6));
    }

    #[test]
    fn test_silu_zero() {
        let mut v = vec![0.0];
        silu(&mut v);
        assert!(approx_eq(v[0], 0.0, 1e-6));
    }
}
