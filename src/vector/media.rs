// Media processing NEON kernels -- color space, histogram, resize

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// BT.601 RGB->YUV. NEON: vld3q_u8 deinterleave, 16 pixels/iter.
pub fn rgb_to_yuv(yuv: &mut [u8], rgb: &[u8]) {
    assert!(
        rgb.len().is_multiple_of(3),
        "rgb_to_yuv: rgb.len() must be multiple of 3"
    );
    assert!(yuv.len() >= rgb.len(), "rgb_to_yuv: yuv buffer too small");
    let npix = rgb.len() / 3;
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let pr = rgb.as_ptr();
            let py = yuv.as_mut_ptr();

            while i + 16 <= npix {
                let src = vld3q_u8(pr.add(i * 3));
                let r = src.0;
                let g = src.1;
                let b = src.2;

                let r_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(r)));
                let r_hi = vreinterpretq_s16_u16(vmovl_high_u8(r));
                let g_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(g)));
                let g_hi = vreinterpretq_s16_u16(vmovl_high_u8(g));
                let b_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b)));
                let b_hi = vreinterpretq_s16_u16(vmovl_high_u8(b));
                let bias_128 = vdupq_n_s16(128);
                let y_lo = compute_y(r_lo, g_lo, b_lo, bias_128);
                let y_hi = compute_y(r_hi, g_hi, b_hi, bias_128);
                let u_lo = compute_u(r_lo, g_lo, b_lo, bias_128);
                let u_hi = compute_u(r_hi, g_hi, b_hi, bias_128);
                let v_lo = compute_v(r_lo, g_lo, b_lo, bias_128);
                let v_hi = compute_v(r_hi, g_hi, b_hi, bias_128);
                let y_u8 = vcombine_u8(vqmovun_s16(y_lo), vqmovun_s16(y_hi));
                let u_u8 = vcombine_u8(vqmovun_s16(u_lo), vqmovun_s16(u_hi));
                let v_u8 = vcombine_u8(vqmovun_s16(v_lo), vqmovun_s16(v_hi));

                vst3q_u8(py.add(i * 3), uint8x16x3_t(y_u8, u_u8, v_u8));
                i += 16;
            }
        }
    }

    // Scalar tail
    while i < npix {
        let off = i * 3;
        let r = rgb[off] as i32;
        let g = rgb[off + 1] as i32;
        let b = rgb[off + 2] as i32;
        yuv[off] = (16 + ((66 * r + 129 * g + 25 * b + 128) >> 8)) as u8;
        yuv[off + 1] = (128 + ((-38 * r - 74 * g + 112 * b + 128) >> 8)) as u8;
        yuv[off + 2] = (128 + ((112 * r - 94 * g - 18 * b + 128) >> 8)) as u8;
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn compute_y(r: int16x8_t, g: int16x8_t, b: int16x8_t, bias: int16x8_t) -> int16x8_t {
    let mut acc = vmulq_n_s16(r, 66);
    acc = vmlaq_n_s16(acc, g, 129);
    acc = vmlaq_n_s16(acc, b, 25);
    acc = vaddq_s16(acc, bias);
    acc = vshrq_n_s16::<8>(acc);
    vaddq_s16(acc, vdupq_n_s16(16))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn compute_u(r: int16x8_t, g: int16x8_t, b: int16x8_t, bias: int16x8_t) -> int16x8_t {
    let mut acc = vmulq_n_s16(r, -38);
    acc = vmlaq_n_s16(acc, g, -74);
    acc = vmlaq_n_s16(acc, b, 112);
    acc = vaddq_s16(acc, bias);
    acc = vshrq_n_s16::<8>(acc);
    vaddq_s16(acc, vdupq_n_s16(128))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn compute_v(r: int16x8_t, g: int16x8_t, b: int16x8_t, bias: int16x8_t) -> int16x8_t {
    let mut acc = vmulq_n_s16(r, 112);
    acc = vmlaq_n_s16(acc, g, -94);
    acc = vmlaq_n_s16(acc, b, -18);
    acc = vaddq_s16(acc, bias);
    acc = vshrq_n_s16::<8>(acc);
    vaddq_s16(acc, vdupq_n_s16(128))
}

/// BT.601 YUV->RGB inverse. NEON: i32 arithmetic, vqmovun_s16 saturation.
pub fn yuv_to_rgb(rgb: &mut [u8], yuv: &[u8]) {
    assert!(
        yuv.len().is_multiple_of(3),
        "yuv_to_rgb: yuv.len() must be multiple of 3"
    );
    assert!(rgb.len() >= yuv.len(), "yuv_to_rgb: rgb buffer too small");
    let npix = yuv.len() / 3;
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let ps = yuv.as_ptr();
            let pd = rgb.as_mut_ptr();

            while i + 8 <= npix {
                let src = vld3_u8(ps.add(i * 3));
                let y_raw = src.0;
                let u_raw = src.1;
                let v_raw = src.2;

                let y16 = vreinterpretq_s16_u16(vmovl_u8(y_raw));
                let u16 = vreinterpretq_s16_u16(vmovl_u8(u_raw));
                let v16 = vreinterpretq_s16_u16(vmovl_u8(v_raw));
                let ym = vsubq_s16(y16, vdupq_n_s16(16));
                let um = vsubq_s16(u16, vdupq_n_s16(128));
                let vm = vsubq_s16(v16, vdupq_n_s16(128));
                let ym_lo = vmovl_s16(vget_low_s16(ym));
                let ym_hi = vmovl_high_s16(ym);
                let base_lo = vmulq_n_s32(ym_lo, 298);
                let base_hi = vmulq_n_s32(ym_hi, 298);
                let bias32 = vdupq_n_s32(128);
                let vm_lo = vmovl_s16(vget_low_s16(vm));
                let vm_hi = vmovl_high_s16(vm);
                let um_lo = vmovl_s16(vget_low_s16(um));
                let um_hi = vmovl_high_s16(um);
                let r_lo = vshrq_n_s32::<8>(vaddq_s32(
                    vaddq_s32(base_lo, vmulq_n_s32(vm_lo, 409)),
                    bias32,
                ));
                let r_hi = vshrq_n_s32::<8>(vaddq_s32(
                    vaddq_s32(base_hi, vmulq_n_s32(vm_hi, 409)),
                    bias32,
                ));
                let g_lo = vshrq_n_s32::<8>(vaddq_s32(
                    vsubq_s32(
                        vsubq_s32(base_lo, vmulq_n_s32(um_lo, 100)),
                        vmulq_n_s32(vm_lo, 208),
                    ),
                    bias32,
                ));
                let g_hi = vshrq_n_s32::<8>(vaddq_s32(
                    vsubq_s32(
                        vsubq_s32(base_hi, vmulq_n_s32(um_hi, 100)),
                        vmulq_n_s32(vm_hi, 208),
                    ),
                    bias32,
                ));
                let b_lo = vshrq_n_s32::<8>(vaddq_s32(
                    vaddq_s32(base_lo, vmulq_n_s32(um_lo, 516)),
                    bias32,
                ));
                let b_hi = vshrq_n_s32::<8>(vaddq_s32(
                    vaddq_s32(base_hi, vmulq_n_s32(um_hi, 516)),
                    bias32,
                ));

                let r16 = vcombine_s16(vmovn_s32(r_lo), vmovn_s32(r_hi));
                let g16 = vcombine_s16(vmovn_s32(g_lo), vmovn_s32(g_hi));
                let b16 = vcombine_s16(vmovn_s32(b_lo), vmovn_s32(b_hi));

                let r_u8 = vqmovun_s16(r16);
                let g_u8 = vqmovun_s16(g16);
                let b_u8 = vqmovun_s16(b16);

                vst3_u8(pd.add(i * 3), uint8x8x3_t(r_u8, g_u8, b_u8));
                i += 8;
            }
        }
    }

    // Scalar tail
    while i < npix {
        let off = i * 3;
        let y = yuv[off] as i32;
        let u = yuv[off + 1] as i32;
        let v = yuv[off + 2] as i32;
        let c = 298 * (y - 16);
        let r = (c + 409 * (v - 128) + 128) >> 8;
        let g = (c - 100 * (u - 128) - 208 * (v - 128) + 128) >> 8;
        let b = (c + 516 * (u - 128) + 128) >> 8;
        rgb[off] = r.clamp(0, 255) as u8;
        rgb[off + 1] = g.clamp(0, 255) as u8;
        rgb[off + 2] = b.clamp(0, 255) as u8;
        i += 1;
    }
}

/// Histogram of u8 data into 256 bins. 4-bank strategy, NEON merge.
pub fn histogram_u8(hist: &mut [u32; 256], data: &[u8]) {
    *hist = [0u32; 256];
    let len = data.len();
    if len == 0 {
        return;
    }

    let mut bank = [[0u32; 256]; 4];
    let mut i = 0;
    while i + 16 <= len {
        bank[0][data[i] as usize] += 1;
        bank[1][data[i + 1] as usize] += 1;
        bank[2][data[i + 2] as usize] += 1;
        bank[3][data[i + 3] as usize] += 1;
        bank[0][data[i + 4] as usize] += 1;
        bank[1][data[i + 5] as usize] += 1;
        bank[2][data[i + 6] as usize] += 1;
        bank[3][data[i + 7] as usize] += 1;
        bank[0][data[i + 8] as usize] += 1;
        bank[1][data[i + 9] as usize] += 1;
        bank[2][data[i + 10] as usize] += 1;
        bank[3][data[i + 11] as usize] += 1;
        bank[0][data[i + 12] as usize] += 1;
        bank[1][data[i + 13] as usize] += 1;
        bank[2][data[i + 14] as usize] += 1;
        bank[3][data[i + 15] as usize] += 1;
        i += 16;
    }

    // Scalar tail
    while i < len {
        bank[i & 3][data[i] as usize] += 1;
        i += 1;
    }

    let mut j = 0;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let p0 = bank[0].as_ptr();
        let p1 = bank[1].as_ptr();
        let p2 = bank[2].as_ptr();
        let p3 = bank[3].as_ptr();
        let pd = hist.as_mut_ptr();
        while j + 4 <= 256 {
            let s01 = vaddq_u32(vld1q_u32(p0.add(j)), vld1q_u32(p1.add(j)));
            let s23 = vaddq_u32(vld1q_u32(p2.add(j)), vld1q_u32(p3.add(j)));
            vst1q_u32(pd.add(j), vaddq_u32(s01, s23));
            j += 4;
        }
    }
    while j < 256 {
        hist[j] = bank[0][j] + bank[1][j] + bank[2][j] + bank[3][j];
        j += 1;
    }
}

/// Bilinear resize f32 image. NEON: 4 dst pixels/iter, manual gather.
pub fn resize_bilinear_f32(
    dst: &mut [f32],
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) {
    assert!(
        src.len() >= src_w * src_h,
        "resize_bilinear_f32: src buffer too small"
    );
    assert!(
        dst.len() >= dst_w * dst_h,
        "resize_bilinear_f32: dst buffer too small"
    );
    if dst_w == 0 || dst_h == 0 || src_w == 0 || src_h == 0 {
        return;
    }

    let x_scale = if dst_w > 1 {
        (src_w as f64 - 1.0) / (dst_w as f64 - 1.0)
    } else {
        0.0
    };
    let y_scale = if dst_h > 1 {
        (src_h as f64 - 1.0) / (dst_h as f64 - 1.0)
    } else {
        0.0
    };
    let max_x = (src_w - 1) as f32;
    let max_y = (src_h - 1) as f32;

    for dy in 0..dst_h {
        let sy_f = (dy as f64 * y_scale) as f32;
        let sy_f = sy_f.min(max_y).max(0.0);
        let sy0 = sy_f as usize;
        let sy1 = (sy0 + 1).min(src_h - 1);
        let fy = sy_f - sy0 as f32;
        let fy_inv = 1.0 - fy;
        let row0 = sy0 * src_w;
        let row1 = sy1 * src_w;
        let dst_row = dy * dst_w;
        let mut dx = 0;

        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                let vfy = vdupq_n_f32(fy);
                let vfy_inv = vdupq_n_f32(fy_inv);
                let ps = src.as_ptr();
                let pd = dst.as_mut_ptr();

                while dx + 4 <= dst_w {
                    let mut sx_f = [0.0f32; 4];
                    let mut sx0 = [0usize; 4];
                    let mut sx1a = [0usize; 4];
                    let mut fxv = [0.0f32; 4];
                    for k in 0..4 {
                        let s = ((dx + k) as f64 * x_scale) as f32;
                        let s = s.min(max_x).max(0.0);
                        sx_f[k] = s;
                        sx0[k] = s as usize;
                        sx1a[k] = (sx0[k] + 1).min(src_w - 1);
                        fxv[k] = s - sx0[k] as f32;
                    }
                    let fx = vld1q_f32(fxv.as_ptr());
                    let fx_inv = vsubq_f32(vdupq_n_f32(1.0), fx);
                    let (mut tl, mut tr) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
                    let (mut bl, mut br) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
                    tl = vsetq_lane_f32::<0>(*ps.add(row0 + sx0[0]), tl);
                    tl = vsetq_lane_f32::<1>(*ps.add(row0 + sx0[1]), tl);
                    tl = vsetq_lane_f32::<2>(*ps.add(row0 + sx0[2]), tl);
                    tl = vsetq_lane_f32::<3>(*ps.add(row0 + sx0[3]), tl);
                    tr = vsetq_lane_f32::<0>(*ps.add(row0 + sx1a[0]), tr);
                    tr = vsetq_lane_f32::<1>(*ps.add(row0 + sx1a[1]), tr);
                    tr = vsetq_lane_f32::<2>(*ps.add(row0 + sx1a[2]), tr);
                    tr = vsetq_lane_f32::<3>(*ps.add(row0 + sx1a[3]), tr);
                    bl = vsetq_lane_f32::<0>(*ps.add(row1 + sx0[0]), bl);
                    bl = vsetq_lane_f32::<1>(*ps.add(row1 + sx0[1]), bl);
                    bl = vsetq_lane_f32::<2>(*ps.add(row1 + sx0[2]), bl);
                    bl = vsetq_lane_f32::<3>(*ps.add(row1 + sx0[3]), bl);
                    br = vsetq_lane_f32::<0>(*ps.add(row1 + sx1a[0]), br);
                    br = vsetq_lane_f32::<1>(*ps.add(row1 + sx1a[1]), br);
                    br = vsetq_lane_f32::<2>(*ps.add(row1 + sx1a[2]), br);
                    br = vsetq_lane_f32::<3>(*ps.add(row1 + sx1a[3]), br);

                    let top = vfmaq_f32(vmulq_f32(tl, fx_inv), tr, fx);
                    let bot = vfmaq_f32(vmulq_f32(bl, fx_inv), br, fx);
                    let res = vfmaq_f32(vmulq_f32(top, vfy_inv), bot, vfy);

                    vst1q_f32(pd.add(dst_row + dx), res);
                    dx += 4;
                }
            }
        }

        // Scalar tail
        while dx < dst_w {
            let sx_f = (dx as f64 * x_scale) as f32;
            let sx_f = sx_f.min(max_x).max(0.0);
            let sx0 = sx_f as usize;
            let sx1 = (sx0 + 1).min(src_w - 1);
            let fx = sx_f - sx0 as f32;
            let tl = src[row0 + sx0];
            let tr = src[row0 + sx1];
            let bl = src[row1 + sx0];
            let br = src[row1 + sx1];
            let top = tl + fx * (tr - tl);
            let bot = bl + fx * (br - bl);
            dst[dst_row + dx] = top + fy * (bot - top);
            dx += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn yuv_ref(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
        let (r, g, b) = (r as i32, g as i32, b as i32);
        (
            (16 + ((66 * r + 129 * g + 25 * b + 128) >> 8)) as u8,
            (128 + ((-38 * r - 74 * g + 112 * b + 128) >> 8)) as u8,
            (128 + ((112 * r - 94 * g - 18 * b + 128) >> 8)) as u8,
        )
    }

    #[test]
    fn test_rgb_to_yuv_known() {
        let mut yuv = vec![0u8; 48];
        rgb_to_yuv(&mut yuv, &vec![0u8; 48]);
        for i in 0..16 {
            assert_eq!((yuv[i * 3], yuv[i * 3 + 1], yuv[i * 3 + 2]), (16, 128, 128));
        }
    }

    #[test]
    fn test_rgb_to_yuv_various() {
        for &(r, g, b) in &[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)] {
            let mut yuv = vec![0u8; 3];
            rgb_to_yuv(&mut yuv, &[r, g, b]);
            let e = yuv_ref(r, g, b);
            assert_eq!((yuv[0], yuv[1], yuv[2]), e, "({},{},{})", r, g, b);
        }
    }

    #[test]
    fn test_yuv_to_rgb_black_white() {
        let mut rgb = vec![0u8; 3];
        yuv_to_rgb(&mut rgb, &[16, 128, 128]);
        assert!(rgb.iter().all(|&c| c <= 1));
        yuv_to_rgb(&mut rgb, &[235, 128, 128]);
        assert!(rgb.iter().all(|&c| c >= 253));
    }

    #[test]
    fn test_rgb_yuv_roundtrip() {
        let mut yuv = vec![0u8; 3];
        let mut out = vec![0u8; 3];
        rgb_to_yuv(&mut yuv, &[128, 128, 128]);
        yuv_to_rgb(&mut out, &yuv);
        for c in 0..3 {
            assert!((128i32 - out[c] as i32).abs() <= 2, "ch {}: {}", c, out[c]);
        }
    }

    #[test]
    fn test_histogram_uniform() {
        let mut hist = [0u32; 256];
        histogram_u8(&mut hist, &(0..=255).collect::<Vec<u8>>());
        assert!(hist.iter().all(|&h| h == 1));
    }

    #[test]
    fn test_histogram_repeated() {
        let mut hist = [0u32; 256];
        histogram_u8(&mut hist, &vec![42u8; 1000]);
        assert_eq!(hist[42], 1000);
        assert_eq!(hist.iter().sum::<u32>(), 1000);
    }

    #[test]
    fn test_histogram_empty() {
        let mut hist = [99u32; 256];
        histogram_u8(&mut hist, &[]);
        assert!(hist.iter().all(|&h| h == 0));
    }

    #[test]
    fn test_resize_identity() {
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; 16];
        resize_bilinear_f32(&mut dst, &src, 4, 4, 4, 4);
        for i in 0..16 {
            assert!((dst[i] - src[i]).abs() < 1e-4, "[{}]", i);
        }
    }

    #[test]
    fn test_resize_upsample() {
        let mut dst = vec![0.0f32; 16];
        resize_bilinear_f32(&mut dst, &[0.0, 1.0, 2.0, 3.0], 2, 2, 4, 4);
        assert!((dst[0]).abs() < 1e-4);
        assert!((dst[3] - 1.0).abs() < 1e-4);
        assert!((dst[12] - 2.0).abs() < 1e-4);
        assert!((dst[15] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_resize_downsample() {
        let mut dst = vec![0.0f32; 4];
        resize_bilinear_f32(&mut dst, &vec![7.5f32; 16], 4, 4, 2, 2);
        assert!(dst.iter().all(|&v| (v - 7.5).abs() < 1e-4));
    }
}
