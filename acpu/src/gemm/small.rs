//! Direct AMX path for small matrices.
//! 32×32 pair-load kernel when m,n both ≥32 and divisible by 32.
//! 16×64 fallback for all other cases.

use super::{MR, NR};
use crate::matrix::tile;

/// Direct AMX matmul_f32 for small matrices where n*k ≤ 131K.
/// `first_k`: if true, skip preload_c and use fma_first (C = A*B, not C += A*B).
#[cfg(target_arch = "aarch64")]
pub(super) fn matmul_f32_amx_direct(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    first_k: bool,
) {
    super::ensure_amx();

    let n_mr = m.div_ceil(MR);
    let bs = n * 4;
    let n_nr = n.div_ceil(NR);

    // Pair32 for larger sizes (≥256). For 64-128, the 16×64 path has lower
    // pack overhead (16-wide vs 32-wide interleaved transpose).
    let use_pair = m.is_multiple_of(32) && n.is_multiple_of(32) && m >= 32 && n >= 32;

    if use_pair {
        matmul_f32_pair32(a, b, c, m, n, k, n_mr, n_nr, bs, first_k);
    } else {
        // 16x64 path always accumulates (C += A*B). Zero C first for set semantics.
        if first_k {
            c.fill(0.0);
        }
        matmul_f32_16x64(a, b, c, m, n, k, n_mr, n_nr, bs, false);
    }
}

/// 32×32 pair-load path: interleaved A pack + pair LDY.
/// 7 ops/k-step (1 LDY pair + 2 LDX + 4 FMA) vs 9 in 16×64.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn matmul_f32_pair32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    _m: usize,
    n: usize,
    k: usize,
    n_mr: usize,
    n_nr: usize,
    bs: usize,
    first_k: bool,
) {
    let n_pairs = n_mr / 2;
    let a_need = n_pairs * k * MR * 2;
    let a_src_id = a.as_ptr() as usize;
    let a_dims = (_m, k, 0usize);

    let (mut a_pack, skip_pack) = super::PACK_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        let cached_hit = cache.a_src == a_src_id && cache.a_dims == a_dims && cache.a.is_some();
        let buf = super::cached_buf(&mut cache.a, a_need);
        let skip = cached_hit && buf.len >= a_need;
        (buf, skip)
    });
    let a_buf = a_pack.as_mut_slice();

    let b_aligned = (b.as_ptr() as usize).is_multiple_of(128) && bs.is_multiple_of(128);

    // K-blocked computation: process K in chunks of KC.
    // Within each KC block, B-strip (KC×128 bytes) fits in L1 (192KB).
    // A is packed for full K (cached across calls), but kernel processes KC at a time.
    // First KC block: fma_first (no preload). Subsequent: preload+acc from C.
    // K-block only when B > 4MB (doesn't fit in L2).
    // For smaller B, process all K in one pass — avoids preload/store overhead.
    let b_bytes = n * k * 4;
    let kc_max = if b_bytes <= 4 * 1024 * 1024 { k } else { 512 };

    // Pack all A for full K (cached across calls).
    if !skip_pack {
        for pair in 0..n_pairs {
            let off = pair * k * MR * 2;
            pack_a_interleaved_neon(a, k, pair * 2 * MR, k, &mut a_buf[off..]);
        }
    }

    unsafe {
        let b_base = b.as_ptr();

        let mut pc = 0usize;
        while pc < k {
            let kc = (k - pc).min(kc_max);
            let is_first_kc = pc == 0 && first_k;

            // Process pairs 2-at-a-time sharing B in X registers (AI=32).
            // For each B-column block, process pair0 then pair1 without reloading B.
            let mut pair = 0;
            while pair + 2 <= n_pairs {
                let a_off0 = pair * k * MR * 2 + pc * MR * 2;
                let a_off1 = (pair + 1) * k * MR * 2 + pc * MR * 2;
                let ap0 = a_buf.as_ptr().add(a_off0) as *const u8;
                let ap1 = a_buf.as_ptr().add(a_off1) as *const u8;
                let ir0 = pair * 2;
                let ir1 = (pair + 1) * 2;
                let mut jr = 0usize;

                while jr + 2 <= n_nr {
                    let bp = b_base.add(pc * n + jr * NR) as *const u8;
                    let c0 = c.as_mut_ptr().add(ir0 * MR * n + jr * NR);
                    let c1 = c.as_mut_ptr().add(ir1 * MR * n + jr * NR);

                    // Pair0: compute into Z tiles
                    if is_first_kc {
                        crate::matrix::kern::kern_32x32_first(ap0, bp, kc, bs);
                    } else {
                        tile::preload_c(c0, n, 0);
                        tile::preload_c(c0.add(NR), n, 1);
                        tile::preload_c(c0.add(MR * n), n, 2);
                        tile::preload_c(c0.add(MR * n + NR), n, 3);
                        tile::microkernel_32x32_acc_nopairx(ap0, bp, kc, bs);
                    }
                    tile::store_c(c0, n, 0);
                    tile::store_c(c0.add(NR), n, 1);
                    tile::store_c(c0.add(MR * n), n, 2);
                    tile::store_c(c0.add(MR * n + NR), n, 3);

                    // Pair1: B is warm in L1 from pair0's access pattern.
                    // Re-reads B from L1 (~4 cycles/line) not L3 (~30 cycles).
                    if is_first_kc {
                        crate::matrix::kern::kern_32x32_first(ap1, bp, kc, bs);
                    } else {
                        tile::preload_c(c1, n, 0);
                        tile::preload_c(c1.add(NR), n, 1);
                        tile::preload_c(c1.add(MR * n), n, 2);
                        tile::preload_c(c1.add(MR * n + NR), n, 3);
                        tile::microkernel_32x32_acc_nopairx(ap1, bp, kc, bs);
                    }
                    tile::store_c(c1, n, 0);
                    tile::store_c(c1.add(NR), n, 1);
                    tile::store_c(c1.add(MR * n), n, 2);
                    tile::store_c(c1.add(MR * n + NR), n, 3);

                    jr += 2;
                }
                pair += 2;
            }

            // Odd remaining pair
            while pair < n_pairs {
                let a_off = pair * k * MR * 2 + pc * MR * 2;
                let ap = a_buf.as_ptr().add(a_off) as *const u8;
                let ir = pair * 2;
                let mut jr = 0usize;
                while jr + 2 <= n_nr {
                    let c0 = c.as_mut_ptr().add(ir * MR * n + jr * NR);
                    let bp = b_base.add(pc * n + jr * NR) as *const u8;
                    if is_first_kc {
                        crate::matrix::kern::kern_32x32_first(ap, bp, kc, bs);
                    } else {
                        tile::preload_c(c0, n, 0);
                        tile::preload_c(c0.add(NR), n, 1);
                        tile::preload_c(c0.add(MR * n), n, 2);
                        tile::preload_c(c0.add(MR * n + NR), n, 3);
                        if b_aligned {
                            tile::microkernel_32x32_acc(ap, ap, bp, std::ptr::null(), kc, bs);
                        } else {
                            tile::microkernel_32x32_acc_nopairx(ap, bp, kc, bs);
                        }
                    }
                    tile::store_c(c0, n, 0);
                    tile::store_c(c0.add(NR), n, 1);
                    tile::store_c(c0.add(MR * n), n, 2);
                    tile::store_c(c0.add(MR * n + NR), n, 3);
                    jr += 2;
                }
                pair += 1;
            }
            pc += kc;
        }
    }

    super::PACK_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        cache.a_src = a_src_id;
        cache.a_dims = a_dims;
        cache.a = Some(a_pack);
    });
}

/// NEON-accelerated interleaved A pack for 2 strips (32 rows).
/// Output stride = 32 floats (128 bytes) per k column for pair LDY.
#[cfg(target_arch = "aarch64")]
pub(super) fn pack_a_interleaved_neon(
    a: &[f32],
    lda: usize,
    row_start: usize,
    kc: usize,
    dst: &mut [f32],
) {
    use core::arch::aarch64::*;

    let mut rows = [0usize; 32];
    for (i, row) in rows.iter_mut().enumerate() {
        *row = (row_start + i) * lda;
    }

    let mut p = 0;
    while p + 4 <= kc {
        unsafe {
            let d = dst.as_mut_ptr();
            let ap = a.as_ptr();
            for ig in 0..8u32 {
                let i = (ig * 4) as usize;
                let r0 = vld1q_f32(ap.add(rows[i] + p));
                let r1 = vld1q_f32(ap.add(rows[i + 1] + p));
                let r2 = vld1q_f32(ap.add(rows[i + 2] + p));
                let r3 = vld1q_f32(ap.add(rows[i + 3] + p));
                let lo01 = vzip1q_f32(r0, r1);
                let hi01 = vzip2q_f32(r0, r1);
                let lo23 = vzip1q_f32(r2, r3);
                let hi23 = vzip2q_f32(r2, r3);
                let c0 = vreinterpretq_f32_f64(vzip1q_f64(
                    vreinterpretq_f64_f32(lo01),
                    vreinterpretq_f64_f32(lo23),
                ));
                let c1 = vreinterpretq_f32_f64(vzip2q_f64(
                    vreinterpretq_f64_f32(lo01),
                    vreinterpretq_f64_f32(lo23),
                ));
                let c2 = vreinterpretq_f32_f64(vzip1q_f64(
                    vreinterpretq_f64_f32(hi01),
                    vreinterpretq_f64_f32(hi23),
                ));
                let c3 = vreinterpretq_f32_f64(vzip2q_f64(
                    vreinterpretq_f64_f32(hi01),
                    vreinterpretq_f64_f32(hi23),
                ));
                // Interleaved stride: 32 floats per k-step.
                vst1q_f32(d.add(p * 32 + i), c0);
                vst1q_f32(d.add((p + 1) * 32 + i), c1);
                vst1q_f32(d.add((p + 2) * 32 + i), c2);
                vst1q_f32(d.add((p + 3) * 32 + i), c3);
            }
        }
        p += 4;
    }
    while p < kc {
        for i in 0..32 {
            dst[p * 32 + i] = a[rows[i] + p];
        }
        p += 1;
    }
}

/// Original 16×64 path with interleaved pack/compute.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn matmul_f32_16x64(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    n_mr: usize,
    n_nr: usize,
    bs: usize,
    _first_k: bool,
) {
    let a_need = n_mr * MR * k;
    let mut a_pack = super::PACK_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        super::cached_buf(&mut cache.a, a_need)
    });

    let first_mc = MR.min(m);
    super::pack_a_mr(a, k, 0, 0, first_mc, k, a_pack.as_mut_slice());

    unsafe {
        for ir in 0..n_mr {
            let mr = MR.min(m - ir * MR);
            let ap = a_pack.as_slice().as_ptr().add(ir * k * MR) as *const u8;
            let mut jr = 0usize;

            while jr + 4 <= n_nr && mr == MR && (n - jr * NR) >= 4 * NR {
                let cp = c.as_mut_ptr().add(ir * MR * n + jr * NR);
                for t in 0u8..4 {
                    tile::preload_c(cp.add(t as usize * NR), n, t);
                }
                tile::microkernel_16x64_acc(
                    ap,
                    b.as_ptr().add(jr * NR) as *const u8,
                    b.as_ptr().add((jr + 1) * NR) as *const u8,
                    b.as_ptr().add((jr + 2) * NR) as *const u8,
                    b.as_ptr().add((jr + 3) * NR) as *const u8,
                    k,
                    bs,
                );
                for t in 0u8..4 {
                    tile::store_c(cp.add(t as usize * NR), n, t);
                }
                jr += 4;
            }
            while jr + 2 <= n_nr && mr == MR && (n - jr * NR) >= 2 * NR {
                let cp = c.as_mut_ptr().add(ir * MR * n + jr * NR);
                tile::preload_c(cp, n, 0);
                tile::preload_c(cp.add(NR), n, 1);
                tile::microkernel_16x32_acc(
                    ap,
                    b.as_ptr().add(jr * NR) as *const u8,
                    b.as_ptr().add((jr + 1) * NR) as *const u8,
                    k,
                    bs,
                );
                tile::store_c(cp, n, 0);
                tile::store_c(cp.add(NR), n, 1);
                jr += 2;
            }
            while jr < n_nr {
                let nr = NR.min(n - jr * NR);
                if mr == MR && nr == NR {
                    let cp = c.as_mut_ptr().add(ir * MR * n + jr * NR);
                    tile::preload_c(cp, n, 0);
                    tile::microkernel_16x16_acc(ap, b.as_ptr().add(jr * NR) as *const u8, k, bs);
                    tile::store_c(cp, n, 0);
                } else {
                    for i in 0..mr {
                        for j in 0..nr {
                            let mut acc = 0.0f32;
                            for p in 0..k {
                                acc += a[(ir * MR + i) * k + p] * b[p * n + jr * NR + j];
                            }
                            c[(ir * MR + i) * n + jr * NR + j] += acc;
                        }
                    }
                }
                jr += 1;
            }

            if ir + 1 < n_mr {
                let next_mc = MR.min(m - (ir + 1) * MR);
                let off = (ir + 1) * k * MR;
                super::pack_a_mr(
                    a,
                    k,
                    (ir + 1) * MR,
                    0,
                    next_mc,
                    k,
                    &mut a_pack.as_mut_slice()[off..],
                );
            }
        }
    }

    super::PACK_CACHE.with(|c| {
        c.borrow_mut().a = Some(a_pack);
    });
}
