//! LUT permute benchmark: SME-mode SVE TBL vs NEON TBL.

#[path = "common.rs"]
mod common;
use common::*;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_permute(table: &[u8; 16], idx: &[u8], out: &mut [u8]) {
    use core::arch::aarch64::*;
    let t = vld1q_u8(table.as_ptr());
    let n = idx.len();
    let mut i = 0;
    while i + 16 <= n {
        let v = vld1q_u8(idx.as_ptr().add(i));
        let r = vqtbl1q_u8(t, v);
        vst1q_u8(out.as_mut_ptr().add(i), r);
        i += 16;
    }
    while i < n {
        out[i] = table[(idx[i] & 0x0F) as usize];
        i += 1;
    }
}

fn main() {
    let caps = acpu::scan();
    println!("acpu LUT benchmark — SVE TBL (streaming) vs NEON vqtbl1q_u8");
    println!("chip: {}", caps.chip);
    if !caps.has_sme {
        println!("SKIP: FEAT_SME not present");
        return;
    }
    println!();

    let table: [u8; 16] = [
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26,
        0x27,
    ];

    let sizes: &[usize] = &[64, 256, 1024, 4096, 16384, 65536];

    println!(
        "{:<14} {:>10} {:>10} {:>10}",
        "n bytes", "NEON ns", "SME ns", "speedup"
    );

    let mut wins = 0;
    let mut total = 0;

    for &n in sizes {
        let idx: Vec<u8> = (0..n).map(|i| (i as u8) & 0x0F).collect();
        let mut out_neon = vec![0u8; n];
        let mut out_sme = vec![0u8; n];

        // Warm
        for _ in 0..3 {
            unsafe { neon_permute(&table, &idx, &mut out_neon) };
            acpu::lut::permute_u8(&table, &idx, &mut out_sme).unwrap();
        }
        // Sanity
        for (i, (a, b)) in out_neon.iter().zip(out_sme.iter()).enumerate() {
            assert_eq!(a, b, "mismatch at lane {i} (n={n})");
        }

        let iters = if n < 1024 {
            5000
        } else if n < 16384 {
            500
        } else {
            50
        };

        let neon_ns = best_of(
            || unsafe { neon_permute(&table, &idx, &mut out_neon) },
            iters,
        );
        let sme_ns = best_of(
            || {
                acpu::lut::permute_u8(&table, &idx, &mut out_sme).unwrap();
            },
            iters,
        );

        let ratio = neon_ns as f64 / sme_ns as f64;
        total += 1;
        let status = if ratio > 1.05 {
            wins += 1;
            "WIN"
        } else if ratio > 0.95 {
            "TIE"
        } else {
            "LOSS"
        };
        println!(
            "{:<14} {:>10} {:>10} {:>9.2}× {}",
            n, neon_ns, sme_ns, ratio, status
        );
    }

    println!();
    println!("wins / total: {wins}/{total}");
}
