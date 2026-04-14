//! Crypto benchmark: acpu vs Apple CommonCrypto.
//! AES-128 via AESE/AESMC vs CCCrypt, SHA-256 via SHA256H vs CC_SHA256.

#[path = "common.rs"]
mod common;
use common::*;
use std::time::Instant;

#[link(name = "System", kind = "dylib")]
extern "C" {
    // CC_SHA256(data, len, md) → md
    fn CC_SHA256(data: *const u8, len: u32, md: *mut u8) -> *mut u8;

    // CCCrypt for AES-128-ECB
    fn CCCrypt(
        op: u32,
        alg: u32,
        options: u32,
        key: *const u8,
        key_len: usize,
        iv: *const u8,
        data_in: *const u8,
        data_in_len: usize,
        data_out: *mut u8,
        data_out_avail: usize,
        data_out_moved: *mut usize,
    ) -> i32;
}

const CC_ENCRYPT: u32 = 0;
const CC_AES: u32 = 0;
const CC_ECB: u32 = 2; // kCCOptionECBMode

// ── AES key expansion (same as crypto module tests) ──────────────────────

fn aes128_key_expand(key: &[u8; 16]) -> Vec<[u8; 16]> {
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

fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(60));
        eprintln!("WATCHDOG: 60s timeout");
        std::process::exit(1);
    });

    println!("acpu crypto benchmark vs Apple CommonCrypto");
    println!();

    let mut score = Score::vs("CommonCrypto");

    // ── AES-128: acpu vs CCCrypt ─────────────────────────────────────

    score.hdr("AES-128 ECB (4096 blocks = 64KB)");
    {
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
            0x4f, 0x3c,
        ];
        let round_keys = aes128_key_expand(&key);
        let count = 4096usize;
        let data_len = count * 16;

        // acpu: encrypt 4096 blocks
        let mut blocks = vec![[0x42u8; 16]; count];
        let t_acpu = best_of(
            || {
                acpu::crypto::aes_encrypt_blocks(&mut blocks, &round_keys);
                std::hint::black_box(&blocks);
            },
            100,
        );

        // CommonCrypto: encrypt same data
        let plain = vec![0x42u8; data_len];
        let mut cipher = vec![0u8; data_len];
        let mut out_len: usize = 0;
        let t_apple = best_of(
            || unsafe {
                CCCrypt(
                    CC_ENCRYPT,
                    CC_AES,
                    CC_ECB,
                    key.as_ptr(),
                    16,
                    std::ptr::null(),
                    plain.as_ptr(),
                    data_len,
                    cipher.as_mut_ptr(),
                    data_len,
                    &mut out_len,
                );
                std::hint::black_box(&cipher);
            },
            100,
        );

        score.row("AES-128 4096blk", t_acpu, t_apple);
        let acpu_gbs = data_len as f64 / t_acpu as f64;
        let apple_gbs = data_len as f64 / t_apple as f64;
        println!("  acpu:  {:.2} GB/s", acpu_gbs);
        println!("  apple: {:.2} GB/s", apple_gbs);
    }

    // ── SHA-256: acpu vs CC_SHA256 ───────────────────────────────────

    score.hdr("SHA-256 (64-byte blocks)");
    {
        let block = [0xAAu8; 64];
        let iters = 10_000u64;

        // acpu: raw compression function
        let mut state: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        let t_acpu = best_of(
            || {
                for _ in 0..iters {
                    acpu::crypto::sha256_compress(&mut state, &block);
                }
                std::hint::black_box(&state);
            },
            10,
        );

        // CC_SHA256: full hash (includes padding overhead)
        let mut digest = [0u8; 32];
        let t_apple = best_of(
            || {
                for _ in 0..iters {
                    unsafe {
                        CC_SHA256(block.as_ptr(), 64, digest.as_mut_ptr());
                    }
                }
                std::hint::black_box(&digest);
            },
            10,
        );

        score.row(&format!("SHA-256 {iters}×64B"), t_acpu, t_apple);
        let acpu_gbs = iters as f64 * 64.0 / t_acpu as f64;
        let apple_gbs = iters as f64 * 64.0 / t_apple as f64;
        println!("  acpu:  {:.2} GB/s  (raw compress)", acpu_gbs);
        println!("  apple: {:.2} GB/s  (CC_SHA256, incl. padding)", apple_gbs);
    }

    // ── PMULL throughput ─────────────────────────────────────────────

    println!();
    println!("--- PMULL (carry-less multiply 64×64) ---");
    {
        let mut a = 0xDEADBEEFCAFEBABEu64;
        let b = 0x1234567890ABCDEFu64;
        let iters = 1_000_000u64;
        let t = best_of(
            || {
                let mut acc = 0u128;
                for _ in 0..iters {
                    acc = acc.wrapping_add(acpu::crypto::pmull_64(a, b));
                    a = a.wrapping_add(1);
                }
                std::hint::black_box(acc);
            },
            5,
        );
        println!(
            "  {} ops: {:.0} Mops/s",
            iters,
            iters as f64 / t as f64 * 1000.0
        );
    }

    println!();
    score.summary();
}
