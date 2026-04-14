//! Crypto hardware primitives — AES, SHA-256, PMULL.
//! Uses ARM Crypto Extensions (FEAT_AES, FEAT_SHA256, FEAT_PMULL).

// ---------------------------------------------------------------------------
// AES single-block encrypt
// ---------------------------------------------------------------------------

/// AES encrypt a single 16-byte block in place.
///
/// `round_keys` length determines the variant:
/// - 11 keys = AES-128  (10 rounds)
/// - 13 keys = AES-192  (12 rounds)
/// - 15 keys = AES-256  (14 rounds)
///
/// Each intermediate round: AESE (AddRoundKey + SubBytes + ShiftRows) then
/// AESMC (MixColumns). Final round: AESE without AESMC, then XOR last key.
#[cfg(target_arch = "aarch64")]
pub fn aes_encrypt_block(block: &mut [u8; 16], round_keys: &[[u8; 16]]) {
    assert!(
        matches!(round_keys.len(), 11 | 13 | 15),
        "round_keys length must be 11 (AES-128), 13 (AES-192), or 15 (AES-256)"
    );
    let nr = round_keys.len() - 1; // number of rounds
    unsafe {
        use core::arch::aarch64::*;
        let mut state = vld1q_u8(block.as_ptr());

        // rounds 0..nr-2: AESE + AESMC
        for rk_bytes in &round_keys[..nr - 1] {
            let rk = vld1q_u8(rk_bytes.as_ptr());
            state = vaeseq_u8(state, rk);
            state = vaesmcq_u8(state);
        }

        // final round: AESE (includes AddRoundKey with key[nr-1]) then XOR key[nr]
        let rk_penult = vld1q_u8(round_keys[nr - 1].as_ptr());
        state = vaeseq_u8(state, rk_penult);
        let rk_last = vld1q_u8(round_keys[nr].as_ptr());
        state = veorq_u8(state, rk_last);

        vst1q_u8(block.as_mut_ptr(), state);
    }
}

/// AES encrypt multiple 16-byte blocks with 4-way interleaving.
///
/// Each block is encrypted independently (ECB mode).
/// For blocks.len() >= 4, processes 4 blocks in parallel to exploit
/// instruction-level parallelism in the AES pipeline.
#[cfg(target_arch = "aarch64")]
pub fn aes_encrypt_blocks(blocks: &mut [[u8; 16]], round_keys: &[[u8; 16]]) {
    assert!(
        matches!(round_keys.len(), 11 | 13 | 15),
        "round_keys length must be 11 (AES-128), 13 (AES-192), or 15 (AES-256)"
    );
    let nr = round_keys.len() - 1;
    let len = blocks.len();
    let mut i = 0;

    unsafe {
        use core::arch::aarch64::*;

        // 4-way interleaved path
        while i + 4 <= len {
            let mut s0 = vld1q_u8(blocks[i].as_ptr());
            let mut s1 = vld1q_u8(blocks[i + 1].as_ptr());
            let mut s2 = vld1q_u8(blocks[i + 2].as_ptr());
            let mut s3 = vld1q_u8(blocks[i + 3].as_ptr());

            for rk_bytes in &round_keys[..nr - 1] {
                let rk = vld1q_u8(rk_bytes.as_ptr());
                s0 = vaeseq_u8(s0, rk);
                s0 = vaesmcq_u8(s0);
                s1 = vaeseq_u8(s1, rk);
                s1 = vaesmcq_u8(s1);
                s2 = vaeseq_u8(s2, rk);
                s2 = vaesmcq_u8(s2);
                s3 = vaeseq_u8(s3, rk);
                s3 = vaesmcq_u8(s3);
            }

            let rk_pen = vld1q_u8(round_keys[nr - 1].as_ptr());
            let rk_last = vld1q_u8(round_keys[nr].as_ptr());
            s0 = veorq_u8(vaeseq_u8(s0, rk_pen), rk_last);
            s1 = veorq_u8(vaeseq_u8(s1, rk_pen), rk_last);
            s2 = veorq_u8(vaeseq_u8(s2, rk_pen), rk_last);
            s3 = veorq_u8(vaeseq_u8(s3, rk_pen), rk_last);

            vst1q_u8(blocks[i].as_mut_ptr(), s0);
            vst1q_u8(blocks[i + 1].as_mut_ptr(), s1);
            vst1q_u8(blocks[i + 2].as_mut_ptr(), s2);
            vst1q_u8(blocks[i + 3].as_mut_ptr(), s3);

            i += 4;
        }
    }

    // remainder: scalar path
    while i < len {
        aes_encrypt_block(&mut blocks[i], round_keys);
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// SHA-256 compression
// ---------------------------------------------------------------------------

/// SHA-256 round constants K[0..63].
#[rustfmt::skip]
const K256: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA-256 compression function for one 64-byte block.
///
/// Updates `state` (8 x u32, H0..H7) using the ARM SHA-256 crypto
/// instructions: SHA256H, SHA256H2, SHA256SU0, SHA256SU1.
#[cfg(target_arch = "aarch64")]
pub fn sha256_compress(state: &mut [u32; 8], block: &[u8; 64]) {
    unsafe {
        use core::arch::aarch64::*;

        // load state: ABCD in low, EFGH in high
        let abcd_save = vld1q_u32(state.as_ptr());
        let efgh_save = vld1q_u32(state.as_ptr().add(4));

        let mut abcd = abcd_save;
        let mut efgh = efgh_save;

        // load message schedule W[0..15] as big-endian u32s
        let mut w0 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block.as_ptr())));
        let mut w1 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block.as_ptr().add(16))));
        let mut w2 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block.as_ptr().add(32))));
        let mut w3 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block.as_ptr().add(48))));

        // 16 rounds, 4 at a time
        macro_rules! sha256_4rounds {
            ($w:expr, $k_offset:expr) => {
                let k = vld1q_u32(K256.as_ptr().add($k_offset));
                let wk = vaddq_u32($w, k);
                let tmp = abcd;
                abcd = vsha256hq_u32(abcd, efgh, wk);
                efgh = vsha256h2q_u32(efgh, tmp, wk);
            };
        }

        // Rounds 0-3
        sha256_4rounds!(w0, 0);
        // Rounds 4-7
        sha256_4rounds!(w1, 4);
        // Rounds 8-11
        sha256_4rounds!(w2, 8);
        // Rounds 12-15
        sha256_4rounds!(w3, 12);

        // Rounds 16-63: schedule + compress
        macro_rules! sha256_4rounds_sched {
            ($w0:expr, $w1:expr, $w2:expr, $w3:expr, $k_offset:expr) => {
                // message schedule: w0 = sigma1(w2,w3) + w0 + sigma0(w1)
                $w0 = vsha256su1q_u32(vsha256su0q_u32($w0, $w1), $w2, $w3);
                sha256_4rounds!($w0, $k_offset);
            };
        }

        sha256_4rounds_sched!(w0, w1, w2, w3, 16);
        sha256_4rounds_sched!(w1, w2, w3, w0, 20);
        sha256_4rounds_sched!(w2, w3, w0, w1, 24);
        sha256_4rounds_sched!(w3, w0, w1, w2, 28);

        sha256_4rounds_sched!(w0, w1, w2, w3, 32);
        sha256_4rounds_sched!(w1, w2, w3, w0, 36);
        sha256_4rounds_sched!(w2, w3, w0, w1, 40);
        sha256_4rounds_sched!(w3, w0, w1, w2, 44);

        sha256_4rounds_sched!(w0, w1, w2, w3, 48);
        sha256_4rounds_sched!(w1, w2, w3, w0, 52);
        sha256_4rounds_sched!(w2, w3, w0, w1, 56);
        sha256_4rounds_sched!(w3, w0, w1, w2, 60);

        // add saved state
        abcd = vaddq_u32(abcd, abcd_save);
        efgh = vaddq_u32(efgh, efgh_save);

        vst1q_u32(state.as_mut_ptr(), abcd);
        vst1q_u32(state.as_mut_ptr().add(4), efgh);
    }
}

// ---------------------------------------------------------------------------
// PMULL — carry-less multiplication
// ---------------------------------------------------------------------------

/// 64x64 -> 128 carry-less multiplication using PMULL/PMULL2.
///
/// Used for GCM (Galois/Counter Mode) and other GF(2^128) arithmetic.
#[cfg(target_arch = "aarch64")]
pub fn pmull_64(a: u64, b: u64) -> u128 {
    unsafe {
        use core::arch::aarch64::*;
        let va = vreinterpretq_p64_u64(vdupq_n_u64(a));
        let vb = vreinterpretq_p64_u64(vdupq_n_u64(b));
        vmull_p64(vgetq_lane_p64(va, 0) as u64, vgetq_lane_p64(vb, 0) as u64)
    }
}

// ---------------------------------------------------------------------------
// Scalar fallbacks (non-aarch64)
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "aarch64"))]
pub fn aes_encrypt_block(block: &mut [u8; 16], round_keys: &[[u8; 16]]) {
    let _ = (block, round_keys);
    unimplemented!("AES encrypt requires aarch64 with FEAT_AES");
}

#[cfg(not(target_arch = "aarch64"))]
pub fn aes_encrypt_blocks(blocks: &mut [[u8; 16]], round_keys: &[[u8; 16]]) {
    let _ = (blocks, round_keys);
    unimplemented!("AES encrypt requires aarch64 with FEAT_AES");
}

#[cfg(not(target_arch = "aarch64"))]
pub fn sha256_compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let _ = (state, block);
    unimplemented!("SHA-256 compress requires aarch64 with FEAT_SHA256");
}

#[cfg(not(target_arch = "aarch64"))]
pub fn pmull_64(a: u64, b: u64) -> u128 {
    let _ = (a, b);
    unimplemented!("PMULL requires aarch64 with FEAT_PMULL");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // AES-128 test vector from FIPS 197 Appendix B
    #[test]
    fn test_aes128_encrypt() {
        // Key: 2b7e151628aed2a6abf7158809cf4f3c
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
            0x4f, 0x3c,
        ];

        // Expand key to 11 round keys (AES-128 key schedule)
        let round_keys = aes128_key_expand(&key);

        // Plaintext: 3243f6a8885a308d313198a2e0370734
        let mut block: [u8; 16] = [
            0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37,
            0x07, 0x34,
        ];

        // Expected ciphertext: 3925841d02dc09fbdc118597196a0b32
        let expected: [u8; 16] = [
            0x39, 0x25, 0x84, 0x1d, 0x02, 0xdc, 0x09, 0xfb, 0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a,
            0x0b, 0x32,
        ];

        aes_encrypt_block(&mut block, &round_keys);
        assert_eq!(block, expected);
    }

    #[test]
    fn test_aes128_encrypt_blocks() {
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
            0x4f, 0x3c,
        ];
        let round_keys = aes128_key_expand(&key);

        let plain: [u8; 16] = [
            0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37,
            0x07, 0x34,
        ];
        let expected: [u8; 16] = [
            0x39, 0x25, 0x84, 0x1d, 0x02, 0xdc, 0x09, 0xfb, 0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a,
            0x0b, 0x32,
        ];

        // test with 5 blocks (exercises both 4-way and remainder)
        let mut blocks = vec![plain; 5];
        aes_encrypt_blocks(&mut blocks, &round_keys);
        for b in &blocks {
            assert_eq!(b, &expected);
        }
    }

    #[test]
    fn test_sha256_compress() {
        // SHA-256 of empty string first block (padded)
        // Initial hash values H0..H7
        let mut state: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];

        // Padded empty message: 0x80 followed by zeros, length = 0 in last 8 bytes
        let mut block = [0u8; 64];
        block[0] = 0x80;

        sha256_compress(&mut state, &block);

        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let expected: [u32; 8] = [
            0xe3b0c442, 0x98fc1c14, 0x9afbf4c8, 0x996fb924, 0x27ae41e4, 0x649b934c, 0xa495991b,
            0x7852b855,
        ];
        assert_eq!(state, expected);
    }

    #[test]
    fn test_pmull_64() {
        // PMULL(1, x) = x (carry-less multiply by 1 is identity)
        let r = pmull_64(1, 0xDEADBEEF);
        assert_eq!(r as u64, 0xDEADBEEF);

        // PMULL(0, x) = 0
        let r = pmull_64(0, 0x12345678);
        assert_eq!(r, 0);

        // PMULL(a, b) = PMULL(b, a) — commutativity
        let a = 0x123456789ABCDEF0u64;
        let b = 0xFEDCBA9876543210u64;
        assert_eq!(pmull_64(a, b), pmull_64(b, a));

        // Known value: PMULL(3, 3) = 5 in GF(2)
        // 3 = 0b11, 3 * 3 in GF(2): x*(x+1) + (x+1) = x^2 + x + x + 1 = x^2 + 1 = 5
        let r = pmull_64(3, 3);
        assert_eq!(r as u64, 5);
    }

    // -----------------------------------------------------------------------
    // AES-128 key expansion (test helper only)
    // -----------------------------------------------------------------------

    fn aes128_key_expand(key: &[u8; 16]) -> Vec<[u8; 16]> {
        let rcon: [u8; 10] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36];

        let mut w = vec![0u32; 44];
        for i in 0..4 {
            w[i] = u32::from_be_bytes([key[4 * i], key[4 * i + 1], key[4 * i + 2], key[4 * i + 3]]);
        }

        for i in 4..44 {
            let mut tmp = w[i - 1];
            if i % 4 == 0 {
                tmp = tmp.rotate_left(8);
                tmp = sub_word(tmp) ^ ((rcon[i / 4 - 1] as u32) << 24);
            }
            w[i] = w[i - 4] ^ tmp;
        }

        let mut round_keys = Vec::with_capacity(11);
        for r in 0..11 {
            let mut rk = [0u8; 16];
            for j in 0..4 {
                let bytes = w[4 * r + j].to_be_bytes();
                rk[4 * j..4 * j + 4].copy_from_slice(&bytes);
            }
            round_keys.push(rk);
        }
        round_keys
    }

    fn sub_word(w: u32) -> u32 {
        let b = w.to_be_bytes();
        u32::from_be_bytes([
            SBOX[b[0] as usize],
            SBOX[b[1] as usize],
            SBOX[b[2] as usize],
            SBOX[b[3] as usize],
        ])
    }

    #[rustfmt::skip]
    const SBOX: [u8; 256] = [
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
}
