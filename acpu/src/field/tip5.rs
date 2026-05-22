//! Tip5 permutation + sponge — bit-identical to `twenty_first::Tip5`.
//!
//! Tip5 is the arithmetization-oriented Goldilocks-field hash used by Triton VM.
//! State width 16, 5 rounds. Each round:
//!   1. S-box layer: split-and-lookup on the first 4 elements, x⁷ on the last 12.
//!   2. MDS multiplication by a 16×16 circulant matrix (column 0 fixed).
//!   3. Add round constants.
//!
//! Internal representation matches twenty-first exactly: state is `[u64; 16]`
//! of *Montgomery-form raw* field elements (`a * 2^64 mod p`). The public API
//! accepts and returns canonical (`a mod p`, `0 ≤ a < p`) u64 values, which
//! are converted at the boundary.
//!
//! Reference parameters: <https://eprint.iacr.org/2023/107.pdf>.
//! Reference impl:       twenty-first 1.1.0, `src/tip5/mod.rs`.

// ── field-arithmetic constants (Goldilocks p = 2^64 - 2^32 + 1) ──────────

/// Goldilocks prime.
const P: u64 = 0xffff_ffff_0000_0001;

/// 2^128 mod p — the Montgomery conversion constant `R²`.
const R2: u64 = 0xffff_fffe_0000_0001;

// ── Tip5 parameters ──────────────────────────────────────────────────────

/// State width.
pub const STATE_SIZE: usize = 16;

/// Number of split-and-lookup S-box positions (the first `NUM_SPLIT_AND_LOOKUP`).
pub const NUM_SPLIT_AND_LOOKUP: usize = 4;

/// Sponge capacity (state slots not exposed via absorb/squeeze).
pub const CAPACITY: usize = 6;

/// Sponge rate (state slots exposed via absorb/squeeze).
pub const RATE: usize = 10;

/// Digest length in field elements.
pub const DIGEST_LEN: usize = 5;

/// Permutation rounds.
pub const NUM_ROUNDS: usize = 5;

// ── lookup table (high-degree S-box) ─────────────────────────────────────

/// Tip5 8-bit lookup table. Bit-identical to `twenty_first::tip5::LOOKUP_TABLE`.
pub const LOOKUP_TABLE: [u8; 256] = [
    0, 7, 26, 63, 124, 215, 85, 254, 214, 228, 45, 185, 140, 173, 33, 240, 29, 177, 176, 32, 8,
    110, 87, 202, 204, 99, 150, 106, 230, 14, 235, 128, 213, 239, 212, 138, 23, 130, 208, 6, 44,
    71, 93, 116, 146, 189, 251, 81, 199, 97, 38, 28, 73, 179, 95, 84, 152, 48, 35, 119, 49, 88,
    242, 3, 148, 169, 72, 120, 62, 161, 166, 83, 175, 191, 137, 19, 100, 129, 112, 55, 221, 102,
    218, 61, 151, 237, 68, 164, 17, 147, 46, 234, 203, 216, 22, 141, 65, 57, 123, 12, 244, 54, 219,
    231, 96, 77, 180, 154, 5, 253, 133, 165, 98, 195, 205, 134, 245, 30, 9, 188, 59, 142, 186, 197,
    181, 144, 92, 31, 224, 163, 111, 74, 58, 69, 113, 196, 67, 246, 225, 10, 121, 50, 60, 157, 90,
    122, 2, 250, 101, 75, 178, 159, 24, 36, 201, 11, 243, 132, 198, 190, 114, 233, 39, 52, 21, 209,
    108, 238, 91, 187, 18, 104, 194, 37, 153, 34, 200, 143, 126, 155, 236, 118, 64, 80, 172, 89,
    94, 193, 135, 183, 86, 107, 252, 13, 167, 206, 136, 220, 207, 103, 171, 160, 76, 182, 227, 217,
    158, 56, 174, 4, 66, 109, 139, 162, 184, 211, 249, 47, 125, 232, 117, 43, 16, 42, 127, 20, 241,
    25, 149, 105, 156, 51, 53, 168, 145, 247, 223, 79, 78, 226, 15, 222, 82, 115, 70, 210, 27, 41,
    1, 170, 40, 131, 192, 229, 248, 255,
];

// ── round constants (canonical u64; converted to Montgomery raw at compile) ──

/// Canonical (value) round constants. The Montgomery-raw constants used at
/// runtime are derived from these via `monty_raw` in a `const` block.
const ROUND_CONSTANTS_VALUE: [u64; NUM_ROUNDS * STATE_SIZE] = [
    13_630_775_303_355_457_758,
    16_896_927_574_093_233_874,
    10_379_449_653_650_130_495,
    1_965_408_364_413_093_495,
    15_232_538_947_090_185_111,
    15_892_634_398_091_747_074,
    3_989_134_140_024_871_768,
    2_851_411_912_127_730_865,
    8_709_136_439_293_758_776,
    3_694_858_669_662_939_734,
    12_692_440_244_315_327_141,
    10_722_316_166_358_076_749,
    12_745_429_320_441_639_448,
    17_932_424_223_723_990_421,
    7_558_102_534_867_937_463,
    15_551_047_435_855_531_404,
    17_532_528_648_579_384_106,
    5_216_785_850_422_679_555,
    15_418_071_332_095_031_847,
    11_921_929_762_955_146_258,
    9_738_718_993_677_019_874,
    3_464_580_399_432_997_147,
    13_408_434_769_117_164_050,
    264_428_218_649_616_431,
    4_436_247_869_008_081_381,
    4_063_129_435_850_804_221,
    2_865_073_155_741_120_117,
    5_749_834_437_609_765_994,
    6_804_196_764_189_408_435,
    17_060_469_201_292_988_508,
    9_475_383_556_737_206_708,
    12_876_344_085_611_465_020,
    13_835_756_199_368_269_249,
    1_648_753_455_944_344_172,
    9_836_124_473_569_258_483,
    12_867_641_597_107_932_229,
    11_254_152_636_692_960_595,
    16_550_832_737_139_861_108,
    11_861_573_970_480_733_262,
    1_256_660_473_588_673_495,
    13_879_506_000_676_455_136,
    10_564_103_842_682_358_721,
    16_142_842_524_796_397_521,
    3_287_098_591_948_630_584,
    685_911_471_061_284_805,
    5_285_298_776_918_878_023,
    18_310_953_571_768_047_354,
    3_142_266_350_630_002_035,
    549_990_724_933_663_297,
    4_901_984_846_118_077_401,
    11_458_643_033_696_775_769,
    8_706_785_264_119_212_710,
    12_521_758_138_015_724_072,
    11_877_914_062_416_978_196,
    11_333_318_251_134_523_752,
    3_933_899_631_278_608_623,
    16_635_128_972_021_157_924,
    10_291_337_173_108_950_450,
    4_142_107_155_024_199_350,
    16_973_934_533_787_743_537,
    11_068_111_539_125_175_221,
    17_546_769_694_830_203_606,
    5_315_217_744_825_068_993,
    4_609_594_252_909_613_081,
    3_350_107_164_315_270_407,
    17_715_942_834_299_349_177,
    9_600_609_149_219_873_996,
    12_894_357_635_820_003_949,
    4_597_649_658_040_514_631,
    7_735_563_950_920_491_847,
    1_663_379_455_870_887_181,
    13_889_298_103_638_829_706,
    7_375_530_351_220_884_434,
    3_502_022_433_285_269_151,
    9_231_805_330_431_056_952,
    9_252_272_755_288_523_725,
    10_014_268_662_326_746_219,
    15_565_031_632_950_843_234,
    1_209_725_273_521_819_323,
    6_024_642_864_597_845_108,
];

/// Round constants in Montgomery raw form (matches `BFieldElement::new(v).0`).
const ROUND_CONSTANTS: [u64; NUM_ROUNDS * STATE_SIZE] = {
    let mut out = [0u64; NUM_ROUNDS * STATE_SIZE];
    let mut i = 0;
    while i < NUM_ROUNDS * STATE_SIZE {
        out[i] = to_monty(ROUND_CONSTANTS_VALUE[i]);
        i += 1;
    }
    out
};

/// Montgomery raw value of canonical 1 — i.e. `R = 2^64 mod p = ε`.
const ONE_MONTY: u64 = to_monty(1);

// ── Montgomery primitives (bit-identical to twenty-first BFieldElement) ──

/// Montgomery reduction. `montyred(x) = x * R⁻¹ mod p` where `R = 2^64`.
/// Bit-identical to `BFieldElement::montyred`.
#[inline(always)]
const fn montyred(x: u128) -> u64 {
    let xl = x as u64;
    let xh = (x >> 64) as u64;
    let (a, e) = xl.overflowing_add(xl << 32);
    let b = a.wrapping_sub(a >> 32).wrapping_sub(e as u64);
    let (r, c) = xh.overflowing_sub(b);
    // (1 + !P) == 0xffff_ffff (i.e. ε)
    r.wrapping_sub((1 + !P) * c as u64)
}

/// Canonical → Montgomery raw (matches `BFieldElement::new(v).0`).
#[inline(always)]
const fn to_monty(v: u64) -> u64 {
    montyred((v as u128) * (R2 as u128))
}

/// Montgomery raw → canonical (matches `BFieldElement::value()`).
#[inline(always)]
const fn from_monty(raw: u64) -> u64 {
    montyred(raw as u128)
}

/// `BFieldElement::Add` on raw Montgomery values.
///
/// Implements `a + b = a - (p - b)`. The `if c1` branch corrects ordinary
/// overflow and *also* fixes a degenerate `a ≥ p` when `b` is small enough
/// (`b < p + 2 - 2^32`), which is true for every round constant.
#[inline(always)]
const fn raw_add(a: u64, b: u64) -> u64 {
    let (x1, c1) = a.overflowing_sub(P - b);
    if c1 {
        x1.wrapping_add(P)
    } else {
        x1
    }
}

/// Montgomery multiplication on raw values.
#[inline(always)]
const fn raw_mul(a: u64, b: u64) -> u64 {
    montyred((a as u128) * (b as u128))
}

// ── permutation kernels ──────────────────────────────────────────────────

/// Split-and-lookup S-box on a single raw element. Splits the 8 little-endian
/// bytes of the Montgomery raw u64, looks each up in `LOOKUP_TABLE`, recombines.
#[inline(always)]
fn split_and_lookup(raw: u64) -> u64 {
    let mut bytes = raw.to_le_bytes();
    let mut i = 0;
    while i < 8 {
        bytes[i] = LOOKUP_TABLE[bytes[i] as usize];
        i += 1;
    }
    u64::from_le_bytes(bytes)
}

/// S-box layer: 4 split-and-lookup, 12 x⁷.
#[inline(always)]
fn sbox_layer(state: &mut [u64; STATE_SIZE]) {
    let mut i = 0;
    while i < NUM_SPLIT_AND_LOOKUP {
        state[i] = split_and_lookup(state[i]);
        i += 1;
    }
    while i < STATE_SIZE {
        let s = state[i];
        let sq = raw_mul(s, s);
        let qu = raw_mul(sq, sq);
        state[i] = raw_mul(s, raw_mul(sq, qu));
        i += 1;
    }
}

/// Add round constants from row `round_index`. Matches
/// `state[i] += ROUND_CONSTANTS[round_index * STATE_SIZE + i]`.
#[inline(always)]
fn add_round_constants(state: &mut [u64; STATE_SIZE], round_index: usize) {
    let off = round_index * STATE_SIZE;
    let mut i = 0;
    while i < STATE_SIZE {
        state[i] = raw_add(state[i], ROUND_CONSTANTS[off + i]);
        i += 1;
    }
}

mod mds;
use mds::mds_generated;

/// One Tip5 round.
#[inline(always)]
fn round(state: &mut [u64; STATE_SIZE], round_index: usize) {
    sbox_layer(state);
    mds_generated(state);
    add_round_constants(state, round_index);
}

// ── public API ───────────────────────────────────────────────────────────

/// Apply the full Tip5 permutation in place.
///
/// Input/output are **canonical** Goldilocks values (`0 ≤ a < p`). The
/// function converts to and from twenty-first's internal Montgomery raw form.
#[inline(never)]
pub fn tip5_permute(state: &mut [u64; STATE_SIZE]) {
    // canonical → Montgomery raw
    let mut s = [0u64; STATE_SIZE];
    for i in 0..STATE_SIZE {
        s[i] = to_monty(state[i]);
    }

    for r in 0..NUM_ROUNDS {
        round(&mut s, r);
    }

    // Montgomery raw → canonical
    for i in 0..STATE_SIZE {
        state[i] = from_monty(s[i]);
    }
}

/// Hash two 5-element digests into one. Matches `Tip5::hash_pair`.
///
/// Inputs and output are canonical Goldilocks values.
#[inline(never)]
pub fn tip5_hash_pair(left: [u64; DIGEST_LEN], right: [u64; DIGEST_LEN]) -> [u64; DIGEST_LEN] {
    // Domain::FixedLength: state[RATE..STATE_SIZE] = ONE_MONTY, rest zero.
    let mut s = [0u64; STATE_SIZE];
    for slot in &mut s[RATE..] {
        *slot = ONE_MONTY;
    }
    for i in 0..DIGEST_LEN {
        s[i] = to_monty(left[i]);
        s[DIGEST_LEN + i] = to_monty(right[i]);
    }

    for r in 0..NUM_ROUNDS {
        round(&mut s, r);
    }

    let mut out = [0u64; DIGEST_LEN];
    for i in 0..DIGEST_LEN {
        out[i] = from_monty(s[i]);
    }
    out
}

/// Hash a variable-length sequence of canonical field elements.
/// Matches `Tip5::hash_varlen`.
///
/// Padding: append `[1, 0, 0, …]` until the input is a positive multiple of
/// `RATE`. Padding is always at least one element, even when `input.len()`
/// is already a multiple of `RATE`.
#[inline(never)]
pub fn tip5_hash_varlen(input: &[u64]) -> [u64; DIGEST_LEN] {
    // Domain::VariableLength: state all-zero.
    let mut s = [0u64; STATE_SIZE];

    // Absorb full RATE-sized chunks via overwrite mode.
    let mut chunks = input.chunks_exact(RATE);
    for chunk in chunks.by_ref() {
        for i in 0..RATE {
            s[i] = to_monty(chunk[i]);
        }
        for r in 0..NUM_ROUNDS {
            round(&mut s, r);
        }
    }

    // Pad and absorb the final chunk (always at least one element of padding).
    let rem = chunks.remainder();
    for i in 0..RATE {
        s[i] = if i < rem.len() {
            to_monty(rem[i])
        } else if i == rem.len() {
            ONE_MONTY
        } else {
            0
        };
    }
    for r in 0..NUM_ROUNDS {
        round(&mut s, r);
    }

    let mut out = [0u64; DIGEST_LEN];
    for i in 0..DIGEST_LEN {
        out[i] = from_monty(s[i]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monty_roundtrip() {
        for v in [0u64, 1, 2, 100, P - 1, P - 2, 0xdead_beef_cafe_babe] {
            let v = if v >= P { v - P } else { v };
            assert_eq!(from_monty(to_monty(v)), v);
        }
    }

    #[test]
    fn one_monty_is_r() {
        // R = 2^64 mod p = ε = 0xFFFF_FFFF
        assert_eq!(ONE_MONTY, 0xFFFF_FFFF);
    }

    #[test]
    fn permute_is_deterministic() {
        let mut a: [u64; 16] = core::array::from_fn(|i| i as u64 + 1);
        let mut b = a;
        tip5_permute(&mut a);
        tip5_permute(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn permute_changes_state() {
        let mut s: [u64; 16] = core::array::from_fn(|i| i as u64 + 1);
        let original = s;
        tip5_permute(&mut s);
        assert_ne!(s, original);
        for v in s {
            assert!(v < P);
        }
    }

    #[test]
    fn hash_pair_deterministic() {
        let l = [1u64, 2, 3, 4, 5];
        let r = [6u64, 7, 8, 9, 10];
        assert_eq!(tip5_hash_pair(l, r), tip5_hash_pair(l, r));
        for v in tip5_hash_pair(l, r) {
            assert!(v < P);
        }
    }

    #[test]
    fn hash_varlen_padding_boundary() {
        // Empty input still produces a digest (padding alone absorbs once).
        let d0 = tip5_hash_varlen(&[]);
        let d1 = tip5_hash_varlen(&[0]);
        // Distinct inputs ⇒ (with overwhelming probability) distinct digests.
        assert_ne!(d0, d1);
        // Multiple-of-RATE input gets a *full extra* padding absorb.
        let d10 = tip5_hash_varlen(&[7u64; 10]);
        let d11 = tip5_hash_varlen(&[7u64; 11]);
        assert_ne!(d10, d11);
    }
}
