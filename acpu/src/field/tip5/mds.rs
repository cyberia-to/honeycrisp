//! Tip5 MDS layer — bit-identical transcription of `Tip5::mds_generated`
//! plus `generated_function` from twenty-first 1.1.0.
//!
//! Strategy: split each state element's raw u64 into hi (high 32 bits) and lo
//! (low 32 bits). Run the (pure-`wrapping_*`) generated function on each half
//! independently. Recombine via `s = (lo[r] >> 4) + (hi[r] << 28)`, fold the
//! u128 down to a raw-Montgomery u64.

/// In-place MDS layer. Bit-identical to `Tip5::mds_generated`.
#[inline(always)]
pub fn mds_generated_inplace(state: &mut [u64; 16]) {
    let mut lo = [0u64; 16];
    let mut hi = [0u64; 16];
    let mut i = 0;
    while i < 16 {
        let b = state[i];
        hi[i] = b >> 32;
        lo[i] = b & 0xffff_ffff;
        i += 1;
    }

    let lo = generated_function(lo);
    let hi = generated_function(hi);

    let mut r = 0;
    while r < 16 {
        // s = (lo >> 4) + (hi << 28), computed as u128.
        let s = (lo[r] >> 4) as u128 + ((hi[r] as u128) << 28);
        let s_hi = (s >> 64) as u64;
        let s_lo = s as u64;

        let (res, over) = s_lo.overflowing_add(s_hi.wrapping_mul(0xffff_ffff));
        state[r] = if over {
            res.wrapping_add(0xffff_ffff)
        } else {
            res
        };
        r += 1;
    }
}

/// `Tip5::generated_function` — pure `wrapping_*` arithmetic on `[u64; 16]`.
/// Translated mechanically from twenty-first 1.1.0; do not rearrange.
#[inline(always)]
#[allow(clippy::too_many_lines)]
pub fn generated_function(input: [u64; 16]) -> [u64; 16] {
    let node_34 = input[0].wrapping_add(input[8]);
    let node_38 = input[4].wrapping_add(input[12]);
    let node_36 = input[2].wrapping_add(input[10]);
    let node_40 = input[6].wrapping_add(input[14]);
    let node_35 = input[1].wrapping_add(input[9]);
    let node_39 = input[5].wrapping_add(input[13]);
    let node_37 = input[3].wrapping_add(input[11]);
    let node_41 = input[7].wrapping_add(input[15]);
    let node_50 = node_34.wrapping_add(node_38);
    let node_52 = node_36.wrapping_add(node_40);
    let node_51 = node_35.wrapping_add(node_39);
    let node_53 = node_37.wrapping_add(node_41);
    let node_160 = input[0].wrapping_sub(input[8]);
    let node_161 = input[1].wrapping_sub(input[9]);
    let node_165 = input[5].wrapping_sub(input[13]);
    let node_163 = input[3].wrapping_sub(input[11]);
    let node_167 = input[7].wrapping_sub(input[15]);
    let node_162 = input[2].wrapping_sub(input[10]);
    let node_166 = input[6].wrapping_sub(input[14]);
    let node_164 = input[4].wrapping_sub(input[12]);
    let node_58 = node_50.wrapping_add(node_52);
    let node_59 = node_51.wrapping_add(node_53);
    let node_90 = node_34.wrapping_sub(node_38);
    let node_91 = node_35.wrapping_sub(node_39);
    let node_93 = node_37.wrapping_sub(node_41);
    let node_92 = node_36.wrapping_sub(node_40);
    let node_64 = node_58.wrapping_add(node_59).wrapping_mul(524757);
    let node_67 = node_58.wrapping_sub(node_59).wrapping_mul(52427);
    let node_71 = node_50.wrapping_sub(node_52);
    let node_72 = node_51.wrapping_sub(node_53);
    let node_177 = node_161.wrapping_add(node_165);
    let node_179 = node_163.wrapping_add(node_167);
    let node_178 = node_162.wrapping_add(node_166);
    let node_176 = node_160.wrapping_add(node_164);
    let node_69 = node_64.wrapping_add(node_67);
    let node_397 = node_71
        .wrapping_mul(18446744073709525744)
        .wrapping_sub(node_72.wrapping_mul(53918));
    let node_1857 = node_90.wrapping_mul(395512);
    let node_99 = node_91.wrapping_add(node_93);
    let node_1865 = node_91.wrapping_mul(18446744073709254400);
    let node_1869 = node_93.wrapping_mul(179380);
    let node_1873 = node_92.wrapping_mul(18446744073709509368);
    let node_1879 = node_160.wrapping_mul(35608);
    let node_185 = node_161.wrapping_add(node_163);
    let node_1915 = node_161.wrapping_mul(18446744073709340312);
    let node_1921 = node_163.wrapping_mul(18446744073709494992);
    let node_1927 = node_162.wrapping_mul(18446744073709450808);
    let node_228 = node_165.wrapping_add(node_167);
    let node_1939 = node_165.wrapping_mul(18446744073709420056);
    let node_1945 = node_167.wrapping_mul(18446744073709505128);
    let node_1951 = node_166.wrapping_mul(216536);
    let node_1957 = node_164.wrapping_mul(18446744073709515080);
    let node_70 = node_64.wrapping_sub(node_67);
    let node_702 = node_71
        .wrapping_mul(53918)
        .wrapping_add(node_72.wrapping_mul(18446744073709525744));
    let node_1961 = node_90.wrapping_mul(18446744073709254400);
    let node_1963 = node_91.wrapping_mul(395512);
    let node_1965 = node_92.wrapping_mul(179380);
    let node_1967 = node_93.wrapping_mul(18446744073709509368);
    let node_1970 = node_160.wrapping_mul(18446744073709340312);
    let node_1973 = node_161.wrapping_mul(35608);
    let node_1982 = node_162.wrapping_mul(18446744073709494992);
    let node_1985 = node_163.wrapping_mul(18446744073709450808);
    let node_1988 = node_166.wrapping_mul(18446744073709505128);
    let node_1991 = node_167.wrapping_mul(216536);
    let node_1994 = node_164.wrapping_mul(18446744073709420056);
    let node_1997 = node_165.wrapping_mul(18446744073709515080);
    let node_98 = node_90.wrapping_add(node_92);
    let node_184 = node_160.wrapping_add(node_162);
    let node_227 = node_164.wrapping_add(node_166);
    let node_86 = node_69.wrapping_add(node_397);
    let node_403 = node_1857.wrapping_sub(
        node_99
            .wrapping_mul(18446744073709433780)
            .wrapping_sub(node_1865)
            .wrapping_sub(node_1869)
            .wrapping_add(node_1873),
    );
    let node_271 = node_177.wrapping_add(node_179);
    let node_1891 = node_177.wrapping_mul(18446744073709208752);
    let node_1897 = node_179.wrapping_mul(18446744073709448504);
    let node_1903 = node_178.wrapping_mul(115728);
    let node_1909 = node_185.wrapping_mul(18446744073709283688);
    let node_1933 = node_228.wrapping_mul(18446744073709373568);
    let node_88 = node_70.wrapping_add(node_702);
    let node_708 = node_1961
        .wrapping_add(node_1963)
        .wrapping_sub(node_1965.wrapping_add(node_1967));
    let node_1976 = node_178.wrapping_mul(18446744073709448504);
    let node_1979 = node_179.wrapping_mul(115728);
    let node_87 = node_69.wrapping_sub(node_397);
    let node_897 = node_1865
        .wrapping_add(node_98.wrapping_mul(353264))
        .wrapping_sub(node_1857)
        .wrapping_sub(node_1873)
        .wrapping_sub(node_1869);
    let node_2007 = node_184.wrapping_mul(18446744073709486416);
    let node_2013 = node_227.wrapping_mul(180000);
    let node_89 = node_70.wrapping_sub(node_702);
    let node_1077 = node_98
        .wrapping_mul(18446744073709433780)
        .wrapping_add(node_99.wrapping_mul(353264))
        .wrapping_sub(node_1961.wrapping_add(node_1963))
        .wrapping_sub(node_1965.wrapping_add(node_1967));
    let node_2020 = node_184.wrapping_mul(18446744073709283688);
    let node_2023 = node_185.wrapping_mul(18446744073709486416);
    let node_2026 = node_227.wrapping_mul(18446744073709373568);
    let node_2029 = node_228.wrapping_mul(180000);
    let node_2035 = node_176.wrapping_mul(18446744073709550688);
    let node_2038 = node_176.wrapping_mul(18446744073709208752);
    let node_2041 = node_177.wrapping_mul(18446744073709550688);
    let node_270 = node_176.wrapping_add(node_178);
    let node_152 = node_86.wrapping_add(node_403);
    let node_412 = node_1879.wrapping_sub(
        node_271
            .wrapping_mul(18446744073709105640)
            .wrapping_sub(node_1891)
            .wrapping_sub(node_1897)
            .wrapping_add(node_1903)
            .wrapping_sub(
                node_1909
                    .wrapping_sub(node_1915)
                    .wrapping_sub(node_1921)
                    .wrapping_add(node_1927),
            )
            .wrapping_sub(
                node_1933
                    .wrapping_sub(node_1939)
                    .wrapping_sub(node_1945)
                    .wrapping_add(node_1951),
            )
            .wrapping_add(node_1957),
    );
    let node_154 = node_88.wrapping_add(node_708);
    let node_717 = node_1970.wrapping_add(node_1973).wrapping_sub(
        node_1976
            .wrapping_add(node_1979)
            .wrapping_sub(node_1982.wrapping_add(node_1985))
            .wrapping_sub(node_1988.wrapping_add(node_1991))
            .wrapping_add(node_1994.wrapping_add(node_1997)),
    );
    let node_156 = node_87.wrapping_add(node_897);
    let node_906 = node_1915
        .wrapping_add(node_2007)
        .wrapping_sub(node_1879)
        .wrapping_sub(node_1927)
        .wrapping_sub(
            node_1897
                .wrapping_sub(node_1921)
                .wrapping_sub(node_1945)
                .wrapping_add(
                    node_1939
                        .wrapping_add(node_2013)
                        .wrapping_sub(node_1957)
                        .wrapping_sub(node_1951),
                ),
        );
    let node_158 = node_89.wrapping_add(node_1077);
    let node_1086 = node_2020
        .wrapping_add(node_2023)
        .wrapping_sub(node_1970.wrapping_add(node_1973))
        .wrapping_sub(node_1982.wrapping_add(node_1985))
        .wrapping_sub(
            node_2026
                .wrapping_add(node_2029)
                .wrapping_sub(node_1994.wrapping_add(node_1997))
                .wrapping_sub(node_1988.wrapping_add(node_1991)),
        );
    let node_153 = node_86.wrapping_sub(node_403);
    let node_1237 = node_1909
        .wrapping_sub(node_1915)
        .wrapping_sub(node_1921)
        .wrapping_add(node_1927)
        .wrapping_add(node_2035)
        .wrapping_sub(node_1879)
        .wrapping_sub(node_1957)
        .wrapping_sub(
            node_1933
                .wrapping_sub(node_1939)
                .wrapping_sub(node_1945)
                .wrapping_add(node_1951),
        );
    let node_155 = node_88.wrapping_sub(node_708);
    let node_1375 = node_1982
        .wrapping_add(node_1985)
        .wrapping_add(node_2038.wrapping_add(node_2041))
        .wrapping_sub(node_1970.wrapping_add(node_1973))
        .wrapping_sub(node_1994.wrapping_add(node_1997))
        .wrapping_sub(node_1988.wrapping_add(node_1991));
    let node_157 = node_87.wrapping_sub(node_897);
    let node_1492 = node_1921
        .wrapping_add(
            node_1891
                .wrapping_add(node_270.wrapping_mul(114800))
                .wrapping_sub(node_2035)
                .wrapping_sub(node_1903),
        )
        .wrapping_sub(
            node_1915
                .wrapping_add(node_2007)
                .wrapping_sub(node_1879)
                .wrapping_sub(node_1927),
        )
        .wrapping_sub(
            node_1939
                .wrapping_add(node_2013)
                .wrapping_sub(node_1957)
                .wrapping_sub(node_1951),
        )
        .wrapping_sub(node_1945);
    let node_159 = node_89.wrapping_sub(node_1077);
    let node_1657 = node_270
        .wrapping_mul(18446744073709105640)
        .wrapping_add(node_271.wrapping_mul(114800))
        .wrapping_sub(node_2038.wrapping_add(node_2041))
        .wrapping_sub(node_1976.wrapping_add(node_1979))
        .wrapping_sub(
            node_2020
                .wrapping_add(node_2023)
                .wrapping_sub(node_1970.wrapping_add(node_1973))
                .wrapping_sub(node_1982.wrapping_add(node_1985)),
        )
        .wrapping_sub(
            node_2026
                .wrapping_add(node_2029)
                .wrapping_sub(node_1994.wrapping_add(node_1997))
                .wrapping_sub(node_1988.wrapping_add(node_1991)),
        );

    [
        node_152.wrapping_add(node_412),
        node_154.wrapping_add(node_717),
        node_156.wrapping_add(node_906),
        node_158.wrapping_add(node_1086),
        node_153.wrapping_add(node_1237),
        node_155.wrapping_add(node_1375),
        node_157.wrapping_add(node_1492),
        node_159.wrapping_add(node_1657),
        node_152.wrapping_sub(node_412),
        node_154.wrapping_sub(node_717),
        node_156.wrapping_sub(node_906),
        node_158.wrapping_sub(node_1086),
        node_153.wrapping_sub(node_1237),
        node_155.wrapping_sub(node_1375),
        node_157.wrapping_sub(node_1492),
        node_159.wrapping_sub(node_1657),
    ]
}
