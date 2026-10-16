# Mechanical gate repairs

Admission source remains frozen for independent review. The initial workspace
reds and additional library/test Clippy red are retained before these edits.
No versions, dependencies, numeric tolerances, FFI, AMX encodings or numerical
algorithm changes belong to this unit.

Required-gate findings at admission closure `47bffad9…`:

| File and line | Precise repair |
|---|---|
| aruminium/benches/objc2.rs:300,371,418,433 | rustfmt only; initial fmt output records exact changes. |
| acpu/src/gemm/mod.rs:1094 | Remove the duplicate allow attribute; retain the existing single annotation. |
| acpu/src/sparse/mod.rs:56 | Iterate over the same first min(len,8) columns in the same order. |
| acpu/src/sparse/chebyshev.rs:186 | Named scaled temporary preserves `-d * t1` operand order. |
| acpu/src/sparse/chebyshev.rs:207 | Enumerate the same coefficients from index two; same recurrence order. |
| acpu/examples/trace64.rs:289 | Remove unused pure `N * 4` binding; preserve all address expressions. |

Additional strict library/test findings from
`cargo clippy -p acpu --lib --tests --locked --offline -- -D warnings`:

| File and line | Diagnostic / value-preserving proposal |
|---|---|
| acpu/src/field/goldilocks.rs:299,303 | needless_range_loop; iterate mutable array entries, retaining index formula and pow7 order. |
| acpu/src/field/permute.rs:146,154 | needless_range_loop; enumerate mutable fixed arrays with identical wrapping formula. |
| acpu/src/numeric/bf16.rs:167 | approx_constant on intentional decimal 3.14 input; retain its exact f32 bits with an explicit from_bits literal and decimal comment. |
| acpu/src/sparse/chebyshev.rs:262 | excessive_precision; spell the identical f32 value as 1.266_066, with a Rust bit-equality receipt before substitution. |
| acpu/src/sparse/mod.rs:192 | needless_range_loop; enumerate y from index one, retaining diagnostic index. |
| acpu/src/vector/integer.rs:411 | unnecessary_cast; remove the identity i32 cast. |
| acpu/src/vector/media.rs:417 | needless_range_loop; enumerate three output bytes, preserving tolerance. |
| acpu/src/crypto/mod.rs:362 | useless_vec; fixed [u32;44] test scratch. |
| acpu/src/vector/media.rs:386,467 | useless_vec; borrow identical fixed zero/7.5 arrays. |

Newly exposed diagnostics must retain the same accounting: inspect each and
preserve prior red evidence. Checks run against a fresh source manifest; original
receipts and manifests are immutable. Whole-workspace gates and affected acpu
debug/release tests follow the final reviewed source epoch.
