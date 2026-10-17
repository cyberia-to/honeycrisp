# Changelog

## 0.2.1

Changes since the published 0.2.0 crate:

- Add sparse CSR accumulation/overwrite matvec, Chebyshev heat-kernel matvec,
  Goldilocks NTT, bit-reversal permutation and multilinear evaluation APIs.
- Add checked, bounded Chebyshev coefficient generation with explicit work and
  output limits; avoid overflow in scaled coefficients. Validate sparse and
  field shapes before mutation in release builds.
  ([coefficient repair](https://github.com/cyberia-to/honeycrisp/commit/ff27cef))
- Validate GEMV lengths and checked dimensions before output writes; preserve
  accumulation/overwrite and zero-dimension behavior. Initialize every AMX
  output tile across repeated calls and use NEON beyond supported tile bounds.
  Keep the pure-assembly K-divisible-by-eight precondition.
  ([GEMV repair](https://github.com/cyberia-to/honeycrisp/pull/25))
- Preserve the NeBu development dependency in Cargo's normalized package and
  restore strict lint checks across all packaged examples.
  ([package repair](https://github.com/cyberia-to/honeycrisp/pull/21))

The normalized crate still uses registry unimem 0.2.0. Its reexported `Tape`
therefore does not include the owning workspace's subsequent Tape repair.
