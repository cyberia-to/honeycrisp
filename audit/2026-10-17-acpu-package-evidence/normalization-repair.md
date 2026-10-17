# Proposed NeBu normalization repair

Current owning commit `ff1a91ff3be0b4a7d83f66c6d33e7dd1a563d743`; untouched
normalized acpu archive SHA256
`b99d16febc7ede51d4af9a494477edf9692166e201ab2ec422b03d8a02772a6a`.
[Named example check](checks-1/named-examples.json) actually exits101 with
E0433 in both `bench/zk.rs` and `bench/summary.rs`. The normalized manifest has
no NeBu dev-dependency; Cargo stripped the path-only declaration.

Smallest proposed production diff, root Cargo.toml only:

```diff
-nebu = { path = "../strata/nebu/rs", package = "strata-nebu" }
+nebu = { path = "../strata/nebu/rs", package = "strata-nebu", version = "0.1.1" }
```

This existing published minimum keeps alias, path, selected local Strata
revision, empty defaults and fast-off policy. The local source is already0.1.1;
no local lock graph change is expected. Inspect generated lock before accepting
any difference. After normalization, the named examples must use one registry
strata-nebu0.1.1 with checksum
`f4a881cdb3622cd6b38333246f516c70e3e5c63d2bc0c0e48344d8e7b0392766`.
Registry tier dependencies are recorded separately from source Strata tiers.

No package version bump is part of this ordinary dependency-declaration repair.
Re-run owning strict gates and package normalization, then compile both named
examples and retain target conservation. Root reviews/authorizes this exact diff
before implementation. Current package/all-target failures stay intact.

Separate first strict all-target failure is excessive_precision at
`bench/elementwise.rs:103`, retained in [Clippy receipt](checks-1/clippy-all-targets.json).
The keep-going strict pass will inventory all example lint debt before a distinct
value-preserving hygiene proposal; nothing is suppressed or excluded.
