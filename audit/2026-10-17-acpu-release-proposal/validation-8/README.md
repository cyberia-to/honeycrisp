# Canonical extraction after creation

Root's independent replay of candidate
`2c70b3b929c791ca489bba12d2f5a6daef7d43e7` with validator
`e791fefaf05c612d10d96f201e9d9460b20d502f` refused its freshly extracted
metadata: the requested `/tmp/` root did not exist when `path expand` ran,
but Cargo returned the corresponding existing `/private/tmp/` paths.
`root-failure` preserves that exact metadata receipt and both raw streams.

The only runner change creates the fresh extraction directory before resolving
its canonical path. No historical receipt path is rewritten. `alias-check.nu`
uses the same candidate archive, creates a real `/tmp/` alias, resolves it after
creation, and runs `cargo metadata --no-deps --format-version 1 --locked
--offline`. The corrected source-path check accepts all 40 targets; no compiler,
consumer or API gate runs. Its exact invocation and output are `alias-control.*`.

The final verifier will use a new committed validator bundle, separately from
the unchanged e791 consumer and 3b875c9 owning/package/API producer bundles.
The previous capsule and failed root replay remain immutable. Full `/tmp/`
capsule replay and root independent replay are required before the owner PR.
