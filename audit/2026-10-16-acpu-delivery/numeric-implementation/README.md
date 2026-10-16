# Scaled coefficient implementation

Candidate on `fix/acpu-scaled-chebyshev`, parent
`475dd6d22c4d179aeb0d3a480303d3af4ff6a42c`; Strata remains
`56aedb2d12b3126c601eb333419136d403614dbb`. Final compiler closure
`bf7125964da4ffae87f6f419de482c4295e0975af7f5764018c5a1e11608371c`
contains 255 Honeycrisp and 243 Strata inputs in [sources.json](final/sources.json).
No dependency, version or manifest changes belong to this unit.

The [accepted plan](../../../.claude/plans/chebyshev-scaling.md) became one
private scaled-ratio engine and a checked public entry point. It retains the
prototype recurrence, two terminal seeds, normalization and tail/convergence
policy; production removes audit statistics. Defaults bound output and work,
fallible reservation reports allocation/layout refusal, and the legacy wrapper
panics before matvec output/scratch writes on numeric refusal. The independently
reviewed [prototype evidence](../numeric-prototype/README.md) remains separate.

`verify.nu` confirms the entire unscaled `bessel_i` and matvec function bodies
are byte-identical to the parent. Public unscaled Bessel limitations remain.
The [canonical contract](../../../acpu/specs/README.md) distinguishes estimated
convergence from certified error and separates coefficient work from matvec.

## Executed gates

Every command below used Rust/Cargo/Clippy 1.95.0, `RUSTUP_TOOLCHAIN=1.95.0`,
explicit matching `RUSTC`/`RUSTDOC`, two build jobs and `-Dwarnings` for Rust and
rustdoc. [run-gates.nu](final/run-gates.nu) ran them sequentially on the closure
above; each JSON receipt retains exact arguments, environment, timestamps,
before/after identity and raw stream digests. All eight exited zero.

| Receipt | Command | Observed result |
| --- | --- | --- |
| [fmt](final/fmt.json) | `cargo fmt --all -- --check` | clean |
| [workspace lint](final/clippy-workspace.json) | `cargo clippy --workspace --locked --offline -- -D warnings` | clean |
| [acpu lint](final/clippy-acpu.json) | `cargo clippy -p acpu --lib --tests --locked --offline -- -D warnings` | clean |
| [workspace tests](final/tests-workspace.json) | `cargo test --workspace --locked --offline` | 365 passed, 5 ignored |
| [release build](final/build-release.json) | `cargo build --release --workspace --locked --offline` | built |
| [release tests](final/tests-acpu-release.json) | `cargo test -p acpu --lib --tests --release --locked --offline` | 196 passed, 1 ignored |
| [AMX](final/matmul.json) | `cargo run -p acpu --example matmul --locked --offline` | 64×64 control, max error zero |
| [docs](final/docs-acpu.json) | `cargo doc -p acpu --no-deps --locked --offline` | clean |

Seven authored coefficient tests cover 149 frozen independent references,
128 seeded positive-series comparisons, 29 resolved wide-time integral controls,
tau 100/104 overflow regressions, tiny/subnormal values, exact work/refinement
limits, maximum/default/custom counts and layout refusal after work preflight.
They also retain analytic heat controls and refusal-before-mutation tests.
The separate root-authored two-test suite checks 24 weighted four-cycle heat
actions through tau 10,000 and six bit-preserving refusal cases; its truncation
tail and f32 accumulation allowances are separate. Both suites passed debug
and release. Numerical target `2^-21 |reference| + 2^-149` is tested evidence,
not a global error guarantee. Unresolved oracle values remain excluded and named.
No performance claim comes from the busy-host AMX timing.

## Historical red and verification

[First strict Clippy](first/clippy-acpu.json) exited 101 on test-only
`manual_is_multiple_of` at source closure
`27b20bec86bdce1451b5f58dc19475051d93b361a38b581af015dd75b52ef0e3`.
The exact original test body and raw streams remain in `first/`. Only that test
file changed between epochs: equivalent parity syntax plus requested zero/nonzero
layout-versus-work priority controls. Production and independent tests stayed fixed.

Run `nu audit/2026-10-16-acpu-delivery/numeric-implementation/verify.nu REPO`.
It checks the complete current source inventory/bytes, revision ancestry, both
manifest digests, all nine actual gate/stream records, exact command arguments,
strict compiler/version flags, the one-file historical bridge, unchanged kernels
and byte-exact derivation of the 149-row TSV from the retained oracle.
[verification.json](verification.json) records the result. The final independent
[CODE review](../numeric-review.md#final-production-code-verdict) approves this
exact source and distinguishes inspected evidence from rerun tests.

This is source repair evidence. Package release, version bump and crates.io
publication remain separate owner-controlled work.
