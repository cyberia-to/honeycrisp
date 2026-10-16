# Public sparse/field admission

Base `70dfb1f55e4cb2570d5a78a497be5828a203087b`, Strata
`56aedb2d12b3126c601eb333419136d403614dbb`; input closure
`47bffad98aa5a820aab3bf02d2d19d375bcc0ba76e07d24268b35d39b9426250`.
[Inventory](sources.json) includes new untracked compiler files. Parent audit
retains the unchanged historical gate/probe evidence.

The private Csr view admits the complete immutable partition and all column
indices. Every application checks vector lengths before output clearing,
prefetch or unchecked access. Chebyshev validates every dimension before
coefficient allocation and reuses one admitted view. Field entry points enforce
nonempty powers of two and exact multilinear coordinate counts before private
kernels or recursion. Sparse arithmetic, floating-point operand order, Bessel
scaling and field arithmetic are unchanged. Root inspected the complete guard
before formerly unsafe malformed-input tests were executed.

`cargo test -p acpu --lib --tests --locked --offline` passes 187 tests with one
existing ignored, in both [debug](debug.json) and [release](release.json); the
latter command adds `--release`. The new public integration target contains
five tests: 30 malformed CSR calls through both operations, 51 Chebyshev
shape/partition/index calls, field shape/coordinate cases and independent valid
controls for rectangular sparse arithmetic and two-element heat evolution.
All malformed calls require the appropriate panic and unchanged outputs/scratch.
The huge-order malformed Chebyshev cases establish shape-error precedence.

[Strict library/test Clippy](clippy.json) remains red on historical production
and test hygiene; full diagnostic text is retained. Independent review and the
separate mechanical hygiene unit follow. This unit does not resolve the measured
Chebyshev numerical overflow and does not establish publication readiness.
