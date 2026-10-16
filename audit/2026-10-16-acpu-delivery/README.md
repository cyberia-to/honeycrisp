# acpu delivery prerequisite gates — 2026-10-16

The [admission repair](admission/README.md) is committed as `c5177c5`; the
[final hygiene/tooling gates](delivery-v4/README.md) pass on their recorded source.
The [bounded scaled coefficient repair](numeric-implementation/README.md) now
passes its eight source-bound gates. The version proposal and owner publication
remain separate from these source repairs.

The sections below preserve the original soundness and gate findings before
repair. Baseline:
`70dfb1f55e4cb2570d5a78a497be5828a203087b`; Strata:
`56aedb2d12b3126c601eb333419136d403614dbb`; frozen input SHA256:
`9659c09eb4071c55d11e71e3e96f09b24ef74b42e8ace73006facc5be9293a34`.
[Source inventory](sources.json), [proposed first repair](../../.claude/plans/acpu-delivery.md).

## Required whole-upstream gates

Root `CLAUDE.md:39–45` and `acpu/CLAUDE.md:34–40` name these checks.
[check.nu](check.nu) takes the repository argument and captures actual argv,
pinned rustup 1.95.0, warnings denied, external target, stream hashes and unchanged
before/after inputs. All raw output is retained beside its receipt.
The saved `verification.json` records the original seven receipt/stream pairs,
probe inputs and unchanged baseline compiler files. Its original `verify.nu`
intentionally compares that historical source. `verify-delivery.nu` checks the
historical hygiene epoch; `numeric-implementation/verify.nu` checks the current
numerical epoch. Preserved red outcomes stay red.

| Receipt | Result on the stated source |
|---|---|
| [fmt](fmt.json) | `cargo fmt --all -- --check`: red; only `aruminium/benches/objc2.rs`, repeated by multiple target entries. |
| [Clippy](clippy.json) | `cargo clippy --workspace --locked --offline -- -D warnings`: red; duplicate allow attribute at `acpu/src/gemm/mod.rs:1094`, assign-op and coefficient-loop lints in `sparse/chebyshev.rs:171,192`, prefetch-loop lint in `sparse/mod.rs:61`. Later packages may expose further lints. |
| [Release build](release.json) | `cargo build --release --workspace --locked --offline`: pass. |
| [Workspace tests](tests.json) | `cargo test --workspace --locked --offline`: compilation stops on unused `b_stride` at `acpu/examples/trace64.rs:289`. No whole-workspace test pass is claimed. |
| [AMX example](matmul.json) | `cargo run -p acpu --example matmul --locked --offline`: pass, 64×64 identity control, maximum error zero. `gemm/mod.rs:182–205` routes this shape to the direct AMX path. The captured timing is not performance evidence. |

## Release-blocking soundness

`acpu/src/sparse/mod.rs:16,24` exports safe functions but validates output length
only with debug assertions and never validates columns. The aarch64 path then
uses `x.as_ptr().add(...)` at lines 62/77, unchecked x reads at 88–100 and an
unchecked y write at 104. For example, the safe expression
`csr_matvec(&[0,1], &[1], &[1.0], &[0.0], &mut [0.0])` reaches an out-of-bounds
unchecked read. This is a source-level witness; it was deliberately not executed.
Short y similarly permits unchecked release-mode writes. Rust requires these
indices to be valid even for safe API callers.
[Rust slice safety](https://doc.rust-lang.org/std/primitive.slice.html#method.get_unchecked),
[pointer arithmetic safety](https://doc.rust-lang.org/std/primitive.pointer.html#method.add).

`chebyshev_matvec` is the third public entry into this unsafe CSR path. Its
debug-only scratch checks and unchecked relation between row count and x length
are insufficient. The proposed unit validates the complete CSR partition,
column bounds and all wrapper dimensions before any output mutation or unsafe
operation. It preserves the numerical kernels and public signatures.

## Other newly exported API findings

All new public field/sparse functions were inspected. `pow_gl` accepts its u64
inputs without a shape boundary; ntt_forward already has a release length assert.
Bit reversal and multilinear evaluation depend on debug-only length assertions;
multilinear evaluation also ignores excess coordinates. Branch
`launch/131-field-power-of-two-release`, commit `904dd336…`, changes the length
assertions but leaves point-count admission and CSR safety unresolved.

The isolated [safe probe](safe-input-probe.rs) never calls invalid CSR or empty
recursive polynomial evaluation. Its manifest/lock are retained as text, and
[debug](input-debug.json)/[release](input-release.json) receipts additionally hash
the probe inputs. These successful probe executions report existing bad behavior;
they are not successful correctness gates:

| Input | Debug | Release |
|---|---|---|
| NTT length 3 | Panics | Panics |
| Bit reversal length 3 | Panics | Returns input unchanged |
| Multilinear 3 evaluations / 2 coordinates | Panics | Returns 1 |
| Multilinear 4 evaluations / 1 coordinate | Panics | Panics later by indexing |
| Multilinear 4 evaluations / 3 coordinates | Returns 1 | Returns 1 |
| Chebyshev tau=100, order=20 | First coefficient infinity; none of 21 finite | Same |
| Chebyshev tau=104, order=20 | First coefficient NaN; none of 21 finite | Same |

`bessel_i` casts the unscaled result to f32 at `sparse/chebyshev.rs:93`, and
`chebyshev_coeffs` multiplies it by exp(-tau) at 113. For positive tau, the true
scaled zeroth coefficient is the average of exp(tau*(cos(theta)-1)), hence lies
in (0,1]; this finite bound follows from the primary integral representation.
The observed infinities/NaNs are avoidable intermediate overflow, not true
coefficient overflow. The existing fixed recurrence-depth policy also needs a
defined accuracy domain before claiming a general repair.
[NIST DLMF 10.32.1](https://dlmf.nist.gov/10.32.E1).

The sparse API tests currently cover a short path, accumulate/set, one long row,
small Bessel values and a small-tau smoothing check. They contain no malformed
CSR, large-tau coefficient or independent general matrix-exponential oracle.
The advertised caller-owned-buffer policy also differs from Chebyshev's internal
allocations; that architecture statement needs reconciliation, not concealed
allocation claims. These findings are distinct from Nox's already measured
valid-input NTT/multilinear package contract.

The original next step was numerical-domain/scaling repair and its owning gates;
follow the current result linked above. The acpu-only version proposal remains
separate. No version editing, tag or publication occurred in these repair units.
