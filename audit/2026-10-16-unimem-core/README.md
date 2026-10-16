# Unimem core prerequisite

The no-default Unimem dependency now compiles inside the frozen independent
`#![no_std]` binary with its own panic handler. Default builds retain queue std
backoff. The change adds feature forwarding and core imports; it preserves
Cargo.lock, dependency versions, FFI/native contracts and existing test bodies.
This establishes a dependency prerequisite, not Kadek Image completion or release
readiness. Apple framework, allocator, ownership and physical-address requirements
remain as before.

Source parent: Honeycrisp `9cbd48070a136e24711896c930670fcff8a147fc`.
`parent.json`, `candidate.json` and `candidate-v2.json` bind complete source
identities before/after every recorded command. Current native-test input is clean
committed Strata `56aedb2d12b3126c601eb333419136d403614dbb`; the preserved
[historical packet](../2026-10-13-unimem-core/plan-v1/README.md) uses Strata
`494df7a0973ae10a6cc1c2022e644d70449a29d6` and remains a separate epoch.
Both current parent and candidate use the same sibling and Homebrew Rust 1.95.0
on aarch64-apple-darwin (`tools.json`); setting RUSTUP_TOOLCHAIN did not select a
Rustup compiler. Commands use the existing verified offline registry cache.

Each `records/<name>.json` gives the literal argv, environment, revision, times,
exit code and hashes of its complete stdout/stderr. Counts below come from those
streams, not inferred coverage. Commands below abbreviate the common
`--locked --offline` arguments; records retain them.

| Record / command | Current result |
| --- | --- |
| `parent-no-std` / `nu unimem/scripts/check-no-std.nu` | 101: E0152 duplicate `panic_impl` |
| `candidate-no-std`, `candidate-no-std-v2` / same command | 0: original and guarded gate pass |
| `parent-owner`, `candidate-owner-default` / `cargo test -p unimem` | 0: 53 tests + 13 doctests each |
| `candidate-owner-no-default` / same with `--no-default-features` | 0: 53 tests + 13 doctests |
| `parent-isolated`, `candidate-isolated` / separate native consumer | 0: 44 unchanged native tests each |
| `candidate-clippy-core`, `candidate-clippy-default` / `cargo clippy -p unimem --lib … -- -D warnings` | 0: both modes |
| `candidate-docs-core`, `candidate-docs-default` / `cargo doc -p unimem --no-deps …` | 0: both modes, RUSTDOCFLAGS=-Dwarnings |
| `candidate-owner-fmt` / `cargo fmt -p unimem --check` | 0 |
| `candidate-examples` / `cargo check -p unimem --examples --benches` | 0; compilation only |
| `candidate-drift` / `check-drift.nu` | 0: checksum drift rejected before Cargo |
| `candidate-temp-failure-v2` / `check-temp-failure.nu` | 0: failed mktemp with existing directory and successful-empty stdout rejected without writes/removal |
| `parent-fmt`, `candidate-workspace-fmt` / workspace fmt | 1: same inherited formatting output, first aruminium/benches/objc2.rs:300 |
| `parent-workspace-clippy`, `candidate-workspace-clippy` / strict workspace Clippy | 101: same four ACPU errors; first `error: duplicated attribute` at acpu/src/gemm/mod.rs:1094 |
| `candidate-workspace-tests` / strict workspace tests | 101: first `error: unused import: std::time::Instant` at acpu/bench/summary.rs:8; also unchanged bench FFI declaration errors and Cargo example-name collision warnings |

The isolated native manifest, lock and copied test identities are recorded in
`isolated-native.json`. It has `default-features = false` and no reverse dev
dependencies. Owning no-default tests can unify std through dev dependencies and
do not prove the dependency boundary alone. Four feature-tree views are retained;
the reviewer also inspected locked queue/utils feature behavior directly.
Historical-tag API comparison remains blocked by the tag's obsolete NeBu path.
Separate rustdoc JSON API comparison, current full release workspace build,
native benchmark execution and broader product/dependent/Kelvin gates were not
completed in this source unit; older measurements do not replace current gates.
Public signature preservation was reviewed from the complete diff.

Two preparation failures are retained: `candidate-features-default` passed a
comma-separated option as separate Nu tokens; `candidate-features-default-v2`
corrects only quoting. `candidate-temp-failure` used Nu's reserved `$nu` name;
the retained first control body is in `harness-bridge/`, and `-v2` renames the
local binding. Neither failure changed production or the frozen consumer.

Root requested a narrow `mktemp` guard after source review. `candidate-v2` differs
from `candidate` only in `unimem/scripts/check-no-std.nu`: validate exit status
and nonempty existing directory before writing or cleanup. The original gate is
retained in `harness-bridge/`; all three fixtures retain `frozen-witness.json`
identities. Only the durable gate and its failure controls were rerun afterward.
`finalization.json` records this bridge, raw-record verification and current-source
verification without replacing any earlier epoch.

[Independent Codex CODE review](code-review.md) approves the source delta;
[guard supplement](guard-review.md) covers the only subsequent gate change.
Neither is a different-vendor verdict or an all-green release verdict. The
planning-era “production unchanged” and vendor-review language remains frozen in
the old packet; [execution status](../../.claude/tasks/unimem-core/execution.md)
states the current authorization and lineage. Broader red gates require a `wip:`
commit with the first error; no version, tag or publication is authorized here.
