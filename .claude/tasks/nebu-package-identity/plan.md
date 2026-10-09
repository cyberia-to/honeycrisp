# nebu package identity — plan

One manifest identity repair restores origin-source Cargo resolution for the
unimem gate prerequisite. Root owns implementation/review/commit/PR; this unit
does not overlap the Block constructor's source/spec files.

1. Record `rustc -Vv`, `cargo -V`, host and target: actual toolchain 1.95.0,
   aarch64-apple-darwin, M4 Max/Mac16,5, macOS26.4.1 build25E253. Strata's
   edition2024/rust-version1.89 requires at least Rust1.89; honeycrisp resolver2
   governs this graph. Toolchain failure is distinct from dependency resolution.
   Change only workspace `nebu.package` from `cyb-nebu` to `strata-nebu`, keeping
   alias, path and unspecified feature policy → verify exact one-line manifest
   diff and both benchmark call sites against the pinned public source.
2. Overlay that manifest into fresh full honeycrisp/strata archives; run
   `cargo metadata --format-version 1 --offline` to refresh the lock using the
   existing lock as resolver input → verify metadata resolves every local
   package under these exact archives; no dirty input or path substitution.
   Copy the generated lock back only after comparing package/source/checksum
   tuples. Expected: cyb-nebu0.1.0 becomes strata-nebu0.1.1; four strata tier
   packages become0.1.1; now-unused registry acpu/unimem duplicates disappear,
   with their path-package dependency disambiguators collapsing. Remaining
   registry tuples must be identical. Strata's nonmember dev-deps do not enter
   resolution. Any unexpected delta stops for explanation. No blanket update.
   If offline resolution itself lacks a cache artifact, retain that diagnostic
   and permit one recorded network metadata run using the existing lock; apply
   the same tuple comparison before accepting its result.
   After copying, read live sibling HEAD/status. Only if clean and exactly
   pinned, run a secondary `cargo metadata --locked --offline` in the owned
   worktree, then recheck sibling status. Otherwise record divergence and skip
   that convenience check. All authoritative gates and the dependent Block
   unit use archived exact sources; live sibling state is never a prerequisite.
3. Run `cargo metadata --format-version 1 --locked --offline`,
   `cargo fmt --all -- --check`,
   `cargo clippy --workspace --all-targets --locked -- -D warnings`,
   `cargo build --release --workspace --locked`, `cargo test --workspace --locked`
   and `cargo build -p acpu --release --example bench_zk --example bench_summary
   --locked` → retain commands, exits and first diagnostics with separate private
   targets. Metadata checks resolution; all-target Clippy/test compilation and
   named example linking exercise the changed dependency. Ordinary release
   build checks production only. Benchmark examples are compiled, not executed;
   record that limit against acpu's examples-run doctrine. Run its bounded
   `cargo run -p acpu --example matmul --locked` gate separately, without claiming
   it exercises nebu. Record actual executed/skipped AMX/ANE/Metal test names.
   For attributable downstream errors, archive a separate control at the same
   pins, delete only acpu's nebu dev-dependency in that temporary control and
   refresh its lock. Run the identical gate set. Missing-nebu errors in its
   benchmark examples are expected and cannot classify candidate nebu errors;
   identical diagnostics in untouched common source can. Neither control edits
   nor generated control lock enter the PR. No untested baseline attribution.
4. Execute semver checks for the public crates against v0.2.0 and source parent
   where the real manifests resolve; a resolution/tool/build failure is a red
   diagnostic, not compatibility proof → verify actual compared roots. Source
   remains byte-identical, a separately stated inspection fact. Run registry/
   reverse-closure checks available from origin; never invent ring readiness.
5. Freeze manifest/lock, write `audit/2026-10-09-nebu-package-identity/` and
   task review receipt, then fresh different-vendor CODE review → verify all
   final bytes and honest red gates before ordinary source PR. A required red
   uses `wip:` with first diagnostic and `decision`; owner session already
   authorizes ordinary source merges, never bumps/releases. No green ring claim.

Failures to check: wrong source selected despite correct name; registry lock
churn unrelated to the repair; benchmark API mismatch; current empty defaults
mistaken for retained acceleration; downstream build errors hidden by an
isolated unimem harness. Actual dependency/test failures are recorded and split
into separate reviewed units; this unit does not repair CPU/GPU/ANE source.

Touched files: Cargo.toml, Cargo.lock, this task's research/plan/review and its
audit receipt/reproducer. No specs/source/API/version/feature toggle changes.
Full upstream gates and rung closure remain distinct from metadata success.

Keep `fast` off deliberately: it enables a registry acpu beside the local acpu,
including another driver implementation. This repair preserves the declared
default-feature policy without adding that path; record selected features from
metadata/tree. Benchmark speed and acceleration remain unmeasured. Record the
root CLAUDE.md standalone-nebu URL as deferred documentation drift; the current
package/source identity is settled here and in the audit. Session authorization
is implementation of phase1/2 and ordinary source merges; package identity is
the only declaration change, with no version requirement or release action.
