# acpu 0.2.1 — owner release proposal

Accepted by root on 2026-10-17; re-anchored after ordinary PR25 merged at
`14373cc8a0e18302751844579d94d1f8f80bd8b1`. This exact origin/main revision
is the source base. The preparation branch preserves pushed `9e6cffe699278059c8940f13940727b10569e13e`
and merges the reviewed source without rewriting history. Historical tooling,
approval and receipts remain in their original epochs; `review-epoch-2`
also preserves the pre-re-anchor active files. Root must approve the new
tool closure before export use, bump edits or frozen bump executions.

The existing PR25 Tape gate inventory equals the complete compiler inputs at
this base. Its eleven gates, raw outcomes and portable replay bridge this
tooling-only unit; new script bodies receive separate fresh validation.
Deliverable: a reviewed, source-only `chore: acpu 0.2.1` PR for the owner.
Owner alone merges the bump, tags or publishes. Branch-built archives are
prepublication diagnostics; release delivery requires fresh origin/main inputs
after the owner's merge.

## Exact source scope

1. `acpu/Cargo.toml`: replace `version.workspace = true` with `version = "0.2.1"`.
2. Root `Cargo.lock`: let Cargo regenerate the local acpu version entry offline;
   inspect the entire diff and refuse unrelated dependency/version changes.
3. New `acpu/CHANGELOG.md`: concise 0.2.1 entry derived from the actual published
   0.2.0-to-current source/API delta, linking reviewed admission, numerical and
   package repairs; distinguish added APIs from fixed invalid-input behavior.

Workspace/sibling package versions remain 0.2.0. The existing workspace acpu
requirement `"0.2.0"` already accepts 0.2.1; no owning Nox or dependency pin edits
belong to this PR. Existing 0.2.0 audits stay immutable. Any other required
source correction is a separately reviewed ordinary repair before the bump.
The owning workspace includes the reviewed Tape repair, but the normalized
acpu package still reexports `Tape` from registry unimem0.2.0. This release
does not deliver that dependency's owning-source Tape repair. Do not claim
otherwise in the changelog or package evidence, or expand this into a unimem bump.

## One checkout, two branches

Use the existing `honeycrisp/.worktrees/acpu-delivery/honeycrisp`; preserve its
Strata link/target and revision `56aedb2d12b3126c601eb333419136d403614dbb`.
Keep build/probe outputs under `~/.cache/cyber/honeycrisp/acpu-0.2.1/`.

1. After plan approval, preparation branch owns this plan and new version-aware
   tooling under `audit/2026-10-17-acpu-release-proposal/`. Commit reviewed
   preparation with the exact-source PR25 gate bridge and fresh tooling checks;
   retain wrapper context there. Preserve the old preparation commit and epochs.
2. From a clean checkout, record the preparation commit and export its exact
   runner/support bytes to the external cache. Switch to a new bump branch
   `chore/acpu-0.2.1` from the reviewed origin/main base. No new checkout or
   history rewrite. Update wrapper branch/base context, preserving dependencies.
3. Make only the three source edits. Use an offline Cargo operation to regenerate
   the root lock; inspect exact source/target/dependency deltas. Run all eight
   owning gates below, independent review, then commit the source-only bump.
   No final package can be built before this committed source identity exists.
4. Run immutable preparation tools against that exact clean bump commit;
   receipts and archives go to the external cache while this branch is checked
   out. Confirm the bump diff contains exactly the three permitted paths.
5. After inspection of tracked/untracked/ignored files, switch the same clean
   checkout back to the preparation branch. Retain exact candidate commit,
   archive, normalized inputs, raw receipts and portable verifier in its audit;
   commit/push that evidence separately. Push the bump branch and open its
   source-only PR with links to the evidence branch's immutable commit.

The evidence verifier must bind candidate bytes to its explicit Git revision
and retained archive, rather than treating the preparation branch's unbumped
working tree as the candidate. Both refs remain retained through owner delivery.
Any failed command remains red with its exact source epoch; no gate weakening.

## Required fresh checks

Pin Rust1.95.0, `RUSTFLAGS=-Dwarnings`, `RUSTDOCFLAGS=-Dwarnings`, external
per-task targets, complete before/after owner and Strata source identities.
The eight owning gates are: fmt; strict workspace Clippy; strict acpu all-target
Clippy; workspace tests; workspace release build; acpu release tests; real
matmul; strict acpu docs. Use the exact commands from the merged package audit,
with `--locked --offline` where applicable. Matmul is a correctness smoke.

Before `cargo package -p acpu --locked --offline`, require complete owning
compiler/config/asset inventory equal the committed bump. Verify the untouched
normalized acpu0.2.1 archive's VCS, source inventory, manifest, lock and checksum;
assert the package version and all 40 targets (32 examples, six integration
tests, one library and one binary), freshly derived with locked offline
`cargo metadata --no-deps --format-version 1` at the reviewed base. The two
new GEMV targets are `gemv_admission` and `gemv_independent`; retain independent
test SHA `d595e4799a602ba6b2f46987bac6c22d0cb60d7cf8b841900a388488272568ba`.
Run its strict all-target
Clippy, debug/release library+integration tests, strict docs and real matmul.
No source/manifest overlay, omitted examples or dependency substitution there.

The current source epoch's raw PR25 Tape receipts contain 388 passing and five
ignored workspace tests across 31 summaries; acpu release has 212 passing and
one ignored across eight summaries. Re-derive these totals in the new tooling
validation and demand them from the fresh owning/package gates. Historical
PR21 totals (365/5 workspace, 196/1 acpu) remain historical controls only.
Package debug/release totals are expectations until the untouched candidate
actually runs; no acpu0.2.1 gate result exists at preparation time.

Fetch the official acpu0.2.0 archive/index again; preserve fixed checksum/VCS
and refuse a newly occupied/yanked 0.2.1 rather than selecting another version.
Extract separate full-library rustdoc artifacts for published0.2.0 and candidate
0.2.1 on aarch64-apple-darwin; no crate features exist. Run explicit patch-level
API comparison with the same pinned tool/flags and retain warnings/raw outcomes.
Preserve distinct source/target paths and exact artifact-to-command links.

Replay the unchanged four-test Nox probe in debug/release against normalized
Nox `64d091e5ad3f277cce77b88ae0de6eea4c49b4df` (report any source drift before
selecting another revision), using only an explicit scratch `[patch.crates-io]` to untouched
candidate acpu0.2.1. Record manifest/lock and prove one registry unimem0.2.0 and
strata-nebu0.1.1 identity, NeBu fast off, shared direct/Nox Goldilocks type source.
This diagnostic does not authorize an owning Nox patch or minimum update.

Root independently reviews source/tooling and replays portable evidence before
handoff. After owner publication, a separate Nox change can set `acpu = "0.2.1"`
and run actual unpatched registry acceptance. Post-merge release rebuilding,
tagging and publication remain explicitly separate from this preparation PR.
