# acpu 0.2.1 — owner release proposal

Accepted by root on 2026-10-17. Preparation tooling must be reviewed before
branch switching or bump edits; existing source gates apply unchanged to this
plan/tooling-only unit. Owner-only merge/tag/publication remains mandatory.

Proposed, 2026-10-17. Base `3ffc0d5e6f268d28658b8ad279b7adcfa0735fa6`
is committed origin/main; the reused checkout is clean on
`chore/acpu-release-preparation`. This plan is the only new file so far.
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

## One checkout, two branches

Use the existing `honeycrisp/.worktrees/acpu-delivery/honeycrisp`; preserve its
Strata link/target and revision `56aedb2d12b3126c601eb333419136d403614dbb`.
Keep build/probe outputs under `~/.cache/cyber/honeycrisp/acpu-0.2.1/`.

1. After plan approval, preparation branch owns this plan and new version-aware
   tooling under `audit/2026-10-17-acpu-release-proposal/`. Commit reviewed
   preparation with required owning gates; retain wrapper context there.
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
assert the package version and all 38 existing targets. Run its strict all-target
Clippy, debug/release library+integration tests, strict docs and real matmul.
No source/manifest overlay, omitted examples or dependency substitution there.

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
