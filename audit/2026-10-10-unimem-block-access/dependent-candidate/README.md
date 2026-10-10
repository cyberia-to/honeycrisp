# unimem raw-access dependents — candidate diagnostics

2026-10-10. This is the local diagnostic overlay of the frozen Block-access
implementation on honeycrisp parent `7b148906263da5ea8aabc5ddb0e2e44ff2288ecc`.
Source manifest SHA256:
`1ead683ed4e75cfb0c2edd196e6e18e979f3ca83634e69c3bcdd2bfe72e1b1be`.
The native source worktree is `/Users/master/cyber/honeycrisp-kadek-block-access`.
No consumer source was migrated. Compatibility and ring closure are red.

## Reproduction and integrity

`scenarios.json` records every origin revision, archive and committed lock.
`prepare.nu` re-extracted the same 17 baseline scenarios and reused the same
five retained local locks, without a new dependency resolution. All source
matched the parent baseline before the explicit 16-path Honeycrisp overlay;
15 scenarios close over Honeycrisp. The two Nox scenarios have no Honeycrisp
package, so their unchanged parent check/test evidence is reused only as
owning-rung evidence, never candidate driver coverage.

Baseline `/tmp/unimem-dependent-baseline.cX82Mt/SHA256SUMS` is pinned by
`d2f7485eb0e59b06025b3301e53e5d23bc05ea35fa005656325b6718f16db5d2`.
Candidate `overlay.json` pins every source byte. `nu execute.nu` ran the 37
selected commands sequentially with identical baseline argv, locked/offline,
private targets, two jobs, incremental disabled and exact Rust1.95.0 binaries
(Nika uses the same declared nightly2025-11-26 as parent). Tool executable
hashes and version comparisons are in `toolchains.json`.

`logs/<scenario>/<label>.json` records command, environment, time, actual Cargo
exit, source-overlay SHA and unchanged lock hashes. Adjacent stdout/stderr
are complete and SHA-pinned, including warnings. Wrapper success means only
that evidence was written. `nu verify.nu` independently checked all receipts,
the baseline payload manifest, lock equality, all successful metadata local
paths, final 17 source trees and exact changed-path whitelist. `comparison.json`
retains emitted error/warning headers, source locations and test result lines
for each command on each side; it does not treat matching exit codes as proof
of matching diagnostics. `final-source-comparison.json` retains full tree diffs.
The first verifier attempt expected hashes in parent command receipts; those
actually live in the parent payload manifest. It stopped before writing any
verification artifact and was corrected to verify that manifest directly.

`verification.json`: 37 actual commands, 16 exit0 and 21 nonzero. These include
metadata and repeated build configurations, not 37 independent test suites.
No source/manifests/locks outside the explicit Honeycrisp overlay changed.

## Causal comparison

All commands below map to exact argument arrays and revisions in the receipts.
`C` means the selected all-target check, with Metal/GPU features where named.

| Scenario | Parent | Candidate and interpretation |
|---|---|---|
| Xena | Whole-workspace C fails E0608 in `xena-bench/src/gpudebug.rs:192,195`; scoped hash/reference C and 4 tests pass. | All selected C/tests fail compilation: two E0133 sites `xena-hash/src/scratchpad.rs:34,42` call newly unsafe `as_bytes_mut`. The previous E0608 paths and later GPU calls are masked, not fixed. No candidate tests run. |
| Bita local lock | Metal C, bounded hash5 tests (62 passed, 8 benchmarks filtered), explicit GPU oracle pass. | All selected C/tests fail at the changed API: `metal/mining.rs` emits 13 E0133 and 5 E0596 findings in C. Four shared-receiver methods and a non-mutable local cannot request exclusive mutation; raw imports/views now require safety proofs. Candidate GPU oracle never executes. |
| Erga local lock | C and bootstrap release/difftest pass; workspace tests reach 11 passes and two existing chime overflow failures. | C/tests/release/difftest now fail compiling `blake-bench/src/lib.rs`: E0133 at99/107/114, E0596 at107. Earlier runtime failures are masked. No candidate runtime/difftest result exists. |
| Mir local lock | Default/Bevy C fail missing `Gpu::sync` at `src/gpu.rs:32`. | Same emitted E0599; no tests and no claim that masked paths are compatible. |
| Glia local lock | Workspace C fails E0063 missing `has_attn_bias` at `run/tests/tier5_graph_ir.rs:22`; library C/tests pass. | Same workspace error; library C/tests pass again, 92 tests. Existing warnings remain. |
| Zoya | C and workspace release tests pass. | Same checks pass, 27 tests. No debug-suite claim. |
| Perla | GPU C, bounded workspace tests and explicit no-capture GPU oracle pass. | Same checks pass: 7 bounded workspace tests, `bench_gpu_tile` filtered; repeated exact GPU oracle passes once with no setup-skip output. That repeated oracle is not an additional unique test. |
| Mir/Erga origin locks | Locked metadata requires existing lock update. | Same prerequisite failure; local-resolved diagnostics above remain separate from origin. |
| Bita/Glia origin | No committed lock; locked metadata fails. | Same prerequisite failure. No origin-locked build claim. |
| Evy | BBG feature resolution fails. | Same missing `backend-unimem` requirement. |
| Nika/Trisha | Required exact vendor closure unavailable. | Same missing chaff/tasm-lib manifest; no moving or dirty vendor substitute. |
| Cyb | Authoritative Nu origin unavailable; isolated closure deliberately incomplete. | Same first missing staged cybergraph manifest. This is a setup limitation, not proof cybergraph is absent upstream. |

The API breaks are intentional removal of unsafe operations exposed as safe,
not baseline failures. They require consumer migrations with initialization,
owner lifetime, CPU/GPU exclusion and completion/visibility proofs; inserting
`unsafe` mechanically would not discharge those obligations. No consumer fixes,
version/pin edits, releases or compatibility approval are part of this unit.
Public semver evidence is supplied separately by the native implementation.

Warnings remain in full logs. `comparison.json` preserves their emitted sets;
compilation stops earlier in Xena/Bita/Erga and can mask old warnings or errors.
No global zero-warning or unchanged-diagnostic claim follows. Unavailable
Cyb/Nika/Trisha/Evy closures, full downstream suites, cross-platform checks,
format/Clippy/semver of consumers and Kelvin drift are not counted green.
`README.pre-execution.md` and `preparation.json` intentionally preserve the
historical pre-overlay state, superseded by these completed receipts.
