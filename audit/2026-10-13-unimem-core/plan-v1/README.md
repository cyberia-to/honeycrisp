# Unimem core boundary: PLAN evidence

Research and proposed implementation on Honeycrisp
`9cbd48070a136e24711896c930670fcff8a147fc`, with committed Strata
`494df7a0973ae10a6cc1c2022e644d70449a29d6` supplying the existing dev dependency.
Production remains unchanged. Fresh different-vendor PLAN review is pending at
this frozen evidence epoch; no implementation approval is asserted.

The standalone own-panic-handler consumer fails E0152 because unimem depends on
std. Existing owning tests pass but can re-enable std through reverse dev
dependencies. An isolated consumer runs all 44 existing native integration tests
without those edges. The proposed feature policy preserves default std queue
behavior and exposes a no-default path; native frameworks, allocation/access
contracts and physical-memory questions remain unchanged/open as recorded.

[Research](../../../.claude/tasks/unimem-core/research.md) and
[PLAN](../../../.claude/tasks/unimem-core/plan.md) name all gates and limitations.
Parent records 03/06 show 53 owning tests + 13 doctests and 44 isolated native
tests. Records 11/14/15 pass release workspace build, pipeline and bench execution.
Workspace format, Clippy and test compilation remain red (09/10/12). Actual last
tag v0.2.0 cannot resolve its historical NeBu path (16/17); compatibility, broader
rung/product/dependent suites and Kelvin checks are unverified. Bench timings are
raw execution evidence without a new performance guarantee. `decision` applies;
no version, tag or publication is part of this source unit.

`evidence.tar.gz` retains the exact 184,360-byte readable PLAN packet, all 284
indexed support files, complete streams/receipts, source archives and checksummed
primary dependency sources, proposed gate/templates, preparation history and
verifier. `manifest.json` identifies the archive and review input. The full 168
registry archives and compiler outputs are omitted; their acquisition identities
and workspace-lock checksums are retained. The 49 direct primary source bodies
are present. No omitted support body is claimed to have been read by the reviewer.

Run `nu verify.nu --output <fresh-result.json>` from this directory. It hashes the
archive, extracts into an isolated temporary directory, reconstructs the readable
packet, checks source modes/blobs/bodies and streams, then removes that temporary
directory. It uses Nu, Git hash-object and standard macOS tar/stat tools, without
Ruby/Python. `--git` additionally compares tree membership with the original local
Git repositories; the default replay verifies the archived identity chain. The
actual initial replay with Git is preserved inside the capsule. New replays write
only to the requested output and temporary directory.

The packet's initial preparation copy had literal newline escapes in its footer;
the final packet replaces those two escapes with newline bytes. No source/evidence
body changed; the earlier unsubmitted copy remains at
`/tmp/unimem-core-plan-preparation-v1`. Nu setup failures and the Homebrew-versus-
Rustup durable-gate distinction are retained explicitly in `support/preparation.txt`.
