# acpu0.2.1 preparation tooling — review stage

Accepted [plan](../../.claude/plans/acpu-release-proposal.md), committed as
`0995cf9e51b5858df13f8349b083d409e0b306f5`. Owning compiler inputs still equal
the previous green source/package epoch; no version edit, branch switch,
candidate package or publication has occurred. Existing 0.2.0 audits are read-only.

This directory contains Nu-only orchestration. `common.nu` imports the previous
audit's generic byte/receipt/archive helpers unchanged. `replay.nu` binds raw
receipts to fixed tool/source identities and rederives the reported outcomes.
`export.nu` exports
exact committed tooling and that helper with relative paths preserved, so the
source-only bump checkout needs no tooling or audit additions.

After independent tooling review and its preparation commit, use explicit paths:

| Runner | Inputs and evidence |
| --- | --- |
| `export.nu REPO PREP_COMMIT TOOLS` | Fresh external immutable tool bundle; commit/hash for every file. |
| `owning.nu REPO RECORDS/owning TARGET TOOLS` | Eight fresh owning gates; complete source before/after each. |
| `package.nu REPO RECORDS/package TARGET TOOLS RECORDS/owning` | Exactly three committed bump paths; manifest-only package version and generated lock identity; official baseline plus untouched candidate; all package gates and targets. |
| `api.nu RECORDS/package RECORDS/api TARGET TOOLS` | Separate full-library baseline/candidate rustdoc artifacts; explicit pinned comparison. |
| `consumer.nu NOX RECORDS/package RECORDS/consumer TARGET TOOLS` | Fixed committed normalized Nox, unchanged four-test probe, isolated candidate override, registry type identity proof. |
| `verify.nu REPO RECORDS TOOLS FRESH_TMP` | Replay explicit candidate Git revision and archive even on unbumped preparation branch; no recorded historical cwd execution. |
| `retain.nu pack RECORDS TOOLS CAPSULE FRESH_STAGE` | Keep raw records, crate/API/probe artifacts and tools once; omit extracted package copies. |
| `retain.nu unpack CAPSULE FRESH_TMP` | Verify compressed inventory/hash before portable extraction. |

All target/output paths must be explicit external cache/tmp paths while the bump
branch is active. Successful evidence later belongs on the preparation branch;
the owner-facing bump PR contains only acpu/Cargo.toml, root Cargo.lock and
acpu/CHANGELOG.md. Any red remains attached to its original epoch.

Validation here concerns tooling only. Candidate-specific gates will run after
source review/commit; no test result for acpu0.2.1 exists yet. Owner-only bump
merge/tag/publication and fresh post-merge origin/main delivery remain required.

`review-epoch-1` preserves the first review's source bytes and manifest unchanged.
`validation-2` records parsed entrypoints and the real unbumped-source refusal.
`validation-3` preserves the first failed guard test and its exact source bodies:
Nu `each` deferred an error value. The eager-loop repair passed `validation-4`:
one real retained registry control, four synthetic corruption refusals and exact
current owner bytes equal the previously green source epoch.

`validation-5` records the repaired closure in `sources.json`; its compact JSON
SHA-256 is `a685d5b1df3f959933002de58508e1d0e59cc2555afee371ec71ccc2bf08abb3`.
The exact commands and revisions are in its receipts and the accepted plan.
Its replay controls check installed API binary hashes and raw versions, compiler
environment selection, raw historical test summaries, actual matmul output,
normalized package targets, consumer manifest/source paths, direct NeBu version,
clean VCS, and the independently fixed Nox archive checksum. Synthetic changes
are explicitly labeled and rejected; real inputs remain unchanged.

The prior raw workspace evidence contains 365 passing and 5 ignored tests across
28 summaries; each acpu library/integration run contains 196 passing and 1 ignored
test across 6 summaries. These numbers are replayed from the merged PR21 receipts,
identified in `replay.nu` and `replay-check.nu`; they are not acpu0.2.1 results.
No compiler gate was rerun for this tooling-only unit. `tooling-review.json`
freezes the active review inputs, including the unchanged shared helper.
