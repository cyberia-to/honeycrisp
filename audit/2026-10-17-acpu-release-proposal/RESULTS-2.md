# acpu 0.2.1 — final portable validation capsule

Use `candidate-2` for handoff. Candidate source remains the unchanged three-file
bump `2c70b3b929c791ca489bba12d2f5a6daef7d43e7`, based on reviewed main
`14373cc8a0e18302751844579d94d1f8f80bd8b1`. The acpu archive is unchanged:
SHA-256 `74967ea98831cd29d69fd3cef0cc08a8236f66d8d2fdcb2e99223428488d1159`.

`candidate-2/evidence.tar.gz` has SHA-256
`0c7ad9278b6af4d451a511cf514e1bee599745a633ef929900dbd98eae627d07`.
Its companion `bundle.json` contains the complete file inventory. It retains
the original owning/package/API records, successful consumer, all three tool
bundles, prior validator receipts, and the root `/tmp` alias failure.

All commands and scope limits in [RESULTS.md](RESULTS.md) remain unchanged:
eight owning gates; five normalized-package gates; 40 matching targets including
32 examples; patch API comparison with 229 pass and 31 skip; four Nox probe
tests in both debug and release. Workspace tests have 388 pass / five ignored;
acpu owning release and package debug/release each have 212 pass / one ignored.
These are the original commands against candidate `2c70b3b`, not rerun gates.
No source, compiler, package, API or consumer command was repeated for this
validator-only repair.

## Three exact bundles

| Role | Committed revision | Manifest SHA-256 | Capsule path |
| --- | --- | --- | --- |
| Owning/acpu package/API producer | `3b875c9850f8ebe232a6650aaff7913a28773803` | `5c74d87123d860ce25800f300cac8f2425770d6ba9bf57a33489603000bb0811` | `tools/` |
| Successful Nox consumer | `e791fefaf05c612d10d96f201e9d9460b20d502f` | `78f184494b7095f5804202a9cd661e1975b945577e8f8f705774e5aade170b9b` | `records/validation-tools/` |
| Final validator | `11c6397dc0b187eb95d2198c7463edad98f82638` | `5b2dd7663dd4688f92b29ef3554fbd0701cbdad139b5b28742d16203bf19b87b` | `records/validator-tools/` |

Root's first independent replay refused Cargo's `/private/tmp` metadata when
the newly requested extraction path was `/tmp`. The complete failed receipt
and raw streams remain in `validation-8/root-failure`. The only subsequent
runner change resolves the extraction path after creating its directory; it
does not rewrite historical paths or results. `verification-candidate-2`
records successful pack/unpack and full replay from real `/tmp` paths, using
the canonical repository as the explicit Git argument. The verifier requires
its executing file to belong to the supplied exported validator bundle.

After unpacking to a fresh external `UNPACK`, use a different fresh extraction
directory for the replay:

```text
nu UNPACK/records/validator-tools/audit/2026-10-17-acpu-release-proposal/verify.nu REPO UNPACK/records UNPACK/tools UNPACK/records/validation-tools UNPACK/records/validator-tools FRESH_TMP
```

The original package summary remains absent after its old tooling refusal;
the final verifier independently checks every successful raw command, test
total, archive/source/VCS identity, metadata target and tool identity. Earlier
capsules and failure epochs are immutable.

The normalized crate still uses registry unimem0.2.0, so its reexported `Tape`
does not deliver the owning workspace's later Tape repair. The diagnostic
still describes only pinned Nox `64d091e5`; Nox subsequently advanced to
`262de214521c47d18cd539040280f22e4b1fd8a4`, including PR36 production/CLI
changes. Live adoption must validate current Nox separately, without relabeling
this pinned diagnostic as latest-source or unpatched registry acceptance.

Only the owner may merge the source-only bump, tag or publish. Fresh delivery
must use pushed default-branch inputs after that merge. Root independent replay
of this final capsule precedes opening the owner-facing bump PR.
