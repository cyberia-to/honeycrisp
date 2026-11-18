# Byte-preserving consumer and verifier repair

This supersedes the continuation proposed in `validation-repair-1.md` without
changing its retained evidence. Candidate source remains
`2c70b3b929c791ca489bba12d2f5a6daef7d43e7`; owning, package and API production
tools remain `3b875c9850f8ebe232a6650aaff7913a28773803`.

The original consumer packaged the unchanged normalized Nox successfully, then
refused before creating a probe or running tests. Direct assignment of
`git show` output removed its final newline in Nu. The committed and working
probe still have the required SHA
`eb0e4498e99aa7de6ddcaca18ee39ec38a160ebccd322dc30a8e070a6cfae77d`;
the stripped value has SHA
`7b0a9de300af3070ddf53fafdd8f646797aa0594911524a4b44ef4f7080f4d97`.
`validation-7/newline-check.nu` demonstrates that exact difference, checks all
ten original exported files against Git, and admits the unchanged candidate's
exact manifest bytes. Its receipt names both repositories and tool bundle.

The coherent repair reads the probe, verifier tool bodies and both manifests
through `complete.stdout`, checking Git exit codes. Other Git body readers in
`git-inputs`, `export.nu` and the unchanged shared package helper already use
that form; parsed lock readers intentionally compare TOML values. All typed
`checked` argument lists are strings, including both metadata version arguments.
All other multiline predicates already have enclosing parentheses.

The API result passed under the original production tools: the explicit
patch-level comparison returned 229 passing checks and 31 skips, with no semver
update required. Its command, source and tool identities remain in candidate-1
records. No successful compiler, package or API command is repeated here.

The corrected consumer accepts its actual new bundle plus the original package
producer bundle. Existing `result.tooling` records the consumer's actual bundle;
the unchanged package-identity hash links it to the old producer evidence.
The verifier takes explicit producer, consumer and validator bundle arguments,
checks every manifest file against its own Git commit, and compares the existing
receipt identities with the appropriate bundle. Producer and API identities
remain old; consumer and validator identities are new. No broad schema is added.

The original consumer directory and package's missing `result.json` remain
unchanged. A separate final records directory receives byte-checked copies of
the existing owning/package/API/proposal evidence and a fresh consumer output.
Raw failures are retained separately. The final verifier must run from the
committed exported validation bundle and record all three bundle hashes.
Final pack/unpack and root replay precede the owner-only bump PR.

Commands after review:

```text
consumer.nu NOX PACKAGE NEW_CONSUMER TARGET CONSUMER_TOOLS PRODUCER_TOOLS
verify.nu REPO RECORDS PRODUCER_TOOLS CONSUMER_TOOLS VALIDATOR_TOOLS FRESH_TMP
```
