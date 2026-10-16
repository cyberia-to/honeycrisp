# unimem temporary-directory guard — CODE supplement

Verdict: APPROVE. No blocking findings in this harness-only correction.

Independent Codex reviewer: `/root/xtask_code_review`, 2026-10-16.
Parent remains `9cbd48070a136e24711896c930670fcff8a147fc`. This supplements
`/tmp/unimem-core-code-review-20261016.md`; its source approval and gate limits
remain unchanged. No source edits or test reruns were performed by this reviewer.

The gate now captures `mktemp` completion explicitly and rejects nonzero exit,
empty stdout, a nonexistent path, or a non-directory before any file writes or
cleanup. This closes the unsafe assumption that an external nonzero exit throws
in Nushell. The original gate is retained byte-for-byte in `harness-bridge`.
The consumer and manifest/lock fixtures retain their previously reviewed hashes.

Inspected actual evidence: `candidate-no-std-v2` succeeds for the corrected gate.
`candidate-temp-failure-v2` succeeds as a control: a failed `mktemp` returning an
existing sentinel directory and a successful `mktemp` returning empty stdout
both make the gate fail with its explicit diagnostic; the existing directory
contents and empty working directory remain unchanged. The control script's
assertions substantiate those outcomes. Both receipts bind the unchanged
candidate-v2 source epoch. No broad gate result is inferred from this supplement.

SHA-256 identities:

| Path | SHA-256 |
| --- | --- |
| `unimem/scripts/check-no-std.nu` | `e57d1e818a2283893774b2727d4e10511df4b76e9978a28eedcd77771c0391ce` |
| `audit/2026-10-16-unimem-core/harness-bridge/check-no-std.nu.before.txt` | `12d37297970207f930f46030026a02f7325056333b1a9b4b627cf29b8f3ae9bf` |
| `audit/2026-10-16-unimem-core/check-temp-failure.nu` | `4252646d05467d75dd8f22c89cd95b421fdac804b0dacac81dbbdffc87faa841` |
| `audit/2026-10-16-unimem-core/records/candidate-temp-failure-v2.json` | `70c6431007eb16fe142ff71199e33cd7ecbee8f473bfbb1b2e63975f8ca49f92` |
| `audit/2026-10-16-unimem-core/records/candidate-no-std-v2.json` | `2f783995247c82a97340a64ee8c859e73316d158c13cf35457b689bd51a9e8a4` |
