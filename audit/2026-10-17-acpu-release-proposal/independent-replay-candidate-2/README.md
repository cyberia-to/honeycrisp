# Independent final capsule replay

Root reported successful independent unpack and replay on 2026-10-17. The
unpack was `/tmp/acpu-candidate2-root-20261017a`, from the immutable `candidate-2`
capsule at preparation commit `bae9306dfc0e57ee9630d26138636e50f8d3285b`.
The capsule SHA-256 remains
`0c7ad9278b6af4d451a511cf514e1bee599745a633ef929900dbd98eae627d07`.

Root ran the following command with canonical Honeycrisp as the Git argument:

```text
nu /tmp/acpu-candidate2-root-20261017a/records/validator-tools/audit/2026-10-17-acpu-release-proposal/verify.nu /Users/master/cyber/honeycrisp /tmp/acpu-candidate2-root-20261017a/records /tmp/acpu-candidate2-root-20261017a/tools /tmp/acpu-candidate2-root-20261017a/records/validation-tools /tmp/acpu-candidate2-root-20261017a/records/validator-tools /tmp/acpu-candidate2-root-extract-20261017a
```

`result.json` is a byte-for-byte copy of root's
`/tmp/acpu-candidate2-root-result-20261017a.json`. The three `target-replay`
files are byte-for-byte copies from the root extraction directory. Their
actual metadata receipt records exit zero, canonical `/private/tmp` paths,
the command, compiler environment, timestamps and stream hashes. `copies.json`
binds each retained copy to its original path and SHA-256. No outer command
receipt was synthesized from the reported completion.

The result verifies source `2c70b3b929c791ca489bba12d2f5a6daef7d43e7`, version
0.2.1, unchanged archive `74967ea98831cd29d69fd3cef0cc08a8236f66d8d2fdcb2e99223428488d1159`,
eight owning gates, five package gates, API comparison, two consumer modes and
the three tool manifests listed in [RESULTS-2.md](../RESULTS-2.md). Original
compiler/API/consumer commands were reused with their exact producing bundle.
The earlier root alias failure and every historical capsule remain unchanged.

Root also reviewed the preparation diff against `origin/main`: only `audit/`
and `.claude/` differ. The source-only bump remains a separate owner decision.
