# Independent final delivery CODE supplement

APPROVE — Codex `/root/xtask_code_review`, 2026-10-16, scoped source review.
This supersedes the delivery-source identity in `../delivery-final/independent-code-review.md`;
its earlier observations and actual gate history remain intact. Current complete
owner/Strata closure is
`45468f4172e1592f6c9fec7ce56a1af5553aa7ec93b714836b18775138d08352`.
The independently rehashed inventory is in `independent-source-verification.json`;
`independent-delivery.json` binds the complete retained patch from admission
commit `c5177c56e4551c9896a2869c3b23d0c7bac9a6f9`.

Since the earlier approved `5faaae626d09eab9d2b4c989a77defd27889ba6cc5a589d06cc296d07c42cddd`
epoch, seven files first changed for warning/target hygiene (`0453b62f...`).
The three benchmark files remove unused imports, one unused mutability marker
and redundant expression parentheses; evaluation order and values are unchanged.
The two examples remove a pure unused constant and private, uncalled
`Snapshot::diff`; snapshot capture, AMX operations and operands remain unchanged.

The root-authorized `autobenches = false` settings remove ten accidentally
auto-discovered targets from the two library packages. This is an intentional
target-set change: those files belong to the separate workspace benchmark
packages. I independently ran locked, offline Cargo metadata and recomputed
the before/after comparison in `independent-targets.json`. All five explicit
`metal-benches`/`rane-benches` binaries remain. The other removed auto-targets
are modules imported by those binaries; their files remain present. All other
targets, including the previously reviewed forty examples, are conserved.

The final bridge from `0453b62f...` changes documentation only: array expressions
are code-marked, the RoPE formula uses a text fence, and `Matrix::new` now states
its actual behavior (AMX_SET followed by Ok, unsupported hardware may trap).
The latter removes a false promise of a recoverable error; it does not implement
a hardware probe or change the pre-existing safe API. No arithmetic, unsafe
operation, tolerance, admission guard or test assertion changed in this bridge.

Verification questions checked: (1) current file inventory and complete hashes
match the new closure; (2) no target disappeared outside the ten approved
auto-target copies; (3) every standalone benchmark binary and helper source is
retained; (4) the documentation bridge contains no executable change; (5) the
earlier SDK-backed ABI correction and admission guards are unchanged. No new
source blocker was found. No broad tests were rerun by this reviewer.

Final `delivery-v4` owning gates are still collecting at this review point;
their source-bound receipts, not earlier epochs, determine integration status.
The separate numerical Chebyshev overflow issue remains open. This approval
does not authorize or claim owner-only crate publication.

## Receipt supplement, 2026-10-16 23:04 UTC

The ten final owning gates are now present and green on the same `45468f41...`
closure. `independent-gate-verification.json` independently rehashes all 493
owner/sibling inputs, all twenty raw output streams and each before/after source
identity. Recorded commands prove workspace tests 356 passed / 5 ignored,
release acpu tests 187 passed / 1 ignored, fmt, both strict Clippy commands,
release workspace build and strict acpu docs. The actual AMX, Metal and ANE
example runs pass their own result checks. This reviewer inspected those output
checks; these are device-execution witnesses, not performance claims. An extra
all-target compile is still running and is not included in these ten results.
Complete current delivery patch SHA256:
`917c95794981224827cb6237a9b23da01bdc0eccd75e51d813165c9777a1f05d`.
