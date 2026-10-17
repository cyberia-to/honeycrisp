# Candidate 1 validation repair

The unchanged, pushed three-file bump is
`2c70b3b929c791ca489bba12d2f5a6daef7d43e7`, based on
`14373cc8a0e18302751844579d94d1f8f80bd8b1`. Production tools were exported from
`3b875c9850f8ebe232a6650aaff7913a28773803`; their manifest SHA is
`5c74d87123d860ce25800f300cac8f2425770d6ba9bf57a33489603000bb0811`.
The candidate archive SHA is
`74967ea98831cd29d69fd3cef0cc08a8236f66d8d2fdcb2e99223428488d1159`.

All eight owning commands and all five normalized-package commands returned
zero under those actual tools. The final package metadata replay then refused
its locally stored integer `1` because `checked` requires `list<string>`.
`records/tooling-failure/target-inventory.*` preserves the real refusal with the
original exported tool. No package `result.json` was written or reconstructed.

The repair quotes that stored argument and the corresponding consumer metadata
argument in `verify.nu`. All other `checked` call sites have string arguments.
`validation-6/consumer-metadata.*` checks the real retained PR21 metadata receipt
with the explicit string argument; no original receipt is changed.
`validation-6/target-replay.*` invokes the
corrected helper on the same immutable package and owner metadata receipts:
`cargo metadata --no-deps --format-version 1 --locked --offline` yields 40
matching targets. Raw package `cargo test --lib --tests [--release] --locked
--offline` receipts each contain 212 passing, one ignored, eight summaries.
These results belong to candidate 1; no compiler command was repeated.

An attempt to retain this partial epoch exposed a second Nu runtime syntax
error: the multiline selection predicate in `retain.nu` interpreted `and` as
a new command. `validation-6/retention-before.*` preserves that refusal.
Parenthesizing the same three clauses repairs it without changing selection.
The real pack/unpack round trip is retained in `retention-after.*` and `unpack.*`.
`candidate-1-incomplete-retained` preserves all candidate-1 records so far,
both crate archives and the actual production tools; it omits only reproducible
extracted package copies. No raw receipt, archive or original tool was edited.

The corrected validation closure is separate from the production closure.
After root review, the unchanged production API and consumer runners can use
the existing package identity and exact original tools. The corrected verifier
can then replay their actual receipts and archives, with the original production
bundle passed explicitly. Its own committed/exported closure and invocation
must be retained separately. It does not require the absent package summary:
it derives package results from the successful commands and raw evidence.
This preserves evidence identities without rerunning unchanged compiler gates
or claiming that the original package runner completed.

No candidate source change, bump merge, tag or publication is authorized here.
