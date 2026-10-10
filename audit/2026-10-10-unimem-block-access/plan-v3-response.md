# PLAN v3 correction response

Base stays7b148906263da5ea8aabc5ddb0e2e44ff2288ecc. No production change,
candidate overlay/gate or source commit. Root owns the fresh vendor review.
V2 full documents, brief, input hashes, response and verdict are preserved at
`/tmp/unimem-block-access-plan-v2-archive`; SHA256SUMS:
`1cea80a27c66f99ef5bc1143bebcb25f0b1f962ef21288f7a13fd43327d8fca8`.

## C1 — executable lint reachability, observed rather than assumed

The two suggested all-targets alternatives were independently run on exact
parent archives with rustup1.95/LLVM22.1.2. Package selection without --no-deps
still stops on acpu lints; full workspace -W warnings also fails on existing
auto-discovered benchmark source/dependency errors. Neither proves all targets
were linted. Preserve both full -D and -W red gates and emitted diagnostics.

Added and executed:

- `cargo clippy -p unimem -p aruminium --lib --no-deps --locked --offline -- -D warnings -D clippy::missing_safety_doc`: exit0, no warnings; reaches the libraries containing all eight methods. Required green after implementation.
- `cargo clippy -p unimem -p aruminium -p rane --tests --no-deps --locked --offline --message-format=json -- -W warnings`: exit0, eight existing needless_range_loop diagnostics; completes all changed fixture modules without benchmark autodiscovery.

Candidate compares complete emitted message multisets including duplicate
counts, source text and child messages; line shifts remain in raw evidence.
Every added/removed finding is inspected; -D stderr gets per-code/source
comparison too. A matching failed set never proves unvisited-target coverage.
No suppression or unrelated lint/bench repair. Full raw receipt and normalized
JSON/script paths are in research and the new primary review supplement.

## R2 — complete unchanged-constructor inventory

The26 changed-method calls include warm INSIDE start_warm; calls TO the unchanged
start_warm signature are excluded by the word-boundary regex. A separate scan
of all157 exact-source Rust files, with equality against prior file hashes,
finds five constructor calls, not the review's suggested two:
alloc.rs:7, bandwidth.rs:41, pipeline.rs:267, layout.rs:28, roundtrip.rs:271.
tape.rs:35 is a definition. No example/bench migration is introduced.

## R3/R4 — precise spec corrections

Move both the experiments row and Layer1's duplicate table/adjacent throughput
projection verbatim to the same owner audit with missing original run/machine
provenance; link it from the compact spec. Other unrelated performance tables
remain out of scope. The lock statement now says exactly that creation locks
once, drop unlocks once and no lock/unlock or synchronization surrounds a typed
view. No universal IOSurfaceLock semantics are asserted.

## Small execution notes

Preserve mutants::skip. Add named vecadd and existing64³ matmul invocations.
Prepend the actual toolchain bin and record path/hash/version for Cargo external
subcommands: initial inherited Homebrew cargo-clippy caused E0514, separately
retained as tooling evidence. All successful ordinary parent gates remain
valid; only the two affected external subcommands were repeated under matching
tools. Existing native runs establish neither initialization soundness nor
universal import geometry. No row6/global-safety/release closure is claimed.

V3 changes only the two task docs. Existing review source selection is unchanged;
`REVIEW-V3-SHA256SUMS` adds complete supplemental scan/gate evidence. Full raw
archives remain local. Avoid carrying both v1 and v2 verdicts as duplicate
review context; the complete v2 verdict already independently checks v1 fixes.
