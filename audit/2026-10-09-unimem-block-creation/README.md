# Checked native Block creation

2026-10-09, Apple Silicon. Implementation `6715335a3283e22218ad5f762316f2c735acb56f`
on merged nebu-identity parent `6560974dcd1a77b4b2e040a2b2a4187b702ace0b`.
This records a local combined candidate, with intentional API breaks and red
workspace gates. It establishes no release, rung, GPU, pinning or Kadek row-6 closure.

## Source and review identity

[source-freeze.sha256](source-freeze.sha256) retains the original nine
source/test/spec and two task-document hashes; manifest SHA-256
`e28991a40b49a908ff1882ba14079217f39570187d2374c7601943c59368db87`.
The implementation commit contains exactly those nine frozen product files.
The original full-overlay input was local nebu commit
`76b2f26432cca15607ab1dd18e488df68db51ca0`, strata
`56aedb2d12b3126c601eb333419136d403614dbb`, and those nine files; no task docs
entered compilation. [Original source record](overlay/sources.json) and archive
hashes remain unchanged. [Equivalence receipt](reproduction/source-equivalence.command)
is `git diff --exit-code 76b2f264 6715335^ -- Cargo.toml Cargo.lock src acpu unimem rane aruminium`,
exit 0: the merged parent has the measured dependency/build-source bytes.

Fresh different-vendor [PLAN](reviews/plan.txt) and [CODE](reviews/code.txt)
reviews both returned APPROVE; reviewers ran no commands. Brief SHA-256:

- PLAN: `4cba3dcdb74d4da3dd689178d320930b9e237fc2bfbd2e8bab4d7d2761b66080`.
- CODE: `7ef783a031bce898ba0e657c25da136146318f55b96bf1d28d3de355eb871ac5`.

[Invocation provenance](reviews/invocations.md) distinguishes coordinator
session-recorded exits from captured process receipts.

The raw CODE review's phrase “full `slice::from_raw_parts` contract” is too broad.
This unit checks numeric and mapping bounds. Initialization, safe mutable-view
aliasing, raw-handle/import lifetime, publication, physical residency, portable
support and opaque CF allocator fallibility remain unresolved by this unit.
The review separately identifies these exclusions; its original text is preserved.

## Measured gates

Each receipt links to an exact `.command` record, process `.exit`, and raw output.
Isolated receipts use one combined `.log`; overlay/baseline preserve stdout/stderr.
All numbers below derive from those commands and the frozen source above.

| Receipt | Exit | Observation |
|---|---:|---|
| [Isolated debug](isolated/logs/debug-final.command), [release](isolated/logs/release.command) | 0 | Each: 50 native tests + 2 stable doctests |
| [Isolated Clippy](isolated/logs/clippy-final.command), [fmt](isolated/logs/fmt-final.command), [docs](isolated/logs/docs.command), [build](isolated/logs/build-final.command) | 0 | Clippy `--all-targets -- -D warnings`; no warnings |
| [Nightly doctests](isolated/logs/nightly-doctests.command) | 0 | 2 tests; actual matching nightly RUSTC/RUSTDOC enforce E0451/E0599 |
| [Native alignment](isolated/logs/alignment.command) | 0 | API alignment agreement, row shape and exact extent; includes 256 MiB |
| [Semver origin](isolated/logs/semver-origin.command), [tag](isolated/logs/semver-tag.command) | 100 | Each: 202 checks, 201 pass, 1 fail, 58 skip; five added exhaustive MemError variants |
| [Workspace fmt](overlay/gates/fmt.command) | 1 | Baseline-identical aruminium formatting output after root normalization |
| [Workspace Clippy](overlay/gates/clippy.command) | 101 | Baseline-identical 16 unique acpu diagnostic/location pairs |
| [Workspace release](overlay/gates/release.command) | 0 | Four pre-existing output-name collision warnings retained |
| [Workspace tests](overlay/gates/tests.command) | 0 | 336 pass / 0 fail / 5 ignore; baseline 320 / 0 / 5; all 25 source-warning/location pairs identical |
| [unimem build](overlay/gates/unimem-build.command), [tests](overlay/gates/unimem-tests.command) | 0 | 50 native tests + 2 doctests |
| [Pipeline](overlay/gates/pipeline.command) | 0 | Both `reference verified: true` and `unimem verified: true`; CPU/AMX and ANE executed |
| [Bench](overlay/gates/unimem-bench.command) | 0 | All 16 Criterion cases reached analysis; smoke execution only |

Pipeline checks ANE identity output and finite FFN values; it has no executed
GPU stage or full transformer golden comparison. Bench timings remain raw
observations in the log, with no optimization claim. Five workspace ignores
are unchanged (one acpu test, four aruminium doctests). See
[computed results](overlay/results.json) and [baseline comparison](overlay/baseline-comparison.json).

The first isolated Clippy failure is preserved. One new test lint and two old
roundtrip lints were corrected: `is_ok`, nonnull `ptr::dangling`, and the exact
3.14f32 bits `0x4048_f5c3`. Final debug/release cover those corrections.
Standalone manifests drop unrelated dev dependencies/examples/benches only
inside the harness; exact runtime dependencies and final per-candidate locks
are in [isolated/manifests](isolated/manifests). Original root/tag revisions are
`9593c7218e14e2b5b816d9008b45ab60b9580cf3` /
`96fa4f4509d7a966cb22a4bb9e12ac753cff43ae`.

## Compatibility and contract extent

[Compiler probes](probes) independently show four raw-FFI source breaks:
CFNumber type constant/parameter i32 → isize, CFDictionary capacity i64 → isize,
and IOSurfaceCreate mutable → const dictionary input. The baseline compiles;
the candidate reports exactly four E0308 errors. These were not reported by
semver-checks. CFIndex is ABI-equivalent to i64 on Apple LP64; no previous runtime
miscompile is claimed. Private plan construction fails E0451, the absent
unchecked constructor E0599. No version bump or compatibility aliases are included.

The two specs add 51 lines / 3150 bytes by [recorded `wc -lc`](reproduction/spec-size.command): README 480/17636 → 502/19371;
API sketch 307/7884 → 336/9299. This adds a checked-constructor contract; it is
not a strict-refinement size-reduction claim. Seven source/test files are
225/205/333/83/49/143/450 lines in manifest order, all within 500.

## Reproduce

Requires macOS Apple Silicon, Nu, Git and the retained Rust toolchains/dependencies.
Measured versions: Homebrew Rust/Cargo 1.95.0, rustc `59807616e`, semver-checks
0.51.0; nightly-2025-11-26 rustc `80d8f292d`; macOS 26.4.1/25E253, Mac16,5,
16384-byte pages. Exact version commands are in `isolated/logs/`.

From this audit directory, with repositories containing the pinned commits:

```nu
shasum -a 256 -c ARTIFACTS.sha256
nu check-records.nu
nu prepare.nu /tmp/block-repro --honeycrisp /path/to/honeycrisp --strata /path/to/strata
nu replay.nu /tmp/block-repro isolated debug-final
nu replay.nu /tmp/block-repro isolated build-origin
nu replay.nu /tmp/block-repro isolated build-final
nu replay.nu /tmp/block-repro isolated ffi-origin
nu replay.nu /tmp/block-repro isolated ffi-current
nu replay.nu /tmp/block-repro overlay tests
```

`prepare` only archives exact commits, applies checked hashes and copies the
preserved standalone manifests/locks; no build, working-tree source, dependency
shim or shared target is used. It requires a fresh destination. `replay` accepts
any preserved isolated/overlay receipt except checkout-only `diff`; use table
receipt names to run the remaining gates, including semver and actual nightly.
It records and returns each process's real exit, including expected red gates.
Original absolute paths stay in historical receipts; reproduction relocates
them to private prepared roots. `original/` retains the original harness scripts.

[Reproduction checks](reproduction) preserve prepare/record verification and
source-equivalence receipts. Preparation succeeds, refuses an existing output,
and replay preserves both toolchain exit 0 and formatting exit 1. The first
record-check script accidentally included
compiler summary lines as warnings; that exit 1 is retained. Filtering summaries
reproduces the original 25 source-warning pairs, exit 0; no product source changed.
