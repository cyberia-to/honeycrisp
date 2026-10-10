# Frozen Block-access candidate — native validation

2026-10-10. Local reviewed overlay, not an origin release. Honeycrisp parent
`7b148906263da5ea8aabc5ddb0e2e44ff2288ecc`, Strata
`56aedb2d12b3126c601eb333419136d403614dbb`; exact Git archives, distinct parent
and candidate trees/targets. Source manifest `/tmp/unimem-block-access-source-v1.sha256`
SHA256 `1ead683ed4e75cfb0c2edd196e6e18e979f3ca83634e69c3bcdd2bfe72e1b1be`
contains all16 changed/new paths. No manifests, locks, versions or consumer
sources changed. Source freeze still matches both owned worktree and candidate.

`nu run.nu candidate <gate>` recorded19 commands. Two explicit semver follow-ups
bring the native receipt total to21; wrapper completion does not mean every
Cargo exit passed. `candidate-results.json` stores exact command arrays, timestamps,
exits and paths. Each `receipts/candidate/<label>/` retains command.nuon,
stdout, stderr, result.json, executable paths/hashes/versions and lock equality.
The parent receipts use the identical committed input and compiler. The first
parent Clippy attempt used an unintended Homebrew cargo-clippy and failed E0514;
that receipt is retained, and `clippy-pinned` is the comparison baseline.

Stable gates use actual rustup1.95.0 compiler commit59807616, LLVM22.1.2, with
matching cargo/clippy/rustfmt/rustdoc binaries, explicit RUSTC/RUSTDOC and toolchain
bin prepended to PATH. Homebrew LLVM22.1.3 is not the executed compiler here.
Nightly doctests use actual nightly2025-11-26 (Rust1.93), both compiler and
rustdoc. Four jobs maximum; total live native targets remained below20GB.
No Miri or cross-platform/native physical-address proof is claimed.

## Commands and observed results

Every command is Cargo via `rustup run 1.95.0 cargo`, locked and
CARGO_NET_OFFLINE=true, unless noted. Full argument arrays and compiler hashes
are authoritative in the adjacent receipts.

| Label / arguments | Exit and observation |
|---|---|
| clippy-owned-lib: `clippy -p unimem -p aruminium --lib --no-deps --locked --offline -- -D warnings -D clippy::missing_safety_doc` | 0, zero warnings; reaches all8 changed unsafe public methods. Content still requires review. |
| clippy-fixtures-warnings: `clippy -p unimem -p aruminium -p rane --tests --no-deps --locked --offline --message-format=json -- -W warnings` | 0; same8 existing rane warnings; complete normalized diagnostic multiset added0/removed0. |
| nightly-doc: actual nightly `test -p unimem -p aruminium --doc --locked` | 0;14 compile-fail cases pass, including12 new;4 preexisting aruminium examples ignored. Named E0133/E0596/E0499/E0502 checked by nightly rustdoc. |
| tests: `test --workspace --locked` | 0;351 passed,0 failed,5 ignored across25 result summaries. Existing example/FFI/output-name warnings remain. Parent336 passed; delta is3 native tests plus12 doctests. |
| unimem-release: `test -p unimem --release --locked` | 0;66 passed,0 ignored. All7 block_access tests, block_creation and roundtrip execute. |
| gpu-block / gpu-raw: `test -p aruminium --test integration wrap_block_vecadd` / `gpu_buffer_wrap`, each `--locked` | Both0, one selected test each, no setup skip. |
| ane: `test -p rane --test hardware run_direct_with_unimem_block --locked` | 0, one selected test. Retains inherited private-ANE completion/visibility assumption; no public guarantee inferred. |
| docs: `doc -p unimem -p aruminium --no-deps --locked`, RUSTDOCFLAGS=-Dwarnings | 0. |
| gpu-example: `run -p aruminium --example vecadd --locked --offline` | 0. |
| ane-example: `run -p rane --example matmul --locked --offline` | 0; example source fixes64×64 and takes no size argument. |
| release: `build --release --workspace --locked` | 0; existing output-name collision warning retained. |
| unimem-build: `build -p unimem --locked` | 0. |
| unimem-bench: `bench -p unimem --locked` | 0; full raw measurements retained, no performance conclusion or comparison asserted. |
| pipeline: `run -p unimem --example pipeline --release --locked` | 0; bounded existing example, not a full transformer golden test. |
| fmt: `fmt --all -- --check` | 1; preexisting aruminium/benches/objc2.rs formatting. Output equals exact pinned parent after archive-path normalization. |
| clippy: `clippy --workspace --all-targets --locked -- -D warnings` | 101; same4 emitted acpu findings. After removing only Cargo compile progress, all diagnostic bytes equal pinned parent. Failure masks later targets. |
| clippy-warnings: `clippy --workspace --all-targets --locked --offline --message-format=json -- -W warnings` | 101; target masking differs, detailed below. |
| semver-parent: `semver-checks --workspace --exclude metal-benches --exclude rane-benches --baseline-root <parent>/honeycrisp --verbose` | 101; aruminium detects newly unsafe Gpu::wrap; acpu/honeycrisp/rane202 checks pass each. Tool then encounters ambiguous unimem manifests inside prior audit snapshots. |
| semver-parent-unimem: `semver-checks --manifest-path <candidate>/honeycrisp/unimem/Cargo.toml -p unimem --baseline-root <parent>/honeycrisp/unimem --verbose` | 100, intended breaking diagnostic:202 checks,200 pass,2 fail,58 skip. Detects7 unsafe additions and4 shared→exclusive receiver changes, each duplicated by public reexports. |
| semver-tag-unimem: same explicit crate command against exact v0.2.0 archive96fa4f4509d7a966cb22a4bb9e12ac753cff43ae | 101 during metadata, missing unpinned historical ../nebu/rs. No substituted sibling/edited historical manifest and no tag compatibility verdict. |

Semver tool0.51.0 creates its own scratch manifests/locks and selected registry
crossbeam-queue0.3.14; this API diagnostic is distinct from the locked workspace
build/tests. All21 receipts confirm production Cargo.lock unchanged.
The two follow-up scripts are `/tmp/unimem-block-access-semver-followup.nu` and
`/tmp/unimem-block-access-semver-tag.nu`; first workspace failure is retained.

## Red-gate attribution and limits

Strict Clippy findings: duplicated_attributes acpu/src/gemm/mod.rs1094;
assign_op_pattern acpu/src/sparse/chebyshev.rs171; needless_range_loop there192
and acpu/src/sparse/mod.rs61. `candidate-clippy-strict-comparison.json` preserves
the exact comparison against parent/clippy-pinned. The preliminary comparison
mistakenly selected the old tooling-failed parent/clippy receipt; its false result
is retained in `candidate-clippy-strict-comparison-before-label-fix.json` and was
corrected without rerunning or modifying gate evidence.

Full -W parent emitted4 warnings,36 errors,2 failure notes; candidate emitted10
warnings,1 error,1 failure note. Candidate still fails E0433 in the unchanged
standalone aruminium/benches/aruminium.rs (too many leading super keywords).
The35 old wgpu_bench E0433 errors and its failure note were not emitted by the
candidate's failed parallel invocation; they are masked, not fixed. The6 new
emitted warnings are5 rane test diagnostics independently present in the complete
parent fixture gate, plus1 unchanged rane/examples/matmul.rs warning. Source
hashes and every addition/removal classification are retained in
`candidate-clippy-warning-attribution.json`; raw JSON remains adjacent to each
command. Thus no equal full-workspace diagnostic-set or complete lint coverage
claim is made. Fixture-only sets are exactly equal including duplicate count,
source text and child notes, with line coordinates retained in raw evidence.

`candidate-source-postcheck.json` verifies all450 parent Honeycrisp and244 Strata
files, all452 candidate Honeycrisp files (14 replacements+2 new) and244 Strata
files, with no hash mismatch/additional path. `candidate-freeze-postcheck.json`
checks all16 owned source paths again. The evidence generator's original wrong
strict-comparison label and its corrected script are preserved; no Cargo rerun
was needed. Owned-file rustfmt and worktree diff-check passed; global fmt stays red.

No safe CPU ownership abstraction, Grid/Buffer soundness closure, CPU/GPU fence
object, global initialization, native physical contiguity, external compatibility,
release or Kadek foundation row6 closure is claimed. The exact native fixture
proves only its checked mapping and synchronized operations on this host.
