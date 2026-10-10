# Exact-parent execution receipt

2026-10-10. Honeycrisp `7b148906263da5ea8aabc5ddb0e2e44ff2288ecc`,
strata `56aedb2d12b3126c601eb333419136d403614dbb`; immutable prepared archives,
preserved Cargo.lock, private target. No candidate overlay or gate, production
edit, plan edit, dependency edit, semver run or source commit.

Commands and stdout/stderr/exit are in `../receipts/parent/<label>/` relative to
this file. `command.nuon` records
exact arguments, compiler paths and environment; later receipts also retain
resolved executable paths, SHA256/version outputs and runner SHA. All compile
gates use rustup 1.95.0 rustc59807616/LLVM22.1.2, explicit RUSTC/RUSTDOC, jobs4,
locked/offline resolution. Resource snapshots live in the coordinator results.

The initial Clippy attempt selected Homebrew cargo-clippy from inherited PATH
despite `rustup run 1.95.0 cargo`, causing E0514 against rustup-built dependency
metadata. That is a tooling failure, retained under clippy, not a source failure.
The corrected runner prepends the actual toolchain bin directory and records
executable version/hash evidence; fmt and Clippy were repeated once under
fmt-pinned/clippy-pinned. Initial fmt also used inherited external subcommand
resolution. Built-in Cargo build/test/doc/native gates use the explicitly pinned
compiler and remain valid. `toolchain-correction.json` records both resolutions.

Observed results (each row names a retained command receipt):

| Effective label | Exit | Evidence |
|---|---:|---|
| fmt-pinned | 1 | Existing aruminium/benches/objc2.rs formatting at300/371/418/433 |
| clippy-pinned | 101 | Four acpu library diagnostics observed before the gate stopped; other findings may be masked |
| clippy-owned | 101 | Selecting unimem/aruminium without --no-deps still stops on the same acpu diagnostics |
| clippy-owned-nodeps | 101 | All-targets reaches selected packages but has existing test lints and auto-discovered benchmark dependency errors |
| clippy-warnings | 101 | Workspace/all-targets -W warnings: four emitted acpu warnings,36 E0433 errors in aruminium benchmark targets, two failure notes; other targets can still be masked |
| clippy-owned-lib | 0 | Selected libraries --lib --no-deps -D warnings -D clippy::missing_safety_doc actually reaches unimem and aruminium; no warnings |
| clippy-fixtures-warnings | 0 | Selected unimem/aruminium/rane --tests --no-deps -W warnings completes with eight existing needless_range_loop warnings |
| release | 0 | cargo build --release --workspace --locked |
| tests | 0 | cargo test --workspace --locked:336 passed,0 failed,5 ignored across24 result lines |
| docs | 0 | cargo doc -p unimem -p aruminium --no-deps --locked, RUSTDOCFLAGS=-Dwarnings |
| gpu-block / gpu-raw / ane | 0 each | Explicit existing native integration filters |
| unimem-release / unimem-build | 0 each | Existing release suite includes block_creation and roundtrip; new block_access absent |
| unimem-bench / pipeline | 0 each | Existing benchmark command; pipeline prints both reference/unimem verified:true |
| gpu-example | 0 | Named vecadd command reports1024 verified sums |
| ane-example | 0 | Named matmul defaults to64×64×64 in source, reports4096 verified ones; no size CLI option |

The workspace test command also reports existing matmul output-name collisions
and acpu example warnings. An exit0 does not mean a warning-free gate. The four
observed workspace Clippy findings are duplicated_attributes at gemm/mod.rs:1094,
assign_op_pattern at sparse/chebyshev.rs:171, and needless_range_loop at
chebyshev.rs:192 and sparse/mod.rs:61; this is not a count of all latent findings.
The parent-requested clippy-owned command also stops on those acpu path-dependency
lints; package selection alone does not establish lint reachability. All-targets
--no-deps gets further but stops on pre-existing auto-discovered benchmark imports
(compare/objc2). The full workspace -W command independently encounters standalone
bench aruminium.rs's leading-super error and wgpu_bench.rs's missing wgpu/pollster.
These failed invocations do not prove all targets were linted. No dependency or
lint suppression was added. The library-only strict gate supplies the reachable
missing_safety_doc check. The complete selected fixture run covers all changed
test modules without benchmark autodiscovery; its eight warnings are at
rane/tests/hardware.rs:49/140/187, rane/tests/integration.rs:37/43, and
aruminium/tests/integration.rs:30/35/52.

Both JSON warning receipts retain every emitted diagnostic, normalized comparison
keys and summaries. `diagnostics.nu` records target/kind/code/message, source text,
child messages and duplicate counts; comparison ignores moved line/column numbers,
retaining them in diagnostics.json. Later candidate comparison must inspect all
added/removed findings, not only exit status. A failed run's set remains only the
emitted set. The strict -D raw stderr also remains a mandatory per-location/code
comparison. The temporary script's self-comparison has zero differences; this is
helper validation, not a candidate gate.

Native execution is an existing-behavior check, not proof of Rust initialization,
aliasing or universal IOSurface geometry. Direct ANE uses the existing private-API
synchronous-success assumption; rane's unlock-before-dispatch documentation and
Block's lifetime lock remain distinct, unmodified contracts here. No portability,
Miri, all-unimem safety or Kadek row6 closure is claimed.

The coordinator's first invocation failed to parse because Nu resolved `du -sk`
as its builtin. Changing that temporary helper to `^du` fixed it before any Cargo
gate; `unimem-block-access-parent-baseline-v1.nu` preserves that script. Initial
preflight also found the private target directory absent, as expected before
the first build. These are scaffolding events, excluded from Cargo gate totals.

`final.json` and `source-postcheck.json` here, plus `PARENT-BASELINE-SHA256SUMS` in
the scaffold root, provide final outcomes and source integrity. Archive postcheck
covers450 honeycrisp and244 strata files in each of parent/candidate and detects
additional paths. Successful commands were not rerun unchanged; the two retries
correct the observed subcommand toolchain mismatch. All performance output stays
raw evidence for these commands/revisions, without a performance conclusion.
