# unimem raw-access dependents — baseline diagnostics

2026-10-10. Only committed parent honeycrisp
`7b148906263da5ea8aabc5ddb0e2e44ff2288ecc` is used; no candidate overlay or
consumer source migration. This is local dependency evidence, not a release,
compatibility approval, warning-free result or full ring closure.

## Reproduction inputs and commands

`scenarios.json` pins every extracted repository, revision, archive SHA256 and
committed lock SHA256. Archives are the 19 exact-origin inputs catalogued in
`/tmp/unimem-access-origins.knYGlE/archives.json`; their acquisition and registry
relationships are in `/tmp/unimem-block-access-dependent-plan.md`.
`prepare.nu`, `prepare-local.nu`, and `prepare-drift.nu` create separate sibling
roots from those archives. No working-tree source is an input. The latter two
scripts were explicitly authorized for local lock diagnostics after recording
the origin-locked failures; they do not edit source/manifests or the origin roots.

`commands.json` and `logs/<scenario>/<label>.json` record each actual Cargo
command, cwd, explicit environment, times, exit, and lock hash before/after.
Adjacent `.stdout`/`.stderr` retain complete output, including metadata's full
source paths and every warning. `run.nu` returns successfully when it has
written a receipt; the Cargo exit is the receipt's `exit_code`, not the wrapper's.
Initial wrapper parser errors were corrected before Cargo began; they are not
compiler/test outcomes. Commands execute rustup 1.95.0 Cargo/RUSTC/RUSTDOC,
except Nika's declared nightly-2025-11-26. `tool-environment.json` pins versions.
Private targets use two build jobs and disabled incremental compilation.

Every scenario first attempted `metadata --format-version 1 --locked --offline`.
Missing cache artifacts were fetched only with `fetch --locked`, followed by
locked offline metadata. `metadata-paths.json` verifies every local package
stays inside its scenario and records the actual driver manifests. Nox's local
closure has no honeycrisp package; its successful rung does not exercise unimem.

## Observed results

`C` below means `cargo check --workspace --all-targets --locked --offline`.
All named tests/checks likewise preserve their scenario lock and run offline.
Warnings remain visible and were not promoted to errors or suppressed.

| Repository | Origin-locked result | Additional authorized local diagnostic |
|---|---|---|
| mir | Metadata 101: existing lock needs update. | Minimal seeded update; C and `--features bevy-plugin` check both 101, E0599 `Gpu::sync` absent at `src/gpu.rs:32`. No tests run. |
| erga | Metadata 101: existing lock needs update. | Minimal seeded update; C passes. Workspace tests 101: `chime::tests::{strikes_differ,renders_are_audible_and_unclipped}` overflow at `rs/app/src/chime.rs:132`; 11 preceding/completed tests passed, later binaries were not run. Declared bootstrap release build and bounded GPU difftest pass. |
| nox | Metadata 101: existing lock needs update. | Minimal seeded update; C and workspace tests pass, 161 tests. This is the registry owning rung, not driver call coverage. |
| bita | Metadata 101: no committed Cargo.lock. | Fresh local lock; C with `rbtc-hash5/metal-gpu` passes. Bounded hash5 Metal test: 62 passed, 8 filtered; not a full-workspace test result. |
| glia | Metadata 101: no committed Cargo.lock. | Fresh local lock; C 101, E0063 missing `has_attn_bias` at `run/tests/tier5_graph_ir.rs:22`. Library-only check and test pass: 92 tests. Integration suite remains blocked. |
| xena | Metadata passes; C 101. | Unchanged `crates/xena-bench/src/gpudebug.rs:192,195` indexes `()` (E0608). Scoped `xena-hash` + `xena-reference` all-target check and tests pass: 4 tests. No full-workspace test pass. |
| zoya | Metadata and C pass. | Full workspace release tests pass: 27 tests, none filtered. Debug tests were not run; source inspection found six repeated cache-generation tests, so optimized execution was chosen. |
| perla | Initial metadata needs cached bitflags2.12.0; locked fetch then metadata passes. | C with `perla-pow/gpu` passes. Workspace test with that feature passes 7 tests; `bench_gpu_tile` explicitly filtered. |
| evy | Metadata 101: BBG lacks `backend-unimem`. | The default workspace already fails feature resolution; optional probe/storage compilation or tests would not bypass that shared resolver failure and were not repeated. |
| nika | Metadata 101: missing `.vendor/nockchain/crates/chaff/Cargo.toml`. | Exact gitlink origin fetch was unavailable in preparation. No moving-branch or dirty vendor substitute, compilation, or tests. |
| trisha | Metadata 101: missing `.vendor/tasm-lib/Cargo.toml`. | Generated vendor recipes remain unexecuted. Existing sibling version conflicts are separately catalogued; no compile/test retry. |
| cyb | Metadata 101: first unstaged path is `../cybergraph/Cargo.toml`. | Full closure was deliberately not assembled once authoritative Nu origin was unavailable. This observed error is an incomplete isolated setup, not proof cybergraph is missing upstream. No compile/product gate. |

## Lock treatment and immutable-source proof

Origin scenarios retain every committed lock byte-for-byte. Bita/glia origin
scenarios still have no lock. Local locks are retained under `retained-locks/`,
with hashes and complete diffs; reuse these exact files for future candidate
comparisons, without a second resolution.

- Bita/glia generated one new lock each with `generate-lockfile --offline`.
- Mir/erga started from the committed lock and ran metadata without `--locked`:
  the only diff is removal of the stale `libc` dependency from unimem's entry.
- Nox likewise retained its seeded graph: its own package version changes
  0.1.2→0.2.0, plus seven existing path packages update to their pinned source
  versions (Hemera, lens-core and five strata packages). No registry versions
  change in this diff.

These five scenarios are explicitly local-resolved, never origin-locked or
release candidates. `source-comparison.json` contains full `diff -qr` receipts
against the exact extracted originals: the only differences are those five
root Cargo.lock files. `failing-source.json` also pins the unchanged files behind
observed code failures, including Glia's failing integration fixture.

## Test bounds and remaining gates

Bita filtering is intentional: seven `*throughput*` tests plus
`qhash_lc_bench::tests::attribution` are benchmark workloads. The GPU throughput
test alone traverses batches through 2^26 nonces, and attribution repeats 200,000
hashes per phase after warmup. The bounded run keeps the independent 16-nonce
GPU expectation comparison and 256-nonce easy-hit test. Its eight excluded names:

```text
metal::mining::tests::gpu_mining_throughput
miner::tests::concurrent_throughput
sha256_mb::tests::sha256_throughput_compare
qhash_lc::tests::lc_throughput
qhash_lc::tests::lc_throughput_parallel
qhash_lc_cpu::tests::cpu_throughput
qhash::tests::qhash_throughput
qhash_lc_bench::tests::attribution
```

Perla's excluded `tests::bench_gpu_tile` exercises 768×512×2176 panels with
batches 1/4/8 and repeated timing passes. Its retained jackpot oracle uses
16×16 panels and k128. Both Bita and Perla test code can return early if Metal
setup fails, so the specific oracle was repeated with `--exact --nocapture` in
`test-gpu-observed`. Both repeated single-test runs pass with no skip output;
the source guards print on setup failure. The silent-return easy-hit test alone
still does not establish hardware execution. No global driver safety claim follows.

Zoya cache tests instantiate 16,776,896-byte caches (actual constant, despite
stale 256MB comments) and 1,024-item test datasets, not the mining DAG. Xena's
pair/quad fixtures use bounded 543,744-byte scratchpads. Erga tests use 4,096-row
synthetic tables and local unsigned wallet fixtures. Its declared release build
used `RUSTC_BOOTSTRAP=1 cargo build --release --locked --offline`; the isolated
`cargo run -p erga-cli --bin erga --release --locked --offline -- difftest`
then passed on Apple M4 Max, comparing 512 nonces over a 4,096-row table. The
explicit CLI argument selects only the offline oracle. No live wallet transaction,
network pool, miner, packaging, installation or large epoch-table build ran.

Remaining unexecuted commands are not counted green: full Bita/Glia/Xena/Mir
workspace tests; Perla GPU benchmark; Zoya debug suite; unavailable Nika/Trisha/
Cyb closures and product/platform gates; strict Clippy/format/semver gates for
these downstream repositories. Candidate validation is not authorized here.
Re-run the same locked scenarios only after the owner's candidate source freeze.

At final observation private build targets occupy 5.2GiB, below the 30GB limit;
no cleanup was necessary. `SHA256SUMS` covers the compact receipts, helpers,
retained locks and full log streams; build products and source archives are not
part of that receipt payload. No repository file, commit, PR or candidate changed.
