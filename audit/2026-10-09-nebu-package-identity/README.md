# Restore origin nebu package resolution

Implementation `76b2f26432cca15607ab1dd18e488df68db51ca0`, parent honeycrisp
`9593c7218e14e2b5b816d9008b45ab60b9580cf3`, strata
`56aedb2d12b3126c601eb333419136d403614dbb`. Serves Kadek foundation row6's
unimem workspace prerequisite. [Evidence](evidence.json) records every command,
working directory, exit, output and tool version below; transitive metadata
stdout is hash-pinned with its local-path/feature subset retained.

Only workspace dependency `nebu.package` changes, `cyb-nebu` → `strata-nebu`;
the alias/path and version/feature declarations stay identical. Cargo refreshes
the existing lock: five local strata packages resolve at0.1.1, and obsolete
registry acpu/unimem duplicates disappear. Every remaining registry tuple is
identical; no blanket update. The resolved nebu node selects `default`, whose
feature list is empty; its four dependencies are the strata tiers. `fast` stays
off deliberately, avoiding another registry driver beside local acpu. Benchmark
speed/acceleration are unmeasured. Root CLAUDE.md's standalone-nebu repository URL
is deferred documentation drift; the settled source identity is recorded here.

## Executed gates

2026-10-09, Mac16,5/M4 Max, macOS26.4.1 build25E253, aarch64-apple-darwin,
Rust1.95.0 `59807616e1fa2540724bfbac14d7976d7e4a3860`, Cargo1.95.0.
Complete Git archives and separate targets under
`/tmp/honeycrisp-nebu-closure.GUEZG7` are authoritative inputs. Original dirty
honeycrisp source is excluded. Strata's edition2024/rust-version1.89 compiles
under this toolchain; consuming honeycrisp resolver2 governs the graph.

| Command (Cargo unless stated) | Exit / observed scope |
|---|---|
| Parent `metadata --format-version 1 --locked --offline` | 101: no matching package named `cyb-nebu` |
| Candidate metadata, same arguments | 0; all12 local package paths stay inside the archives |
| `tree -e features -p acpu --locked --offline` | 0; nebu `fast` absent |
| `fmt --all -- --check` | 1; untouched aruminium benchmark formatting |
| `clippy --workspace --all-targets --locked --offline -- -D warnings` | 101;16 unique source diagnostics, first `duplicated attribute` |
| `build --release --workspace --locked --offline` | 0; existing benchmark output-name collision warnings |
| `test --workspace --locked --offline` | 0;320 passed,5 ignored; existing example/FFI declaration warnings retained |
| `build -p acpu --release --example bench_zk --example bench_summary --locked --offline` | 0; links both nebu consumers; existing warnings retained; benchmarks not executed |
| `run -p acpu --example matmul --locked --offline` | 0; bounded64×64 result PASS, reported max error0 |
| `semver-checks --workspace --exclude metal-benches --exclude rane-benches --baseline-root <parent>/honeycrisp --verbose` | 0; five public crates, each202 checks pass/58 skip; API smoke only |
| Same semver command, baseline tag `v0.2.0` / `96fa4f4509d7a966cb22a4bb9e12ac753cff43ae` | 101; tag requires absent historical `../nebu/rs/Cargo.toml`; no tag-compatibility verdict |

The five ignored cases are acpu `gemm::gemv_kern::tests::t_small` and four
aruminium doctests; evidence retains the individual executed test names.
The matmul observation is a correctness smoke, with no performance claim.
Passing exits with warnings do not satisfy the zero-warning gate.

The semver tool generates scratch manifests/locks and resolves newer registry
dependencies (including crossbeam-queue0.3.14), despite the unchanged production
lock. Those final scratch files and full verbose logs are retained. Its API
comparison is distinct from the locked workspace tests. `git diff --exit-code
<parent> <implementation> -- src acpu unimem rane aruminium` exits0: all source,
specification and test bytes remain identical, independently of tool coverage.

## Diagnostic control and limits

A second archive at the same pins omits only acpu's nebu dev dependency and
refreshes its temporary lock. Identical commands reproduce all16 Clippy
diagnostics and byte-identical fmt output after path normalization. Release and
bounded matmul succeed. Its full tests and two benchmark examples fail with
E0433 for the deliberately absent nebu; these are control failures and do not
attribute any candidate error. Control changes never enter the repository.

After copying the accepted lock into the owned worktree, a convenience metadata
check also exits0. Live strata was clean at the exact pin before and after;
archive evidence remains authoritative for this unit and the following Block
unit. Registry `94af9de717658b3e912bcfccc85409fd29b23a10` places honeycrisp in
nox/layer1; rung/dependent/kelvin/release gates were not established here.

The session authorizes implementation and ordinary source merges. Versions,
release pins, tags and publishing are excluded. Red lint/format/tag gates require
a `wip:` receipt and `decision` handling; they are preserved, not waived by this
package repair. PLAN review moved from [requested corrections](plan-review-initial.txt)
to [APPROVE](plan-review-approved.txt). [CODE review](code-review-approved.txt) approved the committed manifest/lock and gate attribution.

## Reproduce

`nu reproduce.nu /path/to/honeycrisp /path/to/strata` archives the exact commits,
creates independent targets and records each gate's exit/output. It deliberately
retains the known parent/control failures; script completion means the commands
were recorded, not that all gates passed. The sources must exist in those Git
object databases. No Python or external test harness dependency is introduced.

The checked-in runner was executed against these pins under
`/tmp/nebu-identity-replay.PN072I` and completed with exit0;
[replay evidence](replay-evidence.json) preserves all gate outcomes. The repeated
workspace tests again pass320/ignore5; known red gates remain red. Initial runner
syntax/argument errors occurred before its final replay. In particular, Nu's
literal `--output=$tar` briefly created an agent-owned archive named `$tar` in the
original honeycrisp checkout. It was moved to the scratch directory immediately;
all pre-existing dirty files remained untouched and the original status was
restored. The runner now passes `--output $tar` as separate arguments.
