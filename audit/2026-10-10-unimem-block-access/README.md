# Raw Block access and Metal import contracts

Execution receipt, 2026-10-10. Source is frozen and native/dependent gates are
complete. [CODE review](code-review-approved.txt) conditionally approved the
source with one precommit requirement: preserve the moved historical material
in the repository in this same unit. This audit supplies that requirement;
its exact snippets and link target are checked before staging with the source.
No compatibility approval is claimed.

Parent Honeycrisp `7b148906263da5ea8aabc5ddb0e2e44ff2288ecc`, sibling Strata
`56aedb2d12b3126c601eb333419136d403614dbb`. The [source manifest](source-files.sha256),
SHA `1ead683ed4e75cfb0c2edd196e6e18e979f3ca83634e69c3bcdd2bfe72e1b1be`, pins
every changed file. All results below use that overlay on the exact Git archives;
the [source freeze](source-freeze.json) preserves its original observation time.

Six Block typed views require unsafe initialization/exclusion obligations; mutable views require an exclusive receiver. Warm keeps its fixed16KiB stride, and raw Metal imports require runtime page alignment, a single current VM region, retained backing and completion discipline. Constructors/allocation/drop remain unchanged. These boundaries do not establish global driver soundness, physical-address availability, permanent page pinning, native frame ownership or Kadek foundation row6 completion.

Fresh different-vendor [PLAN v3 review](plan-review-approved.txt) returned
APPROVE before implementation. [The response](plan-v3-response.md) retains
earlier review corrections. CODE review examines all safety contracts, source,
fixtures and both native and dependent evidence; no four-vendor council is claimed.

## Historical IOSurface claims

The following text is preserved verbatim from `git show 7b148906263da5ea8aabc5ddb0e2e44ff2288ecc:unimem/specs/README.md` (full source SHA6475bd489d1d3a05f0cadc0dfa66b23fb8ea1dde81c19a8b5f175c40f5f4d01f). It is historical material moved out of the contract, not a new result. Inspection found the probe source, but no original raw output, machine, invocation or measurement revision supporting these numbers. Source-history import revision1d1b866d is not a run revision. The performance projection remains unverified. Current native probe evidence must be reported separately and cannot retroactively supply provenance for this text.

```text
### Measured performance (from experiment)

| Size | Alloc | Write throughput | Read throughput |
|------|-------|-----------------|----------------|
| 4 KB | 18 us | 22.8 GB/s | 22.8 GB/s |
| 1 MB | 15 us | 23.6 GB/s | 23.5 GB/s |
| 16 MB | 17 us | 23.6 GB/s | 18.9 GB/s |
| 256 MB | 20 us | 23.1 GB/s | 19.5 GB/s |

Throughput measured with volatile u64, single thread, no SIMD. With NEON/AMX: 60-70+ GB/s expected.

| IOSurface: pinned, contiguous VM region, ~20us alloc, ~23 GB/s write | experiments/iosurface_probe |
```

The original full-source SHA metadata accidentally omitted its terminal newline
when captured by Nushell. The [old metadata](historical-snippets-before-metadata-correction.json)
and [corrected metadata](historical-snippets.json) retain that history. The exact
table and row bytes never changed. The initial source-freeze JSON deliberately
references the old metadata SHA; it does not invalidate the unchanged source files.

## Native validation

[Native summary](native/candidate-summary.md), [commands](native/candidate-results.json)
and [full receipts](native-receipts.tar.gz) record compiler binaries, hashes, environment,
source/lock identities, times and actual exits. Parent and candidate use matching
Rustup 1.95.0 compiler/Cargo/Clippy binaries; negative diagnostic-code doctests use
the actual nightly-2025-11-26 compiler and rustdoc. The native host is Mac16,5
arm64, macOS 26.4.1/25E253. No moving toolchain or dirty sibling is a build input.

| Gate on frozen source | Observed result |
|---|---|
| `test --workspace --locked` | 351 pass, 5 ignored; parent 336 pass/5 ignored. Delta is 3 native tests and 12 doctests; moved tests are not counted new. |
| `test -p unimem --release --locked` | 66 pass, none ignored; includes all new access tests and existing creation/roundtrip targets. |
| Actual nightly unimem/aruminium doctests | 14 compile-fail cases pass, including 12 new expected-code witnesses; 4 preexisting examples ignored. |
| Selected GPU raw/import and private ANE fixtures; vecadd/matmul examples | All exit 0; private ANE completion/visibility remains an explicit inherited assumption. |
| Selected library Clippy with `-D warnings -D clippy::missing_safety_doc` | Exit 0, no warnings; contract content independently reviewed. |
| Completing selected fixture lint with `-W warnings` | Exit 0; all 8 emitted preexisting warnings match the parent's complete diagnostic multiset. |
| Strict owning rustdoc, release workspace build, unimem build/bench and pipeline | All exit 0; existing output-name warnings retained. Bench output is evidence of execution, not a performance claim. |
| Whole-workspace fmt | Exit 1; same existing objc2.rs formatting output. |
| Whole-workspace strict Clippy | Exit 101; same 4 emitted acpu diagnostics, first duplicated attribute in gemm/mod.rs:1094. Later targets are masked. |
| Whole-workspace `-W` Clippy | Exit 101; reachable diagnostic sets differ with parallel compilation. New emitted warnings are source-verified preexisting; old unreported benchmark failures remain masked. |
| Explicit exact-parent unimem semver check | Exit 100: 202 checks, 200 pass/2 fail/58 skip; detects 7 unsafe additions and 4 exclusive-receiver changes. |

The initial workspace semver invocation detects newly unsafe `Gpu::wrap`, then
fails on duplicate historical manifests embedded in old audit snapshots. The
explicit unimem command resolves that ambiguity and preserves the actual API
breaks. Last-tag resolution separately fails on the unpinned historical nebu path;
there is no successful tag-compatibility verdict. Semver scratch dependency
resolution is separate from locked source build/test evidence. The native summary
details every emitted-versus-masked finding and earlier tooling correction.

## Dependent compatibility

[Parent summary](dependent-parent/README.md) and
[candidate comparison](dependent-candidate/README.md) identify the exact origin
inputs and retained diagnostic locks. The [machine-readable comparison](dependent-candidate/comparison.json)
and [verification](dependent-candidate/verification.json) retain all 37 identical
command pairs: 16 candidate exits are zero, 21 nonzero, including metadata gates.

Xena, Bita and Erga newly fail at changed unsafe or exclusive-receiver calls;
some earlier errors are masked by these compile failures. Those are intentional
API breaks requiring consumer migration, not preexisting failures. Perla's
bounded GPU suite/oracle, Zoya release suite and Glia library suite still pass;
Mir and Glia workspace failures remain. Missing origin locks, unavailable vendor
closures and incomplete isolated Cyb closure are retained as separate limitations.
Unchanged Nox evidence exercises the owning rung, not the driver API.

The [masked-call inventory](dependent-migration-inventory.json) pins additional
source paths that compilation did not reach. At Erga `848e17e`,
`rs/rtable-bench/src/lib.rs:58` imports a Block; `build_parallel(&self):76`
requests mutable bytes at 80, and `cpu_checksum(&self):106` reads at 107.
Its CLI calls both at `main.rs:52,111`. The earlier blake-bench error masks
these obligations. At Xena `6ea943a`, `xena-gpu/src/lib.rs:44–45` imports the
caller-provided Block; xena-bench has CPU views in gpuverify at 44/58, gpubench
at 22/75, cpugpu at 55, power at 147 and gpudebug at 171. Their GPU calls occur
at 55, 35/85, 65, 158 and 182 respectively. Some mutable CPU views survive
across submission and later CPU use; initialization, borrow scoping and owner
lifetime require review, beyond merely repairing the first emitted error.

Complete stdout/stderr, commands, helper scripts, source inventories, lock evidence
and their original hash manifests are retained compactly in
[parent payload](dependent-parent.tar.gz) and [candidate payload](dependent-candidate.tar.gz).
Each archive was extracted into a fresh directory and its entire `SHA256SUMS`
verified. To inspect, extract an archive into an empty directory and run
`shasum -a 256 -c SHA256SUMS` there. Readable summaries above are copies from those
payloads. The native archive preserves full raw logs, including their original
whitespace, and both native manifests verified after a fresh extraction.

## Remaining work and authorization

No full-ring or zero-warning workspace claim follows. External migrations need
initialization, retention, exclusion and completion proofs; mechanical unsafe
annotations are insufficient. Safe native image ownership/accounting, fences,
Grid/Buffer soundness and Kadek row 6 remain open. The physical-address/pinning
semantic question is unresolved and its contract was not silently weakened.

This spec-touching, breaking source change requires `decision` classification;
the owner's explicit audit/fix/merge instruction covers ordinary source work.
Versions, sibling pins, tags and releases remain owner-controlled. Any commit
while required global gates stay red uses `wip:` and quotes the first error.
Original dirty worktrees under `~/cyber/honeycrisp` and `~/cyber/kadek` are preserved.
