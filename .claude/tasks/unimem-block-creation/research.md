# Checked Block creation — research

2026-10-09 · source-only · serves: kadek foundation row 6 retained unimem
storage, with row 3 admission; upstream unimem spec § Layer 1 / error model.
No builds, gates, production edits or allocation experiments ran in this task.

## Authority and checkout

Read `~/cyber/cyberia/dev.md`, upstream `CLAUDE.md`, `unimem/CLAUDE.md`, and
`unimem/specs/{README,api-sketch}.md`. Drivers expose concrete hardware access;
specs precede code, raw FFI matches the SDK, source files stay ≤500 lines.
The FFI signature correction and compact existing-spec edits are explicit plan
items for review; dependencies, spec structure and hardware policy stay outside.

Commands: `git -C ~/cyber/honeycrisp ls-remote --symref origin HEAD`,
`git -C ~/cyber/honeycrisp fetch origin main`, then
`git -C ~/cyber/honeycrisp worktree add -b fix/unimem-block-creation
~/cyber/honeycrisp-kadek-block-creation 9593c7218e14e2b5b816d9008b45ab60b9580cf3`.
Origin main and the owned checkout are that commit; its unimem tree is
`0cfda068bf512d4ce2a6cad5b0eafb85a0591800`. Original dirty trees are excluded.
`git rev-parse v0.2.0` gives `96fa4f4509d7a966cb22a4bb9e12ac753cff43ae`.

## Current source and consumers

`unimem/src/block.rs::open` rejects zero, casts usize to i64 unchecked, creates
a mutable CF dictionary, six allocated strings and six numbers without checking
their returned pointers. The dictionary retains its entries; local Create-rule
references to keys/numbers are never released. `cf_str` also allocates a CString
and unwraps. Create/lock/base-address failures have manual cleanup; returned
extent, slice isize bounds and typed base alignment are unchecked.

`ffi.rs` declares CFNumberType as i32, whereas SDK CFNumber.h uses CFIndex.
Dictionary capacity is i64 rather than the SDK's pointer-width signed CFIndex.
`IOSurfaceCreate` takes immutable CFDictionaryRef in the SDK. Helpers are used
only by Block; raw FFI is public, so signature fixes have source compatibility
consequences. `MemError` is exhaustively matchable and re-exported by
acpu/rane/aruminium; adding precise variants is another intentional source break.

Searches: `rg -n 'MemError::|cf_str\(|cf_i64\(|Block::open' unimem rane
aruminium src -g '*.rs'`; `git grep -n -E 'unimem::ffi|use unimem::.*MemError'
9593c72 -- '*.rs'`. Named consumers: Tape derives capacity from `block.size()`;
Grid/Layout use Tape; rane hardware tests and unimem pipeline/ane_matmul use
handles; aruminium `Gpu::wrap` uses address and actual size. Existing roundtrip
tests assert actual size ≥ requested, not byte equality. No in-tree external
MemError exhaustive match or call to unimem's raw CF declarations was found.

## Primary native evidence

`xcrun --show-sdk-path` returns `/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk`.
Under `System/Library/Frameworks/`, read:

| Header / clauses | SHA-256 (`shasum -a 256 <header>`) |
|---|---|
| `IOSurface.framework/Headers/IOSurfaceRef.h`:24–38,171,212,350–390 | `8ee2890f7a5392ac2040c2e0b3d3cb773cdebe3627ad3452881fc7d20a759bd5` |
| `CoreFoundation.framework/Headers/CFNumber.h`:31–74 | `ff5cfd4c1bf54d05d495ce80c9ca1405f8ebaef8161f3e21dce590b070f342dd` |
| `CoreFoundation.framework/Headers/CFDictionary.h`:202–284,311–380 | `cde527652ea4af4304bdd53a4ff72cf2c8c5b05988ae9572218e46f2a8c0797b` |
| `CoreFoundation.framework/Headers/CFBase.h`:558–565 | `259ad5fb0c71d37737e38db6617e8373e6266c9a2482b25b7fb1eb10d28f1736` |

AllocSize names total buffer extent, including planes. Calculated BytesPerRow
should always obey its native property alignment; the current spec's row=size
needs a focused correction. PropertyMaximum describes all-device image
compatibility, not a universal raw-byte allocation limit; it must not arbitrarily
reject large existing raw buffers. CFDictionaryCreate accepts stack arrays,
copies callback structures, and retains values using those callbacks. Creating
an immutable six-entry dictionary avoids the void SetValue growth operation.
Borrowed kIOSurface constants avoid allocated string keys. Returned-null checks
do not prove that every opaque framework allocation failure returns normally.

Supporting measurement: kadek's corrected
`audit/2026-10-09-unimem-extent/{README.md,probe.rs,stdout.tsv}` in
`~/cyber/kadek-unimem-extent`, documentation base `6e1f9920a9ab629da5cec1f24ca73a93537da202`.
The receipt owns root's exact compile/run/cmp commands and 54 matching extents;
this task did not rerun them. Corrected probe SHA-256
`5db5ad6ef8035c6e8768cf2040361600ba7786938649fe5d3e41f188a4b63920`, stdout
`8d9c8a02b736da70734467216553b5746d7f45398e6e3bce63d6c6173b520a22`.
Documented total-size input/output semantics support pre-reserving the explicit
extent; post-create equality is defense in depth under that platform premise.
The counted domain is API-visible backing, plus separately charged Rust owners;
CF/kernel bookkeeping, residency and physical pages remain distinct domains.

## Scope and gate prerequisites

A checked concrete plan can stand independently as a constructor repair. It
does not reserve Kadek's pool or certify initialization/publication. A later
private writer must initialize every visible plane byte before immutable output;
this constructor adds no blanket native-initialization claim. Public mutable
views/raw handles, GPU import lifetime, completion, physical-address wording and
non-Apple/no_std support remain separate. Earlier decomposition is in
`/tmp/kadek-unimem-integration-plan.md`; its conservative extent uncertainty is
superseded by the corrected measurement's stated platform interpretation.

Fresh `git -C ~/cyber/strata ls-remote --symref origin HEAD` gives main
`56aedb2d12b3126c601eb333419136d403614dbb`; `git show <pin>:nebu/rs/Cargo.toml`
names `strata-nebu`, while honeycrisp Cargo.toml requires `cyb-nebu` at that path.
This source mismatch prevents calling full origin workspace closure ready; no
build failure was executed here. Isolated exact-source unimem gates are useful
but cannot replace the required workspace/dependent gates. Dependency alignment
needs its own reviewed unit, without dirty sibling inputs or compatibility shims.
Committed soft3 registry `94af9de717658b3e912bcfccc85409fd29b23a10`, read with
`git show <pin>:release/components.toml`, places honeycrisp under nox, layer 1;
this is a pinned registry read, not a fresh soft3-origin or dependent gate claim.
