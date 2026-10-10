# Raw Block access — research

2026-10-09/10 · serves Kadek foundation row6's native ownership prerequisite and
unimem's raw CPU access contract. No production edits or candidate gates.
Initial source research is followed by explicitly delegated exact-parent gates
below. [Plan](plan.md) awaits a fresh vendor verdict; root owns review invocation.

## Authority and base

Owned checkout `/Users/master/cyber/honeycrisp-kadek-block-access`, branch
`fix/unimem-block-access`, HEAD/origin/main
`7b148906263da5ea8aabc5ddb0e2e44ff2288ecc` (Block PR13 merged23:59:02Z).
Root fast-forwarded from implementation `6715335a3283e22218ad5f762316f2c735acb56f`;
`git diff --exit-code 6715335 HEAD -- Cargo.toml Cargo.lock src unimem acpu rane
aruminium` returns0: only parent task/audit artifacts were added. Nebu identity
is already repaired by `6560974dcd1a77b4b2e040a2b2a4187b702ace0b`, with strata
`56aedb2d12b3126c601eb333419136d403614dbb`. The earlier scratch note's 9593c72
base and unresolved-nebu assertion are stale.

Read `~/cyber/cyberia/dev.md`, root/unimem/aruminium/rane `CLAUDE.md`, native
specs, source and the frozen parent task's research/plan in
`~/cyber/honeycrisp-kadek-block-creation/.claude/tasks/unimem-block-creation/`.
Preserve those artifacts as parent evidence. The merged package repair receipt
is `audit/2026-10-09-nebu-package-identity/README.md`; the completed Block receipt
is `audit/2026-10-09-unimem-block-creation/README.md`. These distinguish passing
native/workspace execution from pre-existing red fmt/lint gates. Initial source
research ran no measurements; the later parent-only receipt below records its
own commands/compiler and does not replace those historical receipts.

Commands: `git status --short`, `git rev-parse HEAD origin/main`, `git log -4
--oneline`, `rg -n 'as_(bytes|f32|u16)(_mut)?\(|\.warm\(|\.wrap\(' --glob
'*.rs'`, scoped `cat`/`sed -n`; external `git ls-remote origin HEAD`, then
`git grep -n -E ... <pin> -- '*.toml' '*.rs'` and `git show <pin>:<path>`.
All sibling source is immutable Git content, never dirty build input.

## Actual access graph and contract holes

`unimem/src/block.rs:114–220`: Block owns/locks/releases an IOSurface and is
Send+Sync. All six byte/u16/f32 slice accessors are safe; the mutable three take
`&self` with `mut_from_ref` suppressed. Shared calls can create overlapping Rust
references. Checked creation proves pointer alignment, extent and lifetime,
not initial byte values, alias exclusion or CPU/device completion. Creation locks
once and drop unlocks once; no lock/unlock occurs around a typed view, so Block
performs no per-access synchronization. This is a source claim, not a universal
statement about IOSurfaceLock's API semantics.
The typed slices cover actual `size()`, `size()/2*2` or `size()/4*4` bytes,
including native padding, not just requested bytes. Forming a whole typed view
then slicing its initialized prefix does not establish the whole-view contract.
Rust's primary [from_raw_parts](https://doc.rust-lang.org/std/slice/fn.from_raw_parts.html)
and [mutable counterpart](https://doc.rust-lang.org/std/slice/fn.from_raw_parts_mut.html)
require initialized values throughout the returned slice and preserve their
respective shared/exclusive access requirements; checked pointer arithmetic alone
does not satisfy them. Both primary pages were read during this research.

`tape.rs:35–55,127`: `warm(&self)` writes one zero byte at a fixed16KiB stride while
`block()` exposes the same backing owner. These offsets do not imply the runtime
VM page size, and warming does not initialize all
bytes. Raw takes outlive a Rust borrow; `&mut Tape` alone cannot express every
external alias/device duty. Safe `start_warm` has the stronger fresh-owner proof.
`clear(&self)` changes only the cursor; it establishes no retained output owner.

`grid.rs:75–95`: Grid exposes `&Tape`; Cell exposes a raw pointer and an already
unsafe `bytes(&mut self,len)`. The latter's contract says bounded/exclusive but
omits initialization. No safe typed view should be inferred through this route.
Independent gaps remain: `Grid::new` multiplies constants unchecked;
`give(Cell)` does not establish the cell's originating grid, and safe `take`
uses the returned index in pointer arithmetic. A foreign high index can exceed
the receiving allocation. These need a separate ownership/arithmetic repair;
this task must not claim all safe Grid operations are sound.

`aruminium/src/device.rs:137–177`: `buffer_wrap` is already unsafe, while
`wrap(&Block)` is safe and returns a Buffer with no Block lifetime/retained owner.
Block must outlive imports and all encoded/in-flight accesses. Buffer's own safe
closure read/write methods and `as_bytes` (`buffer.rs:69–143`) separately permit
aliasing or in-flight reads; this unit marks the import boundary, not a full GPU
API repair. Commands wait/status/error exist (`command.rs:193–219`), but no fence
is attached to Block. `rane/src/model.rs:170–201` contains already-unsafe
`Program::run_direct`, its `build_request` is at342–402 and low-level Objective-C
bindings are in `rane/src/ffi.rs`. The private
`evaluateWithQoS:options:request:error:` call returns a success flag; the existing
caller assumes evaluation and memory visibility are complete on success.
No public documented Apple ANE completion specification was established here;
retain that dependency assumption explicitly when migrating its existing test.

### Metal page and VM-region compatibility

Apple's [no-copy buffer contract](https://developer.apple.com/documentation/metal/mtldevice/makebuffer(bytesnocopy:length:options:deallocator:))
requires page alignment of both pointer and imported extent, and confinement to
one VM region. This is stronger than BlockPlan's IOSurface property alignment:
creation checks exact returned size and lane alignment, not VM pages. The existing
vecadd fixture requests1024bytes and therefore lacks this proof; a previous
successful run cannot waive the documented precondition. The import must never
round length beyond the actual owned extent. Change the fixture's allocation
request, not BlockPlan/open or the imported length.

Apple's [getpagesize](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/getpagesize.3.html)
returns the process/system VM page size, distinct from an assumed hardware page.
Use a test-local C declaration and check its positive result. Neither the public
[base-address](https://developer.apple.com/documentation/iosurface/iosurfacegetbaseaddress(_:))
nor [allocation-size](https://developer.apple.com/documentation/iosurface/iosurfacegetallocsize(_:))
page establishes a universal single-region IOSurface guarantee; none is claimed.
For native fixtures, query and check the full actual mapping before import with
`mach_vm_region_recurse`, whose SDK declaration and pinned XNU implementation
are cited below.
The test keeps its fresh owner live and performs no remapping after the query.

SDK26.4.1 (`xcrun --show-sdk-version`), real path
`/Library/Developer/CommandLineTools/SDKs/MacOSX26.4.sdk`, pins this test-only FFI:
`unistd.h:572` getpagesize; `mach/mach_init.h:80–81` task port global;
`mach/mach_vm.h:299–313` recurse signature;
`mach/vm_region.h:54,298–324` packed4 short-info structure/count;
`mach/vm_prot.h:87–88` RW flags. The short structure is48bytes/12integer words,
with is_submap at byte32; ABI assertions protect the Rust test mirror. Passing
depth=u32::MAX reaches a leaf; require is_submap=0, returned count12, RW access,
and checked address+size containment. RW is required by these read/write fixtures;
it is not an added universal Metal import precondition. The recurse API creates no object-name
send right. This is a current mapping check, not an ownership or completion fence.
Apple XNU `f6217f891ac0bb64f3d375211650a4c1ff8ca1ea`,
[osfmk/vm/vm_map.c](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/vm/vm_map.c):15150–15180,15304–15312,15485–15514,
independently supplies
short-info count, maximum-depth, leaf and returned-range semantics.

Full primary content and acquisition commands/checksums are frozen in
`/tmp/kadek-unimem-block-access-primary/{sources.json,sdk-inputs.json,SHA256SUMS}`.
SHA256SUMS hash: `e386277ba595b3e3ece2b736815acc635853d8608a427ec94a3f05e8c0469013`.
The pack contains Metal/IOSurface Markdown, getpagesize HTML, exact SDK headers
and the pinned full XNU file; two failed Mach Markdown fetches are recorded,
with SDK/XNU primary evidence used instead. No probe/build was run during this
primary-source acquisition; later parent-only gates are recorded separately.
The separate `REVIEW-SHA256SUMS` selects full no-copy/IOSurface pages, the complete
getpagesize manual body, complete XNU recurse function and SDK ABI excerpts with
full-file hashes and inclusive ranges. This compact review selection preserves
the original full archive and does not truncate any reviewed function body.

### Shared Metal visibility and historical spec correction

Apple's [CPU/GPU synchronization sample](https://developer.apple.com/documentation/metal/synchronizing-cpu-and-gpu-work)
requires CPU writes to finish before committing commands that reference the shared
buffer, and excludes CPU reuse until the relevant GPU work completes. Its ordered
single-thread write/encode/commit path is the fixture's CPU→GPU proof. Apple's
[compute sample](https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu)
uses storageModeShared, commits then waits, and states completed results are CPU
visible. [waitUntilCompleted](https://developer.apple.com/documentation/metal/mtlcommandbuffer/waituntilcompleted())
waits for GPU commands and completion handlers. These public API rules supersede
`buffer.rs:55–58`'s unsupported standalone `dmb ish` prescription. A CPU barrier
alone establishes neither GPU completion nor Rust alias exclusion. Update only
contents_ptr's documentation; its raw-pointer signature and Buffer behavior stay.
This concerns current MTLCommandQueue/MTLCommandBuffer shared storage, not Metal4
queues, managed/private storage, or an ANE ordering guarantee.

Full new primary Markdown and curl commands/checksums: cache `sources-v2.json`,
`metal-cpu-gpu-sync.md`, `metal-compute.md`, `metal-wait.md`.
`REVIEW-V2-SHA256SUMS` selects the original inputs, full wait page, complete
relevant article sections with inclusive line ranges/full-source hashes, and the
compact complete caller inventory. SHA256:
`09d22d36516f1f26c7daef7c1c9c386407cf3f19a3bfea361d21e1508f918f40`.
Full acquisition files remain cached; no contained sample method is truncated.
Full Apple/SDK/XNU content remains a temporary acquisition cache. The later public
audit retains URLs, revisions, hashes and original project probes, not those pages.

At base7b148906, `unimem/specs/README.md:143,171–172,203` and
`api-sketch.md:89,160`, plus unimem/CLAUDE.md's immutable-Block gotcha, incorrectly
infer immutable storage/no races from owner
fields or lock lifetime. README:171 universally promises one VM region; :460
reports it with ~20us/~23GB/s and a probe pointer. Inspecting the entire tracked
probe directory finds only Cargo.toml/lock and source, no raw result or machine/run
receipt. Source `experiments/iosurface_probe/src/main.rs:452–531,690–741` queries
4KB/1MB/16MB/256MB, but that source alone proves no observed outcome. Git history
records its import at1d1b866dd45b991b73d1f28b9bd3f736b5fcd65d; that is not a run
revision. Layer1's “Measured performance (from experiment)” table repeats these
four sizes/numbers without run provenance. Move both the :460 row and that table
with its following throughput/projection sentence verbatim to the owner audit
as historical reported claims with missing machine/invocation/run revision;
remove the universal invariant and link the scoped evidence. Other unrelated
performance sections stay outside this access repair. No fresh measurement is
substituted for either historical claim.
The adjacent CLAUDE.md:63 page-size gotcha needs the same fixed-stride/runtime-page
distinction as README:240; no broader instruction-file cleanup is planned.
Parent's separately reported C ABI and region probes are follow-up evidence with
their own commands/source hashes, not provenance for the historical row.

## Every direct in-repository caller found

| File / function | Existing initialization and required proof |
|---|---|
| unimem/tests/roundtrip.rs, block_slice_lengths / block_slice_pointer | No initialization; initialize full actual size via raw bounded write before constructing any view, even for len/pointer checks |
| same, block_cross_view_f32_bytes / block_cross_view_u16_bytes | Only first element assigned through full mutable slice; initialize actual backing first, then sequential nonoverlapping views |
| same, tape_warm_after_start | Fresh tape, no takes yet; mutable binding and justified unsafe warm; start_warm proves freshness internally |
| same, cell_bytes | First/last bytes only, whole slice already formed; initialize selected256-byte cell extent through raw pointer before unsafe view |
| unimem/tests/block_creation.rs, native_alignment_extent_and_raw_row_boundaries | Already writes every actual byte 0xA5 before all three read views; retain oracle, add justified unsafe calls |
| aruminium/tests/integration.rs, wrap_block_vecadd | A/B filled via full typed mutable views before initialization; C GPU writes only256 floats, native padding remains unproved. Zero entire actual A/B/C via raw writes, then fill A/B; end views before import; scoped buffers/commands stay alive through wait and status check; read C only after completion |
| same, gpu_buffer_wrap | Hardcoded16KiB manual allocation then uninitialized f32 slice; use owning page-sized Block, runtime page/leaf-region preflight and actual-extent raw initialization before direct buffer_wrap; retain owner through buffer drop |
| rane/tests/hardware.rs, run_direct_with_unimem_block | Input writes program extent only; output and padding uninitialized. Zero full actual input/output before views; end input borrow before run_direct; retain owners through synchronous evaluation and read output only after successful return |

No production caller of the changed Block slice methods or `Gpu::wrap` was
found in this checkout beyond their definitions. `Tape::start_warm` is a
production internal caller; Layout, benches/examples call that safe constructor.
Root facade, aruminium and rane re-export Block; source changes reach dependents.
Raw-pointer tests initializing only endpoints remain valid because they read
only those endpoints and never form whole typed slices.

Complete scan is `/tmp/unimem-block-access-caller-inventory.json`, generated by
`/tmp/unimem-block-access-v2-research.nu`: `git ls-tree -r --name-only 7b148906`,
then `git show <pin>:<every tracked .rs>` and the bare-name expression
`\b(as_bytes|as_bytes_mut|as_f32|as_f32_mut|as_u16|as_u16_mut|warm|wrap)\b`.
All157 tracked Rust files have a hash, line count, lexical matches and explicit
changed-call count, including zero. Manual type classification yields26 calls
in the five files above, including the warm invocation INSIDE start_warm at
tape.rs:37; calls TO unchanged start_warm are excluded. The bare-name regex cannot
match start_warm because underscore is a word character. Definitions/comments and Buffer's
unrelated as_bytes are excluded; rane/model.rs:44 and probe/compile.rs:153 call
str::as_bytes. Broad bare names also cover associated calls/function references.
Private block/tests.rs, aruminium/src/tests.rs and render/tests.rs, every example,
bench member, probe module and experiment have zero changed-API calls. The review
selection contains full private block/aruminium tests and unimem ffi, plus full
rane/model.rs and probe/compile.rs to resolve the two str::as_bytes matches.
Unmodified zero-call examples/bench/probe/experiment bodies are represented by
the complete per-file inventory/commands/hashes; complete source stays available
at the pinned commit. No local Rust file is excerpted. Inventory includes the
historical IOSurface probe's exact SHA; its body is omitted for context size.
No example/bench migration is planned;
only inspected test fixtures need added initialization.
Inventory SHA256: `a066fde1cec16f9a0c1613e2efc232ab852d621da4ba9740aa06a2c0668fdef6`.

Separate exact-source `\bstart_warm\b` scan across the same157 Rust files finds
five unchanged constructor calls: unimem/benches/alloc.rs:7,
benches/bandwidth.rs:41, examples/pipeline.rs:267, src/layout.rs:28 and
tests/roundtrip.rs:271; src/tape.rs:35 is its definition. Commands, every source
hash and explicit zero/nonzero matches are in `/tmp/unimem-block-access-start-warm.json`,
generated by the adjacent `.nu` script. This corrects v2 review's suggested two
call sites and stale line numbers with committed source evidence.

## Exact-parent lint reachability and gates

Root delegated parent-only validation while PLANv2 was frozen. Immutable
7b148906/strata56aedb2 archives and preserved lock, rustup1.95.0/rustc59807616,
LLVM22.1.2, jobs4; `/tmp/unimem-block-access-validation-6eQqO1/parent-baseline/README.md`
and `final.json` record21 receipts including two superseded tool-path attempts.
The original fmt/Clippy selected Homebrew subcommands; Clippy failed E0514.
Prepending the chosen toolchain bin fixed resolution; actual paths/hashes/versions
are retained. Homebrew's LLVM22.1.3 is a different toolchain despite the same
rustc commit. No source/lock change, candidate overlay or candidate gate occurred.

Pinned fmt remains red; workspace -D warnings and selected packages without
--no-deps stop on four emitted acpu diagnostics, possibly masking other findings.
Selected all-targets --no-deps reaches further but hits existing test lints and
benchmark imports. Full workspace -W warnings also exits101: four emitted
warnings,36 E0433 benchmark errors and two failure notes. Thus neither suggested
all-targets alternative alone establishes C1's required lint coverage.
The actual strict selected-library --lib --no-deps -Dwarnings
-Dclippy::missing_safety_doc exits0 and reaches both unimem and aruminium.
Selected unimem/aruminium/rane --tests --no-deps -Wwarnings exits0 with eight
existing needless_range_loop warnings. These complementary candidate gates must
reach all eight newly unsafe methods and changed fixture modules. Keep full -D
and -W gates red/attributed; compare complete emitted diagnostics and masking,
not only exits. No new allow or unrelated lint/benchmark repair is authorized.

Workspace tests passed336/ignored5; release build, strict docs, existing native
filters, unimem release/build/bench, pipeline, named vecadd and bounded64³ matmul
all exit0. Exact commands/raw outputs are retained; these are execution results,
not safety or performance proofs. Parent/candidate archives remain unchanged:
450honeycrisp +244strata files per archive; no extra source paths. Final live
target was1849676KiB. `PARENT-BASELINE-SHA256SUMS` SHA256:
`e620a15100c4f5205d5c470de6852c9f8a8fcb456ce1914cc7c0e799a78f022b`.
Raw JSON diagnostics plus `parent-baseline/diagnostics.nu` support exact emitted
message multiset comparison (duplicate counts preserved; coordinate shifts
ignored only in comparison keys). Source coordinates remain in diagnostics.json.
The v2 task docs/brief/inputs/response/verdict are preserved in
`/tmp/unimem-block-access-plan-v2-archive`; its SHA256SUMS hashes to
`1cea80a27c66f99ef5bc1143bebcb25f0b1f962ef21288f7a13fd43327d8fca8`.
Baseline final.json records that still-frozen v2 plan, before these corrections.
Primary-cache `REVIEW-V3-SHA256SUMS` preserves every v2 input and adds the compact
complete start_warm scan/procedure and actual C1 gate evidence/comparison script;
SHA256 `2cc06f416881bf4593c6c96fb2e719f05e9cfb642ba60215eafe0fe4f2b2e422`.

## Fresh external inventory

Fresh remote HEAD pins and immutable manifest scan (`unimem|aruminium|honeycrisp`):

| Repo | Pin | Relevant result |
|---|---|---|
| cyb | 4e4711f1ba4de640e60e29304b6180c94f20c9d7 | Direct path dependencies; no changed accessor/import caller found |
| mir | b9c832d5011d5a719bec6021afef123be35490ab | Direct path dependencies; changed calls below |
| evy | 2f23bfb97fe2904e3bcba3f116b473ed6cbec483 | Direct dependencies; probe only calls unchanged Block::open; bbg backend path below |
| bbg | a3c9a0c22bfb78ef891265f9e2cd25838ef77c2d | Dormant unimem storage source; backend-unimem feature/dependency unwired |
| strata | 56aedb2d12b3126c601eb333419136d403614dbb | nebu registry acpu path feature stays off; no direct changed API caller |
| soft3 | 94af9de717658b3e912bcfccc85409fd29b23a10 | Component/release registry only |
| spark | 5399d36e99637a9d36018ecf88f3d4d6a80cc9a9 | No matching manifest dependency |
| file | fc666a5880ff874adae655150414b3a45832baf0 | Same |
| prysm | f66cbd82cada5f5e6315b674818bce705b962eb6 | Same |
| optica | 9f0c352cfea30e1ac8ce0feb0d735ffb9fa47ace | Same |
| wysm | bde62d57e4ed2a89fae475ea9e216aeffbd6deb8 | Same |
| lytics | 02c98495deda66c5f3c69ba3d5be650ce658e313 | Same |
| cyber | e5a0b80708d1caffcfedd4428161edd95d573a4f | Same |
| true-cyber | b58589f130f4f80dd8b2752f98704842372aa2d9 | Same |
| bita | cff2b0af4dd0c24c996cdde7283fff04d7e3c6e7 | Direct workspace/optional rbtc-hash5 dependencies; changed calls below |
| erga | 848e17e2522db2f6225ad72d9b1797f8735d26d0 | CLI and benchmark/miner dependencies; changed calls below |
| glia | 89adc925c47205ba8ce0438ffb984f3e1466a92a | run/Cargo.toml aruminium; no changed API caller found |
| nika | 3dfb67415a1b2420300d6f114fad64286f22cc3e | Direct unimem/aruminium; no changed API caller found |
| perla | d06b980d3f29ee829ba0fd264d67581cf79b75f6 | Direct aruminium; no changed API caller found |
| quanta | c7838cbc86f7d74477da1771139f377a8e438072 | Local discovery match absent from committed origin manifests; no dirty input used |
| trisha | 3cb3984eca9e6a1e71cb9fd8e2cc7ea18913fd2d | honeycrisp/Cargo.toml optional aruminium; no changed API caller found |
| xena | 6ea943a2ea3c613e071543cf712287dd087532ba | Direct unimem/aruminium; changed calls below |
| zoya | ef8f582f2562a61c1e5f4adb46f0651aa6b86bbf | crates/progpow aruminium; no changed API caller found |

The additional repositories were discovered with `rg -l --glob Cargo.toml
'honeycrisp/(unimem|aruminium)|^unimem[[:space:]]*=|^aruminium[[:space:]]*='
/Users/master/cyber`, then separately pinned by `git ls-remote origin HEAD`
and inspected with `git grep/show <pin>`. Filesystem discovery is not source
authority; duplicate worktrees were collapsed. Type-resolved follow-up excludes
unrelated Bitcoin Block and glia Tensor methods from changed-call results.

Mir Apple arm re-exports aruminium. Non-Apple `src/gpu.rs:129–143` initializes
via as_f32_mut, and `wrap` at700–708 uses whole as_bytes in its fallback. Imports
retain Vulkan memory/device, not Block. It also expects Block::alloc_size and
Android support absent from pinned upstream. An unsafe annotation alone would
not repair those ownership/platform gaps. This task does not patch mir.

BBG `rs/src/storage/unimem.rs`: four mutable calls (124,138,167,181); immutable
calls in get/iter/tests (86,105,241,256,344). Pool writes initialize one slot,
per-entry writes initialize count*8, neither proves the entire Block view.
It exposes Block and has separate count/capacity concerns. Its cfg is declared
for linting but feature/dependency are not wired (`rs/Cargo.toml:30–34`), so no
executed-feature compatibility claim is possible. Do not mechanically wrap calls
or enable the backend as part of this unit. Fresh inventory is a named source
closure, not all external ecosystem users or a passing ring1 gate.

Erga `rs/blake-bench/src/lib.rs:98–115` wraps an uninitialized Block and exposes
safe mutable bytes through `&self`; `rs/rtable-bench/src/lib.rs:57–58,80,111`
does likewise before parallel table writes. Both declare imported Buffer before
Block so field drop order is appropriate, but this does not establish full-view
initialization, exclusion or completion. Xena `crates/xena-hash/src/scratchpad.rs:
24–42` forms whole byte views before scratchpad initialization; `crates/xena-gpu/
src/lib.rs:45–63` wraps, submits and waits but accepts a shared Block. Benchmark
direct wraps occur in `crates/xena-bench/src/{cpugpu,gpubench,gpudebug,gpuverify,
power}.rs`. These need reviewed caller contracts, not mechanical unsafe blocks.

Bita `crates/rbtc-hash5/src/metal/mining.rs:86–93,130,170,210,252,339–345`
wraps Blocks, forms whole mutable slices then initializes only the counter or
80-byte header, and exposes async GPU futures independently of backing ownership.
Block fields precede their Buffer imports in the struct. Lifetime/exclusion and
full initialization require an independent consumer repair; annotation alone is
insufficient. No external code or feature activation is in this unit.

For later ring receipts, separate causal diagnostics from caller safety proofs:

| Consumer | Change-caused compiler obligation | Independent existing safety/build gap |
|---|---|---|
| xena scratchpad + GPU/bench wraps | unsafe-call diagnostic; mutable scratchpad receivers already &mut | Whole-view initialization and import lifetime/exclusion still need proof; adding unsafe alone is not a sound repair |
| erga ZeroCopyBuf/RTable | unsafe calls plus mutable Block borrow through &self; receiver/ownership redesign | Full extent initialization and parallel exclusion remain unproved |
| bita mining | unsafe calls plus &self mutable access; receiver/ownership redesign | Partial init, async ownership and field drop ordering remain unproved |
| mir non-Apple gpu | New unsafe calls are a separate source regression | Missing alloc_size/platform support already blocks compilation and may mask new diagnostics |
| bbg dormant backend | Unsafe/receiver obligations apply if wired; latent source break | Feature/dependency unwired at parent; no executed feature claim |

For every attempted consumer, compare exact parent and candidate under the same
compiler/feature/closure, record each first diagnostic and whether it masks later
calls, and label new/regressed versus reproduced pre-existing failure per path.
No external unsafe annotation is accepted as proof or applied by this task.

## Exact existing review inputs

All local paths below exist at7b148906 except the two owned task docs. Planned
new `unimem/tests/block_access.rs` and `aruminium/tests/support/import_memory.rs`
are intentionally not existing inputs. Add the selected primary pack and complete
caller inventory above. Zero-call bodies omitted under that inventory remain
unmodified pinned source; this list is the review selection, not the scan's scope.

```text
.claude/tasks/unimem-block-access/research.md
.claude/tasks/unimem-block-access/plan.md
CLAUDE.md
Cargo.toml
Cargo.lock
src/lib.rs
unimem/CLAUDE.md
unimem/Cargo.toml
unimem/src/block.rs
unimem/src/block/creation.rs
unimem/src/block/tests.rs
unimem/src/ffi.rs
unimem/src/tape.rs
unimem/src/grid.rs
unimem/src/lib.rs
unimem/src/layout.rs
unimem/tests/roundtrip.rs
unimem/tests/block_creation.rs
unimem/specs/README.md
unimem/specs/api-sketch.md
aruminium/CLAUDE.md
aruminium/Cargo.toml
aruminium/README.md
aruminium/specs/README.md
aruminium/src/device.rs
aruminium/src/buffer.rs
aruminium/src/command.rs
aruminium/src/encoder.rs
aruminium/src/dispatch.rs
aruminium/src/sync.rs
aruminium/src/lib.rs
aruminium/src/tests.rs
aruminium/src/ffi/mod.rs
aruminium/src/ffi/selectors.rs
aruminium/src/ffi/trampoline.rs
aruminium/tests/integration.rs
rane/CLAUDE.md
rane/Cargo.toml
rane/src/model.rs
rane/src/surface.rs
rane/src/ffi.rs
rane/src/lib.rs
rane/tests/hardware.rs
rane/src/probe/compile.rs
.claude/tasks/unimem-block-creation/research.md
.claude/tasks/unimem-block-creation/plan.md
audit/2026-10-09-unimem-block-creation/README.md
audit/2026-10-09-nebu-package-identity/README.md
```

External primary caller inputs, obtained by `git show <table-pin>:<path>`:
mir `Cargo.toml`, `src/gpu.rs`; bbg `rs/Cargo.toml`,
`rs/src/storage/unimem.rs`; erga `rs/blake-bench/{Cargo.toml,src/lib.rs}`,
`rs/rtable-bench/{Cargo.toml,src/lib.rs}`; xena `Cargo.toml`,
`crates/xena-gpu/{Cargo.toml,src/lib.rs}`,
`crates/xena-hash/{Cargo.toml,src/scratchpad.rs}`; bita `Cargo.toml`,
`crates/rbtc-hash5/{Cargo.toml,src/metal/mining.rs}`. No external source is edited.

## Scope limits

No zeroing in Block::open/plan, allocation semantic change, new owner/lease API,
pool reservation, immutable publication, device fence, portable backend or Kadek
production edit. CPU VA stays CPU VA. Retention, foreign Grid cells, Tape argument
arithmetic and safe GPU/ANE wrapper concurrency remain separately identified
gaps. The only initialization added here belongs to inspected tests that actually
read or form typed references; it is not a native allocator guarantee.
