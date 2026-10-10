# Raw Block access — PLAN v3 candidate

Serves Kadek foundation row6/native ownership prerequisite; upstream unimem CPU
access and aruminium import contracts. Base
`7b148906263da5ea8aabc5ddb0e2e44ff2288ecc`; [research](research.md) records parent
and external pins. Docs only, pending fresh different-vendor PLAN approval.
Parent Block PR13 and nebu identity are merged; no production authorization yet.

## Exact API and safety boundary

Preserve lazy BlockPlan/open, size/layout/error behavior and raw getters. No
new owner type, convenience constructor, eager initialization or dependency.
Change these existing signatures without aliases:

```text
impl Block {
    pub unsafe fn as_bytes(&self) -> &[u8];
    pub unsafe fn as_bytes_mut(&mut self) -> &mut [u8];
    pub unsafe fn as_f32(&self) -> &[f32];
    pub unsafe fn as_f32_mut(&mut self) -> &mut [f32];
    pub unsafe fn as_u16(&self) -> &[u16];
    pub unsafe fn as_u16_mut(&mut self) -> &mut [u16];
}
impl Tape {
    pub unsafe fn warm(&mut self);
    pub fn start_warm(size: usize) -> Result<Self, MemError>; // unchanged
}
impl Gpu {
    pub unsafe fn wrap(&self, block: &unimem::Block) -> Result<Buffer, GpuError>;
}
```

All six typed views require initialized bytes throughout their returned extent,
including native padding. Byte extent=size; typed extent=floor(size/sizeof(T))*
sizeof(T), with trailing partial bytes omitted as before. Block creation proves
alignment/representability; all u8/u16/f32 bit patterns are valid once initialized.
Reads require no overlapping write for the reference lifetime. Mutable views
require exclusive access against every CPU/raw/native alias for that lifetime.
The caller establishes prior device completion and visibility before CPU access,
and prevents device access conflicting with live Rust references. This holds
across threads and independently retained imports. Block holds its creation-time
IOSurfaceLock until drop; no lock/unlock occurs around a typed view, so Block
performs no per-access synchronization.
Constructing a view to call len/as_ptr, or slicing it afterward, does not waive
initialization. Initialize through raw bounded writes before forming the view.

`warm` writes one zero byte at offsets0,16384,32768,… below size, a fixed16KiB
stride independent of runtime VM page size; it requires no conflicting
references, raw-pointer users or device accesses during those writes, and no
live typed value whose validity that overwrite would violate. It neither reads
uninitialized bytes nor initializes the whole allocation. `start_warm` owns a
fresh, unpublished Tape and safely calls unsafe warm under that proof. Retain
the existing byte positions, lazy behavior and `#[mutants::skip]` annotation;
no page-policy change.

`wrap` requires its actual address and actual `size()` to satisfy the current
process's VM page alignment, with the entire imported extent inside one VM
region. BlockPlan's IOSurface property alignment does not establish that Metal
precondition. Pass actual size unchanged: never round an import beyond backing,
truncate it or change Block allocation policy. Incompatible Blocks are outside
this unsafe call's contract, not promised a recoverable native error.
It also requires Block alive until the returned import and all submitted/encoded
uses finish; initialize all bytes that any import CPU view may expose, establish
CPU/device exclusion and visibility, and keep no conflicting Rust references.
For these shared Metal buffers, finish CPU writes before committing commands
that reference them; exclude conflicting CPU access until those GPU commands
finish. Before CPU readback, wait for command-buffer completion and verify success.
The fixture uses same-thread write/encode/commit ordering and submit/wait, as in
Apple's primary samples. Supersede contents_ptr's unsupported `dmb ish` rule with
this shared-storage API ordering, retaining raw-pointer lifetime/init/alias duties.
A bare CPU barrier does not supply device completion or legal Rust aliasing.
The wrapper retains no Block and performs no wait. Preserve native error paths.
Expand existing `buffer_wrap` safety docs with the same alias/init duties; its
signature stays unsafe. Do not make Buffer's independent safe APIs sound by
assertion or claim that returning from an import synchronizes work.

Raw pointers/handles stay safe to obtain and unsafe to dereference/submit.
Replace misleading Send/Sync comments: sharing/moving the owner is supported;
memory access still requires these unsafe contracts. No refcount/fence is added.
Clarify already-unsafe `Cell::bytes` needs initialized len bytes and excludes all
aliases/device access; keep its signature. Grid identity/arithmetic stays open.

## Implementation with decisive verification

1. Compact existing contracts before code → verify exact signatures/duties in
   unimem specs README/API sketch, aruminium specs README and wrap README
   example. Replace “immutable/no data races because locked” wording; consolidate
   repeated rules. Preserve historical evidence in the audit; claim no new timings.
   Replace Block's thread-safety paragraph with: “Block is Send+Sync as an owning
   mapping. Typed CPU views are unsafe: initialize their entire extent, including
   padding, and exclude conflicting CPU/device accesses for each borrow. Mutable
   views require &mut Block. The creation-time lock lasts until drop; Block
   performs no lock/unlock or synchronization around typed access.”
   Update every duplicate: README:143 thread safety, :172 Block invariant, :203
   Tape thread safety, API-sketch:89/160 Send+Sync comments and only the matching
   unimem/CLAUDE.md gotcha (owning mapping, unsafe access obligations). Its adjacent
   page-size gotcha gets only the same fixed-stride/runtime-page distinction.
   Replace README:171's universal single-VM-region guarantee with the raw import caller's
   per-mapping proof duty. Move README:460's IOSurface experiment row verbatim to
   the owner audit, together with Layer1's duplicate “Measured performance (from
   experiment)” table and its throughput/projection sentence. Link that receipt
   and record the actual historical limits:
   probe source enumerates4KB/1MB/16MB/256MB, but no machine/raw output/run command
   or run revision was found. Source history is not measurement provenance.
   The parent's new C probes get separate follow-up receipts. Replace README:240's
   universal16KB claim with fixed-stride warm semantics and runtime import pages.
   Replace Tape's immutable-Block claim with: “Tape's cursor is atomic. Raw takes
   retain no borrow or initialization proof; unsafe warm requires exclusive
   access and writes one byte at a fixed16KiB stride. Safe start_warm operates
   before exposure.”
   Put the exact six view signatures in the existing Block API sketch, the two
   warm signatures in Tape's sketch, and one shared safety paragraph rather than
   duplicating it six times in specs only. Mark the aruminium wrap row unsafe and
   replace its lifetime sentence with the complete retained/in-flight obligation.
   Retain raw buffer_wrap's page requirement, replace hardcoded page numbers with
   runtime VM page size, and add the single-region condition to both import APIs.
   README's import sketch must start from an explicitly compatible initialized
   backing supplied by its caller, rather than imply Block::open(n*4) suffices.
2. Change receivers/unsafe qualifiers and proof comments → verify no remaining
   safe Block typed view, no mut_from_ref allowance, fresh start_warm internal
   proof, unchanged constructor/drop/native error behavior. Explicit unsafe
   blocks around pointer operations remain even inside unsafe functions. Each of
   all eight newly unsafe public functions has its own rustdoc `# Safety` section
   with its applicable duties; no missing_safety_doc suppression. buffer_wrap
   retains/expands its own section. contents_ptr changes documentation only,
   with the same public Metal ordering in both directions and no barrier claim.
3. Migrate every in-tree caller from the research table → verify actual-size
   initialization, borrow termination, ownership and native completion per call:
   - Typed-view tests initialize `address().write_bytes(0,size())` first. Keep
     literal cross-view byte expectations and full extent/padding assertions.
   - Cell test initializes its256-byte extent before forming its unsafe slice.
   - Block creation test already initializes all actual bytes; preserve that
     native oracle. Endpoint-only raw tests stay endpoint-only.
   - GPU fixtures obtain page size via test-local Darwin `getpagesize() -> c_int`,
     checked positive conversion, with no new dependency or public API. Vecadd
     keeps256 logical floats but rounds its allocation request upward using
     checked arithmetic before Block::open. Assert actual size≥logical size,
     actual size%page=0 and address%page=0; never infer native allocation extent
     from that rounded request. Zero exactly actual A/B/C extents through raw
     writes, end mutable views before imports and own Blocks until buffers/commands
     drop. Wait before CPU read; require STATUS_COMPLETED and no command error.
     After submit, wait comes
     before every assertion, error return or owner drop, even on GPU failure.
   - Existing gpu_buffer_wrap uses an owning page-sized Block for raw pointer
     import instead of its manual std allocation; initialize actual backing
     before its f32 view, pass actual size to buffer_wrap and assert returned
     size equality. This preserves the direct raw-import/readback test and
     removes both uninitialized-reference construction and fallible cleanup gaps.
   - Before either native import, a test-only support module calls
     `mach_vm_region_recurse` for `mach_task_self_`, depth=u32::MAX, using SDK's
     short-info layout (`repr(C,packed(4))`,48bytes/12integer words; assert ABI).
     Require success, expected info count, is_submap=0, RW protection, and checked
     containment of the complete actual backing in the returned leaf region.
     This API returns no object-name send right. Fail before native import if
     proof fails; no skip or fallback. Locals retain the fresh mapping through
     import/use/drop, without remap or external aliases. This establishes these
     fixtures' current mapping, not a universal IOSurface single-region promise.
     RW is needed by these read/write fixtures, not an additional Metal-wide
     no-copy precondition. Include the support module from integration.rs.
   - ANE fixture initializes actual input/output extents first; ends input view,
     then existing unsafe synchronous run_direct, then output read after success.
     Keep ANE_LOCK and owning locals. Explicitly record the existing synchronous
     private-API assumption; do not invent a transferable completion guarantee.
4. Add compiler-negative tests in API doctests, with functions taking arguments
   (no hardware execution) → verify E0133 separately for each of six views, warm
   and GPU wrap; E0596 for unsafe mutable access through `&Block`/`&Tape`; E0499
   for simultaneous mutable views; E0502 for read-held-across-mutation. Use
   explicit unsafe calls in borrow-check cases so the intended diagnostic wins,
   and use both borrows afterward so NLL cannot discard the witness.
5. New native `unimem/tests/block_access.rs` → initialize full actual backing for
   awkward requested sizes1/13/alignment+1, verify all slice extents/patterns and
   sequential typed writes. Test warm leaves every initialized byte outside the
   fixed16KiB stride offsets unchanged, zeroes just those offsets, and preserves
   Tape used/free. Keep migrated roundtrip.rs≤500lines (base450); move its four
   Block typed-view tests into block_access.rs if needed rather than cross the cap.
   Move a fully initialized Block to a thread and back; scoped concurrent reads
   are allowed under no-writer proof, writes require exclusive ownership after
   joins. These establish a raw contract, not shared immutable publication.
6. Run the gates below, freeze exact source/commands, fresh vendor CODE review
   and audit receipt → no row6 closure, allocation/fence portability or all-unimem
   safety claim. No sibling migration without its own reviewed task.

## Files and independent review source pack

Production changes: `unimem/src/{block,tape}.rs`, `aruminium/src/device.rs`;
documentation-only safety clarification in `unimem/src/grid.rs` and
`aruminium/src/buffer.rs`.
Tests: `unimem/tests/{roundtrip,block_creation,block_access}.rs`,
`aruminium/tests/integration.rs`, new test-only
`aruminium/tests/support/import_memory.rs`, `rane/tests/hardware.rs`.
Specs/docs: `unimem/specs/{README,api-sketch}.md`,
`aruminium/specs/README.md`, `aruminium/README.md`, and only the adjacent
`unimem/CLAUDE.md` Block Send+Sync/page-size gotchas.
Task docs and later audit receipt only otherwise. No manifests/locks/versions.
Keep new files≤500lines; narrow migrations of existing larger harnesses only.
roundtrip.rs stays≤500; aruminium/tests/integration.rs is already1078lines at
base7b148906 (`wc -l`), a disclosed narrow migration, not a new over-limit file.

Review pack must include both task docs; root/unimem/aruminium/rane CLAUDE.md;
all planned existing paths above; unchanged `unimem/src/block/creation.rs`,
`unimem/src/{lib,layout}.rs`, `aruminium/src/{buffer,command,encoder,lib}.rs`,
`rane/src/{model,surface,ffi,lib}.rs`; full native command support
`aruminium/src/{dispatch,sync}.rs` and `aruminium/src/ffi/{mod,selectors,trampoline}.rs`;
Cargo.toml/lock and relevant crate manifests;
parent Block research/plan and native extent receipt. Include pinned external
mir `src/gpu.rs`/Cargo.toml, bbg `rs/src/storage/unimem.rs`/`rs/Cargo.toml`,
erga `rs/{blake-bench,rtable-bench}/src/lib.rs`, xena
`crates/{xena-gpu/src/lib,xena-hash/src/scratchpad}.rs`, bita
`crates/rbtc-hash5/src/metal/mining.rs` and their manifests at research pins.
`rane/src/model.rs` contains both run_direct and build_request; its FFI is local
`rane/src/ffi.rs`, not an omitted runtime module. Include complete files, not
excerpts. Research lists exact existing inputs plus the checksummed Apple/SDK/XNU
primary-source pack. Full private block/aruminium test modules, unimem ffi.rs and
rane/probe/compile.rs join the pack. The complete157-file inventory/commands/hashes
represents unmodified zero-call examples, bench members, probes and experiments;
their complete bodies stay available at the commit and are omitted for context
size. Include full nonzero ambiguous-call source to prove the str::as_bytes
classification. No local Rust file is excerpted or examples/benches migrated.
block_access.rs and support/import_memory.rs are planned
new tests, not missing review inputs. No cached review inheritance.

## Gates and honest compatibility

Use exact committed parent + owned source overlay in an isolated full sibling
archive closure with strata56aedb2 and its reachable origin inputs, never dirty
siblings. Preserve Cargo.lock and private targets. The nebu name gap is fixed;
do not resurrect the obsolete standalone-only workaround. Parent's known
unrelated warnings/format failures must be reproduced/attributed, not silently
fixed or counted green. If parent merges before implementation, compare its
source delta and update baseline first.

Ordinary gates pin installed rustup1.95.0 (rustc59807616, LLVM22.1.2), with
RUSTUP_TOOLCHAIN and exact matching RUSTC/RUSTDOC/Cargo/fmt/Clippy paths. Prepend
the selected toolchain bin to PATH: rustup run alone did select an incompatible
Homebrew cargo-clippy. Record resolved paths, executable hashes and actual
version outputs, including cargo-fmt/rustfmt and cargo-clippy/clippy-driver. Homebrew
1.95 reports LLVM22.1.3; moving stable1.98 is observed metadata only, not the gate
compiler. Root installed matching1.95 Clippy. The completed exact-parent
baseline has no candidate gate. Scaffolding `/tmp/unimem-block-access-validation-6eQqO1` keeps
separate exact parent/candidate archives, private targets and receipt logs.

Required commands after approval (all Cargo invocations use --locked; offline
where the recorded archive cache permits):

```text
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy --workspace --all-targets --locked --offline --message-format=json -- -W warnings
cargo clippy -p unimem -p aruminium --lib --no-deps --locked --offline -- -D warnings -D clippy::missing_safety_doc
cargo clippy -p unimem -p aruminium -p rane --tests --no-deps --locked --offline --message-format=json -- -W warnings
cargo build --release --workspace --locked
cargo test --workspace --locked
cargo test -p unimem --test block_access --test block_creation --test roundtrip --release --locked
cargo test -p aruminium --test integration wrap_block_vecadd --locked
cargo test -p aruminium --test integration gpu_buffer_wrap --locked
cargo test -p rane --test hardware run_direct_with_unimem_block --locked
cargo build -p unimem --locked
cargo bench -p unimem --locked
cargo run -p unimem --example pipeline --release --locked
cargo run -p aruminium --example vecadd --locked --offline
cargo run -p rane --example matmul --locked --offline
cargo doc -p unimem -p aruminium --no-deps --locked (RUSTDOCFLAGS=-Dwarnings)
git diff --check
```

Workspace -D warnings and package selection without --no-deps stop on existing
acpu lints before dependent targets; neither proves the eight Safety sections.
Exact-parent full workspace -W also fails on auto-discovered benchmark imports;
retain both red gates and compare their emitted diagnostics, not just exits.
The selected strict library command passes on the parent and must pass on the
candidate: it reaches Block/Tape/Gpu and explicitly checks missing_safety_doc.
The selected --tests -W command completes on the parent with eight existing
needless_range_loop warnings; candidate must complete with no new unexplained
finding. Compare complete emitted message multisets (target/kind/code/message,
source text, child messages and duplicate count; keep moved coordinates in raw
evidence). The scaffold's `parent-baseline/diagnostics.nu` normalizes both JSON
warning gates and compares candidate with `--compare <parent-label>`; inspect
every addition/removal and compare the retained -D stderr by code/source location.
Failure can mask targets, so no full-workspace lint coverage is inferred from a
matching failed set. No lint suppression or unrelated benchmark fix is planned.
Matmul's existing source fixes ic/oc/seq=64 and offers no size argument; record
that actual bounded invocation rather than invent a CLI option.

Use actual installed nightly2025-11-26 with both RUSTC/RUSTDOC pinned for
`cargo test -p unimem -p aruminium --doc --locked`; stable rustdoc does not check
named error codes. Existing ignored aruminium examples stay distinguished from
new compile-negative cases. Compile-only borrow tests require no native calls;
the native framework paths have no Miri execution claim.

Semver diagnostic: `cargo semver-checks --workspace --exclude metal-benches
--exclude rane-benches --baseline-root <exact-parent-archive> --verbose`, then
last tag v0.2.0 with its true dependency closure if available. Expect intentional
unsafe/receiver source breaks in unimem/aruminium and reexports; list actual
diagnostics, retain source compiler witnesses if the tool misses receivers.
Do not claim a tag comparison when it fails to resolve historical dependencies.

Ring≥1: attempt the direct pinned consumers in research (cyb/mir/evy/bita/erga/
glia/nika/perla/trisha/xena/zoya) in exact origin archives, including applicable
optional unimem paths and declared checks/tests, with a recorded candidate
dependency overlay distinct from an origin release. Compare parent and candidate
with the same compiler/features. Record per consumer/path the first diagnostic,
whether it masks other call sites, and caused-by-change versus pre-existing.
Xena's unsafe-call-only compiler migration still needs initialization/lifetime/
exclusion proof; erga and bita additionally need receiver/ownership redesign.
Mir's missing alloc_size/platform blockers and BBG's unwired backend are prior
defects, distinct from new unsafe/receiver obligations. BBG's unwired backend is
source-only compatibility debt, not a passing exercised feature. Report those
gates before merge readiness; no release/tag/pin/publishing action in this unit.

## Five failure questions

1. Are complete returned slices initialized, including padding? Raw full-extent
   test initialization and explicit typed-view preconditions; no native-zero claim.
2. Can safe access still bypass the chosen raw boundary? Inventory Block, Tape,
   Grid and wrap; qualify all six views/import/warm, and state unrelated existing
   Grid/Buffer/ANE defects separately instead of asserting global safety.
3. Can mutable aliases or device work outlive an owner/borrow? Receiver checks,
   full-lifetime safety docs, ended fixture borrows, wait/status and owned scopes;
   no accepted mechanical unsafe migration without those facts.
4. Does the repair silently change lazy allocation, charges or sample data?
   Constructors/plan/drop unchanged; initialization only at inspected callers,
   same native and byte oracles, warming only its original fixed-stride bytes.
5. Does green local code imply dependent/whole-image readiness? Record actual
   full/dependent gate failures and intentional source breaks; native owner,
   reservation, retained publication, completion and portability remain future
   units. No approved PLAN or production completion is claimed by these docs.
