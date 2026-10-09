# Checked Block creation — plan

PLAN approved with binding conditions in `/tmp/unimem-block-creation-plan-review.txt`.
Implementation follows those conditions; fresh CODE review remains required. Base:
`9593c7218e14e2b5b816d9008b45ab60b9580cf3`, branch `fix/unimem-block-creation`.
Serves kadek foundation row 6 / row 3 admission and unimem's canonical Block
creation/error contract. [Research](research.md) pins source, SDK and evidence.

## Contract and focused compatibility changes

Keep one raw row: width=requested bytes, height=1, bytes-per-element=1,
pixel-format=0. Align row stride using the native BytesPerRow requirement, then
align its total extent using AllocSize's requirement. This preserves shape and
existing ≥requested-size callers, but intentionally changes small unaligned
requests' returned size. SDK IOSurfaceRef.h:373–382 justifies amending the current
row=size spec. Preserve Block::open, address/size/id/handle and slice signatures.

Public concrete API, no generic allocator/ownership layer:

```rust
Block::plan(requested: usize) -> Result<BlockPlan, MemError>
Block::open(requested: usize) -> Result<Block, MemError> // plan()?.open()
// BlockPlan has private fields; no Default or unchecked constructor.
BlockPlan::requested_size(&self) -> usize
BlockPlan::row_bytes(&self) -> usize
BlockPlan::allocation_size(&self) -> usize
BlockPlan::open(&self) -> Result<Block, MemError>
```

BlockPlan is an immutable Copy/Clone/Debug value holding the exact checked
request, stride and allocation extent. Planning creates no CF object or surface;
it queries native alignment. Open uses stored properties without recomputing or
rounding them. The reusable plan carries numbers only and grants no reservation.
A caller can reserve allocation_size plus its own owner/metadata
before open; this unit implements no reservation ledger. Plan conveys neither
initialized contents nor immutable publication. Keep native initialization
assumptions out of its contract; a future writer initializes visible output.

Correct public raw FFI CFIndex/CFNumberType to isize and IOSurfaceCreate's input
to CFDictionaryRef; add immutable dictionary/property functions and borrowed
keys. Retain otherwise unchanged public raw declarations. Remove private
cf_str/cf_i64 helpers when unused. These type corrections and new MemError
variants break some source uses; record actual semver diagnostics against
v0.2.0 and origin, rather than requiring a fabricated clean comparison.
On the supported Apple LP64 target, isize and i64 have the same ABI; these are
SDK declaration repairs, with no observed prior runtime miscompile asserted.

## Algorithm and exact failure order

1. Amend existing spec Block properties/API/error paragraphs first → verify:
   SDK alignment rationale, retained raw shape, ≥requested extent, precise
   returned-failure scope; keep observed numbers in audit, not specs.
2. Validate request before native object creation → verify: zero → ZeroSize;
   request, row and final extent must fit usize, i64/CFNumber storage and
   isize::MAX slice length. Conversion/checked-add failure → SizeOverflow.
   Query row and allocation alignment; zero → BlockAlignmentInvalid.
   Use remainder-based checked round-up (no power-of-two assumption), then
   cross-check IOSurfaceAlignProperty for both already-representable values;
   disagreement → BlockAlignmentInvalid. Never pass an overflowing value to C.
3. Construct six number values with correct-width FFI and private RAII CF owners;
   borrow the six SDK key pointer constants. Take callback structure addresses
   with `addr_of!`, without forming references to their opaque declarations.
   Build immutable CFDictionaryCreate from
   fixed stack arrays and kCFType callbacks → verify: each returned null yields
   BlockPropertiesFailed and releases all previously owned numbers. Dictionary
   retains entries, then local references release exactly once. No CString,
   Vec, Box, mutable dictionary insertion or library unwrap/expect.
4. IOSurfaceCreate null → existing BlockCreateFailed. Retain an owning surface
   guard; GetAllocSize must equal the planned extent, otherwise
   BlockExtentMismatch { expected, actual }, release without locking.
   Lock failure → existing BlockLockFailed(code), release without unlock.
   After lock, require nonnull base, alignment for existing u16/f32 slices and
   checked address+extent without wrap; failure → BlockAddressInvalid.
   Success transfers the locked owner to Block → verify: every successful
   lock is followed by exactly one unlock before exactly one final release,
   including post-lock errors; no allocation is published on failure.
5. Preserve accessors, Tape/Grid/Layout behavior and owned surface drop → verify:
   existing roundtrip oracles, first/last requested bytes and actual-extent
   lengths. Changing public aliasing and safe GPU-import APIs stays separate.

New unit error variants are exactly SizeOverflow, BlockAlignmentInvalid,
BlockPropertiesFailed, BlockExtentMismatch { expected: usize, actual: usize },
BlockAddressInvalid; existing ZeroSize/BlockCreateFailed/BlockLockFailed remain.
Display uses fixed phrases plus numeric context, with no heap-owned error data.
Planning and creation errors carry no generic OOM recovery promise: validate
returned-null/native error channels; opaque CF/kernel abort behavior is outside
the API's control. Existing “allocation failure always returns” wording narrows
accordingly. Allocation size means API-visible backing; CF/kernel bookkeeping
and residency are not counted as that extent. Equality check defends the stated
platform extent contract; it is not an allocate-first admission strategy.

## Files and decisive tests

Production: `unimem/src/{block,ffi,lib}.rs`; optional private
`unimem/src/block/{creation,tests}.rs` to keep each source ≤500 lines.
Specs: focused edits only in `unimem/specs/{README,api-sketch}.md`.
Tests: new `unimem/tests/block_creation.rs`; retain `tests/roundtrip.rs` oracles.
Task docs and later exact receipts under `audit/`; no manifests/dependency
versions, sibling source, hardware import, public mutable-view or platform edits.

- Pure planning cases: zero, isize/usize boundaries, exact multiples, one above,
  unusual non-power-of-two alignment, zero alignment, arithmetic overflow;
  independent quotient/remainder expectations and native cross-checks.
- Compile-negative examples separately show private plan fields and no unchecked
  constructor; positive path reserves a test counter before the plan is opened.
- Native boundaries around queried row alignment and page size, plus existing
  4096-byte/large roundtrip tests: planned == actual, width preserved, stride
  aligned, address typed-aligned, write/read initialized first/last bytes.
  Query surface width/row through test-only SDK declarations; no new public
  introspection API merely for tests and no reading unknown initial contents.
- A private cfg(test), thread-local fault/trace seam around the concrete native
  calls retains real CF/surface objects. Fail each number creation, dictionary,
  surface creation, lock, extent query, null/misaligned/wrapping base result.
  Observe owned Create/Release and lock/unlock events; exact balance/order and
  no subsequent forbidden call. CF numbers may be cached: count ownership
  acquisitions, not unique pointers. Never pass synthetic handles to CF.
- Production tests use the actual native path. Fault injection does not model
  every CF internal allocation, prove kernel memory reclamation, or serve as a
  native framework Miri test. Fresh source review checks callback retain rules.

## Verification isolation and gates

After approval, archive base and v0.2.0 into fresh temporary roots; overlay only
the named owned changed sources for the local candidate, record their hashes.
Build standalone unimem copies with their exact src/tests, one explicit isolated
workspace manifest: package unimem 0.2.0/edition2021; only runtime dependencies
crossbeam-queue =0.3.12, mutants =0.0.3; no examples/benches or unrelated dev
dependencies. Seed upstream lock's reachable runtime subgraph including
crossbeam-utils 0.8.21; save the adapted manifest/lock and verify checksums against
upstream Cargo.lock. This is a documented gate harness, not a production manifest
change or compatibility shim. All subsequent commands use --locked and a private
target. Never resolve sibling paths from the dirty originals.

Isolated commands: `cargo fmt --check`; `cargo clippy --all-targets --locked --
-D warnings`; `cargo test --locked`; `cargo test --release --locked`;
`cargo doc --no-deps --locked` with RUSTDOCFLAGS=-Dwarnings;
`cargo semver-checks check-release --manifest-path <candidate>/Cargo.toml
--baseline-root <baseline>` against both v0.2.0 and origin. Record intentional
enum/FFI source breaks, unexpected findings and actual exits separately. Run
`git diff --check` in the owned checkout and review source/spec byte deltas.

Required actual upstream gates, from an isolated exact-origin sibling closure:
root `cargo fmt --all -- --check`, `cargo clippy --workspace -- -W warnings`,
`cargo build --release --workspace`, `cargo test --workspace`; unimem
`cargo build`, `cargo test`, `cargo bench`, `cargo run --example pipeline
--release`. Enforce zero warnings and record toolchain/host/commands/exits.
Honeycrisp's cyb-nebu versus strata-origin strata-nebu mismatch is an existing
source prerequisite for those gates. Do not claim isolated tests close it or
silently rename/patch dependencies. Ring ≥1: public unimem APIs reach
acpu/rane/aruminium and downstream mir; dependent gates need their own exact
origin closures. This plan can be reviewed/implemented independently, but full
merge readiness remains conditional on those required gates and dependency fix.

## Five failure questions

1. Can large input truncate or become an oversized Rust slice? Reject before
   Create; independently check rounding overflow, final isize limit and pointer
   range, including unexpected actual extent. Never attempt enormous native allocs.
2. Can reservation use a different extent from creation? One private checked plan
   stores the actual properties; create never silently adjusts them. Preserve
   native extent premise and reject returned mismatch before exposing Block.
3. Can a returned failure leak or double-release a CF/native owner? RAII owns
   every Create result immediately; test each stage and locked/unlocked cleanup.
4. Does aligning the row secretly change consumer semantics? Preserve logical
   width/one-row format, explicitly document padding, verify ≥requested callers
   and actual-size hardware bindings; do not promise old byte-equal small sizes.
5. Could green local tests overstate safety or release readiness? Label the
   isolated manifest, unresolved dependency gate, CF failure scope and unmodified
   aliasing/import/init/platform gaps. Fresh PLAN then CODE review precedes merge;
   no Kadek row closure, decoder admission or fully fallible allocator claim.
