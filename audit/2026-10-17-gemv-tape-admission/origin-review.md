# Origin drift before the acpu version bump

Static review, 2026-10-17. Preparation tooling is preserved at
`9e6cffe699278059c8940f13940727b10569e13e` on
`origin/chore/acpu-release-preparation`. Its reviewed source base remains
`3ffc0d5e6f268d28658b8ad279b7adcfa0735fa6`.

The observed origin revision is
`e86310f564287a720d56e2d8204fd6c7181e6f6a`, merge of PR23. The commands
`git log --oneline 3ffc0d5..e86310f` and `git diff 3ffc0d5 e86310f` identify
changes in `acpu/src/gemm/gemv_amx_asm.rs`, `gemv_kern.rs`, `matvec.rs`, and
`unimem/src/tape.rs`. Each full file was read with `git show e86310f:PATH`;
`gemm/mod.rs`, `unimem/src/block.rs` and the existing specifications supplied
the adjacent contracts. All dimensions and limits below come from those source
commands at this revision. This review executes no candidate gates or unsafe
reproductions. Existing raw counts belong to their recorded earlier epoch.

## Findings in the complete changed paths

PR23 makes existing length and alignment assertions active in release builds.
The following gaps remain in the functions containing those assertions; the
new assertions did not introduce these older behaviors.

- The three GEMV entry points evaluate `k * n` without checked multiplication.
  Broadcast allocation also evaluates `k * 64` and does not check the returned
  pointer for allocation failure before constructing mutable slices.
- Both assembly paths accept `k == 0` and call `alloc_zeroed` with a zero-size
  layout. Their subsequent Z stores can publish previous register contents.
  Empty dimensions need an explicit result before allocation and AMX access.
- Both kernels clear their first-K flag after the first pair or individual
  tile, rather than after the complete tile sweep. Remaining tiles therefore
  accumulate existing Z contents. `ensure_amx` retains its state on the thread;
  repeated calls need an independent oracle with changed inputs.
- `gemv_pure_asm` computes only `n / 16` complete tiles. It has no tail handling.
  Both kernels accept more output tiles than the documented 64-row Z capacity;
  the Rust operand builder masks tile indexes to six bits. `matvec_f32` routes
  dimensions through 2048 to a kernel documented for at most 1024 elements.
- `matvec_f32` documents accumulation. Its NEON paths add the initial output,
  while its AMX call overwrites it. `matvec_f32_set` clears output before the
  inner function validates dimensions.
- `Tape::take` rounds a cursor offset, then adds the backing address. For
  alignments greater than the backing address's actual alignment, a rounded
  offset does not establish an aligned returned pointer. The new acceptance
  test checks `Some`, without checking the pointer remainder. Block bounds
  current offsets by `isize::MAX`; the offset addition alone is not an observed
  overflow under those current invariants.

## Proposed ordinary repairs before a bump

Keep the current preparation commit and earlier evidence intact. Reuse the same
checkout for a separately reviewed ordinary repair branch from the reviewed
origin. Complete acpu admission before allocation or output writes; define empty
dimensions; check broadcast layout and allocation; keep first-K initialization
active for every tile in its sweep. Route oversized tiles and pure-assembly
tails through the existing NEON path. Preserve accumulation through the existing
NEON implementation while retaining AMX overwrite dispatch for the set operation.
AMX instruction encodings and public signatures stay unchanged.

Use separate regression tests for checked dimensions and unchanged sentinel
outputs, empty dimensions, repeated calls on the same thread with changed data,
tile boundaries/tails, and nonzero initial output. Correct Tape alignment using
checked absolute-address arithmetic, derive its relative offset, and keep the
cursor unchanged on refusal. Test actual pointer alignment beyond page size,
invalid alignments, exhaustion and cursor conservation. Update the narrow
existing contracts where their pseudocode or behavior changes.

Run fresh named owning gates from `common.nu`: format, strict workspace Clippy,
strict acpu all-target Clippy, workspace tests, workspace release build, acpu
release library/integration tests, actual matmul and strict acpu documentation.
Also run focused acpu regression tests in debug/release, unimem tests in
debug/release, and `nu unimem/scripts/check-no-std.nu` for the Tape repair.
All commands use Rust1.95.0, strict warning flags and explicit external targets;
Cargo operations use locked/offline inputs where applicable. The isolated
no_std gate's documented invocation must be followed from its owning directory.

After ordinary repairs are reviewed and merged by an authorized merger, record
the new origin revision and derive fresh totals from its raw gate output and
test inventory. Explicitly re-anchor the preparation plan/tooling and review the
new closure. The subsequent version bump remains exactly `acpu/Cargo.toml`, root
`Cargo.lock`, and `acpu/CHANGELOG.md`, followed by fresh committed-package, API
and diagnostic Nox evidence. Owner-only bump merge, tags and publication remain
separate actions.
