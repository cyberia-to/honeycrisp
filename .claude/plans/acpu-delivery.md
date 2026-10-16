# acpu delivery prerequisites

Agent /root/phase2_resume_audit under /root; 2026-10-16. Owner checkout
`honeycrisp/.worktrees/acpu-delivery/honeycrisp`, branch `chore/acpu-delivery`,
base `70dfb1f55e4cb2570d5a78a497be5828a203087b`. Wrapper context preserves Strata
`56aedb2d12b3126c601eb333419136d403614dbb`; initial tracked/untracked/ignored
status is clean. Build cache: `~/.cache/cyber/honeycrisp/acpu-delivery`.

Initial ownership is this plan and `audit/2026-10-16-acpu-delivery/` only.
Run the whole-upstream gates before proposing any source or version changes:
workspace fmt check, strict Clippy, release build, workspace tests, and the real
acpu matmul example. Root/acpu CLAUDE.md define these gates. Record frozen source
bytes, commands, toolchain, dependency revision and complete output; retain reds.
Do not infer performance from this busy machine or from matmul's timing output.

Then inspect newly shipped sparse APIs, field malformed-input contracts and
published/source API compatibility. Existing bounded arithmetic diagnostics in
Nox establish the missing APIs' useful contract, not whole-upstream readiness.
Prefer the existing upstream implementation and package-only 0.2.1 proposal;
no duplicate kernels, fallback, dependency substitution or version edits now.

Review findings with root before ordinary fixes. Keep source fixes separate
from the later version-only PR. Preparing that proposal is authorized in
principle, but only the owner merges a bump, tags or publishes. Preserve this
checkout until the reviewed delivery work and evidence are integrated.

## First source unit for review: public input admission

Current source has a release-blocking soundness hole: safe CSR entry points
can reach unchecked aarch64 reads/writes with arbitrary column indices or short
output slices. Root confirmed this blocks publication. No UB witness is executed.

Proposed contract, retaining the existing safe signatures and panic convention:

- CSR: require a nonempty row-pointer array starting at zero; nondecreasing row
  offsets; final offset equal to the common column/value length; output length
  equal to the row count; every column less than the input length. Validate the
  entire input before pointer arithmetic, prefetch, unchecked loads/stores or
  mutation. Invalid inputs panic in debug and release with a field-specific
  message; all output/scratch arrays remain unchanged. `[0]` with empty data and
  output is the empty CSR. Rectangular matrices, duplicate/unsorted columns and
  empty rows remain valid. No floating-point value restriction belongs to this
  memory-safety unit.
- Chebyshev matvec: require row count equal to x length and d_inv_sqrt/y/t0/t1
  lengths equal to that count, then the same CSR admission before scratch/output
  mutation. Existing recurrence and valid-input floating-point order stay intact.
- Field: require positive power-of-two lengths for public bit reversal and
  multilinear evaluation in every build; multilinear point length must equal
  log2(evaluation length). ntt_forward already rejects invalid lengths.
  Preserve caller-supplied root semantics and the existing algorithms.

Approved edit scope: acpu/src/sparse/{mod,chebyshev}.rs, optional
private sparse/admission.rs, acpu/src/field/{ntt,poly}.rs, focused public regression
tests, and compact contracts in existing acpu/specs/README.md/rustdoc. No version,
dependency, FFI, AMX instruction or kernel scheduling changes. One shared CSR
validator is sufficient; use established kernels after admission.

Regression scope: empty and rectangular CSR; accumulate/set semantics; duplicate
columns; empty rows; scalar and four-wide rows; malformed partition/terminal,
column/value counts, output length and column indices; every Chebyshev shape;
invalid field lengths and both short/extra points. Check panic-before-mutation
and valid arithmetic equivalence in debug/release. Freeze the dangerous CSR
witnesses but execute them only after source review establishes the new checks
precede every unsafe operation. Their parent baseline is the static soundness
proof, never an executed out-of-bounds call. Safe field baseline outputs are
already retained in the audit. The existing unmerged 904dd336 length-check branch
is useful prior work but lacks CSR and point-count validation; do not merge it
blindly or duplicate the algorithms.

Root approved the source contract and complete guard diff before any malformed
CSR test ran. Implementation uses a private immutable Csr view with checked
partition/columns and O(1) vector-length checks at each kernel call. Chebyshev
constructs it once; field wrappers validate once before private kernels/recursion.
The numerical algorithms and sparse unsafe loop contents remain unchanged.

Verification questions resolved by source inspection and public tests:
1. Can a later malformed row mutate an earlier output? Complete admission scans
   the immutable partition and every column before apply; late-column tests retain y.
2. Can a reused view receive different vector lengths? Apply checks both lengths
   before zeroing/prefetch; Chebyshev scratch has its admitted fixed dimensions.
3. Can empty input or a mismatched point reach polynomial recursion? Public
   length and exact-coordinate assertions precede the sole private recursive call.
4. Are valid sparse representations narrowed? Rectangular, empty, duplicate and
   unsorted cases remain accepted; no floating-point restrictions were introduced.
5. Does a shape error allocate coefficients first? All shape checks precede that
   call; huge-order malformed fixtures require the specific shape panic.

## Separate work, still before version-only PR

Hygiene: one formatter file, duplicate allow attribute, three sparse Clippy
patterns, and unused trace64 example binding; preserve initial reds. Whole
workspace tests must then run beyond compilation. Proposed numerical follow-up:
Chebyshev coefficients overflow for finite positive tau=100 and become NaN at
104 although scaled coefficients are bounded. Review the supported numeric
domain and stable scaled evaluation with independent reference cases; do not
invent a silent clamp or arbitrary tau cap. This numerical issue remains open
after memory admission and is not repaired by the four existing Bessel examples.

Admission is committed in `c5177c5` with its historical red Clippy gate retained.
The separately reviewed hygiene unit now passes the required workspace gates,
strict acpu docs/tests, all-target compilation and actual CPU/GPU/ANE matmul
controls; exact receipts are in `audit/2026-10-16-acpu-delivery/delivery-v4/`.
Its authorized scope extensions correct benchmark vDSP stride ABI, unique
example names and parent-package automatic bench discovery, with independent
SDK/metadata proofs. Versions, dependencies and numerical algorithms remain
unchanged. Numerical scaling is the next distinct source unit.

Only after reviewed source fixes and all owning gates: prepare the package-only
acpu 0.2.1 version/lock/changelog proposal with owner approval for the explicit
package-version exception. Full gate/source evidence follows that exact versioned
commit; owner alone merges/tags/publishes. The present source is not release-ready.
