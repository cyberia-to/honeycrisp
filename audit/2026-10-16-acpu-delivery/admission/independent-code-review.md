# Independent admission CODE review

APPROVE for the scoped input-admission implementation. Reviewer:
`/root/xtask_code_review` (Codex), 2026-10-16. This is a source review, not a
release/gate approval: strict Clippy is red in this epoch and requires the
separately planned hygiene unit before integration can claim clean gates.

Reviewed the complete new `acpu/src/sparse/admission.rs`, sparse public and
private kernel paths, Chebyshev caller/recurrence, field NTT/interpolation
entry points, canonical admission clause and `tests/input_admission.rs` against
parent `70dfb1f55e4cb2570d5a78a497be5828a203087b`. Exact source closure:
`47bffad98aa5a820aab3bf02d2d19d375bcc0ba76e07d24268b35d39b9426250`.
[sources.json](sources.json) records every path/hash, including pinned Strata
`56aedb2d12b3126c601eb333419136d403614dbb`. The independent
[verification receipt](independent-verification.json) confirms all 250 owner
and 243 sibling files, inventory/revisions, and actual gate/raw-output hashes.
No production/test edits, malformed parent execution or duplicate test runs
were performed by this reviewer.

Initial assessment was that admission closes the unchecked CSR access gap while
preserving valid arithmetic. Five independent questions checked that conclusion:

1. Can any partition reach the kernel with an out-of-bounds interval? Nonempty
   row_ptr, initial zero, monotonic offsets and terminal equality to both entry
   lengths imply every interval is ordered and contained. Both public matvec
   entry points construct the private evidence; search found no bypass caller.
2. Can prefetch, vector gathers, vector loads or stores escape valid slices?
   Every column is checked against the exact admitted input length. `apply`
   rechecks actual x and y sizes before mutation. Four-wide loads occur only
   with four remaining entries; initial/lookahead prefetch indices are contained
   by their loop conditions; scalar tails and y stores use those same bounds.
   On the specified Apple aarch64 target, u32 entry offsets plus the small
   lookahead cannot overflow usize. Safe borrowed f32/u32 slices cannot mutate
   these retained indices behind the evidence.
3. Could a late shape failure leave zeroed output or changed Chebyshev scratch?
   All CSR validation precedes `set` zeroing. Chebyshev validates every supplied
   vector and square row count, then CSR, before coefficient/temporary allocation
   or output writes. Its recurrence reuses the evidence without rescanning CSR.
4. Does moving validation out of field recursion allow an invalid descendant?
   The public entry requires positive power-of-two evaluations and exactly its
   logarithmic coordinate count. Each private split halves evaluations and
   removes one coordinate until the single-element base case. Both permutation
   entry paths establish the same positive power-of-two invariant before writes.
5. Did valid input behavior or scope change unintentionally? Rectangular CSR,
   duplicate/unsorted columns, empty rows and zero-row matrices remain valid;
   original sparse arithmetic and accumulation order are unchanged. Shape
   rejection is deliberately all-build, documented panic-before-mutation.
   Root/canonical field-value preconditions and Chebyshev numerical behavior
   are unchanged. The revised conclusion remains APPROVE within that boundary.

## Quality passes

| Pass | Assessment of this delta |
| --- | --- |
| 1, determinism | Admission is ordered integer validation; existing floating-point kernels are unchanged. No new clocks, randomness or iteration-order dependence. |
| 2, locality | Validation reads the supplied partition/columns linearly; writes remain limited to supplied outputs/scratch. |
| 3, arithmetic | The new size/index proofs cover unsafe pointer access on the declared target. Existing field arithmetic and Chebyshev coefficients are outside this change. |
| 4, crypto | No new secret-bearing diagnostic values or cryptographic primitives. Shape validation is not advertised as constant time. |
| 5, invariants | Private evidence fields cannot be externally forged; every application verifies its current vector dimensions. |
| 6, errors | The existing infallible API retains explicit precondition panics. All newly checked malformed shapes preserve caller output/scratch; no partial mutation path remains for those conditions. |
| 7, adversarial | All CSR rows and columns are checked, including late rows and unused/order-zero Chebyshev input. Computational order/tau resource policy is unchanged. |
| 8, architecture | One private reusable CSR validator; one public admission boundary per field operation. No new public carrier or dependency. |
| 9, readability | The compact validator states the proof at the kernel call; public panic clauses and canonical spec agree. |
| 10, compactness | Validation is factored once; no copied vectors or extra allocation. All changed source files remain below the size limit. |
| 11, performance | CSR adds O(rows + entries) admission per public call and amortizes it across the Chebyshev recurrence. No speed claim or benchmark result is inferred. |
| 12, tests | Five new integration tests exercise malformed partitions, late invalid columns, both mutation modes, empty/rectangular/duplicate positives, NEON-width/tail/prefetch rows, all Chebyshev dimensions and field shape rejection. |

## Observed evidence and limits

At the exact closure above, `cargo test -p acpu --lib --tests --locked --offline`
and the same command with `--release` each report 187 passed and one ignored
(`debug.json`, `release.json`). Both include the five new integration tests.
Their malformed cases assert output/scratch equality after panic; huge-order
malformed Chebyshev witnesses establish admission before coefficient allocation.
The valid two-vertex heat action uses an independent closed-form oracle. Existing
field tests cover valid permutation/NTT and nonconstant interpolation.

`cargo clippy -p acpu --lib --tests --locked --offline -- -D warnings` exits 101
(`clippy.json`): duplicated GEMM attribute, historical sparse loops/assignment,
and pre-existing test warnings. Their unchanged source locations were inspected;
this review neither suppresses them nor treats the failed gate as success.
Mechanical follow-up must retain the failure and supply a new reviewed epoch.

No claim is made that all acpu unsafe paths are audited, that the complete
workspace is green, or that Chebyshev overflow/large-order numeric accuracy is
fixed. The known coefficient overflow remains separately open. No version,
dependency identity, FFI signature, AMX encoding or numerical tolerance changed
in the reviewed admission unit.
