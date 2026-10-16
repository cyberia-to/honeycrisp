# Independent scaled-Chebyshev review

Codex `/root/xtask_code_review`, 2026-10-16. Read-only design/prototype review;
production approval remains pending final implementation and source-bound gates.
Context: branch `fix/acpu-scaled-chebyshev`, base
`d5a78ed4865196241f6fd1e486fbb895fa59579a`.

## Prototype verdict

APPROVE the proposed architecture for implementation, with the evidence limits
and production obligations below. No numerical/source blocker found in the
inspected prototype. Exact reviewed sources:

| Input | SHA256 |
|---|---|
| numeric-design.md | 247d68b1d1a4b34bca0de94b590675a5846da968033520f78fdeac3999ef7f25 |
| numeric-prototype/recurrence.rs | 3673c49f69c15256c7c15ca07e3791ebfcc81c83467ce70ebcb0ef73fd6ae1a3 |
| numeric-prototype/probe.rs | 86a8be6d89d911efde7cfd47e04ea49545cbf09161e8f134113e62dd9d46bb54 |
| numeric-prototype/stress.rs | def0a50644b3aff43047ee3dd9d4160b2c1c3c7f493da0831167fd396b6d9143 |

I independently rehashed both receipt source lists, all eight build/run streams,
the generated reference input and the root-owned reference output. Actual
commands/revisions are recorded in `numeric-prototype/{first,stress}.json`:
rustc 1.95.0 optimized standalone builds and their retained executables, exit 0.
`first` compares 149 resolved coefficients within the frozen proposed target;
one cancellation-limited reference remains null and ten extreme-time requests
return WorkLimit. Maximum observed target fraction is 0.3206240213967898.
`stress` covers 297 requests (264 accepted, 33 work refusals), with maximum
688165 charged steps and two rounds. Its cross-order/parity/finite/mass checks
are structural consistency evidence, not an independent accuracy oracle.
No test was rerun by this reviewer. Seeded random independent controls are still
being prepared by the author and are not included in this verdict.

## Math assessment

The intended sign and scaling are correct for exp(-tau*(I+L_tilde)): c0=q0,
ck=2*(-1)^k*qk, qk=exp(-tau)*I_k(tau). Direct positive normalization avoids the
old overflow from first constructing unscaled f32 Bessel values.
[DLMF10.35](https://dlmf.nist.gov/10.35) supplies the generating function and
normalization identity; [DLMF10.37](https://dlmf.nist.gov/10.37) gives positive
integer-order decay at positive tau. [DLMF3.6(iii)](https://dlmf.nist.gov/3.6.iii)
supports normalized backward recurrence and depth refinement.

The endpoint argument is valid in exact arithmetic for the normalized result,
not just individual ratios. Set f_N=1 and f_(N+1)=r in [0,1]. Each backward f_k
is affine in r with nonnegative coefficients; the finite normalization is also
positive affine. Thus every normalized q_k has form (a+b*r)/(A+B*r), whose
derivative has constant sign. Its extrema occur at seeds0/1. The stored ratio
products and accumulated sum implement that finite normalized solution.
The exact infinite coefficient equals its truncated-normalized counterpart
times one minus the missing normalization mass.

The author's Chernoff expression follows by bounding the positive bilateral
generating-function tail and choosing asinh((N+1)/tau). Its rewritten expression
avoids subtracting two nearby unit-sized values in sqrt(1+u*u)-1. The minimum
positive f32 time and checked usize depth keep intermediate u/term finite in
f64 on this target. Coefficients too small for f64 are already far below the f32
absolute allowance. Finite output and convergence checks precede publication.

Floating evaluation of the tail and endpoint sequences is not outward-rounded
interval arithmetic. Endpoint agreement does not bound common f64 rounding
error. Keep the claimed target measured, with numerical controls, and avoid a
global exact-error or correctly-rounded promise. This is especially relevant
to arbitrary raised work limits beyond the exercised default domain.

## Production obligations

- Keep typed invalid-time/order/output/work/allocation/convergence distinctions,
  explicit defaults of 65536 coefficients and 1000000 coefficient-work units. Define
  each unit and clarify that it does not bound sparse matvec execution or
  allocator latency. Validate arithmetic/capacity and known unavoidable work
  before allocation; do not substitute silent zeros for refusal.
- Legacy coefficient and matvec entry points must document the new panic on
  checked default refusal. The matvec must obtain accepted coefficients before
  writing y/t0/t1; preserve existing CSR validation/error precedence. No claim
  that all later matvec allocations become fallible follows from this unit.
- Preserve the public unscaled bessel_i implementation/behavior. Keep coefficient
  error separate from caller-selected polynomial truncation and from the
  normalized symmetric-Laplacian assumptions needed for a heat-action bound.
- Pin zero/-zero, smallest subnormal, NaN/infinity/negative time, order overflow,
  maximum default output/one-over, exact measured work/one-below and raised
  custom limits. Cover accepted large times with independent accuracy controls;
  stress through order 4096 alone does not cover the maximum 65535 order.
- Retain finite coefficient controls at tau 100/104 and normalization-depth
  transitions. Test analytic zero/two-state operators and a separate dense
  reference, documenting truncation allowance. Do not alter tolerances to hide
  float cancellation or promote the unresolved oracle pair into a zero value.

Verification questions: the signs match the heat expansion; seed extrema apply
after normalization by the affine-ratio argument; omitted tail is a separate
normalization factor; checked depth/work loops terminate under explicit limits;
and prototype success does not imply an interval proof or final production
approval. Final CODE review will bind actual source/API and regression evidence.

## Concrete plan and random-control supplement

The concrete `.claude/plans/chebyshev-scaling.md`, SHA256
`2d337d9eef66d20748575f4614c419b42d669e6d24100e104013eac1ae5e5e69`,
resolves the API, work and requested-storage obligations: its 16K+4(K+1)
element-byte formula and separate bounded initialization accounting are correct.
APPROVE this plan for implementation. One test addition was requested: exercise
the maximum default order at nonzero time, since the zero-time maximum bypasses
the ratio allocation and recurrence. This is a production regression obligation,
not an architectural blocker.

The actual `numeric-prototype/random.json` rustc/run receipt now passes. I read
the independent series/integral helper and rehashed its three source files and
four raw streams. Frozen random source is
`edd9997b56104ba601141296052f9882a533302a186b6884f5b3f807825b507e`,
reference helper `6fc22944c2ff0701d89c148314ec6f8e26bf88ef415b89d6d7670c53768d2c52`;
the recurrence remains `3673c49f...`. Of 2339 sampled cases, 2304 have retained
reference comparisons (2212 positive-series and 92 resolved integral controls);
35 cancellation-limited integral values remain null. There are 746 nonzero
subnormal reference values. Maximum target fraction is 0.4960834420668369;
this is an observed result against estimated references, not exact rounding.
Raised work at tau1e10 and output-cap controls pass; the default extreme-time
refusal remains explicit. Far-tiny series underflow must be described as a
numerical estimate or justified below the absolute allowance, not as a claim
that the mathematical coefficient is exactly zero.

## Final production CODE verdict

APPROVE the scoped coefficient implementation. Independent Codex review by
`/root/xtask_code_review`; no vendor certification or council claim. Reviewed
context is `475dd6d22c4d179aeb0d3a480303d3af4ff6a42c` plus the frozen delta,
with complete owner/Strata input digest
`bf7125964da4ffae87f6f419de482c4295e0975af7f5764018c5a1e11608371c`
in `numeric-implementation/final/sources.json`. No unresolved source blocker.

| Reviewed input | SHA256 |
|---|---|
| acpu/src/sparse/coefficients.rs | 8d645955bb4d05bb357fc7bdcc1e3470f36990bcd14677b405acd23f9330b832 |
| acpu/src/sparse/chebyshev.rs | 1ba456138b4795f1a75f84dd822bf871adbb0b14120459fa67b46015bf639f9a |
| acpu/src/sparse/mod.rs | feb8ae43b2a0abfaf0b9050d658abc2ed051ac1b486acabdc57e2eddb8a3a243 |
| acpu/specs/README.md | 856baef47e18951b6be4dfe6b3eb62853c2c3974c7b266c0c9065dcc383cfd9c |
| acpu/tests/chebyshev_coefficients.rs | 4c38c6e65796747890cae86ae42d93af60b6ac731fae218f44342a0c4ce77e9c |
| acpu/tests/chebyshev_coefficients/reference.rs | 6fc22944c2ff0701d89c148314ec6f8e26bf88ef415b89d6d7670c53768d2c52 |
| acpu/tests/chebyshev_coefficients/independent.tsv | ee45c184ea38b4b479e6ad0ddb7a34eb51a692f8a30693067840aea41b5db469 |
| acpu/tests/chebyshev_independent.rs | ddd1e0cf070c927d67a167022844afef1595b59c925a8e45b41d93a6c0e9646c |

I independently enumerated and rehashed all 255 owner and 243 Strata inputs,
including current tracked/untracked inventory equality and both recorded HEADs.
The sibling remains `56aedb2d12b3126c601eb333419136d403614dbb`.
`nu audit/2026-10-16-acpu-delivery/numeric-implementation/verify.nu REPO`
then passed its source, stream, historical-bridge and unchanged-kernel checks.
I read the retained raw gate output rather than inferring success from source.
No Cargo gate was rerun by this reviewer.

All eight final receipts bind that same digest and exit0: fmt, strict workspace
Clippy, strict acpu library/tests Clippy, workspace tests, release workspace
build, release acpu tests, actual acpu matmul and strict acpu rustdoc. Their exact
commands/tool paths are in `numeric-implementation/final/*.json`: rustc/Cargo
1.95.0, warning-denying Rust/Rustdoc flags. Raw workspace tests report 365 passed
and 5 ignored; release acpu tests report 196 passed and 1 ignored. Both include
the seven coefficient tests and two root-authored independent controls. The
actual 64x64 AMX matmul control reports maximum error0. This is a functional
control, not a coefficient throughput benchmark.

The original strict-Clippy exit101 is retained in `numeric-implementation/first`.
Its first-to-final source bridge changes only the coefficient test file: two
parity expressions and additional impossible-layout/work-priority witnesses.
I inspected that exact diff; production and reference values are unchanged.
I also compared the complete `bessel_i` and `chebyshev_matvec` function spans
against `git show 475dd6d:acpu/src/sparse/chebyshev.rs`: both are byte-identical.

I independently compared parsed binary f64 values and integer orders of every
TSV row to the root oracle `numeric-reference/resolved.stdout` (SHA256
`9b0f038bd81179df5b60581dbc1c50bf03afbbdfe976332820173d1ca26b93e4`).
The 149 rows match exactly after selecting resolved references through tau1e6.
The ten f32::MAX oracle pairs are intentionally excluded from success fixtures
because the default engine refuses their work; the unresolved tau1000/order320
value remains absent. My first selection mistakenly included those ten extreme
pairs and failed only its row-count check; correcting that reviewer selection
required no fixture or source change.

All twelve quality passes were considered. The accepted floating numerical
contract claims measured accuracy, with no cross-platform bit identity or
directed-rounding proof. There is no new unsafe, FFI, global state or crypto
path. Count/depth/work arithmetic is checked; the known minimum pass cost is
admitted before fallible allocation, and every refinement consumes the same
finite budget. Two output-bounded buffers are released on every error; no
allocation grows with recurrence depth. Typed errors and explicit default
wrapper panics match the canonical contract. Existing CSR checks precede
coefficient admission, which precedes all matvec output/scratch writes.
The private module is cohesive, below 500 lines, and adds no dependency or
parallel production algorithm. Resource tests exercise exact/one-less work,
both layout/work precedence paths, maximum nonzero output and raised limits.
Independent series/integral and analytic two-, three- and four-element heat
controls distinguish coefficient error from polynomial truncation. Root's
refusal witness preserves NaN payload and signed-zero bits in all mutable arrays.

The requested nonzero maximum-order prototype control also passed in
`numeric-prototype/limits.json`: 65536 outputs at tau1, 131073 charged steps,
and the raised 65537-output prefix control. Its numerical-zero screen
`underflow-final.json` reports 649 far-tiny series estimates with floating
log upper bounds below -744.72, versus half-f32-subnormal log -103.97; this is
explicitly a numerical screen, not a certified interval proof. Production
tests now retain the maximum-order control. The remaining limits are deliberate:
coefficient work excludes matvec work and allocator latency, old unscaled
`bessel_i` limitations remain, and arbitrary raised budgets carry no global
accuracy guarantee beyond the documented convergence criterion.
