# Bounded scaled Chebyshev coefficients

Active checkout `honeycrisp/.worktrees/acpu-delivery/honeycrisp`, branch
`fix/acpu-scaled-chebyshev`, base `d5a78ed4865196241f6fd1e486fbb895fa59579a`.
The existing wrapper preserves the prior task context, Strata pin and external
cache. Root approved this architecture after the standalone stress evidence;
independent numerical review and randomized reference comparisons precede code.

## API and refusal

Add `try_chebyshev_coeffs(tau, order, ChebyshevLimits) ->
Result<Vec<f32>, ChebyshevError>`. Limits contain `max_coefficients: usize` and
`max_steps: u64`; named defaults are 65,536 and 1,000,000. Errors distinguish
invalid time, order overflow, output limit, work limit, allocation failure and
numerical nonconvergence. Time is finite and nonnegative; negative zero is zero.
Order zero is valid. Zero time returns exactly one followed by zeros.

Keep `chebyshev_coeffs` as the documented infallible default-limits wrapper:
it now panics on refusal. Callers needing other finite budgets use the checked
entry point. `chebyshev_matvec` already obtains coefficients after shape/CSR
validation and before scratch/output mutation; checked refusal therefore occurs
before those writes. The separately public unscaled `bessel_i` is unchanged,
including its current accuracy/overflow limitations. No version/dependency edits.

The work budget covers coefficient generation; matrix-vector work is separate.
One work unit is one tail estimate, one backward index computing both terminal
sequences, or one requested coefficient evaluation. Charge the complete
recurrence/output pass before execution, count every refinement, and preflight
its minimum cost before allocating. Zero time charges one unit per output.
Vector initialization is separately bounded by the output cap: K ratio pairs
and K+1 output slots, at most 2K+1 initialized slots. Requested heap storage is
16K + 4(K+1) bytes, 1,310,704 bytes at the default cap; allocator bookkeeping
and rounding are outside this element-byte count. All capacity arithmetic is
checked and allocation uses fallible reservation. Allocation failure also covers
an unrepresentable vector layout; the output-limit error is the caller's explicit
count cap. No buffer grows with depth.

## One numerical path

Use backward ratios `r[j-1] = 1 / (2j/tau + r[j])`, starting with terminal
ratios zero and one. Track `S[j-1] = 1 + r[j-1] S[j]`, starting S[N]=1;
normalization gives `q[0] = 1/(2S[0]-1)`. Store only the K requested ratios,
then multiply forward to obtain q[k]. Output c[0]=q[0] and
c[k]=2(-1)^k q[k]. This evaluates scaled quantities directly without exp(+tau).

DLMF 10.37 places the true positive terminal ratio in (0,1). In exact arithmetic
every backward value is affine in that terminal ratio, so each truncated
normalized coefficient is a linear-fractional function with positive denominator
and lies between its two endpoint values. This is a derived argument, separate
from floating-point evaluation. Bound omitted normalization through the positive
bilateral generating function (DLMF 10.35.1):
`tail <= 2 exp(tau(cosh(s)-1) - (N+1)s)` with s=asinh((N+1)/tau).
Evaluate the exponent as `(N+1)*(u/(1+hypot(1,u)) - asinh(u)) + ln(2)`.

Start N=max(K+1,16), doubling until the estimated tail is at most 2^-40 and the
next pass fits. Use the two normalized endpoint values plus that tail estimate
to form a coefficient interval. Accept when its width is at most
`2^-25 * upper + 2^-152`, then round the midpoint to f32. Otherwise double N;
nonfinite/invalid bounds or failure to reduce normalized width refuse with
nonconvergence. Every probe/refinement is charged; exhaustion is a work refusal,
including impractical large-time requests. No time cutoff or silent zero exists.

The interval argument is exact-arithmetic reasoning. Rust f64 transcendentals
and rounding are not directed interval arithmetic. The empirical acceptance
target against independent references is `|error| <= 2^-21 |reference| + 2^-149`;
document it as a measured target on retained cases, never a global certified
error bound. Coefficient accuracy and caller-selected truncation error are
separate; the matrix requires the normalized-Laplacian assumptions for heat
approximation claims.

## Implementation and verification

Keep the coefficient engine in a cohesive private sparse submodule with public
types re-exported from the current Chebyshev module; no parallel production
algorithm. The existing matvec recurrence and unscaled Bessel function remain.
Canonical API contracts go in `acpu/specs/README.md` before source edits;
measured evidence stays under `audit/2026-10-16-acpu-delivery/`.

Required witnesses: tau100/104 original overflow; zero/tiny/subnormal time;
normal/subnormal coefficient transitions; high orders and terminal-seed
refinement; exact budget and one-less refusal; invalid/overflow/output limits;
very large finite time refusal; all failures before matvec mutation. Use seeded
random times/orders against independent positive series and refined quadrature,
with unresolved reference values left unresolved. Add independent small-matrix
heat controls with truncation error stated separately. Run source-bound debug,
release, strict Clippy/docs and literal whole-workspace gates, then independent
CODE review before commit. No owner publication actions are authorized.
