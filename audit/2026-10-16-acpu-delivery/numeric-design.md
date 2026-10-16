# Scaled Chebyshev coefficients — proposed separate unit

Status: architecture approved; [concrete implementation plan](../../.claude/plans/chebyshev-scaling.md)
supersedes the open API choices below. No numerical production edit yet. Admission commit
`c5177c5` preserves the original recurrence/scaling. The parent-owned
[independent oracle](numeric-reference/README.md) contains reference acquisition,
actual commands, frozen data and explicit cancellation limits. Its successful
cross-method agreement is evidence, not an interval proof.

## Contract to establish before code

For heat time tau finite and nonnegative, return K+1 coefficients
`c[0] = exp(-tau) I_0(tau)` and
`c[k] = 2 (-1)^k exp(-tau) I_k(tau)` for k>0. Zero time returns exactly
`[1,0,...]`; zero order is valid. Reject negative/nonfinite heat time, K+1
overflow and impossible buffer capacity before any matvec output/scratch write.
Retain the existing panic convention for these infallible entry points.

Proposed coefficient acceptance target:
`abs(computed - exact) <= 2^-21 * abs(exact) + 2^-149` (four f32 epsilons
relative plus one smallest subnormal). This is a proposed measured/analytical
target, not a present guarantee. Require finite signed coefficients; use the
subnormal term only for true tiny results, never to conceal nonconvergence.
K is caller-selected truncation order: accurate coefficients do not promise
an accurate truncated heat action for every K. The matrix also needs the
normalized symmetric-Laplacian assumptions for a spectral heat-error bound.

Do not claim support for the entire finite-f32 tau/order domain from moderate
fixtures. In particular the oracle resolves low-order values at f32::MAX while
Miller depth sufficient for normalization can be computationally infeasible.
Before implementation, review an explicit work-admission/error policy for this
case. A magic tau cutoff or silent zero vector is unacceptable. If practical
work refusal is needed, its limit must be an explicit resource contract with
failure before mutation; it must not be advertised as a mathematical singularity.
Root selected a checked entry point with output/work limits plus a documented
default-limits infallible wrapper. [Prototype evidence](numeric-prototype/README.md)
records independent comparisons and explicit resource refusals.

## Minimal numerical architecture

Keep one shared private recurrence family. Normalize the scaled sequence
directly using `q[0] + 2 sum(q[k],k>=1) = 1`, where q=exp(-tau)I.
Do not construct exp(+tau), or round an unscaled Bessel result to f32, on the
coefficient path. Compute all requested orders together, rescale working values
consistently, and round only final coefficients.

Replace fixed depth max(2k+60,80) with explicit depth/error admission. DLMF
[3.6(iii)](https://dlmf.nist.gov/3.6.iii) prescribes comparing normalized Miller
solutions at increasing depth. Agreement alone is insufficient as a certified
error bound: account separately for omitted normalization tail, boundary seed
error and floating-point error. Positive-order monotonicity gives a terminal
ratio in [0,1]; evaluate bounding terminal conditions or otherwise prove seed
error. [DLMF 10.37](https://dlmf.nist.gov/10.37) supports this ratio bound.

One useful derived tail bound is
`2 sum(q[k], k>N) <= 2 exp(tau*(cosh(s)-1) - s*(N+1))`, s>0,
obtained from the positive bilateral generating function at t=exp(s).
Its minimizer is `s=asinh((N+1)/tau)` for positive tau. Evaluate this expression
without cancellation for small s. This is an inference from
[DLMF 10.35.1/5](https://dlmf.nist.gov/10.35), not a supplied library guarantee.
It can expose why merely normalizing a shallow recurrence is insufficient.

Large-argument asymptotic seeding is a possible continuation if bounded work
over the full finite-f32 domain is required. It needs its own validity/remainder
proof; the asymptotic formula alone is not a uniform accuracy guarantee.
[DLMF 10.40](https://dlmf.nist.gov/10.40),
[10.41](https://dlmf.nist.gov/10.41). Do not bolt on an unreviewed second kernel.

The separately public unscaled `bessel_i` must remain explicit: genuine f32
overflow is different from scaled coefficient overflow. Reconstructing it from
already-rounded/underflowed q is unsafe numerically; review high-order and
negative-argument behavior before sharing that path. No silent change is part
of the present hygiene/admission work.

## Frozen regression obligations

- Pin the independent tau=100/104 witnesses, small/subnormal/zero times,
  order zero, both coefficient parities, higher orders and wider practical tau.
  Preserve the unresolved cancellation-floor oracle entry as unresolved.
- Check the recurrence/normalization error criterion across selected depth
  transitions; specifically demonstrate where the old depth-80 rule is wrong.
- Test order/capacity arithmetic and negative/NaN/infinite tau rejection before
  matvec writes. Exercise explicit work/convergence refusal once its contract
  is reviewed; include f32::MAX as a reference or deliberate refusal witness.
- Compare heat actions against analytic zero/two-element operators and an
  independent dense reference with truncation error separated from coefficient
  error. Retain tolerances and describe their derivation.
- Run debug/release, strict acpu Clippy/docs and the owning workspace gates on
  the exact reviewed source. Version/package proposal follows this unit only
  after its actual supported domain and failures are settled.
