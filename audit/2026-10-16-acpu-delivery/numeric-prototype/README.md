# Bounded scaled-recurrence prototype

Audit-only standalone Rust, importing no production code. The accepted
[implementation plan](../../../.claude/plans/chebyshev-scaling.md) defines the
proposed checked API, output/work bounds and legacy-wrapper refusal semantics.
Frozen recurrence SHA256:
`3673c49f69c15256c7c15ca07e3791ebfcc81c83467ce70ebcb0ef73fd6ae1a3`.
Each receipt names its exact source files, compiler, commands, contextual Git
revision and raw output. `verify.nu REPO` verifies those inputs and streams.

| Receipt | Actual observations |
|---|---|
| [first](first.json) | Root's independent 160-pair reference: 150 accepted, 149 resolved comparisons; one unresolved integral remains null, ten f32::MAX requests refuse for work. Maximum measured target fraction 0.3206240213967898. Every accepted case checks exact work and one-less refusal; ten admission/tiny-time controls pass. |
| [stress](stress.json) | 297 structural cases, 264 accepted / 33 work refusals; finite values, parity, partial mass and cross-order agreement. Maximum 688165 steps, two refinement rounds. These consistency checks supply no independent accuracy claim. |
| [random](random.json) | 2339 seeded/random/fixed-transition cases: 2304 resolved comparisons, including 746 subnormal coefficients; 35 integral estimates remain unresolved. Maximum measured target fraction 0.4960834420668369. Includes positive series, independently refined integrals through accepted high times and custom-budget tau=1e10. Four zero-time count/custom-budget controls pass. |
| [limits](limits.json) | Nonzero tau=1, order=65535: all 65536 coefficients admitted, 131073 steps, finite/parity checks and exact 21-value prefix agreement. Default one-over refuses; raised cap admits 65537; impossible vector layout reports Allocation. |

The target fraction divides error by `2^-21*abs(reference)+2^-149`.
This is measured agreement with floating reference estimates, not a globally
certified bound. The randomized reference code uses the positive defining
series ([DLMF 10.25.2](https://dlmf.nist.gov/10.25.E2)) through tau=700 and scaled
quadrature ([10.32.3](https://dlmf.nist.gov/10.32.E3)) above it. Cancellation-floor
estimates remain null. Refinement is a numerical resolution screen, not directed
interval arithmetic. `random.rs` records its deterministic seed and case domains.

Far-small positive-series values can underflow f64 to numerical zero. They are
not exact mathematical zeros. [The retained screen](underflow-final.json) checks
649 such rows using the termwise upper bound
`2*exp(-tau)*(tau/2)^n/n!*exp(tau^2/(4(n+1)))`: the largest computed log upper is
below -744, versus approximately -104 for half an f32 subnormal. This separate
log calculation has ample numerical margin but is not an interval proof.
The first screen mistakenly used Nushell's approximate float equality to select
zeros; its failure and original source are retained as `underflow-first.*`.
The corrected screen selects the exact serialized zero representation.

The ratio recurrence follows [DLMF 10.29.1](https://dlmf.nist.gov/10.29.E1).
[10.37](https://dlmf.nist.gov/10.37) bounds the positive terminal ratio;
[10.35.1](https://dlmf.nist.gov/10.35.E1) gives the generating function used for
the tail estimate. Endpoint bracketing is an exact-arithmetic linear-fractional
argument; actual f64 endpoints and transcendental evaluation are convergence
estimates. Production implementation and final CODE/gates follow separately.
