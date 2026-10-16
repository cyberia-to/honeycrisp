# Independent scaled-Bessel controls

Root's standalone Rust probe imports no Honeycrisp code or recurrence. It evaluates
the scaled integer-order integral from [NIST DLMF 10.32.3](https://dlmf.nist.gov/10.32.E3)
with composite Simpson integration, and independently evaluates the positive-term
defining series from [DLMF 10.25.2](https://dlmf.nist.gov/10.25.E2) through tau=700.
The primary equations were inspected on 2026-10-16. No Python, external crates,
production dependency or system library is required.

The substitution u=sqrt(tau)*theta resolves the narrow peak at large tau;
`-2*sin(theta/2)^2` preserves the exponent near zero. The omitted integral beyond
u=24 has the explicit Gaussian upper bound in the source, derived from
sin(theta/2)>=theta/pi on [0,pi]. Refinement compares 16384 and 32768 panels.
This supplies measured reference values and a tail bound, not a rigorous bound
on floating-point/quadrature error or a global correctly-rounded function.

Reproduce with `nu audit/2026-10-16-acpu-delivery/numeric-reference/run.nu REPO CACHE LABEL`.
`resolved.json` binds the exact build/run commands, compiler, source hash and
raw streams at contextual Git revision `c5177c56e4551c9896a2869c3b23d0c7bac9a6f9`.
The probe is independent of that revision's production implementation.

Actual results in `resolved.stdout`:

- 160 evaluated pairs: 16 times from zero through f32::MAX, and orders
  0/1/2/5/10/20/40/80/160/320. Both integral refinements complete for every pair.
- 120 pairs also have the independent positive-series control. Their maximum
  series/integral difference is 8.664243548601087e-17; maximum coarse/fine integral
  difference across all pairs is 1.2593572922050639e-16.
- 159 coefficients have a resolved reference. At tau=1000, order=320, the
  integral is too small relative to this probe's cancellation floor; coefficient
  and f32 bits remain null. No zero or fabricated reference replaces it.
- The zeroth coefficient is 0.039944379299096666 at tau=100 and
  0.03916678462145642 at tau=104; these finite positive controls expose the
  existing infinity/NaN behavior. At tau=10000 it is 0.003989472674604732.

The first attempt asserted that every large-time integral was resolved and
stopped at the unresolved pair. `first.*` and `oracle-first.rs.txt` preserve that
actual failure/source. The correction retains the observation with null reference
fields. Exported f32 bits are the rounded numerical estimates; consumers must
choose justified tolerances rather than claim exact mathematical rounding.
This probe does not test a production repair or certify Chebyshev truncation.
