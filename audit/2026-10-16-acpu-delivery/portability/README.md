# Audit replay and retained task context

Audit-only repair on `fix/acpu-audit-portability`, base
`c8450d0f008656a1260c094be7cb0a8f415daff6`. `verify-delivery.nu` now resolves
Honeycrisp from its repository argument and Strata from the declared `../strata`
sibling. Unknown owner names fail explicitly. Historical `inputs[].root` values
remain untouched as provenance; no verification command executes those paths.

Run `nu audit/2026-10-16-acpu-delivery/portability/check.nu REPO CANONICAL`.
The retained checks used canonical Honeycrisp at the base above:

| Receipt | Observed result |
| --- | --- |
| [numeric-canonical](numeric-canonical.json) | Canonical numerical verifier exits zero; all 498 current inputs and eight final gates replay successfully. |
| [historical-refusal](historical-refusal.json) | Updated historical verifier exits one with `inventory changed: honeycrisp; historical delivery-v4 source required`, because the numerical repair changed the older hygiene inventory. |

Each receipt records exact argv, script SHA256, revisions, unchanged replay
inputs, raw streams and expected exit. No compiler input changed or Cargo gate
was rerun. The expected historical refusal is preserved as an actual nonzero exit.

`task-chebyshev.json` and `task-admission-hygiene.json` archive the completed
wrapper contexts byte-for-byte before the portability task replaced `task.json`.
Their paths, branches, agent identity, sibling target/revision, cache and retention
reasons describe those historical tasks. The active wrapper retains this audit
task until root integrates it and inspects the checkout for clean removal.
