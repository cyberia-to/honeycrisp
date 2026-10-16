# Required workspace gate repair

Source base: `c5177c56e4551c9896a2869c3b23d0c7bac9a6f9` (reviewed admission).
Final compiler closure: `0453b62f57181602a3404961fedccb36b64c579b3c222b41a7274745bfb8239d`.
[Exact owner/Strata inputs](sources.json), [runner](run-gates.nu).

The first mechanical bridge preserves arithmetic and test inputs; the retained
[independent review](../admission/independent-hygiene-review.md) includes actual
f32 bit-equivalence checks. Later compiler findings remove unused bindings,
imports, an uncalled private diagnostic helper, and redundant parentheses.
The CSR test tuple alias, fixed arrays and OR-with-zero removal preserve inputs
and AMX instruction/operand bits. Numerical tolerances stay unchanged.

Two separately authorized tooling corrections extend that bridge:

- Shared benchmark vDSP strides now match the installed SDK's signed 64-bit
  `vDSP_Stride`; duplicate numeric-benchmark declarations use the shared ones.
  CBLAS/vForce declarations stay unchanged. [SDK identity and controls](../vdsp-abi/sdk.json)
  and [independent ABI review](../delivery-final/independent-code-review.md)
  bind all eleven signatures to the header and contained forward/reverse calls.
  The standalone C probe is audit evidence; it adds no production dependency.
- Aruminium/Rane examples have unique `aruminium_matmul` / `rane_matmul` names.
  Acpu retains `matmul`. [Before](../examples-before.json) and
  [after](../examples-after.json) metadata retain all forty examples. The renamed
  Aruminium source is byte-identical; Rane changes only its Run-command comment.
  Both driver packages disable automatic bench discovery because `benches/`
  belongs to a separate workspace package. [Target conservation](../target-conservation.json)
  retains all five explicit benchmark binaries and removes only their accidental
  parent-package targets/helper-module targets. No source or dependency is dropped.

## Preserved execution epochs

Each directory retains its manifest, command, complete output and actual exit.
Earlier receipts retain their original status; subsequent results apply only to
the named source epoch.

| Directory | Result and next correction |
|---|---|
| `../hygiene/` | Strict acpu Clippy exposes test-only tuple/literal/array/identity lints after the first bridge. |
| `../hygiene-final/` | Fmt and both Clippy gates pass; workspace test compilation finds unused benchmark mutability and Cargo example-output collisions. |
| `../hygiene-candidate2/` | Workspace compilation exposes conflicting vDSP stride signatures; example collisions remain. |
| `../delivery-final/` | After ABI/example repair, fmt/Clippy pass; workspace compilation finds unused import/redundant parentheses in `bench/integer.rs`. SDK/Rust ABI controls pass; two initial probe link failures are retained. |
| `../build-diagnostic/` | First Cargo command rejects an unsupported flag. Supported all-target check finds accidentally discovered benchmark modules and remaining unused example helpers/imports. |
| `../delivery-v2/` | Same final source as this directory: fmt passes; Clippy dispatches a different rustup toolchain and rejects incompatible cached metadata (E0514). |
| This directory | Explicit `RUSTUP_TOOLCHAIN=1.95.0`, plus pinned Rust executables; actual Clippy version recorded. Read the individual final receipts for results. |

Reproduce final gates with the runner and the environment recorded in each
receipt; caches are external. This repair contains no version/dependency change,
publication or performance claim. Chebyshev scaled-coefficient overflow and its
bounded numerical contract remain a separate open source unit.
