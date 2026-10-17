# Packaged example lint repair — proposed scope

At committed `ff1a91ff3be0b4a7d83f66c6d33e7dd1a563d743`,
`checks-1/clippy-keep-going.json` records the actual strict package command:
`cargo clippy --all-targets --keep-going --locked --offline -- -D warnings`.
It returned 101. Its complete stderr and parsed `lint-inventory.json` retain
35 lint findings across 15 source files, plus the separately reproduced NeBu
normalization errors. No source edits have been applied.

| Paths under acpu/ | Proposed equivalent edit |
| --- | --- |
| bench/cores.rs | Local alias for the existing workload tuple type. |
| bench/sgemm.rs | Inline the three literal output suffixes into format strings. |
| bench/memory.rs | Preserve `3.14159f32` with its exact `from_bits` value; iterate the existing four timing slots. |
| bench/elementwise.rs | Spell `0.7978845608f32` as `0.797_884_6_f32`, after a compiled bit-equality proof. |
| bench/crypto.rs | Replace fixed 44-word temporary Vec with `[0u32; 44]`; same initialization/index order. |
| examples/pair_test.rs | Enumerate all 64 mutable array entries, retaining `(i+1) as f32`. |
| examples/matmul.rs | Iterate all existing output elements in the same order. |
| examples/neon_vs_amx.rs, phase_time.rs, single_vs_par.rs, profile64.rs | Iterate exactly the existing timing-buffer lengths, preserving every timed operation and its order. |
| examples/amx_smoke.rs, amx_opcode_scan.rs | Remove OR with constant zero only; AMX instruction encodings and operand bits remain identical. |
| examples/trace64.rs | Simplify constant offsets `0*N*NR` and `1*N*NR` to zero and `N*NR`, preserving all pointer positions/calls. |
| examples/decompose.rs | Return the three final `median_ns` expressions directly; unchanged closure bodies/evaluation order. |

This bounded repair changes example syntax and a fixed temporary allocation;
library kernels, numeric tolerances, FFI, AMX words, target inventory and package
version remain unchanged. Compile a standalone Rust value proof before edits,
retain full source diff, then run all owning gates and strict acpu all-target
Clippy. Preserve the initial red package epoch. Final normalized verification
uses a subsequent committed source epoch; any necessary intermediate red commit
uses `wip:` and quotes the actual first error, as explicitly authorized by root.
