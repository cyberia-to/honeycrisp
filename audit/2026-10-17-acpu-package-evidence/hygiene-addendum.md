# Newly reachable example lints — proposed two-file extension

`owner-1/clippy-acpu.json` preserves the actual exit101 and ten additional
diagnostics after NeBu became available: nine in `acpu/bench/summary.rs`, one
in `acpu/bench/zk.rs`. The first is the warmup range loop at summary.rs:227.
The prior 15-file delta and source manifest remain retained unchanged.

Proposed summary.rs edits: iterate all 4096 existing warmup entries in order;
reuse proven s2pi/scalar spellings; spell `42.0 as f32` as `42.0_f32`; remove
one redundant borrow; combine identical adjacent formatting threshold arms;
use the same fixed 44-word array as the reviewed crypto example. For quantizing
the benchmark's scalar reference, preserve `round().max(-128).min(127)` exactly
with a named lower-bounded temporary before `.min(127)`. A direct `clamp` rewrite
would change NaN behavior, so the two original operations/order are retained.
zk.rs removes only the redundant `perm_ns as u64` cast (already u64).

No benchmark execution/performance claim, numeric library edits, lint allowances
or target changes. After approval, retain a second source delta and fresh owner
gate epoch. Root Cargo.lock remains unchanged.
