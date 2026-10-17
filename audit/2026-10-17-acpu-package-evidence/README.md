# Committed acpu package evidence — 2026-10-17

Accepted [plan](../../.claude/plans/acpu-package-evidence.md), source base
`c7c9fabdb1e9a9b0be379917540355e89c899dfe`. This unit checks the published
library API and actual normalized package before a version proposal. Source,
dependencies and versions are unchanged; no publication is authorized.

The recreated owned checkout uses the retained wrapper. Its [prior context](wrapper-before.json)
is archived; [current context](wrapper-current.json) preserves the Strata target
and revision `56aedb2d12b3126c601eb333419136d403614dbb` exactly.

## Plan commit gates

[plan-gates.nu](plan-gates.nu) executed the required owning checks at the above
revision with Rust1.95.0, warnings denied and an external task cache. All eight
commands passed: fmt, strict workspace/acpu Clippy, workspace tests, workspace
release build, acpu release tests, real matmul, strict acpu docs. Exact argv,
source identity and raw streams are under [plan-gates/](plan-gates/); the
[verification](plan-verification.json) checks all outcomes/stream hashes.

Full owner/Strata closure SHA256
`fe97d5e30b6436cd27ef9c4bd247729664a396bf4fd4428c157ac4222754251e`.
[Source bridge](source-bridge.json) and [actual prior verifier run](prior-source-verification.json)
prove the complete source byte inventories equal the reviewed numeric repair;
recorded checkout roots/HEADs differ. These are fresh owning gates, separate
from package gates to follow. Matmul output is a correctness smoke; no benchmark
performance claim is made.

Package acquisition, all-target packaged checking, complete published API
comparison and the four-test normalized Nox consumer remain pending. The plan
records the expected path-only NeBu dev-dependency loss; it must be reproduced
without modifying the package before any normalization repair is proposed.
