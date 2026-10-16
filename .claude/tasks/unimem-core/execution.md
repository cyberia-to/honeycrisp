# Unimem core implementation

Root accepted the existing narrow feature/core-import plan on 2026-10-16 under
the owner's standing implementation request. The frozen PLAN's requirement for
a different-vendor review was self-authored; no such verdict exists or is claimed.
Independent Codex CODE review is the current review route under the owner's Node
ban. The historical PLAN packet and parent evidence remain unchanged.

Source parent: Honeycrisp `9cbd48070a136e24711896c930670fcff8a147fc`.
Current native-test sibling: clean committed Strata
`56aedb2d12b3126c601eb333419136d403614dbb`. Historical tests used Strata
`494df7a0973ae10a6cc1c2022e644d70449a29d6`; their records retain that identity.
New parent and candidate inputs are hashed before/after each command. The new
parent repeats the frozen E0152 consumer, owning native tests and isolated native
tests before production. The isolated consumer has no Strata dependency.

Implementation preserves default std queue behavior and adds an explicit
no-default mode: queue alloc is always available, std forwards explicitly,
runtime namespaces use core, and unit tests import their std prelude items.
All three fixture bodies are unchanged from the frozen proposal. The durable
compile-only gate adds a checked temporary-directory creation guard after root
review; its original body and the narrow bridge are retained in the resume audit.
Existing test bodies, native contracts, Cargo.lock and
dependency versions remain unchanged. Rustfmt only reordered the new core import
before the queue import in grid.rs before candidate capture.

Evidence: [resume receipt](../../../audit/2026-10-16-unimem-core/README.md).
Owner branch: `research/unimem-core`; checkout:
`honeycrisp/.worktrees/unimem-core/honeycrisp`. Agent:
`/root/foundation_resume_audit`, Codex session
`01a0d380-ada9-7ff0-8fcf-4057387fde96`.
Independent Codex source CODE review approved the delta; the separate gate
supplement and actual gate results are retained in the resume audit. Owner tests,
isolated tests, owner formatting and strict library Clippy pass. Broader inherited
workspace gates remain red; the historical API/product gates remain unmet.
Retain the checkout for root integration. No commit,
push, merge, version, tag or publication is performed by this implementation agent.
Physical-address, native Image ownership, completion and platform coverage remain
separate requirements.
