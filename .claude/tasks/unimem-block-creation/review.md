# Checked Block creation — review and handoff

Source commit `6715335a3283e22218ad5f762316f2c735acb56f`, parent
`6560974dcd1a77b4b2e040a2b2a4187b702ace0b`. The original nine-file source freeze
remains unchanged. Durable evidence is in
[audit](../../../audit/2026-10-09-unimem-block-creation/README.md).

Fresh different-vendor PLAN review: APPROVE, binding conditions implemented.
Brief SHA-256 `4cba3dcdb74d4da3dd689178d320930b9e237fc2bfbd2e8bab4d7d2761b66080`;
verdict `120092a0a189457d2eb06e279cb83d7d264e5f7ba34f4d10df9f4db40deafbe9`.
Fresh CODE review: APPROVE on the frozen source, no source correction requested.
Brief SHA-256 `7ef783a031bce898ba0e657c25da136146318f55b96bf1d28d3de355eb871ac5`;
verdict `ca816e4e8482ff4e9c8c67ee4ea53cb0b15416dcce31b37533e41e5f26dbd6f9`.
Reviewers ran no commands or gates; original verdicts are preserved verbatim.

## Conditions and bounds

- Copy plan with `open(&self)`; numbers only, no reservation or initialization authority.
- Correct CF widths, `addr_of!` callback addresses, immutable stack-entry dictionary,
  immediate Create ownership and returned-null handling.
- Checked request/row/extent arithmetic and actual native alignment cross-checks;
  returned extent disagreement fails before lock/publication.
- Complete failure cleanup: release before lock; unlock then release after successful lock.
- Unit-only fault seam; compile-fail E0451/E0599 checked by actual matching nightly.
- No production dependencies/version changes or broader ownership framework.

The CODE review's “full slice::from_raw_parts contract” statement overreaches.
Accepted evidence proves numeric and mapping bounds only. Initialization,
mutable-view aliasing, raw handle/import lifetime, publication and physical
pinning remain separate, as the reviewer also explicitly states. No Miri claim
is made for native framework paths, and no opaque-CF universal allocation-failure
guarantee follows from returned-null cleanup.

## Gate disposition

Isolated debug/release each pass 50 native tests and 2 doctests; Clippy, fmt,
docs/build pass. Semver checks against origin and v0.2.0 both exit 100: five
added exhaustive MemError variants. Four separate raw-FFI source breaks are
proved by compiler probes (old declarations compile, new declarations E0308).
These are intentional breaks, not waived green gates or a version bump.

Full overlay uses local nebu `76b2f264` + strata `56aedb2d` + the nine frozen
files. Its bytes match the source commit after the nebu PR merge (recorded
parent-diff exit 0). Workspace release/tests pass, 336 passed/5 ignored;
unimem build/tests/16-case bench and actually verified CPU/AMX+ANE pipeline pass.
Workspace fmt remains exit 1 and all-target Clippy exit 101, with exact baseline
parity: 16 diagnostic/location pairs and 25 test-source warning/location pairs.
The source commit therefore uses `wip:` and quotes the red API/Clippy gates.

Two specs add 51 lines/3150 bytes for the checked-creation contract; this is an
addition, not a spec-refinement claim. Raw commands, exits, outputs, frozen
hashes, compiler probes and isolated manifest/lock adaptations are durable.
The Nu prepare-only reproduction verifies committed source without compiling
dirty siblings; the record check recomputes outcomes from retained logs.

This is a local source-correctness unit. It establishes no release/rung,
GPU-import, initialization/aliasing or complete Kadek image ownership closure.
Root owns remaining integration, audit commit, PR and merge decisions.
