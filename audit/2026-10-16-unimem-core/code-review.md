# unimem no_std prerequisite — independent CODE review

Verdict: APPROVE for the reviewed source delta. No blocking source findings.

Reviewer: independent Codex agent `/root/xtask_code_review`, 2026-10-16.
Parent: `9cbd48070a136e24711896c930670fcff8a147fc`.
Checkout: `/Users/master/cyber/honeycrisp/.worktrees/unimem-core/honeycrisp`.
This is a Codex CODE review, not a different-vendor or council review. No source,
documentation or refs were edited; no tests were rerun by this reviewer.

## Correctness

- Default behavior is preserved: `unimem/default -> std -> crossbeam-queue/std`
  enables the queue's `alloc` and `crossbeam-utils/std` features. Inspection of
  the locked queue 0.3.12 and utils 0.8.21 source confirms the omitted queue
  `default` marker has no separate implementation effect. The std backoff still
  calls `std::thread::yield_now()` under prolonged contention.
- The isolated `default-features = false` mode selects alloc without runtime std.
  Its spinning backoff and allocator requirement are documented. Cargo's additive
  feature unification can restore std through another consumer; owning no-default
  tests therefore cannot by themselves establish the dependency boundary.
- Runtime edits replace standard-library reexports with their core definitions.
  Public signatures, layouts, FFI symbols, ownership, lock/unlock, native creation,
  access safety requirements and queue algorithms are unchanged. `MemError` keeps
  the same Error trait identity. Test changes import prelude items explicitly;
  existing expectations and test bodies are retained.
- The durable Nu gate compiles a separate workspace with its own panic handler,
  detecting an accidental transitive runtime std dependency. It checks normal
  dependency names, workspace package version and locked registry identities;
  uses the frozen lock with `--locked --offline`; derives the dependency path from
  the script; and uses a fresh temporary target. The inert allocator and native
  driver are never executed. Fixture and workspace locks are not rewritten.

## Evidence inspected and limits

Records are under `audit/2026-10-16-unimem-core/records/`, anchored to the source
epochs hashed below. `parent-no-std.json` records the durable gate failing E0152;
`candidate-no-std.json` records the identical gate succeeding. Candidate owner
default and no-default `cargo test -p unimem ... --locked --offline` each pass
53 tests plus 13 doctests. `candidate-isolated.json` records 44 native tests
passing in the separate consumer. Candidate owner formatting passes. Recorded
before/after source identities agree. These observations were read from actual
command receipts and output, not inferred from the implementation.

Workspace formatting and strict Clippy already fail on the parent; the inspected
Clippy output starts with `error: duplicated attribute` in acpu. The plan also
retains a blocked historical-tag API gate and unverified product/dependent gates.
Remaining candidate workspace/build/API/feature-graph records were still being
collected when this source review was issued. This approval does not classify
those gates as green, authorize a release, or establish Kadek Image completion.
Preserve their actual results and the required `wip:` treatment of red gates.

Nonblocking record correction: the retained task plan says “production unchanged”
and requires a fresh different-vendor review. In finalization, identify those as
the planning-era state/policy and state the actual current source/review lineage;
the implementation must not inherit a vendor-review claim from that stale text.

## Reviewed SHA-256 identities

| Path | SHA-256 |
| --- | --- |
| `Cargo.toml` | `7cddb0de95a53bf9182b71d7255a0263718f13c34bb581c5d0aa9d9608eec517` |
| `Cargo.lock` | `b9da0c2f21698c93fccc17695d7c9b2581e824f93e46f305434edcb541b0b678` |
| `unimem/Cargo.toml` | `1fa39e3b8f32df814965dfa5ab613e5453c0f82eb0cc73ab143787897527d1ed` |
| `unimem/CLAUDE.md` | `8dd469b33ea0303e92b6ed02fee3ba94f27b025de0bac8951b51a9a9f76cfd79` |
| `unimem/README.md` | `807c39a1e47b51c139ae92c5ecdb645c49dc1e39130ad3b7e1621863f445733a` |
| `unimem/src/block.rs` | `c663a7d573299cda3e8aec36bf515b2cc5645007840f9bc4c7687b07722a0325` |
| `unimem/src/block/creation.rs` | `f32c0f3efd10be251e83e9aa5787780b6b41f413f741eca899991044c2d03651` |
| `unimem/src/block/tests.rs` | `16a12d6edaaefef3780bf23353109664a77d7e7665cfa1d6153ea0d6462a94b8` |
| `unimem/src/ffi.rs` | `458e716934c8a16814a27fb3d0464757b914bb0e5d8c0c719c3357b8a008db9a` |
| `unimem/src/grid.rs` | `09eb44a83c3c357819779097b512744feef2ca5e8531c47124de246b0782824c` |
| `unimem/src/layout.rs` | `2b6f0054d1075a160fc887de968b827b0cf93072303ea9ad51abf29a41f730e2` |
| `unimem/src/lib.rs` | `11ffac3f6a1ef8d77dcb5ca32ed36adda1282bebd47338b5908bf9d4bc491cc5` |
| `unimem/src/tape.rs` | `7665fad68b385e94d27a52ab866cf092735281265722c335e03fb39144137975` |
| `unimem/scripts/check-no-std.nu` | `12d37297970207f930f46030026a02f7325056333b1a9b4b627cf29b8f3ae9bf` |
| `unimem/fixtures/no-std/Cargo.toml.txt` | `050ef868b26b59c312fed776a1a416827a965c03d1dda3f54f91c44d2ee07544` |
| `unimem/fixtures/no-std/Cargo.lock.txt` | `9b3453ff7a632620f0b0c3274eab3b597346d03eec0ff5023d4affe298d2a387` |
| `unimem/fixtures/no-std/consumer.rs` | `e11677d9d349c914b48854e8d0bb1a9a2b4d97c13c518fcf63f0301b606748d7` |
| `audit/2026-10-16-unimem-core/parent.json` | `952f4ac2974fa4d8b55217ce2d762baf066e586e934b9889e25cc6f336431b70` |
| `audit/2026-10-16-unimem-core/candidate.json` | `cf780b0c9abb169bbe2c68a8367db537d6a912962fd7a66721c4b93124beb23f` |
| `audit/2026-10-16-unimem-core/frozen-witness.json` | `4486c583704cb04901bf098c13742639c6c62683e3ef81d173e52f28b0cd0b45` |
