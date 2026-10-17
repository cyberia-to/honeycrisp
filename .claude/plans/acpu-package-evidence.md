# acpu normalized package and published API — proposed unit

Read-only preparation, 2026-10-17. Honeycrisp canonical/origin HEAD is
`c7c9fabdb1e9a9b0be379917540355e89c899dfe`, clean. No checkout, code, manifest,
version or dependency edits have been made for this unit.

## Placement and ownership

After root approval, recreate only
`/Users/master/cyber/honeycrisp/.worktrees/acpu-delivery/honeycrisp`, branch
`chore/acpu-package-evidence`, from that committed origin/main. `/.worktrees/`
is already excluded. Reuse the retained wrapper; archive its current task.json
under the new owning audit before updating task/branch/base/retention metadata.
Keep its existing Strata symlink target `/Users/master/cyber/strata` and revision
`56aedb2d12b3126c601eb333419136d403614dbb` unchanged. Cache:
`~/.cache/cyber/honeycrisp/acpu-package-evidence/`.
Own `.claude/plans/acpu-package-evidence.md` and
`audit/2026-10-17-acpu-package-evidence/` only; Nox and production inputs stay fixed.

## Smallest executable scope

1. Acquire official acpu0.2.0 archive/index with raw argv, responses/checksums,
   and compare complete source inventory/bodies with published VCS
   `96fa4f4509d7a966cb22a4bb9e12ac753cff43ae`. Archive SHA256 is
   `81089272e2a248781b1e2c38ad7247a98bea63a602c0a445dc63e955b2035375`.
   Require current owning compiler/config/asset inventory equal committed HEAD
   before diagnostic `cargo package -p acpu --locked --offline --allow-dirty`;
   audit-only dirt is permitted. Bind all package files, normalized/original
   manifests, generated lock and VCS identity. No source overlay or manifest fix.
2. Compare the complete library API on its sole supported target,
   aarch64-apple-darwin. acpu declares no feature matrix. Produce old/current
   rustdoc JSON in separate source/target roots using pinned Rust1.95.0 and
   explicit identical extraction flags; retain warnings. Run installed
   cargo-semver-checks0.51.0 with explicit baseline/current JSON and
   `--release-type patch`. No API filtering, shared-target discovery or claimed
   behavior/performance proof. Package target inventory separately covers bins,
   examples and tests that a library API checker does not adjudicate.
3. On the untouched normalized current package, run locked/offline debug and
   release `cargo test --lib --tests`, strict docs, and strict all-target Clippy;
   retain initial failures. Run bounded real `matmul` correctness, not performance
   benchmarks. This exercises merged admission/scaled-coefficient tests with
   registry unimem rather than the workspace's local unimem. Verify inclusion
   of the coefficient test's `independent.tsv` include_str input.
4. Replay the exact existing four-test Nox contract probe against normalized
   committed Nox and acpu packages. The approved scratch consumer override
   selects only the untouched acpu archive; record its manifest/lock and call it
   a prepublication diagnostic. Require one registry unimem and one registry
   strata-nebu identity, fast off, with matching Nox/direct Goldilocks type source.
   Run debug/release; preserve original probe semantics and hash. No owning Nox
   edits; actual unpatched registry acceptance waits for owner publication.
5. Reuse prior whole-workspace gate evidence only after exact current closure
   verification, and run required owning fmt/strict Clippy/release build/workspace
   tests/real matmul before the audit commit as required by Honeycrisp rules.
   Preserve all source epochs, exact command streams and a portable capsule
   verifier. Root independent CODE review precedes the ordinary audit commit.

## Concrete packaging concern and limits

Current root Cargo.toml:41 declares NeBu with path/package but no version.
Previous normalized acpu manifest drops it; `acpu/bench/{zk,summary}.rs` use it
and remain declared examples. Thus all-target packaged checking is expected to
expose unresolved NeBu even if default `cargo package` verification passes.
Existing `fix/nebu-package-identity` only corrected the package name and explicitly
left its version unspecified; that merged repair does not solve normalization.
Measure the failure first and return a separate minimal source/manifest proposal.

Fresh official index reads still list acpu0.2.0 and unimem0.2.0 only. Unimem SHA256
`a91794fe345844d09b134c55fde134bc9d51593964b906dde2f667162e49918d`; its published
VCS metadata declares dirty=true, so its archive checksum is authoritative.
Registry strata-nebu0.1.1 SHA256
`f4a881cdb3622cd6b38333246f516c70e3e5c63d2bc0c0e48344d8e7b0392766` has empty
defaults; source Strata and registry tier versions remain separately identified.

The occupied current version0.2.0 archive remains diagnostic. An acpu-only0.2.1
bump is only a later proposal after actual package/API outcomes. Owner alone
merges a bump, tags or publishes. Nox's later consumer repair must raise its acpu
minimum to0.2.1, not merely refresh a lock; none of those edits belongs here.
