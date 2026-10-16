# Unimem core dependency boundary

Stage: active; evidence: measured. Serves Kadek foundation row 6 and its native
Image dependency, within Cyb. Source is honeycrisp origin/main
`9cbd48070a136e24711896c930670fcff8a147fc`; owned worktree `honeycrisp-unimem-core`.
Production and specifications remain unchanged. The owner's original worktree
and open Honeycrisp PR11 are separate inputs and remain untouched.

## Current boundary

Unimem has no crate feature policy or `no_std` attribute. Its runtime `std` uses
are pointers, slices, atomics, formatting, error traits, marker types, memory
operations and FFI types, all available in core on tested Rust 1.95. Test-only
fault injection also uses Vec, HashMap, RefCell and thread-local storage.

The normal external dependencies are mutants 0.0.3 (host procedural macro) and
crossbeam-queue 0.3.12, through crossbeam-utils 0.8.21 in the workspace lock.
Crossbeam queue enables std by default; its alloc feature exports SegQueue on
targets with pointer atomics. Unimem's private Grid is the workspace's only
direct queue consumer. Crossbeam's no_std backoff spins where its std mode can
yield the thread. Disabling std globally would therefore change default runtime
behavior. Preserve that default and expose an explicit no-default-features path.

`cargo check -p unimem --lib --locked --offline` passes on the exact parent with
Rust 1.95 and `-Dwarnings`; capture `/tmp/unimem-core-parent-check-v1.*`.
The external compile-only binary `/tmp/unimem-core-consumer-parent` is no_std,
has its own panic handler and inert allocator, and references `Block::plan`.
`cargo check --locked --offline --manifest-path <consumer>/Cargo.toml` fails
with E0152: duplicate `panic_impl`, first defined in std through unimem. This
is a compiler dependency witness, not execution of the inert allocator or driver.
Capture `/tmp/unimem-core-parent-consumer-v2.*` uses the same locked queue/utils
versions as the workspace. Initial v1 resolved newer cached queue/utils and is
retained separately; the temporary consumer lock alone was corrected. The project
manifest, dependency versions and lock were not edited.

## Scope and unresolved requirements

The driver remains Apple Silicon/IOSurface as the owning CLAUDE specifies.
Framework linkage, supported platform set, raw access safety and all native
creation/destruction operations remain unchanged. A no_std library may still
call its platform frameworks and require an allocator for Grid.

This prerequisite removes one runtime library dependency under an explicit
feature selection. It does not implement Kadek's pooled native Image owner,
completion/readback, non-Apple backends, resident-memory guarantees or a physical
address. The pending Kadek address-contract question is independent of it.
PR11's unmerged platform modules predate current checked creation and unsafe
view contracts; adopting that whole branch is outside this unit.

## Measured parent and primary inputs

Artifacts: `/tmp/unimem-core-research`, indexed by `sources.json`,
`consumer-inputs.json`, `registry-acquisition.json` and `records/`. Build inputs
are Git exports of Honeycrisp 9cbd480 above and Strata
`494df7a0973ae10a6cc1c2022e644d70449a29d6`; fresh origin/main observations matched.
All 734 file modes, Git blobs and bodies were verified at extraction, and hashes/
modes again after records 01–15. These are boundary checks, not per-run guards.
No original sibling working tree was a compiler input.

The private offline registry contains 168 official archives, 28,540,128 bytes,
verified against the exact workspace lock, with zero downloads. Crossbeam queue,
utils and mutants primary bodies match those archives (12/31/6 files). Queue std
forwards alloc and utils/std. Utils' Backoff::snooze yields only with std and spins
without it. Neither package consumes the removed `default` marker via source or
build-script cfg/environment queries. Utils' build script configures atomics and
sanitizer support; host build-script/proc-macro std is allowed by this runtime
boundary.

Numbered commands use Rustup 1.95.0, jobs 2, incremental off, private Cargo home
and denied Rust/rustdoc warnings. Each receipt retains argv/env/times/raw streams.

| Parent record / command | Observed result |
|---|---|
| 02 `metadata --locked --offline` | Exact committed closure resolved. |
| 03 `test -p unimem --locked --offline` | 53 tests and 13 doctests passed. |
| 04 standalone no_std `check --locked --offline` | E0152 duplicate panic_impl; std enters through unimem. |
| 05 standalone std `check --locked --offline` | Passed. |
| 06 isolated native `test --locked --offline` | 44 verbatim existing integration tests passed. |
| 07, 13 feature trees | Queue default/std/alloc and utils/std confirmed. |
| 09 `fmt --all --check` | Existing diff, first aruminium/benches/objc2.rs:300. |
| 10 strict workspace Clippy | Four ACPU errors; first duplicated attribute at gemm/mod.rs:1094. |
| 11 locked/offline release workspace build | Passed. |
| 12 locked/offline workspace tests | Existing example-name warnings, then unused std::time::Instant at acpu/bench/crypto.rs:7; tests never run. |
| 14 locked/offline release pipeline | Exit 0; reference/unimem verification both true. |
| 15 locked/offline owning bench | Exit 0. Raw execution evidence, no performance guarantee/regression verdict. |
| 16 exact-tag locked metadata; 17 owning semver against tag | Both stop at absent historical ../nebu/rs/Cargo.toml. No compatibility result. |

Actual last tag: v0.2.0 `96fa4f4509d7a966cb22a4bb9e12ac753cff43ae`;
all 224 exported files verified, no sibling/manifest substitution. The committed
block-access audit already records intentional unsafe/exclusive-receiver breaks;
that historical evidence does not close this actual tag gate.

Preparation failures remain: record 01 used an unpacked cache without vendor
checksum files; 08 split tree-edge arguments. Earlier Nu CLI/parser failures
occurred before Cargo; they are described separately, not counted as gates.
Root's initial consumer resolved newer queue/utils; its v2 and every new consumer
use the workspace versions. Production manifests/lock remain unchanged.

## Durable discriminator

Owning tests have reverse dev-dependencies on ACPU/Rane/Aruminium, which request
default unimem. Thus `test -p unimem --no-default-features` can re-enable std.
The standalone own-panic-handler consumer is the decisive dependency witness.
The isolated native consumer runs unchanged tests without reverse dev-dependencies;
its std test harness does not itself enable unimem's std feature.

Proposed Nu gate and `.txt` templates: `/tmp/unimem-core-gate-draft/unimem/`.
It derives the owning path, checks three locked registry identities plus unimem
version/dependency names, uses a fresh temporary target and caller toolchain/cache,
and runs only `cargo check --locked --offline`. The inert allocator and Block::plan
are never executed. Only that temporary directory is deleted.
`durable-parent-v1` measures E0152 through the exact script on a parent copy.
It uses caller PATH's Homebrew Rust 1.95 (`durable-tools.json`), distinct from the
numbered Rustup runs. `durable-drift-v1` changes one queue checksum nibble in a
separate fixture copy and fails before Cargo with the named dependency-drift error.

## Radius and remaining delivery

Soft3 registry `2661a79db1df4c2c3881f98462332bf2641f5c68` puts Honeycrisp in
Nox (`part_of = "nox"`). Cyb is an actual reverse path consumer and the product
served by this Kadek prerequisite. Root facade, ACPU, Rane and Aruminium request
default unimem; only unimem directly consumes queue. Current Nox and Strata's
optional ACPU paths use registry dependencies, so their ordinary builds do not
exercise this unpublished driver overlay.

`inventory/` pins manifest observations for Nox, Erga, Cyb, Kadek, Strata and the
previously identified Xena/Bita/Mir/Glia/Zoya/Perla/Evy/Nika/Trisha consumers. The
latter are committed local origin-tracking observations, not fresh remote/build
results. Historical audit manifests are not live declarations. Current Kadek has
no live Honeycrisp Cargo dependency. This inventory is not a global completeness
claim and does not prove downstream compatibility.

The additive feature policy reaches at least minor/ring 1. Rung product/direct-
dependent suites, specification/Kelvin checks and tag compatibility remain
unverified; workspace reds remain. This is a `decision` ordinary source unit under
standing owner implementation/review/merge authorization, never a release/bump or
full phase closure. No wider unrelated repair is proposed.
