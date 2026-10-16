# Preserve default behavior and expose unimem without std

Review-ready; production unchanged at `9cbd48070a136e24711896c930670fcff8a147fc`.
Serves Kadek phase 1 / native Image dependency. Fresh different-vendor PLAN review
and substantive finding resolution precede production; fresh CODE approval
precedes ordinary source acceptance. No version, tag, release or publication.

1. Freeze the exact parent, existing native tests, compile-only consumer and
   crossbeam feature/source inputs → verify the existing driver builds; the
   explicit no-default consumer fails E0152, while its ordinary std control builds.
   Keep initial newer-cache resolution separate from the final workspace-pinned
   consumer. Preserve literal consumer source and lock for the candidate. The
   parent already passes 53 tests + 13 doctests and 44 isolated native tests;
   the standalone no_std consumer and proposed durable gate both fail E0152.
2. Give unimem `default = ["std"]` and `std = ["crossbeam-queue/std"]`; request
   crossbeam-queue with default features disabled and alloc enabled at its workspace
   declaration, retaining version requirements/lock → verify default dependency
   features preserve the prior std-enabled queue/utils behavior, and the isolated
   no-default target closure contains no runtime std dependency. Cargo feature
   unification can re-enable std when another consumer requests it; document this.
3. Add `#![no_std]`, use core for existing runtime namespaces, and import test-only
   std/prelude items explicitly → verify no changed FFI signatures, safety contract,
   allocation algorithm, public types or native layout; the frozen no_std consumer
   now compiles with the same source/lock. Its inert allocator remains compile-only.
   Add test-only `extern crate std` and explicit `thread_local`, `vec`, `Vec`
   imports; preserve existing test bodies/events. Integration tests/examples are
   separate std crates. Runtime unimem itself needs no std namespace import.
4. Install the frozen Nu gate and three fixture files → verify parent E0152,
   candidate pass, checksum-drift rejection before Cargo, and unchanged workspace/
   template locks. Derive paths from the script; respect caller Cargo/toolchain/
   CARGO_HOME. Compile in one temporary target, clean only that directory, never
   execute the inert allocator/driver. Compare locked queue/utils/mutants identities,
   unimem version and normal dependency names. Offline cache misses remain real
   prerequisite failures; no silent pin updates. Add the named gate to CLAUDE.md.
5. Run native owning tests/docs under default and no-default features, the existing
   native examples/benches, fmt and strict workspace Clippy/build/tests → retain
   actual commands, source epochs and any pre-existing broader failures. Bench
   execution is not a new numerical performance guarantee. Do not silently replace
   native calls with mocks to claim a no_std runtime result. Also run the isolated
   44 native tests with identical bodies/lock and only the dependency path changed.
   Owning no-default tests can unify std through reverse dev-dependencies and are
   insufficient alone. Retain default, explicit std, isolated no-default and owning
   test feature trees. Compare actual std/alloc cfgs; the unused queue `default`
   marker can disappear. Source inspection establishes its lack of cfg consumers.
6. Check public API from separate parent/candidate rustdoc JSON and actual last tag
   availability; record additive feature policy and reverse consumers → preserve
   real gate gaps. Fresh CODE review sees the complete small delta, exact controls
   and compact audit. Commit/push an ordinary PR under standing owner authorization;
   no Image phase completion or delivery claim follows. Select the explicit unimem
   package to avoid preexisting audit manifest ambiguity. Preserve actual disjoint
   compilation roots/input identities. The exact tag gate already fails at the
   historical NeBu path and stays unmet; a repaired tag harness is no substitute.

Files: root Cargo.toml (queue feature defaults only), unimem/Cargo.toml,
unimem/src/{lib,block,ffi,tape,grid,layout}.rs, block/creation.rs and the minimum
test-only imports in block/tests.rs. Existing tests keep their expectations.
One concise README usage note explains Apple-only no-default operation and the
allocator/feature-unification boundary; canonical specifications remain unchanged.
Additional files: unimem/CLAUDE.md, unimem/scripts/check-no-std.nu, and
unimem/fixtures/no-std/{consumer.rs,Cargo.toml.txt,Cargo.lock.txt}. The templates
are never repository Cargo manifests. Frozen proposed bodies are in
`/tmp/unimem-core-gate-draft/unimem/`; no production copy is installed before review.
Research, reviewed plan and observations stay in task/audit directories. Keep
replay tools in Nu/Rust; temporary historical capture helpers are evidence.

Five likely failures and their checks: transitive std remains (own panic-handler
consumer); default queue backoff changes (feature graph equality); test prelude
breakage (both feature-mode native suites); namespace replacement changes public
traits or FFI (full diff/API comparison); no_std gets confused with platform or
native ownership closure (explicit remaining requirements in research/report).

The core-only feature mode may spin under queue contention where std yields;
that is the dependency's explicit no_std policy, not default-mode equivalence.
Grid remains an allocating legacy utility, with no new memory/work bound. Block
still exposes raw capabilities; no alternate storage type or synchronization layer
is introduced. If a real gate requires a wider semantic change, stop this source
unit at the measured boundary and bring that change into its own reviewed plan.

Required candidate commands (locked/offline where Cargo supports them): owning
`build`, `test` default/no-default, standalone durable check, isolated native tests,
`bench`, release pipeline; workspace `fmt --all --check`, strict Clippy, release
build, tests; explicit parent API comparison. Use separate parent/candidate compiler
targets, jobs 2/incremental off, declared compiler/warning flags. Preserve original
baseline records 01–17 including preparation errors. Repeat an affected gate only
for a source/failure/input change. No performance threshold or benchmark improvement
claim is introduced by this namespace/feature boundary.

Computed floor: additive feature policy at least minor/ring 1. Registry Honeycrisp
belongs to Nox; concrete reverse consumers include Cyb and the pinned manifest
inventory in research. Rung product/dependent/specification/Kelvin gates remain
unverified, tag comparison blocked, workspace fmt/Clippy/tests red. Keep `decision`
and any required `wip:` first-error commit wording under standing ordinary-source
authorization. No dependency versions, Cargo.lock, sibling pins, FFI signatures,
allocation/access semantics or release actions change. Final CODE review receives
these limitations, not an inferred all-green or full-native-Image claim.
