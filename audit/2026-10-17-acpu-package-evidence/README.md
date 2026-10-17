# acpu package delivery evidence — 2026-10-17

Source repair `64866774f221112a6e83a55e557902a79b68722f` retains the existing
NeBu0.1.1 dev dependency during Cargo normalization and clears equivalent
example syntax. Package/library versions, kernels, AMX words and root lock stay
unchanged. Accepted [plan](../../.claude/plans/acpu-package-evidence.md),
[complete source diff](source-repair-final.patch), [literal proof](literal-proof/run.stdout)
and [root source verification](owner-verification-root.json) retain the scope.

| Check | Exact evidence and outcome |
| --- | --- |
| Owning gates | [owner-2](owner-2/): eight green commands from [owner-gates.nu](owner-gates.nu), including strict all-target acpu Clippy; closure `c02d2c897dbfadfa9e8fbbe398b5c261efc9252b21c5c459e0cd6308a2119e53`. |
| Committed normalized package | [package-3](package-3/identities.json): `cargo package -p acpu --locked --offline --allow-dirty`, after complete owning bytes equal HEAD; SHA256 `acd3929c3472443c92b977ed789e9494715bfbd9278a8d1e52fe6d1fdb9938a3`. |
| Package checks | [checks-2](checks-2/): strict `cargo clippy --all-targets --keep-going --locked --offline -- -D warnings`; debug/release `cargo test --lib --tests` each 196 passed, one ignored; strict docs; real matmul PASS. All use locked/offline inputs and Rust1.95.0. |
| Published library API | [api-2](api-2/comparison.stderr): explicit old/current rustdoc JSON, sole supported aarch64-apple-darwin target, no declared crate features; cargo-semver-checks0.51.0 patch comparison: 229 pass, 31 skip. |
| Targets | [metadata comparison](target-inventory/result.json): all 38 targets, including 32 examples, conserved across owning source and both normalized package epochs. |
| Nox contract | [consumer-2](consumer-2/result.json): normalized Nox `64d091e5ad3f277cce77b88ae0de6eea4c49b4df`; exact existing four-test probe passes debug/release (48 NTT, 48 multilinear, noncanonical residue controls, 40 backend comparisons). |

The consumer's isolated `[patch.crates-io]` selects the exact untouched acpu
archive. Its [manifest](consumer-2/consumer-Cargo.toml.txt), lock and metadata
prove one registry unimem0.2.0 and one strata-nebu0.1.1 identity, NeBu fast off,
and identical direct/Nox Goldilocks type source. This is a prepublication
diagnostic; actual unpatched registry acceptance still awaits owner publication.
The local Strata target remains `56aedb2d12b3126c601eb333419136d403614dbb`;
source and registry dependency identities remain separately recorded.

Official acpu0.2.0 SHA256 is
`81089272e2a248781b1e2c38ad7247a98bea63a602c0a445dc63e955b2035375`, with VCS
`96fa4f4509d7a966cb22a4bb9e12ac753cff43ae`. The complete shipped source inventory
matches that revision. [Package identities](package-3/identities.json) retain
the fresh official index response, archive/VCS and normalized manifest/lock.
Registry unimem/NeBu checksums are pinned in the consumer result; unimem's
published VCS declares dirty=true, so its archive checksum is authoritative.

Historical reds remain verbatim: [checks-1](checks-1/) contains omitted NeBu and
35 initial lint findings; [owner-1](owner-1/) contains ten newly reachable lints.
`package-1`, `consumer-1` and `verification-1` preserve tooling mistakes and
reproductions (generated lock inventory, explicit TOML parsing, quoted filename).
The earlier green API/test epoch remains distinct. Matmul is a correctness
smoke, with no performance claim. API comparison does not establish semantics.

Portable replay: `nu audit/2026-10-17-acpu-package-evidence/verify.nu REPO FRESH_TMP`.
[bundle.json](bundle.json) binds the compressed package/API/probe capsule;
the verifier resolves current repositories from arguments and retains historical
recorded paths as evidence. [Wrapper history](wrapper-before.json) and
[current context](wrapper-current.json) preserve task/dependency lifetime.

Next release proposal: acpu-only0.2.1, followed by Nox requiring at least0.2.1
and a fresh unpatched registry check. Owner alone merges the version bump,
tags or publishes. No release-ready verdict or publication occurs in this unit.
