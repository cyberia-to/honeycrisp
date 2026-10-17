# acpu 0.2.1 owner proposal — validated branch evidence

Source-only bump `2c70b3b929c791ca489bba12d2f5a6daef7d43e7` descends from
reviewed origin/main `14373cc8a0e18302751844579d94d1f8f80bd8b1`. Its complete
diff is `acpu/Cargo.toml`, Cargo-generated root `Cargo.lock`, and
`acpu/CHANGELOG.md`. Workspace and sibling package versions remain unchanged.
No bump merge, tag or publication has occurred. This branch archive is a
prepublication diagnostic; owner delivery requires fresh default-branch inputs
after the owner merges the bump.

The capsule is `candidate-1/evidence.tar.gz`, SHA-256
`d14d591e4e36e5e489494fb1b2c51f49bd1281b2f588366a3c4d3ac666a0d288`.
`candidate-1/bundle.json` lists every retained file and checksum. The untouched
normalized acpu archive has SHA-256
`74967ea98831cd29d69fd3cef0cc08a8236f66d8d2fdcb2e99223428488d1159`, clean VCS
identity at the bump commit, and source bytes verified against that Git revision.

## Results and exact commands

All owning and acpu package results below belong to the bump commit above.
Commands use Rust 1.95.0, `RUSTFLAGS=-Dwarnings`, `RUSTDOCFLAGS=-Dwarnings`,
locked offline resolution and explicit external target directories. Raw
commands, environment, streams and source inventories are inside the capsule.

| Scope | Commands and observed result |
| --- | --- |
| Owning eight gates | `cargo fmt --all -- --check`; `cargo clippy --workspace --locked --offline -- -D warnings`; `cargo clippy -p acpu --all-targets --keep-going --locked --offline -- -D warnings`; `cargo test --workspace --locked --offline`; `cargo build --release --workspace --locked --offline`; `cargo test -p acpu --lib --tests --release --locked --offline`; `cargo run -p acpu --example matmul --locked --offline`; `cargo doc -p acpu --no-deps --locked --offline`. All returned zero. |
| Raw owning tests | Workspace: 388 passing, five ignored, 31 summaries. Acpu release: 212 passing, one ignored, eight summaries. These totals are rederived from the two test commands above. |
| Untouched normalized package | `cargo clippy --all-targets --keep-going --locked --offline -- -D warnings`; `cargo test --lib --tests --locked --offline`; the same tests with `--release`; `cargo doc --no-deps --locked --offline`; `cargo run --example matmul --locked --offline`. All returned zero. Both test modes: 212 passing, one ignored, eight summaries. Both owning/package matmul runs report zero maximum error and PASS. |
| Package targets | `cargo metadata --no-deps --format-version 1 --locked --offline`, both owner and normalized package, plus fresh extracted-archive replay: identical 40 targets, including 32 examples and six integrations. `gemv_admission` and `gemv_independent` are present. |
| API comparison | Separate complete-library rustdoc JSON for official acpu0.2.0 and this candidate, target `aarch64-apple-darwin`, Rust 1.95.0; explicit `cargo-semver-checks 0.51.0 check-release --baseline-rustdoc … --current-rustdoc … --release-type patch --color never`: 229 pass, 31 skip, no semver update required. Exact binary hashes, flags, artifacts and commands are retained. |
| Nox diagnostic | Unchanged four-test probe from Nox `64d091e5ad3f277cce77b88ae0de6eea4c49b4df`, normalized Nox archive SHA `37b40ed9894b3656315555b58afaec86058b6eb122e453244bb858a34fe1ff5b`; `cargo test --test source_contract --locked --offline` and the same command with `--release`: four pass in each mode. |

The Nox scratch manifest uses only an explicit candidate acpu patch. Metadata
proves one registry unimem0.2.0 and one registry strata-nebu0.1.1, NeBu fast off,
and the same direct/Nox Goldilocks type source. This is not unpatched registry
acceptance and makes no owning Nox dependency change. The normalized acpu crate
still reexports registry unimem0.2.0's `Tape`; it does not deliver the owning
workspace's later Tape repair. Strata remains at
`56aedb2d12b3126c601eb333419136d403614dbb`.

After this consumer completed, Nox master advanced to
`262de214521c47d18cd539040280f22e4b1fd8a4` through PR39, incorporating the
production/CLI changes merged by PR36. The retained diagnostic continues to
describe only fixed Nox `64d091e5ad3f277cce77b88ae0de6eea4c49b4df` and its
unchanged archive/probe. Live release adoption must validate the latest Nox
source separately; this evidence is not relabeled as current-Nox acceptance.

## Preserved tooling epochs

Owning, acpu package and API production used exact tools exported from
`3b875c9850f8ebe232a6650aaff7913a28773803`, manifest SHA
`5c74d87123d860ce25800f300cac8f2425770d6ba9bf57a33489603000bb0811`.
The package runner stopped after its successful commands at a metadata-argv
type error. Its missing `result.json` remains absent. The original consumer
stopped before tests because a Git-output assignment stripped the probe's final
newline. Nu retention also required parentheses around a multiline predicate.
All failures, raw receipts, original tools and intermediate capsules remain.

The reviewed repairs are committed at
`e791fefaf05c612d10d96f201e9d9460b20d502f`, manifest SHA
`78f184494b7095f5804202a9cd661e1975b945577e8f8f705774e5aade170b9b`.
The successful consumer and validator use this actual bundle. No owning/acpu
package/API gate was repeated; fresh consumer execution includes its own Nox
normalization command. Existing successful receipts were copied byte-for-byte
into the final records; original records remain unchanged.

The capsule retains original production tools under `tools/` and repaired
consumer/validator tools under `records/validation-tools/`. The verifier checks
each bundle against its own Git commit and requires its executing file to come
from the declared validator bundle. It derives package success from commands,
raw counts, source/archive/VCS checks and fresh target metadata, not the absent
package summary. `verification-candidate-1` retains pack/unpack and successful
relocated replay using the canonical repository as the explicit Git argument.

After unpacking to an unused external directory `UNPACK`, run the committed
exported verifier as follows, with a separate unused extraction directory:

```text
nu UNPACK/records/validation-tools/audit/2026-10-17-acpu-release-proposal/verify.nu REPO UNPACK/records UNPACK/tools UNPACK/records/validation-tools UNPACK/records/validation-tools FRESH_TMP
```

Root independent replay and the owner's merge/publication decision remain the
handoff boundary. The independent GEMV test stays byte-identical at SHA
`d595e4799a602ba6b2f46987bac6c22d0cb60d7cf8b841900a388488272568ba`.
