# GEMV and Tape admission before release re-anchoring

Root approved this ordinary repair on 2026-10-17 after independently reviewing
`e86310f564287a720d56e2d8204fd6c7181e6f6a`. Same checkout, branch
`fix/gemv-tape-admission`; Strata stays at
`56aedb2d12b3126c601eb333419136d403614dbb`. Preparation commit
`9e6cffe699278059c8940f13940727b10569e13e` is pushed and preserved. No version,
public signature, specification structure or AMX instruction encoding changes.

1. Reconcile the existing acpu specification's gemm section with the existing
   `matvec_f32` accumulation and `matvec_f32_set` overwrite contracts. Admit
   checked dimensions before output writes; define empty results. Fix GEMV
   broadcast allocation checks, all-tile first-K initialization and tile limits;
   use the existing NEON path for accumulation and unsupported AMX dimensions.
   Preserve the pure-assembly requirement that K is divisible by eight.
   Regression tests cover invalid dimensions without mutation, empty shapes,
   repeated data, tails/row limits and nonzero output. Root reviews the source
   before final gates; commit this acpu logical unit separately.
2. Reconcile Tape's existing allocation pseudocode with its requested-alignment
   contract. Round the absolute address with checked arithmetic; refuse excess
   extent without moving the cursor. Assert actual returned pointer alignment,
   including alignment beyond the backing page, and test invalid alignment,
   arithmetic refusal and exhaustion. Root reviews before final gates; commit
   this Tape logical unit separately.

Each source unit runs these exact fresh owning gates, using Rust1.95.0,
`RUSTFLAGS=-Dwarnings`, `RUSTDOCFLAGS=-Dwarnings`, locked/offline Cargo inputs
and explicit targets below `~/.cache/cyber/honeycrisp/acpu-0.2.1/admission-repair`:

```
cargo fmt --all -- --check
cargo clippy --workspace --locked --offline -- -D warnings
cargo clippy -p acpu --all-targets --keep-going --locked --offline -- -D warnings
cargo test --workspace --locked --offline
cargo build --release --workspace --locked --offline
cargo test -p acpu --lib --tests --release --locked --offline
cargo run -p acpu --example matmul --locked --offline
cargo doc -p acpu --no-deps --locked --offline
```

The acpu unit also runs its focused regression integration target in debug and
release. The Tape unit also runs `cargo test -p unimem --locked --offline`,
`cargo test -p unimem --release --locked --offline`, and
`nu unimem/scripts/check-no-std.nu`; that compile-only script owns its temporary
target and deletes it. Receipts retain exact commands, environment, complete
before/after owning and Strata input identities, and raw outcomes in `audit/`.
Known zero-allocation undefined behavior will not be executed as a reproduction.

Fresh raw test totals belong to these source epochs. After ordinary repairs
reach origin through an authorized merger, root reviews a newly anchored
release plan/tool closure. The eventual bump still changes exactly
`acpu/Cargo.toml`, root `Cargo.lock` and `acpu/CHANGELOG.md`. Owner-only bump
merge, tags and publication remain separate.
