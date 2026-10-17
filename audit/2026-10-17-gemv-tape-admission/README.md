# GEMV admission and Tape alignment repair evidence

Ordinary source repair from origin
`e86310f564287a720d56e2d8204fd6c7181e6f6a`, preceding a separately reviewed acpu
version bump. The [accepted plan](../../.claude/plans/gemv-tape-admission.md)
and [static review](origin-review.md) retain scope. Source assessment and plan
commits are `616a781` and `84ea611`; the plan's ignored `.claude/` path required
an explicit forced add in the latter commit. No history was rewritten.

## acpu source epoch

Root reviewed the existing spec, three GEMV files and both integration targets
before gates. `acpu-1/sources.json` binds the complete owner and Strata inputs
at `84ea611df44ec1bb250e5e17a739d4a0d0b3118a` plus the recorded source edits.
The compact snapshot SHA-256 is
`2e07b4bab71c2e94631e2fe357fc27de0713d83fd44d37201c68889a858b7670`.
`gates.nu REPO OUTPUT TARGET acpu` ran the exact commands in each receipt under
Rust1.95.0 and strict warning flags, with locked/offline dependencies and an
external target. Complete source inventories remained equal after every gate.

All twelve commands passed. [Raw totals](acpu-1/result.json) are rederived from
the commands below at that source epoch:

| Command | Raw outcome |
| --- | --- |
| `cargo test --workspace --locked --offline` | 383 passed, 5 ignored, 30 summaries |
| `cargo test -p acpu --lib --tests --release --locked --offline` | 212 passed, 1 ignored, 8 summaries |
| `cargo test -p acpu --test gemv_admission [--release] --locked --offline` | 4 passed in each mode |
| `cargo test -p acpu --test gemv_independent [--release] --locked --offline` | 2 passed in each mode |

Format, both strict Clippy commands, release workspace build and strict acpu
docs passed. `cargo run -p acpu --example matmul --locked --offline` printed
`max error: 0.00e0` and `result: PASS`; it is a correctness smoke. Root authored
the independent integer oracle, retained unchanged as SHA-256
`d595e4799a602ba6b2f46987bac6c22d0cb60d7cf8b841900a388488272568ba`.
No known zero-allocation undefined behavior was executed on the parent source.

An exploratory `cargo +1.95.0 fmt --all` failed before formatting because PATH
selected a direct Cargo executable, which rejects `+toolchain` syntax. The
successful invocation was `rustup run 1.95.0 cargo fmt --all`; all captured gates
select the toolchain executable directory explicitly. This command-selection
error has no associated compiler/test outcome.

The GEMV repair conserves AMX instruction words, public signatures, package
versions and dependency identities. It restores accumulation through existing
NEON kernels and limits AMX overwrite dispatch to its supported tile capacity.
Timing and performance acceptance are outside this correctness audit.

The source/evidence unit is committed as
`b38bed1952a562054d29350020fcd5dea37b5ed2`. Replay it with
`nu audit/2026-10-17-gemv-tape-admission/verify.nu REPO acpu-1 b38bed1952a562054d29350020fcd5dea37b5ed2`.
The verifier binds the complete source inventory to that Git revision, checks
the commands/environments/raw hashes, and independently parses test totals.
Historical recorded paths are evidence only. `verification-acpu-1` retains the
successful replay and the explicit refusal when given the older origin source.

Tape repair and its fresh evidence follow as a separate logical commit. This
branch contains no release candidate or publication authorization.
