# Native image planes

This local Apple driver unit adds checked two-plane allocation and retained
IOSurface import to the existing `Block`. Supported pixel formats are explicit
video/full-range NV12 and P010. It provides native allocation identity, numeric
geometry and ownership. Kadek Image integration, literal physical addresses,
device completion and other platforms remain separate work.

Base is Honeycrisp `61714b1b6337c967b8e1a1ecc4304970352f14c7`; final capture HEAD
is plan commit `4a3079cc49ea086121324af826091c559e5d8120` with the recorded local
implementation. Strata stays `56aedb2d12b3126c601eb333419136d403614dbb` for the
existing workspace NeBu dev-dependency. Manifests, lockfile, versions and acpu
source are unchanged. This is correctness evidence, with no release or timing
verdict. Root reviewed the final source before freeze; see `source-review.md`.

## Fixed inputs and replay

`freeze.json` SHA256:
`1d3dd625fe5a5ee18528e50e40e8c395cd9b9fddad6ea7b009e5d1be3a54a1a0`.
Its 299-file compiler-input inventory SHA256 is
`e37fe7a30ff0dac0bf740833ea94de21e503e703b73a275111e4f2f1ac3ff260`.
The inventory covers both repositories' Rust, manifests/config, no_std fixture
templates/script and formatting configuration. Specs and plan have separate
authority hashes. The final ten commands bind HEAD before/after, source bytes,
runner and tool bytes, actual clean environment, streams and exit status.

On a checkout containing these source bytes and the pinned Strata checkout:

```text
nu audit/2026-10-17-native-image-planes/verify.nu REPO STRATA
```

Replay checks current compiler bytes and the complete audit seal without
executing historical code, compilers or native frameworks. Repository arguments
replace the original machine paths. It supports a later documentation/merge
HEAD with identical compiler inputs. `--live` additionally requires original
capture HEADs and tool binaries. The evidence retains tool/dependency identities;
it does not bundle compilers, the SDK or registry sources.

To reproduce commands in a fresh audit epoch, use the exact argument arrays
in `freeze.json` with the environment recorded by `run.nu`. The capture scripts
refuse to overwrite receipts. `final.nu REPO CACHE` is the finite original
launcher, not an instruction to overwrite this completed epoch.

## Acceptance

Acceptance preceded production: `acceptance-missing-api.json` records exit 101
with E0432/E0599 for the absent APIs; `acceptance-initial.tar.gz` retains those
test bytes. The parent package tests, strict Clippy and formatting passed on
base source inventory `b93cac88f527900f606390991de13b7b78eef153366e3ab0202343feb6002566`.

Current acceptance includes literal odd geometry and overflow/end bounds;
complete final-row padding and disjoint plane extents; rejected counts, formats,
element dimensions and addresses; allocation observer plus positive control;
every CF creation stage and retain/lock/validation cleanup; and compile-fail
private-field, unchecked-constructor and unsafe-import examples.

Real native tests cover all four formats at 5×3 and 16×8 in both directions:
planned IOSurface to CoreVideo, and independently CoreVideo-created IOSurface
to retained Block. They verify shared identity, actual plane dimensions/strides,
initialized samples, full padding/gaps and survival after original-owner drop.
CoreVideo metadata/sample observations use real CPU locks. No device readiness
is inferred from retaining an object or acquiring those locks.

`final-debug` and `final-release` each report 85 passing tests: 15 library,
7 access, 5 creation, 4 image, 32 roundtrip, 5 tape and 17 doctests. Their commands
are `cargo test -p unimem --locked --offline` and the same with `--release`,
under the frozen environment and source identity above.

All ten final commands exited zero. `workspace-test` reports 400 passing and
5 ignored tests under `cargo test --workspace --locked --offline`; the ignored
cases remain visible in its raw output. The other final checks were strict
unimem all-target and workspace Clippy, unimem docs, workspace formatting,
isolated own-panic-handler no_std compilation, workspace release build and the
release pipeline example. Exact arguments are in the freeze and each receipt.

## Capture history and limits

Native gates use Rust 1.95.0 with absolute `RUSTC`/`RUSTDOC` and `rustup run`,
strict warning flags, offline lockfiles, a dedicated target cache and SDK 26.4.1.
The SDK's IOSurface and CoreVideo header hashes are recorded in the freeze.
No benchmark, Miri or sanitizer result is claimed for this native FFI unit.

Development receipts remain historical: the first native success has a small
source archive; later receipts retain source inventories and runner identities.
The final inventory adds no_std templates/script and formatter configuration to
the earlier capture scope. Compiler bytes did not change during final gates.

Two launcher syntax errors occurred before compiler execution: the initial
`run.nu` condition and a multiline external invocation in `final.nu`. Exact
failed bodies are retained under `runners/`; the latter has an explicit failure
receipt. The corrected final launcher changed no frozen compiler/runner bytes.
Raw receipts preserve the acceptance red and all prior results.
