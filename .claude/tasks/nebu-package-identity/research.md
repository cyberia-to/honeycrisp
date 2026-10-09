# nebu package identity — research

2026-10-09 · serves: Kadek foundation row 6 native storage, via unimem's
required full-workspace gates; discovered from `unimem-block-creation`.

`git ls-remote --symref origin HEAD` pins honeycrisp main to
`9593c7218e14e2b5b816d9008b45ab60b9580cf3` and strata main to
`56aedb2d12b3126c601eb333419136d403614dbb`. Root and acpu CLAUDE.md and
cyberia/dev.md were read. Original honeycrisp is dirty; only committed archives
and the owned `fix/nebu-package-identity` worktree are task inputs.

Honeycrisp workspace dependency `nebu` points at `../strata/nebu/rs` with
package `cyb-nebu`; the pinned source names `strata-nebu` 0.1.1, library `nebu`.
Only acpu's dev dependency uses it. Its two benchmark examples call
`Goldilocks::new`, `ntt::ntt` and `ntt::intt`, all present in the pinned source.
There is no production API use. Strata's four path tier crates inherit 0.1.1.
The checked-in lock still has `cyb-nebu` 0.1.0 and a registry acpu dependency.
Current strata-nebu defaults are empty; `fast` explicitly enables registry
acpu. Keeping the existing dependency's unspecified feature policy now selects
those current defaults. This unit will not assert unchanged benchmark speed
or enable acceleration implicitly.

Executed baseline: `nu /tmp/kadek-nebu-closure-baseline.nu` archives both exact
commits into `/tmp/honeycrisp-nebu-closure.GUEZG7/{honeycrisp,strata}` and runs
`cargo metadata --format-version 1 --locked --offline`. Exit 101:
`error: no matching package named cyb-nebu found`, searched the isolated strata
path, required by local acpu 0.2.0. Raw stderr/exit are retained there. This is
an executed resolution failure, before compilation. Two initial Nu parser/
command invocation errors occurred before any archives or Cargo; corrected
script uses quoted revisions and external `^mktemp`.

Source checks: `git grep -n 'path =' <strata-pin> -- '**/Cargo.toml'`,
`rg -n '\bnebu\b' acpu --glob '*.rs'`, reads of both workspace manifests,
strata `nebu/rs/{Cargo.toml,lib.rs,field.rs,ntt.rs}` and tier manifests.
All non-registry paths close over these two complete repositories. The package
fix does not change dependency path, version requirements, public APIs or specs.
No build/test/Clippy/semver result is known yet. Full workspace may expose
independent existing code errors once package resolution works.

Read-only soft3 registry pin `94af9de717658b3e912bcfccc85409fd29b23a10`
places honeycrisp in nox, layer 1. This is not a current origin/rung validation.
No release pin, version, tag or publication is in scope.

Correction-round inspection: `rustc -Vv` reports1.95.0
`59807616e1fa2540724bfbac14d7976d7e4a3860`, LLVM22.1.3, aarch64-apple-darwin;
Cargo1.95.0 `f2d3ce0bd`. `git -C ../strata rev-parse HEAD` is the above pin and
`git status --porcelain` is empty. Authoritative gates still use archives.
[Cargo resolver documentation](https://doc.rust-lang.org/cargo/reference/resolver.html#features)
distinguishes lock resolution from selected compile features; nonmember
dev-dependencies are ignored. Its resolver-versions section gives authority
to the consuming workspace's resolver, so strata's resolver3 is not inherited.
Toolchain >=1.89 is required by that source's package declaration. No Cargo
package-identity repair has yet been applied or gated.
