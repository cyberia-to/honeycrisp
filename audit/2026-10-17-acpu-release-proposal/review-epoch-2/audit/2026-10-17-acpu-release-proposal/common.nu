export use ../2026-10-17-acpu-package-evidence/support.nu *

export const version = '0.2.1'
export const base = '3ffc0d5e6f268d28658b8ad279b7adcfa0735fa6'
export const nox_revision = '64d091e5ad3f277cce77b88ae0de6eea4c49b4df'
export const nox_archive_sha256 = '37b40ed9894b3656315555b58afaec86058b6eb122e453244bb858a34fe1ff5b'
export const strata_revision = '56aedb2d12b3126c601eb333419136d403614dbb'
export const probe_sha256 = 'eb0e4498e99aa7de6ddcaca18ee39ec38a160ebccd322dc30a8e070a6cfae77d'

export def fresh [path: string] {
    require (not ($path | path exists)) $'output already exists: ($path)'
    mkdir $path
}

export def environment [target: string] {
    let prefix = (^rustup which --toolchain 1.95.0 rustc | str trim | path dirname)
    {PATH: ([$prefix ($env.HOME | path join .cargo/bin) /opt/homebrew/bin /usr/bin /bin] | str join (char esep)),
        RUSTUP_TOOLCHAIN: '1.95.0', RUSTC: ($prefix | path join rustc), RUSTDOC: ($prefix | path join rustdoc),
        RUSTFLAGS: '-Dwarnings', RUSTDOCFLAGS: '-Dwarnings', CARGO_BUILD_JOBS: '2', CARGO_INCREMENTAL: '0',
        CARGO_TARGET_DIR: ($target | path expand)}
}

export def snapshot [repo: string] {
    let sibling = ($repo | path dirname | path join strata | path expand)
    let head = (^git -C $sibling rev-parse HEAD | str trim)
    require ($head == $strata_revision) 'Strata revision drift'
    {owner_head: (^git -C $repo rev-parse HEAD | str trim), owner: (owner-files $repo),
        strata_head: $head, strata: (committed-owner $sibling)}
}

export def tool-bundle [directory: string] {
    let file = ($directory | path join tooling.json)
    let manifest = (open $file)
    for input in $manifest.files {
        require ((digest ($directory | path join $input.path)) == $input.sha256) 'exported tooling changed'
    }
    {manifest: $manifest, sha256: (digest $file)}
}

export def owning-gates [] {
    [
        {name: fmt, argv: [cargo fmt --all -- --check]}
        {name: clippy-workspace, argv: [cargo clippy --workspace --locked --offline -- -D warnings]}
        {name: clippy-acpu, argv: [cargo clippy -p acpu --all-targets --keep-going --locked --offline -- -D warnings]}
        {name: tests-workspace, argv: [cargo test --workspace --locked --offline]}
        {name: build-release, argv: [cargo build --release --workspace --locked --offline]}
        {name: tests-acpu-release, argv: [cargo test -p acpu --lib --tests --release --locked --offline]}
        {name: matmul, argv: [cargo run -p acpu --example matmul --locked --offline]}
        {name: docs-acpu, argv: [cargo doc -p acpu --no-deps --locked --offline]}
    ]
}

export def package-gates [] {
    [
        {name: clippy, argv: [cargo clippy --all-targets --keep-going --locked --offline -- -D warnings]}
        {name: debug, argv: [cargo test --lib --tests --locked --offline]}
        {name: release, argv: [cargo test --lib --tests --release --locked --offline]}
        {name: docs, argv: [cargo doc --no-deps --locked --offline]}
        {name: matmul, argv: [cargo run --example matmul --locked --offline]}
    ]
}

export def git-inputs [root: string, revision: string] {
    ^git -C $root ls-tree -r --name-only $revision | lines | sort
    | where {|p| not ($p | str starts-with audit/) and not ($p | str starts-with .claude/)}
    | each {|p| let body = (do { ^git -C $root show $'($revision):($p)' } | complete)
        require ($body.exit_code == 0) 'Git source is unavailable'
        {path: $p, sha256: ($body.stdout | hash sha256)}
    }
}

export def candidate-revision [root: string, revision: string] {
    let head = (^git -C $root rev-parse $revision | str trim)
    let ancestor = (do { ^git -C $root merge-base --is-ancestor $base $head } | complete)
    require ($ancestor.exit_code == 0) 'candidate does not descend from reviewed base'
    let diff = (^git -C $root diff --name-only $base $head | lines | sort)
    require ($diff == [Cargo.lock acpu/CHANGELOG.md acpu/Cargo.toml]) 'bump must contain exactly three source paths'
    let manifest = (^git -C $root show $'($head):acpu/Cargo.toml')
    let old_manifest = (^git -C $root show $'($base):acpu/Cargo.toml')
    require ($manifest == ($old_manifest | str replace 'version.workspace = true' $'version = "($version)"')) 'crate manifest extends beyond exact package bump'
    require (($manifest | from toml).package.version == $version) 'candidate version differs'
    let locked = (^git -C $root show $'($head):Cargo.lock' | from toml)
    let original = (^git -C $root show $'($base):Cargo.lock' | from toml)
    let packages = ($original.package | each {|pkg|
        if $pkg.name == acpu { $pkg | update version $version } else { $pkg }
    })
    let expected = ($original | update package $packages)
    require ($locked == $expected) 'lock changes extend beyond local acpu version'
    let source = (git-inputs $root $head)
    {head: $head, files: $source, source_sha256: ($source | to json -r | hash sha256)}
}

export def candidate [root: string] {
    let source = (candidate-revision $root HEAD)
    require ((committed-owner $root) == $source.files) 'candidate working inputs differ from committed source'
    $source
}

export def registry-identities [metadata: record, locked: record] {
    mut result = []
    for expected in [
        {name: unimem, version: '0.2.0', checksum: a91794fe345844d09b134c55fde134bc9d51593964b906dde2f667162e49918d}
        {name: strata-nebu, version: '0.1.1', checksum: f4a881cdb3622cd6b38333246f516c70e3e5c63d2bc0c0e48344d8e7b0392766}
    ] {
        let packages = ($metadata.packages | where name == $expected.name)
        let locks = ($locked.package | where name == $expected.name)
        require (($packages | length) == 1 and ($locks | length) == 1) 'duplicate/missing registry identity'
        let pkg = ($packages | first)
        let lock = ($locks | first)
        require ($pkg.version == $expected.version and ($pkg.source | str starts-with 'registry+')
            and $lock.version == $expected.version and $lock.source == $pkg.source and $lock.checksum == $expected.checksum) 'registry source/version/checksum differs'
        let features = ($metadata.resolve.nodes | where id == $pkg.id | first | get features)
        if $expected.name == strata-nebu { require ('fast' not-in $features) 'NeBu fast enabled' }
        $result = ($result | append ($expected | merge {id: $pkg.id, features: $features}))
    }
    let direct = ($metadata.resolve.nodes | where id == $metadata.resolve.root | first)
    let nox = ($metadata.packages | where name == cyber-nox)
    require (($nox | length) == 1) 'Nox identity is not unique'
    let node = ($metadata.resolve.nodes | where id == ($nox | first | get id) | first)
    require (($direct.deps | where name == nebu | first | get pkg) == ($node.deps | where name == nebu | first | get pkg)) 'direct/Nox Goldilocks types differ'
    $result
}
