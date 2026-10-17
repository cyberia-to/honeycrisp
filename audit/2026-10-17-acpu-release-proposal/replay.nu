use common.nu *

# Fixed binaries independently retained by merged PR21, api-2/tools.json.
# These are macOS arm64 build identities; recorded paths are never executed.
export const api_tools = [
    {name: rustc, sha256: b829b733131d4e1673eeebd1f34d06ae1e9ff4977b051313cf42e2a9e79ecf1c, version: 'rustc 1.95.0 (59807616e 2026-04-14)'}
    {name: rustdoc, sha256: 59bc6a1caf617032284a2edc2aa03916396e70589c435a8b51e3957887f90ce5, version: 'rustdoc 1.95.0 (59807616e 2026-04-14)'}
    {name: cargo, sha256: c512bff73c86143b557463f021d0c3d5b0490d97d65040ba59ea2b3427784758, version: 'cargo 1.95.0 (f2d3ce0bd 2026-03-21)'}
    {name: cargo-semver-checks, sha256: a4fcf9eae2dd1ee88f91dc71f1236b49faa7d90165d9c326dbf2f1fb7dd41c18, version: 'cargo-semver-checks 0.51.0'}
]

export def checked [folder: string, name: string, argv: list<string>, strict: bool] {
    let r = (open ($folder | path join $'($name).json'))
    require ($r.command == $argv and $r.exit_code == 0 and $r.expected_exit_code == 0) $'command/outcome differs: ($name)'
    for stream in [stdout stderr] {
        require ((digest ($folder | path join $'($name).($stream)')) == ($r | get $'($stream)_sha256')) 'raw stream differs'
    }
    if $strict {
        require ($r.env.RUSTUP_TOOLCHAIN == '1.95.0' and $r.env.RUSTFLAGS == '-Dwarnings'
            and $r.env.RUSTDOCFLAGS == '-Dwarnings') 'strict pinned environment differs'
    }
    $r
}

export def api-environment [envs: record, binaries: list] {
    let rustc = ($binaries | where name == rustc | first | get path)
    let rustdoc = ($binaries | where name == rustdoc | first | get path)
    let cargo = ($binaries | where name == cargo | first | get path)
    require (($rustc | path dirname) == ($cargo | path dirname) and ($rustdoc | path dirname) == ($cargo | path dirname)
        and $envs.RUSTC == $rustc and $envs.RUSTDOC == $rustdoc
        and ($envs.PATH | split row (char esep) | first) == ($cargo | path dirname)
        and $envs.RUSTUP_TOOLCHAIN == '1.95.0' and $envs.RUSTC_BOOTSTRAP == '1'
        and $envs.RUSTFLAGS == '' and $envs.RUSTDOCFLAGS == '') 'API executable/environment binding differs'
}

export def api-binaries [folder: string, api: record] {
    require (($api.binaries.name | sort) == [cargo rustc rustdoc]) 'API binary inventory differs'
    let binaries = ($api.binaries | append ($api.semver | insert name cargo-semver-checks))
    for expected in $api_tools {
        let binary = ($binaries | where name == $expected.name | first)
        require ($binary.sha256 == $expected.sha256 and ($binary.path | path basename) == $expected.name) 'API binary hash/name differs'
        let name = (if $expected.name == cargo-semver-checks { 'tool' } else { $'tool-($expected.name)' })
        let r = (checked $folder $name [$binary.path --version] false)
        require ($r.cwd == $api.root and (open --raw ($folder | path join $'($name).stdout') | str trim) == $expected.version) 'API tool version/cwd differs'
        api-environment $r.env $api.binaries
    }
    $binaries
}

export def test-summary [raw: string] {
    let lines = ($raw | lines | where {|line| $line | str starts-with 'test result: '})
    let rows = ($lines | parse -r '^test result: ok[.] (?<passed>[0-9]+) passed; (?<failed>[0-9]+) failed; (?<ignored>[0-9]+) ignored; (?<measured>[0-9]+) measured; (?<filtered>[0-9]+) filtered out;')
    require (($rows | length) > 0 and ($rows | length) == ($lines | length)) 'test summary missing or failed'
    for row in $rows { require ($row.failed == '0' and $row.measured == '0' and $row.filtered == '0') 'tests failed, measured or filtered' }
    {rows: ($rows | length), passed: ($rows.passed | into int | math sum), ignored: ($rows.ignored | into int | math sum)}
}

# Counts rederived from PR25 tape-1/tests-{workspace,acpu-release}.stdout,
# whose complete compiler inventory equals base 14373cc8:
# 3f9e1b15736009afd9daefecebcfa79dfabc8ddd1eaa5deea91c9acc97f3220a,
# aaeeb1c08a3c813b96e8a4615a9756dcd68aefedd7dabd013233fbfb5416ef81.
export def test-count [folder: string, name: string, workspace: bool] {
    let expected = (if $workspace { {rows: 31, passed: 388, ignored: 5} } else { {rows: 8, passed: 212, ignored: 1} })
    let actual = (test-summary (open --raw ($folder | path join $'($name).stdout')))
    require ($actual == $expected) $'raw test totals differ: ($name)'
    $actual
}

export def matmul-success [raw: string] {
    let lines = ($raw | lines | each {|line| $line | str trim})
    require (($lines | where {|line| $line | str starts-with 'acpu::matmul_f32'}) == ['acpu::matmul_f32 64x64 x 64x64']
        and ($lines | where {|line| $line | str starts-with 'max error:'}) == ['max error: 0.00e0']
        and ($lines | where {|line| $line | str starts-with 'result:'}) == ['result:    PASS']) 'real matmul correctness output differs'
}

export def normalized-targets [metadata: record, root: string] {
    let packages = ($metadata.packages | where name == acpu)
    require (($packages | length) == 1) 'metadata acpu is not unique'
    let package = ($packages | first)
    require ($package.manifest_path == ($root | path join Cargo.toml)) 'target metadata source differs'
    $package.targets | each {|t| $t | update src_path ($t.src_path | path relative-to $root)} | sort-by name
}

export def target-inventory [folder: string, current: string, owner: string] {
    let argv = [cargo metadata --no-deps --format-version '1' --locked --offline]
    let packaged = (checked $folder metadata $argv true)
    let owning = (checked $folder owner-metadata $argv true)
    require ($packaged.cwd == $current and $owning.cwd == $owner) 'target metadata cwd differs'
    let targets = (normalized-targets (open --raw ($folder | path join metadata.stdout) | from json) $current)
    let original = (normalized-targets (open --raw ($folder | path join owner-metadata.stdout) | from json) ($owner | path join acpu))
    require ($targets == $original and ($targets | length) == 40
        and ($targets | where {|t| 'example' in $t.kind} | length) == 32) 'normalized target inventory differs'
    require (($targets | where {|t| 'test' in $t.kind} | get name) == [chebyshev_coefficients chebyshev_independent gemv_admission gemv_independent input_admission integration]) 'integration target inventory differs'
    $targets
}

export def clean-vcs [vcs: record, revision: string, path: string] {
    require ($vcs.git.sha1 == $revision and $vcs.path_in_vcs == $path
        and not ($vcs.git.dirty? | default false)) 'package VCS is not clean committed source'
}

export def consumer-manifest [manifest: record, metadata: record, scratch: string, acpu: string, nox: string, candidate_version: string, contract: string = acpu-release-contract] {
    let expected = {package: {name: $contract, version: '0.0.0', edition: '2024', publish: false}, workspace: {},
        dependencies: {nox: {package: cyber-nox, path: $nox, features: [honeycrisp]},
            acpu: $'=($candidate_version)', nebu: {package: strata-nebu, version: '=0.1.1'}},
        patch: {'crates-io': {acpu: {path: $acpu}}}, test: [{name: source_contract, path: probe.rs}]}
    require ($manifest == $expected) 'consumer manifest dependency/path scope differs'
    for item in [{name: acpu, root: $acpu, version: $candidate_version}, {name: cyber-nox, root: $nox, version: '0.2.0'},
        {name: $contract, root: $scratch, version: '0.0.0'}] {
        let matches = ($metadata.packages | where name == $item.name)
        require (($matches | length) == 1) 'consumer package identity is not unique'
        let pkg = ($matches | first)
        require ($pkg.version == $item.version and $pkg.source == null and $pkg.manifest_path == ($item.root | path join Cargo.toml)) 'consumer selected a different normalized source'
    }
    require ($metadata.workspace_root == $scratch and $metadata.resolve.root == ($metadata.packages | where name == $contract | first | get id)) 'consumer metadata root differs'
}
