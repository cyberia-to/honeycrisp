use common.nu *
use replay.nu *

def main [package: string, output: string, target: string, tools: string] {
    let pair = ($package | path expand)
    let out = ($output | path expand)
    let target = ($target | path expand)
    let identity = (open ($pair | path join identities.json))
    let tooling = (tool-bundle ($tools | path expand))
    require ($tooling == $identity.tooling) 'API uses different tooling'
    fresh $out
    let envs = (environment $target | upsert RUSTC_BOOTSTRAP '1' | upsert RUSTFLAGS '' | upsert RUSTDOCFLAGS '')
    let semver = (which cargo-semver-checks | first | get path)
    capture $out tool [$semver --version] $out $envs | ignore
    let binaries = ([{name: rustc, path: $envs.RUSTC}, {name: rustdoc, path: $envs.RUSTDOC},
        {name: cargo, path: ($envs.RUSTC | path dirname | path join cargo)}] | each {|tool|
        capture $out $'tool-($tool.name)' [$tool.path --version] $out $envs | ignore
        $tool | insert sha256 (digest $tool.path)
    })
    let tool_identity = {root: $out, binaries: $binaries, semver: {path: $semver, sha256: (digest $semver)}}
    let tools_before = (api-binaries $out $tool_identity)
    let cargo = ($binaries | where name == cargo | first | get path)
    let artifacts = ([baseline current] | each {|side|
        let input = ($identity | get $side)
        let package_version = (if $side == baseline { '0.2.0' } else { $version })
        let source = ($pair | path join $side $'acpu-($package_version)')
        require ((files $source) == $input.files) 'API package changed'
        let manifest = (open ($source | path join Cargo.toml))
        require ($manifest.features? == null and $manifest.package.version == $package_version) 'unexpected API feature/version matrix'
        let envs = ($envs | upsert CARGO_TARGET_DIR ($target | path join $side))
        capture $out $side [$cargo rustdoc --lib --locked --offline --target aarch64-apple-darwin -- -Z unstable-options --output-format json] $source $envs | ignore
        let path = ($envs.CARGO_TARGET_DIR | path join aarch64-apple-darwin/doc/acpu.json)
        let retained = ($out | path join $'($side).rustdoc.json')
        cp $path $retained
        require ((files $source) == $input.files) 'API package changed during extraction'
        {side: $side, path: $path, retained: ($retained | path basename), sha256: (digest $path)}
    })
    let argv = [$semver check-release --baseline-rustdoc $artifacts.0.path --current-rustdoc $artifacts.1.path --release-type patch --color never]
    capture $out comparison $argv $out $envs | ignore
    for tool in $tools_before { require ((digest $tool.path) == $tool.sha256) 'API executable changed during gate' }
    require ((tool-bundle ($tools | path expand)) == $tooling) 'tooling changed'
    $tool_identity | merge {package_identity_sha256: (digest ($pair | path join identities.json)), artifacts: $artifacts, tooling: $tooling,
        scope: 'complete library; aarch64-apple-darwin; no declared crate features'}
    | to json | save ($out | path join result.json)
}
