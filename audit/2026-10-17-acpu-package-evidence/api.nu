use support.nu *

def main [repo: string, pair: string, pair_cache: string, cache: string, epoch: string] {
    let root = ($repo | path expand)
    let audit = ($root | path join audit/2026-10-17-acpu-package-evidence)
    let identity_path = ($audit | path join $pair identities.json)
    let identity = (open $identity_path)
    let cache = ($cache | path expand)
    let out = ($audit | path join $epoch)
    require (not ($out | path exists) and not ($cache | path exists)) 'API output/cache exists'
    require (($epoch | str replace -r '^[a-z0-9-]+$' '') == '') 'invalid epoch'
    require ((committed-owner $root) == $identity.owner_files) 'owner inputs changed'
    mkdir $out $cache
    let tooling = ([support.nu api.nu] | each {|p| {path: $p, sha256: (digest ($audit | path join $p))}})
    $tooling | to json | save ($out | path join tooling.json)
    let environment = ($identity.environment | merge {RUSTC_BOOTSTRAP: '1', RUSTFLAGS: '', RUSTDOCFLAGS: ''})
    let tools = ([cargo rustc rustdoc cargo-semver-checks] | each {|tool|
        let path = (with-env $environment { which $tool | first | get path })
        capture $out $'tool-($tool)' [$path --version] $root $environment | ignore
        {name: $tool, path: $path, sha256: (digest $path)}
    })
    $tools | to json | save ($out | path join tools.json)
    mut artifacts = []
    for side in [baseline current] {
        let package = ($pair_cache | path expand | path join $side acpu-0.2.0)
        let input = ($identity | get $side)
        require ((files $package) == $input.files) 'package inputs changed'
        let manifest = (open ($package | path join Cargo.toml))
        require (not ('features' in ($manifest | columns))) 'unexpected feature surface requires a new plan'
        let command_env = ($environment | upsert CARGO_TARGET_DIR ($cache | path join $side))
        let argv = [cargo rustdoc --lib --locked --offline --target aarch64-apple-darwin -- -Z unstable-options --output-format json]
        capture $out $side $argv $package $command_env | ignore
        let path = ($command_env.CARGO_TARGET_DIR | path join aarch64-apple-darwin doc acpu.json)
        $artifacts = ($artifacts | append {side: $side, path: $path, sha256: (digest $path)})
        capture $out $'($side)-tree' [cargo tree --locked --offline -e features] $package $command_env | ignore
        require ((files $package) == $input.files) 'package changed during API extraction'
    }
    let old = ($artifacts | where side == baseline | first)
    let new = ($artifacts | where side == current | first)
    capture $out comparison [cargo-semver-checks check-release --baseline-rustdoc $old.path --current-rustdoc $new.path --release-type patch --color never] $root $environment | ignore
    require ((committed-owner $root) == $identity.owner_files) 'owner inputs changed during API comparison'
    for file in $tooling { require ((digest ($audit | path join $file.path)) == $file.sha256) 'API tooling changed' }
    {pair: $pair, pair_sha256: (digest $identity_path), artifacts: $artifacts,
        tooling_sha256: ($tooling | to json --raw | hash sha256), scope: 'complete library API; aarch64-apple-darwin; no crate features', release_ready: false}
    | to json | save ($out | path join result.json)
    {out: $out} | to json
}
