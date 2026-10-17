use common.nu *
use replay.nu *

def main [nox_repo: string, package: string, output: string, target: string, tools: string] {
    let nox = ($nox_repo | path expand)
    let pair = ($package | path expand)
    let out = ($output | path expand)
    let envs = (environment $target)
    let input = (open ($pair | path join identities.json))
    let acpu = ($pair | path join current $'acpu-($version)')
    require ((files $acpu) == $input.current.files) 'candidate package changed'
    let nox_inputs = (committed-owner $nox)
    require ((^git -C $nox rev-parse HEAD | str trim) == $nox_revision) 'Nox source drift needs review'
    let tooling = (tool-bundle ($tools | path expand))
    require ($tooling == $input.tooling) 'consumer tooling differs'
    fresh $out
    capture $out package [cargo package -p cyber-nox --locked --offline] $nox $envs | ignore
    let archive = ($out | path join nox.crate)
    cp ($envs.CARGO_TARGET_DIR | path join package/cyber-nox-0.2.0.crate) $archive
    require ((digest $archive) == $nox_archive_sha256) 'Nox archive differs from independent PR21 anchor'
    let normalized = (extract $archive ($out | path join nox) cyber-nox-0.2.0)
    let vcs = (open ($normalized | path join .cargo_vcs_info.json))
    require ($vcs.git.sha1 == $nox_revision and $vcs.path_in_vcs == rs and not ($vcs.git.dirty? | default false)) 'Nox package VCS differs'
    let package_files = (files $normalized)
    let source = ($package_files | where {|f| $f.path not-in ['.cargo_vcs_info.json' Cargo.lock Cargo.toml]}
        | each {|f| $f | update path (if $f.path == Cargo.toml.orig { 'rs/Cargo.toml' } else { $'rs/($f.path)' })} | sort-by path)
    require ($source == ($nox_inputs | where {|f| ($f.path | str starts-with rs/) and $f.path != 'rs/Cargo.lock'})) 'normalized Nox source differs'
    let probe_path = 'audit/2026-10-16-honeycrisp-backend-design/probe.rs'
    let probe = (^git -C $nox show $'($nox_revision):($probe_path)')
    require (($probe | hash sha256) == $probe_sha256 and (digest ($nox | path join $probe_path)) == $probe_sha256) 'original probe changed'
    let scratch = ($out | path join scratch)
    mkdir $scratch
    $probe | save ($scratch | path join probe.rs)
    {package: {name: acpu-release-contract, version: '0.0.0', edition: '2024', publish: false}, workspace: {},
        dependencies: {nox: {package: cyber-nox, path: $normalized, features: [honeycrisp]},
            acpu: $'=($version)', nebu: {package: strata-nebu, version: '=0.1.1'}},
        patch: {'crates-io': {acpu: {path: $acpu}}}, test: [{name: source_contract, path: probe.rs}]}
    | to toml | save ($scratch | path join Cargo.toml)
    capture $out lock [cargo generate-lockfile --offline] $scratch $envs | ignore
    let before = (files $scratch)
    capture $out metadata [cargo metadata --format-version 1 --locked --offline] $scratch $envs | ignore
    let metadata = (open --raw ($out | path join metadata.stdout) | from json)
    let lock = (open --raw ($scratch | path join Cargo.lock) | from toml)
    let dependencies = (registry-identities $metadata $lock)
    consumer-manifest (open ($scratch | path join Cargo.toml)) $metadata $scratch $acpu $normalized $version
    let selected = ($metadata.packages | where name == acpu)
    require (($selected | length) == 1 and ($selected | first | get version) == $version
        and ($selected | first | get manifest_path | path dirname) == $acpu) 'consumer selected another acpu'
    capture $out tree [cargo tree -e features --locked --offline] $scratch $envs | ignore
    capture $out debug [cargo test --test source_contract --locked --offline] $scratch $envs | ignore
    capture $out release [cargo test --test source_contract --release --locked --offline] $scratch $envs | ignore
    require ((files $scratch) == $before and (files $acpu) == $input.current.files and (files $normalized) == $package_files) 'consumer/package inputs changed'
    require ((committed-owner $nox) == $nox_inputs and (tool-bundle ($tools | path expand)) == $tooling) 'Nox/tooling inputs changed'
    {package_identity_sha256: (digest ($pair | path join identities.json)), nox_head: $nox_revision, nox_inputs: $nox_inputs,
        nox: {root: $normalized, sha256: (digest $archive), vcs: $vcs, files: $package_files},
        consumer: {root: $scratch, files: $before, probe_sha256: ($probe | hash sha256)},
        dependencies: $dependencies, tooling: $tooling, diagnostic_override: true, unpatched_registry_acceptance: false}
    | to json | save ($out | path join result.json)
}
