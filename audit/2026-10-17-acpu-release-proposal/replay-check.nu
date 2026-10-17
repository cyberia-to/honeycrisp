use common.nu *
use replay.nu *

def rejects [operation: closure, message: string] {
    let outcome = (try { do $operation | ignore; {rejected: false, message: ''} } catch {|e| {rejected: true, message: $e.msg}})
    require ($outcome.rejected and ($outcome.message | str contains $message)) 'guard accepted corrupt input or refused for another reason'
}

# Exercises replay guards with real PR21/PR25 receipts and synthetic damage.
# Fresh work is limited to --version and metadata; no 0.2.1 source or gate exists.
def main [repo: string, output: string, scratch: string] {
    let root = ($repo | path expand)
    let out = ($output | path expand)
    let tmp = ($scratch | path expand)
    fresh $out
    fresh $tmp
    let old = ($root | path join audit/2026-10-17-acpu-package-evidence)
    let tools = ($out | path join tools)
    fresh $tools
    let envs = (environment ($tmp | path join target) | upsert RUSTC_BOOTSTRAP '1' | upsert RUSTFLAGS '' | upsert RUSTDOCFLAGS '')
    mut binaries = []
    for expected in $api_tools {
        let path = (if $expected.name == cargo-semver-checks { which cargo-semver-checks | first | get path } else { $envs.RUSTC | path dirname | path join $expected.name })
        let name = (if $expected.name == cargo-semver-checks { 'tool' } else { $'tool-($expected.name)' })
        capture $tools $name [$path --version] $tools $envs | ignore
        $binaries = ($binaries | append {name: $expected.name, path: $path, sha256: (digest $path)})
    }
    require (($binaries | select name sha256 | sort-by name) == (open ($old | path join api-2/tools.json) | select name sha256 | sort-by name)) 'installed API tools differ from PR21'
    let api = {root: $tools, binaries: ($binaries | where name != cargo-semver-checks), semver: ($binaries | where name == cargo-semver-checks | first | reject name)}
    api-binaries $tools $api | ignore
    rejects { api-binaries $tools ($api | update semver.sha256 changed) } 'API binary hash/name differs'
    rejects { api-environment ($envs | update RUSTC '/synthetic/wrong/rustc') $api.binaries } 'API executable/environment binding differs'
    let damaged = ($out | path join synthetic-api-version)
    cp -r $tools $damaged
    'rustdoc 1.94.0 (synthetic damage)' | save -f ($damaged | path join tool-rustdoc.stdout)
    open ($damaged | path join tool-rustdoc.json) | update stdout_sha256 (digest ($damaged | path join tool-rustdoc.stdout)) | to json | save -f ($damaged | path join tool-rustdoc.json)
    rejects { api-binaries $damaged $api } 'API tool version/cwd differs'
    let historical_workspace = (test-summary (open --raw ($old | path join owner-2/tests-workspace.stdout)))
    let historical_acpu = {rows: 6, passed: 196, ignored: 1}
    require ($historical_workspace == {rows: 28, passed: 365, ignored: 5}) 'historical workspace totals differ'
    for item in [owner-2/tests-acpu-release checks-2/tests-debug checks-2/tests-release] {
        require ((test-summary (open --raw ($old | path join $'($item).stdout'))) == $historical_acpu) 'historical acpu totals differ'
    }
    let current = ($root | path join audit/2026-10-17-gemv-tape-admission/tape-1)
    let workspace = (test-count $current tests-workspace true)
    let release = (test-count $current tests-acpu-release false)
    rejects { test-count ($old | path join owner-2) tests-workspace true } 'raw test totals differ'
    let raw = (open --raw ($old | path join owner-2/tests-acpu-release.stdout))
    rejects { test-summary ($raw | str replace '126 passed; 0 failed' '125 passed; 1 failed') } 'tests failed, measured or filtered'
    let matmul = (open --raw ($old | path join owner-2/matmul.stdout))
    matmul-success $matmul
    matmul-success (open --raw ($old | path join checks-2/matmul.stdout))
    rejects { matmul-success ($matmul | str replace PASS FAIL) } 'real matmul correctness output differs'
    let previous = (open ($old | path join consumer-2/result.json))
    let manifest = (open --raw ($old | path join consumer-2/consumer-Cargo.toml.txt) | from toml)
    let metadata = (open --raw ($old | path join consumer-2/metadata.stdout) | from json)
    let scratch_root = $previous.consumer.root
    let acpu_root = $manifest.patch.crates-io.acpu.path
    let nox_root = $manifest.dependencies.nox.path
    consumer-manifest $manifest $metadata $scratch_root $acpu_root $nox_root '0.2.0' acpu-delivery-contract
    rejects { consumer-manifest ($manifest | update dependencies.nox.path '/synthetic/other/nox') $metadata $scratch_root $acpu_root $nox_root '0.2.0' acpu-delivery-contract } 'consumer manifest dependency/path scope differs'
    rejects { consumer-manifest ($manifest | update dependencies.nebu.version '0.1.1') $metadata $scratch_root $acpu_root $nox_root '0.2.0' acpu-delivery-contract } 'consumer manifest dependency/path scope differs'
    let drift = ($metadata | update packages ($metadata.packages | each {|p| if $p.name == cyber-nox { $p | update manifest_path '/synthetic/other/nox/Cargo.toml' } else { $p }}))
    rejects { consumer-manifest $manifest $drift $scratch_root $acpu_root $nox_root '0.2.0' acpu-delivery-contract } 'consumer selected a different normalized source'
    let vcs = {git: {sha1: $nox_revision}, path_in_vcs: rs}
    clean-vcs $vcs $nox_revision rs
    rejects { clean-vcs ($vcs | upsert git.dirty true) $nox_revision rs } 'package VCS is not clean committed source'
    let capsule = ($old | path join evidence.tar.gz)
    require ((digest $capsule) == '0058a7f32b9dfb691f4592539bc2e33ac59ffc0a2f666ee056b0eb736ff33f7c') 'prior capsule differs'
    capture $out extract-prior [tar -xzf $capsule -C $tmp current-final.crate normalized-nox.crate] $root {} | ignore
    require ((digest ($tmp | path join normalized-nox.crate)) == $nox_archive_sha256) 'prior Nox archive anchor differs'
    require ((digest ($tmp | path join current-final.crate)) == 'acd3929c3472443c92b977ed789e9494715bfbd9278a8d1e52fe6d1fdb9938a3') 'prior package differs'
    let package = (extract ($tmp | path join current-final.crate) ($tmp | path join extracted) acpu-0.2.0)
    capture $out metadata [cargo metadata --no-deps --format-version 1 --locked --offline] $package (environment ($tmp | path join target)) | ignore
    let targets = (normalized-targets (open --raw ($out | path join metadata.stdout) | from json) $package)
    require ($targets == (open ($old | path join target-inventory/result.json)).targets and ($targets | length) == 38) 'real target replay differs'
    let meta = (open --raw ($out | path join metadata.stdout) | from json)
    rejects { normalized-targets $meta '/synthetic/other/package' } 'target metadata source differs'
    capture $out current-metadata [cargo metadata --no-deps --format-version 1 --locked --offline] $root (environment ($tmp | path join target)) | ignore
    let current_targets = (normalized-targets (open --raw ($out | path join current-metadata.stdout) | from json) ($root | path join acpu))
    require (($current_targets | length) == 40 and ($current_targets | where {|t| 'example' in $t.kind} | length) == 32
        and ($current_targets | where {|t| 'test' in $t.kind} | get name) == [chebyshev_coefficients chebyshev_independent gemv_admission gemv_independent input_admission integration]) 'current target replay differs'
    $current_targets | to json | save ($out | path join current-targets.json)
    {real_api_tools: 4, current_raw_owner_workspace: $workspace, current_raw_acpu_release: $release,
        historical_workspace: $historical_workspace, historical_acpu_and_package: $historical_acpu,
        real_matmul_controls: 2, historical_normalized_targets: ($targets | length), current_targets: ($current_targets | length), synthetic_corruption_refusals: 11,
        candidate_gates_executed: false, prior_capsule_sha256: (digest $capsule)} | to json
}
