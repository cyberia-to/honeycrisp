use ../common.nu *

def main [repo: string, output: string, bundle_path: string] {
    let root = ($repo | path expand)
    let out = ($output | path expand)
    let bundle = ($bundle_path | path expand)
    let dir = ($root | path join audit/2026-10-17-acpu-release-proposal)
    let revision = (^git -C $root rev-parse HEAD | str trim)
    capture $out export [nu ($dir | path join export.nu) $root $revision $bundle] $root {} | ignore
    let verified = (tool-bundle $bundle)
    for f in $verified.manifest.files {
        require ((^git -C $root show $'($revision):($f.path)' | hash sha256) == $f.sha256) 'export differs from committed tool'
    }
    $verified.manifest | to json | save ($out | path join tooling.json)
    for f in ($verified.manifest.files | where {|f| $f.path != 'audit/2026-10-17-acpu-package-evidence/support.nu'}) {
        let name = ($f.path | path basename | str replace '.nu' '')
        capture $out $'help-($name)' [nu ($bundle | path join $f.path) --help] $root {} | ignore
    }
    let base_tools = ($bundle | path join audit/2026-10-17-acpu-release-proposal)
    let refusal = ($'use ($base_tools)/common.nu *; ' + 'candidate $env.PWD')
    capture $out unbumped-refusal [nu -c $refusal] $root {} 1 | ignore
    require (open --raw ($out | path join unbumped-refusal.stderr) | str contains 'bump must contain exactly three source paths') 'exported candidate guard refusal differs'
    let replay = ($'use ($base_tools)/common.nu *; use ($base_tools)/replay.nu *; ' + '
        let root = $env.PWD
        let epoch = ($root | path join audit/2026-10-17-gemv-tape-admission/tape-1)
        let current = (open ($epoch | path join sources.json)).before.owner
        require ((committed-owner $root) == $current and (git-inputs $root $base) == $current) source-drift
        {workspace: (test-count $epoch tests-workspace true), acpu_release: (test-count $epoch tests-acpu-release false)} | to json')
    capture $out replay [nu -c $replay] $root {} | ignore
    {revision: $revision, manifest_sha256: $verified.sha256, files: ($verified.manifest.files | length), parsed_runners: 9,
        source_bridge_and_raw_replay: passed, candidate_refusal: passed, candidate_gates_executed: false}
    | to json | save ($out | path join result.json)
}
