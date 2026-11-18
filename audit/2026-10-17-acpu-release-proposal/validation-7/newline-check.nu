use ../common.nu *

def main [repo: string, nox_repo: string, production_tools: string] {
    let root = ($repo | path expand)
    let nox = ($nox_repo | path expand)
    let bundle = (tool-bundle ($production_tools | path expand))
    let path = 'audit/2026-10-16-honeycrisp-backend-design/probe.rs'
    let naive = (^git -C $nox show $'($nox_revision):($path)')
    let exact = (do { ^git -C $nox show $'($nox_revision):($path)' } | complete)
    require ($exact.exit_code == 0 and ($exact.stdout | hash sha256) == $probe_sha256
        and (digest ($nox | path join $path)) == $probe_sha256) 'exact committed/working probe differs'
    require (($naive | hash sha256) != $probe_sha256 and ($naive + (char nl)) == $exact.stdout) 'newline-loss witness differs'
    for tool in $bundle.manifest.files {
        let source = (do { ^git -C $root show $'($bundle.manifest.revision):($tool.path)' } | complete)
        require ($source.exit_code == 0 and ($source.stdout | hash sha256) == $tool.sha256) 'exact committed tool differs'
    }
    let candidate = (candidate-revision $root '2c70b3b929c791ca489bba12d2f5a6daef7d43e7')
    {probe_sha256: ($exact.stdout | hash sha256), stripped_sha256: ($naive | hash sha256),
        omitted: 'one final newline', committed_tool_files: ($bundle.manifest.files | length),
        candidate: $candidate.head, candidate_manifest_exact_bytes: passed} | to json
}
