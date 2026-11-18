use common.nu *

# Export only committed runner bytes, preserving their relative helper import.
def main [repo: string, revision: string, destination: string] {
    let root = ($repo | path expand)
    let out = ($destination | path expand)
    fresh $out
    let paths = ([common.nu export.nu owning.nu package.nu api.nu consumer.nu verify.nu retain.nu]
        | each {|p| $'audit/2026-10-17-acpu-release-proposal/($p)'}
        | append 'audit/2026-10-17-acpu-package-evidence/support.nu' | sort)
    let files = ($paths | each {|p|
        let body = (do { ^git -C $root show $'($revision):($p)' } | complete)
        require ($body.exit_code == 0) $'missing committed tool: ($p)'
        mkdir ($out | path join $p | path dirname)
        $body.stdout | save ($out | path join $p)
        {path: $p, sha256: ($body.stdout | hash sha256)}
    })
    {revision: (^git -C $root rev-parse $revision | str trim), files: $files}
    | to json | save ($out | path join tooling.json)
    tool-bundle $out | to json
}
