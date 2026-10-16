def check [ok: bool, message: string] {
    if not $ok { error make {msg: $message} }
}
def main [repo: path, destination: path] {
    let out = ($repo | path expand | path join audit/2026-10-16-acpu-delivery/numeric-reference)
    for epoch in [{name: first, source: oracle-first.rs.txt, exit: 101}, {name: resolved, source: oracle.rs, exit: 0}] {
        let r = (open ($out | path join $'($epoch.name).json'))
        check ($r.compile_exit == 0 and $r.source_unchanged and $r.run_exit == $epoch.exit) $'result: ($epoch.name)'
        check ((open --raw ($out | path join $epoch.source) | hash sha256) == $r.source_sha256) $'source: ($epoch.name)'
        for stream in [stdout stderr] {
            check ((open --raw ($out | path join $'($epoch.name).($stream)') | hash sha256) == ($r | get $'($stream)_sha256')) $'run stream: ($epoch.name): ($stream)'
            check ((open --raw ($out | path join $'($epoch.name)-build.($stream)') | hash sha256) == ($r | get $'build_($stream)_sha256')) $'build stream: ($epoch.name): ($stream)'
        }
    }
    let data = (open --raw ($out | path join resolved.stdout) | from json)
    check ($data.count == 160 and ($data.rows | length) == 160) 'row count'
    check (($data.rows | where reference_method == positive_series | length) == 120) 'series count'
    let missing = ($data.rows | where coefficient == null)
    check (($missing | length) == 1 and $missing.0.tau == 1000 and $missing.0.n == 320) 'unresolved control'
    {verified: true, checked_at: (date now | format date '%+'), results: ($data | reject rows)}
        | to json | save $destination
}
