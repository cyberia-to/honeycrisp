def main [repo: path] {
    let root = ($repo | path expand)
    let out = ($root | path join audit/2026-10-16-acpu-delivery/numeric-prototype)
    for name in [first stress random limits] {
        let r = (open ($out | path join $'($name).json'))
        if $r.compile_exit != 0 or $r.run_exit != 0 or $r.source_before_after != equal {
            error make {msg: $'failed prototype: ($name)'}
        }
        for source in $r.sources {
            if (open --raw ($out | path join $source.path) | hash sha256) != $source.sha256 {
                error make {msg: $'source changed: ($source.path)'}
            }
        }
        for stream in [stdout stderr] {
            if (open --raw ($out | path join $'($name).($stream)') | hash sha256) != ($r | get $'($stream)_sha256') {
                error make {msg: $'run stream changed: ($name)'}
            }
            if (open --raw ($out | path join $'($name)-build.($stream)') | hash sha256) != ($r | get $'build_($stream)_sha256') {
                error make {msg: $'build stream changed: ($name)'}
            }
        }
    }
    let first = (open ($out | path join first.json))
    if (open --raw ($out | path join first-reference.tsv) | hash sha256) != $first.input_sha256 {
        error make {msg: 'reference TSV changed'}
    }
    if (open --raw ($root | path join audit/2026-10-16-acpu-delivery/numeric-reference/resolved.stdout) | hash sha256) != $first.reference_sha256 {
        error make {msg: 'independent reference changed'}
    }
    {verified: true, recurrence_sha256: (open --raw ($out | path join recurrence.rs) | hash sha256),
        receipts: [first stress random limits]} | to json
}
