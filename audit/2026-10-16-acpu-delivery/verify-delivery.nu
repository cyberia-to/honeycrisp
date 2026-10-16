# Verify preserved historical evidence and the final source epoch; no gate reruns.
def main [repo: string] {
    let root = ($repo | path expand)
    let audit = ($root | path join audit/2026-10-16-acpu-delivery)
    let final = ($audit | path join delivery-v4)
    let frozen = (open ($final | path join sources.json))
    for input in $frozen.inputs {
        let ancestry = (do { ^git -C $input.root merge-base --is-ancestor $input.head HEAD } | complete)
        if $ancestry.exit_code != 0 { error make {msg: $'source base is not an ancestor: ($input.name)'} }
        let paths = (^git -C $input.root ls-files --cached --others --exclude-standard | lines
            | where {|p| not ($p | str starts-with audit/) and not ($p | str starts-with .claude/)}
            | sort | uniq | where {|p| ($input.root | path join $p | path exists)})
        if $paths != $input.files.path { error make {msg: $'inventory changed: ($input.name)'} }
        for f in $input.files {
            if (open --raw ($input.root | path join $f.path) | hash sha256) != $f.sha256 {
                error make {msg: $'source changed: ($input.name)/($f.path)'}
            }
        }
    }
    let manifests = (glob ($audit | path join '**/sources.json') | each {|path|
        let m = (open $path)
        if ('inputs' in ($m | columns)) {
            if ($m.inputs | to json --raw | hash sha256) != $m.source_sha256 {
                error make {msg: $'manifest changed: ($path)'}
            }
            {path: ($path | path relative-to $audit), source_sha256: $m.source_sha256}
        }
    } | compact)
    let receipts = (glob ($audit | path join '**/*.json') | each {|path|
        let r = (open $path)
        if ($r | describe | str starts-with record) and ('stdout_sha256' in ($r | columns)) and ('exit_code' in ($r | columns)) {
            for stream in [stdout stderr] {
                let raw = ($path | str replace -r '\.json$' $'.($stream)')
                if (open --raw $raw | hash sha256) != ($r | get $'($stream)_sha256') {
                    error make {msg: $'stream changed: ($raw)'}
                }
            }
            if ('source_sha256' in ($r | columns)) {
                let manifest = (open ($path | path dirname | path join sources.json))
                if $r.source_sha256 != $manifest.source_sha256 or $r.source_before_after != equal {
                    error make {msg: $'receipt source mismatch: ($path)'}
                }
            }
            {path: ($path | path relative-to $audit), exit_code: $r.exit_code}
        }
    } | compact)
    let gates = [fmt clippy-workspace clippy-acpu tests-workspace build-release tests-acpu-release matmul docs-acpu aruminium-matmul rane-matmul check-all-targets]
    let final_results = ($gates | each {|name|
        let r = (open ($final | path join $'($name).json'))
        if $r.exit_code != 0 or $r.source_sha256 != $frozen.source_sha256 {
            error make {msg: $'final gate failed or moved: ($name)'}
        }
        let tests = (open --raw ($final | path join $'($name).stdout') | lines
            | parse -r '^test result: ok\. (?<passed>\d+) passed; (?<failed>\d+) failed; (?<ignored>\d+) ignored;')
        {name: $name, command: $r.command, exit_code: $r.exit_code,
            passed: (if ($tests | is-empty) { null } else { $tests.passed | into int | math sum }),
            ignored: (if ($tests | is-empty) { null } else { $tests.ignored | into int | math sum })}
    })
    {verified: true, source_sha256: $frozen.source_sha256, manifests: $manifests,
        historical_receipts: $receipts, final_gates: $final_results} | to json
}
