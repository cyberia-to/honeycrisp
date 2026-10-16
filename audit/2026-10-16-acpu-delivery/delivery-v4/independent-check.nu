# Independent final inventory/receipt verification, without running owner gates.
def main [repo: string, receipts_json: string, destination: string] {
    let root = ($repo | path expand)
    let audit = ($root | path join audit/2026-10-16-acpu-delivery/delivery-v4)
    let frozen = (open ($audit | path join sources.json))
    if ($frozen.inputs | to json --raw | hash sha256) != $frozen.source_sha256 {
        error make {msg: 'manifest identity differs'}
    }
    for input in $frozen.inputs {
        if (^git -C $input.root rev-parse HEAD | str trim) != $input.head {
            # Integration may append commits; exact file/inventory checks below
            # remain mandatory and the recorded revision must be an ancestor.
            let ancestor = (do { ^git -C $input.root merge-base --is-ancestor $input.head HEAD } | complete)
            if $ancestor.exit_code != 0 {
                error make {msg: $'recorded revision is not an ancestor: ($input.name)'}
            }
        }
        let paths = (^git -C $input.root ls-files --cached --others --exclude-standard | lines | where {|p|
            not ($p | str starts-with audit/) and not ($p | str starts-with .claude/)
        } | sort | uniq | where {|p| ($input.root | path join $p | path exists)})
        if $paths != $input.files.path { error make {msg: $'inventory changed: ($input.name)'} }
        for file in $input.files {
            if (open --raw ($input.root | path join $file.path) | hash sha256) != $file.sha256 {
                error make {msg: $'file changed: ($input.name)/($file.path)'}
            }
        }
    }
    let receipts = ($receipts_json | from json | each {|name|
        let receipt = (open ($audit | path join $'($name).json'))
        if $receipt.source_sha256 != $frozen.source_sha256 or $receipt.source_before_after != equal {
            error make {msg: $'receipt epoch mismatch: ($name)'}
        }
        if $receipt.exit_code != 0 { error make {msg: $'required gate failed: ($name)'} }
        for stream in [stdout stderr] {
            if (open --raw ($audit | path join $'($name).($stream)') | hash sha256) != ($receipt | get $'($stream)_sha256') {
                error make {msg: $'raw mismatch: ($name).($stream)'}
            }
        }
        for extra in $receipt.extra_inputs {
            let path = if ($extra.path | str starts-with '/') { $extra.path } else { $root | path join $extra.path }
            if (open --raw $path | hash sha256) != $extra.sha256 { error make {msg: $'extra changed: ($extra.path)'} }
        }
        let results = (open --raw ($audit | path join $'($name).stdout') | lines
            | parse -r '^test result: ok\. (?<passed>\d+) passed; (?<failed>\d+) failed; (?<ignored>\d+) ignored;')
        {name: $name, command: $receipt.command, exit_code: $receipt.exit_code,
            tests: (if ($results | is-empty) { null } else {
                {passed: ($results.passed | into int | math sum),
                 failed: ($results.failed | into int | math sum),
                 ignored: ($results.ignored | into int | math sum)}
            })}
    })
    let sdk = (open ($root | path join audit/2026-10-16-acpu-delivery/vdsp-abi/sdk.json))
    if (open --raw $sdk.header | hash sha256) != $sdk.header_sha256 { error make {msg: 'SDK header changed'} }
    {verified: true, source_sha256: $frozen.source_sha256, recorded: (date now | format date '%+'),
        inventories: ($frozen.inputs | each {|i| {name: $i.name, head: $i.head, files: ($i.files | length)}}),
        sdk_header_sha256: $sdk.header_sha256, receipts: $receipts
    } | to json | save $destination
}
