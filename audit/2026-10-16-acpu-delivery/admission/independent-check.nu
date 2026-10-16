# Independent read-only review identity and raw receipt verification.
def main [repo: string, destination: string] {
    let root = ($repo | path expand)
    let audit = ($root | path join audit/2026-10-16-acpu-delivery/admission)
    let frozen = (open ($audit | path join sources.json))
    if ($frozen.inputs | to json --raw | hash sha256) != $frozen.source_sha256 {
        error make {msg: 'manifest hash mismatch'}
    }
    for input in $frozen.inputs {
        if (^git -C $input.root rev-parse HEAD | str trim) != $input.head {
            error make {msg: $'revision changed: ($input.name)'}
        }
        let paths = (^git -C $input.root ls-files --cached --others --exclude-standard | lines | where {|p|
            not ($p | str starts-with audit/) and not ($p | str starts-with .claude/)
        } | sort)
        if $paths != $input.files.path { error make {msg: $'inventory changed: ($input.name)'} }
        for file in $input.files {
            if (open --raw ($input.root | path join $file.path) | hash sha256) != $file.sha256 {
                error make {msg: $'file changed: ($input.name)/($file.path)'}
            }
        }
    }
    let gates = [debug release clippy] | each {|name|
        let receipt = (open ($audit | path join $'($name).json'))
        if $receipt.source_sha256 != $frozen.source_sha256 or $receipt.source_before_after != equal {
            error make {msg: $'receipt epoch mismatch: ($name)'}
        }
        for stream in [stdout stderr] {
            if (open --raw ($audit | path join $'($name).($stream)') | hash sha256) != ($receipt | get $'($stream)_sha256') {
                error make {msg: $'raw mismatch: ($name).($stream)'}
            }
        }
        {name: $name, command: $receipt.command, exit_code: $receipt.exit_code}
    }
    {source_sha256: $frozen.source_sha256, recorded: (date now | format date '%+'),
        verified: true, inventories: ($frozen.inputs | each {|i| {name: $i.name, head: $i.head, files: ($i.files | length)}}),
        gates: $gates
    } | to json | save $destination
}
