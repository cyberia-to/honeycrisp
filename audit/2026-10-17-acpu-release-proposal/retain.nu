use common.nu *

# Retain actual records/archives once, omitting reproducible extracted source copies.
def 'main pack' [evidence: string, tools: string, output: string, staging: string] {
    let records = ($evidence | path expand)
    let tools = ($tools | path expand)
    let out = ($output | path expand)
    let stage = ($staging | path expand)
    fresh $out
    fresh $stage
    let tooling = (tool-bundle $tools)
    let selected = (files $records | where {|f|
        (not ($f.path | str starts-with package/baseline/) and not ($f.path | str starts-with package/current/)
        and not ($f.path | str starts-with consumer/nox/))
    })
    for file in $selected {
        let dest = ($stage | path join records $file.path)
        mkdir ($dest | path dirname)
        cp ($records | path join $file.path) $dest
        require ((digest $dest) == $file.sha256) 'record changed while retaining'
    }
    for file in (files $tools) {
        let dest = ($stage | path join tools $file.path)
        mkdir ($dest | path dirname)
        cp ($tools | path join $file.path) $dest
        require ((digest $dest) == $file.sha256) 'tool changed while retaining'
    }
    let inventory = (files $stage)
    capture $out archive [tar -czf ($out | path join evidence.tar.gz) -C $stage ...$inventory.path] $out {} | ignore
    {files: $inventory, tooling: $tooling, capsule_sha256: (digest ($out | path join evidence.tar.gz)),
        candidate: (open ($records | path join package/identities.json)).candidate.head,
        publication_authorized: false} | to json | save ($out | path join bundle.json)
}

def 'main unpack' [capsule: string, destination: string] {
    let source = ($capsule | path expand)
    let out = ($destination | path expand)
    let manifest = (open ($source | path join bundle.json))
    let archive = ($source | path join evidence.tar.gz)
    require ((digest $archive) == $manifest.capsule_sha256) 'capsule checksum differs'
    let listing = (do { ^tar -tzf $archive } | complete)
    require ($listing.exit_code == 0 and ($listing.stdout | lines | sort) == ($manifest.files.path | sort)) 'capsule inventory differs'
    require ($manifest.files.path | all {|p| not ($p | str starts-with '/') and '..' not-in ($p | split row '/')}) 'unsafe capsule member'
    fresh $out
    let result = (do { ^tar -xzf $archive -C $out } | complete)
    require ($result.exit_code == 0 and (files $out) == $manifest.files) 'capsule contents differ'
    {records: ($out | path join records), tools: ($out | path join tools)} | to json
}

def main [] { help main }
