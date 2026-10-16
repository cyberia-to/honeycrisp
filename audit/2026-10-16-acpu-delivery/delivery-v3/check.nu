# Conservatively fingerprint every tracked owner/sibling file except audit/task records.
def inputs [repo: string] {
    [
        {name: honeycrisp, root: $repo}
        {name: strata, root: ($repo | path dirname | path join strata | path expand)}
    ] | each {|r|
        let files = (^git -C $r.root ls-files --cached --others --exclude-standard | lines | where {|p|
            not ($p | str starts-with audit/) and not ($p | str starts-with .claude/)
        } | sort | uniq | where {|p| ($r.root | path join $p | path exists)} | each {|p| {path: $p, sha256: (open --raw ($r.root | path join $p) | hash sha256)}})
        {name: $r.name, root: $r.root, head: (^git -C $r.root rev-parse HEAD | str trim), files: $files}
    }
}
def main [repo: string, name: string, argv_json: string, --extra-files: string = '[]'] {
    let root = ($repo | path expand)
    cd $root
    let out = 'audit/2026-10-16-acpu-delivery/delivery-v3'
    let before = (inputs $root)
    let hash = ($before | to json --raw | hash sha256)
    let manifest = ($out | path join sources.json)
    if ($manifest | path exists) {
        if (open $manifest).source_sha256 != $hash { error make {msg: 'source epoch changed'} }
    } else { {source_sha256: $hash, inputs: $before} | to json | save $manifest }
    let argv = ($argv_json | from json)
    let extras = ($extra_files | from json | each {|p| {path: $p, sha256: (open --raw $p | hash sha256)}})
    let started = (date now | format date '%+')
    let r = (do { ^($argv | first) ...($argv | skip 1) } | complete)
    if $before != (inputs $root) { error make {msg: 'source changed during gate'} }
    for extra in $extras {
        if (open --raw $extra.path | hash sha256) != $extra.sha256 { error make {msg: 'diagnostic changed during gate'} }
    }
    $r.stdout | save ($out | path join $'($name).stdout')
    $r.stderr | save ($out | path join $'($name).stderr')
    {command: $argv, cwd: $root, source_sha256: $hash, source_before_after: equal,
        started: $started, finished: (date now | format date '%+'), exit_code: $r.exit_code,
        rustc: (^rustc --version | str trim), cargo: (^cargo --version | str trim),
        rustup_toolchain: $env.RUSTUP_TOOLCHAIN, clippy: (^cargo clippy --version | str trim), rustc_path: $env.RUSTC, rustdoc_path: $env.RUSTDOC, target: $env.CARGO_TARGET_DIR,
        rustflags: $env.RUSTFLAGS, rustdocflags: $env.RUSTDOCFLAGS, jobs: $env.CARGO_BUILD_JOBS, extra_inputs: $extras,
        stdout_sha256: ($r.stdout | hash sha256), stderr_sha256: ($r.stderr | hash sha256)
    } | to json | save ($out | path join $'($name).json')
    print {name: $name, exit_code: $r.exit_code}
    if $r.exit_code != 0 { exit $r.exit_code }
}
