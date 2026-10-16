use std/assert

def identity [path: string] {
    let body = (open --raw $path | into binary)
    {bytes: ($body | bytes length), sha256: ($body | hash sha256)}
}

def verify [freeze: record] {
    for source in $freeze.repositories {
        assert equal (^git -C $source.root rev-parse HEAD | str trim) $source.head
        for file in $source.files {
            assert equal (identity ($source.root | path join $file.path)) {bytes: $file.bytes, sha256: $file.sha256}
        }
    }
}

def main [repo: string, epoch: string, name: string, command_json: string] {
    let repo = ($repo | path expand)
    let command = ($command_json | from json)
    let output = $env.FILE_PWD
    let freeze_path = ($output | path join ($epoch + '.json'))
    if not ($freeze_path | path exists) {
        let roots = [$repo ($repo | path dirname | path join strata | path expand)]
        let sources = ($roots | each {|root|
            let tracked = (^git -C $root ls-files -z | split row (char nul)
                | where {|p| $p != '' and not ($p | str starts-with 'audit/') and not ($p | str starts-with '.claude/') })
            let new = if $root == $repo {
                ^git -C $root ls-files --others --exclude-standard -z | split row (char nul)
                    | where {|p| $p | str starts-with 'unimem/' }
            } else { [] }
            let files = ($tracked | append $new | uniq | sort | each {|path|
                {path: $path} | merge (identity ($root | path join $path))
            })
            {root: $root, head: (^git -C $root rev-parse HEAD | str trim), files: $files}
        })
        {epoch: $epoch, repositories: $sources} | to json | save $freeze_path
    }
    let freeze = (open $freeze_path)
    verify $freeze
    let records = ($output | path join records)
    mkdir $records
    let stem = ($records | path join $name)
    assert (not (($stem + '.json') | path exists)) 'Never overwrite a receipt'
    cd $repo
    let started = (date now | format date '%+')
    let result = (do { ^($command | first) ...($command | skip 1) } | complete)
    let finished = (date now | format date '%+')
    $result.stdout | save --raw ($stem + '.stdout')
    $result.stderr | save --raw ($stem + '.stderr')
    verify $freeze
    {
        epoch: $epoch, parent: ($freeze.repositories | first | get head), cwd: $repo,
        started: $started, finished: $finished, argv: $command, exit_code: $result.exit_code,
        source_manifest: (identity $freeze_path), source_before_after: equal,
        environment: ($env | select -o CARGO_HOME CARGO_TARGET_DIR RUSTUP_TOOLCHAIN RUSTFLAGS RUSTDOCFLAGS CARGO_INCREMENTAL CARGO_BUILD_JOBS CARGO_NET_OFFLINE),
        runner: (identity ($output | path join capture.nu)),
        stdout: (identity ($stem + '.stdout')), stderr: (identity ($stem + '.stderr'))
    } | to json | save ($stem + '.json')
    print {name: $name, exit_code: $result.exit_code, started: $started, finished: $finished}
    if $result.exit_code != 0 { print $result.stderr }
    exit $result.exit_code
}
