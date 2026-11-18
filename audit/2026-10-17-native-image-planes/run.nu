def require [ok: bool, message: string] { if not $ok { error make {msg: $message} } }
def sha [p: path] { open --raw $p | into binary | hash sha256 }
export def sources [root: path] {
    [[$root honeycrisp] [($root | path join ../strata | path expand) strata]] | each {|pair|
        ^git -C $pair.0 ls-files --cached --others --exclude-standard | lines | uniq | sort
            | where {|p| ((not ($p | str starts-with 'audit/')) and
                (($p | str ends-with '.rs') or ($p | str starts-with '.cargo/') or
                (($p | path basename) in [Cargo.toml Cargo.lock Cargo.toml.txt Cargo.lock.txt rust-toolchain.toml rustfmt.toml .rustfmt.toml])
                or $p == 'unimem/scripts/check-no-std.nu')) }
            | each {|p| {path: $'($pair.1)/($p)', sha256: (sha ($pair.0 | path join $p))}}
    } | flatten
}
export def identities [] {
    let rustup = ($env.HOME | path join .cargo/bin/rustup)
    [$rustup /opt/homebrew/bin/nu] | append ([cargo rustc rustdoc cargo-clippy clippy-driver cargo-fmt rustfmt] | each {|n|
        ^$rustup which --toolchain '1.95.0' $n | str trim
    }) | each {|p| {path: $p, sha256: (sha $p)}}
}
def main [repo: path, cache: path, name: string, argv_json: string, --freeze: path] {
    let root = ($repo | path expand)
    let cache = ($cache | path expand)
    cd $root
    let out = ($root | path join audit/2026-10-17-native-image-planes)
    require (not ($out | path join $'($name).json' | path exists)) 'retain prior receipt'
    let rustup = ($env.HOME | path join .cargo/bin/rustup)
    let toolchain = '1.95.0'
    let rustc = (^$rustup which --toolchain $toolchain rustc | str trim)
    let rustdoc = (^$rustup which --toolchain $toolchain rustdoc | str trim)
    let cargo = (^$rustup which --toolchain $toolchain cargo | str trim)
    let command = ([$rustup run $toolchain] | append ($argv_json | from json))
    let run_env = {HOME: $env.HOME, CARGO_HOME: ($env.HOME | path join .cargo),
        RUSTUP_HOME: ($env.HOME | path join .rustup),
        PATH: ([($rustc | path dirname) ($rustup | path dirname) /opt/homebrew/bin /usr/bin /bin /usr/sbin /sbin] | str join ':'),
        RUSTC: $rustc, RUSTDOC: $rustdoc, RUSTFLAGS: '-Dwarnings', RUSTDOCFLAGS: '-Dwarnings',
        CARGO_BUILD_JOBS: '2', CARGO_INCREMENTAL: '0', CARGO_NET_OFFLINE: 'true',
        CARGO_TARGET_DIR: ($cache | path join target), LANG: C, LC_ALL: C, TZ: UTC}
    let inherited = ([RUSTC RUSTDOC RUSTC_WRAPPER RUSTC_WORKSPACE_WRAPPER RUSTFLAGS RUSTDOCFLAGS CARGO_ENCODED_RUSTFLAGS RUSTUP_TOOLCHAIN]
        | each {|k| {name: $k, value: ($env | get -o $k)}})
    let env_args = ($run_env | transpose key value | each {|r| $'($r.key)=($r.value)'})
    let tools = (identities)
    let before = (sources $root)
    let digest = ($before | to json --raw | hash sha256)
    mkdir ($out | path join inputs)
    let manifest = $'inputs/($digest).json'
    if not ($out | path join $manifest | path exists) { $before | to json --raw | save ($out | path join $manifest) }
    let heads = ([$root ($root | path join ../strata | path expand)] | each {|p| ^git -C $p rev-parse HEAD | str trim})
    let runner = (sha ($out | path join run.nu))
    let runner_source = $'runners/($runner).nu.txt'
    if not ($out | path join $runner_source | path exists) { cp ($out | path join run.nu) ($out | path join $runner_source) }
    let freeze_sha = (if $freeze == null { null } else {
        let fixed = (open $freeze)
        let gate = ($fixed.gates | where name == $name | first)
        require ($digest == $fixed.source_sha256 and $heads == $fixed.heads) 'source/HEAD differs from freeze'
        require ($tools == $fixed.tools and $runner == $fixed.runner_sha256) 'tool/runner differs from freeze'
        require (($argv_json | from json) == $gate.argv) 'command differs from freeze'
        sha $freeze
    })
    let started = (date now | format date '%+')
    let result = (do { ^/usr/bin/env -i ...$env_args ...$command } | complete)
    let after = (sources $root)
    let heads_after = ([$root ($root | path join ../strata | path expand)] | each {|p| ^git -C $p rev-parse HEAD | str trim})
    let tools_after = (identities)
    $result.stdout | save ($out | path join $'($name).stdout')
    $result.stderr | save ($out | path join $'($name).stderr')
    {command: $command, launcher: ['/usr/bin/env' '-i'], environment: $run_env, inherited_tool_overrides: $inherited,
        heads_before: $heads, heads_after: $heads_after, cwd: $root,
        source_manifest: $manifest, source_sha256: $digest, sources_equal: ($before == $after),
        runner_sha256: $runner, runner_source: $runner_source, tools: $tools, tools_equal: ($tools == $tools_after),
        freeze_sha256: $freeze_sha,
        versions: ([$cargo $rustc $rustdoc] | each {|p| {path: $p, result: (do { ^/usr/bin/env -i ...$env_args $p -Vv } | complete)}}),
        started: $started, finished: (date now | format date '%+'), exit_code: $result.exit_code,
        stdout_sha256: ($result.stdout | hash sha256), stderr_sha256: ($result.stderr | hash sha256)
    } | to json --raw | save ($out | path join $'($name).json')
    require ($before == $after and $heads == $heads_after) 'source or HEAD changed during command'
    require ((sha ($out | path join run.nu)) == $runner) 'runner changed during command'
    require ($tools == $tools_after) 'tools changed during command'
    print {name: $name, exit_code: $result.exit_code, source_sha256: $digest}
    if $result.exit_code != 0 { print --stderr $result.stderr; exit $result.exit_code }
}
