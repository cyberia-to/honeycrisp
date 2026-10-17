use ../2026-10-17-acpu-package-evidence/support.nu *

def git-files [root: string, revision: string] {
    mut result = []
    for p in (^git -C $root ls-tree -r --name-only $revision | lines | sort
        | where {|p| not ($p | str starts-with audit/) and not ($p | str starts-with .claude/)}) {
        let body = (do { ^git -C $root show $'($revision):($p)' } | complete)
        require ($body.exit_code == 0) 'Git input unavailable'
        $result = ($result | append {path: $p, sha256: ($body.stdout | hash sha256)})
    }
    $result
}

def main [repo: string, epoch: string, revision: string] {
    let root = ($repo | path expand)
    let out = ($root | path join audit/2026-10-17-gemv-tape-admission $epoch)
    let source = (open ($out | path join sources.json))
    let result = (open ($out | path join result.json))
    let head = (^git -C $root rev-parse $revision | str trim)
    require ((git-files $root $head) == $source.before.owner) 'committed source differs from gated inputs'
    require ($source.before == $result.after and ($source.before | to json -r | hash sha256) == $result.source_sha256) 'before/after source identity differs'
    let sibling = ($root | path dirname | path join strata | path expand)
    require ($source.before.strata.revision == '56aedb2d12b3126c601eb333419136d403614dbb'
        and (git-files $sibling $source.before.strata.revision) == $source.before.strata.files) 'Strata source differs'
    let historical_root = (open ($out | path join fmt.json)).cwd
    for file in $source.tooling {
        let path = ($file.path | path relative-to $historical_root)
        require ((^git -C $root show $'($head):($path)' | hash sha256) == $file.sha256) 'gate tooling differs from commit'
    }
    mut gates = [
        {name: fmt, argv: [cargo fmt --all -- --check]}
        {name: clippy-workspace, argv: [cargo clippy --workspace --locked --offline -- -D warnings]}
        {name: clippy-acpu, argv: [cargo clippy -p acpu --all-targets --keep-going --locked --offline -- -D warnings]}
        {name: tests-workspace, argv: [cargo test --workspace --locked --offline]}
        {name: build-release, argv: [cargo build --release --workspace --locked --offline]}
        {name: tests-acpu-release, argv: [cargo test -p acpu --lib --tests --release --locked --offline]}
        {name: matmul, argv: [cargo run -p acpu --example matmul --locked --offline]}
        {name: docs-acpu, argv: [cargo doc -p acpu --no-deps --locked --offline]}
    ]
    if $source.scope == acpu {
        for mode in [debug release] {
            for target in [gemv_admission gemv_independent] {
                $gates = ($gates | append {name: $'($target)-($mode)', argv: ([cargo test -p acpu --test $target]
                    | append (if $mode == release { [--release] } else { [] }) | append [--locked --offline])})
            }
        }
    } else {
        require ($source.scope == tape) 'unknown gate scope'
        $gates = ($gates | append [
            {name: unimem-debug, argv: [cargo test -p unimem --locked --offline]}
            {name: unimem-release, argv: [cargo test -p unimem --release --locked --offline]}
            {name: unimem-no-std, argv: [nu unimem/scripts/check-no-std.nu]}
        ])
    }
    let envs = $source.environment
    require ($envs.RUSTUP_TOOLCHAIN == '1.95.0' and $envs.RUSTFLAGS == '-Dwarnings'
        and $envs.RUSTDOCFLAGS == '-Dwarnings' and ($envs.RUSTC | path dirname) == ($envs.RUSTDOC | path dirname)
        and ($envs.PATH | split row (char esep) | first) == ($envs.RUSTC | path dirname)) 'pinned gate environment differs'
    mut totals = []
    for gate in $gates {
        let r = (open ($out | path join $'($gate.name).json'))
        require ($r.command == $gate.argv and $r.exit_code == 0 and $r.expected_exit_code == 0
            and $r.cwd == $historical_root and $r.env == $envs) 'gate command/outcome/environment differs'
        for stream in [stdout stderr] {
            require ((digest ($out | path join $'($gate.name).($stream)')) == ($r | get $'($stream)_sha256')) 'raw stream differs'
        }
        if 'test' in $gate.argv {
            let lines = (open --raw ($out | path join $'($gate.name).stdout') | lines | where {|l| $l | str starts-with 'test result: '})
            let rows = ($lines | parse -r '^test result: ok[.] (?<passed>[0-9]+) passed; 0 failed; (?<ignored>[0-9]+) ignored; 0 measured; 0 filtered out;')
            require (($rows | length) > 0 and ($rows | length) == ($lines | length)) 'raw tests failed or were filtered'
            $totals = ($totals | append {name: $gate.name, summaries: ($rows | length), passed: ($rows.passed | into int | math sum), ignored: ($rows.ignored | into int | math sum)})
        }
    }
    require ($totals == $result.tests and ($gates | length) == $result.gates) 'raw totals differ from report'
    let matmul = (open --raw ($out | path join matmul.stdout) | lines | each {|l| $l | str trim})
    require ('max error: 0.00e0' in $matmul and 'result:    PASS' in $matmul) 'matmul correctness differs'
    {revision: $head, epoch: $epoch, gates: ($gates | length), tests: $totals, source_sha256: $result.source_sha256} | to json
}
