use ../2026-10-17-acpu-package-evidence/support.nu *

def snapshot [root: string] {
    let strata = ($root | path dirname | path join strata | path expand)
    let revision = (^git -C $strata rev-parse HEAD | str trim)
    require ($revision == '56aedb2d12b3126c601eb333419136d403614dbb') 'Strata revision changed'
    {head: (^git -C $root rev-parse HEAD | str trim), owner: (owner-files $root),
        strata: {revision: $revision, files: (committed-owner $strata)}}
}

def summaries [raw: string] {
    let lines = ($raw | lines | where {|line| $line | str starts-with 'test result: '})
    let rows = ($lines | parse -r '^test result: ok[.] (?<passed>[0-9]+) passed; (?<failed>[0-9]+) failed; (?<ignored>[0-9]+) ignored; (?<measured>[0-9]+) measured; (?<filtered>[0-9]+) filtered out;')
    require (($rows | length) > 0 and ($rows | length) == ($lines | length)) 'missing or failed test summaries'
    for r in $rows { require ($r.failed == '0' and $r.measured == '0' and $r.filtered == '0') 'filtered or failed tests' }
    {summaries: ($rows | length), passed: ($rows.passed | into int | math sum), ignored: ($rows.ignored | into int | math sum)}
}

def main [repo: string, output: string, target: string, scope: string] {
    require ($scope in [acpu tape]) 'scope must be acpu or tape'
    let root = ($repo | path expand)
    let out = ($output | path expand)
    require (not ($out | path exists)) 'output exists'
    mkdir $out
    let prefix = (^rustup which --toolchain 1.95.0 rustc | str trim | path dirname)
    let envs = {PATH: ([$prefix ($env.HOME | path join .cargo/bin) /opt/homebrew/bin /usr/bin /bin] | str join (char esep)),
        RUSTUP_TOOLCHAIN: '1.95.0', RUSTC: ($prefix | path join rustc), RUSTDOC: ($prefix | path join rustdoc),
        RUSTFLAGS: '-Dwarnings', RUSTDOCFLAGS: '-Dwarnings', CARGO_BUILD_JOBS: '2', CARGO_INCREMENTAL: '0',
        CARGO_TARGET_DIR: ($target | path expand)}
    let before = (snapshot $root)
    let tooling = ([($env.FILE_PWD | path join gates.nu) ($env.FILE_PWD | path join ../2026-10-17-acpu-package-evidence/support.nu)]
        | each {|p| {path: ($p | path expand), sha256: (digest $p)}})
    {scope: $scope, before: $before, environment: $envs, tooling: $tooling} | to json | save ($out | path join sources.json)
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
    if $scope == acpu {
        for mode in [debug release] {
            for target in [gemv_admission gemv_independent] {
                let argv = ([cargo test -p acpu --test $target] | append (if $mode == release { [--release] } else { [] }) | append [--locked --offline])
                $gates = ($gates | append {name: $'($target)-($mode)', argv: $argv})
            }
        }
    } else {
        $gates = ($gates | append [
            {name: unimem-debug, argv: [cargo test -p unimem --locked --offline]}
            {name: unimem-release, argv: [cargo test -p unimem --release --locked --offline]}
            {name: unimem-no-std, argv: [nu unimem/scripts/check-no-std.nu]}
        ])
    }
    mut tests = []
    for gate in $gates {
        print $gate.name
        capture $out $gate.name $gate.argv $root $envs | ignore
        require ((snapshot $root) == $before) 'source changed during gate'
        for file in $tooling { require ((digest $file.path) == $file.sha256) 'gate tooling changed' }
        let raw = (open --raw ($out | path join $'($gate.name).stdout'))
        if 'test' in $gate.argv { $tests = ($tests | append ({name: $gate.name} | merge (summaries $raw))) }
    }
    let matmul = (open --raw ($out | path join matmul.stdout))
    require (($matmul | str contains 'max error: 0.00e0') and ($matmul | str contains 'result:    PASS')) 'matmul correctness failed'
    {scope: $scope, gates: ($gates | length), tests: $tests, after: (snapshot $root),
        source_sha256: ($before | to json -r | hash sha256)} | to json | save ($out | path join result.json)
}
