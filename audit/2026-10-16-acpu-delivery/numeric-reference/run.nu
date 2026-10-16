def main [repo: path, cache: path, label: string] {
    let root = ($repo | path expand)
    let out = ($root | path join audit/2026-10-16-acpu-delivery/numeric-reference)
    let source = ($out | path join oracle.rs)
    let build = ($cache | path expand)
    mkdir $build
    let binary = ($build | path join $'scaled-bessel-oracle-($label)')
    let digest = (open --raw $source | hash sha256)
    let argv = [rustc --edition=2021 -O -D warnings $source -o $binary]
    let compile = (^($argv | first) ...($argv | skip 1) | complete)
    $compile.stdout | save ($out | path join $'($label)-build.stdout')
    $compile.stderr | save ($out | path join $'($label)-build.stderr')
    let run = if $compile.exit_code == 0 { ^$binary | complete } else { null }
    if $run != null {
        $run.stdout | save ($out | path join $'($label).stdout')
        $run.stderr | save ($out | path join $'($label).stderr')
    }
    let receipt = {command: $argv, run_command: [$binary],
        context_revision: (^git -C $root rev-parse HEAD | str trim),
        tool: (^rustc --version | str trim), source_sha256: $digest,
        source_unchanged: ($digest == (open --raw $source | hash sha256)),
        compile_exit: $compile.exit_code,
        build_stdout_sha256: ($compile.stdout | hash sha256),
        build_stderr_sha256: ($compile.stderr | hash sha256),
        run_exit: $run.exit_code?, stdout_sha256: ($run.stdout? | default '' | hash sha256),
        stderr_sha256: ($run.stderr? | default '' | hash sha256)}
    $receipt | to json | save ($out | path join $'($label).json')
    if $run == null { exit $compile.exit_code }
    if $run.exit_code != 0 { print --stderr $run.stderr; exit $run.exit_code }
    $run.stdout | from json | reject rows | print
}
