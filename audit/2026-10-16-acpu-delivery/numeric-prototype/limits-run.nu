def main [repo: path, cache: path, label: string] {
    let root = ($repo | path expand)
    let out = ($root | path join audit/2026-10-16-acpu-delivery/numeric-prototype)
    let files = ([limits.rs recurrence.rs] | each {|p| let path = ($out | path join $p); {path: $p, sha256: (open --raw $path | hash sha256)}})
    let build = ($cache | path expand)
    mkdir $build
    let binary = ($build | path join $'scaled-limits-($label)')
    let argv = [rustc --edition=2021 -O -D warnings ($out | path join limits.rs) -o $binary]
    let compile = (^($argv | first) ...($argv | skip 1) | complete)
    $compile.stdout | save ($out | path join $'($label)-build.stdout')
    $compile.stderr | save ($out | path join $'($label)-build.stderr')
    let run = if $compile.exit_code == 0 { ^$binary | complete } else { null }
    if $run != null {
        $run.stdout | save ($out | path join $'($label).stdout')
        $run.stderr | save ($out | path join $'($label).stderr')
    }
    for f in $files {
        if (open --raw ($out | path join $f.path) | hash sha256) != $f.sha256 { error make {msg: 'prototype source changed'} }
    }
    {command: $argv, run_command: [$binary], context_revision: (^git -C $root rev-parse HEAD | str trim),
        tool: (^rustc --version | str trim), sources: $files, compile_exit: $compile.exit_code,
        build_stdout_sha256: ($compile.stdout | hash sha256), build_stderr_sha256: ($compile.stderr | hash sha256),
        run_exit: $run.exit_code?, stdout_sha256: ($run.stdout? | default '' | hash sha256),
        stderr_sha256: ($run.stderr? | default '' | hash sha256), source_before_after: equal
    } | to json | save ($out | path join $'($label).json')
    if $run == null { exit $compile.exit_code }
    if $run.exit_code != 0 { print --stderr $run.stderr; exit $run.exit_code }
    $run.stdout | from json  | print
}
