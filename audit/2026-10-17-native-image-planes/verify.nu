def require [ok: bool, message: string] { if not $ok { error make {msg: $message} } }
def sha [p: path] { open --raw $p | into binary | hash sha256 }
def source-list [root: path, sibling: path] {
    [[$root honeycrisp] [$sibling strata]] | each {|pair|
        ^git -C $pair.0 ls-files --cached --others --exclude-standard | lines | uniq | sort
            | where {|p| ((not ($p | str starts-with 'audit/')) and
                (($p | str ends-with '.rs') or ($p | str starts-with '.cargo/') or
                (($p | path basename) in [Cargo.toml Cargo.lock Cargo.toml.txt Cargo.lock.txt rust-toolchain.toml rustfmt.toml .rustfmt.toml])
                or $p == 'unimem/scripts/check-no-std.nu')) }
            | each {|p| {path: $'($pair.1)/($p)', sha256: (sha ($pair.0 | path join $p))}}
    } | flatten
}
def tests [path: path] {
    open --raw $path | lines | parse -r '^test result: ok\. (?<n>\d+) passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;'
        | get n | into int
}
def receipt [out: path, name: string] {
    let r = (open ($out | path join $'($name).json'))
    require ($r.sources_equal and $r.heads_before == $r.heads_after) $'($name): changed source or HEAD'
    require ((sha ($out | path join $r.source_manifest)) == $r.source_sha256) $'($name): manifest hash'
    require ((sha ($out | path join $'runners/($r.runner_sha256).nu.txt')) == $r.runner_sha256) $'($name): runner hash'
    for stream in [stdout stderr] {
        require ((sha ($out | path join $'($name).($stream)')) == ($r | get $'($stream)_sha256')) $'($name): ($stream) hash'
    }
    require ($r.command.1 == run and $r.command.2 == '1.95.0') $'($name): toolchain selection'
    require ($r.environment.RUSTFLAGS == '-Dwarnings' and $r.environment.RUSTDOCFLAGS == '-Dwarnings') $'($name): warning policy'
    require ($r.environment.CARGO_NET_OFFLINE == 'true') $'($name): offline policy'
    require (($r.environment.RUSTC | path basename) == rustc and ($r.environment.RUSTDOC | path basename) == rustdoc) $'($name): explicit compilers'
    for binary in [$r.environment.RUSTC $r.environment.RUSTDOC] {
        require (($r.tools | where path == $binary | length) == 1) $'($name): compiler identity missing'
    }
    require (($r.versions | all {|v| $v.result.exit_code == 0})) $'($name): version command failed'
    $r
}

# Replay hashes and conclusions without running framework/compiler or historical helpers.
def main [repo: path, strata: path, --unsealed, --live] {
    let root = ($repo | path expand)
    let sibling = ($strata | path expand)
    let out = ($root | path join audit/2026-10-17-native-image-planes)
    if not $unsealed {
        let seal = (open ($out | path join inventory.json))
        let actual = (glob ($out | path join '**/*') | where {|p| ($p | path type) == file}
            | each {|p| $p | path relative-to $out} | where {|p| $p != inventory.json} | sort)
        require ($actual == ($seal | get path)) 'audit file set differs from seal'
        for entry in $seal {
            require ((sha ($out | path join $entry.path)) == $entry.sha256) $'sealed file differs: ($entry.path)'
        }
    }
    let fixed = (open ($out | path join freeze.json))
    let freeze_sha = (sha ($out | path join freeze.json))
    require ($freeze_sha == '1d3dd625fe5a5ee18528e50e40e8c395cd9b9fddad6ea7b009e5d1be3a54a1a0') 'freeze identity'
    require ((sha ($out | path join $fixed.source_manifest)) == $fixed.source_sha256) 'frozen manifest identity'
    let inputs = (source-list $root $sibling)
    require ($inputs == (open ($out | path join $fixed.source_manifest))) 'current source bytes/set differ from final inputs'
    require ((sha ($out | path join run.nu)) == $fixed.runner_sha256) 'current runner differs from final runner'
    for entry in $fixed.authority_files {
        require ((sha ($root | path join $entry.path)) == $entry.sha256) $'authority file differs: ($entry.path)'
    }
    let expected_names = [final-debug final-release final-clippy final-doc final-fmt final-no-std
        workspace-test workspace-release workspace-clippy pipeline]
    require (($fixed.gates | get name) == $expected_names) 'finite gate set differs'
    let verdicts = ($fixed.gates | each {|gate|
        let r = (receipt $out $gate.name)
        require ($r.freeze_sha256 == $freeze_sha and $r.source_sha256 == $fixed.source_sha256) $'($gate.name): freeze/source binding'
        require ($r.heads_before == $fixed.heads and $r.tools == $fixed.tools and $r.tools_equal) $'($gate.name): HEAD/tools binding'
        require ($r.runner_sha256 == $fixed.runner_sha256) $'($gate.name): final runner binding'
        require (($r.command | skip 3) == $gate.argv and $r.launcher == ['/usr/bin/env' '-i']) $'($gate.name): actual command'
        require ($r.environment == (open ($out | path join final-debug.json)).environment) $'($gate.name): environment differs'
        {name: $gate.name, exit_code: $r.exit_code}
    })
    for name in [final-debug final-release] {
        require ((tests ($out | path join $'($name).stdout')) == [15 7 5 4 32 5 17]) $'($name): expected all 85 tests'
    }
    for name in $expected_names {
        require (($verdicts | where name == $name | first | get exit_code) == 0) $'owning gate red: ($name)'
    }
    let workspace = (open --raw ($out | path join workspace-test.stdout) | lines
        | parse -r '^test result: ok\. (?<passed>\d+) passed; 0 failed; (?<ignored>\d+) ignored;'
        | update passed {into int} | update ignored {into int} | math sum)
    require ($workspace == {passed: 400, ignored: 5}) 'workspace test totals differ'
    for name in [parent-test parent-clippy parent-fmt development-native-1 development-native-2 development-clippy] {
        let r = (receipt $out $name)
        require ($r.exit_code == 0) $'historical green differs: ($name)'
    }
    let red = (receipt $out acceptance-missing-api)
    require ($red.exit_code == 101) 'acceptance-first red differs'
    let diagnostic = (open --raw ($out | path join acceptance-missing-api.stderr))
    require (($diagnostic | str contains 'E0432') and ($diagnostic | str contains 'E0599')) 'missing API red diagnostic differs'
    let launcher = (open ($out | path join final-launcher-failure.json))
    require ($launcher.gates_started == 0) 'launcher failure scope'
    require ((sha ($out | path join runners/final-initial-parse-failure.nu.txt)) == $launcher.launcher_sha256) 'failed launcher identity'
    if $live {
        let heads = ([$root $sibling] | each {|p| ^git -C $p rev-parse HEAD | str trim})
        require ($heads == $fixed.heads) 'live capture HEAD differs (ordinary portable replay supports later commits with identical compiler bytes)'
        for tool in $fixed.tools {
            require ((sha $tool.path) == $tool.sha256) $'live tool differs: ($tool.path)'
        }
    }
    print {freeze_sha256: $freeze_sha, source_files: ($inputs | length), final_gates: $verdicts,
        native_tests_per_profile: 85, acceptance_first_exit: $red.exit_code, sealed: (not $unsealed)}
}
