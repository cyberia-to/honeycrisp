# Read-only replay against a supplied canonical checkout; no Cargo gates.
def main [repo: string, canonical: string] {
    let root = ($repo | path expand)
    let target = ($canonical | path expand)
    let relative = 'audit/2026-10-16-acpu-delivery'
    let out = ($root | path join $relative portability)
    let cases = [
        {name: numeric-canonical, script: ($target | path join $relative numeric-implementation/verify.nu), expected: 0}
        {name: historical-refusal, script: ($root | path join $relative verify-delivery.nu), expected: 1}
    ]
    let results = ($cases | each {|case|
        let source = (open --raw $case.script | hash sha256)
        let before = (^git -C $target rev-parse HEAD | str trim)
        let started = (date now | format date '%+')
        let result = (do { ^nu $case.script $target } | complete)
        if $source != (open --raw $case.script | hash sha256) or $before != (^git -C $target rev-parse HEAD | str trim) {
            error make {msg: 'replay inputs moved'}
        }
        $result.stdout | save ($out | path join $'($case.name).stdout')
        $result.stderr | save ($out | path join $'($case.name).stderr')
        let receipt = {
            command: [nu $case.script $target], source_revision: (^git -C $root rev-parse HEAD | str trim),
            target_revision: $before, probe_source_sha256: $source, source_unchanged: true,
            started: $started, finished: (date now | format date '%+'), exit_code: $result.exit_code,
            expected_exit_code: $case.expected, stdout_sha256: ($result.stdout | hash sha256),
            stderr_sha256: ($result.stderr | hash sha256)
        }
        $receipt | to json | save ($out | path join $'($case.name).json')
        if $result.exit_code != $case.expected { error make {msg: $'unexpected outcome: ($case.name)'} }
        if $case.name == historical-refusal and not ($result.stderr | str contains 'inventory changed: honeycrisp; historical delivery-v4 source') {
            error make {msg: 'historical replay did not refuse the changed canonical inventory'}
        }
        {name: $case.name, exit_code: $result.exit_code, expected_exit_code: $case.expected}
    })
    {verified: true, checks: $results} | to json
}
