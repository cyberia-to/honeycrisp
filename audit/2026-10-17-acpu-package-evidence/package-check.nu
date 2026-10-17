use support.nu *

# Runs one command inside an untouched normalized package, with an external target.
def main [repo: string, pair: string, pair_cache: string, target: string, epoch: string, name: string, argv_json: string, expected: int = 0] {
    let root = ($repo | path expand)
    let audit = ($root | path join audit/2026-10-17-acpu-package-evidence)
    let identity_file = ($audit | path join $pair identities.json)
    let identity = (open $identity_file)
    let package = ($pair_cache | path expand | path join current acpu-0.2.0)
    let out = ($audit | path join $epoch)
    require (($epoch | str replace -r '^[a-z0-9-]+$' '') == '') 'invalid epoch'
    require (($name | str replace -r '^[a-z0-9-]+$' '') == '') 'invalid gate name'
    require ((committed-owner $root) == $identity.owner_files) 'owner differs from package inputs'
    require ((files $package) == $identity.current.files) 'package inputs changed'
    mkdir $out
    let source = {pair: $pair, pair_sha256: (digest $identity_file),
        tooling: ([support.nu package-check.nu] | each {|p| {path: $p, sha256: (digest ($audit | path join $p))}})}
    let source_file = ($out | path join sources.json)
    if ($source_file | path exists) { require ((open $source_file) == $source) 'check source epoch changed' } else { $source | to json | save $source_file }
    let environment = ($identity.environment | upsert CARGO_TARGET_DIR ($target | path expand))
    let argv = ($argv_json | from json)
    let result = (capture $out $name $argv $package $environment $expected)
    require ((files $package) == $identity.current.files and (committed-owner $root) == $identity.owner_files) 'inputs changed during package check'
    {name: $name, exit_code: $result.exit_code, expected: $expected} | to json
}
