use support.nu *

def main [repo: string, cache: string, pair: string, api: string, consumer: string] {
    let root = ($repo | path expand)
    let cache = ($cache | path expand)
    let audit = ($root | path join audit/2026-10-17-acpu-package-evidence)
    let stage = ($cache | path join retained)
    require (not ($audit | path join evidence.tar.gz | path exists)) 'capsule exists'
    let pair_id = (open ($audit | path join $pair identities.json))
    let api_id = (open ($audit | path join $api result.json))
    let consumer_id = (open ($audit | path join $consumer result.json))
    require ((committed-owner $root) == $pair_id.owner_files) 'final owner changed'
    for entry in [
        {from: ($cache | path join $pair target/package/acpu-0.2.0.crate), to: current-final.crate, expected: $pair_id.current.sha256}
        {from: ($cache | path join $api baseline/aarch64-apple-darwin/doc/acpu.json), to: published-final-api.json,
            expected: ($api_id.artifacts | where side == baseline | first | get sha256)}
        {from: ($cache | path join $api current/aarch64-apple-darwin/doc/acpu.json), to: final-api.json,
            expected: ($api_id.artifacts | where side == current | first | get sha256)}
        {from: ($cache | path join $consumer package-target/package/cyber-nox-0.2.0.crate), to: normalized-nox.crate, expected: $consumer_id.nox_archive.sha256}
        {from: ($cache | path join $consumer consumer/probe.rs), to: consumer-probe.rs, expected: $consumer_id.consumer.probe_sha256}
    ] {
        require ((digest $entry.from) == $entry.expected) 'retained artifact changed'
        cp $entry.from ($stage | path join $entry.to)
    }
    let inventory = (files $stage)
    capture $audit retention [tar -czf ($audit | path join evidence.tar.gz) -C $stage ...$inventory.path] $root {} | ignore
    {schema: 1, pair: $pair, api: $api, consumer: $consumer,
        capsule_sha256: (digest ($audit | path join evidence.tar.gz)), files: $inventory,
        owner_head: $pair_id.owner_head, owner_sha256: $pair_id.owner_sha256,
        release_ready: false} | to json | save ($audit | path join bundle.json)
}
