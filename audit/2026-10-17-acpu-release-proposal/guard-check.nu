use common.nu *

# Pure proof checks use retained real metadata, then explicit synthetic corruption.
def rejects [operation: closure, message: string] {
    let outcome = (try { do $operation | ignore; {rejected: false, message: ''} } catch {|e| {rejected: true, message: $e.msg}})
    require ($outcome.rejected and ($outcome.message | str contains $message)) 'guard accepted corrupt input or refused for another reason'
}

def main [repo: string] {
    let root = ($repo | path expand)
    let old = ($root | path join audit/2026-10-17-acpu-package-evidence/consumer-2)
    let metadata = (open --raw ($old | path join metadata.stdout) | from json)
    let locked = (open --raw ($old | path join consumer-Cargo.lock.txt) | from toml)
    let expected = (open ($old | path join result.json))
    require ((registry-identities $metadata $locked) == $expected.dependencies) 'real prior proof differs'
    let nebu = ($metadata.packages | where name == strata-nebu | first)
    let duplicate = ($metadata | update packages ($metadata.packages | append $nebu))
    rejects { registry-identities $duplicate $locked } 'duplicate/missing registry identity'
    let wrong_lock = ($locked | update package ($locked.package | each {|p|
        if $p.name == strata-nebu { $p | update checksum 'changed' } else { $p }
    }))
    rejects { registry-identities $metadata $wrong_lock } 'registry source/version/checksum differs'
    let fast = ($metadata | update resolve.nodes ($metadata.resolve.nodes | each {|node|
        if $node.id == $nebu.id { $node | update features ($node.features | append fast) } else { $node }
    }))
    rejects { registry-identities $fast $locked } 'NeBu fast enabled'
    let wrong_type = ($metadata | update resolve.nodes ($metadata.resolve.nodes | each {|node|
        if $node.id == $metadata.resolve.root {
            $node | update deps ($node.deps | each {|dep| if $dep.name == nebu { $dep | update pkg 'different-type-source' } else { $dep }})
        } else { $node }
    }))
    rejects { registry-identities $wrong_type $locked } 'direct/Nox Goldilocks types differ'
    let source = (open ($root | path join audit/2026-10-17-gemv-tape-admission/tape-1/sources.json)).before.owner
    require ((owner-files $root) == $source and (git-inputs $root $base) == $source) 'compiler source differs from reviewed merge and its green gates'
    {real_registry_control: passed, synthetic_corruption_refusals: 4,
        source_bytes_equal_pr25_green_epoch: true, source_base: $base, candidate_gates_executed: false,
        metadata_sha256: (digest ($old | path join metadata.stdout)), lock_sha256: (digest ($old | path join consumer-Cargo.lock.txt))} | to json
}
