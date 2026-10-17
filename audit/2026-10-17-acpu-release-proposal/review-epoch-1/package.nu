use common.nu *

def main [repo: string, output: string, target: string, tools: string, owning: string] {
    let root = ($repo | path expand)
    let out = ($output | path expand)
    let committed = (candidate $root)
    let owner = (open ($owning | path expand | path join sources.json))
    require ($owner.inputs.owner == $committed.files) 'owning gates cover different source bytes'
    for gate in (owning-gates) {
        let r = (open ($owning | path join $'($gate.name).json'))
        require ($r.command == $gate.argv and $r.exit_code == 0) 'owning gate is not green'
    }
    fresh $out
    let tooling = (tool-bundle ($tools | path expand))
    require ($tooling == $owner.tooling) 'tool bundle differs from owning gates'
    let before = (snapshot $root)
    let envs = (environment $target)
    let index = ($out | path join registry-index.jsonl)
    capture $out index [curl --fail --location --silent --show-error --output $index 'https://index.crates.io/ac/pu/acpu'] $root {} | ignore
    let records = (open --raw $index | lines | each {|line| $line | from json})
    require (($records | where vers == $version | is-empty)) 'candidate version is already occupied, including yanked releases'
    let published = ($records | where vers == '0.2.0' | first)
    require ($published.cksum == '81089272e2a248781b1e2c38ad7247a98bea63a602c0a445dc63e955b2035375' and not $published.yanked) 'published baseline identity changed'
    let archive = ($out | path join published.crate)
    capture $out archive [curl --fail --location --silent --show-error --output $archive 'https://static.crates.io/crates/acpu/acpu-0.2.0.crate'] $root {} | ignore
    require ((digest $archive) == $published.cksum) 'published checksum differs'
    let baseline = (extract $archive ($out | path join baseline) acpu-0.2.0)
    let baseline_vcs = (open ($baseline | path join .cargo_vcs_info.json))
    require ($baseline_vcs.git.sha1 == '96fa4f4509d7a966cb22a4bb9e12ac753cff43ae' and $baseline_vcs.path_in_vcs == acpu) 'published VCS differs'
    let baseline_files = (package-source $root $baseline_vcs.git.sha1 $baseline)
    capture $out package [cargo package -p acpu --locked --offline] $root $envs | ignore
    let built = ($envs.CARGO_TARGET_DIR | path join package $'acpu-($version).crate')
    let candidate_archive = ($out | path join candidate.crate)
    cp $built $candidate_archive
    let current = (extract $candidate_archive ($out | path join current) $'acpu-($version)')
    let vcs = (open ($current | path join .cargo_vcs_info.json))
    require ($vcs.git.sha1 == $committed.head and $vcs.path_in_vcs == acpu and not ($vcs.git.dirty? | default false)) 'candidate package is not clean committed source'
    let inputs = (package-source $root $committed.head $current)
    require ((open ($current | path join Cargo.toml)).package.version == $version) 'normalized version differs'
    let normalized = (open ($current | path join Cargo.toml))
    require ($normalized.dev-dependencies.nebu.package == strata-nebu and $normalized.dev-dependencies.nebu.version == '0.1.1') 'NeBu normalization regressed'
    {candidate: $committed, snapshot: $before, tooling: $tooling, environment: $envs,
        published: $published, index_sha256: (digest $index),
        baseline: {root: $baseline, archive: $archive, sha256: (digest $archive), vcs: $baseline_vcs, files: $baseline_files},
        current: {root: $current, archive: $candidate_archive, sha256: (digest $candidate_archive), vcs: $vcs, files: $inputs},
        publication_authorized: false} | to json | save ($out | path join identities.json)
    for gate in (package-gates) {
        capture $out $gate.name $gate.argv $current $envs | ignore
        require ((files $current) == $inputs and (snapshot $root) == $before) 'package/owner changed during gate'
    }
    capture $out metadata [cargo metadata --no-deps --format-version 1 --locked --offline] $current $envs | ignore
    let targets = (open --raw ($out | path join metadata.stdout) | from json | get packages | where name == acpu | first | get targets)
    require (($targets | length) == 38 and ($targets | where {|t| 'example' in $t.kind} | length) == 32) 'target count differs'
    capture $out owner-metadata [cargo metadata --no-deps --format-version 1 --locked --offline] $root $envs | ignore
    let owning_targets = (open --raw ($out | path join owner-metadata.stdout) | from json | get packages | where name == acpu | first | get targets)
    let relative = {|rows, base| $rows | each {|t| $t | update src_path ($t.src_path | path relative-to $base)} | sort-by name }
    require ((do $relative $targets $current) == (do $relative $owning_targets ($root | path join acpu))) 'packaged targets differ from owner'
    require ((candidate $root) == $committed and (snapshot $root) == $before and (files $current) == $inputs) 'packaging inputs changed'
    require ((tool-bundle ($tools | path expand)) == $tooling) 'tooling changed'
    {gates: 5, targets: 38, examples: 32, package_sha256: (digest $candidate_archive), candidate_commit: $committed.head}
    | to json | save ($out | path join result.json)
}
