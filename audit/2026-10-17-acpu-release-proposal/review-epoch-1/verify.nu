use common.nu *

def checked [folder: string, name: string, argv: list<string>, strict: bool] {
    let r = (open ($folder | path join $'($name).json'))
    require ($r.command == $argv and $r.exit_code == 0) $'command/outcome differs: ($name)'
    for stream in [stdout stderr] {
        require ((digest ($folder | path join $'($name).($stream)')) == ($r | get $'($stream)_sha256')) 'raw stream differs'
    }
    if $strict {
        require ($r.env.RUSTUP_TOOLCHAIN == '1.95.0' and $r.env.RUSTFLAGS == '-Dwarnings'
            and $r.env.RUSTDOCFLAGS == '-Dwarnings') 'strict pinned environment differs'
    }
    $r
}

# Replay an explicit candidate revision even while the checkout is on the audit branch.
def main [repo: string, evidence: string, tools: string, extraction: string] {
    let root = ($repo | path expand)
    let records = ($evidence | path expand)
    let tmp = ($extraction | path expand)
    fresh $tmp
    let tooling = (tool-bundle ($tools | path expand))
    for tool in $tooling.manifest.files {
        let committed = (^git -C $root show $'($tooling.manifest.revision):($tool.path)')
        require (($committed | hash sha256) == $tool.sha256) 'tool bundle differs from its Git revision'
    }
    let pair_dir = ($records | path join package)
    let pair = (open ($pair_dir | path join identities.json))
    let candidate = (candidate-revision $root $pair.candidate.head)
    require ($candidate == $pair.candidate and $pair.tooling == $tooling) 'candidate source/tool identity differs'
    let owner_dir = ($records | path join owning)
    let owner = (open ($owner_dir | path join sources.json))
    require ($owner.inputs.owner == $candidate.files and $owner.tooling == $tooling) 'owning gates covered different candidate bytes'
    let sibling = ($root | path dirname | path join strata | path expand)
    require ($owner.inputs.strata_head == $strata_revision and $owner.inputs.strata == (git-inputs $sibling $strata_revision)) 'recorded Strata source differs'
    for gate in (owning-gates) { checked $owner_dir $gate.name $gate.argv true | ignore }
    checked $pair_dir package [cargo package -p acpu --locked --offline] true | ignore
    for gate in (package-gates) { checked $pair_dir $gate.name $gate.argv true | ignore }
    let baseline_archive = ($pair_dir | path join published.crate)
    let candidate_archive = ($pair_dir | path join candidate.crate)
    require ((digest $baseline_archive) == '81089272e2a248781b1e2c38ad7247a98bea63a602c0a445dc63e955b2035375'
        and (digest $candidate_archive) == $pair.current.sha256) 'crate archive differs'
    let index = (open --raw ($pair_dir | path join registry-index.jsonl) | lines | each {|line| $line | from json})
    require ((digest ($pair_dir | path join registry-index.jsonl)) == $pair.index_sha256
        and ($index | where vers == $version | is-empty)) 'recorded candidate availability differs'
    require (($index | where vers == '0.2.0' | first) == $pair.published and not $pair.published.yanked) 'published index record differs'
    let baseline = (extract $baseline_archive ($tmp | path join baseline) acpu-0.2.0)
    let current = (extract $candidate_archive ($tmp | path join current) $'acpu-($version)')
    for item in [{root: $baseline, side: baseline, revision: '96fa4f4509d7a966cb22a4bb9e12ac753cff43ae'},
        {root: $current, side: current, revision: $candidate.head}] {
        let input = ($pair | get $item.side)
        require ((package-source $root $item.revision $item.root) == $input.files) 'archive source inventory/bytes differ'
        let vcs = (open ($item.root | path join .cargo_vcs_info.json))
        require ($vcs == $input.vcs and $vcs.git.sha1 == $item.revision and $vcs.path_in_vcs == acpu) 'archive VCS differs'
    }
    require ((open ($current | path join Cargo.toml)).package.version == $version) 'normalized version differs'
    let api_dir = ($records | path join api)
    let api = (open ($api_dir | path join result.json))
    require ($api.package_identity_sha256 == (digest ($pair_dir | path join identities.json)) and $api.tooling == $tooling) 'API package/tool link differs'
    let artifacts = ([baseline current] | each {|side|
        let r = (checked $api_dir $side [cargo rustdoc --lib --locked --offline --target aarch64-apple-darwin -- -Z unstable-options --output-format json] false)
        let a = ($api.artifacts | where side == $side | first)
        require ($r.cwd == ($pair | get $side | get root) and $r.env.RUSTUP_TOOLCHAIN == '1.95.0'
            and $r.env.RUSTC_BOOTSTRAP == '1' and $r.env.RUSTFLAGS == '' and $r.env.RUSTDOCFLAGS == '') 'API extraction scope differs'
        require ($a.path == ($r.env.CARGO_TARGET_DIR | path join aarch64-apple-darwin/doc/acpu.json)
            and (digest ($api_dir | path join $a.retained)) == $a.sha256) 'API artifact link differs'
        $a | insert target $r.env.CARGO_TARGET_DIR
    })
    require ($artifacts.0.path != $artifacts.1.path and $artifacts.0.target != $artifacts.1.target and $pair.baseline.root != $pair.current.root) 'API sides share an artifact/source/target'
    checked $api_dir comparison [cargo-semver-checks check-release --baseline-rustdoc $artifacts.0.path --current-rustdoc $artifacts.1.path --release-type patch --color never] false | ignore
    let summary = (open --raw ($api_dir | path join comparison.stderr))
    require (($summary | str contains '229 checks: 229 pass, 31 skip') and ($summary | str contains 'Summary no semver update required')) 'raw API summary differs; review required'
    let consumer_dir = ($records | path join consumer)
    let consumer = (open ($consumer_dir | path join result.json))
    require ($consumer.package_identity_sha256 == (digest ($pair_dir | path join identities.json)) and $consumer.tooling == $tooling
        and $consumer.nox_head == $nox_revision and $consumer.diagnostic_override and not $consumer.unpatched_registry_acceptance) 'consumer source/scope differs'
    let nox_archive = ($consumer_dir | path join nox.crate)
    require ((digest $nox_archive) == $consumer.nox.sha256) 'Nox archive differs'
    let nox = (extract $nox_archive ($tmp | path join nox) cyber-nox-0.2.0)
    require ((files $nox) == $consumer.nox.files and (open ($nox | path join .cargo_vcs_info.json)) == $consumer.nox.vcs
        and $consumer.nox.vcs.git.sha1 == $nox_revision and $consumer.nox.vcs.path_in_vcs == rs) 'Nox files/VCS differ'
    let source = (files $nox | where {|f| $f.path not-in ['.cargo_vcs_info.json' Cargo.lock Cargo.toml]}
        | each {|f| $f | update path (if $f.path == Cargo.toml.orig { 'rs/Cargo.toml' } else { $'rs/($f.path)' })} | sort-by path)
    require ($source == ($consumer.nox_inputs | where {|f| ($f.path | str starts-with rs/) and $f.path != 'rs/Cargo.lock'})) 'Nox source/body link differs'
    let scratch = ($consumer_dir | path join scratch)
    require ((files $scratch) == $consumer.consumer.files and (digest ($scratch | path join probe.rs)) == $probe_sha256
        and $consumer.consumer.probe_sha256 == $probe_sha256) 'consumer inputs differ'
    let manifest = (open ($scratch | path join Cargo.toml))
    let lock = (open --raw ($scratch | path join Cargo.lock) | from toml)
    let metadata = (open --raw ($consumer_dir | path join metadata.stdout) | from json)
    require ((registry-identities $metadata $lock) == $consumer.dependencies) 'registry identity summary differs'
    require ($manifest.dependencies.acpu == $'=($version)' and $manifest.dependencies.nox.features == [honeycrisp]
        and ($manifest.patch.crates-io | columns) == [acpu] and $manifest.patch.crates-io.acpu.path == $pair.current.root) 'consumer patch scope differs'
    let selected = ($metadata.packages | where name == acpu)
    require (($selected | length) == 1 and ($selected | first | get version) == $version
        and ($selected | first | get manifest_path | path dirname) == $pair.current.root) 'consumer selected a different acpu'
    for mode in [debug release] {
        let argv = (if $mode == debug { [cargo test --test source_contract --locked --offline] } else { [cargo test --test source_contract --release --locked --offline] })
        checked $consumer_dir $mode $argv true | ignore
        let raw = (open --raw ($consumer_dir | path join $'($mode).stdout'))
        require ($raw | str contains 'test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;') 'raw consumer count differs'
        for name in [raw_field_outputs_can_be_noncanonical upstream_multilinear_matches_independent_hypercube_sum upstream_ntt_matches_independent_dft_in_both_directions nox_source_backend_matches_cpu_result_budget_and_trace] {
            require ($raw | str contains $'test ($name) ... ok') 'original consumer test missing'
        }
    }
    for receipt in (glob ($records | path join '**/*.json')) {
        let r = (open $receipt)
        if ($r | describe | str starts-with record) and $r.stdout_sha256? != null and $r.stderr_sha256? != null {
            let stem = ($receipt | str replace -r '\.json$' '')
            for stream in [stdout stderr] { require ((digest $'($stem).($stream)') == ($r | get $'($stream)_sha256')) 'raw historical stream differs' }
        }
    }
    {candidate: $candidate.head, version: $version, package_sha256: $pair.current.sha256,
        owning: 8, package: 5, consumer_modes: 2, api: verified, publication_authorized: false,
        scope: 'replay of committed branch inputs; fresh origin/main delivery required after owner merge'} | to json
}
