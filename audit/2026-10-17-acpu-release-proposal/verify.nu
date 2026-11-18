use common.nu *
use replay.nu *

# Replay an explicit candidate revision even while the checkout is on the audit branch.
def main [repo: string, evidence: string, tools: string, consumer_tools: string, validation_tools: string, extraction: string] {
    let root = ($repo | path expand)
    let records = ($evidence | path expand)
    let requested = ($extraction | path expand)
    require (($env.CURRENT_FILE | path expand) == ($validation_tools | path expand | path join audit/2026-10-17-acpu-release-proposal/verify.nu)) 'verifier must execute from the declared exported validation bundle'
    fresh $requested
    let tmp = ($requested | path expand)
    let tooling = (tool-bundle ($tools | path expand))
    let consumer_tooling = (tool-bundle ($consumer_tools | path expand))
    let validation_tooling = (tool-bundle ($validation_tools | path expand))
    for bundle in [$tooling $consumer_tooling $validation_tooling] {
        for tool in $bundle.manifest.files {
            let committed = (do { ^git -C $root show $'($bundle.manifest.revision):($tool.path)' } | complete)
            require ($committed.exit_code == 0 and ($committed.stdout | hash sha256) == $tool.sha256) 'tool bundle differs from its Git revision'
        }
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
    let packaging = (checked $pair_dir package [cargo package -p acpu --locked --offline] true)
    for gate in (owning-gates) {
        let r = (checked $owner_dir $gate.name $gate.argv true)
        require ($r.cwd == $packaging.cwd) 'owning gate cwd differs'
    }
    for gate in (package-gates) {
        let r = (checked $pair_dir $gate.name $gate.argv true)
        require ($r.cwd == $pair.current.root) 'normalized package gate cwd differs'
    }
    test-count $owner_dir tests-workspace true | ignore
    test-count $owner_dir tests-acpu-release false | ignore
    for mode in [debug release] { test-count $pair_dir $mode false | ignore }
    for folder in [$owner_dir $pair_dir] { matmul-success (open --raw ($folder | path join matmul.stdout)) }
    let targets = (target-inventory $pair_dir $pair.current.root $packaging.cwd)
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
        if $item.side == current { clean-vcs $vcs $candidate.head acpu }
    }
    require ((open ($current | path join Cargo.toml)).package.version == $version) 'normalized version differs'
    capture $tmp target-replay [cargo metadata --no-deps --format-version 1 --locked --offline] $current (environment ($tmp | path join target)) | ignore
    require ((normalized-targets (open --raw ($tmp | path join target-replay.stdout) | from json) $current) == $targets
        and (files $current) == $pair.current.files) 'fresh normalized target replay differs'
    let api_dir = ($records | path join api)
    let api = (open ($api_dir | path join result.json))
    require ($api.package_identity_sha256 == (digest ($pair_dir | path join identities.json)) and $api.tooling == $tooling) 'API package/tool link differs'
    api-binaries $api_dir $api | ignore
    let cargo = ($api.binaries | where name == cargo | first | get path)
    let artifacts = ([baseline current] | each {|side|
        let r = (checked $api_dir $side [$cargo rustdoc --lib --locked --offline --target aarch64-apple-darwin -- -Z unstable-options --output-format json] false)
        api-environment $r.env $api.binaries
        let a = ($api.artifacts | where side == $side | first)
        require ($r.cwd == ($pair | get $side | get root) and $r.env.RUSTUP_TOOLCHAIN == '1.95.0'
            and $r.env.RUSTC_BOOTSTRAP == '1' and $r.env.RUSTFLAGS == '' and $r.env.RUSTDOCFLAGS == '') 'API extraction scope differs'
        require ($a.path == ($r.env.CARGO_TARGET_DIR | path join aarch64-apple-darwin/doc/acpu.json)
            and (digest ($api_dir | path join $a.retained)) == $a.sha256) 'API artifact link differs'
        $a | insert target $r.env.CARGO_TARGET_DIR
    })
    require ($artifacts.0.path != $artifacts.1.path and $artifacts.0.target != $artifacts.1.target and $pair.baseline.root != $pair.current.root) 'API sides share an artifact/source/target'
    let comparison = (checked $api_dir comparison [$api.semver.path check-release --baseline-rustdoc $artifacts.0.path --current-rustdoc $artifacts.1.path --release-type patch --color never] false)
    require ($comparison.cwd == $api.root) 'API comparison cwd differs'
    api-environment $comparison.env $api.binaries
    let summary = (open --raw ($api_dir | path join comparison.stderr))
    require (($summary | str contains '229 checks: 229 pass, 31 skip') and ($summary | str contains 'Summary no semver update required')) 'raw API summary differs; review required'
    let consumer_dir = ($records | path join consumer)
    let consumer = (open ($consumer_dir | path join result.json))
    require ($consumer.package_identity_sha256 == (digest ($pair_dir | path join identities.json)) and $consumer.tooling == $consumer_tooling
        and $consumer.nox_head == $nox_revision and $consumer.diagnostic_override and not $consumer.unpatched_registry_acceptance) 'consumer source/scope differs'
    let nox_archive = ($consumer_dir | path join nox.crate)
    require ((digest $nox_archive) == $nox_archive_sha256 and $consumer.nox.sha256 == $nox_archive_sha256) 'Nox archive differs from independent PR21 anchor'
    let nox = (extract $nox_archive ($tmp | path join nox) cyber-nox-0.2.0)
    require ((files $nox) == $consumer.nox.files and (open ($nox | path join .cargo_vcs_info.json)) == $consumer.nox.vcs
        and $consumer.nox.vcs.git.sha1 == $nox_revision and $consumer.nox.vcs.path_in_vcs == rs) 'Nox files/VCS differ'
    clean-vcs $consumer.nox.vcs $nox_revision rs
    let source = (files $nox | where {|f| $f.path not-in ['.cargo_vcs_info.json' Cargo.lock Cargo.toml]}
        | each {|f| $f | update path (if $f.path == Cargo.toml.orig { 'rs/Cargo.toml' } else { $'rs/($f.path)' })} | sort-by path)
    require ($source == ($consumer.nox_inputs | where {|f| ($f.path | str starts-with rs/) and $f.path != 'rs/Cargo.lock'})) 'Nox source/body link differs'
    let scratch = ($consumer_dir | path join scratch)
    require ((files $scratch) == $consumer.consumer.files and (digest ($scratch | path join probe.rs)) == $probe_sha256
        and $consumer.consumer.probe_sha256 == $probe_sha256) 'consumer inputs differ'
    let manifest = (open ($scratch | path join Cargo.toml))
    let lock = (open --raw ($scratch | path join Cargo.lock) | from toml)
    let metadata = (open --raw ($consumer_dir | path join metadata.stdout) | from json)
    let metadata_receipt = (checked $consumer_dir metadata [cargo metadata --format-version '1' --locked --offline] true)
    require ($metadata_receipt.cwd == $consumer.consumer.root) 'consumer metadata cwd differs'
    require ((registry-identities $metadata $lock) == $consumer.dependencies) 'registry identity summary differs'
    require ($consumer.nox.root == ($consumer.consumer.root | path dirname | path join nox cyber-nox-0.2.0)) 'consumer normalized Nox root differs'
    consumer-manifest $manifest $metadata $consumer.consumer.root $pair.current.root $consumer.nox.root $version
    for mode in [debug release] {
        let argv = (if $mode == debug { [cargo test --test source_contract --locked --offline] } else { [cargo test --test source_contract --release --locked --offline] })
        let r = (checked $consumer_dir $mode $argv true)
        require ($r.cwd == $consumer.consumer.root) 'consumer test cwd differs'
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
        production_tools: $tooling.sha256, consumer_tools: $consumer_tooling.sha256, validation_tools: $validation_tooling.sha256,
        scope: 'replay of committed branch inputs; fresh origin/main delivery required after owner merge'} | to json
}
