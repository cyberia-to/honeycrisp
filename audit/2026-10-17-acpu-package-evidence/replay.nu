use support.nu *

export def api-links [audit: string, epoch: string, identity: record, pair: record] {
    let rows = ([baseline current] | each {|side|
        let receipt = (open ($audit | path join $epoch $'($side).json'))
        let artifact = ($identity.artifacts | where side == $side | first)
        require ($receipt.command == [cargo rustdoc --lib --locked --offline --target aarch64-apple-darwin -- -Z unstable-options --output-format json]
            and $receipt.exit_code == 0 and $receipt.cwd == ($pair | get $side | get root)) 'API extraction source/command differs'
        let derived = ($receipt.env.CARGO_TARGET_DIR | path join aarch64-apple-darwin/doc/acpu.json)
        require ($artifact.path == $derived) 'API artifact did not come from recorded target'
        {cwd: $receipt.cwd, target: $receipt.env.CARGO_TARGET_DIR, artifact: $artifact.path}
    })
    require ($rows.0.cwd != $rows.1.cwd and $rows.0.target != $rows.1.target and $rows.0.artifact != $rows.1.artifact) 'API sides share an artifact/source/target root'
    let summary = (open --raw ($audit | path join $epoch comparison.stderr))
    require ($summary | str contains '229 checks: 229 pass, 31 skip') 'API raw summary differs'
    require ($summary | str contains 'Summary no semver update required') 'API raw verdict differs'
}

export def consumer-links [audit: string, epoch: string, identity: record, pair: record, extraction: string] {
    let folder = ($audit | path join $epoch)
    let metadata = (open --raw ($folder | path join metadata.stdout) | from json)
    let locked = (open --raw ($folder | path join consumer-Cargo.lock.txt) | from toml)
    let manifest = (open --raw ($folder | path join consumer-Cargo.toml.txt) | from toml)
    for entry in [
        {retained: ($folder | path join consumer-Cargo.toml.txt), source: Cargo.toml}
        {retained: ($folder | path join consumer-Cargo.lock.txt), source: Cargo.lock}
        {retained: ($extraction | path join consumer-probe.rs), source: probe.rs}
    ] {
        let recorded = ($identity.consumer.files | where path == $entry.source)
        require (($recorded | length) == 1 and (digest $entry.retained) == ($recorded | first | get sha256)) 'retained consumer input differs'
    }
    require (($identity.consumer.files.path | sort) == [Cargo.lock Cargo.toml probe.rs]) 'unexpected consumer input inventory'
    require ($manifest.patch.crates-io.acpu.path == $pair.current.root and ($manifest.patch.crates-io | columns) == [acpu]
        and $manifest.dependencies.acpu == '=0.2.0' and $manifest.dependencies.nebu.version == '=0.1.1'
        and $manifest.dependencies.nox.features == [honeycrisp]) 'consumer override/feature scope differs'
    let deps = ([
        {name: unimem, version: '0.2.0', checksum: a91794fe345844d09b134c55fde134bc9d51593964b906dde2f667162e49918d}
        {name: strata-nebu, version: '0.1.1', checksum: f4a881cdb3622cd6b38333246f516c70e3e5c63d2bc0c0e48344d8e7b0392766}
    ] | each {|expected|
        let packages = ($metadata.packages | where name == $expected.name)
        let locks = ($locked.package | where name == $expected.name)
        require (($packages | length) == 1 and ($locks | length) == 1) 'duplicate registry identity'
        let pkg = ($packages | first)
        let lock = ($locks | first)
        require ($pkg.version == $expected.version and ($pkg.source | str starts-with 'registry+')
            and $lock.version == $expected.version and $lock.source == $pkg.source and $lock.checksum == $expected.checksum) 'registry version/source/checksum differs'
        let features = ($metadata.resolve.nodes | where id == $pkg.id | first | get features)
        if $expected.name == strata-nebu { require ('fast' not-in $features) 'NeBu fast is enabled' }
        $expected | merge {id: $pkg.id, features: $features}
    })
    require ($deps == $identity.dependencies) 'consumer summary differs from metadata/lock'
    let direct = ($metadata.resolve.nodes | where id == $metadata.resolve.root | first)
    let nox = ($metadata.packages | where name == cyber-nox)
    let acpu = ($metadata.packages | where name == acpu)
    require (($nox | length) == 1 and ($acpu | length) == 1) 'duplicate Nox/acpu package'
    require (($nox | first | get manifest_path | path dirname) == $manifest.dependencies.nox.path
        and ($acpu | first | get manifest_path | path dirname) == $pair.current.root) 'consumer selected different packages'
    let node = ($metadata.resolve.nodes | where id == ($nox | first | get id) | first)
    require (($direct.deps | where name == nebu | first | get pkg) == ($node.deps | where name == nebu | first | get pkg)) 'direct/Nox Goldilocks types differ'
    for mode in [debug release] {
        let raw = (open --raw ($folder | path join $'($mode).stdout'))
        require ($raw | str contains 'test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;') 'consumer raw result differs'
        for test in [raw_field_outputs_can_be_noncanonical upstream_multilinear_matches_independent_hypercube_sum upstream_ntt_matches_independent_dft_in_both_directions nox_source_backend_matches_cpu_result_budget_and_trace] {
            require ($raw | str contains $'test ($test) ... ok') 'original consumer test missing'
        }
    }
}

export def archive-links [root: string, extraction: string, pair: record, consumer: record] {
    let archive = ($extraction | path join published-acpu-0.2.0.crate)
    require ((digest $archive) == '81089272e2a248781b1e2c38ad7247a98bea63a602c0a445dc63e955b2035375') 'official archive checksum differs'
    let published = (extract $archive ($extraction | path join published) acpu-0.2.0)
    let vcs = (open ($published | path join .cargo_vcs_info.json))
    require ($vcs.git.sha1 == '96fa4f4509d7a966cb22a4bb9e12ac753cff43ae' and $vcs.path_in_vcs == acpu
        and $vcs == $pair.baseline.vcs) 'published VCS differs'
    require ((package-source $root $vcs.git.sha1 $published) == $pair.baseline.files) 'published source inventory/bytes differ'
    let nox = (extract ($extraction | path join normalized-nox.crate) ($extraction | path join nox) cyber-nox-0.2.0)
    let nox_vcs = (open ($nox | path join .cargo_vcs_info.json))
    require ((files $nox) == $consumer.nox_archive.files and $nox_vcs == $consumer.nox_archive.vcs
        and $nox_vcs.git.sha1 == $consumer.nox_head and $nox_vcs.path_in_vcs == rs) 'normalized Nox files/VCS differ'
    let source = (files $nox | where {|f| $f.path not-in ['.cargo_vcs_info.json' Cargo.lock Cargo.toml]}
        | each {|f| $f | update path (if $f.path == Cargo.toml.orig { 'rs/Cargo.toml' } else { $'rs/($f.path)' })} | sort-by path)
    let expected = ($consumer.nox_inputs | where {|f| ($f.path | str starts-with rs/) and $f.path != 'rs/Cargo.lock'})
    require ($source == $expected) 'normalized Nox source differs from recorded owning input bytes'
}
