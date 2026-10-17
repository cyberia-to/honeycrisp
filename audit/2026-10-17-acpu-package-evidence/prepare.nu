use support.nu *

def main [repo: string, cache: string, epoch: string] {
    let root = ($repo | path expand)
    let cache = ($cache | path expand)
    let audit = ($root | path join audit/2026-10-17-acpu-package-evidence)
    let out = ($audit | path join $epoch)
    require (($epoch | str replace -r '^[a-z0-9-]+$' '') == '') 'invalid epoch'
    require (not ($out | path exists) and not ($cache | path exists)) 'epoch/cache exists'
    mkdir $out $cache
    let before = (committed-owner $root)
    let head = (^git -C $root rev-parse HEAD | str trim)
    let prefix = (^rustup which --toolchain 1.95.0 rustc | str trim | path dirname)
    let environment = {PATH: ([$prefix ($env.HOME | path join .cargo/bin) /opt/homebrew/bin /usr/bin /bin] | str join (char esep)),
        RUSTUP_TOOLCHAIN: '1.95.0', RUSTC: ($prefix | path join rustc), RUSTDOC: ($prefix | path join rustdoc),
        RUSTFLAGS: '-Dwarnings', RUSTDOCFLAGS: '-Dwarnings', CARGO_BUILD_JOBS: '2', CARGO_INCREMENTAL: '0',
        CARGO_TARGET_DIR: ($cache | path join target)}
    [support.nu prepare.nu] | each {|p| {path: $p, sha256: (digest ($audit | path join $p))}}
    | to json | save ($out | path join tooling.json)
    let index = ($out | path join registry-index.jsonl)
    capture $out index [curl --fail --location --silent --show-error --output $index 'https://index.crates.io/ac/pu/acpu'] $root {} | ignore
    let published = (open --raw $index | lines | each {|x| $x | from json} | where vers == '0.2.0' | first)
    require ($published.cksum == '81089272e2a248781b1e2c38ad7247a98bea63a602c0a445dc63e955b2035375' and not $published.yanked) 'published acpu identity differs'
    let archive = ($cache | path join baseline.crate)
    capture $out archive [curl --fail --location --silent --show-error --output $archive 'https://static.crates.io/crates/acpu/acpu-0.2.0.crate'] $root {} | ignore
    require ((digest $archive) == $published.cksum) 'published archive differs'
    let baseline = (extract $archive ($cache | path join baseline) acpu-0.2.0)
    let baseline_vcs = (open ($baseline | path join .cargo_vcs_info.json))
    require ($baseline_vcs.git.sha1 == '96fa4f4509d7a966cb22a4bb9e12ac753cff43ae' and $baseline_vcs.path_in_vcs == acpu) 'published VCS differs'
    let baseline_files = (package-source $root $baseline_vcs.git.sha1 $baseline)
    capture $out package [cargo package -p acpu --locked --offline --allow-dirty] $root $environment | ignore
    let current_archive = ($environment.CARGO_TARGET_DIR | path join package acpu-0.2.0.crate)
    let current = (extract $current_archive ($cache | path join current) acpu-0.2.0)
    let current_vcs = (open ($current | path join .cargo_vcs_info.json))
    require ($current_vcs.git.sha1 == $head and $current_vcs.path_in_vcs == acpu) 'current package VCS differs'
    let current_files = (package-source $root $head $current)
    require ($before == (committed-owner $root)) 'owner changed while packaging'
    {owner_head: $head, owner_files: $before, owner_sha256: ($before | to json --raw | hash sha256), environment: $environment,
        baseline: {root: $baseline, archive: $archive, sha256: (digest $archive), vcs: $baseline_vcs, files: $baseline_files},
        current: {root: $current, archive: $current_archive, sha256: (digest $current_archive), vcs: $current_vcs, files: $current_files},
        published: $published, index_sha256: (digest $index), release_ready: false}
    | to json | save ($out | path join identities.json)
    {out: $out, current: $current} | to json
}
