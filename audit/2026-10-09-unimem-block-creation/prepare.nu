# Prepare exact-source scratch roots; this script never builds or edits a repository.
def checked [program: string, arguments: list<string>] {
    let result = (run-external $program ...$arguments | complete)
    if $result.exit_code != 0 { error make {msg: $"($program) failed with exit ($result.exit_code): ($result.stderr)"} }
    $result.stdout
}

def archive [repo: string, revision: string, destination: string, paths: list<string>] {
    mkdir $destination
    let tar = $"($destination).tar"
    checked 'git' (['-C', $repo, 'archive', '--format=tar', '--output', $tar, $revision] | append $paths) | ignore
    checked 'tar' ['-xf', $tar, '-C', $destination] | ignore
}

def main [output: string, --honeycrisp: string, --strata: string] {
    let audit = $env.FILE_PWD
    let output = ($output | path expand)
    if $honeycrisp == null or $strata == null { error make {msg: '--honeycrisp and --strata Git repositories are required'} }
    if ($output | path exists) { error make {msg: 'output must be a new private directory'} }
    let files = (open --raw ($audit | path join 'source-freeze.sha256') | lines | parse '{sha256}  {path}' | where {|row| $row.path starts-with 'unimem/' })
    if ($files | length) != 9 { error make {msg: 'expected nine frozen overlay files'} }
    mkdir $output
    let block_revision = '6715335a3283e22218ad5f762316f2c735acb56f'
    let overlay = ($output | path join 'block-source')
    archive $honeycrisp $block_revision $overlay $files.path
    for row in $files {
        if (open --raw ($overlay | path join $row.path) | hash sha256) != $row.sha256 { error make {msg: $"frozen source mismatch: ($row.path)"} }
    }
    for row in [
        {name: candidate, revision: '9593c7218e14e2b5b816d9008b45ab60b9580cf3'},
        {name: origin, revision: '9593c7218e14e2b5b816d9008b45ab60b9580cf3'},
        {name: tag, revision: '96fa4f4509d7a966cb22a4bb9e12ac753cff43ae'}
    ] {
        let destination = ($output | path join 'isolated' $row.name)
        archive $honeycrisp $row.revision $destination ['unimem']
        for name in ['Cargo.toml', 'Cargo.lock'] {
            cp ($audit | path join 'isolated/manifests' $row.name $name) ($destination | path join 'unimem' $name)
        }
    }
    archive $honeycrisp '76b2f26432cca15607ab1dd18e488df68db51ca0' ($output | path join 'overlay/honeycrisp') []
    archive $strata '56aedb2d12b3126c601eb333419136d403614dbb' ($output | path join 'overlay/strata') []
    for destination in ['isolated/candidate', 'overlay/honeycrisp'] {
        for row in $files {
            let path = ($output | path join $destination $row.path)
            mkdir ($path | path dirname)
            cp ($overlay | path join $row.path) $path
            if (open --raw $path | hash sha256) != $row.sha256 { error make {msg: $"copied source mismatch: ($row.path)"} }
        }
    }
    for path in (glob ($audit | path join 'probes/*.rs')) { cp $path ($output | path join 'isolated') }
    {
        classification: 'local reconstruction of measured candidate; not release',
        honeycrisp: '76b2f26432cca15607ab1dd18e488df68db51ca0',
        strata: '56aedb2d12b3126c601eb333419136d403614dbb',
        block_revision: $block_revision,
        overlay: $files, manifest_adaptation: 'isolated/manifests only',
        builds_run: false
    } | to json | save ($output | path join 'prepared.json')
    print $"Prepared and verified nine frozen files in both candidates: ($output)"
}
