def checked [program: string, arguments: list<string>] {
    let result = (run-external $program ...$arguments | complete)
    if $result.exit_code != 0 {
        error make {msg: $"($program) failed: ($result.stderr)"}
    }
    $result.stdout
}

let source = '/Users/master/cyber/honeycrisp-kadek-block-creation'
cd $source
checked 'shasum' ['-c', '/tmp/unimem-block-creation-source-freeze.sha256'] | print
let root = (checked 'mktemp' ['-d', '/tmp/unimem-block-overlay.XXXXXX'] | str trim)
mkdir ($root | path join 'honeycrisp') ($root | path join 'strata') ($root | path join 'gates')
for entry in [
    {name: honeycrisp, repo: $source, revision: '76b2f26432cca15607ab1dd18e488df68db51ca0'},
    {name: strata, repo: '/Users/master/cyber/strata', revision: '56aedb2d12b3126c601eb333419136d403614dbb'}
] {
    let archive = ($root | path join $"($entry.name).tar")
    checked 'git' ['-C', $entry.repo, 'archive', '--format=tar', '--output', $archive, $entry.revision] | ignore
    checked 'tar' ['-xf', $archive, '-C', ($root | path join $entry.name)] | ignore
}
let overlay = (open --raw '/tmp/unimem-block-creation-source-freeze.sha256' | lines | parse '{sha256}  {path}' | where {|entry| $entry.path starts-with 'unimem/' })
if ($overlay | length) != 9 { error make {msg: 'expected nine frozen source/test/spec overlay paths'} }
for entry in $overlay {
    let destination = ($root | path join 'honeycrisp' $entry.path)
    mkdir ($destination | path dirname)
    cp ($source | path join $entry.path) $destination
    let actual = (open --raw $destination | hash sha256)
    if $actual != $entry.sha256 { error make {msg: $"overlay mismatch: ($entry.path)"} }
}
{
    classification: 'local combined candidate; not origin/release',
    honeycrisp: '76b2f26432cca15607ab1dd18e488df68db51ca0',
    strata: '56aedb2d12b3126c601eb333419136d403614dbb',
    frozen_overlay_manifest_sha256: 'e28991a40b49a908ff1882ba14079217f39570187d2374c7601943c59368db87',
    overlay: $overlay,
    target: ($root | path join 'target')
} | to json | save ($root | path join 'sources.json')
checked 'shasum' ['-a', '256', ($root | path join 'honeycrisp.tar'), ($root | path join 'strata.tar'), ($root | path join 'sources.json')] | save ($root | path join 'sources.sha256')
$root | save --force '/tmp/unimem-block-overlay-root.txt'
print $root
