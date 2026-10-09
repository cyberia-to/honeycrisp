def checked [program: string, arguments: list<string>] {
    let result = (run-external $program ...$arguments | complete)
    if $result.exit_code != 0 {
        error make {msg: $"($program) ($arguments | str join ' ') failed: ($result.stderr)"}
    }
}

let source = '/Users/master/cyber/honeycrisp-kadek-block-creation'
let root = '/tmp/unimem-block-creation-validation'
if ($root | path exists) { error make {msg: 'validation root already exists'} }
mkdir $root
let manifest = '[package]
name = "unimem"
version = "0.2.0"
edition = "2021"
license = "LicenseRef-Cyber"
autoexamples = false
autobenches = false

[workspace]

[dependencies]
crossbeam-queue = "=0.3.12"
mutants = "=0.0.3"
'
let lock = (open --raw ($source | path join 'Cargo.lock') | from toml)
let packages = ($lock.package | where {|package| $package.name in ['crossbeam-queue', 'crossbeam-utils', 'mutants'] })
let seeded = {version: $lock.version, package: ($packages | append {name: 'unimem', version: '0.2.0', dependencies: ['crossbeam-queue', 'mutants']})}
for row in [
    {name: candidate, revision: '9593c7218e14e2b5b816d9008b45ab60b9580cf3'},
    {name: origin, revision: '9593c7218e14e2b5b816d9008b45ab60b9580cf3'},
    {name: tag, revision: '96fa4f4509d7a966cb22a4bb9e12ac753cff43ae'}
] {
    let output = ($root | path join $row.name)
    mkdir $output
    let archive = ($root | path join $"($row.name).tar")
    checked 'git' ['-C', $source, 'archive', '--format=tar', '--output', $archive, $row.revision, 'unimem']
    checked 'tar' ['-xf', $archive, '-C', $output]
    $manifest | save ($output | path join 'unimem/Cargo.toml') --force
    $seeded | to toml | save ($output | path join 'unimem/Cargo.lock') --force
}
let files = [
    'unimem/src/block.rs', 'unimem/src/block/creation.rs',
    'unimem/src/block/tests.rs', 'unimem/src/ffi.rs', 'unimem/src/lib.rs',
    'unimem/tests/block_creation.rs', 'unimem/tests/roundtrip.rs'
]
for file in $files {
    let destination = ($root | path join 'candidate' $file)
    mkdir ($destination | path dirname)
    cp ($source | path join $file) $destination
}
checked 'shasum' (['-a', '256'] | append ($files | each {|file| $source | path join $file }))
print $"Created exact-source isolated candidates in ($root); manifest/lock adaptation is scoped to this harness. No original sibling paths."
