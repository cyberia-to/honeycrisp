# Read-only Cargo inventory; no owner gates are run.
def main [repo: string] {
    let root = ($repo | path expand)
    let audit = ($root | path join audit/2026-10-16-acpu-delivery)
    let result = (do { ^cargo metadata --manifest-path ($root | path join Cargo.toml) --no-deps --format-version 1 --locked --offline } | complete)
    if $result.exit_code != 0 { error make {msg: $result.stderr} }
    let metadata = ($result.stdout | from json)
    let current = ($metadata.packages | each {|p| {name: $p.name, targets: ($p.targets | select name kind src_path)}})
    let expected = (open ($audit | path join targets-after.json))
    if $current != $expected { error make {msg: 'actual target inventory differs from recorded post-bridge inventory'} }
    let before = (open ($audit | path join targets-before.json))
    let changes = ($current | each {|p|
        let old = ($before | where name == $p.name | get 0.targets)
        {package: $p.name, retained: ($p.targets | length),
         removed: ($old | where {|t| not ($t in $p.targets)}),
         added: ($p.targets | where {|t| not ($t in $old)})}
    })
    if $changes != (open ($audit | path join target-conservation.json)) {
        error make {msg: 'independent target comparison differs'}
    }
    let standalone = ($current | where name in [metal-benches rane-benches])
    if ($standalone.targets | flatten | where {|t| $t.kind != [bin]} | length) != 0 {
        error make {msg: 'standalone benchmark target kind changed'}
    }
    if ($standalone.targets | flatten | length) != 5 {
        error make {msg: 'standalone benchmark targets missing'}
    }
    {verified: true, command: [cargo metadata --no-deps --format-version 1 --locked --offline],
     recorded: (date now | format date '%+'), changes: $changes, standalone: $standalone
    } | to json | save ($audit | path join delivery-v4/independent-targets.json)
}
