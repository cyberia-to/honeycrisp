def require [condition: bool, message: string] {
    if not $condition { error make {msg: $message} }
}

def body [source: string, start: string, end: string] {
    $source | split row $start | get 1 | split row $end | first
}

def main [repo: string] {
    let root = ($repo | path expand)
    cd $root
    let dir = 'audit/2026-10-16-acpu-delivery/numeric-implementation'
    let final = (open ($dir | path join final/sources.json))
    let first = (open ($dir | path join first/sources.json))
    for manifest in [$first $final] {
        require (($manifest.inputs | to json --raw | hash sha256) == $manifest.source_sha256) 'manifest digest mismatch'
    }
    let checked = ($final.inputs | each {|input|
        let location = if $input.name == 'honeycrisp' { $root } else { $root | path dirname | path join strata }
        let ancestor = (do { ^git -C $location merge-base --is-ancestor $input.head HEAD } | complete)
        require ($ancestor.exit_code == 0) 'input revision is not an ancestor'
        let inventory = (^git -C $location ls-files --cached --others --exclude-standard | lines | where {|p|
            not ($p | str starts-with audit/) and not ($p | str starts-with .claude/)
        } | sort | uniq | where {|p| ($location | path join $p | path exists)})
        require ($inventory == ($input.files | get path)) $'current source inventory changed: ($input.name)'
        for file in $input.files {
            require ((open --raw ($location | path join $file.path) | hash sha256) == $file.sha256) $'source changed: ($input.name)/($file.path)'
        }
        {name: $input.name, files: ($input.files | length), recorded_head: $input.head}
    })
    let changed = ($first.inputs | each {|old|
        let current = ($final.inputs | where name == $old.name | first)
        require (($old.files | get path) == ($current.files | get path)) 'unexpected source inventory change'
        $old.files | where {|file| ($current.files | where path == $file.path | first | get sha256) != $file.sha256} | each {|file| $'($old.name)/($file.path)'}
    } | flatten)
    require ($changed == ['honeycrisp/acpu/tests/chebyshev_coefficients.rs']) 'unexpected first-to-final source bridge'
    let old_test_hash = ($first.inputs | where name == honeycrisp | first | get files | where path == 'acpu/tests/chebyshev_coefficients.rs' | first | get sha256)
    require ((open --raw ($dir | path join first/chebyshev_coefficients.rs.txt) | hash sha256) == $old_test_hash) 'initial red test body missing'
    let commands = {
        fmt: [cargo fmt --all -- --check],
        clippy-workspace: [cargo clippy --workspace --locked --offline -- -D warnings],
        clippy-acpu: [cargo clippy -p acpu --lib --tests --locked --offline -- -D warnings],
        tests-workspace: [cargo test --workspace --locked --offline],
        build-release: [cargo build --release --workspace --locked --offline],
        tests-acpu-release: [cargo test -p acpu --lib --tests --release --locked --offline],
        matmul: [cargo run -p acpu --example matmul --locked --offline],
        docs-acpu: [cargo doc -p acpu --no-deps --locked --offline]
    }
    let gates = ([{epoch: first, names: [clippy-acpu], expected: 101},
        {epoch: final, names: [fmt clippy-workspace clippy-acpu tests-workspace build-release tests-acpu-release matmul docs-acpu], expected: 0}]
        | each {|set| $set.names | each {|name|
            let base = ($dir | path join $set.epoch $name)
            let receipt = (open $'($base).json')
            let expected_hash = if $set.epoch == first { $first.source_sha256 } else { $final.source_sha256 }
            require ($receipt.source_sha256 == $expected_hash and $receipt.exit_code == $set.expected and $receipt.source_before_after == equal) $'gate outcome mismatch: ($base)'
            require ($receipt.command == ($commands | get $name)) $'gate command mismatch: ($base)'
            require ($receipt.rustup_toolchain == '1.95.0' and $receipt.rustflags == '-Dwarnings' and $receipt.rustdocflags == '-Dwarnings') $'strict toolchain environment mismatch: ($base)'
            require (($receipt.rustc | str starts-with 'rustc 1.95.0 ') and ($receipt.cargo | str starts-with 'cargo 1.95.0 ') and ($receipt.clippy | str starts-with 'clippy 0.1.95 ')) $'compiler version mismatch: ($base)'
            require (($receipt.rustc_path | str ends-with '/1.95.0-aarch64-apple-darwin/bin/rustc') and ($receipt.rustdoc_path | str ends-with '/1.95.0-aarch64-apple-darwin/bin/rustdoc')) $'compiler executable mismatch: ($base)'
            for stream in [stdout stderr] {
                require ((open --raw $'($base).($stream)' | hash sha256) == ($receipt | get $'($stream)_sha256')) $'raw stream mismatch: ($base).($stream)'
            }
            {epoch: $set.epoch, name: $name, exit_code: $receipt.exit_code}
        }} | flatten)
    let parent = (^git show '475dd6d22c4d179aeb0d3a480303d3af4ff6a42c:acpu/src/sparse/chebyshev.rs')
    let current = (open --raw acpu/src/sparse/chebyshev.rs)
    let unchanged = ([
        {name: bessel_i, start: 'pub fn bessel_i(', end: '// ── Chebyshev coefficients'}
        {name: chebyshev_matvec, start: 'pub fn chebyshev_matvec(', end: '// ── tests'}
    ] | each {|function|
        let old_body = (body $parent $function.start $function.end)
        let new_body = (body $current $function.start $function.end)
        require ($old_body == $new_body) $'existing kernel changed: ($function.name)'
        {name: $function.name, body_sha256: ($new_body | hash sha256)}
    })
    let oracle = (open --raw audit/2026-10-16-acpu-delivery/numeric-reference/resolved.stdout | from json | get rows)
    let expected = ($oracle | where {|row| $row.tau <= 1e6 and $row.coefficient != null} | select tau n coefficient)
    let retained = (open --raw acpu/tests/chebyshev_coefficients/independent.tsv)
    require ($retained == ($expected | to tsv) and ($expected | length) == 149) 'independent reference selection changed'
    {source_sha256: $final.source_sha256, inputs: $checked, gates: $gates,
        current_inventory_exact: true, gate_commands_exact: true, strict_toolchain_verified: true,
        historical_bridge: $changed, unchanged_functions: $unchanged, retained_reference_rows: ($expected | length)} | to json
}
