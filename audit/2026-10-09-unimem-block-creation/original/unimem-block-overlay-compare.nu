def diagnostics [path: string] {
    let lines = (open --raw $path | lines)
    $lines | enumerate | where {|entry| ($entry.item starts-with 'error: ') and not ($entry.item starts-with 'error: could not compile') } | each {|entry|
        let locations = ($lines | skip ($entry.index + 1) | take 6 | parse -r '(?P<location>[a-zA-Z0-9_/.-]+\.rs:[0-9]+:[0-9]+)')
        {message: $entry.item, location: ($locations | get -o 0.location | default 'missing')}
    } | uniq | sort-by location message
}

let root = '/tmp/unimem-block-overlay.fXgCHW'
let baseline = '/tmp/honeycrisp-nebu-closure.GUEZG7'
let before = (diagnostics ($baseline | path join 'candidate-gates/clippy.stderr'))
let after = (diagnostics ($root | path join 'gates/clippy.stderr'))
let old_format = (open --raw ($baseline | path join 'candidate-gates/fmt.stdout') | str replace --all '/private/tmp/' '/tmp/' | str replace --all $baseline '<ROOT>')
let new_format = (open --raw ($root | path join 'gates/fmt.stdout') | str replace --all '/private/tmp/' '/tmp/' | str replace --all $root '<ROOT>')
let comparison = {
    baseline: $baseline,
    candidate: $root,
    fmt_output_equal_after_root_normalization: ($old_format == $new_format),
    baseline_unique_clippy_count: ($before | length),
    candidate_unique_clippy_count: ($after | length),
    clippy_diagnostics_equal: ($before == $after),
    baseline_clippy: $before,
    candidate_clippy: $after
}
$comparison | to json | save --force ($root | path join 'baseline-comparison.json')
$comparison | reject baseline_clippy candidate_clippy | to json | print
if not (($old_format == $new_format) and ($before == $after)) { exit 1 }
