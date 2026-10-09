def totals [path: string] {
    let rows = (open --raw $path | lines | parse -r 'test result: ok. (?P<passed>[0-9]+) passed; (?P<failed>[0-9]+) failed; (?P<ignored>[0-9]+) ignored')
    {passed: ($rows.passed | into int | math sum), failed: ($rows.failed | into int | math sum), ignored: ($rows.ignored | into int | math sum)}
}

def source_warnings [path: string] {
    let lines = (open --raw $path | lines)
    $lines | enumerate | where {|entry| ($entry.item starts-with 'warning: ') and not ($entry.item =~ 'generated [0-9]+ warning') } | each {|entry|
        let location = ($lines | skip ($entry.index + 1) | take 3 | parse -r '^\s*-->\s+(?P<location>.+\.rs:[0-9]+:[0-9]+)' | get -o 0.location)
        {message: $entry.item, location: $location}
    } | where {|entry| $entry.location != null } | uniq | sort-by location message
}

let root = '/tmp/unimem-block-overlay.fXgCHW'
let baseline = '/tmp/honeycrisp-nebu-closure.GUEZG7'
let previous = (source_warnings ($baseline | path join 'candidate-gates/tests.stderr'))
let current = (source_warnings ($root | path join 'gates/tests.stderr'))
let verification = (open --raw ($root | path join 'gates/pipeline.stdout') | lines | parse -r '^\s*(?P<mode>reference|unimem) verified:\s+(?P<verified>true|false)$')
let benchmark_cases = (open --raw ($root | path join 'gates/unimem-bench.stderr') | lines | parse -r '^Benchmarking (?P<name>.+): Analyzing$' | get name)
let gates = (ls ($root | path join 'gates') | where {|entry| $entry.name ends-with '.exit'} | each {|entry| {gate: ($entry.name | path basename | str replace '.exit' ''), exit: (open --raw $entry.name | str trim | into int)} } | sort-by gate)
let overlay = (open ($root | path join 'sources.json')).overlay
for entry in $overlay {
    let actual = (open --raw ($root | path join 'honeycrisp' $entry.path) | hash sha256)
    if $actual != $entry.sha256 { error make {msg: $"source changed: ($entry.path)"} }
}
let comparison = (open ($root | path join 'baseline-comparison.json'))
let output = {
    classification: 'local combined candidate; not origin/release',
    gates: $gates,
    workspace_tests: (totals ($root | path join 'gates/tests.stdout')),
    baseline_tests: (totals ($baseline | path join 'candidate-gates/tests.stdout')),
    unimem_tests: (totals ($root | path join 'gates/unimem-tests.stdout')),
    pipeline: $verification,
    pipeline_verified: (($verification | length) == 2 and ($verification | all {|entry| $entry.verified == 'true'})),
    benchmark_cases: $benchmark_cases,
    benchmark_case_count: ($benchmark_cases | length),
    fmt_equal_to_baseline: $comparison.fmt_output_equal_after_root_normalization,
    clippy_equal_to_baseline: $comparison.clippy_diagnostics_equal,
    clippy_unique_diagnostics: $comparison.candidate_unique_clippy_count,
    test_source_warnings_equal: ($previous == $current),
    test_source_warning_count: ($current | length),
    baseline_source_warnings: $previous,
    candidate_source_warnings: $current,
    frozen_overlay_unchanged: true
}
$output | to json | save --force ($root | path join 'results.json')
$output | reject baseline_source_warnings candidate_source_warnings | to json | print
