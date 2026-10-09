# Recompute comparison claims from durable original logs, without build targets.
def diagnostics [path: string, prefix: string, count: int] {
    let lines = (open --raw $path | lines)
    $lines | enumerate | where {|entry| ($entry.item starts-with $prefix) and not ($entry.item starts-with 'error: could not compile') and not ($entry.item =~ 'generated [0-9]+ warning') } | each {|entry|
        let location = ($lines | skip ($entry.index + 1) | take $count | parse -r '^\s*-->\s+(?P<location>.+\.rs:[0-9]+:[0-9]+)' | get -o 0.location)
        {message: $entry.item, location: $location}
    } | where {|entry| $entry.location != null } | uniq | sort-by location message
}

def totals [path: string] {
    let rows = (open --raw $path | lines | parse -r 'test result: ok. (?P<passed>[0-9]+) passed; (?P<failed>[0-9]+) failed; (?P<ignored>[0-9]+) ignored')
    {passed: ($rows.passed | into int | math sum), failed: ($rows.failed | into int | math sum), ignored: ($rows.ignored | into int | math sum)}
}

def main [] {
    let audit = $env.FILE_PWD
    let baseline = ($audit | path join 'baseline')
    let gates = ($audit | path join 'overlay/gates')
    let old_format = (open --raw ($baseline | path join 'fmt.stdout') | str replace --all '/private/tmp/' '/tmp/' | str replace --all '/tmp/honeycrisp-nebu-closure.GUEZG7' '<ROOT>')
    let new_format = (open --raw ($gates | path join 'fmt.stdout') | str replace --all '/private/tmp/' '/tmp/' | str replace --all '/tmp/unimem-block-overlay.fXgCHW' '<ROOT>')
    let old_errors = (diagnostics ($baseline | path join 'clippy.stderr') 'error: ' 6)
    let new_errors = (diagnostics ($gates | path join 'clippy.stderr') 'error: ' 6)
    let old_warnings = (diagnostics ($baseline | path join 'tests.stderr') 'warning: ' 3)
    let new_warnings = (diagnostics ($gates | path join 'tests.stderr') 'warning: ' 3)
    let verification = (open --raw ($gates | path join 'pipeline.stdout') | lines | parse -r '^\s*(?P<mode>reference|unimem) verified:\s+(?P<verified>true|false)$')
    let cases = (open --raw ($gates | path join 'unimem-bench.stderr') | lines | parse -r '^Benchmarking (?P<name>.+): Analyzing$' | get name)
    let result = {
        fmt_equal: ($old_format == $new_format),
        clippy_equal: ($old_errors == $new_errors), clippy_count: ($new_errors | length),
        warnings_equal: ($old_warnings == $new_warnings), warning_count: ($new_warnings | length),
        baseline_tests: (totals ($baseline | path join 'tests.stdout')),
        overlay_tests: (totals ($gates | path join 'tests.stdout')),
        unimem_tests: (totals ($gates | path join 'unimem-tests.stdout')),
        pipeline_verified: (($verification | length) == 2 and ($verification | all {|row| $row.verified == 'true'})),
        benchmark_cases: ($cases | length)
    }
    $result | to json | print
    if not ($result.fmt_equal and $result.clippy_equal and $result.warnings_equal and $result.pipeline_verified
        and $result.clippy_count == 16 and $result.warning_count == 25 and $result.benchmark_cases == 16
        and $result.baseline_tests == {passed: 320, failed: 0, ignored: 5}
        and $result.overlay_tests == {passed: 336, failed: 0, ignored: 5}
        and $result.unimem_tests == {passed: 52, failed: 0, ignored: 0}) { exit 1 }
}
