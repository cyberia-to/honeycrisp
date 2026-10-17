use ../common.nu *
use ../replay.nu *

def main [archive: string, output: string, extraction: string] {
    let out = ($output | path expand)
    fresh $out
    let requested = ($extraction | path expand)
    fresh $requested
    let canonical = ($requested | path expand)
    require ($requested != $canonical and ($requested | str starts-with '/tmp/')
        and ($canonical | str starts-with '/private/tmp/')) 'actual macOS tmp alias witness absent'
    require ((digest $archive) == '74967ea98831cd29d69fd3cef0cc08a8236f66d8d2fdcb2e99223428488d1159') 'candidate archive differs'
    let current = (extract $archive ($canonical | path join current) acpu-0.2.1)
    capture $out metadata [cargo metadata --no-deps --format-version '1' --locked --offline] $current (environment ($canonical | path join target)) | ignore
    let targets = (normalized-targets (open --raw ($out | path join metadata.stdout) | from json) $current)
    require (($targets | length) == 40) 'target count differs'
    {requested_before_creation: $requested, canonical_after_creation: $canonical,
        manifest: ($current | path join Cargo.toml), targets: ($targets | length),
        historical_receipts_changed: false, compiler_gates_executed: false} | to json
}
