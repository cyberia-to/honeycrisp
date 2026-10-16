# Verify historical receipts and their unchanged compiler inputs; documentation
# commits may advance HEAD, but must retain the reviewed source ancestor.
def main [repo: string] {
    let root = ($repo | path expand)
    let out = ($root | path join audit/2026-10-16-acpu-delivery)
    let manifest = (open ($out | path join sources.json))
    if ($manifest.inputs | to json --raw | hash sha256) != $manifest.source_sha256 {
        error make {msg: 'invalid source inventory hash'}
    }
    for input in $manifest.inputs {
        let current = if $input.name == honeycrisp { $root } else {
            $root | path dirname | path join strata | path expand
        }
        let ancestry = (do { ^git -C $current merge-base --is-ancestor $input.head HEAD } | complete)
        if $ancestry.exit_code != 0 { error make {msg: $'missing ancestor: ($input.name)'} }
        let files = (^git -C $current ls-files | lines | where {|p|
            not ($p | str starts-with audit/) and not ($p | str starts-with .claude/)
        } | sort | each {|p| {path: $p, sha256: (open --raw ($current | path join $p) | hash sha256)}})
        if $files != $input.files { error make {msg: $'compiler input drift: ($input.name)'} }
    }
    let expected = [
        {name: fmt, code: 1, argv: [cargo fmt --all -- --check]}
        {name: clippy, code: 101, argv: [cargo clippy --workspace --locked --offline -- -D warnings]}
        {name: release, code: 0, argv: [cargo build --release --workspace --locked --offline]}
        {name: tests, code: 101, argv: [cargo test --workspace --locked --offline]}
        {name: matmul, code: 0, argv: [cargo run -p acpu --example matmul --locked --offline]}
        {name: input-debug, code: 0, argv: [cargo run --manifest-path /tmp/honeycrisp-acpu-safe-input-probe/Cargo.toml --locked --offline]}
        {name: input-release, code: 0, argv: [cargo run --manifest-path /tmp/honeycrisp-acpu-safe-input-probe/Cargo.toml --release --locked --offline]}
    ]
    for gate in $expected {
        let receipt = (open ($out | path join $'($gate.name).json'))
        if $receipt.source_sha256 != $manifest.source_sha256 or $receipt.source_before_after != equal or $receipt.exit_code != $gate.code or $receipt.command != $gate.argv {
            error make {msg: $'receipt mismatch: ($gate.name)'}
        }
        for stream in [stdout stderr] {
            let actual = (open --raw ($out | path join $'($gate.name).($stream)') | hash sha256)
            if $actual != ($receipt | get $'($stream)_sha256') {
                error make {msg: $'stream mismatch: ($gate.name).($stream)'}
            }
        }
        for extra in ($receipt | get --optional extra_inputs | default []) {
            let saved = match ($extra.path | path basename) {
                Cargo.toml => 'probe-manifest.toml.txt'
                Cargo.lock => 'probe-lock.toml.txt'
                safe-input-probe.rs => 'safe-input-probe.rs'
                _ => { error make {msg: 'unrecognized diagnostic input'} }
            }
            if (open --raw ($out | path join $saved) | hash sha256) != $extra.sha256 {
                error make {msg: $'diagnostic mismatch: ($saved)'}
            }
        }
    }
    {verified: ($expected | length), source_sha256: $manifest.source_sha256,
        verdict: 'Historical evidence intact; admission and numerical repairs remain required.'} | to json
}
