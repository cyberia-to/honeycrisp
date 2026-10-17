use run.nu [sources identities]
def sha [p: path] { open --raw $p | into binary | hash sha256 }
def main [repo: path] {
    let root = ($repo | path expand)
    let out = ($root | path join audit/2026-10-17-native-image-planes)
    if ($out | path join freeze.json | path exists) { error make {msg: 'freeze exists'} }
    let inventory = (sources $root)
    let digest = ($inventory | to json --raw | hash sha256)
    let manifest = $'inputs/($digest).json'
    if not ($out | path join $manifest | path exists) { $inventory | to json --raw | save ($out | path join $manifest) }
    let sdk = (^xcrun --show-sdk-path | str trim)
    let headers = [System/Library/Frameworks/IOSurface.framework/Headers/IOSurfaceRef.h
        System/Library/Frameworks/CoreVideo.framework/Headers/CVPixelBuffer.h
        System/Library/Frameworks/CoreVideo.framework/Headers/CVPixelBufferIOSurface.h]
    let gates = [
        {name: final-debug, argv: [cargo test -p unimem --locked --offline]}
        {name: final-release, argv: [cargo test -p unimem --release --locked --offline]}
        {name: final-clippy, argv: [cargo clippy -p unimem --all-targets --locked --offline -- -D warnings]}
        {name: final-doc, argv: [cargo doc -p unimem --no-deps --locked --offline]}
        {name: final-fmt, argv: [cargo fmt --all -- --check]}
        {name: final-no-std, argv: [nu unimem/scripts/check-no-std.nu]}
        {name: workspace-test, argv: [cargo test --workspace --locked --offline]}
        {name: workspace-release, argv: [cargo build --workspace --release --locked --offline]}
        {name: workspace-clippy, argv: [cargo clippy --workspace --locked --offline -- -D warnings]}
        {name: pipeline, argv: [cargo run -p unimem --example pipeline --release --locked --offline]}
    ]
    {created: (date now | format date '%+'),
        heads: ([$root ($root | path join ../strata | path expand)] | each {|p| ^git -C $p rev-parse HEAD | str trim}),
        source_manifest: $manifest, source_sha256: $digest,
        runner_sha256: (sha ($out | path join run.nu)), tools: (identities),
        machine: {os: (^sw_vers), architecture: (^uname -m | str trim)},
        sdk: {path: $sdk, version: (^xcrun --show-sdk-version | str trim),
            headers: ($headers | each {|p| {path: $p, sha256: (sha ($sdk | path join $p))}})},
        authority_files: ([.claude/plans/native-image-planes.md unimem/specs/README.md unimem/specs/api-sketch.md]
            | each {|p| {path: $p, sha256: (sha ($root | path join $p))}}),
        gates: $gates,
        scope: 'Local workspace/sibling Rust and manifests/config plus no_std templates/script; dependency identities are locked, registry sources/tool binaries are not bundled'
    } | to json | save ($out | path join freeze.json)
    print {source_sha256: $digest, freeze_sha256: (sha ($out | path join freeze.json)), gates: ($gates | length)}
}
