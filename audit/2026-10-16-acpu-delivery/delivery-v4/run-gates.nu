def main [repo: string] {
    let root = ($repo | path expand)
    let check = ($root | path join audit/2026-10-16-acpu-delivery/delivery-v4/check.nu)
    for gate in [
        {name: fmt, argv: [cargo fmt --all -- --check]}
        {name: clippy-workspace, argv: [cargo clippy --workspace --locked --offline -- -D warnings]}
        {name: clippy-acpu, argv: [cargo clippy -p acpu --lib --tests --locked --offline -- -D warnings]}
        {name: tests-workspace, argv: [cargo test --workspace --locked --offline]}
        {name: build-release, argv: [cargo build --release --workspace --locked --offline]}
        {name: tests-acpu-release, argv: [cargo test -p acpu --lib --tests --release --locked --offline]}
        {name: matmul, argv: [cargo run -p acpu --example matmul --locked --offline]}
        {name: docs-acpu, argv: [cargo doc -p acpu --no-deps --locked --offline]}
        {name: aruminium-matmul, argv: [cargo run -p aruminium --example aruminium_matmul --locked --offline]}
        {name: rane-matmul, argv: [cargo run -p rane --example rane_matmul --locked --offline]}
    ] {
        print $'Running ($gate.name)'
        let result = (do { ^nu $check $root $gate.name ($gate.argv | to json --raw) } | complete)
        print $result.stdout
        if $result.exit_code != 0 {
            print --stderr $result.stderr
            exit $result.exit_code
        }
    }
}
