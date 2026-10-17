def main [repo: path, cache: path] {
    let root = ($repo | path expand)
    let out = ($root | path join audit/2026-10-17-native-image-planes)
    let freeze = ($out | path join freeze.json)
    let fixed = (open $freeze)
    for gate in $fixed.gates {
        let args = [($out | path join run.nu) $root $cache $gate.name
            ($gate.argv | to json --raw) --freeze $freeze]
        let result = (do { ^nu ...$args } | complete)
        print --no-newline $result.stdout
        if $result.exit_code != 0 { print --stderr $result.stderr }
    }
}
