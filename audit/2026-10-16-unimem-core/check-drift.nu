use std/assert

# Deliberately corrupt only a temporary witness checksum; admission must stop
# before compiling. The real workspace and frozen witness remain untouched.
def main [repo: string] {
    let repo = ($repo | path expand)
    let temp = (^mktemp -d -t unimem-fixture-drift | str trim)
    let result = (try {
        mkdir ($temp | path join unimem scripts) ($temp | path join unimem fixtures no-std)
        for file in [Cargo.toml Cargo.lock unimem/Cargo.toml unimem/scripts/check-no-std.nu unimem/fixtures/no-std/Cargo.toml.txt unimem/fixtures/no-std/Cargo.lock.txt unimem/fixtures/no-std/consumer.rs] {
            cp ($repo | path join $file) ($temp | path join $file)
        }
        let path = ($temp | path join unimem fixtures no-std Cargo.lock.txt)
        let original = (open --raw $path)
        let changed = ($original | str replace '0f58bbc28f91df819d0aa2a2c00cd19754769c2fad90579b3592b1c9ba7a3115' '1f58bbc28f91df819d0aa2a2c00cd19754769c2fad90579b3592b1c9ba7a3115')
        assert ($original != $changed)
        $changed | save --raw --force $path
        do { ^nu ($temp | path join unimem scripts check-no-std.nu) } | complete
    } catch {|error| {stdout: '', stderr: $error.msg, exit_code: -1} })
    rm -r $temp
    assert ($result.exit_code > 0)
    assert ($result.stderr | str contains 'no_std fixture dependency drift: crossbeam-queue')
    assert (not ($result.stderr | str contains 'Checking'))
    {status: pass, expected_rejection: $result, workspace_changed: false} | to json
}
