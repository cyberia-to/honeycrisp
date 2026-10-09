def --wrapped main [root: string, name: string, directory: string, program: string, ...arguments: string] {
    let logs = ($root | path join 'gates')
    let target = ($root | path join 'target')
    {directory: $directory, environment: {CARGO_TARGET_DIR: $target}, program: $program, arguments: $arguments} | to nuon | save --force ($logs | path join $"($name).command")
    cd $directory
    let result = (with-env {CARGO_TARGET_DIR: $target} { run-external $program ...$arguments | complete })
    $result.stdout | save --force ($logs | path join $"($name).stdout")
    $result.stderr | save --force ($logs | path join $"($name).stderr")
    $result.exit_code | into string | save --force ($logs | path join $"($name).exit")
    print ($result.stdout + $result.stderr)
    exit $result.exit_code
}
