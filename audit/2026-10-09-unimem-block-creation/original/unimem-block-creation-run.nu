def --wrapped main [name: string, directory: string, program: string, ...arguments: string] {
    let logs = '/tmp/unimem-block-creation-validation/logs'
    mkdir $logs
    {directory: $directory, program: $program, arguments: $arguments} | to nuon | save --force ($logs | path join $"($name).command")
    cd $directory
    let result = (run-external $program ...$arguments | complete)
    ($result.stdout + $result.stderr) | save --force ($logs | path join $"($name).log")
    $result.exit_code | into string | save --force ($logs | path join $"($name).exit")
    print ($result.stdout + $result.stderr)
    exit $result.exit_code
}
