# Replay one preserved gate with fresh private paths; return its actual exit code.
def relocate [value: string, prepared: string] {
    $value
    | str replace --all '/tmp/unimem-block-creation-validation' ($prepared | path join 'isolated')
    | str replace --all '/tmp/unimem-block-overlay.fXgCHW' ($prepared | path join 'overlay')
    | str replace --all '/Users/master/.rustup/' (($env.HOME | path join '.rustup') + '/')
}

def main [prepared: string, suite: string, receipt: string] {
    if $suite not-in ['isolated', 'overlay'] { error make {msg: 'suite must be isolated or overlay'} }
    if $suite == 'isolated' and $receipt == 'diff' { error make {msg: 'diff is original-checkout evidence, not an archived build gate'} }
    let prepared = ($prepared | path expand)
    let folder = if $suite == 'isolated' { 'logs' } else { 'gates' }
    let command = (open --raw ($env.FILE_PWD | path join $suite $folder $"($receipt).command") | from nuon)
    let directory = (relocate $command.directory $prepared)
    let program = (relocate $command.program $prepared)
    let arguments = ($command.arguments | each {|value| relocate $value $prepared })
    let target = if $suite == 'overlay' { $prepared | path join 'overlay/target' } else { $directory | path join 'target' }
    let environment = {CARGO_TARGET_DIR: $target}
    let logs = ($prepared | path join 'receipts' $suite)
    mkdir $logs
    {directory: $directory, program: $program, arguments: $arguments, environment: $environment}
    | to nuon | save --force ($logs | path join $"($receipt).command")
    cd $directory
    let result = (with-env $environment { run-external $program ...$arguments | complete })
    $result.stdout | save --force ($logs | path join $"($receipt).stdout")
    $result.stderr | save --force ($logs | path join $"($receipt).stderr")
    $result.exit_code | into string | save --force ($logs | path join $"($receipt).exit")
    print ($result.stdout + $result.stderr)
    exit $result.exit_code
}
