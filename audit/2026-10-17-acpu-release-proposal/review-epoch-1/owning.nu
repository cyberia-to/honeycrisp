use common.nu *

# Owning gates precede the source commit; every candidate byte is retained.
def main [repo: string, output: string, target: string, tools: string] {
    let root = ($repo | path expand)
    let out = ($output | path expand)
    fresh $out
    let tooling = (tool-bundle ($tools | path expand))
    let inputs = (snapshot $root)
    let envs = (environment $target)
    {inputs: $inputs, tooling: $tooling, environment: $envs} | to json | save ($out | path join sources.json)
    for gate in (owning-gates) {
        capture $out $gate.name $gate.argv $root $envs | ignore
        require ((snapshot $root) == $inputs) 'source changed during owning gate'
        require ((tool-bundle ($tools | path expand)) == $tooling) 'tooling changed during owning gate'
    }
    {gates: 8, source_sha256: ($inputs | to json -r | hash sha256)} | to json | save ($out | path join result.json)
}
