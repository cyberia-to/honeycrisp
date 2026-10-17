def main [repo: path] {
    let out = ($repo | path expand | path join audit/2026-10-17-native-image-planes)
    if ($out | path join inventory.json | path exists) { error make {msg: 'seal exists; preserve original evidence'} }
    let files = (glob ($out | path join '**/*') | where {|p| ($p | path type) == file} | sort)
    $files | each {|p| {path: ($p | path relative-to $out), sha256: (open --raw $p | into binary | hash sha256)}}
        | to json --raw | save ($out | path join inventory.json)
    print {files: ($files | length), sha256: (open --raw ($out | path join inventory.json) | hash sha256)}
}
