def main [--output: path, --git] {
    let root = $env.FILE_PWD
    let m = (open ($root | path join manifest.json))
    let archive = ($root | path join evidence.tar.gz)
    let bytes = (open --raw $archive | into binary)
    if ($bytes | bytes length) != $m.archive_bytes or ($bytes | hash sha256) != $m.archive_sha256 {
        error make {msg: 'PLAN capsule archive identity'}
    }
    let scratch = (^mktemp -d -t unimem-core-plan | str trim)
    let unpacked = (^tar -xzf $archive -C $scratch | complete)
    if $unpacked.exit_code != 0 { error make {msg: $unpacked.stderr} }
    let pack = ($scratch | path join payload)
    if (open --raw ($pack | path join verify.nu) | hash sha256) != $m.verifier_sha256 {
        error make {msg: 'inner verifier identity'}
    }
    let args = (if $git { [$pack '--git'] } else { [$pack] })
    let replay = (^nu ($pack | path join verify.nu) ...$args | complete)
    if $replay.exit_code != 0 { error make {msg: $replay.stderr} }
    let r = ($replay.stdout | from json)
    if $r.review_sha256 != $m.review_sha256 or $r.review_bytes != $m.review_bytes {
        error make {msg: 'replayed review identity'}
    }
    rm -r $scratch
    let result = ($r | insert archive_sha256 $m.archive_sha256)
    if $output != null { $result | to json | save $output }
    print ($result | to json)
}
