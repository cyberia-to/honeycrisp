# Run from any directory: nu reproduce.nu /path/to/honeycrisp /path/to/strata
def capture [cwd: path, logs: path, name: string, args: list<string>, target: path] {
  mkdir $logs
  cd $cwd
  let result = (with-env {CARGO_TARGET_DIR: $target} { ^cargo ...$args | complete })
  {cwd: $cwd, command: ([cargo] | append $args), target: $target,
    exit: $result.exit_code, stdout: $result.stdout, stderr: $result.stderr}
    | to json | save ($logs | path join $'($name).json')
  print $'($name): ($result.exit_code)'
}

def archive [repo: path, revision: string, dest: path] {
  mkdir $dest
  let tar = ($dest | path dirname | path join $'(($dest | path basename)).tar')
  ^git -C $repo archive --format=tar --output $tar $revision
  if $env.LAST_EXIT_CODE != 0 { error make {msg: 'git archive failed'} }
  ^tar -xf $tar -C $dest
  if $env.LAST_EXIT_CODE != 0 { error make {msg: 'tar extraction failed'} }
}

def main [honeycrisp_repo: path, strata_repo: path] {
  let root = (^mktemp -d /tmp/nebu-identity-replay.XXXXXX | str trim)
  print $root
  for tree in [
    {name: candidate, rev: '76b2f26432cca15607ab1dd18e488df68db51ca0'}
    {name: parent, rev: '9593c7218e14e2b5b816d9008b45ab60b9580cf3'}
    {name: control, rev: '9593c7218e14e2b5b816d9008b45ab60b9580cf3'}
    {name: tag, rev: '96fa4f4509d7a966cb22a4bb9e12ac753cff43ae'}
  ] {
    archive $honeycrisp_repo $tree.rev ($root | path join $tree.name honeycrisp)
    archive $strata_repo '56aedb2d12b3126c601eb333419136d403614dbb' ($root | path join $tree.name strata)
  }
  let parent = ($root | path join parent honeycrisp)
  capture $parent ($root | path join parent-logs) metadata [metadata --format-version '1' --locked --offline] ($root | path join parent-target)
  let control = ($root | path join control honeycrisp)
  let manifest = ($control | path join acpu Cargo.toml)
  let text = (open --raw $manifest)
  if not ($text | str contains 'nebu.workspace = true') { error make {msg: 'missing control seam'} }
  $text | str replace 'nebu.workspace = true' '# Diagnostic control: omitted nebu dev-dependency.' | save -f $manifest
  capture $control ($root | path join control-logs) resolve [metadata --format-version '1' --offline] ($root | path join control-target)
  for name in [candidate control] {
    let cwd = ($root | path join $name honeycrisp)
    let logs = ($root | path join $'($name)-logs')
    let target = ($root | path join $'($name)-target')
    for gate in [
      {name: metadata, args: [metadata --format-version '1' --locked --offline]}
      {name: features, args: [tree -e features -p acpu --locked --offline]}
      {name: fmt, args: [fmt --all -- --check]}
      {name: clippy, args: [clippy --workspace --all-targets --locked --offline -- -D warnings]}
      {name: release, args: [build --release --workspace --locked --offline]}
      {name: tests, args: [test --workspace --locked --offline]}
      {name: examples, args: [build -p acpu --release --example bench_zk --example bench_summary --locked --offline]}
      {name: matmul, args: [run -p acpu --example matmul --locked --offline]}
    ] { capture $cwd $logs $gate.name $gate.args $target }
  }
  for name in [parent tag] {
    capture ($root | path join candidate honeycrisp) ($root | path join candidate-logs) $'semver-($name)' [
      semver-checks --workspace --exclude metal-benches --exclude rane-benches
      --baseline-root ($root | path join $name honeycrisp) --verbose
    ] ($root | path join semver-target)
  }
  print 'Read every recorded exit. Known baseline/control failures remain failures.'
}
