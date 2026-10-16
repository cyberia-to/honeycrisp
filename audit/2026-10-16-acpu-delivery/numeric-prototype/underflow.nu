# Positive-series zeros are numerical underflow. Bound the corresponding exact
# scaled magnitude by exp(-tau)*(tau/2)^n/n!*exp(tau^2/(4(n+1))).
# This follows termwise from (n+j)! >= n!*(n+1)^j. The log evaluation is f64.
def main [repo: path] {
    let root = ($repo | path expand)
    let input = ($root | path join audit/2026-10-16-acpu-delivery/numeric-prototype/random.stdout)
    let zeros = (open --raw $input | from json | get rows | where {|r|
        $r.status == ok and $r.reference_method? == series and (($r.reference? | into string) in ['0' '0.0' '-0' '-0.0'])
    })
    let bounds = ($zeros | each {|r|
        if $r.order == 0 { error make {msg: 'unexpected zero zeroth coefficient'} }
        let factorial = (1..$r.order | each {|j| $j | math ln} | math sum)
        let bound = (2 | math ln) + $r.order * ($r.tau / 2 | math ln) - $factorial - $r.tau + $r.tau * $r.tau / (4 * ($r.order + 1))
        if $bound >= (-150 * (2 | math ln)) { error make {msg: 'underflow lacks a sufficient magnitude margin'} }
        $bound
    })
    {command: [nu audit/2026-10-16-acpu-delivery/numeric-prototype/underflow.nu $root],
        input_sha256: (open --raw $input | hash sha256), numerical_zero_rows: ($zeros | length),
        largest_log_upper: ($bounds | math max), half_subnormal_log: (-150 * (2 | math ln)),
        floating_log_evaluation: true, certified_interval: false} | to json
}
