# Independent mechanical bridge review

APPROVE the proposed bridge, `/root/xtask_code_review` (Codex), 2026-10-16.
Reviewed `../hygiene-proposal.patch`, SHA256
`e641254187f42f1da7940f4fc0e775f4005df64e0875ae3f65d3d2f38ae75163`,
against admission source closure
`47bffad98aa5a820aab3bf02d2d19d375bcc0ba76e07d24268b35d39b9426250`.
This approves the exact proposed delta; fresh final gates/source identity remain
necessary after applying it.

The sparse prefetch iterator visits exactly the original first min(len, PF)
columns in the same order. The Chebyshev iterator visits indices 2..=order:
successful coefficient construction yields precisely order+1 values. The named
scaled temporary keeps the original multiply operand/evaluation order, including
exceptional floating-point operands. Kernel calculations and numerical bounds
remain unchanged. The duplicate GEMM attribute and unused example local have
no execution effect. Remaining changes are test-only array/iterator substitutions
or rustfmt layout; they preserve original values and assertion domains.

Independent standalone literal check, run from the task checkout with:

```sh
/Users/master/.rustup/toolchains/1.95.0-aarch64-apple-darwin/bin/rustc audit/2026-10-16-acpu-delivery/admission/independent-literals.rs -o /tmp/acpu-admission-independent-literals
/tmp/acpu-admission-independent-literals
```

Actual output retained in `independent-literals.stdout`: original decimal
3.14f32 and the proposed from_bits value both encode `4048f5c3`; old/new Bessel
oracle spellings both encode `3fa20e73`. No tolerance or oracle value changes.
No Cargo test rerun, production edit or unsafe malformed-parent execution was
performed by this reviewer. Chebyshev numerical overflow remains separately open.
