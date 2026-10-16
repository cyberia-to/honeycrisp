# Final hygiene and tooling gates

Base `c5177c56e4551c9896a2869c3b23d0c7bac9a6f9`; compiler closure
`45468f4172e1592f6c9fec7ce56a1af5553aa7ec93b714836b18775138d08352`.
[Source inventory](sources.json), [actual commands](run-gates.nu),
[historical epoch/repair bridge](../delivery-v3/README.md).

All required workspace gates pass on this exact source. Warnings are denied;
Rust, rustdoc and rustup subcommands are pinned to 1.95.0. Each receipt retains
complete output and verifies the owner/Strata inputs before and after execution.

| Gate / receipt | Actual result |
|---|---|
| [Fmt](fmt.json), [workspace Clippy](clippy-workspace.json), [strict acpu lib/tests Clippy](clippy-acpu.json) | Pass, zero warnings. |
| [Workspace tests](tests-workspace.json) | 356 passed, 5 existing ignored, including doctests. |
| [Release workspace build](build-release.json) | Pass. |
| [Release acpu tests](tests-acpu-release.json) | 187 passed, 1 existing ignored. |
| [Strict acpu docs](docs-acpu.json) | Pass. |
| [Acpu matmul](matmul.json) | Actual AMX 64×64 identity control, maximum error zero. |
| [Aruminium matmul](aruminium-matmul.json) | Actual GPU 64×64×64 control, maximum error zero. |
| [Rane matmul](rane-matmul.json) | Actual ANE execution, all 4096 outputs equal one. |
| [All-target compilation](check-all-targets.json) | Pass after removing accidental parent-package bench discovery; explicit benchmark-package binaries remain. |

The last bridge from `0453b62f…` changes documentation only: formula/indexing
code spans, a text code fence and truthful `Matrix::new` behavior. It executes
AMX_SET then returns `Ok`; unsupported hardware may trap rather than return the
previously documented error. The earlier strict rustdoc failure remains in
`../delivery-v3/docs-acpu.*` alongside its passing required runtime gates.

`nu audit/2026-10-16-acpu-delivery/verify-delivery.nu <repository>` checks historical raw streams/manifests
and final source identity without rerunning gates. No measured performance,
numeric-overflow repair, version bump or publication is claimed. The remaining
Chebyshev coefficient contract/algorithm is a separate numerical unit.
