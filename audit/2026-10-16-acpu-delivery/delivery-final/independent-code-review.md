# Independent delivery CODE supplement

APPROVE — `/root/xtask_code_review` (Codex), 2026-10-16. This supplements the
retained admission and proposed-hygiene reviews. Exact final source closure:
`5faaae626d09eab9d2b4c989a77defd27889ba6cc5a589d06cc296d07c42cddd`.
Base `c5177c56e4551c9896a2869c3b23d0c7bac9a6f9` commits the initial admission
unit and its actual red Clippy evidence. [Complete delivery delta](independent-delivery.patch),
including the new renamed example bodies, has SHA256
`6292e4a032e068048fcd1544adc5bad18bc61fe7d73f11188c5310c838341066`.
[sources.json](sources.json) supplies every current owner/sibling source hash.

The applied mechanical delta matches the earlier approved proposal. Its extra
test changes only name the existing CSR tuple type, retain the bit-identical
3.14f32 input, substitute fixed arrays and remove bitwise OR with zero in two
AMX test operands. Actual opcodes, operand bits, numerical controls and tolerance
remain unchanged. Removing the unused bench_zk mutability has no execution effect.

The separate, root-authorized benchmark ABI correction changes only the stride
parameters in the eleven shared vDSP declarations from i32 to i64 and removes
two duplicate numeric-benchmark declarations. I independently read the installed
SDK's typedef and every affected declaration: macOS LP64 uses signed long for
vDSP_Stride and unsigned long for vDSP_Length. Their parameter positions/types
match the final Rust declarations. CBLAS and vForce signatures remain unchanged.
SDK version/path/header identity are in `../vdsp-abi/sdk.json`, header SHA256
`8cdbe0b8d48d3a3e7d9790df5e75f4d5b8f2f1e7cef55c9968c801624985c6ba`.

[Independent ABI verification](independent-abi-verification.json) checks the
complete source inventory, SDK identity, four actual build/run receipts and raw
streams, plus proof source/binary hashes. The SDK C control proves signed
64-bit stride/length sizes and reverse addition. The Rust control imports the
actual shared declarations and exercises forward addition, contained negative
stride addition/multiplication/reduction. Every pointer traversal stays within
its array; the malformed historical declarations were not executed. These are
audit probes, with no new production C-toolchain dependency. The two failed
standalone Rust link attempts remain in their original receipts.

The root-authorized example-name repair preserves all forty Cargo example
targets. I independently ran read-only Cargo metadata and normalized only the
two approved names; [the recorded result](independent-examples.json) matches the
original target inventory and has unique output names. Aruminium's renamed body
is byte-identical; Rane's only body change updates its Run-command comment.
Manifest and active documentation references agree. The source runner correctly
includes existing tracked/untracked paths after these renames.

No remaining source blocker was found. Final workspace/build/release/docs gates
were still collecting when this supplement was written; their actual status
must be read from the owning receipts. Earlier red epochs remain red. This
approval makes no claim that Chebyshev numerical overflow is fixed or that acpu
has been published; numerical repair and owner-only publication remain separate.
