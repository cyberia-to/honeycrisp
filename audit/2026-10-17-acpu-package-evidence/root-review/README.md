# Independent root review

CODE APPROVE for the ordinary package repair and diagnostic evidence.
Reviewed source `64866774f221112a6e83a55e557902a79b68722f`; compiler inputs
remain fixed. This verdict authorizes no version bump merge or publication.

The NeBu declaration keeps the existing path/package and adds its published
version so Cargo retains the examples' dependency. Root read the full 17-file
example diff: constant spellings preserve the compiled bits; loop extents,
timed operations and AMX operands remain; fixed AES scratch becomes an array.
The quantization reference preserves round/max/min order and NaN behavior.
No library kernel, package version or root lock changed. The earlier missing
dependency and both lint epochs remain actual failed evidence.

Root independently checked all eight owning receipts, raw streams and current
file bytes before the source commit (`owner-verification-root.json`). Final
`verify.nu REPO FRESH_CACHE` replay passed; [verification.json](verification.json)
binds the normalized package, published baseline, source inventory, registry
type identities and retained probe. Review requested additional replay links
for separate rustdoc roots/artifacts, consumer metadata/lock/probe and published
archive/source identities; final `replay.nu` supplies those checks.

Root separately reran the explicit API comparator on the reconstructed capsule
artifacts: [command](api-comparison.json), [raw result](api-comparison.stderr),
229 passed checks and 31 skipped. This reuses the independently bound extraction
artifacts, not a new compilation, and makes no numerical/performance claim.
Normalized package and four-test consumer results retain their recorded source
and commands. The consumer override remains a prepublication diagnostic;
unpatched registry delivery and owner release decisions remain open.
