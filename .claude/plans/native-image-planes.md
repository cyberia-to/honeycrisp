# Native image planes through the existing Block

Owner: Codex `/root/nox_perf_finish`, delegated by `/root` on 2026-10-17.
Base: `61714b1b6337c967b8e1a1ecc4304970352f14c7`. Exact owned checkout,
cache and unchanged Strata dependency are recorded in `../task.json`.

Root approved actual multi-plane creation/retained import and real CoreVideo
witnesses, additive SDK-matching FFI and existing-spec additions. Current raw
Block planning/allocation semantics stay intact. No version/dependency/tag,
publication, acpu/Nox mutation, other storage family, fake fence or PA claim.
This is an Apple driver prerequisite; Kadek integration stays with root.

## Implementation

1. Add bounded immutable native format/layout/plan APIs for NV12 and P010,
   explicit video/full ranges. Plan checked ceiling geometry, full plane
   allocation extents, native row/offset/allocation alignment without allocation.
2. Create real plane dictionaries/array and an IOSurface, returning existing
   Block. Query actual count, format, dimensions, lane/element dimensions,
   row strides, addresses and total extent; require agreement with the plan.
3. Add unsafe retain-import of a live IOSurface into Block. Retain its owner,
   lock successfully, validate actual supported layout and preserve valid
   padding. Failure balances this constructor's ownership. No initialization,
   alias exclusion or asynchronous-completion promise follows from import.
4. Share numeric/native validation and existing RAII cleanup where practical.
   All new source files remain below 500 lines. Extend existing spec files
   before production source. Public sample views retain existing unsafe duties.

## Acceptance and review

Write focused acceptance before production: literal odd/even geometry,
arithmetic/alignment/end bounds and malformed metadata; no-allocation plan with
positive allocation control; real native create/query and initialization;
independent CoreVideo wrapping and reverse import, shared identity/sample bytes,
original-owner drop; CF creation/retain/lock/validation failure cleanup and no
double-release/unlock; private fields/unchecked constructors/unsafe imports.
Use injected failures only for failure paths and real FFI for native success.
Record framework rejection honestly; do not silently round public dimensions.

Run scoped native debug/release, strict all-target unimem Clippy, docs/doctests,
formatting and isolated no_std own-panic gate with exact tools/clean environment.
Retain baseline and final source inventories, commands, streams and results in
the audit; use temporary development logs rather than redundant audit capsules.
Real FFI is native tested; pure helpers may receive targeted Miri.
No benchmark is required for this correctness unit.

Review questions: Does creation have real native plane metadata? Do imports
retain the same allocation without claiming readiness? Are all bounds/plane
overlap checks independent of padding and pointer provenance? Does every
returned failure unwind precisely its own native references/locks? Does one
package identity and existing raw Block behavior survive? Answer from code,
independent CoreVideo tests and source-bound receipts before final review.

Send implementation and acceptance to root before the final frozen gates/commit.
Prepare an ordinary reviewable commit/PR after review; root controls merge.
