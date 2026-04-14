#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== Building hyp_probe ==="
cargo build 2>&1

echo ""
echo "=== Signing with hypervisor entitlement ==="
codesign --sign - --entitlements hyp.plist --force target/debug/hyp_probe 2>&1

echo ""
echo "=== Running hyp_probe ==="
target/debug/hyp_probe
