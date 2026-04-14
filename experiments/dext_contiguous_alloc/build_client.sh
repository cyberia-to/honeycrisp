#!/usr/bin/env bash
set -euo pipefail

#
# Build the Rust userspace client for the DEXT contiguous allocation experiment.
#
# The client can run in two modes:
#   1. With DEXT installed: full experiment (map DEXT buffer, analyze, wrap)
#   2. Standalone: uses IOSurface allocation for comparison measurements
#

cd "$(dirname "$0")"

BUILD_DIR="build/client"

echo "============================================================"
echo "  Building DEXT Contiguous Allocation Client"
echo "============================================================"

# ── Build with cargo ──
echo ""
echo "--- Building Rust client ---"
cd client
cargo build --release 2>&1
cd ..

# ── Copy binary ──
mkdir -p "${BUILD_DIR}"
cp client/target/release/dext_contiguous_client "${BUILD_DIR}/"

echo ""
echo "--- Build complete ---"
echo "  Binary: ${BUILD_DIR}/dext_contiguous_client"
echo ""
echo "  To run (no special entitlements needed for standalone mode):"
echo "    ${BUILD_DIR}/dext_contiguous_client"
echo ""
echo "  For DEXT mode, first install the DEXT (see build_dext.sh)."
echo ""
