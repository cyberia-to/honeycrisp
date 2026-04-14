#!/bin/bash
# build_client.sh -- Build and codesign the Rust userspace client
#
# The client links against IOKit.framework, CoreFoundation.framework,
# and IOSurface.framework via Rust's #[link] attributes.
#
# Codesigning is required for IOKit user client access on recent macOS.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLIENT_DIR="${SCRIPT_DIR}/client"

echo "=== Building CybMem Client ==="
echo ""

# ---- Step 1: Build with Cargo ----
echo "[*] Building with cargo (release)..."
cd "${CLIENT_DIR}"
cargo build --release 2>&1

BINARY="${CLIENT_DIR}/target/release/cybmem-client"
if [ ! -f "${BINARY}" ]; then
    echo "[!] Build failed: binary not found at ${BINARY}"
    exit 1
fi

echo "[+] Built: ${BINARY}"
echo ""

# ---- Step 2: Codesign ----
# The client needs to be signed to open IOKit connections.
# Ad-hoc signing works for local development.
echo "[*] Code signing (ad-hoc)..."
codesign --sign - --force "${BINARY}" 2>&1 || {
    echo "[!] Code signing failed. The client may not be able to open DEXT connections."
}
echo "[+] Signed: ${BINARY}"
echo ""

# ---- Step 3: Verify ----
echo "[*] Verifying binary..."
file "${BINARY}"
codesign -vv "${BINARY}" 2>&1 || true
echo ""

echo "=== Client build complete ==="
echo "Binary: ${BINARY}"
echo ""
echo "Usage:"
echo "  ${BINARY}"
echo ""
echo "NOTE: The CybMemDriver DEXT must be loaded first."
echo "      See run.sh for the full workflow."
