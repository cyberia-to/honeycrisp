#!/usr/bin/env bash
set -euo pipefail

#
# Build and run the DEXT contiguous allocation experiment.
#
# This script:
#   1. Attempts to build the DEXT (may fail without DriverKit SDK)
#   2. Builds the Rust client (always succeeds if Rust is installed)
#   3. Runs the client (standalone mode if DEXT is not installed)
#

cd "$(dirname "$0")"

echo "============================================================"
echo "  DEXT Contiguous Allocation Experiment"
echo "  PATH 1: DEXT alloc contiguous -> IOSurface -> ANE"
echo "============================================================"
echo ""

# ── Step 1: Try building the DEXT ──
echo "--- Step 1: Build DEXT (optional) ---"
echo ""

DEXT_OK=0
if xcrun --sdk driverkit --show-sdk-path &>/dev/null; then
    echo "  DriverKit SDK found, attempting DEXT build..."
    if bash build_dext.sh 2>&1; then
        DEXT_OK=1
        echo ""
        echo "  DEXT build succeeded."
    else
        echo ""
        echo "  DEXT build failed (see errors above)."
        echo "  Continuing with client-only (standalone) mode."
    fi
else
    echo "  DriverKit SDK not available."
    echo "  Skipping DEXT build; client will run in standalone mode."
fi
echo ""

# ── Step 2: Build the client ──
echo "--- Step 2: Build client ---"
echo ""
bash build_client.sh 2>&1
echo ""

# ── Step 3: Check if DEXT is installed ──
echo "--- Step 3: Check DEXT status ---"
echo ""

DEXT_INSTALLED=0
if ioreg -l 2>/dev/null | grep -q "CybMemAllocDriver"; then
    echo "  CybMemAllocDriver found in IORegistry!"
    DEXT_INSTALLED=1
else
    echo "  CybMemAllocDriver NOT found in IORegistry."
    if [ "$DEXT_OK" = "1" ]; then
        echo ""
        echo "  The DEXT was built but is not installed. To install:"
        echo "    sudo systemextensionsctl developer on"
        echo "    sudo cp -r build/dext/CybMemAllocDriver.dext /Library/DriverExtensions/"
        echo "    # Approve in System Settings > Privacy & Security"
        echo "    # Then re-run this script"
    fi
fi
echo ""

# ── Step 4: Run the client ──
echo "--- Step 4: Running experiment ---"
echo ""

if [ "$DEXT_INSTALLED" = "1" ]; then
    echo "  Mode: FULL (DEXT connected)"
else
    echo "  Mode: STANDALONE (no DEXT, using IOSurface for comparison)"
fi
echo ""

build/client/dext_contiguous_client

echo ""
echo "--- Experiment complete ---"
