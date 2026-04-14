#!/bin/bash
# run.sh -- Full workflow: build, load DEXT, run client, unload DEXT
#
# This script orchestrates the complete experiment:
#   1. Build the DEXT (if sources changed)
#   2. Build the Rust client
#   3. Load the DEXT into the kernel
#   4. Run the client
#   5. Unload the DEXT
#
# Requirements:
#   - macOS 13+ (Ventura or later) for DriverKit 22+
#   - System Integrity Protection (SIP) must be configured to allow DEXTs:
#       * Either fully disabled: csrutil disable
#       * Or reduced mode: csrutil enable --without kext
#   - For development loading via kmutil, you need sudo access
#   - For system extension activation, you need a Developer ID
#
# Alternatively, this script supports a --client-only mode that skips
# DEXT loading and just runs the client (useful if the DEXT is already loaded).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEXT_BUNDLE="${SCRIPT_DIR}/build/CybMemDriver.dext"
CLIENT_BINARY="${SCRIPT_DIR}/client/target/release/cybmem-client"

CLIENT_ONLY=false
SKIP_BUILD=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --client-only)
            CLIENT_ONLY=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --client-only   Skip DEXT load/unload, just run the client"
            echo "  --skip-build    Skip building, use existing binaries"
            echo "  --help          Show this help"
            echo ""
            echo "Environment:"
            echo "  SURFACE_SIZE    IOSurface size in bytes (default: compiled-in)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "  CybMemDriver -- IOSurface PA Experiment"
echo "  PATH 2: IOSurface -> DEXT -> PA list"
echo "=========================================="
echo ""

# ---- Step 1: Build ----
if [ "${SKIP_BUILD}" = false ]; then
    echo "--- Step 1: Build DEXT ---"
    bash "${SCRIPT_DIR}/build_dext.sh"
    echo ""

    echo "--- Step 2: Build Client ---"
    bash "${SCRIPT_DIR}/build_client.sh"
    echo ""
fi

# ---- Step 2: Load DEXT ----
if [ "${CLIENT_ONLY}" = false ]; then
    echo "--- Step 3: Load DEXT ---"

    if [ ! -d "${DEXT_BUNDLE}" ]; then
        echo "[!] DEXT bundle not found at ${DEXT_BUNDLE}"
        echo "    Run build_dext.sh first."
        exit 1
    fi

    echo "[*] Loading DEXT with kmutil..."
    echo "    Bundle: ${DEXT_BUNDLE}"
    echo ""

    # kmutil load requires sudo and SIP disabled/reduced
    sudo kmutil load -p "${DEXT_BUNDLE}" 2>&1 || {
        echo ""
        echo "[!] Failed to load DEXT. Possible causes:"
        echo "    - SIP is enabled (run 'csrutil status' to check)"
        echo "    - Missing entitlements / signing"
        echo "    - DEXT binary has link errors"
        echo ""
        echo "    To disable SIP for development:"
        echo "      1. Boot to Recovery Mode (hold power on M1/M2)"
        echo "      2. Run: csrutil disable"
        echo "      3. Reboot"
        echo ""
        echo "    Alternatively, use systemextensionsctl for proper activation:"
        echo "      sudo systemextensionsctl developer on"
        echo "      (then load via System Preferences > Security)"
        echo ""
        echo "    Continuing with --client-only to see if DEXT is already loaded..."
        CLIENT_ONLY=true
    }

    if [ "${CLIENT_ONLY}" = false ]; then
        echo "[+] DEXT loaded"
        # Give IOKit a moment to process matching
        sleep 1
    fi
    echo ""
fi

# ---- Step 3: Run Client ----
echo "--- Step 4: Run Client ---"

if [ ! -f "${CLIENT_BINARY}" ]; then
    echo "[!] Client binary not found at ${CLIENT_BINARY}"
    echo "    Run build_client.sh first."
    exit 1
fi

echo "[*] Running: ${CLIENT_BINARY}"
echo ""
echo "-------- Client Output --------"
"${CLIENT_BINARY}" 2>&1
CLIENT_EXIT=$?
echo "-------------------------------"
echo ""

if [ ${CLIENT_EXIT} -eq 0 ]; then
    echo "[+] Client exited successfully"
else
    echo "[!] Client exited with code ${CLIENT_EXIT}"
fi
echo ""

# ---- Step 4: Unload DEXT ----
if [ "${CLIENT_ONLY}" = false ]; then
    echo "--- Step 5: Unload DEXT ---"
    echo "[*] Unloading DEXT..."

    sudo kmutil unload -b com.cyb.CybMemDriver 2>&1 || {
        echo "[!] Failed to unload DEXT (may already be unloaded)"
    }

    echo "[+] DEXT unloaded"
    echo ""
fi

echo "=========================================="
echo "  Experiment complete"
echo "=========================================="
