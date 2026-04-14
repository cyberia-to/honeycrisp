#!/usr/bin/env bash
set -euo pipefail

#
# Build the CybMemAllocDriver DriverKit extension.
#
# Prerequisites:
#   - Xcode with DriverKit SDK installed
#   - Apple Developer account with DriverKit entitlement (for signing)
#   - macOS 14+ (Sonoma) for current DriverKit APIs
#
# For local development/testing:
#   sudo systemextensionsctl developer on
#   csrutil status  # SIP must allow system extensions
#

cd "$(dirname "$0")"

DEXT_DIR="dext"
BUILD_DIR="build/dext"
BUNDLE_NAME="CybMemAllocDriver.dext"
BUNDLE_DIR="${BUILD_DIR}/${BUNDLE_NAME}"

echo "============================================================"
echo "  Building CybMemAllocDriver DEXT"
echo "============================================================"

# ── Locate DriverKit SDK ──
DRIVERKIT_SDK=$(xcrun --sdk driverkit --show-sdk-path 2>/dev/null || true)
if [ -z "$DRIVERKIT_SDK" ]; then
    echo ""
    echo "ERROR: DriverKit SDK not found."
    echo "Install via: xcode-select --install"
    echo "Or ensure Xcode.app has DriverKit support."
    echo ""
    echo "Alternatively, check available SDKs:"
    echo "  xcodebuild -showsdks | grep driver"
    echo ""
    echo "NOTE: DriverKit compilation requires the DriverKit SDK which"
    echo "includes the DriverKit headers and .iig compiler (iig tool)."
    echo "Without it, this DEXT cannot be compiled from command line."
    echo ""
    echo "For a quick test without DEXT, run the client in standalone mode:"
    echo "  ./build_client.sh && ./build/client/dext_contiguous_client"
    exit 1
fi

echo "  DriverKit SDK: ${DRIVERKIT_SDK}"

# ── Locate iig compiler ──
IIG=$(xcrun --sdk driverkit --find iig 2>/dev/null || true)
if [ -z "$IIG" ]; then
    echo ""
    echo "ERROR: iig compiler not found in DriverKit SDK."
    echo "The .iig interface compiler is required to process CybMemAllocDriver.iig"
    echo ""
    echo "This tool ships with Xcode's DriverKit support."
    echo "Ensure you have a full Xcode installation (not just Command Line Tools)."
    exit 1
fi

echo "  iig compiler:  ${IIG}"

# ── Clean and prepare ──
rm -rf "${BUILD_DIR}"
mkdir -p "${BUNDLE_DIR}/Contents/MacOS"
mkdir -p "${BUILD_DIR}/gen"

# ── Run iig to generate C++ headers ──
echo ""
echo "--- Generating headers from .iig ---"
"${IIG}" \
    --sdk "${DRIVERKIT_SDK}" \
    --target arm64e-apple-macos \
    "${DEXT_DIR}/CybMemAllocDriver.iig" \
    --output-dir "${BUILD_DIR}/gen" \
    2>&1 || {
    echo ""
    echo "WARNING: iig compilation failed."
    echo "The .iig file may need adjustment for your SDK version."
    echo "Check ${DEXT_DIR}/CybMemAllocDriver.iig for syntax issues."
    echo ""
    echo "Common issues:"
    echo "  - Missing DriverKit headers"
    echo "  - API differences between SDK versions"
    echo "  - arm64e vs arm64 target mismatch"
    exit 1
}

echo "  Generated headers in ${BUILD_DIR}/gen/"
ls -la "${BUILD_DIR}/gen/"

# ── Compile C++ source ──
echo ""
echo "--- Compiling CybMemAllocDriver.cpp ---"

DRIVERKIT_CLANG=$(xcrun --sdk driverkit --find clang++ 2>/dev/null || xcrun --find clang++)
DRIVERKIT_SYSROOT="${DRIVERKIT_SDK}"

"${DRIVERKIT_CLANG}" \
    -isysroot "${DRIVERKIT_SYSROOT}" \
    -I "${BUILD_DIR}/gen" \
    -I "${DEXT_DIR}" \
    -target arm64e-apple-macos \
    -fmodules \
    -fcxx-modules \
    -std=gnu++20 \
    -fno-rtti \
    -fno-exceptions \
    -D__DRIVERKIT__=1 \
    -D__DriverKit__=1 \
    -c "${DEXT_DIR}/CybMemAllocDriver.cpp" \
    -o "${BUILD_DIR}/CybMemAllocDriver.o" \
    2>&1

echo "  Compiled successfully"

# ── Link ──
echo ""
echo "--- Linking ---"

"${DRIVERKIT_CLANG}" \
    -isysroot "${DRIVERKIT_SYSROOT}" \
    -target arm64e-apple-macos \
    -Xlinker -kext \
    -nostdlib \
    -lDriverKit \
    "${BUILD_DIR}/CybMemAllocDriver.o" \
    -o "${BUNDLE_DIR}/Contents/MacOS/CybMemAllocDriver" \
    2>&1

echo "  Linked successfully"

# ── Assemble bundle ──
echo ""
echo "--- Assembling DEXT bundle ---"
cp "${DEXT_DIR}/Info.plist" "${BUNDLE_DIR}/Contents/"

echo "  Bundle: ${BUNDLE_DIR}"
ls -la "${BUNDLE_DIR}/Contents/"
ls -la "${BUNDLE_DIR}/Contents/MacOS/"

# ── Code sign ──
echo ""
echo "--- Code signing ---"
echo "  NOTE: For local development, we sign ad-hoc."
echo "  For deployment, sign with a Developer ID + DriverKit entitlement."

codesign --sign - \
    --entitlements "${DEXT_DIR}/CybMemAllocDriver.entitlements" \
    --force \
    "${BUNDLE_DIR}" \
    2>&1 || {
    echo ""
    echo "WARNING: Code signing failed."
    echo "The DEXT bundle was assembled but is not signed."
    echo "You may need to sign with a proper Developer ID."
}

echo ""
echo "============================================================"
echo "  Build complete: ${BUNDLE_DIR}"
echo "============================================================"
echo ""
echo "  To install (requires developer mode + SIP permitting):"
echo "    sudo systemextensionsctl developer on"
echo "    sudo cp -r ${BUNDLE_DIR} /Library/DriverExtensions/"
echo "    # Then approve in System Settings > Privacy & Security"
echo ""
echo "  To check status:"
echo "    systemextensionsctl list"
echo "    ioreg -l | grep CybMemAlloc"
echo ""
