# DEXT IOSurface PA Resolution Experiment

## What This Tests

**PATH 2: IOSurface alloc -> DEXT reads physical addresses**

This experiment validates whether a DriverKit extension (DEXT) can resolve
the physical addresses backing an IOSurface allocation. The goal is to build
a userspace-accessible physical memory map for IOSurface buffers -- the same
buffers used by the Apple Neural Engine and GPU.

## Architecture

```
 Userspace (client)              DEXT (DriverKit userspace)
 ==================              =========================
 1. IOSurfaceCreate()
 2. Write test pattern
 3. IOServiceOpen() ----------> CybMemDriver::Start()
 4. IOConnectCallStructMethod   CybMemDriver::ExternalMethod()
    {surface_id, length} ------>   |
                                   v
                                 IOBufferMemoryDescriptor::Create()
                                 IODMACommand::Create()
                                 IODMACommand::PrepareForDMA()
                                   |
    <----- {PA segments} --------  |
 5. Print PA map
 6. Verify data integrity
 7. IOServiceClose() ---------> CybMemDriver::Stop()
```

## Key DriverKit APIs

| API | Purpose |
|-----|---------|
| `IOUserClient::ExternalMethod` | Dispatch userspace calls from `IOConnectCallStructMethod` |
| `IODMACommand::Create` | Create DMA command object with device context |
| `IODMACommand::PrepareForDMA` | Resolve IOMemoryDescriptor to physical address segments |
| `IOBufferMemoryDescriptor::Create` | Allocate kernel-accessible memory buffer |
| `IOUserClient::CreateMemoryDescriptorFromClient` | Wrap client VA range as IOMemoryDescriptor |

## Physical Address Access in DriverKit

DriverKit DEXTs run in userspace but have privileged IOKit access. Unlike
kernel extensions (KEXTs), DEXTs do not have `getPhysicalSegment()`. Instead:

1. **IODMACommand::PrepareForDMA()** -- resolves an IOMemoryDescriptor into
   physical address segments (up to 32 per call). On Apple Silicon with
   DART (Device Address Resolution Table), these are IOVA addresses that
   map to physical RAM through the IOMMU translation tables.

2. **IOUserClient::CreateMemoryDescriptorFromClient()** -- wraps a virtual
   address range from the client process into an IOMemoryDescriptor that
   can be used with IODMACommand.

## File Structure

```
dext_iosurface_pa/
  dext/
    CybMemDriver.iig           # DriverKit interface definition
    CybMemDriver.cpp           # Implementation (ExternalMethod, DMA, PA resolution)
    Info.plist                  # IOKit matching personalities
    CybMemDriver.entitlements  # DriverKit entitlements for codesigning
  client/
    Cargo.toml                 # Rust project
    src/main.rs                # IOSurface creation, DEXT connection, PA readout
  build_dext.sh                # Compile DEXT (requires Xcode with iig tool)
  build_client.sh              # Cargo build + codesign
  run.sh                       # Full workflow: build, load, run, unload
  README.md                    # This file
```

## Prerequisites

- macOS 13+ (Ventura) or later
- Xcode (full install, not just Command Line Tools) for the `iig` tool
- SIP disabled or in reduced mode for development DEXT loading:
  ```
  # Boot to Recovery Mode, then:
  csrutil disable
  # Or reduced mode:
  csrutil enable --without kext
  ```
- Rust toolchain (for the client)

## Building

```bash
# Build everything
./build_dext.sh    # Compile DEXT (needs Xcode)
./build_client.sh  # Build Rust client

# Or build and run the full workflow
./run.sh
```

## Running

```bash
# Full workflow (builds, loads DEXT, runs client, unloads DEXT)
./run.sh

# Client only (if DEXT is already loaded)
./run.sh --client-only

# Skip build step
./run.sh --skip-build
```

## Expected Results

### Success Case
```
=== Physical Address Map ===
Total length:   65536 bytes
Segment count:  4

Seg#     Phys Addr       Length    Pages
--------------------------------------------------
0        0x0000800100000  16384 B      1
1        0x0000800104000  16384 B      1
2        0x0000800108000  16384 B      1
3        0x000080010c000  16384 B      1
--------------------------------------------------
Total mapped: 65536 bytes (4 pages)
[+] Full surface coverage verified
[+] Data integrity OK -- test pattern intact
```

### Expected Failure Modes

1. **DEXT not found**: The IOServiceGetMatchingService call fails because
   the DEXT is not loaded. Solve by loading via `kmutil load` or
   `systemextensionsctl`.

2. **IODMACommand returns IOVA not PA**: On Apple Silicon, DART translates
   device addresses. The returned "physical" addresses are actually IOVAs.
   This is correct behavior -- the DART maps IOVAs to true physical addresses.

3. **Permission denied**: Without proper entitlements or SIP configuration,
   the DEXT cannot load or the client cannot connect.

4. **PrepareForDMA fails**: The IODMACommand may fail if the memory
   descriptor is not properly prepared or the device context is wrong.

## What This Proves

If successful, this experiment demonstrates that:

1. A DEXT can receive IOSurface metadata from userspace
2. IODMACommand can resolve memory descriptors to physical address segments
3. The physical address map can be passed back to userspace
4. IOSurface data remains intact throughout the process

This establishes PATH 2 as viable for the unimem project's goal of
physical memory address resolution for IOSurface-backed buffers.

## Limitations

- DriverKit DEXTs see IOVA (DART-translated addresses) on Apple Silicon,
  not raw physical addresses. An additional DART table walk would be needed
  to get true physical addresses.
- The current implementation creates a test IOBufferMemoryDescriptor rather
  than resolving the client's actual IOSurface memory. A production version
  would use `CreateMemoryDescriptorFromClient()` with the IOSurface's VA.
- DEXT loading requires SIP disabled or proper Apple Developer provisioning.
- Maximum 32 physical segments per IODMACommand call (sufficient for most
  IOSurface sizes).
