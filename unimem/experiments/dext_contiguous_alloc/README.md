# DEXT Contiguous Allocation Experiment

**PATH 1**: DEXT allocates contiguous physical pages, wraps as IOSurface for ANE/GPU visibility.

## Goal

Test whether a DriverKit extension (DEXT) can:

1. Allocate physically contiguous memory via `IOBufferMemoryDescriptor`
2. Map that memory into userspace via `IOConnectMapMemory64`
3. Create an IOSurface backed by the same physical pages
4. Have the ANE accept that IOSurface for computation

## Architecture

```
+-------------------+     IOConnectMapMemory64     +-------------------+
| CybMemAllocDriver | --------------------------> | Rust Client       |
| (DEXT, kernel)    |     ExternalMethod           | (userspace)       |
|                   | <--------------------------- |                   |
| IOBufferMemory    |                              | IOSurface wrap    |
| Descriptor with   |                              | attempt           |
| contiguous pages  |                              |                   |
+-------------------+                              | ANE visibility    |
                                                   | test              |
                                                   +-------------------+
```

## Key Technical Details

### kIOMemoryPhysicallyContiguous

This flag (`0x00000008`) is the kernel-side mechanism for requesting physically
contiguous memory allocation. However:

- **Kernel IOKit** (`IOBufferMemoryDescriptor::inTaskWithPhysicalMask`): Supports it.
- **DriverKit userspace** (`IOBufferMemoryDescriptor::Create`): Does NOT support this
  flag directly. The DriverKit API accepts `kIOMemoryDirectionInOut` and alignment hints,
  but the contiguous flag is filtered out.

**Implication**: The DEXT path relies on the kernel's allocator heuristics. For small
allocations (under ~2MB), the kernel often gives contiguous pages. For larger sizes,
we need to inspect the physical layout via `IODMACommand` segment walking.

### IOSurface Backing

There is no public userspace API to create an IOSurface backed by pre-existing
physical pages. IOSurface always allocates its own backing store. Possible approaches:

1. **DEXT creates the surface** kernel-side (via IOSurface kernel APIs)
2. **Kernel kext** with IOSurfaceRootUserClient access
3. **IOSurfaceLookupFromMachPort** if the DEXT can export a surface mach port

The client tests approach (3) and measures IOSurface characteristics for comparison.

### ANE Visibility

ANE uses IOSurface IDs to reference tensor buffers. If we can create an IOSurface
that is backed by our contiguous memory, ANE should accept it like any other surface.
The ANE access patterns are based on `~/git/rane/` (private framework bindings).

## File Structure

```
dext_contiguous_alloc/
  dext/
    CybMemAllocDriver.iig          -- DriverKit interface definition
    CybMemAllocDriver.cpp          -- DEXT implementation
    Info.plist                     -- Bundle metadata + IOKit personality
    CybMemAllocDriver.entitlements -- Required entitlements
  client/
    Cargo.toml                     -- Rust project config
    src/main.rs                    -- Userspace client (IOKit + IOSurface + benchmarks)
  build_dext.sh                    -- Build the DEXT bundle (requires DriverKit SDK)
  build_client.sh                  -- Build the Rust client
  run.sh                           -- Build all + run experiment
  README.md                        -- This file
```

## Building and Running

### Quick Start (standalone, no DEXT)

```bash
./build_client.sh
build/client/dext_contiguous_client
```

This runs in standalone mode, creating a normal IOSurface for comparison
measurements (throughput, latency, VM analysis, page dispositions).

### Full Experiment (with DEXT)

```bash
# 1. Enable system extension developer mode
sudo systemextensionsctl developer on

# 2. Build everything
./run.sh

# 3. If DEXT build succeeds, install it
sudo cp -r build/dext/CybMemAllocDriver.dext /Library/DriverExtensions/

# 4. Approve in System Settings > Privacy & Security

# 5. Re-run
./run.sh
```

### Prerequisites

- **Rust** (stable, any recent version)
- **Xcode** with DriverKit SDK (for DEXT build only)
- **macOS 14+** (Sonoma) on Apple Silicon
- System extension developer mode enabled (for DEXT installation)

## Expected Results

### Without DEXT (standalone)

The client creates a 16MB IOSurface and measures:
- Sequential read/write throughput (expect ~40-60 GB/s on M-series)
- Random access latency (expect ~5-15 ns/access)
- VM region structure (typically 1 contiguous VA region)
- Page disposition uniformity (typically uniform object IDs)
- IOSurface mach port roundtrip (should succeed)
- ANE framework availability check

### With DEXT

Additionally measures:
- DEXT buffer physical segment count (1 = contiguous, >1 = fragmented)
- IOConnectMapMemory64 mapping latency
- Throughput comparison: DEXT buffer vs IOSurface
- Whether IOSurface wrapping of DEXT memory is feasible

### Key Questions This Experiment Answers

| Question | How We Test | Expected Answer |
|----------|------------|-----------------|
| Can DEXT allocate contiguous PA? | Segment count from ExternalMethod | Likely NO for large sizes (kernel-only flag) |
| Can DEXT buffer be mapped to userspace? | IOConnectMapMemory64 | YES (via CopyClientMemoryForType) |
| Can mapped memory become IOSurface? | IOSurfaceCreate with various properties | NO (no public API for custom backing) |
| ANE accepts the surface? | rane-style evaluation | YES if IOSurface created normally |
| Throughput difference? | Benchmark comparison | Likely similar (same unified memory) |

## Next Steps

If this experiment confirms that:
- DEXT cannot provide `kIOMemoryPhysicallyContiguous` (kernel-only)
- IOSurface cannot be backed by arbitrary physical pages from userspace

Then explore:
- **PATH 2**: Kernel kext with `IOBufferMemoryDescriptor::inTaskWithPhysicalMask` +
  `kIOMemoryPhysicallyContiguous`, creating IOSurface kernel-side
- **PATH 3**: `IODMACommand` with contiguous specification in DEXT
- **PATH 4**: Hypervisor.framework for stage-2 page table control
  (see `experiments/hyp_probe/`)
