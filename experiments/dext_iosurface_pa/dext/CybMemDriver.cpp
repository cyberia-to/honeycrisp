/*
 * CybMemDriver.cpp -- DriverKit extension: resolve IOSurface VA to PA
 *
 * Flow:
 *   1. Client creates IOSurface, locks it, gets base VA
 *   2. Client sends (VA, length) to this DEXT via IOConnectCallStructMethod
 *   3. DEXT calls CreateMemoryDescriptorFromClient(VA, length) → IOMemoryDescriptor
 *   4. DEXT calls IODMACommand::PrepareForDMA(descriptor) → physical segments
 *   5. Physical segments returned to client
 *
 * On Apple Silicon the "physical addresses" from IODMACommand are DART IOVAs —
 * the addresses NVMe/PCIe devices would use to access this memory via DMA.
 */

#include <os/log.h>
#include "CybMemDriver.h"
#include <DriverKit/IOMemoryDescriptor.h>
#include <DriverKit/IOBufferMemoryDescriptor.h>
#include <DriverKit/IODMACommand.h>
#include <DriverKit/IOLib.h>
#include <DriverKit/OSData.h>
#include <DriverKit/IOUserClient.h>

#define LOG(fmt, ...) os_log(OS_LOG_DEFAULT, "CybMemDriver: " fmt, ##__VA_ARGS__)

// Forward declaration
static kern_return_t
sResolvePA(OSObject * target, void * reference,
           IOUserClientMethodArguments * arguments);

static const IOUserClientMethodDispatch sMethods[kCybMemMethodCount] = {
    [kCybMemMethodResolvePA] = {
        .function                 = sResolvePA,
        .checkCompletionExists    = 0,
        .checkScalarInputCount    = 0,
        .checkStructureInputSize  = sizeof(CybMemInput),
        .checkScalarOutputCount   = 0,
        .checkStructureOutputSize = kIOUserClientVariableStructureSize,
    },
};

// --- IVars ---

struct CybMemDriver_IVars {
    IOService * provider;
};

// --- Lifecycle ---

bool CybMemDriver::init()
{
    LOG("init");
    if (!super::init()) return false;
    ivars = IONewZero(CybMemDriver_IVars, 1);
    return ivars != nullptr;
}

void CybMemDriver::free()
{
    LOG("free");
    IOSafeDeleteNULL(ivars, CybMemDriver_IVars, 1);
    super::free();
}

kern_return_t CybMemDriver::Start(IOService * provider)
{
    LOG("Start");
    kern_return_t ret = super::Start(provider);
    if (ret != kIOReturnSuccess) return ret;
    ivars->provider = provider;
    ret = RegisterService();
    if (ret != kIOReturnSuccess) {
        LOG("RegisterService failed: 0x%x", ret);
        return ret;
    }
    LOG("ready");
    return kIOReturnSuccess;
}

kern_return_t CybMemDriver::Stop(IOService * provider)
{
    LOG("Stop");
    return super::Stop(provider);
}

// --- ExternalMethod dispatch ---

kern_return_t CybMemDriver::ExternalMethod(
    uint64_t                            selector,
    IOUserClientMethodArguments       * arguments,
    const IOUserClientMethodDispatch  * dispatch,
    OSObject                          * target,
    void                              * reference)
{
    if (selector >= kCybMemMethodCount) {
        return kIOReturnBadArgument;
    }
    return super::ExternalMethod(selector, arguments,
                                 &sMethods[selector], this, nullptr);
}

kern_return_t CybMemDriver::CopyClientMemoryForType(
    uint64_t type, uint64_t * options, IOMemoryDescriptor ** memory)
{
    return kIOReturnUnsupported;
}

// --- Core: resolve client VA → physical segments ---

static kern_return_t
sResolvePA(OSObject * target, void * reference,
           IOUserClientMethodArguments * arguments)
{
    CybMemDriver * self = OSRequiredCast(CybMemDriver, target);
    kern_return_t ret;

    // Parse input
    if (!arguments->structureInput) {
        LOG("no input");
        return kIOReturnBadArgument;
    }

    const CybMemInput * input =
        reinterpret_cast<const CybMemInput *>(
            arguments->structureInput->getBytesNoCopy());
    if (!input) return kIOReturnBadArgument;

    uint64_t clientVA   = input->client_va;
    uint64_t byteLength = input->byte_length;
    uint32_t surfaceId  = input->surface_id;

    LOG("resolve: surface_id=%u VA=0x%llx len=%llu", surfaceId, clientVA, byteLength);

    if (byteLength == 0 || clientVA == 0) {
        return kIOReturnBadArgument;
    }

    // Step 1: Wrap client's VA range into an IOMemoryDescriptor
    // CreateMemoryDescriptorFromClient takes IOAddressSegment array
    // describing the client's virtual address regions to wrap.
    IOAddressSegment clientSegments[1];
    clientSegments[0].address = clientVA;
    clientSegments[0].length  = byteLength;

    IOMemoryDescriptor * clientMD = nullptr;
    ret = self->CreateMemoryDescriptorFromClient(
        kIOMemoryDirectionOutIn,   // options: bidirectional
        1,                         // segmentsCount
        clientSegments,            // client VA segments
        &clientMD);

    if (ret != kIOReturnSuccess || !clientMD) {
        LOG("CreateMemoryDescriptorFromClient failed: 0x%x", ret);
        return ret;
    }

    LOG("wrapped client VA into IOMemoryDescriptor");

    // Step 2: Create IODMACommand for PA resolution
    IODMACommandSpecification dmaSpec;
    memset(&dmaSpec, 0, sizeof(dmaSpec));
    dmaSpec.options        = kIODMACommandSpecificationNoOptions;
    dmaSpec.maxAddressBits = 64;

    IODMACommand * dmaCmd = nullptr;
    ret = IODMACommand::Create(
        self->ivars->provider,
        kIODMACommandCreateNoOptions,
        &dmaSpec,
        &dmaCmd);

    if (ret != kIOReturnSuccess || !dmaCmd) {
        LOG("IODMACommand::Create failed: 0x%x", ret);
        OSSafeReleaseNULL(clientMD);
        return ret;
    }

    // Step 3: PrepareForDMA — resolves to physical/IOVA segments
    uint64_t         dmaFlags = 0;
    uint32_t         segCount = CYBMEM_MAX_SEGMENTS;
    IOAddressSegment segments[CYBMEM_MAX_SEGMENTS];
    memset(segments, 0, sizeof(segments));

    ret = dmaCmd->PrepareForDMA(
        kIODMACommandPrepareForDMANoOptions,
        clientMD,
        0,              // offset
        0,              // length (0 = entire descriptor)
        &dmaFlags,
        &segCount,
        segments);

    if (ret != kIOReturnSuccess) {
        LOG("PrepareForDMA failed: 0x%x", ret);
        OSSafeReleaseNULL(dmaCmd);
        OSSafeReleaseNULL(clientMD);
        return ret;
    }

    LOG("PrepareForDMA: %u segments, flags=0x%llx", segCount, dmaFlags);

    // Step 4: Pack results
    CybMemOutput output;
    memset(&output, 0, sizeof(output));
    output.num_segments = segCount;
    output.flags        = (uint32_t)dmaFlags;
    output.total_length = byteLength;

    for (uint32_t i = 0; i < segCount && i < CYBMEM_MAX_SEGMENTS; i++) {
        output.segments[i].address = segments[i].address;
        output.segments[i].length  = segments[i].length;
        LOG("  seg[%u]: addr=0x%llx len=%llu", i, segments[i].address, segments[i].length);
    }

    // Cleanup DMA
    dmaCmd->CompleteDMA(kIODMACommandCompleteDMANoOptions);
    OSSafeReleaseNULL(dmaCmd);
    OSSafeReleaseNULL(clientMD);

    // Return to userspace
    arguments->structureOutput = OSData::withBytes(&output, sizeof(output));
    if (!arguments->structureOutput) {
        return kIOReturnNoMemory;
    }

    LOG("returned %u segments", segCount);
    return kIOReturnSuccess;
}
