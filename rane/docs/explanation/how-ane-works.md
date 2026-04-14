# how ANE access works

Apple Neural Engine is a fixed-function accelerator on Apple Silicon.
Apple provides no public API. this crate reaches ANE through the same
private ObjC classes that CoreML uses internally.

## the problem

Apple locks ANE behind three barriers:

1. IOKit driver (H11ANEIn) requires `com.apple.ane.iokit-user-access`
   entitlement — only `/usr/libexec/aned` has it
2. three private frameworks (AppleNeuralEngine, ANECompiler,
   ANEServices) have no headers and no documentation
3. the compilation pipeline accepts MIL (Model Intermediate Language)
   text, a format documented nowhere outside Apple

## how rane bypasses all three

barrier 1: aned is an XPC service. any process can talk to it through
the ObjC runtime. the crate calls `objc_msgSend` directly — no ObjC
compiler needed, `objc_msgSend` is just a C function in libobjc.dylib.

barrier 2: `dlopen` loads the private frameworks at runtime. class and
selector names are discovered through the ObjC runtime (`objc_getClass`,
`sel_registerName`). no headers, no linking.

barrier 3: MIL format was reverse-engineered from CoreML model
bundles. the crate generates MIL text programmatically.

## the data path

CPU and ANE share memory through IOSurfaces — kernel-managed shared
memory objects. creating an IOSurface requires no entitlement (it is
a public macOS API). the ANE reads/writes the same physical memory
pages the CPU sees. no copies.

IOSurfaces must be locked before CPU access and unlocked before ANE
dispatch. the lock/unlock protocol ensures cache coherence between
CPU and ANE.

## why MIL text and not bytecode

the ANE bytecode format is undocumented and changes between chip
generations. MIL text is the stable interface — AppleNeuralEngine.framework
compiles it to the correct bytecode for the current hardware. this
makes the crate forward-compatible: same MIL works on M1, M2, M3, M4.

## dynamic weights

traditional neural engine usage embeds weights in the model at compile
time. this crate uses a different pattern: weights are packed into the
input IOSurface alongside activations. the MIL program slices them
apart and uses them as matmul operands.

this means one compiled kernel handles all 28 layers of a transformer —
different weights are staged into the input surface before each dispatch.
compile once, run 28 times with different data.

## fp16 everywhere

ANE operates exclusively in fp16 (IEEE 754 half-precision). all data
entering or leaving the ANE must be fp16. the crate provides inline
NEON assembly for fast f32↔fp16 conversion on ARM64.
