# rane — API specification

the public interface for Apple Neural Engine access from Rust.

## concepts

| concept | what it is |
|---------|-----------|
| kernel | a compiled ANE program — one MIL function → one hardware dispatch |
| surface | shared-memory tensor buffer (IOSurface) — the only data path CPU↔ANE |
| program | MIL text describing tensor operations — compiled to ANE bytecode at runtime |
| blob | binary weight data (fp16 or int8) with a header — embedded in or alongside MIL |

## lifecycle

```
program  →  compile  →  load  →  run  →  unload
  MIL text    bytecode    SRAM     execute   free
```

compile: MIL text + optional weight blobs → ANE bytecode (via aned daemon)
load: upload bytecode into ANE SRAM
run: dispatch on hardware, block until done
unload: free SRAM (automatic on drop)

## Program

the central type. owns a compiled ANE model.

| method | signature | semantics |
|--------|-----------|-----------|
| compile | `(program, weights) → Result<Program>` | compile MIL to bytecode. weights: `[(&str, &[u8])]` |
| load | `(&mut self) → Result<()>` | upload bytecode to ANE SRAM |
| run | `(&self, input, output) → Result<()>` | execute on ANE hardware (synchronous) |
| unload | `(&mut self) → Result<()>` | free SRAM. idempotent |
| drop | automatic | unload + cleanup temp directory |

### apple mapping

| rane method | ObjC class | ObjC selector |
|------------|-----------|---------------|
| compile | _ANEInMemoryModelDescriptor | modelWithMILText:weights:optionsPlist: |
| | _ANEInMemoryModel | inMemoryModelWithDescriptor: |
| | _ANEInMemoryModel | compileWithQoS:options:error: |
| load | _ANEInMemoryModel | loadWithQoS:options:error: |
| run | _ANEInMemoryModel | evaluateWithQoS:options:request:error: |
| unload | _ANEInMemoryModel | unloadWithQoS:error: |

run internally wraps IOSurfaces in `_ANEIOSurfaceObject` and builds `_ANERequest`.

## surface

shared-memory tensor buffer backed by IOSurface. zero-copy between CPU and ANE.

| method | signature | semantics |
|--------|-----------|-----------|
| new | `(bytes) → Result<Surface>` | allocate by byte size |
| with_shape | `(channels, spatial) → Result<Surface>` | allocate for `[1, C, 1, S]` fp16 tensor |
| read | `(&self, \|&[u16]\|) → R` | lock, read fp16 data, unlock |
| write | `(&self, \|&mut [u16]\|) → R` | lock, write fp16 data, unlock |
| id | `(&self) → u32` | IOSurface ID |
| size | `(&self) → usize` | allocation in bytes |
| drop | automatic | CFRelease |

### apple mapping

| rane method | system call |
|------------|------------|
| new | IOSurfaceCreate(dict) |
| read | IOSurfaceLock(kRead) → closure → IOSurfaceUnlock |
| write | IOSurfaceLock(0) → closure → IOSurfaceUnlock |
| drop | CFRelease |

## program

MIL text builder. produces text consumed by `compile`.

| method | signature | semantics |
|--------|-----------|-----------|
| matmul | `(ic, oc, seq) → Program` | dynamic matmul: weights packed in input spatial |
| from_text | `(mil, in_ch, in_sp, out_ch, out_sp) → Program` | custom MIL |
| input_shape | `(&self) → (channels, spatial)` | input tensor dimensions |
| output_shape | `(&self) → (channels, spatial)` | output tensor dimensions |
| input_size | `(&self) → usize` | channels × spatial × 2 |
| output_size | `(&self) → usize` | channels × spatial × 2 |
| as_str | `(&self) → &str` | raw MIL text |

## conversion

fp16↔f32 conversion via inline NEON assembly (ARM64) with software fallback.

| function | signature | semantics |
|----------|-----------|-----------|
| f32_to_fp16 | `(f32) → u16` | f32 → IEEE 754 half-precision |
| fp16_to_f32 | `(u16) → f32` | IEEE 754 half-precision → f32 |
| cast_f32_f16 | `(&mut [u16], &[f32])` | bulk NEON-vectorized f32→fp16 |
| cast_f16_f32 | `(&mut [f32], &[u16])` | bulk NEON-vectorized fp16→f32 |

## blob

binary weight format for MIL `BLOBFILE` references.

| function | signature | semantics |
|----------|-----------|-----------|
| pack_weights | `(&[u16]) → Vec<u8>` | wrap fp16 data with 128-byte header |

### fp16 blob layout

```
offset  size  content
0x00    4     0x01 (version)
0x04    4     0x02 (format flag)
0x40    4     0xDEADBEEF (magic, little-endian)
0x44    4     0x01 (dtype: fp16)
0x48    4     data size in bytes
0x50    4     data offset (128)
0x80    var   fp16[] weights
```

## errors

```
SurfaceCreationFailed(String)   IOSurface allocation failed
ClassNotFound(&str)             ObjC class missing from framework
DescriptorCreationFailed        MIL descriptor rejected
ModelCreationFailed             model object allocation failed
CompilationFailed(String)       MIL→bytecode compilation error
LoadFailed(String)              SRAM upload failed
EvalFailed(String)              hardware execution failed
UnloadFailed(String)            SRAM release failed
Io(io::Error)                   filesystem error
```

## MIL operations

operations verified on ANE hardware:

| group | operations |
|-------|-----------|
| arithmetic | add, sub, mul, matmul |
| shape | reshape, transpose, slice_by_size, concat |
| activation | softmax, sigmoid |
| conv | conv (1×1 = dense layer) |
| type | cast (fp16↔fp32), quantize (fp16→int8), dequantize (int8→fp16) |
| data | const |

rejected by ANE (CPU-only): reduce_mean, rsqrt, reduce_sum, pow.

## tensor layout

all tensors are 4D: `[batch, channels, height, width]`.

for 1D sequences: `[1, C, 1, S]` — model dimension is channels, sequence length is spatial.

### dynamic weight packing

weights and activations in one surface:

```
input [1, IC, 1, SEQ + OC]:
  spatial[0..SEQ]       activations
  spatial[SEQ..SEQ+OC]  weight matrix (transposed)
```

the MIL program slices, reshapes, and matmuls internally.

## execution model

- one kernel = one compiled MIL function = one hardware dispatch
- run is synchronous — blocks until ANE completes
- surfaces are reusable: write new data, run again
- multiple kernels can be loaded simultaneously
- ANE executes serially on hardware (one dispatch at a time)
- weight staging is handled by the runtime layer (cyb/llm), not the driver

## driver stack

```
rane crate (objc_msgSend FFI)
  → AppleNeuralEngine.framework (dlopen at runtime)
    → ANECompiler.framework (MIL → bytecode)
      → XPC to aned daemon
        → IOKit H11ANEIn driver
          → ANE hardware
```

three private frameworks, loaded once via dlopen:
- AppleNeuralEngine — ObjC classes, MIL validation, XPC interface
- ANECompiler — bytecode generation (ANECCompile)
- ANEServices — device/program lifecycle (aned internal)

all user code goes through XPC to aned. aned holds the IOKit entitlement and validates all bytecode before hardware dispatch.
