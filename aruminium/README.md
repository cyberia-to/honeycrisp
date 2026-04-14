# aruminium

the lightest metal.

pure Rust Apple Metal GPU driver. zero external dependencies. direct `objc_msgSend` FFI to Metal.framework — no objc runtime, no Swift, no headers.

```rust,ignore
let device = aruminium::Gpu::open()?;
let queue = device.new_command_queue()?;

let lib = device.compile(r#"
    #include <metal_stdlib>
    kernel void add(device float *a [[buffer(0)]],
                    device float *b [[buffer(1)]],
                    device float *c [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
        c[id] = a[id] + b[id];
    }
"#)?;

let pipe = device.pipeline(&lib.function("add")?)?;
let buf = device.buffer(n * 4)?;

let cmd = queue.commands()?;
let enc = cmd.encoder()?;
enc.bind(&pipe);
enc.bind_buffer(&buf, 0, 0);
enc.launch((n, 1, 1), (256, 1, 1));
enc.finish();
cmd.submit();
cmd.wait();
```

## numbers

M1 Pro 16-core:

```text
buffer create (1 MB):     0.01 ms
shader compile:           0.01 ms
dispatch overhead:        0.21 ms CPU / 0.18 ms GPU
SAXPY 16M floats:         143 GB/s
fp16 conversion:          58-72 GB/s
```

vs objc2-metal (standard Rust Metal binding):

```text
batch encode:    1.13x faster
pipelined:       1.79x faster
inference sim:   1.11x faster (300 layers)
SAXPY:           143 vs 140 GB/s
```

same GPU, same work. aruminium is lighter.

## zero-copy memory

GPU buffers can wrap `unimem::Block` — IOSurface-backed pinned memory shared with CPU (acpu) and ANE (rane):

```rust,ignore
use aruminium::{Gpu, Block};

let block = Block::open(n * 4)?;
let gpu = Gpu::open()?;
let buf = gpu.wrap(&block)?;  // MTLBuffer over same physical pages
// GPU reads/writes block's memory directly — zero copies
```

one allocation. three devices. no copies.

## api

```rust,ignore
// device
Gpu::open() -> Result<Gpu>
device.name() -> String
device.has_unified_memory() -> bool

// buffers
device.buffer(bytes) -> Result<Buffer>
device.buffer_with_data(&[u8]) -> Result<Buffer>
buf.read(|&[u8]|)
buf.write_f32(|&mut [f32]|)

// shaders
device.compile(&str) -> Result<ShaderLib>
lib.function(&str) -> Result<Shader>
device.pipeline(&Shader) -> Result<Pipeline>

// dispatch
queue.commands() -> Result<Commands>
cmd.encoder() -> Result<Encoder>
enc.bind(&pipe)
enc.bind_buffer(&buf, offset, index)
enc.launch_groups(grid, group)
enc.finish()
cmd.submit()
cmd.wait()

// sync
Fence, Event, SharedEvent

// profiling
cmd.gpu_time() -> f64
pipeline.static_threadgroup_memory_length() -> usize

// fp16
aruminium::f32_to_fp16(f32) -> u16
aruminium::fp16_to_f32(u16) -> f32
aruminium::cast_f32_f16(&mut [u16], &[f32])  // bulk NEON
aruminium::cast_f16_f32(&mut [f32], &[u16])  // bulk NEON
```

## build

```text
cargo build --release
cargo run --example vecadd
cargo run --example matmul
cargo run --release -p metal-benches --bin bench
```

requires macOS with Metal-capable GPU.

## contributing

we welcome pull requests. especially:

- **performance** — faster dispatch, better batching, tighter FFI
- **safety** — soundness fixes, better error handling, UB elimination
- **hardware** — tested only on M1 Pro. if you have M2/M3/M4 or any other Apple Silicon — your benchmarks and fixes are gold

fork, branch, PR. keep it simple: `cargo fmt`, `cargo clippy`, `cargo test`, examples run.

## license

don't trust. don't fear. don't beg.
