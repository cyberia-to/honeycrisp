# tutorial

step-by-step: from zero to GPU compute in aruminium.

## prerequisites

- macOS with Metal-capable GPU (any Mac from 2012+)
- Rust toolchain (`rustup`)

## step 1: create project

```bash
cargo new my-gpu-app
cd my-gpu-app
```

add aruminium to `Cargo.toml`:

```toml
[dependencies]
aruminium = { path = "../aruminium" }  # or git URL
```

## step 2: discover the GPU

```rust
use aruminium::{Gpu, GpuError};

fn main() -> Result<(), GpuError> {
    let device = Gpu::open()?;
    println!("GPU: {}", device.name());
    println!("Unified memory: {}", device.has_unified_memory());
    println!("Max buffer: {} MB", device.max_buffer_length() / (1024 * 1024));
    Ok(())
}
```

```
$ cargo run
GPU: Apple M1 Pro
Unified memory: true
Max buffer: 5461 MB
```

## step 3: allocate GPU buffers

create three buffers for vector addition: A + B = C.

```rust
let n = 1024usize;
let buf_a = device.buffer(n * 4)?;  // 1024 floats = 4096 bytes
let buf_b = device.buffer(n * 4)?;
let buf_c = device.buffer(n * 4)?;
```

these are shared-mode buffers: CPU and GPU see the same memory.

write data from CPU:

```rust
buf_a.write_f32(|data| {
    for i in 0..n {
        data[i] = i as f32;
    }
});
buf_b.write_f32(|data| {
    for i in 0..n {
        data[i] = (n - i) as f32;
    }
});
```

## step 4: write a Metal shader

Metal Shading Language (MSL) is C++-like. each thread gets its
position via `[[thread_position_in_grid]]`:

```rust
let source = r#"
    #include <metal_stdlib>
    using namespace metal;
    kernel void vecadd(device float *a [[buffer(0)]],
                       device float *b [[buffer(1)]],
                       device float *c [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
        c[id] = a[id] + b[id];
    }
"#;
```

## step 5: compile and create pipeline

```rust
let lib = device.compile(source)?;
let func = lib.function("vecadd")?;
let pipeline = device.pipeline(&func)?;
```

this compiles MSL to GPU bytecode at runtime. the pipeline is
a reusable compiled state — create once, dispatch many times.

## step 6: encode and dispatch

```rust
let queue = device.new_command_queue()?;
let cmd = queue.commands()?;
let enc = cmd.encoder()?;

enc.bind(&pipeline);
enc.bind_buffer(&buf_a, 0, 0);  // buffer, offset, index
enc.bind_buffer(&buf_b, 0, 1);
enc.bind_buffer(&buf_c, 0, 2);
enc.launch(
    (n, 1, 1),   // grid: 1024 threads total
    (64, 1, 1),  // threadgroup: 64 threads per group
);
enc.finish();

cmd.submit();
cmd.wait();
```

## step 7: read results

```rust
buf_c.read_f32(|data| {
    for i in 0..n {
        let expected = n as f32;  // i + (n - i) = n
        assert!((data[i] - expected).abs() < 1e-6);
    }
    println!("PASS: {} additions verified", n);
});
```

## step 8: add GPU timing

```rust
cmd.submit();
cmd.wait();
println!("GPU time: {:.3} ms", cmd.gpu_time() * 1000.0);
```

## step 9: pass parameters via push

for small constants (uniforms), use `push` instead of a buffer:

```rust
#[repr(C)]
struct Params { n: u32, scale: f32 }
let params = Params { n: 1024, scale: 2.0 };

let bytes = unsafe {
    std::slice::from_raw_parts(
        &params as *const Params as *const u8,
        std::mem::size_of::<Params>(),
    )
};
enc.push(bytes, 3);  // bind at index 3
```

in the shader:

```metal
kernel void my_kernel(constant Params &p [[buffer(3)]],
                      ...) {
    float x = p.scale * input[id];
}
```

## complete example

```rust
use aruminium::{Gpu, GpuError};

fn main() -> Result<(), GpuError> {
    let device = Gpu::open()?;
    let queue = device.new_command_queue()?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void vecadd(device float *a [[buffer(0)]],
                           device float *b [[buffer(1)]],
                           device float *c [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
            c[id] = a[id] + b[id];
        }
    "#;

    let lib = device.compile(source)?;
    let func = lib.function("vecadd")?;
    let pipeline = device.pipeline(&func)?;

    let n = 1024usize;
    let buf_a = device.buffer(n * 4)?;
    let buf_b = device.buffer(n * 4)?;
    let buf_c = device.buffer(n * 4)?;

    buf_a.write_f32(|d| {
        for i in 0..n { d[i] = i as f32; }
    });
    buf_b.write_f32(|d| {
        for i in 0..n { d[i] = (n - i) as f32; }
    });

    let cmd = queue.commands()?;
    let enc = cmd.encoder()?;
    enc.bind(&pipeline);
    enc.bind_buffer(&buf_a, 0, 0);
    enc.bind_buffer(&buf_b, 0, 1);
    enc.bind_buffer(&buf_c, 0, 2);
    enc.launch((n, 1, 1), (64, 1, 1));
    enc.finish();
    cmd.submit();
    cmd.wait();

    buf_c.read_f32(|d| {
        let ok = d.iter().enumerate().all(|(i, &v)| {
            (v - n as f32).abs() < 1e-6
        });
        println!("{}: {} additions, GPU {:.3} ms",
            if ok { "PASS" } else { "FAIL" },
            n,
            cmd.gpu_time() * 1000.0);
    });

    Ok(())
}
```

## next steps

- see `examples/matmul.rs` for 2D dispatch with struct parameters
- see `docs/guide.md` for Dispatch, batch encoding, pipelining
- see `docs/explanations.md` for why the architecture works this way
