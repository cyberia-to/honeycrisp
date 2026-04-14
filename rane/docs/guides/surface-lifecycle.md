# surface lifecycle

how to create, write, read, and release Buffer buffers.

## create by size

```rust
let surface = Buffer::new(8192)?; // 8 KB
```

## create by tensor shape

for a 4D tensor `[1, channels, 1, spatial]` in fp16:

```rust
let surface = Buffer::with_shape(64, 128)?; // 64 channels × 128 spatial × 2 bytes = 16 KB
```

## write fp16 data

surface must be locked before CPU access. `write` handles lock/unlock:

```rust
surface.write(|data: &mut [u16]| {
    for i in 0..data.len() {
        data[i] = f32_to_fp16(1.0);
    }
});
```

## read fp16 data

```rust
surface.read(|data: &[u16]| {
    let first = fp16_to_f32(data[0]);
    println!("first element: {}", first);
});
```

## pass to ANE

surfaces are passed directly to `model.run()`. the ANE reads/writes
the same physical memory — no copies.

```rust
model.run(&input_surface, &output_surface)?;
```

**important:** do not hold a lock while ANE is running. `read` and
`write` lock and unlock automatically. calling `run()` between
lock/unlock will deadlock.

## release

surfaces are released automatically on drop. the underlying IOSurface
is freed via `CFRelease`.

## bulk fp16 conversion

for large buffers, use vectorized NEON conversion:

```rust
let f32_data: Vec<f32> = vec![1.0; 1024];
let mut fp16_data: Vec<u16> = vec![0; 1024];
rane::cast_f32_f16(&mut fp16_data, &f32_data);

let mut back: Vec<f32> = vec![0.0; 1024];
rane::cast_f16_f32(&mut back, &fp16_data);
```
