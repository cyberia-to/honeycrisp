# first ANE dispatch

compile a matmul kernel, run it on Apple Neural Engine, read the result.

## prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Rust toolchain (`rustup`)

## step 1: create a project

```bash
cargo new ane-hello && cd ane-hello
```

add rane to Cargo.toml:

```toml
[dependencies]
rane = { path = "../rane" }  # or from crates.io when published
```

## step 2: write the program

```rust
use rane::{Source, Buffer, Program, f32_to_fp16, fp16_to_f32};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build a 64×64 matmul: y = x @ W
    let program = rane::mil::matmul(64, 64, 64);

    // compile MIL text → ANE bytecode
    let mut model = Program::compile(&program, &[])?;

    // upload bytecode to ANE SRAM
    model.load()?;

    // allocate shared-memory surfaces (IOSurface-backed)
    let input = Buffer::new(program.input_size())?;
    let output = Buffer::new(program.output_size())?;

    // fill input: activations = 1.0, weights = identity matrix
    let (ic, sp) = program.input_shape();
    let seq = 64;
    input.write(|d| {
        for ch in 0..ic {
            for s in 0..seq {
                d[ch * sp + s] = f32_to_fp16(1.0);
            }
            for o in 0..64 {
                d[ch * sp + seq + o] = if ch == o { f32_to_fp16(1.0) } else { 0 };
            }
        }
    });

    // dispatch on ANE hardware
    model.run(&input, &output)?;

    // read result
    output.read(|d| {
        let val = fp16_to_f32(d[0]);
        println!("output[0] = {}", val); // should be 1.0
    });

    Ok(())
}
```

## step 3: run

```bash
cargo run
```

output:

```
output[0] = 1
```

## what happened

1. `matmul(64, 64, 64)` generated MIL text — Apple's model intermediate language
2. `compile()` sent the MIL to `aned` daemon via XPC, which compiled it to ANE bytecode
3. `load()` uploaded the bytecode to ANE SRAM
4. `run()` dispatched the kernel on ANE hardware using IOSurface shared memory
5. `read()` read the fp16 result directly from the IOSurface — zero copies
