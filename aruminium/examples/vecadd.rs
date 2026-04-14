//! Vector addition on Metal GPU — minimal compute example

use aruminium::{Gpu, GpuError};

fn main() -> Result<(), GpuError> {
    let device = Gpu::open()?;
    println!("Device: {}", device.name());

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
        for i in 0..n {
            d[i] = i as f32;
        }
    });
    buf_b.write_f32(|d| {
        for i in 0..n {
            d[i] = (n - i) as f32;
        }
    });

    let cmd = queue.commands()?;
    let enc = cmd.encoder()?;
    enc.bind(&pipeline);
    enc.bind_buffer(&buf_a, 0, 0);
    enc.bind_buffer(&buf_b, 0, 1);
    enc.bind_buffer(&buf_c, 0, 2);
    enc.launch((n, 1, 1), (pipeline.thread_execution_width(), 1, 1));
    enc.finish();
    cmd.submit();
    cmd.wait();

    buf_c.read_f32(|d| {
        let mut ok = true;
        for i in 0..n {
            let expected = n as f32;
            if (d[i] - expected).abs() > 1e-6 {
                println!("FAIL: c[{}] = {} (expected {})", i, d[i], expected);
                ok = false;
                break;
            }
        }
        if ok {
            println!("PASS: {} vector additions verified (all = {})", n, n);
        }
    });

    Ok(())
}
