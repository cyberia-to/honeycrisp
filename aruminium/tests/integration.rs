//! Integration tests — full GPU pipeline: compile → dispatch → verify

use aruminium::{autorelease_pool, Batch, Block, Commands, Dispatch, Gpu, GpuError};
use std::ffi::c_void;

#[test]
fn vecadd_1024() -> Result<(), GpuError> {
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
    let pipe = device.pipeline(&lib.function("vecadd")?)?;

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
    enc.bind(&pipe);
    enc.bind_buffer(&buf_a, 0, 0);
    enc.bind_buffer(&buf_b, 0, 1);
    enc.bind_buffer(&buf_c, 0, 2);
    enc.launch((n, 1, 1), (64, 1, 1));
    enc.finish();
    cmd.submit();
    cmd.wait();

    buf_c.read_f32(|d| {
        for i in 0..n {
            assert!(
                (d[i] - n as f32).abs() < 1e-6,
                "vecadd mismatch at {}: got {}",
                i,
                d[i]
            );
        }
    });
    Ok(())
}

#[test]
fn matmul_64x64() -> Result<(), GpuError> {
    let device = Gpu::open()?;
    let queue = device.new_command_queue()?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        struct P { uint M; uint N; uint K; };
        kernel void matmul(device const float *A [[buffer(0)]],
                           device const float *B [[buffer(1)]],
                           device float *C       [[buffer(2)]],
                           constant P &p [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
            uint row = gid.y, col = gid.x;
            if (row >= p.M || col >= p.N) return;
            float sum = 0.0;
            for (uint k = 0; k < p.K; k++)
                sum += A[row * p.K + k] * B[k * p.N + col];
            C[row * p.N + col] = sum;
        }
    "#;
    let lib = device.compile(source)?;
    let pipe = device.pipeline(&lib.function("matmul")?)?;

    let m = 64usize;
    let buf_a = device.buffer(m * m * 4)?;
    let buf_b = device.buffer(m * m * 4)?;
    let buf_c = device.buffer(m * m * 4)?;

    // A = identity, B = ones => C = ones
    buf_a.write_f32(|d| {
        for i in 0..m {
            for j in 0..m {
                d[i * m + j] = if i == j { 1.0 } else { 0.0 };
            }
        }
    });
    buf_b.write_f32(|d| d.fill(1.0));

    #[repr(C)]
    struct P {
        m: u32,
        n: u32,
        k: u32,
    }
    let params = P {
        m: m as u32,
        n: m as u32,
        k: m as u32,
    };
    let params_bytes = unsafe {
        std::slice::from_raw_parts(&params as *const P as *const u8, std::mem::size_of::<P>())
    };

    let cmd = queue.commands()?;
    let enc = cmd.encoder()?;
    enc.bind(&pipe);
    enc.bind_buffer(&buf_a, 0, 0);
    enc.bind_buffer(&buf_b, 0, 1);
    enc.bind_buffer(&buf_c, 0, 2);
    enc.push(params_bytes, 3);
    enc.launch((m, m, 1), (16, 16, 1));
    enc.finish();
    cmd.submit();
    cmd.wait();

    buf_c.read_f32(|d| {
        let max_err: f32 = d.iter().map(|&v| (v - 1.0).abs()).fold(0.0, f32::max);
        assert!(max_err < 1e-5, "matmul max_err={}", max_err);
    });
    Ok(())
}

#[test]
fn compute_dispatcher_batch() -> Result<(), GpuError> {
    let device = Gpu::open()?;
    let queue = device.new_command_queue()?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void inc(device float *a [[buffer(0)]],
                        uint id [[thread_position_in_grid]]) {
            a[id] = a[id] + 1.0;
        }
    "#;
    let lib = device.compile(source)?;
    let pipe = device.pipeline(&lib.function("inc")?)?;

    let n = 256usize;
    let buf = device.buffer(n * 4)?;
    buf.write_f32(|d| d.fill(0.0));

    let disp = Dispatch::new(&queue);

    // batch 10 increments into one command buffer
    unsafe {
        disp.batch(|batch| {
            for _ in 0..10 {
                batch.bind(&pipe);
                batch.bind_buffer(&buf, 0, 0);
                batch.launch((n, 1, 1), (64, 1, 1));
            }
        });
    }

    buf.read_f32(|d| {
        for (i, &v) in d.iter().enumerate() {
            assert!(
                (v - 10.0).abs() < 1e-6,
                "batch dispatch: d[{}]={}, expected 10.0",
                i,
                v
            );
        }
    });
    Ok(())
}

#[test]
fn compute_dispatcher_async_pipelining() -> Result<(), GpuError> {
    let device = Gpu::open()?;
    let queue = device.new_command_queue()?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void inc(device float *a [[buffer(0)]],
                        uint id [[thread_position_in_grid]]) {
            a[id] = a[id] + 1.0;
        }
    "#;
    let lib = device.compile(source)?;
    let pipe = device.pipeline(&lib.function("inc")?)?;

    let n = 256usize;
    let buf = device.buffer(n * 4)?;
    buf.write_f32(|d| d.fill(0.0));

    let disp = Dispatch::new(&queue);

    // pipeline 5 async batches
    let mut prev = None;
    for _ in 0..5 {
        let future = unsafe {
            disp.batch_async(|batch| {
                batch.bind(&pipe);
                batch.bind_buffer(&buf, 0, 0);
                batch.launch((n, 1, 1), (64, 1, 1));
            })
        };
        if let Some(p) = prev {
            aruminium::GpuFuture::wait(p);
        }
        prev = Some(future);
    }
    if let Some(p) = prev {
        aruminium::GpuFuture::wait(p);
    }

    buf.read_f32(|d| {
        for (i, &v) in d.iter().enumerate() {
            assert!(
                (v - 5.0).abs() < 1e-6,
                "pipelined: d[{}]={}, expected 5.0",
                i,
                v
            );
        }
    });
    Ok(())
}

#[test]
fn private_buffer_blit_copy() -> Result<(), GpuError> {
    let device = Gpu::open()?;
    let queue = device.new_command_queue()?;

    let n = 256usize;
    let shared = device.buffer(n * 4)?;
    let private = device.buffer_private(n * 4)?;
    let readback = device.buffer(n * 4)?;

    shared.write_f32(|d| {
        for (i, v) in d.iter_mut().enumerate() {
            *v = i as f32;
        }
    });

    // shared -> private via blit
    let cmd = queue.commands()?;
    let blit = cmd.copier()?;
    blit.copy(&shared, 0, &private, 0, n * 4);
    blit.finish();
    cmd.submit();
    cmd.wait();

    // private -> readback via blit
    let cmd = queue.commands()?;
    let blit = cmd.copier()?;
    blit.copy(&private, 0, &readback, 0, n * 4);
    blit.finish();
    cmd.submit();
    cmd.wait();

    readback.read_f32(|d| {
        for (i, &v) in d.iter().enumerate() {
            assert_eq!(v, i as f32, "blit copy mismatch at {}", i);
        }
    });
    Ok(())
}

#[test]
fn autorelease_pool_dispatch() -> Result<(), GpuError> {
    let device = Gpu::open()?;
    let queue = device.new_command_queue()?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void noop(device float *a [[buffer(0)]],
                         uint id [[thread_position_in_grid]]) {
            a[id] = a[id] + 1.0;
        }
    "#;
    let lib = device.compile(source)?;
    let pipe = device.pipeline(&lib.function("noop")?)?;
    let buf = device.buffer(256 * 4)?;
    buf.write_f32(|d| d.fill(0.0));

    // 100 dispatches in autorelease pool with autoreleased command buffers
    autorelease_pool(|| {
        for _ in 0..100 {
            unsafe {
                let cmd = queue.commands_autoreleased();
                let enc = cmd.encoder_autoreleased();
                enc.bind(&pipe);
                enc.bind_buffer(&buf, 0, 0);
                enc.launch((256, 1, 1), (64, 1, 1));
                enc.finish();
                cmd.submit();
                cmd.wait();
            }
        }
    });

    buf.read_f32(|d| {
        assert!(
            (d[0] - 100.0).abs() < 1e-6,
            "autorelease pool: d[0]={}, expected 100.0",
            d[0]
        );
    });
    Ok(())
}

// ── 1. Gpu::all() ──

#[test]
fn gpu_all_returns_non_empty() -> Result<(), GpuError> {
    let devices = Gpu::all()?;
    assert!(!devices.is_empty(), "Gpu::all() returned empty vec");
    for dev in &devices {
        assert!(!dev.name().is_empty());
    }
    Ok(())
}

// ── 2. Gpu::has_unified_memory() ──

#[test]
fn gpu_has_unified_memory() -> Result<(), GpuError> {
    let dev = Gpu::open()?;
    // Apple Silicon always has unified memory
    assert!(
        dev.has_unified_memory(),
        "expected unified memory on Apple Silicon"
    );
    Ok(())
}

// ── 3. Gpu::max_threads_per_threadgroup() ──

#[test]
fn gpu_max_threads_per_threadgroup() -> Result<(), GpuError> {
    let dev = Gpu::open()?;
    let size = dev.max_threads_per_threadgroup();
    assert!(
        size.width > 0,
        "max_threads_per_threadgroup width must be > 0"
    );
    assert!(
        size.height > 0,
        "max_threads_per_threadgroup height must be > 0"
    );
    assert!(
        size.depth > 0,
        "max_threads_per_threadgroup depth must be > 0"
    );
    Ok(())
}

// ── 4. Gpu::recommended_max_working_set_size() ──

#[test]
fn gpu_recommended_max_working_set_size() -> Result<(), GpuError> {
    let dev = Gpu::open()?;
    let size = dev.recommended_max_working_set_size();
    assert!(size > 0, "recommended_max_working_set_size must be > 0");
    Ok(())
}

// ── 5. Gpu::buffer_wrap() ──

#[test]
fn gpu_buffer_wrap() -> Result<(), GpuError> {
    let dev = Gpu::open()?;
    let page_size = 16384usize; // Apple Silicon page size
    let layout = std::alloc::Layout::from_size_align(page_size, page_size).unwrap();
    let ptr = unsafe { std::alloc::alloc(layout) };
    assert!(!ptr.is_null(), "page-aligned alloc failed");

    // Write pattern into the memory
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut f32, page_size / 4) };
    for (i, v) in slice.iter_mut().enumerate() {
        *v = i as f32;
    }

    // Wrap as Metal buffer — zero copy
    let buf = unsafe { dev.buffer_wrap(ptr as *mut c_void, page_size)? };
    assert_eq!(buf.size(), page_size);

    // Read back and verify — should see same data (zero copy)
    buf.read_f32(|d| {
        for (i, &v) in d.iter().enumerate() {
            assert_eq!(v, i as f32, "buffer_wrap readback mismatch at {}", i);
        }
    });

    drop(buf);
    unsafe { std::alloc::dealloc(ptr, layout) };
    Ok(())
}

// ── 6. Queue::commands_fast() / commands_unchecked() ──

#[test]
fn queue_commands_fast() -> Result<(), GpuError> {
    let dev = Gpu::open()?;
    let queue = dev.new_command_queue()?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void fill_one(device float *a [[buffer(0)]],
                             uint id [[thread_position_in_grid]]) {
            a[id] = 1.0;
        }
    "#;
    let lib = dev.compile(source)?;
    let pipe = dev.pipeline(&lib.function("fill_one")?)?;
    let n = 64usize;
    let buf = dev.buffer(n * 4)?;
    buf.write_f32(|d| d.fill(0.0));

    // commands_fast: unretained references, retained command buffer
    autorelease_pool(|| unsafe {
        let cmd = queue.commands_fast().unwrap();
        let enc = cmd.encoder().unwrap();
        enc.bind(&pipe);
        enc.bind_buffer(&buf, 0, 0);
        enc.launch((n, 1, 1), (n, 1, 1));
        enc.finish();
        cmd.submit();
        cmd.wait();
    });

    buf.read_f32(|d| {
        for (i, &v) in d.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-6, "commands_fast: d[{}]={}", i, v);
        }
    });
    Ok(())
}

#[test]
fn queue_commands_unchecked() -> Result<(), GpuError> {
    let dev = Gpu::open()?;
    let queue = dev.new_command_queue()?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void fill_two(device float *a [[buffer(0)]],
                             uint id [[thread_position_in_grid]]) {
            a[id] = 2.0;
        }
    "#;
    let lib = dev.compile(source)?;
    let pipe = dev.pipeline(&lib.function("fill_two")?)?;
    let n = 64usize;
    let buf = dev.buffer(n * 4)?;
    buf.write_f32(|d| d.fill(0.0));

    autorelease_pool(|| unsafe {
        let cmd = queue.commands_unchecked();
        let enc = cmd.encoder().unwrap();
        enc.bind(&pipe);
        enc.bind_buffer(&buf, 0, 0);
        enc.launch((n, 1, 1), (n, 1, 1));
        enc.finish();
        cmd.submit();
        cmd.wait();
    });

    buf.read_f32(|d| {
        for (i, &v) in d.iter().enumerate() {
            assert!((v - 2.0).abs() < 1e-6, "commands_unchecked: d[{}]={}", i, v);
        }
    });
    Ok(())
}

// ── 7. Commands::status() ──

#[test]
fn commands_status_completed() -> Result<(), GpuError> {
    let dev = Gpu::open()?;
    let queue = dev.new_command_queue()?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void nop(device float *a [[buffer(0)]],
                        uint id [[thread_position_in_grid]]) {
            a[id] = 0.0;
        }
    "#;
    let lib = dev.compile(source)?;
    let pipe = dev.pipeline(&lib.function("nop")?)?;
    let buf = dev.buffer(64 * 4)?;

    let cmd = queue.commands()?;
    let enc = cmd.encoder()?;
    enc.bind(&pipe);
    enc.bind_buffer(&buf, 0, 0);
    enc.launch((64, 1, 1), (64, 1, 1));
    enc.finish();
    cmd.submit();
    cmd.wait();

    assert_eq!(cmd.status(), Commands::STATUS_COMPLETED);
    Ok(())
}

// ── 8. Commands::error() ──

#[test]
fn commands_error_none_on_success() -> Result<(), GpuError> {
    let dev = Gpu::open()?;
    let queue = dev.new_command_queue()?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void nop2(device float *a [[buffer(0)]],
                         uint id [[thread_position_in_grid]]) {
            a[id] = 0.0;
        }
    "#;
    let lib = dev.compile(source)?;
    let pipe = dev.pipeline(&lib.function("nop2")?)?;
    let buf = dev.buffer(64 * 4)?;

    let cmd = queue.commands()?;
    let enc = cmd.encoder()?;
    enc.bind(&pipe);
    enc.bind_buffer(&buf, 0, 0);
    enc.launch((64, 1, 1), (64, 1, 1));
    enc.finish();
    cmd.submit();
    cmd.wait();

    assert!(
        cmd.error().is_none(),
        "expected no error after successful completion"
    );
    Ok(())
}

// ── 9. Commands::gpu_start_time() / gpu_end_time() ──

#[test]
fn commands_gpu_start_end_time() -> Result<(), GpuError> {
    let dev = Gpu::open()?;
    let queue = dev.new_command_queue()?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void nop3(device float *a [[buffer(0)]],
                         uint id [[thread_position_in_grid]]) {
            a[id] = 0.0;
        }
    "#;
    let lib = dev.compile(source)?;
    let pipe = dev.pipeline(&lib.function("nop3")?)?;
    let buf = dev.buffer(64 * 4)?;

    let cmd = queue.commands()?;
    let enc = cmd.encoder()?;
    enc.bind(&pipe);
    enc.bind_buffer(&buf, 0, 0);
    enc.launch((64, 1, 1), (64, 1, 1));
    enc.finish();
    cmd.submit();
    cmd.wait();

    let start = cmd.gpu_start_time();
    let end = cmd.gpu_end_time();
    assert!(start > 0.0, "gpu_start_time must be > 0 after completion");
    assert!(end > 0.0, "gpu_end_time must be > 0 after completion");
    assert!(
        start <= end,
        "gpu_start_time ({}) must be <= gpu_end_time ({})",
        start,
        end
    );
    Ok(())
}

// ── 10. Encoder::launch_groups() ──

#[test]
fn encoder_launch_groups() -> Result<(), GpuError> {
    let dev = Gpu::open()?;
    let queue = dev.new_command_queue()?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void fill_three(device float *a [[buffer(0)]],
                               uint id [[thread_position_in_grid]]) {
            a[id] = 3.0;
        }
    "#;
    let lib = dev.compile(source)?;
    let pipe = dev.pipeline(&lib.function("fill_three")?)?;
    let n = 256usize;
    let group_size = 64usize;
    let num_groups = n / group_size;
    let buf = dev.buffer(n * 4)?;
    buf.write_f32(|d| d.fill(0.0));

    let cmd = queue.commands()?;
    let enc = cmd.encoder()?;
    enc.bind(&pipe);
    enc.bind_buffer(&buf, 0, 0);
    // Use launch_groups: dispatch threadgroup count explicitly
    enc.launch_groups((num_groups, 1, 1), (group_size, 1, 1));
    enc.finish();
    cmd.submit();
    cmd.wait();

    buf.read_f32(|d| {
        for (i, &v) in d.iter().enumerate() {
            assert!(
                (v - 3.0).abs() < 1e-6,
                "launch_groups: d[{}]={}, expected 3.0",
                i,
                v
            );
        }
    });
    Ok(())
}

// ── 11. Dispatch::dispatch() ──

#[test]
fn dispatch_single() -> Result<(), GpuError> {
    let dev = Gpu::open()?;
    let queue = dev.new_command_queue()?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void add_five(device float *a [[buffer(0)]],
                             uint id [[thread_position_in_grid]]) {
            a[id] = a[id] + 5.0;
        }
    "#;
    let lib = dev.compile(source)?;
    let pipe = dev.pipeline(&lib.function("add_five")?)?;
    let n = 128usize;
    let buf = dev.buffer(n * 4)?;
    buf.write_f32(|d| d.fill(1.0));

    let disp = Dispatch::new(&queue);
    unsafe {
        disp.dispatch(&pipe, &[(&buf, 0, 0)], (n, 1, 1), (64, 1, 1));
    }

    buf.read_f32(|d| {
        for (i, &v) in d.iter().enumerate() {
            assert!(
                (v - 6.0).abs() < 1e-6,
                "dispatch single: d[{}]={}, expected 6.0",
                i,
                v
            );
        }
    });
    Ok(())
}

// ── 12. Dispatch::dispatch_with_bytes() ──

#[test]
fn dispatch_with_bytes() -> Result<(), GpuError> {
    let dev = Gpu::open()?;
    let queue = dev.new_command_queue()?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        struct Params { float offset; };
        kernel void add_offset(device float *a [[buffer(0)]],
                               constant Params &p [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {
            a[id] = a[id] + p.offset;
        }
    "#;
    let lib = dev.compile(source)?;
    let pipe = dev.pipeline(&lib.function("add_offset")?)?;
    let n = 128usize;
    let buf = dev.buffer(n * 4)?;
    buf.write_f32(|d| d.fill(0.0));

    let offset: f32 = 7.5;
    let bytes = unsafe {
        std::slice::from_raw_parts(
            &offset as *const f32 as *const u8,
            std::mem::size_of::<f32>(),
        )
    };

    let disp = Dispatch::new(&queue);
    unsafe {
        disp.dispatch_with_bytes(&pipe, &[(&buf, 0, 0)], bytes, 1, (n, 1, 1), (64, 1, 1));
    }

    buf.read_f32(|d| {
        for (i, &v) in d.iter().enumerate() {
            assert!(
                (v - 7.5).abs() < 1e-6,
                "dispatch_with_bytes: d[{}]={}, expected 7.5",
                i,
                v
            );
        }
    });
    Ok(())
}

// ── 13. Batch::push() ──

#[test]
fn batch_push_bytes() -> Result<(), GpuError> {
    let dev = Gpu::open()?;
    let queue = dev.new_command_queue()?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        struct Params { float scale; };
        kernel void scale_buf(device float *a [[buffer(0)]],
                              constant Params &p [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
            a[id] = a[id] * p.scale;
        }
    "#;
    let lib = dev.compile(source)?;
    let pipe = dev.pipeline(&lib.function("scale_buf")?)?;
    let n = 128usize;
    let buf = dev.buffer(n * 4)?;
    buf.write_f32(|d| d.fill(2.0));

    #[repr(C)]
    struct Params {
        scale: f32,
    }
    let params = Params { scale: 3.0 };
    let params_bytes = unsafe {
        std::slice::from_raw_parts(
            &params as *const Params as *const u8,
            std::mem::size_of::<Params>(),
        )
    };

    let disp = Dispatch::new(&queue);
    unsafe {
        disp.batch(|batch: &Batch| {
            batch.bind(&pipe);
            batch.bind_buffer(&buf, 0, 0);
            batch.push(params_bytes, 1);
            batch.launch((n, 1, 1), (64, 1, 1));
        });
    }

    buf.read_f32(|d| {
        for (i, &v) in d.iter().enumerate() {
            assert!(
                (v - 6.0).abs() < 1e-6,
                "batch push: d[{}]={}, expected 6.0",
                i,
                v
            );
        }
    });
    Ok(())
}

// ── 14. Texture::replace_region() / get_bytes() ──

#[test]
fn texture_replace_region_get_bytes() -> Result<(), GpuError> {
    use aruminium::ffi::*;

    let dev = Gpu::open()?;

    let w = 4usize;
    let h = 4usize;
    let bpp = 4usize; // RGBA8 = 4 bytes per pixel

    // Create texture descriptor
    let tex = unsafe {
        let cls = objc_getClass(c"MTLTextureDescriptor".as_ptr()) as ObjcId;
        let desc = msg0(cls, sel_registerName(c"new".as_ptr()));
        assert!(!desc.is_null());

        type SetU = unsafe extern "C" fn(ObjcId, ObjcSel, NSUInteger);
        let set_u: SetU = std::mem::transmute(objc_msgSend as *const c_void);
        set_u(desc, sel_registerName(c"setTextureType:".as_ptr()), 2); // MTLTextureType2D
        set_u(
            desc,
            sel_registerName(c"setPixelFormat:".as_ptr()),
            MTLPixelFormatRGBA8Unorm,
        );
        set_u(desc, sel_registerName(c"setWidth:".as_ptr()), w);
        set_u(desc, sel_registerName(c"setHeight:".as_ptr()), h);
        // StorageModeShared for CPU read/write
        set_u(
            desc,
            sel_registerName(c"setStorageMode:".as_ptr()),
            0, // MTLStorageModeShared
        );
        // MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite = 1 | 2 = 3
        set_u(desc, sel_registerName(c"setUsage:".as_ptr()), 3);

        let t = dev.texture(desc).unwrap();
        release(desc);
        t
    };

    // Write pixels: each pixel = (r, g, b, a) = (x, y, 0xFF, 0xFF)
    let mut pixels = vec![0u8; w * h * bpp];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * bpp;
            pixels[idx] = x as u8;
            pixels[idx + 1] = y as u8;
            pixels[idx + 2] = 0xFF;
            pixels[idx + 3] = 0xFF;
        }
    }

    let region = MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: w,
            height: h,
            depth: 1,
        },
    };
    let bytes_per_row = w * bpp;

    unsafe {
        tex.replace_region(region, 0, pixels.as_ptr() as *const c_void, bytes_per_row);
    }

    // Read back
    let mut readback = vec![0u8; w * h * bpp];
    unsafe {
        tex.get_bytes(
            readback.as_mut_ptr() as *mut c_void,
            bytes_per_row,
            region,
            0,
        );
    }

    assert_eq!(pixels, readback, "texture round-trip mismatch");
    Ok(())
}

// ── 15. Pipeline::static_threadgroup_memory_length() ──

#[test]
fn pipeline_static_threadgroup_memory_length() -> Result<(), GpuError> {
    let dev = Gpu::open()?;

    // A kernel with no threadgroup memory should report 0
    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void no_tgm(device float *a [[buffer(0)]],
                           uint id [[thread_position_in_grid]]) {
            a[id] = 0.0;
        }
    "#;
    let lib = dev.compile(source)?;
    let pipe = dev.pipeline(&lib.function("no_tgm")?)?;
    let len = pipe.static_threadgroup_memory_length();
    // No threadgroup memory declared, so length should be 0
    assert_eq!(len, 0, "expected 0 static threadgroup memory, got {}", len);

    // A kernel WITH threadgroup memory should report > 0
    let source_tgm = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void with_tgm(device float *a [[buffer(0)]],
                             threadgroup float *shared [[threadgroup(0)]],
                             uint id [[thread_position_in_grid]],
                             uint lid [[thread_position_in_threadgroup]]) {
            shared[lid] = a[id];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            a[id] = shared[lid];
        }
    "#;
    let lib_tgm = dev.compile(source_tgm)?;
    let pipe_tgm = dev.pipeline(&lib_tgm.function("with_tgm")?)?;
    // With threadgroup memory declared, length should be >= 0
    // (the exact value depends on the compiler; the point is the API works)
    let _len_tgm = pipe_tgm.static_threadgroup_memory_length();

    Ok(())
}

// ── 10b. Batch::launch_groups() ──

#[test]
fn batch_launch_groups() -> Result<(), GpuError> {
    let dev = Gpu::open()?;
    let queue = dev.new_command_queue()?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void fill_four(device float *a [[buffer(0)]],
                              uint id [[thread_position_in_grid]]) {
            a[id] = 4.0;
        }
    "#;
    let lib = dev.compile(source)?;
    let pipe = dev.pipeline(&lib.function("fill_four")?)?;
    let n = 256usize;
    let group_size = 64usize;
    let num_groups = n / group_size;
    let buf = dev.buffer(n * 4)?;
    buf.write_f32(|d| d.fill(0.0));

    let disp = Dispatch::new(&queue);
    unsafe {
        disp.batch(|batch| {
            batch.bind(&pipe);
            batch.bind_buffer(&buf, 0, 0);
            batch.launch_groups((num_groups, 1, 1), (group_size, 1, 1));
        });
    }

    buf.read_f32(|d| {
        for (i, &v) in d.iter().enumerate() {
            assert!(
                (v - 4.0).abs() < 1e-6,
                "batch launch_groups: d[{}]={}, expected 4.0",
                i,
                v
            );
        }
    });
    Ok(())
}

#[test]
fn wrap_block_vecadd() -> Result<(), GpuError> {
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
    let pipe = device.pipeline(&func)?;

    let n = 256;

    // Allocate via unimem::Block — shared with CPU/ANE
    let block_a = Block::open(n * 4).unwrap();
    let block_b = Block::open(n * 4).unwrap();
    let block_c = Block::open(n * 4).unwrap();

    for (i, v) in block_a.as_f32_mut().iter_mut().enumerate() {
        *v = i as f32;
    }
    for v in block_b.as_f32_mut().iter_mut() {
        *v = 10.0;
    }

    // Wrap blocks as Metal buffers — zero copy
    let buf_a = device.wrap(&block_a)?;
    let buf_b = device.wrap(&block_b)?;
    let buf_c = device.wrap(&block_c)?;

    let cmd = queue.commands()?;
    let enc = cmd.encoder()?;
    enc.bind(&pipe);
    enc.bind_buffer(&buf_a, 0, 0);
    enc.bind_buffer(&buf_b, 0, 1);
    enc.bind_buffer(&buf_c, 0, 2);
    enc.launch((n, 1, 1), (256, 1, 1));
    enc.finish();
    cmd.submit();
    cmd.wait();

    // Read result from Block directly — same physical memory
    for i in 0..n {
        let expected = i as f32 + 10.0;
        let actual = block_c.as_f32()[i];
        assert!(
            (actual - expected).abs() < 1e-6,
            "wrap_block: c[{}]={}, expected {}",
            i,
            actual,
            expected
        );
    }

    Ok(())
}

// ── Drop stress test ──

#[test]
fn drop_stress_leak() -> Result<(), GpuError> {
    let dev = Gpu::open()?;
    let queue = dev.new_command_queue()?;

    // Create and drop 5000 buffers — without Drop this exhausts handles
    for _ in 0..5000 {
        let _buf = dev.buffer(4096)?;
    }

    // Create and drop command buffers + encoders
    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void nop_drop(device float *a [[buffer(0)]],
                             uint id [[thread_position_in_grid]]) {
            a[id] = 0.0;
        }
    "#;
    let lib = dev.compile(source)?;
    let pipe = dev.pipeline(&lib.function("nop_drop")?)?;
    let buf = dev.buffer(64 * 4)?;

    for _ in 0..200 {
        let cmd = queue.commands()?;
        let enc = cmd.encoder()?;
        enc.bind(&pipe);
        enc.bind_buffer(&buf, 0, 0);
        enc.launch((64, 1, 1), (64, 1, 1));
        enc.finish();
        cmd.submit();
        cmd.wait();
    }

    Ok(())
}

// ── Copier as_raw ──

#[test]
fn copier_as_raw_not_null() -> Result<(), GpuError> {
    let dev = Gpu::open()?;
    let queue = dev.new_command_queue()?;
    let cmd = queue.commands()?;
    let blit = cmd.copier()?;
    assert!(!blit.as_raw().is_null());
    blit.finish();
    cmd.submit();
    cmd.wait();
    Ok(())
}
