//! aruminium driver benchmarks — measures Metal.framework overhead

use aruminium::Gpu;
use std::time::Instant;

use super::shaders::{NOOP as NOOP_SRC, SAXPY as SAXPY_SRC};

// ── basic: device, buffer, shader ──

pub fn device_discovery(iters: usize) -> f64 {
    let _ = Gpu::open().unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = Gpu::open().unwrap();
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

pub fn buffer_creation(iters: usize, size: usize) -> f64 {
    let dev = Gpu::open().unwrap();
    for _ in 0..3 {
        let _ = dev.buffer(size).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = dev.buffer(size).unwrap();
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

pub fn shader_compile(iters: usize) -> f64 {
    let dev = Gpu::open().unwrap();
    let _ = dev.compile(SAXPY_SRC).unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = dev.compile(SAXPY_SRC).unwrap();
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

// ── encode: overhead, unchecked, batch ──

pub fn encode_overhead(iters: usize) -> f64 {
    let dev = Gpu::open().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let lib = dev.compile(NOOP_SRC).unwrap();
    let pipe = dev.pipeline(&lib.function("noop").unwrap()).unwrap();
    let buf = dev.buffer(256 * 4).unwrap();
    warmup(&queue, &pipe, &buf);

    let t0 = Instant::now();
    let mut last_cmd = None;
    for _ in 0..iters {
        let cmd = queue.commands().unwrap();
        let enc = cmd.encoder().unwrap();
        enc.bind(&pipe);
        enc.bind_buffer(&buf, 0, 0);
        enc.launch((256, 1, 1), (64, 1, 1));
        enc.finish();
        cmd.submit();
        last_cmd = Some(cmd);
    }
    if let Some(cmd) = last_cmd {
        cmd.wait();
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

pub fn encode_unchecked(iters: usize) -> f64 {
    let dev = Gpu::open().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let lib = dev.compile(NOOP_SRC).unwrap();
    let pipe = dev.pipeline(&lib.function("noop").unwrap()).unwrap();
    let buf = dev.buffer(256 * 4).unwrap();
    warmup(&queue, &pipe, &buf);

    let t0 = Instant::now();
    let mut last_cmd = None::<aruminium::Commands>;
    for _ in 0..iters {
        unsafe {
            let cmd = queue.commands_unchecked();
            let enc = cmd.encoder_unchecked();
            enc.bind(&pipe);
            enc.bind_buffer(&buf, 0, 0);
            enc.launch((256, 1, 1), (64, 1, 1));
            enc.finish();
            cmd.submit();
            last_cmd = Some(cmd);
        }
    }
    if let Some(cmd) = last_cmd {
        cmd.wait();
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

pub fn batch_encode(batch_size: usize, iters: usize) -> f64 {
    use aruminium::Dispatch;
    let dev = Gpu::open().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let lib = dev.compile(NOOP_SRC).unwrap();
    let pipe = dev.pipeline(&lib.function("noop").unwrap()).unwrap();
    let buf = dev.buffer(256 * 4).unwrap();
    let disp = Dispatch::new(&queue);

    for _ in 0..3 {
        unsafe {
            disp.batch(|b| {
                for _ in 0..batch_size {
                    b.bind(&pipe);
                    b.bind_buffer(&buf, 0, 0);
                    b.launch((256, 1, 1), (64, 1, 1));
                }
            });
        }
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe {
            disp.batch(|b| {
                for _ in 0..batch_size {
                    b.bind(&pipe);
                    b.bind_buffer(&buf, 0, 0);
                    b.launch((256, 1, 1), (64, 1, 1));
                }
            });
        }
    }
    t0.elapsed().as_secs_f64() / (iters * batch_size) as f64
}

pub fn encode_encoder(batch_size: usize, iters: usize) -> f64 {
    let dev = Gpu::open().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let lib = dev.compile(NOOP_SRC).unwrap();
    let pipe = dev.pipeline(&lib.function("noop").unwrap()).unwrap();
    let buf = dev.buffer(256 * 4).unwrap();

    for _ in 0..3 {
        let cmd = queue.commands().unwrap();
        let enc = cmd.encoder().unwrap();
        for _ in 0..batch_size {
            enc.bind(&pipe);
            enc.bind_buffer(&buf, 0, 0);
            enc.launch((256, 1, 1), (64, 1, 1));
        }
        enc.finish();
        cmd.submit();
        cmd.wait();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let cmd = queue.commands().unwrap();
        let enc = cmd.encoder().unwrap();
        for _ in 0..batch_size {
            enc.bind(&pipe);
            enc.bind_buffer(&buf, 0, 0);
            enc.launch((256, 1, 1), (64, 1, 1));
        }
        enc.finish();
        cmd.submit();
        cmd.wait();
    }
    t0.elapsed().as_secs_f64() / (iters * batch_size) as f64
}

// ── dispatch: overhead, unchecked, autoreleased, raw, IMP, pipelined ──

pub fn dispatch_overhead(iters: usize) -> f64 {
    let dev = Gpu::open().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let lib = dev.compile(NOOP_SRC).unwrap();
    let pipe = dev.pipeline(&lib.function("noop").unwrap()).unwrap();
    let buf = dev.buffer(256 * 4).unwrap();
    warmup(&queue, &pipe, &buf);

    let t0 = Instant::now();
    for _ in 0..iters {
        let cmd = queue.commands().unwrap();
        let enc = cmd.encoder().unwrap();
        enc.bind(&pipe);
        enc.bind_buffer(&buf, 0, 0);
        enc.launch((256, 1, 1), (64, 1, 1));
        enc.finish();
        cmd.submit();
        cmd.wait();
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

pub fn dispatch_unchecked(iters: usize) -> f64 {
    let dev = Gpu::open().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let lib = dev.compile(NOOP_SRC).unwrap();
    let pipe = dev.pipeline(&lib.function("noop").unwrap()).unwrap();
    let buf = dev.buffer(256 * 4).unwrap();
    warmup(&queue, &pipe, &buf);

    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe {
            let cmd = queue.commands_unchecked();
            let enc = cmd.encoder_unchecked();
            enc.bind(&pipe);
            enc.bind_buffer(&buf, 0, 0);
            enc.launch((256, 1, 1), (64, 1, 1));
            enc.finish();
            cmd.submit();
            cmd.wait();
        }
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

pub fn dispatch_autoreleased(iters: usize) -> f64 {
    let dev = Gpu::open().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let lib = dev.compile(NOOP_SRC).unwrap();
    let pipe = dev.pipeline(&lib.function("noop").unwrap()).unwrap();
    let buf = dev.buffer(256 * 4).unwrap();
    warmup(&queue, &pipe, &buf);

    let t0 = Instant::now();
    aruminium::autorelease_pool(|| {
        for _ in 0..iters {
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
    t0.elapsed().as_secs_f64() / iters as f64
}

pub fn dispatch_raw(iters: usize) -> f64 {
    use aruminium::ffi::*;
    use std::ffi::c_void;

    let dev = Gpu::open().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let lib = dev.compile(NOOP_SRC).unwrap();
    let pipe = dev.pipeline(&lib.function("noop").unwrap()).unwrap();
    let buf = dev.buffer(256 * 4).unwrap();
    warmup(&queue, &pipe, &buf);

    let q = queue.as_raw();
    let p = pipe.as_raw();
    let b = buf.as_raw();
    let (s_cb, s_ce) = (SEL_commandBuffer(), SEL_computeCommandEncoder());
    let (s_sp, s_sb) = (
        SEL_setComputePipelineState(),
        SEL_setBuffer_offset_atIndex(),
    );
    let (s_dt, s_ee) = (SEL_dispatchThreads(), SEL_endEncoding());
    let (s_cm, s_wt) = (SEL_commit(), SEL_waitUntilCompleted());

    type F0 = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
    type FV = unsafe extern "C" fn(ObjcId, ObjcSel);
    type F1 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId);
    type F3 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, NSUInteger, NSUInteger);
    type FD = unsafe extern "C" fn(ObjcId, ObjcSel, MTLSize, MTLSize);
    let f0: F0 = unsafe { std::mem::transmute(objc_msgSend as *const c_void) };
    let fv: FV = unsafe { std::mem::transmute(objc_msgSend as *const c_void) };
    let f1: F1 = unsafe { std::mem::transmute(objc_msgSend as *const c_void) };
    let f3: F3 = unsafe { std::mem::transmute(objc_msgSend as *const c_void) };
    let fd: FD = unsafe { std::mem::transmute(objc_msgSend as *const c_void) };
    let grid = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    let group = MTLSize {
        width: 64,
        height: 1,
        depth: 1,
    };

    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe {
            let cmd = f0(q, s_cb);
            objc_retainAutoreleasedReturnValue(cmd);
            let enc = f0(cmd, s_ce);
            objc_retainAutoreleasedReturnValue(enc);
            f1(enc, s_sp, p);
            f3(enc, s_sb, b, 0, 0);
            fd(enc, s_dt, grid, group);
            fv(enc, s_ee);
            fv(cmd, s_cm);
            fv(cmd, s_wt);
            objc_release(enc);
            objc_release(cmd);
        }
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

pub fn dispatch_imp(iters: usize) -> f64 {
    use aruminium::Dispatch;
    let dev = Gpu::open().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let lib = dev.compile(NOOP_SRC).unwrap();
    let pipe = dev.pipeline(&lib.function("noop").unwrap()).unwrap();
    let buf = dev.buffer(256 * 4).unwrap();
    let disp = Dispatch::new(&queue);
    for _ in 0..10 {
        unsafe { disp.dispatch(&pipe, &[(&buf, 0, 0)], (256, 1, 1), (64, 1, 1)) };
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe { disp.dispatch(&pipe, &[(&buf, 0, 0)], (256, 1, 1), (64, 1, 1)) };
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

pub fn dispatch_pipelined(iters: usize) -> f64 {
    use aruminium::Dispatch;
    let dev = Gpu::open().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let lib = dev.compile(NOOP_SRC).unwrap();
    let pipe = dev.pipeline(&lib.function("noop").unwrap()).unwrap();
    let buf = dev.buffer(256 * 4).unwrap();
    let disp = Dispatch::new(&queue);
    for _ in 0..10 {
        unsafe { disp.dispatch(&pipe, &[(&buf, 0, 0)], (256, 1, 1), (64, 1, 1)) };
    }

    let t0 = Instant::now();
    let mut prev = None;
    for _ in 0..iters {
        let future = unsafe {
            disp.batch_async(|batch| {
                batch.bind(&pipe);
                batch.bind_buffer(&buf, 0, 0);
                batch.launch((256, 1, 1), (64, 1, 1));
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
    t0.elapsed().as_secs_f64() / iters as f64
}

// ── compute: SAXPY, inference sim ──

pub fn large_compute(iters: usize, n: usize) -> f64 {
    let dev = Gpu::open().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let lib = dev.compile(SAXPY_SRC).unwrap();
    let pipe = dev.pipeline(&lib.function("saxpy").unwrap()).unwrap();
    let buf_x = dev.buffer(n * 4).unwrap();
    let buf_y = dev.buffer(n * 4).unwrap();
    buf_x.write_f32(|d| {
        for (i, v) in d.iter_mut().enumerate() {
            *v = i as f32;
        }
    });
    buf_y.write_f32(|d| d.fill(1.0));
    let a_bytes = 2.0f32.to_le_bytes();

    for _ in 0..3 {
        let cmd = queue.commands().unwrap();
        let enc = cmd.encoder().unwrap();
        enc.bind(&pipe);
        enc.bind_buffer(&buf_x, 0, 0);
        enc.bind_buffer(&buf_y, 0, 1);
        enc.push(&a_bytes, 2);
        enc.launch((n, 1, 1), (256, 1, 1));
        enc.finish();
        cmd.submit();
        cmd.wait();
    }
    buf_y.write_f32(|d| d.fill(1.0));

    let t0 = Instant::now();
    for _ in 0..iters {
        let cmd = queue.commands().unwrap();
        let enc = cmd.encoder().unwrap();
        enc.bind(&pipe);
        enc.bind_buffer(&buf_x, 0, 0);
        enc.bind_buffer(&buf_y, 0, 1);
        enc.push(&a_bytes, 2);
        enc.launch((n, 1, 1), (256, 1, 1));
        enc.finish();
        cmd.submit();
        cmd.wait();
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

pub fn large_compute_imp(iters: usize, n: usize) -> f64 {
    use aruminium::Dispatch;
    let dev = Gpu::open().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let lib = dev.compile(SAXPY_SRC).unwrap();
    let pipe = dev.pipeline(&lib.function("saxpy").unwrap()).unwrap();
    let buf_x = dev.buffer(n * 4).unwrap();
    let buf_y = dev.buffer(n * 4).unwrap();
    buf_x.write_f32(|d| {
        for (i, v) in d.iter_mut().enumerate() {
            *v = i as f32;
        }
    });
    buf_y.write_f32(|d| d.fill(1.0));
    let ab = 2.0f32.to_le_bytes();
    let disp = Dispatch::new(&queue);

    for _ in 0..3 {
        unsafe {
            disp.dispatch_with_bytes(
                &pipe,
                &[(&buf_x, 0, 0), (&buf_y, 0, 1)],
                &ab,
                2,
                (n, 1, 1),
                (256, 1, 1),
            );
        }
    }
    buf_y.write_f32(|d| d.fill(1.0));

    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe {
            disp.dispatch_with_bytes(
                &pipe,
                &[(&buf_x, 0, 0), (&buf_y, 0, 1)],
                &ab,
                2,
                (n, 1, 1),
                (256, 1, 1),
            );
        }
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

pub fn inference_sim(layers: usize, iters: usize) -> f64 {
    use aruminium::Dispatch;
    let dev = Gpu::open().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let src = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void matmul_k(device float *a [[buffer(0)]], device float *b [[buffer(1)]],
                             device float *c [[buffer(2)]], uint id [[thread_position_in_grid]]) { c[id] = a[id] * b[id]; }
        kernel void relu_k(device float *x [[buffer(0)]], uint id [[thread_position_in_grid]]) { x[id] = max(x[id], 0.0f); }
        kernel void add_k(device float *a [[buffer(0)]], device float *b [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) { a[id] = a[id] + b[id]; }
    "#;
    let lib = dev.compile(src).unwrap();
    let matmul = dev.pipeline(&lib.function("matmul_k").unwrap()).unwrap();
    let relu = dev.pipeline(&lib.function("relu_k").unwrap()).unwrap();
    let add = dev.pipeline(&lib.function("add_k").unwrap()).unwrap();
    let n = 4096usize;
    let buf_a = dev.buffer(n * 4).unwrap();
    let buf_b = dev.buffer(n * 4).unwrap();
    let buf_c = dev.buffer(n * 4).unwrap();
    let disp = Dispatch::new(&queue);

    for _ in 0..3 {
        unsafe {
            dispatch_layers(
                &disp, &matmul, &relu, &add, &buf_a, &buf_b, &buf_c, n, layers,
            );
        }
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe {
            dispatch_layers(
                &disp, &matmul, &relu, &add, &buf_a, &buf_b, &buf_c, n, layers,
            );
        }
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

// ── helpers ──

fn warmup(queue: &aruminium::Queue, pipe: &aruminium::Pipeline, buf: &aruminium::Buffer) {
    for _ in 0..10 {
        let cmd = queue.commands().unwrap();
        let enc = cmd.encoder().unwrap();
        enc.bind(pipe);
        enc.bind_buffer(buf, 0, 0);
        enc.launch((256, 1, 1), (64, 1, 1));
        enc.finish();
        cmd.submit();
        cmd.wait();
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn dispatch_layers(
    disp: &aruminium::Dispatch,
    matmul: &aruminium::Pipeline,
    relu: &aruminium::Pipeline,
    add: &aruminium::Pipeline,
    buf_a: &aruminium::Buffer,
    buf_b: &aruminium::Buffer,
    buf_c: &aruminium::Buffer,
    n: usize,
    layers: usize,
) {
    disp.batch(|b| {
        for _ in 0..layers {
            b.bind(matmul);
            b.bind_buffer(buf_a, 0, 0);
            b.bind_buffer(buf_b, 0, 1);
            b.bind_buffer(buf_c, 0, 2);
            b.launch((n, 1, 1), (256, 1, 1));
            b.bind(relu);
            b.bind_buffer(buf_c, 0, 0);
            b.launch((n, 1, 1), (256, 1, 1));
            b.bind(add);
            b.bind_buffer(buf_a, 0, 0);
            b.bind_buffer(buf_c, 0, 1);
            b.launch((n, 1, 1), (256, 1, 1));
        }
    });
}
