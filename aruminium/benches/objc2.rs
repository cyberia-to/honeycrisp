use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use std::time::Instant;

use super::shaders::{NOOP as NOOP_SRC, SAXPY as SAXPY_SRC};

fn get_device() -> Retained<ProtocolObject<dyn MTLDevice>> {
    MTLCreateSystemDefaultDevice().expect("no metal device")
}

pub fn device_discovery(iters: usize) -> f64 {
    let _ = get_device();
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = get_device();
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

pub fn buffer_creation(iters: usize, size: usize) -> f64 {
    let dev = get_device();
    for _ in 0..3 {
        let _ = dev
            .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
            .unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = dev
            .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
            .unwrap();
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

pub fn shader_compile(iters: usize) -> f64 {
    let dev = get_device();
    let src = NSString::from_str(SAXPY_SRC);
    let _ = dev.newLibraryWithSource_options_error(&src, None).unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = dev.newLibraryWithSource_options_error(&src, None).unwrap();
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

pub fn encode_overhead(iters: usize) -> f64 {
    let dev = get_device();
    let queue = dev.newCommandQueue().unwrap();
    let src = NSString::from_str(NOOP_SRC);
    let lib = dev.newLibraryWithSource_options_error(&src, None).unwrap();
    let fname = NSString::from_str("noop");
    let func = lib.newFunctionWithName(&fname).unwrap();
    let pipe = dev
        .newComputePipelineStateWithFunction_error(&func)
        .unwrap();
    let buf = dev
        .newBufferWithLength_options(256 * 4, MTLResourceOptions::StorageModeShared)
        .unwrap();
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

    for _ in 0..10 {
        let cmd = queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pipe);
        unsafe { enc.setBuffer_offset_atIndex(Some(&buf), 0, 0) };
        enc.dispatchThreads_threadsPerThreadgroup(grid, group);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    let t0 = Instant::now();
    let mut last = None;
    for _ in 0..iters {
        let cmd = queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pipe);
        unsafe { enc.setBuffer_offset_atIndex(Some(&buf), 0, 0) };
        enc.dispatchThreads_threadsPerThreadgroup(grid, group);
        enc.endEncoding();
        cmd.commit();
        last = Some(cmd);
    }
    if let Some(cmd) = last {
        cmd.waitUntilCompleted();
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

pub fn dispatch_overhead(iters: usize) -> f64 {
    let dev = get_device();
    let queue = dev.newCommandQueue().unwrap();
    let src = NSString::from_str(NOOP_SRC);
    let lib = dev.newLibraryWithSource_options_error(&src, None).unwrap();
    let fname = NSString::from_str("noop");
    let func = lib.newFunctionWithName(&fname).unwrap();
    let pipe = dev
        .newComputePipelineStateWithFunction_error(&func)
        .unwrap();
    let buf = dev
        .newBufferWithLength_options(256 * 4, MTLResourceOptions::StorageModeShared)
        .unwrap();

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

    for _ in 0..10 {
        let cmd = queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pipe);
        unsafe { enc.setBuffer_offset_atIndex(Some(&buf), 0, 0) };
        enc.dispatchThreads_threadsPerThreadgroup(grid, group);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        let cmd = queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pipe);
        unsafe { enc.setBuffer_offset_atIndex(Some(&buf), 0, 0) };
        enc.dispatchThreads_threadsPerThreadgroup(grid, group);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

pub fn large_compute(iters: usize, n: usize) -> f64 {
    let dev = get_device();
    let queue = dev.newCommandQueue().unwrap();
    let src = NSString::from_str(SAXPY_SRC);
    let lib = dev.newLibraryWithSource_options_error(&src, None).unwrap();
    let fname = NSString::from_str("saxpy");
    let func = lib.newFunctionWithName(&fname).unwrap();
    let pipe = dev
        .newComputePipelineStateWithFunction_error(&func)
        .unwrap();
    let buf_x = dev
        .newBufferWithLength_options(n * 4, MTLResourceOptions::StorageModeShared)
        .unwrap();
    let buf_y = dev
        .newBufferWithLength_options(n * 4, MTLResourceOptions::StorageModeShared)
        .unwrap();

    // fill
    unsafe {
        let ptr = buf_x.contents().as_ptr() as *mut f32;
        for i in 0..n {
            *ptr.add(i) = i as f32;
        }
        let ptr = buf_y.contents().as_ptr() as *mut f32;
        for i in 0..n {
            *ptr.add(i) = 1.0;
        }
    }

    let a_val: f32 = 2.0;
    let a_bytes = a_val.to_le_bytes();
    let grid = MTLSize {
        width: n,
        height: 1,
        depth: 1,
    };
    let group = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };

    for _ in 0..3 {
        let cmd = queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pipe);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&buf_x), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&buf_y), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(a_bytes.as_ptr() as *mut _).unwrap(),
                a_bytes.len(),
                2,
            );
        }
        enc.dispatchThreads_threadsPerThreadgroup(grid, group);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    // reset y
    unsafe {
        let ptr = buf_y.contents().as_ptr() as *mut f32;
        for i in 0..n {
            *ptr.add(i) = 1.0;
        }
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        let cmd = queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pipe);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&buf_x), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&buf_y), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(a_bytes.as_ptr() as *mut _).unwrap(),
                a_bytes.len(),
                2,
            );
        }
        enc.dispatchThreads_threadsPerThreadgroup(grid, group);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

/// Batch encode: N dispatches per cmd buffer, amortized cost
pub fn batch_encode(batch_size: usize, iters: usize) -> f64 {
    let dev = get_device();
    let queue = dev.newCommandQueue().unwrap();
    let src = NSString::from_str(NOOP_SRC);
    let lib = dev.newLibraryWithSource_options_error(&src, None).unwrap();
    let fname = NSString::from_str("noop");
    let func = lib.newFunctionWithName(&fname).unwrap();
    let pipe = dev
        .newComputePipelineStateWithFunction_error(&func)
        .unwrap();
    let buf = dev
        .newBufferWithLength_options(256 * 4, MTLResourceOptions::StorageModeShared)
        .unwrap();
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

    for _ in 0..3 {
        let cmd = queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        for _ in 0..batch_size {
            enc.setComputePipelineState(&pipe);
            unsafe { enc.setBuffer_offset_atIndex(Some(&buf), 0, 0) };
            enc.dispatchThreads_threadsPerThreadgroup(grid, group);
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        let cmd = queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        for _ in 0..batch_size {
            enc.setComputePipelineState(&pipe);
            unsafe { enc.setBuffer_offset_atIndex(Some(&buf), 0, 0) };
            enc.dispatchThreads_threadsPerThreadgroup(grid, group);
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
    t0.elapsed().as_secs_f64() / (iters * batch_size) as f64
}

/// Inference sim: 3 kernels × N layers, single batch
pub fn inference_sim(layers: usize, iters: usize) -> f64 {
    let dev = get_device();
    let queue = dev.newCommandQueue().unwrap();
    let src = NSString::from_str(
        r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void matmul_k(device float *a [[buffer(0)]], device float *b [[buffer(1)]],
                             device float *c [[buffer(2)]], uint id [[thread_position_in_grid]]) {
            c[id] = a[id] * b[id];
        }
        kernel void relu_k(device float *x [[buffer(0)]], uint id [[thread_position_in_grid]]) {
            x[id] = max(x[id], 0.0f);
        }
        kernel void add_k(device float *a [[buffer(0)]], device float *b [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
            a[id] = a[id] + b[id];
        }
    "#,
    );
    let lib = dev.newLibraryWithSource_options_error(&src, None).unwrap();
    let matmul = dev
        .newComputePipelineStateWithFunction_error(
            &lib.newFunctionWithName(&NSString::from_str("matmul_k"))
                .unwrap(),
        )
        .unwrap();
    let relu = dev
        .newComputePipelineStateWithFunction_error(
            &lib.newFunctionWithName(&NSString::from_str("relu_k"))
                .unwrap(),
        )
        .unwrap();
    let add = dev
        .newComputePipelineStateWithFunction_error(
            &lib.newFunctionWithName(&NSString::from_str("add_k"))
                .unwrap(),
        )
        .unwrap();
    let n: usize = 4096;
    let buf_a = dev
        .newBufferWithLength_options(n * 4, MTLResourceOptions::StorageModeShared)
        .unwrap();
    let buf_b = dev
        .newBufferWithLength_options(n * 4, MTLResourceOptions::StorageModeShared)
        .unwrap();
    let buf_c = dev
        .newBufferWithLength_options(n * 4, MTLResourceOptions::StorageModeShared)
        .unwrap();
    let grid = MTLSize {
        width: n,
        height: 1,
        depth: 1,
    };
    let group = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };

    for _ in 0..3 {
        let cmd = queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        for _ in 0..layers {
            enc.setComputePipelineState(&matmul);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_a), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&buf_b), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&buf_c), 0, 2);
            }
            enc.dispatchThreads_threadsPerThreadgroup(grid, group);
            enc.setComputePipelineState(&relu);
            unsafe { enc.setBuffer_offset_atIndex(Some(&buf_c), 0, 0) };
            enc.dispatchThreads_threadsPerThreadgroup(grid, group);
            enc.setComputePipelineState(&add);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_a), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&buf_c), 0, 1);
            }
            enc.dispatchThreads_threadsPerThreadgroup(grid, group);
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        let cmd = queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        for _ in 0..layers {
            enc.setComputePipelineState(&matmul);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_a), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&buf_b), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&buf_c), 0, 2);
            }
            enc.dispatchThreads_threadsPerThreadgroup(grid, group);
            enc.setComputePipelineState(&relu);
            unsafe { enc.setBuffer_offset_atIndex(Some(&buf_c), 0, 0) };
            enc.dispatchThreads_threadsPerThreadgroup(grid, group);
            enc.setComputePipelineState(&add);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_a), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&buf_c), 0, 1);
            }
            enc.dispatchThreads_threadsPerThreadgroup(grid, group);
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
    t0.elapsed().as_secs_f64() / iters as f64
}
