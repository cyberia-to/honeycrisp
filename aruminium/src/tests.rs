//! Unit tests for aruminium

#[cfg(test)]
mod fp16_tests {
    use crate::{cast_f16_f32, cast_f32_f16, f32_to_fp16, fp16_to_f32};

    #[test]
    fn zero_round_trip() {
        assert_eq!(fp16_to_f32(f32_to_fp16(0.0)), 0.0);
    }

    #[test]
    fn negative_zero_round_trip() {
        let nz = f32_to_fp16(-0.0);
        let back = fp16_to_f32(nz);
        assert!(back == 0.0 || back == -0.0);
    }

    #[test]
    fn one_round_trip() {
        let h = f32_to_fp16(1.0);
        assert_eq!(fp16_to_f32(h), 1.0);
    }

    #[test]
    fn negative_one_round_trip() {
        let h = f32_to_fp16(-1.0);
        assert_eq!(fp16_to_f32(h), -1.0);
    }

    #[test]
    fn infinity() {
        let h = f32_to_fp16(f32::INFINITY);
        assert_eq!(fp16_to_f32(h), f32::INFINITY);
    }

    #[test]
    fn negative_infinity() {
        let h = f32_to_fp16(f32::NEG_INFINITY);
        assert_eq!(fp16_to_f32(h), f32::NEG_INFINITY);
    }

    #[test]
    fn nan_round_trip() {
        let h = f32_to_fp16(f32::NAN);
        assert!(fp16_to_f32(h).is_nan());
    }

    #[test]
    fn small_values() {
        // fp16 max is 65504, smallest normal ~6.1e-5
        for &v in &[0.5, 0.25, 0.1, 100.0, 1000.0, 65504.0] {
            let h = f32_to_fp16(v);
            let back = fp16_to_f32(h);
            let rel_err = ((back - v) / v).abs();
            assert!(
                rel_err < 0.002,
                "round-trip error for {}: got {}, err={}",
                v,
                back,
                rel_err
            );
        }
    }

    #[test]
    fn overflow_to_infinity() {
        // values > 65504 should clamp to infinity
        let h = f32_to_fp16(100000.0);
        assert_eq!(fp16_to_f32(h), f32::INFINITY);
    }

    #[test]
    fn underflow_to_zero() {
        // very small values should flush to zero
        let h = f32_to_fp16(1e-10);
        assert_eq!(fp16_to_f32(h), 0.0);
    }

    #[test]
    fn subnormal_fp16() {
        // smallest fp16 subnormal = 2^(-24) ≈ 5.96e-8
        // a value like 2^(-14)/1024 = 2^(-24)
        let h: u16 = 0x0001; // smallest subnormal
        let v = fp16_to_f32(h);
        assert!(v > 0.0 && v < 1e-6, "subnormal fp16: {}", v);
    }

    #[test]
    fn bulk_cast_f16_f32_matches_scalar() {
        let src: Vec<u16> = (0..1000).collect();
        let mut bulk = vec![0.0f32; 1000];
        cast_f16_f32(&mut bulk, &src);
        for (i, &h) in src.iter().enumerate() {
            let scalar = fp16_to_f32(h);
            assert!(
                (bulk[i] - scalar).abs() < 1e-10 || (bulk[i].is_nan() && scalar.is_nan()),
                "mismatch at {}: bulk={} scalar={}",
                i,
                bulk[i],
                scalar
            );
        }
    }

    #[test]
    fn bulk_cast_f32_f16_matches_scalar() {
        let src: Vec<f32> = (0..1000).map(|i| i as f32 * 0.1).collect();
        let mut bulk = vec![0u16; 1000];
        cast_f32_f16(&mut bulk, &src);
        for (i, &v) in src.iter().enumerate() {
            let scalar = f32_to_fp16(v);
            assert_eq!(
                bulk[i], scalar,
                "mismatch at {}: bulk={} scalar={}",
                i, bulk[i], scalar
            );
        }
    }

    #[test]
    fn bulk_tail_handling() {
        // test sizes that don't divide evenly by 32 or 8
        for n in [0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 100] {
            let src: Vec<u16> = (0..n as u16).collect();
            let mut dst = vec![0.0f32; n];
            cast_f16_f32(&mut dst, &src);
            for i in 0..n {
                let expected = fp16_to_f32(src[i]);
                assert!(
                    (dst[i] - expected).abs() < 1e-10 || (dst[i].is_nan() && expected.is_nan()),
                    "tail fail at n={} i={}",
                    n,
                    i
                );
            }
        }
    }

    #[test]
    fn all_u16_round_trip_no_panic() {
        // verify no panic for all 65536 possible fp16 values
        for i in 0..=u16::MAX {
            let f = fp16_to_f32(i);
            let _ = f32_to_fp16(f);
        }
    }

    #[test]
    fn tiny_exponents_no_panic() {
        // exponents -20..-24 previously caused shift >= 32 panic in debug
        // on NEON these may produce valid fp16 subnormals; on soft they flush to zero
        // the key invariant: no panic, and round-trip is consistent
        for exp in -30..=-15i32 {
            if (127 + exp) < 0 {
                continue;
            }
            let f32_bits = ((127 + exp) as u32) << 23;
            let v = f32::from_bits(f32_bits);
            let h = f32_to_fp16(v); // must not panic
            let back = fp16_to_f32(h);
            // back should be close to v or zero (flushed)
            assert!(
                back == 0.0 || (back - v).abs() / v.max(1e-30) < 1.0,
                "exp {}: v={} h={} back={}",
                exp,
                v,
                h,
                back
            );
        }
    }
}

#[cfg(test)]
mod error_tests {
    use crate::GpuError;

    #[test]
    fn error_display_all_variants() {
        let variants: Vec<GpuError> = vec![
            GpuError::DeviceNotFound,
            GpuError::BufferCreationFailed("test".into()),
            GpuError::LibraryCompilationFailed("bad msl".into()),
            GpuError::FunctionNotFound("missing".into()),
            GpuError::PipelineCreationFailed("fail".into()),
            GpuError::CommandBufferError("oops".into()),
            GpuError::EncoderCreationFailed,
            GpuError::QueueCreationFailed,
            GpuError::TextureCreationFailed("tex".into()),
        ];
        for v in &variants {
            let s = format!("{}", v);
            assert!(!s.is_empty(), "empty display for {:?}", v);
        }
    }

    #[test]
    fn error_is_error_trait() {
        let e: Box<dyn std::error::Error> = Box::new(GpuError::DeviceNotFound);
        assert!(!e.to_string().is_empty());
    }
}

#[cfg(test)]
mod device_tests {
    use crate::Gpu;

    #[test]
    fn open_works() {
        let dev = Gpu::open().unwrap();
        let name = dev.name();
        assert!(!name.is_empty());
        assert!(name.len() > 5, "name too short: {}", name);
        assert!(
            name.starts_with("Apple"),
            "expected name starting with 'Apple', got: {}",
            name
        );
        assert!(dev.max_buffer_length() > 1024 * 1024);
        assert!(dev.recommended_max_working_set_size() > 1_000_000);
        assert!(!dev.as_raw().is_null());
    }

    #[test]
    fn buffer_create_and_access() {
        let dev = Gpu::open().unwrap();
        let buf = dev.buffer(1024).unwrap();
        assert_eq!(buf.size(), 1024);
        assert!(buf.is_shared());
        assert!(!buf.as_raw().is_null());
        buf.write(|d| d[0] = 42);
        buf.read(|d| assert_eq!(d[0], 42));
    }

    #[test]
    fn buffer_read() {
        let dev = Gpu::open().unwrap();
        let data = vec![1u8, 2, 3, 4];
        let buf = dev.buffer_with_data(&data).unwrap();
        buf.read(|d| {
            assert_eq!(&d[..4], &[1, 2, 3, 4]);
        });
    }

    #[test]
    fn private_buffer_not_shared() {
        let dev = Gpu::open().unwrap();
        let buf = dev.buffer_private(1024).unwrap();
        assert!(!buf.is_shared());
    }

    #[test]
    #[should_panic(expected = "private buffer")]
    fn private_buffer_read_panics() {
        let dev = Gpu::open().unwrap();
        let buf = dev.buffer_private(1024).unwrap();
        buf.read(|_| {});
    }

    #[test]
    fn shader_compile_and_function() {
        let dev = Gpu::open().unwrap();
        let src = r#"
            #include <metal_stdlib>
            kernel void test_fn(device float *a [[buffer(0)]],
                               uint id [[thread_position_in_grid]]) {
                a[id] = 1.0;
            }
        "#;
        let lib = dev.compile(src).unwrap();
        assert!(!lib.as_raw().is_null());
        let names = lib.function_names();
        assert!(names.contains(&"test_fn".to_string()));
        let func = lib.function("test_fn").unwrap();
        assert_eq!(func.name(), "test_fn");
        assert!(!func.as_raw().is_null());
    }

    #[test]
    fn shader_compile_error() {
        let dev = Gpu::open().unwrap();
        let result = dev.compile("not valid msl!!!");
        assert!(result.is_err());
    }

    #[test]
    fn function_not_found() {
        let dev = Gpu::open().unwrap();
        let src = r#"
            #include <metal_stdlib>
            kernel void exists(device float *a [[buffer(0)]],
                              uint id [[thread_position_in_grid]]) { a[id] = 0; }
        "#;
        let lib = dev.compile(src).unwrap();
        let result = lib.function("does_not_exist");
        assert!(result.is_err());
    }

    #[test]
    fn pipeline_properties() {
        let dev = Gpu::open().unwrap();
        let src = r#"
            #include <metal_stdlib>
            kernel void k(device float *a [[buffer(0)]],
                         uint id [[thread_position_in_grid]]) { a[id] = 0; }
        "#;
        let lib = dev.compile(src).unwrap();
        let func = lib.function("k").unwrap();
        let pipe = dev.pipeline(&func).unwrap();
        assert!(pipe.max_total_threads_per_threadgroup() > 1);
        assert!(pipe.thread_execution_width() > 1);
        assert!(!pipe.as_raw().is_null());
    }

    #[test]
    fn gpu_timing() {
        let dev = Gpu::open().unwrap();
        let queue = dev.new_command_queue().unwrap();
        assert!(!queue.as_raw().is_null());
        let src = r#"
            #include <metal_stdlib>
            kernel void k(device float *a [[buffer(0)]],
                         uint id [[thread_position_in_grid]]) { a[id] = 0; }
        "#;
        let lib = dev.compile(src).unwrap();
        let pipe = dev.pipeline(&lib.function("k").unwrap()).unwrap();
        let buf = dev.buffer(256 * 4).unwrap();

        let cmd = queue.commands().unwrap();
        assert!(!cmd.as_raw().is_null());
        let enc = cmd.encoder().unwrap();
        assert!(!enc.as_raw().is_null());
        enc.bind(&pipe);
        enc.bind_buffer(&buf, 0, 0);
        enc.launch((256, 1, 1), (64, 1, 1));
        enc.finish();
        cmd.submit();
        cmd.wait();

        assert!(cmd.gpu_time() > 0.0);
        assert!(cmd.gpu_time() < 1.0); // should be microseconds, not seconds
    }

    #[test]
    fn sync_primitives() {
        let dev = Gpu::open().unwrap();
        let fence = dev.fence().unwrap();
        assert!(!fence.as_raw().is_null());
        let event = dev.event().unwrap();
        assert!(!event.as_raw().is_null());
        let se = dev.shared_event().unwrap();
        assert!(!se.as_raw().is_null());
        assert_eq!(se.signaled_value(), 0);
    }

    #[test]
    fn buffer_zero_size_rejected() {
        let dev = Gpu::open().unwrap();
        assert!(dev.buffer(0).is_err());
        assert!(dev.buffer_private(0).is_err());
    }

    #[test]
    fn texture_create_and_properties() {
        use crate::ffi::*;
        let dev = Gpu::open().unwrap();

        // Create a texture descriptor for 16x16 RGBA8
        unsafe {
            let cls = objc_getClass(c"MTLTextureDescriptor".as_ptr()) as ObjcId;
            let desc = msg0(cls, sel_registerName(c"new".as_ptr()));
            assert!(!desc.is_null());

            // setTextureType: MTLTextureType2D = 2
            type SetU = unsafe extern "C" fn(ObjcId, ObjcSel, NSUInteger);
            let set_u: SetU = std::mem::transmute(objc_msgSend as *const std::ffi::c_void);
            set_u(desc, sel_registerName(c"setTextureType:".as_ptr()), 2);
            set_u(
                desc,
                sel_registerName(c"setPixelFormat:".as_ptr()),
                MTLPixelFormatRGBA8Unorm,
            );
            set_u(desc, sel_registerName(c"setWidth:".as_ptr()), 16);
            set_u(desc, sel_registerName(c"setHeight:".as_ptr()), 16);

            let tex = dev.texture(desc).unwrap();
            assert!(!tex.as_raw().is_null());
            assert_eq!(tex.width(), 16);
            assert_eq!(tex.height(), 16);
            assert_eq!(tex.depth(), 1);
            assert_eq!(tex.pixel_format(), MTLPixelFormatRGBA8Unorm);

            release(desc);
        }
    }

    #[test]
    fn command_buffer_status_constants() {
        use crate::Commands;
        assert_eq!(Commands::STATUS_NOT_ENQUEUED, 0);
        assert_eq!(Commands::STATUS_COMPLETED, 4);
        assert_eq!(Commands::STATUS_ERROR, 5);
    }
}

#[cfg(test)]
mod sharing_tests {
    //! Tests for `Gpu::from_raw` / `Queue::from_raw_shared` — the wgpu
    //! device-sharing surface. Verifies retain/release semantics by
    //! allocating from both wrappers, dropping in arbitrary order, and
    //! relying on the existing leak/double-free coverage of the Drop impls.
    use crate::{Gpu, Queue};

    #[test]
    fn from_raw_round_trip() {
        let g1 = Gpu::open().unwrap();
        let raw = g1.as_raw();
        let g2 = unsafe { Gpu::from_raw(raw) }.unwrap();

        // Same underlying device: identical name + unified-memory flag.
        assert_eq!(g1.name(), g2.name());
        assert_eq!(g1.has_unified_memory(), g2.has_unified_memory());
        assert_eq!(g1.as_raw(), g2.as_raw());

        // Both wrappers can independently allocate on the shared device.
        let b1 = g1.buffer(1024).unwrap();
        let b2 = g2.buffer(1024).unwrap();
        drop(b1);
        drop(b2);

        // Drop the wrapper first, then the original — original retain
        // count must still be > 0 after g2's release.
        drop(g2);
        // g1 must still work after g2 was dropped.
        let _b3 = g1.buffer(1024).unwrap();
        drop(g1);
    }

    #[test]
    fn from_raw_null_returns_error() {
        assert!(unsafe { Gpu::from_raw(std::ptr::null_mut()) }.is_err());
    }

    #[test]
    fn queue_from_raw_shared_works() {
        let g = Gpu::open().unwrap();
        let q1 = g.new_command_queue().unwrap();
        let q_raw = q1.as_raw();
        let q2 = unsafe { Queue::from_raw_shared(q_raw) };

        assert_eq!(q1.as_raw(), q2.as_raw());

        // Both wrappers must produce valid command buffers.
        let _c1 = q1.commands().unwrap();
        let _c2 = q2.commands().unwrap();

        // Drop one wrapper; the other must remain usable.
        drop(q2);
        let _c3 = q1.commands().unwrap();
    }

    #[test]
    fn from_raw_with_queue_pair() {
        let g0 = Gpu::open().unwrap();
        let q0 = g0.new_command_queue().unwrap();
        let (g, q) = unsafe { Gpu::from_raw_with_queue(g0.as_raw(), q0.as_raw()) }.unwrap();
        assert_eq!(g.as_raw(), g0.as_raw());
        assert_eq!(q.as_raw(), q0.as_raw());
        let _ = q.commands().unwrap();
    }
}
