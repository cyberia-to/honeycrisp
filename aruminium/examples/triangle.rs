//! Minimum useful raster example — render a triangle into an offscreen
//! BGRA8 texture and verify pixels via getBytes:.
//!
//! Confirms:
//!   - `Gpu::render_target` produces a render-capable color texture
//!   - `Gpu::render_pipeline` compiles a vertex+fragment program
//!   - `Commands::render_encoder` records a clear + draw
//!   - the GPU actually wrote color into the target

use aruminium::ffi::*;
use aruminium::{
    ColorAttachmentDesc, Gpu, GpuError, PrimitiveType, RenderPassDescriptor, RenderPipelineSpec,
};

const SHADER: &str = r#"
    #include <metal_stdlib>
    using namespace metal;

    struct VOut {
        float4 pos [[position]];
        float4 color;
    };

    vertex VOut vmain(uint vid [[vertex_id]]) {
        float2 verts[3] = {
            float2(-0.8, -0.8),
            float2( 0.8, -0.8),
            float2( 0.0,  0.8),
        };
        float3 colors[3] = {
            float3(1.0, 0.0, 0.0),
            float3(0.0, 1.0, 0.0),
            float3(0.0, 0.0, 1.0),
        };
        VOut o;
        o.pos = float4(verts[vid], 0.0, 1.0);
        o.color = float4(colors[vid], 1.0);
        return o;
    }

    fragment float4 fmain(VOut v [[stage_in]]) {
        return v.color;
    }
"#;

fn main() -> Result<(), GpuError> {
    let device = Gpu::open()?;
    println!("Device: {}", device.name());
    let queue = device.new_command_queue()?;

    let lib = device.compile(SHADER)?;
    let vfn = lib.function("vmain")?;
    let ffn = lib.function("fmain")?;

    let spec = RenderPipelineSpec::color(MTLPixelFormatBGRA8Unorm);
    let pipeline = device.render_pipeline(&vfn, &ffn, &spec)?;
    println!(
        "RenderPipeline OK ({} color attachments, sample_count={})",
        pipeline.color_attachments(),
        pipeline.sample_count(),
    );

    let w: u32 = 256;
    let h: u32 = 256;
    let target = device.render_target(w, h, MTLPixelFormatBGRA8Unorm)?;

    let mut pass = RenderPassDescriptor::new();
    pass.color_attachment(
        0,
        ColorAttachmentDesc::clear(&target, [0.05, 0.05, 0.10, 1.0]),
    );

    let cmd = queue.commands()?;
    {
        let enc = cmd.render_encoder(&pass)?;
        enc.bind(&pipeline);
        enc.set_viewport(0.0, 0.0, w as f64, h as f64, 0.0, 1.0);
        enc.draw(PrimitiveType::Triangle, 0, 3);
        enc.end();
    }
    cmd.submit();
    cmd.wait();

    // Read back the rendered pixels to verify the triangle was rasterized.
    // Render-target storage is Private, so go through a blit into a shared
    // staging buffer for CPU readback.
    let bytes_per_pixel = 4usize;
    let bytes_per_row = w as usize * bytes_per_pixel;
    let total = bytes_per_row * h as usize;
    let staging = device.buffer(total)?;

    let cmd2 = queue.commands()?;
    unsafe {
        let blit = aruminium::ffi::msg0(cmd2.as_raw(), aruminium::ffi::SEL_blitCommandEncoder());
        aruminium::ffi::retain(blit);
        let sel = sel_registerName(
            c"copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:".as_ptr(),
        );
        type F = unsafe extern "C" fn(
            ObjcId,
            ObjcSel,
            ObjcId,
            NSUInteger,
            NSUInteger,
            MTLOrigin,
            MTLSize,
            ObjcId,
            NSUInteger,
            NSUInteger,
            NSUInteger,
        );
        let f: F = std::mem::transmute(objc_msgSend as *const std::ffi::c_void);
        f(
            blit,
            sel,
            target.as_raw(),
            0,
            0,
            MTLOrigin { x: 0, y: 0, z: 0 },
            MTLSize {
                width: w as usize,
                height: h as usize,
                depth: 1,
            },
            staging.as_raw(),
            0,
            bytes_per_row,
            total,
        );
        msg0_void(blit, SEL_endEncoding());
        release(blit);
    }
    cmd2.submit();
    cmd2.wait();

    // Centroid of the triangle should not be the clear color.
    let cx = (w / 2) as usize;
    let cy = (h / 2) as usize;
    let mut center_bgra = [0u8; 4];
    staging.read(|d| {
        let i = cy * bytes_per_row + cx * bytes_per_pixel;
        center_bgra.copy_from_slice(&d[i..i + 4]);
    });
    let clear_bgra = [
        (0.10_f64 * 255.0).round() as u8,
        (0.05_f64 * 255.0).round() as u8,
        (0.05_f64 * 255.0).round() as u8,
        255,
    ];
    if center_bgra == clear_bgra {
        println!(
            "FAIL: center pixel is still the clear color {:?}",
            center_bgra
        );
        return Ok(());
    }
    println!(
        "PASS: rendered {}x{} triangle, center pixel BGRA = {:?}",
        w, h, center_bgra
    );

    Ok(())
}
