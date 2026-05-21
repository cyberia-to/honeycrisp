//! Textured quad — phase 2 demo.
//!
//! Builds a quad with a vertex buffer (Float2 pos + Float2 uv per vertex)
//! described by an MTLVertexDescriptor. Samples a small procedural texture
//! in the fragment shader. Uses indexed draw (drawIndexedPrimitives) over
//! 6 indices forming 2 triangles.

use aruminium::ffi::*;
use aruminium::{
    ColorAttachmentDesc, Gpu, GpuError, IndexType, PrimitiveType, RenderPassDescriptor,
    RenderPipelineSpec, VertexAttribute, VertexBufferLayout, VertexDescriptor, VertexFormat,
    VertexStep,
};

const SHADER: &str = r#"
    #include <metal_stdlib>
    using namespace metal;

    struct V    { float2 pos [[attribute(0)]]; float2 uv [[attribute(1)]]; };
    struct VOut { float4 pos [[position]]; float2 uv; };

    vertex VOut vmain(V v [[stage_in]]) {
        VOut o;
        o.pos = float4(v.pos, 0.0, 1.0);
        o.uv  = v.uv;
        return o;
    }

    fragment float4 fmain(VOut v [[stage_in]],
                          texture2d<float> tex [[texture(0)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        return tex.sample(s, v.uv);
    }
"#;

fn main() -> Result<(), GpuError> {
    let dev = Gpu::open()?;
    println!("Device: {}", dev.name());
    let queue = dev.new_command_queue()?;

    let lib = dev.compile(SHADER)?;
    let vfn = lib.function("vmain")?;
    let ffn = lib.function("fmain")?;

    let vd = VertexDescriptor::new()
        .with_attribute(VertexAttribute {
            shader_location: 0,
            format: VertexFormat::Float2,
            offset: 0,
            buffer_index: 0,
        })
        .with_attribute(VertexAttribute {
            shader_location: 1,
            format: VertexFormat::Float2,
            offset: 8,
            buffer_index: 0,
        })
        .with_layout(VertexBufferLayout {
            buffer_index: 0,
            stride: 16,
            step: VertexStep::PerVertex,
            step_rate: 1,
        });

    let spec = RenderPipelineSpec::color(MTLPixelFormatBGRA8Unorm).with_vertex_descriptor(vd);
    let pipeline = dev.render_pipeline(&vfn, &ffn, &spec)?;

    // Quad: 4 vertices (pos.xy, uv.xy). Two triangles via 6-index buffer.
    #[rustfmt::skip]
    let verts: [f32; 16] = [
        -0.8, -0.8, 0.0, 1.0,
         0.8, -0.8, 1.0, 1.0,
         0.8,  0.8, 1.0, 0.0,
        -0.8,  0.8, 0.0, 0.0,
    ];
    let vb = dev.buffer(verts.len() * 4)?;
    vb.write_f32(|d| d.copy_from_slice(&verts));

    let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
    let ib_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(indices.as_ptr() as *const u8, indices.len() * 2) };
    let ib = dev.buffer_with_data(ib_bytes)?;

    // Procedural 4x4 BGRA8 checker texture, then upload via a staging-style
    // texture: we use a plain `render_target` (RenderTarget|ShaderRead, private)
    // and a staging buffer + blit; here we use a shader-readable shared texture
    // instead via texture descriptor.
    let tex = make_sampled_texture(&dev, 4, 4)?;

    let target = dev.render_target(256, 256, MTLPixelFormatBGRA8Unorm)?;
    let mut pass = RenderPassDescriptor::new();
    pass.color_attachment(0, ColorAttachmentDesc::clear(&target, [0.0, 0.0, 0.0, 1.0]));

    let cmd = queue.commands()?;
    {
        let enc = cmd.render_encoder(&pass)?;
        enc.bind(&pipeline);
        enc.set_viewport(0.0, 0.0, 256.0, 256.0, 0.0, 1.0);
        enc.set_vertex_buffer(0, &vb, 0);
        enc.set_fragment_texture(0, &tex);
        enc.draw_indexed(PrimitiveType::Triangle, 6, IndexType::UInt16, &ib, 0);
        enc.end();
    }
    cmd.submit();
    cmd.wait();

    println!(
        "PASS: textured_quad rendered (pipeline color_attachments={}, sample_count={})",
        pipeline.color_attachments(),
        pipeline.sample_count(),
    );
    Ok(())
}

/// Build a 2D BGRA8 texture, ShaderRead, Shared storage, with a 4x4 checker
/// pattern written via replaceRegion.
fn make_sampled_texture(dev: &Gpu, w: u32, h: u32) -> Result<aruminium::Texture, GpuError> {
    unsafe {
        let cls = objc_getClass(c"MTLTextureDescriptor".as_ptr()) as ObjcId;
        let desc = msg0(cls, sel_registerName(c"new".as_ptr()));
        type SetU = unsafe extern "C" fn(ObjcId, ObjcSel, NSUInteger);
        let set_u: SetU = std::mem::transmute(objc_msgSend as *const std::ffi::c_void);
        set_u(desc, sel_registerName(c"setTextureType:".as_ptr()), 2); // 2D
        set_u(
            desc,
            sel_registerName(c"setPixelFormat:".as_ptr()),
            MTLPixelFormatBGRA8Unorm,
        );
        set_u(
            desc,
            sel_registerName(c"setWidth:".as_ptr()),
            w as NSUInteger,
        );
        set_u(
            desc,
            sel_registerName(c"setHeight:".as_ptr()),
            h as NSUInteger,
        );
        set_u(
            desc,
            sel_registerName(c"setUsage:".as_ptr()),
            MTLTextureUsageShaderRead,
        );
        // Shared storage so we can replaceRegion from CPU.
        set_u(desc, sel_registerName(c"setStorageMode:".as_ptr()), 0); // Shared
        let tex = dev.texture(desc)?;
        release(desc);

        // Write checker pixels.
        let mut pixels = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let dark = ((x + y) & 1) == 0;
                let c: u8 = if dark { 64 } else { 240 };
                pixels[i] = c; // B
                pixels[i + 1] = c; // G
                pixels[i + 2] = c; // R
                pixels[i + 3] = 255;
            }
        }
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: w as usize,
                height: h as usize,
                depth: 1,
            },
        };
        tex.replace_region(
            region,
            0,
            pixels.as_ptr() as *const std::ffi::c_void,
            (w * 4) as usize,
        );
        Ok(tex)
    }
}
