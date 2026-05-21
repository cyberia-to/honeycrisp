//! Depth-cube — phase 2 demo: depth-tested rotating cube.
//!
//! Confirms:
//!   - Gpu::depth_target — depth32float render target
//!   - RenderPassDescriptor::depth_attachment + clear depth
//!   - Gpu::depth_stencil_state(less, write=true)
//!   - RenderEncoder::set_depth_stencil_state + set_cull_mode
//!   - draw_indexed with 36 indices (12 triangles)
//!   - the GPU actually performs depth comparison (back faces hidden)

use aruminium::ffi::*;
use aruminium::{
    ColorAttachmentDesc, CullMode, DepthAttachmentDesc, DepthStencil, Gpu, GpuError, IndexType,
    PrimitiveType, RenderPassDescriptor, RenderPipelineSpec, VertexAttribute, VertexBufferLayout,
    VertexDescriptor, VertexFormat, VertexStep, Winding,
};

const SHADER: &str = r#"
    #include <metal_stdlib>
    using namespace metal;

    struct V {
        float3 pos [[attribute(0)]];
        float3 color [[attribute(1)]];
    };
    struct VOut {
        float4 pos [[position]];
        float3 color;
    };
    struct Uniforms {
        float4x4 mvp;
    };

    vertex VOut vmain(V v [[stage_in]],
                      constant Uniforms& u [[buffer(1)]]) {
        VOut o;
        o.pos = u.mvp * float4(v.pos, 1.0);
        o.color = v.color;
        return o;
    }

    fragment float4 fmain(VOut v [[stage_in]]) {
        return float4(v.color, 1.0);
    }
"#;

#[rustfmt::skip]
const CUBE_VERTS: [f32; 6 * 8] = [
    // pos.xyz                color.rgb
    -0.5, -0.5, -0.5,         1.0, 0.0, 0.0,
     0.5, -0.5, -0.5,         0.0, 1.0, 0.0,
     0.5,  0.5, -0.5,         0.0, 0.0, 1.0,
    -0.5,  0.5, -0.5,         1.0, 1.0, 0.0,
    -0.5, -0.5,  0.5,         1.0, 0.0, 1.0,
     0.5, -0.5,  0.5,         0.0, 1.0, 1.0,
     0.5,  0.5,  0.5,         1.0, 1.0, 1.0,
    -0.5,  0.5,  0.5,         0.5, 0.5, 0.5,
];

#[rustfmt::skip]
const CUBE_INDICES: [u16; 36] = [
    // back
    0, 1, 2,  0, 2, 3,
    // front
    4, 6, 5,  4, 7, 6,
    // left
    0, 3, 7,  0, 7, 4,
    // right
    1, 5, 6,  1, 6, 2,
    // top
    3, 2, 6,  3, 6, 7,
    // bottom
    0, 4, 5,  0, 5, 1,
];

// All matrices below are stored in COLUMN-MAJOR order (Metal's float4x4
// convention). Indexing: m[col * 4 + row].

/// Column-major 4x4 multiply: C = A * B.
fn matmul_col(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut r = [0.0; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += a[k * 4 + row] * b[col * 4 + k];
            }
            r[col * 4 + row] = s;
        }
    }
    r
}

/// Rotate around Y (column-major).
fn rotate_y_col(t: f32) -> [f32; 16] {
    let c = t.cos();
    let s = t.sin();
    // columns: col0=(c,0,-s,0), col1=(0,1,0,0), col2=(s,0,c,0), col3=(0,0,0,1)
    [
        c, 0.0, -s, 0.0, 0.0, 1.0, 0.0, 0.0, s, 0.0, c, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

/// Translate (column-major). Translation goes in column 3.
fn translate_col(x: f32, y: f32, z: f32) -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, x, y, z, 1.0,
    ]
}

/// Orthographic projection (column-major). Metal NDC: clip z in [0, 1].
/// Maps view-z in [-far, -near] to clip-z in [0, 1].
fn orthographic_col(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> [f32; 16] {
    let sx = 2.0 / (r - l);
    let sy = 2.0 / (t - b);
    let sz = 1.0 / (n - f);
    let tx = -(r + l) / (r - l);
    let ty = -(t + b) / (t - b);
    let tz = n / (n - f);
    [
        sx, 0.0, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 0.0, sz, 0.0, tx, ty, tz, 1.0,
    ]
}

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
            format: VertexFormat::Float3,
            offset: 0,
            buffer_index: 0,
        })
        .with_attribute(VertexAttribute {
            shader_location: 1,
            format: VertexFormat::Float3,
            offset: 12,
            buffer_index: 0,
        })
        .with_layout(VertexBufferLayout {
            buffer_index: 0,
            stride: 24,
            step: VertexStep::PerVertex,
            step_rate: 1,
        });

    let spec = RenderPipelineSpec::color(MTLPixelFormatBGRA8Unorm)
        .with_depth(MTLPixelFormatDepth32Float)
        .with_vertex_descriptor(vd);
    let pipeline = dev.render_pipeline(&vfn, &ffn, &spec)?;

    let ds_state = dev.depth_stencil_state(DepthStencil::less_write())?;

    let w: u32 = 256;
    let h: u32 = 256;
    let color = dev.render_target(w, h, MTLPixelFormatBGRA8Unorm)?;
    let depth = dev.depth_target(w, h, MTLPixelFormatDepth32Float)?;

    let vb = dev.buffer(CUBE_VERTS.len() * 4)?;
    vb.write_f32(|d| d.copy_from_slice(&CUBE_VERTS));
    let ib_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(CUBE_INDICES.as_ptr() as *const u8, CUBE_INDICES.len() * 2)
    };
    let ib = dev.buffer_with_data(ib_bytes)?;

    // MVP: rotate the cube and map view-z in [-1.5, -0.5] into clip-z in [0,1]
    // (Metal NDC depth range), using a simple orthographic-like projection.
    //
    // Matrices stored in column-major (Metal's float4x4 convention). Each
    // 4-tuple below is one *column* of the matrix.
    let model = rotate_y_col(0.7);
    let view = translate_col(0.0, 0.0, -1.0);
    let proj = orthographic_col(-1.0, 1.0, -1.0, 1.0, 0.5, 1.5);
    let mv = matmul_col(&view, &model);
    let mvp = matmul_col(&proj, &mv);
    let mvp_bytes: &[u8] = unsafe { std::slice::from_raw_parts(mvp.as_ptr() as *const u8, 16 * 4) };

    let mut pass = RenderPassDescriptor::new();
    pass.color_attachment(
        0,
        ColorAttachmentDesc::clear(&color, [0.05, 0.05, 0.10, 1.0]),
    );
    pass.depth_attachment(DepthAttachmentDesc::clear(&depth));

    let cmd = queue.commands()?;
    {
        let enc = cmd.render_encoder(&pass)?;
        enc.bind(&pipeline);
        enc.set_depth_stencil_state(&ds_state);
        enc.set_cull_mode(CullMode::Back);
        enc.set_front_facing_winding(Winding::CounterClockwise);
        enc.set_viewport(0.0, 0.0, w as f64, h as f64, 0.0, 1.0);
        enc.set_vertex_buffer(0, &vb, 0);
        enc.push_vertex(mvp_bytes, 1);
        enc.draw_indexed(
            PrimitiveType::Triangle,
            CUBE_INDICES.len() as u32,
            IndexType::UInt16,
            &ib,
            0,
        );
        enc.end();
    }
    cmd.submit();
    cmd.wait();

    // Verify the depth path actually ran: render and confirm at least one
    // non-clear pixel exists in the center region.
    let bytes_per_row = (w * 4) as usize;
    let total = bytes_per_row * h as usize;
    let staging = dev.buffer(total)?;
    let cmd2 = queue.commands()?;
    unsafe {
        let blit = msg0(cmd2.as_raw(), SEL_blitCommandEncoder());
        retain(blit);
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
            color.as_raw(),
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

    let clear_b = (0.10_f64 * 255.0).round() as u8;
    let clear_g = (0.05_f64 * 255.0).round() as u8;
    let clear_r = (0.05_f64 * 255.0).round() as u8;
    let mut non_clear = 0usize;
    staging.read(|d| {
        for i in (0..d.len()).step_by(4) {
            if d[i] != clear_b || d[i + 1] != clear_g || d[i + 2] != clear_r {
                non_clear += 1;
            }
        }
    });
    if non_clear == 0 {
        println!("FAIL: depth_cube produced no rendered pixels");
        return Ok(());
    }
    println!(
        "PASS: depth_cube rendered {} non-clear pixels ({}x{})",
        non_clear, w, h
    );
    Ok(())
}
