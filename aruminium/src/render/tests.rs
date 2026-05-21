//! Unit tests for the render submodule (phase 1).

use crate::ffi::*;
use crate::{
    ColorAttachmentDesc, Gpu, LoadAction, PrimitiveType, RenderPassDescriptor, RenderPipelineSpec,
    StoreAction,
};

const VFRAG: &str = r#"
    #include <metal_stdlib>
    using namespace metal;
    struct VOut { float4 pos [[position]]; };
    vertex VOut vmain(uint vid [[vertex_id]]) {
        float2 v[3] = { float2(-1,-1), float2(1,-1), float2(0,1) };
        VOut o; o.pos = float4(v[vid], 0.0, 1.0); return o;
    }
    fragment float4 fmain(VOut v [[stage_in]]) { return float4(1.0); }
"#;

fn compile(dev: &Gpu) -> (crate::Shader, crate::Shader) {
    let lib = dev.compile(VFRAG).unwrap();
    (
        lib.function("vmain").unwrap(),
        lib.function("fmain").unwrap(),
    )
}

#[test]
fn render_target_create() {
    let dev = Gpu::open().unwrap();
    let tex = dev.render_target(64, 32, MTLPixelFormatBGRA8Unorm).unwrap();
    assert_eq!(tex.width(), 64);
    assert_eq!(tex.height(), 32);
    assert_eq!(tex.pixel_format(), MTLPixelFormatBGRA8Unorm);
}

#[test]
fn render_target_zero_size_rejected() {
    let dev = Gpu::open().unwrap();
    assert!(dev.render_target(0, 16, MTLPixelFormatBGRA8Unorm).is_err());
    assert!(dev.render_target(16, 0, MTLPixelFormatBGRA8Unorm).is_err());
}

#[test]
fn render_pipeline_build_basic() {
    let dev = Gpu::open().unwrap();
    let (v, f) = compile(&dev);
    let spec = RenderPipelineSpec::color(MTLPixelFormatBGRA8Unorm);
    let p = dev.render_pipeline(&v, &f, &spec).unwrap();
    assert_eq!(p.color_attachments(), 1);
    assert_eq!(p.sample_count(), 1);
    assert!(!p.as_raw().is_null());
}

#[test]
fn render_pipeline_requires_color_attachment() {
    let dev = Gpu::open().unwrap();
    let (v, f) = compile(&dev);
    let spec = RenderPipelineSpec {
        color_attachments: vec![],
    };
    assert!(dev.render_pipeline(&v, &f, &spec).is_err());
}

#[test]
fn pass_descriptor_color_attachment() {
    let dev = Gpu::open().unwrap();
    let tex = dev.render_target(16, 16, MTLPixelFormatBGRA8Unorm).unwrap();
    let mut pass = RenderPassDescriptor::new();
    pass.color_attachment(
        0,
        ColorAttachmentDesc {
            texture: &tex,
            load_action: LoadAction::Clear,
            store_action: StoreAction::Store,
            clear_color: [1.0, 0.0, 0.0, 1.0],
            level: 0,
            slice: 0,
        },
    );
    assert!(!pass.as_raw().is_null());
}

#[test]
fn pass_descriptor_clear_helper() {
    let dev = Gpu::open().unwrap();
    let tex = dev.render_target(16, 16, MTLPixelFormatBGRA8Unorm).unwrap();
    let mut pass = RenderPassDescriptor::new();
    pass.color_attachment(0, ColorAttachmentDesc::clear(&tex, [0.0, 0.5, 0.0, 1.0]));
    assert!(!pass.as_raw().is_null());
}

#[test]
fn encoder_lifecycle() {
    let dev = Gpu::open().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let (v, f) = compile(&dev);
    let spec = RenderPipelineSpec::color(MTLPixelFormatBGRA8Unorm);
    let pipe = dev.render_pipeline(&v, &f, &spec).unwrap();
    let tex = dev.render_target(8, 8, MTLPixelFormatBGRA8Unorm).unwrap();

    let mut pass = RenderPassDescriptor::new();
    pass.color_attachment(0, ColorAttachmentDesc::clear(&tex, [0.1, 0.2, 0.3, 1.0]));

    let cmd = queue.commands().unwrap();
    let enc = cmd.render_encoder(&pass).unwrap();
    enc.bind(&pipe);
    enc.set_viewport(0.0, 0.0, 8.0, 8.0, 0.0, 1.0);
    enc.set_scissor(0, 0, 8, 8);
    enc.draw(PrimitiveType::Triangle, 0, 3);
    enc.end();
    cmd.submit();
    cmd.wait();
    assert_eq!(cmd.status(), crate::Commands::STATUS_COMPLETED);
}

#[test]
fn encoder_drops_without_end() {
    // Drop should still call endEncoding so the command buffer is in a
    // valid state — verify by submitting + waiting after drop.
    let dev = Gpu::open().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let (v, f) = compile(&dev);
    let spec = RenderPipelineSpec::color(MTLPixelFormatBGRA8Unorm);
    let pipe = dev.render_pipeline(&v, &f, &spec).unwrap();
    let tex = dev.render_target(8, 8, MTLPixelFormatBGRA8Unorm).unwrap();

    let mut pass = RenderPassDescriptor::new();
    pass.color_attachment(0, ColorAttachmentDesc::clear(&tex, [0.0; 4]));

    let cmd = queue.commands().unwrap();
    {
        let enc = cmd.render_encoder(&pass).unwrap();
        enc.bind(&pipe);
        enc.draw(PrimitiveType::Triangle, 0, 3);
        // Drop without calling .end()
    }
    cmd.submit();
    cmd.wait();
    assert_eq!(cmd.status(), crate::Commands::STATUS_COMPLETED);
}

#[test]
fn multiple_color_attachments_spec() {
    let spec = RenderPipelineSpec::colors(&[MTLPixelFormatBGRA8Unorm, MTLPixelFormatRGBA16Float]);
    assert_eq!(spec.color_attachments.len(), 2);
    assert_eq!(spec.color_attachments[0].format, MTLPixelFormatBGRA8Unorm);
    assert_eq!(spec.color_attachments[1].format, MTLPixelFormatRGBA16Float);
}
