//! Reference: metal-rs (the `metal` crate) — most-used direct Metal wrapper in Rust.
//! wgpu's Metal backend uses metal-rs internally.

use metal::*;
use std::time::Instant;

const TRIANGLE: &str = r#"
    #include <metal_stdlib>
    using namespace metal;
    vertex float4 vmain(uint vid [[vertex_id]]) {
        float2 v[3] = { float2(-1,-1), float2(1,-1), float2(0,1) };
        return float4(v[vid], 0.0, 1.0);
    }
    fragment float4 fmain() { return float4(1.0); }
"#;

fn make_pipeline(device: &DeviceRef) -> RenderPipelineState {
    let lib = device
        .new_library_with_source(TRIANGLE, &CompileOptions::new())
        .unwrap();
    let vfn = lib.get_function("vmain", None).unwrap();
    let ffn = lib.get_function("fmain", None).unwrap();
    let desc = RenderPipelineDescriptor::new();
    desc.set_vertex_function(Some(&vfn));
    desc.set_fragment_function(Some(&ffn));
    desc.color_attachments()
        .object_at(0)
        .unwrap()
        .set_pixel_format(MTLPixelFormat::BGRA8Unorm);
    device.new_render_pipeline_state(&desc).unwrap()
}

fn make_target(device: &DeviceRef) -> Texture {
    let td = TextureDescriptor::new();
    td.set_texture_type(MTLTextureType::D2);
    td.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
    td.set_width(16);
    td.set_height(16);
    td.set_storage_mode(MTLStorageMode::Private);
    td.set_usage(MTLTextureUsage::RenderTarget);
    device.new_texture(&td)
}

pub fn render_pass_overhead(iters: usize) -> f64 {
    let device = Device::system_default().expect("no metal device");
    let queue = device.new_command_queue();
    let rpipeline = make_pipeline(&device);
    let target = make_target(&device);

    let rpass = RenderPassDescriptor::new();
    let att = rpass.color_attachments().object_at(0).unwrap();
    att.set_texture(Some(&target));
    att.set_load_action(MTLLoadAction::Clear);
    att.set_clear_color(MTLClearColor::new(0.0, 0.0, 0.0, 0.0));
    att.set_store_action(MTLStoreAction::Store);

    for _ in 0..5 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_render_command_encoder(rpass);
        enc.set_render_pipeline_state(&rpipeline);
        enc.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_render_command_encoder(rpass);
        enc.set_render_pipeline_state(&rpipeline);
        enc.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

pub fn render_batch_encode(n_draws: usize, iters: usize) -> f64 {
    let device = Device::system_default().expect("no metal device");
    let queue = device.new_command_queue();
    let rpipeline = make_pipeline(&device);
    let target = make_target(&device);

    let rpass = RenderPassDescriptor::new();
    let att = rpass.color_attachments().object_at(0).unwrap();
    att.set_texture(Some(&target));
    att.set_load_action(MTLLoadAction::Clear);
    att.set_clear_color(MTLClearColor::new(0.0, 0.0, 0.0, 0.0));
    att.set_store_action(MTLStoreAction::Store);

    for _ in 0..5 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_render_command_encoder(rpass);
        enc.set_render_pipeline_state(&rpipeline);
        for _ in 0..n_draws {
            enc.draw_primitives(MTLPrimitiveType::Triangle, 0, 0);
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_render_command_encoder(rpass);
        enc.set_render_pipeline_state(&rpipeline);
        for _ in 0..n_draws {
            enc.draw_primitives(MTLPrimitiveType::Triangle, 0, 0);
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    t0.elapsed().as_secs_f64() / (iters * n_draws) as f64
}
