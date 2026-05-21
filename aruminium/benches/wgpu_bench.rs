//! Reference: wgpu with Metal backend — what Bevy uses; measures cross-platform abstraction overhead.

use std::borrow::Cow;
use std::time::Instant;

const WGSL_TRIANGLE: &str = r#"
@vertex
fn vmain(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 0.0,  1.0),
    );
    return vec4<f32>(pos[vid], 0.0, 1.0);
}

@fragment
fn fmain() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
"#;

struct Ctx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    texture_view: wgpu::TextureView,
}

fn setup() -> Ctx {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("no wgpu adapter");

    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL_TRIANGLE)),
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vmain"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fmain"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Bgra8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        multiview_mask: None,
        cache: None,
    });

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 16,
            height: 16,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bgra8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&Default::default());

    Ctx {
        device,
        queue,
        pipeline,
        texture_view,
    }
}

fn draw(ctx: &Ctx, n_draws: usize) {
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &ctx.texture_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&ctx.pipeline);
        for _ in 0..n_draws {
            rpass.draw(0..3, 0..1);
        }
    }
    let idx = ctx.queue.submit([encoder.finish()]);
    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(idx),
            timeout: None,
        })
        .unwrap();
}

pub fn render_pass_overhead(iters: usize) -> f64 {
    let ctx = setup();
    for _ in 0..5 {
        draw(&ctx, 1);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        draw(&ctx, 1);
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

pub fn render_batch_encode(n_draws: usize, iters: usize) -> f64 {
    let ctx = setup();
    for _ in 0..5 {
        draw(&ctx, n_draws);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        draw(&ctx, n_draws);
    }
    t0.elapsed().as_secs_f64() / (iters * n_draws) as f64
}
