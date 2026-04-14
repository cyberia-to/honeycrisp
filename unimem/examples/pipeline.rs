//! Simulated transformer layer through all Apple Silicon compute units.
//!
//! Each unit does what it's best at:
//!   CPU (NEON): RMSNorm, SiLU, Softmax, RoPE — element-wise ops
//!   AMX:        QKV projection, attention matmul, FFN — matrix multiply
//!   GPU:        same matmul for comparison (shader compile cost amortized)
//!   ANE:        matmul via compiled MIL program
//!
//! Two modes: standard (Vec, separate allocs, copies) vs unimem (one tape, zero-copy)
//!
//! Run: cargo run --example pipeline --release

use std::time::Instant;

// Transformer dimensions (small model — fits in cache, shows alloc/copy overhead)
const DIM: usize = 512; // hidden dimension
const HEADS: usize = 8; // attention heads
const HEAD_DIM: usize = 64; // DIM / HEADS
const SEQ: usize = 128; // sequence length
const FFN_DIM: usize = 1376; // ~2.7x DIM (Llama-style)

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Transformer layer pipeline: unimem vs standard ===");
    println!("    dim={DIM} heads={HEADS} seq={SEQ} ffn={FFN_DIM}\n");

    // Run reference first (warms caches, loads frameworks)
    let ref_result = run_standard()?;
    let cyb_result = run_cybmem()?;

    // Comparison
    println!("\n=== Comparison ===\n");
    println!(
        "  {:28} {:>12} {:>12} {:>8}",
        "Stage", "Standard", "unimem", "Speedup"
    );
    println!(
        "  {:28} {:>12} {:>12} {:>8}",
        "-----", "--------", "-------", "-------"
    );

    let pairs = [
        ("alloc + fill", ref_result.alloc, cyb_result.alloc),
        ("CPU: rmsnorm", ref_result.rmsnorm, cyb_result.rmsnorm),
        ("AMX: qkv projection", ref_result.qkv, cyb_result.qkv),
        ("CPU: rope", ref_result.rope, cyb_result.rope),
        ("AMX: attention matmul", ref_result.attn, cyb_result.attn),
        ("CPU: softmax", ref_result.softmax, cyb_result.softmax),
        ("AMX: attn @ V", ref_result.attn_v, cyb_result.attn_v),
        ("AMX: ffn up+gate", ref_result.ffn_up, cyb_result.ffn_up),
        ("CPU: silu", ref_result.silu, cyb_result.silu),
        ("AMX: ffn down", ref_result.ffn_down, cyb_result.ffn_down),
        (
            "ANE: compile+load",
            ref_result.ane_compile,
            cyb_result.ane_compile,
        ),
        ("ANE: matmul run", ref_result.ane_run, cyb_result.ane_run),
        ("TOTAL", ref_result.total, cyb_result.total),
    ];

    for (name, r, c) in &pairs {
        let speedup = r.as_nanos() as f64 / c.as_nanos().max(1) as f64;
        println!("  {:28} {:>12.1?} {:>12.1?} {:>7.1}x", name, r, c, speedup);
    }

    println!("\n  reference verified: {}", ref_result.pass);
    println!("  unimem verified:   {}", cyb_result.pass);

    Ok(())
}

struct LayerResult {
    alloc: std::time::Duration,
    rmsnorm: std::time::Duration,
    qkv: std::time::Duration,
    rope: std::time::Duration,
    attn: std::time::Duration,
    softmax: std::time::Duration,
    attn_v: std::time::Duration,
    ffn_up: std::time::Duration,
    silu: std::time::Duration,
    ffn_down: std::time::Duration,
    ane_compile: std::time::Duration,
    ane_run: std::time::Duration,
    total: std::time::Duration,
    pass: bool,
}

fn run_standard() -> Result<LayerResult, Box<dyn std::error::Error>> {
    println!(">>> STANDARD (Vec + separate allocs) <<<\n");
    let t_total = Instant::now();

    // Alloc
    let t = Instant::now();
    let mut x = vec![0.01f32; SEQ * DIM];
    let rms_weight = vec![1.0f32; DIM];
    let wq = vec![0.01f32; DIM * DIM];
    let wk = vec![0.01f32; DIM * DIM];
    let wv = vec![0.01f32; DIM * DIM];
    let w_up = vec![0.01f32; DIM * FFN_DIM];
    let w_gate = vec![0.01f32; DIM * FFN_DIM];
    let w_down = vec![0.01f32; FFN_DIM * DIM];
    let freqs = vec![1.0f32; HEAD_DIM]; // simplified RoPE freqs
    let mut norm_out = vec![0.0f32; DIM];
    let mut q = vec![0.0f32; SEQ * DIM];
    let mut k = vec![0.0f32; SEQ * DIM];
    let mut v = vec![0.0f32; SEQ * DIM];
    let mut scores = vec![0.0f32; SEQ * SEQ];
    let mut attn_out = vec![0.0f32; SEQ * DIM];
    let mut ffn_up_out = vec![0.0f32; SEQ * FFN_DIM];
    let mut ffn_gate_out = vec![0.0f32; SEQ * FFN_DIM];
    let mut ffn_out = vec![0.0f32; SEQ * DIM];
    let t_alloc = t.elapsed();

    // CPU: RMSNorm per token
    let t = Instant::now();
    for tok in 0..SEQ {
        let row = &x[tok * DIM..(tok + 1) * DIM];
        acpu::vector::normalize(&mut norm_out, row, &rms_weight, 1e-5);
        x[tok * DIM..(tok + 1) * DIM].copy_from_slice(&norm_out);
    }
    let t_rmsnorm = t.elapsed();

    // AMX: QKV projection
    let t = Instant::now();
    acpu::matmul_f32(&x, &wq, &mut q, SEQ, DIM, DIM);
    acpu::matmul_f32(&x, &wk, &mut k, SEQ, DIM, DIM);
    acpu::matmul_f32(&x, &wv, &mut v, SEQ, DIM, DIM);
    let t_qkv = t.elapsed();

    // CPU: RoPE
    let t = Instant::now();
    for tok in 0..SEQ {
        let mut q_tok = vec![0.0f32; HEAD_DIM];
        acpu::vector::rotate(&mut q_tok, &q[tok * DIM..tok * DIM + HEAD_DIM], &freqs, tok);
        q[tok * DIM..tok * DIM + HEAD_DIM].copy_from_slice(&q_tok);
    }
    let t_rope = t.elapsed();

    // AMX: attention scores = Q @ K^T (simplified: first head only, full seq)
    let t = Instant::now();
    acpu::matmul_f32(
        &q[..SEQ * HEAD_DIM],
        &k[..SEQ * HEAD_DIM],
        &mut scores,
        SEQ,
        SEQ,
        HEAD_DIM,
    );
    let t_attn = t.elapsed();

    // CPU: softmax per row
    let t = Instant::now();
    for row in 0..SEQ {
        acpu::vector::softmax(&mut scores[row * SEQ..(row + 1) * SEQ]);
    }
    let t_softmax = t.elapsed();

    // AMX: attn_out = scores @ V
    let t = Instant::now();
    acpu::matmul_f32(
        &scores,
        &v[..SEQ * HEAD_DIM],
        &mut attn_out[..SEQ * HEAD_DIM],
        SEQ,
        HEAD_DIM,
        SEQ,
    );
    let t_attn_v = t.elapsed();

    // AMX: FFN up + gate
    let t = Instant::now();
    acpu::matmul_f32(&x, &w_up, &mut ffn_up_out, SEQ, FFN_DIM, DIM);
    acpu::matmul_f32(&x, &w_gate, &mut ffn_gate_out, SEQ, FFN_DIM, DIM);
    let t_ffn_up = t.elapsed();

    // CPU: SiLU(gate) * up
    let t = Instant::now();
    acpu::vector::silu(&mut ffn_gate_out);
    for i in 0..ffn_up_out.len() {
        ffn_up_out[i] *= ffn_gate_out[i];
    }
    let t_silu = t.elapsed();

    // AMX: FFN down
    let t = Instant::now();
    acpu::matmul_f32(&ffn_up_out, &w_down, &mut ffn_out, SEQ, DIM, FFN_DIM);
    let t_ffn_down = t.elapsed();

    // ANE: small matmul
    let t = Instant::now();
    let program = rane::mil::matmul(64, 64, 64);
    let mut model = rane::Program::compile(&program, &[])?;
    model.load()?;
    let t_ane_compile = t.elapsed();

    let ane_in = rane::Buffer::new(program.input_size())?;
    let ane_out = rane::Buffer::new(program.output_size())?;
    fill_ane_identity(&ane_in, &program);

    let t = Instant::now();
    model.run(&ane_in, &ane_out)?;
    let t_ane_run = t.elapsed();

    let ane_ok = ane_out.read(|d| {
        let (oc, osp) = program.output_shape();
        d[..oc * osp].iter().all(|&v| rane::fp16_to_f32(v) == 1.0)
    });
    model.unload()?;

    let pass = ane_ok && ffn_out.iter().all(|v| v.is_finite());
    let t_total = t_total.elapsed();

    println!("  alloc:        {:?}", t_alloc);
    println!("  CPU rmsnorm:  {:?}", t_rmsnorm);
    println!("  AMX qkv:      {:?}", t_qkv);
    println!("  CPU rope:     {:?}", t_rope);
    println!("  AMX attn:     {:?}", t_attn);
    println!("  CPU softmax:  {:?}", t_softmax);
    println!("  AMX attn@V:   {:?}", t_attn_v);
    println!("  AMX ffn up:   {:?}", t_ffn_up);
    println!("  CPU silu:     {:?}", t_silu);
    println!("  AMX ffn down: {:?}", t_ffn_down);
    println!("  ANE compile:  {:?}", t_ane_compile);
    println!("  ANE run:      {:?}", t_ane_run);
    println!("  TOTAL:        {:?}", t_total);
    println!("  pass:         {}\n", pass);

    Ok(LayerResult {
        alloc: t_alloc,
        rmsnorm: t_rmsnorm,
        qkv: t_qkv,
        rope: t_rope,
        attn: t_attn,
        softmax: t_softmax,
        attn_v: t_attn_v,
        ffn_up: t_ffn_up,
        silu: t_silu,
        ffn_down: t_ffn_down,
        ane_compile: t_ane_compile,
        ane_run: t_ane_run,
        total: t_total,
        pass,
    })
}

fn run_cybmem() -> Result<LayerResult, Box<dyn std::error::Error>> {
    println!(">>> CYB-MEM (tape, zero-copy) <<<\n");
    let t_total = Instant::now();

    // All memory from one tape
    let total_bytes = SEQ * DIM * 4 +       // x
        DIM * 4 +             // rms_weight
        DIM * DIM * 4 * 3 +   // wq, wk, wv
        DIM * FFN_DIM * 4 * 2 + // w_up, w_gate
        FFN_DIM * DIM * 4 +   // w_down
        HEAD_DIM * 4 +         // freqs
        DIM * 4 +             // norm_out
        SEQ * DIM * 4 * 3 +   // q, k, v
        SEQ * SEQ * 4 +       // scores
        SEQ * DIM * 4 +       // attn_out
        SEQ * FFN_DIM * 4 * 2 + // ffn_up, ffn_gate
        SEQ * DIM * 4 +       // ffn_out
        1024 * 1024; // padding

    let t = Instant::now();
    let tape = unimem::Tape::start_warm(total_bytes)?;

    macro_rules! arena_slice {
        ($n:expr) => {{
            let ptr = tape.take($n * 4, 64).unwrap() as *mut f32;
            unsafe { std::slice::from_raw_parts_mut(ptr, $n) }
        }};
    }

    let x = arena_slice!(SEQ * DIM);
    let rms_weight = arena_slice!(DIM);
    let wq = arena_slice!(DIM * DIM);
    let wk = arena_slice!(DIM * DIM);
    let wv = arena_slice!(DIM * DIM);
    let w_up = arena_slice!(DIM * FFN_DIM);
    let w_gate = arena_slice!(DIM * FFN_DIM);
    let w_down = arena_slice!(FFN_DIM * DIM);
    let freqs = arena_slice!(HEAD_DIM);
    let norm_out = arena_slice!(DIM);
    let q = arena_slice!(SEQ * DIM);
    let k = arena_slice!(SEQ * DIM);
    let v = arena_slice!(SEQ * DIM);
    let scores = arena_slice!(SEQ * SEQ);
    let attn_out = arena_slice!(SEQ * DIM);
    let ffn_up_out = arena_slice!(SEQ * FFN_DIM);
    let ffn_gate_out = arena_slice!(SEQ * FFN_DIM);
    let ffn_out = arena_slice!(SEQ * DIM);

    // Fill weights
    x.fill(0.01);
    rms_weight.fill(1.0);
    wq.fill(0.01);
    wk.fill(0.01);
    wv.fill(0.01);
    w_up.fill(0.01);
    w_gate.fill(0.01);
    w_down.fill(0.01);
    freqs.fill(1.0);
    let t_alloc = t.elapsed();

    // CPU: RMSNorm
    let t = Instant::now();
    for tok in 0..SEQ {
        let row = &x[tok * DIM..(tok + 1) * DIM];
        acpu::vector::normalize(norm_out, row, rms_weight, 1e-5);
        x[tok * DIM..(tok + 1) * DIM].copy_from_slice(norm_out);
    }
    let t_rmsnorm = t.elapsed();

    // AMX: QKV
    let t = Instant::now();
    acpu::matmul_f32(x, wq, q, SEQ, DIM, DIM);
    acpu::matmul_f32(x, wk, k, SEQ, DIM, DIM);
    acpu::matmul_f32(x, wv, v, SEQ, DIM, DIM);
    let t_qkv = t.elapsed();

    // CPU: RoPE
    let t = Instant::now();
    for tok in 0..SEQ {
        let mut q_tok = [0.0f32; HEAD_DIM];
        acpu::vector::rotate(&mut q_tok, &q[tok * DIM..tok * DIM + HEAD_DIM], freqs, tok);
        q[tok * DIM..tok * DIM + HEAD_DIM].copy_from_slice(&q_tok);
    }
    let t_rope = t.elapsed();

    // AMX: attention
    let t = Instant::now();
    acpu::matmul_f32(
        &q[..SEQ * HEAD_DIM],
        &k[..SEQ * HEAD_DIM],
        scores,
        SEQ,
        SEQ,
        HEAD_DIM,
    );
    let t_attn = t.elapsed();

    // CPU: softmax
    let t = Instant::now();
    for row in 0..SEQ {
        acpu::vector::softmax(&mut scores[row * SEQ..(row + 1) * SEQ]);
    }
    let t_softmax = t.elapsed();

    // AMX: attn @ V
    let t = Instant::now();
    acpu::matmul_f32(
        scores,
        &v[..SEQ * HEAD_DIM],
        &mut attn_out[..SEQ * HEAD_DIM],
        SEQ,
        HEAD_DIM,
        SEQ,
    );
    let t_attn_v = t.elapsed();

    // AMX: FFN
    let t = Instant::now();
    acpu::matmul_f32(x, w_up, ffn_up_out, SEQ, FFN_DIM, DIM);
    acpu::matmul_f32(x, w_gate, ffn_gate_out, SEQ, FFN_DIM, DIM);
    let t_ffn_up = t.elapsed();

    // CPU: SiLU
    let t = Instant::now();
    acpu::vector::silu(ffn_gate_out);
    for i in 0..ffn_up_out.len() {
        ffn_up_out[i] *= ffn_gate_out[i];
    }
    let t_silu = t.elapsed();

    // AMX: FFN down
    let t = Instant::now();
    acpu::matmul_f32(ffn_up_out, w_down, ffn_out, SEQ, DIM, FFN_DIM);
    let t_ffn_down = t.elapsed();

    // ANE
    let t = Instant::now();
    let program = rane::mil::matmul(64, 64, 64);
    let mut model = rane::Program::compile(&program, &[])?;
    model.load()?;
    let t_ane_compile = t.elapsed();

    let cyb_in = unimem::Block::open(program.input_size())?;
    let cyb_out = unimem::Block::open(program.output_size())?;
    fill_ane_identity_raw(&cyb_in, &program);

    let t = Instant::now();
    unsafe { model.run_direct(cyb_in.handle(), cyb_out.handle())? };
    let t_ane_run = t.elapsed();

    let ane_ok = unsafe {
        let d =
            std::slice::from_raw_parts(cyb_out.address() as *const u16, program.output_size() / 2);
        let (oc, osp) = program.output_shape();
        d[..oc * osp].iter().all(|&v| rane::fp16_to_f32(v) == 1.0)
    };
    model.unload()?;

    let pass = ane_ok && ffn_out.iter().all(|v| v.is_finite());
    let t_total = t_total.elapsed();

    println!(
        "  alloc (tape): {:?}  ({:.1} MB used / {:.1} MB cap)",
        t_alloc,
        tape.used() as f64 / 1e6,
        tape.total() as f64 / 1e6
    );
    println!("  CPU rmsnorm:   {:?}", t_rmsnorm);
    println!("  AMX qkv:       {:?}", t_qkv);
    println!("  CPU rope:      {:?}", t_rope);
    println!("  AMX attn:      {:?}", t_attn);
    println!("  CPU softmax:   {:?}", t_softmax);
    println!("  AMX attn@V:    {:?}", t_attn_v);
    println!("  AMX ffn up:    {:?}", t_ffn_up);
    println!("  CPU silu:      {:?}", t_silu);
    println!("  AMX ffn down:  {:?}", t_ffn_down);
    println!("  ANE compile:   {:?}", t_ane_compile);
    println!("  ANE run:       {:?}", t_ane_run);
    println!("  TOTAL:         {:?}", t_total);
    println!("  pass:          {}\n", pass);

    Ok(LayerResult {
        alloc: t_alloc,
        rmsnorm: t_rmsnorm,
        qkv: t_qkv,
        rope: t_rope,
        attn: t_attn,
        softmax: t_softmax,
        attn_v: t_attn_v,
        ffn_up: t_ffn_up,
        silu: t_silu,
        ffn_down: t_ffn_down,
        ane_compile: t_ane_compile,
        ane_run: t_ane_run,
        total: t_total,
        pass,
    })
}

fn fill_ane_identity(surface: &rane::Buffer, program: &rane::Source) {
    let (_, in_sp) = program.input_shape();
    surface.write(|data| {
        for ch in 0..64 {
            for s in 0..64 {
                data[ch * in_sp + s] = rane::f32_to_fp16(1.0);
            }
            for o in 0..64 {
                data[ch * in_sp + 64 + o] = if ch == o { rane::f32_to_fp16(1.0) } else { 0 };
            }
        }
    });
}

fn fill_ane_identity_raw(surface: &unimem::Block, program: &rane::Source) {
    let (_, in_sp) = program.input_shape();
    unsafe {
        let data =
            std::slice::from_raw_parts_mut(surface.address() as *mut u16, program.input_size() / 2);
        for ch in 0..64 {
            for s in 0..64 {
                data[ch * in_sp + s] = rane::f32_to_fp16(1.0);
            }
            for o in 0..64 {
                data[ch * in_sp + 64 + o] = if ch == o { rane::f32_to_fp16(1.0) } else { 0 };
            }
        }
    }
}
