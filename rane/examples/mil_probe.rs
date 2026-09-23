//! MIL operation probe — what does ANECCompile actually accept?
//!
//! Tests micro-programs to find:
//!   1. Does fp32 add compile at all?
//!   2. Does final fp32 cast work after fp16 add chain?
//!   3. Does intermediate cast(fp16→fp32→fp16) work?
//!   4. Does the fp16 add saturate at 65504, or wrap, or accumulate differently?
//!
//! Run: cargo run -p rane --example mil_probe --release

use rane::{mil, Program};

const BUILD_INFO: &str = concat!(
    "{{\"coremlc-component-MIL\", \"3510.2.1\"}, ",
    "{\"coremlc-version\", \"3505.4.1\"}, ",
    "{\"coremltools-component-milinternal\", \"\"}, ",
    "{\"coremltools-version\", \"9.0\"}}",
);

fn header_simple(ch: usize, sp: usize) -> String {
    format!(
        "program(1.3)\n[buildInfo = dict<string, string>({info})]\n{{\n    func main<ios18>(tensor<fp16, [1, {ch}, 1, {sp}]> x) {{\n",
        info=BUILD_INFO, ch=ch, sp=sp,
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("MIL operation probe — what does ANE actually allow?\n");

    // Probe 1: minimum fp32 add — just one matmul + cast + add(fp32, fp32)
    probe(
        "P1: matmul → cast(fp32) → add(fp32 self,self)",
        &mil_p1(),
        true,
    );

    // Probe 2: 2 matmuls, sum in fp16, terminal cast to fp32
    probe(
        "P2: matmul + matmul → add(fp16) → cast(fp32)",
        &mil::matmul_split_fp16_cast(64, 64, 64, 2).text,
        false,
    );

    // Probe 3: matmul → cast(fp32) → cast(fp16) → output (round-trip)
    probe("P3: matmul → cast(fp32) → cast(fp16)", &mil_p3(), true);

    // Probe 4: add(fp32 const, fp32 cast result)
    probe(
        "P4: matmul → cast(fp32) → add(fp32, fp32_const)",
        &mil_p4(),
        true,
    );

    // Probe 5: try `reduce_sum` (does it have fp32 accumulator?)
    probe("P5: 2 matmuls → concat → reduce_sum", &mil_p5(), true);

    // Probe 6: cast each chunk to fp32 BEFORE reduce_sum (the dream path)
    probe(
        "P6: 2 matmuls → cast each fp32 → concat fp32 → reduce_sum fp32",
        &mil_p6(),
        true,
    );

    // Probe 7: stack fp16 chunks → reduce_sum → cast(fp32) terminal
    probe(
        "P7: 2 matmuls → stack fp16 → reduce_sum fp16 → cast fp32",
        &mil_p7(),
        true,
    );

    // Probe 8: try reduce_sum with explicit output_dtype="fp32"
    probe(
        "P8: stack fp16 → reduce_sum[output_dtype=fp32]",
        &mil_p8(),
        true,
    );

    // Probe 9: try built-in `linear` op (dense layer, may have separate accumulator)
    probe("P9: linear with fp16 weight", &mil_p9(), true);

    // Probe 10: try `matmul` with fp32 inputs directly (after cast)
    probe("P10: cast x → fp32 matmul fp32", &mil_p10(), true);

    // Probe 11: scale-down trick — multiply by 0.5, then matmul, then maybe multiply by 2 outside
    probe("P11: mul(0.5) → matmul → cast(fp32)", &mil_p11(), true);

    // Probe 12: try quantize → matmul int8 (the holy grail — int32 accumulator hardware path)
    probe("P12: quantize → matmul → dequantize", &mil_p12(), true);

    // Probe 13: matmul with explicit fp32 output_dtype param (some MIL versions support this)
    probe("P13: matmul[output_dtype=fp32]", &mil_p13(), true);

    // Probe 14: conv2d 1x1 (alternate matmul path with possibly different accumulator)
    probe("P14: conv2d 1x1", &mil_p14(), true);

    // Probe 15: quantize with proper MIL signature: quantize(x=, scale=, zero_point=, output_dtype=)
    probe(
        "P15: quantize(x,scale,zero_point,output_dtype=int8) + int8 matmul",
        &mil_p15(),
        true,
    );

    // Probe 16: same with axis attribute
    probe("P16: quantize w/ axis + matmul", &mil_p16(), true);

    // Probe 17: constexpr_affine_dequantize on weight + matmul
    probe(
        "P17: constexpr_affine_dequantize weight + matmul",
        &mil_p17(),
        true,
    );

    // Probe 18: cast matmul result to int8 (does ANE use int32 accumulator behind?)
    probe("P18: matmul → cast(int8)", &mil_p18(), true);

    // Probe 19: cast to uint8
    probe("P19: matmul → cast(uint8)", &mil_p19(), true);

    // Probe 20: cast to int16
    probe("P20: matmul → cast(int16)", &mil_p20(), true);

    // Probe 21: input as int8, matmul int8
    probe("P21: int8 input + matmul int8", &mil_p21(), true);

    // Probe 22: explicit fp_to_int_clamped
    probe("P22: fp_to_int_clamped", &mil_p22(), true);

    // Probe 23: constexpr_affine_dequantize WITHOUT axis (per-tensor)
    probe(
        "P23: constexpr_affine_dequantize per-tensor + matmul",
        &mil_p23(),
        true,
    );

    // Probe 24: with per-channel scale tensor (rank 1)
    probe(
        "P24: constexpr_affine_dequantize per-channel + matmul",
        &mil_p24(),
        true,
    );

    // Probe 25: matmul → int8 output as RAW output type (no cast)
    probe(
        "P25: matmul int8 output declared directly",
        &mil_p25(),
        true,
    );

    // Probe 26: try output_dtype attribute on matmul
    probe("P26: matmul[output_dtype=int8]", &mil_p26(), true);

    // Probe 27: matmul → mul(scale) → cast(int16)
    probe("P27: matmul → mul(1/256) → cast(int8)", &mil_p27(), true);

    // Probe 28: matmul → mul → matmul (chain for byte extraction)
    probe("P28: matmul → mul(1/65536) → cast(int8)", &mil_p28(), true);

    Ok(())
}

fn probe(label: &str, mil_text: &str, dump: bool) {
    println!("=== {label} ===");
    if dump {
        // Save MIL to /tmp for inspection
        let path = format!("/tmp/probe_{}.mil", label.split(':').next().unwrap_or("p"));
        let _ = std::fs::write(&path, mil_text);
    }
    let src = rane::Source {
        text: mil_text.to_string(),
        input_channels: 64,
        input_spatial: 128,
        output_channels: 64,
        output_spatial: 64,
        output_dtype: rane::OutputDtype::Fp32,
    };
    match Program::compile(&src, &[]) {
        Ok(_) => println!("  ✓ COMPILED\n"),
        Err(e) => {
            let s = format!("{e}");
            // Save full error to file for inspection
            let safe_label = label.split(':').next().unwrap_or("p").replace(' ', "_");
            let path = format!("/tmp/probe_err_{safe_label}.txt");
            let _ = std::fs::write(&path, &s);
            // Show as much as we can — last 800 chars often contains the real reason
            let tail: String = s
                .chars()
                .rev()
                .take(800)
                .collect::<String>()
                .chars()
                .rev()
                .collect();
            println!("  ✗ (tail) {tail}\n");
        }
    }
}

// P1: one matmul → cast(fp32) → fp32 add(self, self)
fn mil_p1() -> String {
    let mut m = header_simple(64, 128);
    mil::gen_dyn_matmul(&mut m, "mm", 64, 64, 64, 0, 64, "x");
    m += "        tensor<fp32, [1,64,1,64]> f = cast(x=mm_y, dtype=string(\"fp32\"))[name=string(\"f\")];\n";
    m += "        tensor<fp32, [1,64,1,64]> s = add(x=f, y=f)[name=string(\"s\")];\n";
    m += &mil::mil_footer("s");
    m
}

// P3: matmul → cast(fp32) → cast(fp16) → output
fn mil_p3() -> String {
    let mut m = header_simple(64, 128);
    mil::gen_dyn_matmul(&mut m, "mm", 64, 64, 64, 0, 64, "x");
    m += "        tensor<fp32, [1,64,1,64]> f = cast(x=mm_y, dtype=string(\"fp32\"))[name=string(\"f\")];\n";
    m += "        tensor<fp16, [1,64,1,64]> h = cast(x=f, dtype=string(\"fp16\"))[name=string(\"h\")];\n";
    m += &mil::mil_footer("h");
    m
}

// P4: matmul → cast(fp32) → add(fp32, fp32_const)
fn mil_p4() -> String {
    let mut m = header_simple(64, 128);
    mil::gen_dyn_matmul(&mut m, "mm", 64, 64, 64, 0, 64, "x");
    m += "        tensor<fp32, [1,64,1,64]> f = cast(x=mm_y, dtype=string(\"fp32\"))[name=string(\"f\")];\n";
    // fp32 constant tensor with value 0.0
    m += "        tensor<fp32, [1,64,1,64]> z = const()[name=string(\"z\"), val=tensor<fp32, [1,64,1,64]>([";
    for i in 0..4096 {
        if i > 0 {
            m += ",";
        }
        m += "0.0";
    }
    m += "])];\n";
    m += "        tensor<fp32, [1,64,1,64]> s = add(x=f, y=z)[name=string(\"s\")];\n";
    m += &mil::mil_footer("s");
    m
}

// P6: 2 matmuls → cast EACH to fp32 → concat fp32 → reduce_sum
fn mil_p6() -> String {
    let mut m = header_simple(64, 128);
    mil::gen_dyn_matmul_chunk(&mut m, "c0", 0, 32, 64, 64, 64, "x");
    mil::gen_dyn_matmul_chunk(&mut m, "c1", 32, 32, 64, 64, 64, "x");
    m += "        tensor<fp32, [1,64,1,64]> f0 = cast(x=c0_y, dtype=string(\"fp32\"))[name=string(\"f0\")];\n";
    m += "        tensor<fp32, [1,64,1,64]> f1 = cast(x=c1_y, dtype=string(\"fp32\"))[name=string(\"f1\")];\n";
    m += "        tensor<fp32, [1,128,1,64]> cc = concat(values=(f0, f1), axis=int32(1), interleave=bool(false))[name=string(\"cc\")];\n";
    m += "        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,2,64,64])];\n";
    m += "        tensor<fp32, [1,2,64,64]> r2 = reshape(shape=rsh, x=cc)[name=string(\"r2\")];\n";
    m += "        tensor<int32, [1]> axes = const()[name=string(\"axes\"), val=tensor<int32, [1]>([1])];\n";
    m += "        bool kd = const()[name=string(\"kd\"), val=bool(false)];\n";
    m += "        tensor<fp32, [1,64,64]> rs = reduce_sum(x=r2, axes=axes, keep_dims=kd)[name=string(\"rs\")];\n";
    m += "        tensor<int32, [4]> rsho = const()[name=string(\"rsho\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp32, [1,64,1,64]> y = reshape(shape=rsho, x=rs)[name=string(\"y\")];\n";
    m += &mil::mil_footer("y");
    m
}

// P7: stack 2 matmuls fp16 → reduce_sum fp16 → cast(fp32) terminal
fn mil_p7() -> String {
    let mut m = header_simple(64, 128);
    mil::gen_dyn_matmul_chunk(&mut m, "c0", 0, 32, 64, 64, 64, "x");
    mil::gen_dyn_matmul_chunk(&mut m, "c1", 32, 32, 64, 64, 64, "x");
    m += "        tensor<fp16, [1,128,1,64]> cc = concat(values=(c0_y, c1_y), axis=int32(1), interleave=bool(false))[name=string(\"cc\")];\n";
    m += "        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,2,64,64])];\n";
    m += "        tensor<fp16, [1,2,64,64]> r2 = reshape(shape=rsh, x=cc)[name=string(\"r2\")];\n";
    m += "        tensor<int32, [1]> axes = const()[name=string(\"axes\"), val=tensor<int32, [1]>([1])];\n";
    m += "        bool kd = const()[name=string(\"kd\"), val=bool(false)];\n";
    m += "        tensor<fp16, [1,64,64]> rs = reduce_sum(x=r2, axes=axes, keep_dims=kd)[name=string(\"rs\")];\n";
    m += "        tensor<int32, [4]> rsho = const()[name=string(\"rsho\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp16, [1,64,1,64]> r = reshape(shape=rsho, x=rs)[name=string(\"r\")];\n";
    m += "        tensor<fp32, [1,64,1,64]> y = cast(x=r, dtype=string(\"fp32\"))[name=string(\"y\")];\n";
    m += &mil::mil_footer("y");
    m
}

// P8: explicit reduce_sum output_dtype=fp32
fn mil_p8() -> String {
    let mut m = header_simple(64, 128);
    mil::gen_dyn_matmul_chunk(&mut m, "c0", 0, 32, 64, 64, 64, "x");
    mil::gen_dyn_matmul_chunk(&mut m, "c1", 32, 32, 64, 64, 64, "x");
    m += "        tensor<fp16, [1,128,1,64]> cc = concat(values=(c0_y, c1_y), axis=int32(1), interleave=bool(false))[name=string(\"cc\")];\n";
    m += "        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,2,64,64])];\n";
    m += "        tensor<fp16, [1,2,64,64]> r2 = reshape(shape=rsh, x=cc)[name=string(\"r2\")];\n";
    m += "        tensor<int32, [1]> axes = const()[name=string(\"axes\"), val=tensor<int32, [1]>([1])];\n";
    m += "        bool kd = const()[name=string(\"kd\"), val=bool(false)];\n";
    // Output as fp32 directly
    m += "        tensor<fp32, [1,64,64]> rs = reduce_sum(x=r2, axes=axes, keep_dims=kd, output_dtype=string(\"fp32\"))[name=string(\"rs\")];\n";
    m += "        tensor<int32, [4]> rsho = const()[name=string(\"rsho\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp32, [1,64,1,64]> y = reshape(shape=rsho, x=rs)[name=string(\"y\")];\n";
    m += &mil::mil_footer("y");
    m
}

// P9: linear op
fn mil_p9() -> String {
    let mut m = header_simple(64, 128);
    // slice activations [1,64,1,64]
    m += "        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n";
    m += "        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp16, [1,64,1,64]> a = slice_by_size(x=x, begin=ba, size=sa)[name=string(\"a\")];\n";
    // weights from input (slice)
    m += "        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,64])];\n";
    m += "        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp16, [1,64,1,64]> w = slice_by_size(x=x, begin=bw, size=sw)[name=string(\"w\")];\n";
    m += "        tensor<int32, [2]> wsh = const()[name=string(\"wsh\"), val=tensor<int32, [2]>([64,64])];\n";
    m += "        tensor<fp16, [64,64]> w2 = reshape(shape=wsh, x=w)[name=string(\"w2\")];\n";
    m += "        tensor<int32, [2]> ash = const()[name=string(\"ash\"), val=tensor<int32, [2]>([64,64])];\n";
    m += "        tensor<fp16, [64,64]> a2 = reshape(shape=ash, x=a)[name=string(\"a2\")];\n";
    m += "        tensor<fp16, [64,64]> ly = linear(x=a2, weight=w2)[name=string(\"ly\")];\n";
    m += "        tensor<int32, [4]> osh = const()[name=string(\"osh\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp16, [1,64,1,64]> r = reshape(shape=osh, x=ly)[name=string(\"r\")];\n";
    m += "        tensor<fp32, [1,64,1,64]> y = cast(x=r, dtype=string(\"fp32\"))[name=string(\"y\")];\n";
    m += &mil::mil_footer("y");
    m
}

// P10: cast x to fp32 first, then matmul on fp32 tensors
fn mil_p10() -> String {
    let mut m = header_simple(64, 128);
    m += "        tensor<fp32, [1,64,1,128]> xf = cast(x=x, dtype=string(\"fp32\"))[name=string(\"xf\")];\n";
    m += "        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n";
    m += "        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp32, [1,64,1,64]> a = slice_by_size(x=xf, begin=ba, size=sa)[name=string(\"a\")];\n";
    m += "        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,64])];\n";
    m += "        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp32, [1,64,1,64]> w = slice_by_size(x=xf, begin=bw, size=sw)[name=string(\"w\")];\n";
    // Set up matmul
    m += "        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,64,64])];\n";
    m += "        tensor<fp32, [1,1,64,64]> a2 = reshape(shape=ra, x=a)[name=string(\"a2\")];\n";
    m += "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n";
    m += "        tensor<fp32, [1,1,64,64]> a3 = transpose(perm=pm, x=a2)[name=string(\"a3\")];\n";
    m += "        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,64,64])];\n";
    m += "        tensor<fp32, [1,1,64,64]> W = reshape(shape=rw, x=w)[name=string(\"W\")];\n";
    m += "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n";
    m += "        tensor<fp32, [1,1,64,64]> yh = matmul(transpose_x=bF, transpose_y=bF, x=a3, y=W)[name=string(\"yh\")];\n";
    m += "        tensor<fp32, [1,1,64,64]> yt = transpose(perm=pm, x=yh)[name=string(\"yt\")];\n";
    m += "        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp32, [1,64,1,64]> y = reshape(shape=ro, x=yt)[name=string(\"y\")];\n";
    m += &mil::mil_footer("y");
    m
}

// P11: mul by 0.5 (scale down inputs)
fn mil_p11() -> String {
    let mut m = header_simple(64, 128);
    m += "        tensor<fp16, []> half = const()[name=string(\"half\"), val=fp16(0.5)];\n";
    m += "        tensor<fp16, [1,64,1,128]> xh = mul(x=x, y=half)[name=string(\"xh\")];\n";
    mil::gen_dyn_matmul(&mut m, "mm", 64, 64, 64, 0, 64, "xh");
    m += "        tensor<fp32, [1,64,1,64]> y = cast(x=mm_y, dtype=string(\"fp32\"))[name=string(\"y\")];\n";
    m += &mil::mil_footer("y");
    m
}

// P12: quantize → matmul → dequantize (try int8 GEMM hardware path)
fn mil_p12() -> String {
    let mut m = header_simple(64, 128);
    // slice activations and weights as fp16
    m += "        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n";
    m += "        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp16, [1,64,1,64]> a = slice_by_size(x=x, begin=ba, size=sa)[name=string(\"a\")];\n";
    m += "        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,64])];\n";
    m += "        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp16, [1,64,1,64]> w = slice_by_size(x=x, begin=bw, size=sw)[name=string(\"w\")];\n";
    // Quantize to int8 with scale=1.0 (so int8 values == fp16 values for inputs ≤127)
    m += "        tensor<fp16, []> scale = const()[name=string(\"scale\"), val=fp16(1.0)];\n";
    m += "        tensor<int8, [1,64,1,64]> aq = quantize(input=a, scale=scale, output_dtype=string(\"int8\"))[name=string(\"aq\")];\n";
    m += "        tensor<int8, [1,64,1,64]> wq = quantize(input=w, scale=scale, output_dtype=string(\"int8\"))[name=string(\"wq\")];\n";
    // Try matmul on int8
    m += "        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,64,64])];\n";
    m += "        tensor<int8, [1,1,64,64]> a2 = reshape(shape=ra, x=aq)[name=string(\"a2\")];\n";
    m += "        tensor<int8, [1,1,64,64]> W = reshape(shape=ra, x=wq)[name=string(\"W\")];\n";
    m += "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n";
    m += "        tensor<int32, [1,1,64,64]> yh = matmul(transpose_x=bF, transpose_y=bF, x=a2, y=W)[name=string(\"yh\")];\n";
    m += "        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<int32, [1,64,1,64]> y = reshape(shape=ro, x=yh)[name=string(\"y\")];\n";
    m += &mil::mil_footer("y");
    m
}

// P13: matmul with explicit output_dtype param
fn mil_p13() -> String {
    let mut m = header_simple(64, 128);
    m += "        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n";
    m += "        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp16, [1,64,1,64]> a = slice_by_size(x=x, begin=ba, size=sa)[name=string(\"a\")];\n";
    m += "        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,64])];\n";
    m += "        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp16, [1,64,1,64]> w = slice_by_size(x=x, begin=bw, size=sw)[name=string(\"w\")];\n";
    m += "        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,64,64])];\n";
    m += "        tensor<fp16, [1,1,64,64]> a2 = reshape(shape=ra, x=a)[name=string(\"a2\")];\n";
    m += "        tensor<fp16, [1,1,64,64]> W = reshape(shape=ra, x=w)[name=string(\"W\")];\n";
    m += "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n";
    m += "        tensor<fp32, [1,1,64,64]> yh = matmul(transpose_x=bF, transpose_y=bF, x=a2, y=W, output_dtype=string(\"fp32\"))[name=string(\"yh\")];\n";
    m += "        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp32, [1,64,1,64]> y = reshape(shape=ro, x=yh)[name=string(\"y\")];\n";
    m += &mil::mil_footer("y");
    m
}

// P14: conv2d 1x1 (alternative matmul-like op)
fn mil_p14() -> String {
    let mut m = header_simple(64, 128);
    // Use weight as input slice
    m += "        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,64])];\n";
    m += "        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp16, [1,64,1,64]> w = slice_by_size(x=x, begin=bw, size=sw)[name=string(\"w\")];\n";
    // Reshape weight to conv2d format [out_channels, in_channels, kH, kW] = [64, 64, 1, 1]
    m += "        tensor<int32, [4]> wsh = const()[name=string(\"wsh\"), val=tensor<int32, [4]>([64,64,1,1])];\n";
    m += "        tensor<fp16, [64,64,1,1]> wk = reshape(shape=wsh, x=w)[name=string(\"wk\")];\n";
    // Slice activations [1,64,1,64]
    m += "        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n";
    m += "        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp16, [1,64,1,64]> a = slice_by_size(x=x, begin=ba, size=sa)[name=string(\"a\")];\n";
    m += "        tensor<int32, [2]> strides = const()[name=string(\"strides\"), val=tensor<int32, [2]>([1,1])];\n";
    m += "        tensor<int32, [2]> dilations = const()[name=string(\"dilations\"), val=tensor<int32, [2]>([1,1])];\n";
    m += "        tensor<int32, [4]> pad = const()[name=string(\"pad\"), val=tensor<int32, [4]>([0,0,0,0])];\n";
    m += "        string padt = string(\"valid\");\n";
    m += "        tensor<fp16, [1,64,1,64]> r = conv(x=a, weight=wk, strides=strides, pad_type=padt, pad=pad, dilations=dilations, groups=int32(1))[name=string(\"r\")];\n";
    m += "        tensor<fp32, [1,64,1,64]> y = cast(x=r, dtype=string(\"fp32\"))[name=string(\"y\")];\n";
    m += &mil::mil_footer("y");
    m
}

// P15: proper quantize op (x, scale, zero_point, output_dtype) + int8 matmul
fn mil_p15() -> String {
    let mut m = header_simple(64, 128);
    // slice a, w as fp16
    m += "        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n";
    m += "        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp16, [1,64,1,64]> a = slice_by_size(x=x, begin=ba, size=sa)[name=string(\"a\")];\n";
    m += "        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,64])];\n";
    m += "        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp16, [1,64,1,64]> w = slice_by_size(x=x, begin=bw, size=sw)[name=string(\"w\")];\n";
    // quantize params
    m += "        tensor<fp16, []> sc = const()[name=string(\"sc\"), val=fp16(1.0)];\n";
    m += "        tensor<int8, []> zp = const()[name=string(\"zp\"), val=int8(0)];\n";
    m += "        tensor<int8, [1,64,1,64]> aq = quantize(x=a, scale=sc, zero_point=zp, output_dtype=string(\"int8\"))[name=string(\"aq\")];\n";
    m += "        tensor<int8, [1,64,1,64]> wq = quantize(x=w, scale=sc, zero_point=zp, output_dtype=string(\"int8\"))[name=string(\"wq\")];\n";
    // matmul on int8 tensors
    m += "        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,64,64])];\n";
    m += "        tensor<int8, [1,1,64,64]> a2 = reshape(shape=ra, x=aq)[name=string(\"a2\")];\n";
    m += "        tensor<int8, [1,1,64,64]> W2 = reshape(shape=ra, x=wq)[name=string(\"W2\")];\n";
    m += "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n";
    m += "        tensor<int32, [1,1,64,64]> yh = matmul(transpose_x=bF, transpose_y=bF, x=a2, y=W2)[name=string(\"yh\")];\n";
    m += "        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<int32, [1,64,1,64]> y = reshape(shape=ro, x=yh)[name=string(\"y\")];\n";
    m += &mil::mil_footer("y");
    m
}

// P16: with axis attribute
fn mil_p16() -> String {
    let mut m = header_simple(64, 128);
    m += "        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n";
    m += "        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp16, [1,64,1,64]> a = slice_by_size(x=x, begin=ba, size=sa)[name=string(\"a\")];\n";
    m += "        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,64])];\n";
    m += "        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp16, [1,64,1,64]> w = slice_by_size(x=x, begin=bw, size=sw)[name=string(\"w\")];\n";
    m += "        tensor<fp16, []> sc = const()[name=string(\"sc\"), val=fp16(1.0)];\n";
    m += "        tensor<int8, []> zp = const()[name=string(\"zp\"), val=int8(0)];\n";
    m += "        tensor<int32, []> ax = const()[name=string(\"ax\"), val=int32(-1)];\n";
    m += "        tensor<int8, [1,64,1,64]> aq = quantize(x=a, scale=sc, zero_point=zp, axis=ax, output_dtype=string(\"int8\"))[name=string(\"aq\")];\n";
    m += "        tensor<int8, [1,64,1,64]> wq = quantize(x=w, scale=sc, zero_point=zp, axis=ax, output_dtype=string(\"int8\"))[name=string(\"wq\")];\n";
    m += "        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,64,64])];\n";
    m += "        tensor<int8, [1,1,64,64]> a2 = reshape(shape=ra, x=aq)[name=string(\"a2\")];\n";
    m += "        tensor<int8, [1,1,64,64]> W2 = reshape(shape=ra, x=wq)[name=string(\"W2\")];\n";
    m += "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n";
    m += "        tensor<int32, [1,1,64,64]> yh = matmul(transpose_x=bF, transpose_y=bF, x=a2, y=W2)[name=string(\"yh\")];\n";
    m += "        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<int32, [1,64,1,64]> y = reshape(shape=ro, x=yh)[name=string(\"y\")];\n";
    m += &mil::mil_footer("y");
    m
}

// P17: use constexpr_affine_dequantize for STATIC int8 weight, then matmul against fp16 activation
fn mil_p17() -> String {
    let mut m = header_simple(64, 128);
    // slice activation
    m += "        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n";
    m += "        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp16, [1,64,1,64]> a = slice_by_size(x=x, begin=ba, size=sa)[name=string(\"a\")];\n";
    // Build a small int8 weight constant
    let mut weight_str = String::from("[");
    for i in 0..(64 * 64) {
        if i > 0 {
            weight_str += ",";
        }
        weight_str += "1";
    }
    weight_str += "]";
    m += &format!("        tensor<int8, [64,64]> wq = const()[name=string(\"wq\"), val=tensor<int8, [64,64]>({weight_str})];\n");
    m += "        tensor<fp16, []> sc = const()[name=string(\"sc\"), val=fp16(1.0)];\n";
    m += "        tensor<int8, []> zp = const()[name=string(\"zp\"), val=int8(0)];\n";
    // dequantize the int8 weight back to fp16 — the compiler should treat this as int8 weight
    m += "        tensor<fp16, [64,64]> wf = constexpr_affine_dequantize(quantized_data=wq, scale=sc, zero_point=zp, axis=int32(0))[name=string(\"wf\")];\n";
    // reshape activation [64,64] for matmul
    m += "        tensor<int32, [2]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [2]>([64,64])];\n";
    m += "        tensor<fp16, [64,64]> a2 = reshape(shape=rsh, x=a)[name=string(\"a2\")];\n";
    m += "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n";
    m += "        tensor<fp16, [64,64]> ym = matmul(transpose_x=bF, transpose_y=bF, x=a2, y=wf)[name=string(\"ym\")];\n";
    m += "        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp16, [1,64,1,64]> yr = reshape(shape=ro, x=ym)[name=string(\"yr\")];\n";
    m += "        tensor<fp32, [1,64,1,64]> y = cast(x=yr, dtype=string(\"fp32\"))[name=string(\"y\")];\n";
    m += &mil::mil_footer("y");
    m
}

// P18: matmul → cast to int8
fn mil_p18() -> String {
    let mut m = header_simple(64, 128);
    mil::gen_dyn_matmul(&mut m, "mm", 64, 64, 64, 0, 64, "x");
    m += "        tensor<int8, [1,64,1,64]> y = cast(x=mm_y, dtype=string(\"int8\"))[name=string(\"y\")];\n";
    m += &mil::mil_footer("y");
    m
}

// P19: matmul → cast to uint8
fn mil_p19() -> String {
    let mut m = header_simple(64, 128);
    mil::gen_dyn_matmul(&mut m, "mm", 64, 64, 64, 0, 64, "x");
    m += "        tensor<uint8, [1,64,1,64]> y = cast(x=mm_y, dtype=string(\"uint8\"))[name=string(\"y\")];\n";
    m += &mil::mil_footer("y");
    m
}

// P20: matmul → cast to int16
fn mil_p20() -> String {
    let mut m = header_simple(64, 128);
    mil::gen_dyn_matmul(&mut m, "mm", 64, 64, 64, 0, 64, "x");
    m += "        tensor<int16, [1,64,1,64]> y = cast(x=mm_y, dtype=string(\"int16\"))[name=string(\"y\")];\n";
    m += &mil::mil_footer("y");
    m
}

// P21: input declared as int8
fn mil_p21() -> String {
    let mut m = format!(
        "program(1.3)\n[buildInfo = dict<string, string>({info})]\n{{\n    func main<ios18>(tensor<int8, [1, 64, 1, 128]> x) {{\n",
        info=BUILD_INFO,
    );
    // slice int8 activations
    m += "        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n";
    m += "        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<int8, [1,64,1,64]> a = slice_by_size(x=x, begin=ba, size=sa)[name=string(\"a\")];\n";
    m += "        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,64])];\n";
    m += "        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<int8, [1,64,1,64]> w = slice_by_size(x=x, begin=bw, size=sw)[name=string(\"w\")];\n";
    m += "        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,64,64])];\n";
    m += "        tensor<int8, [1,1,64,64]> a2 = reshape(shape=ra, x=a)[name=string(\"a2\")];\n";
    m += "        tensor<int8, [1,1,64,64]> W2 = reshape(shape=ra, x=w)[name=string(\"W2\")];\n";
    m += "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n";
    m += "        tensor<int32, [1,1,64,64]> yh = matmul(transpose_x=bF, transpose_y=bF, x=a2, y=W2)[name=string(\"yh\")];\n";
    m += "        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<int32, [1,64,1,64]> y = reshape(shape=ro, x=yh)[name=string(\"y\")];\n";
    m += &mil::mil_footer("y");
    m
}

// P22: matmul → fp_to_int_clamped (force integer cast on accumulator)
fn mil_p22() -> String {
    let mut m = header_simple(64, 128);
    mil::gen_dyn_matmul(&mut m, "mm", 64, 64, 64, 0, 64, "x");
    // Try fp_to_int (some MIL versions have this)
    m += "        tensor<int32, [1,64,1,64]> y = cast(x=mm_y, dtype=string(\"int32\"))[name=string(\"y\")];\n";
    m += &mil::mil_footer("y");
    m
}

// P23: constexpr_affine_dequantize per-tensor (no axis)
fn mil_p23() -> String {
    let mut m = header_simple(64, 128);
    m += "        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n";
    m += "        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp16, [1,64,1,64]> a = slice_by_size(x=x, begin=ba, size=sa)[name=string(\"a\")];\n";
    // int8 weight const (small)
    let weight_vals = vec!["1"; 64 * 64].join(",");
    m += &format!("        tensor<int8, [64,64]> wq = const()[name=string(\"wq\"), val=tensor<int8, [64,64]>([{weight_vals}])];\n");
    // Per-tensor: scalar scale, scalar zero_point
    m += "        tensor<fp16, []> sc = const()[name=string(\"sc\"), val=fp16(1.0)];\n";
    m += "        tensor<int8, []> zp = const()[name=string(\"zp\"), val=int8(0)];\n";
    m += "        tensor<fp16, [64,64]> wf = constexpr_affine_dequantize(quantized_data=wq, zero_point=zp, scale=sc)[name=string(\"wf\")];\n";
    m += "        tensor<int32, [2]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [2]>([64,64])];\n";
    m += "        tensor<fp16, [64,64]> a2 = reshape(shape=rsh, x=a)[name=string(\"a2\")];\n";
    m += "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n";
    m += "        tensor<fp16, [64,64]> ym = matmul(transpose_x=bF, transpose_y=bF, x=a2, y=wf)[name=string(\"ym\")];\n";
    m += "        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp16, [1,64,1,64]> y = reshape(shape=ro, x=ym)[name=string(\"y\")];\n";
    m += &mil::mil_footer("y");
    m
}

// P24: per-channel (rank-1 scale and zero_point)
fn mil_p24() -> String {
    let mut m = header_simple(64, 128);
    m += "        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n";
    m += "        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp16, [1,64,1,64]> a = slice_by_size(x=x, begin=ba, size=sa)[name=string(\"a\")];\n";
    let weight_vals = vec!["1"; 64 * 64].join(",");
    m += &format!("        tensor<int8, [64,64]> wq = const()[name=string(\"wq\"), val=tensor<int8, [64,64]>([{weight_vals}])];\n");
    // Per-channel: scale and zero_point have shape [64] (one per output channel)
    let scale_vals = vec!["1.0"; 64].join(",");
    let zp_vals = vec!["0"; 64].join(",");
    m += &format!("        tensor<fp16, [64]> sc = const()[name=string(\"sc\"), val=tensor<fp16, [64]>([{scale_vals}])];\n");
    m += &format!("        tensor<int8, [64]> zp = const()[name=string(\"zp\"), val=tensor<int8, [64]>([{zp_vals}])];\n");
    m += "        tensor<fp16, [64,64]> wf = constexpr_affine_dequantize(quantized_data=wq, zero_point=zp, scale=sc, axis=int32(0))[name=string(\"wf\")];\n";
    m += "        tensor<int32, [2]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [2]>([64,64])];\n";
    m += "        tensor<fp16, [64,64]> a2 = reshape(shape=rsh, x=a)[name=string(\"a2\")];\n";
    m += "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n";
    m += "        tensor<fp16, [64,64]> ym = matmul(transpose_x=bF, transpose_y=bF, x=a2, y=wf)[name=string(\"ym\")];\n";
    m += "        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp16, [1,64,1,64]> y = reshape(shape=ro, x=ym)[name=string(\"y\")];\n";
    m += &mil::mil_footer("y");
    m
}

// P25: matmul with int8 output declared directly (skip cast)
fn mil_p25() -> String {
    let mut m = header_simple(64, 128);
    // slice activations and weights
    m += "        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n";
    m += "        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp16, [1,64,1,64]> a = slice_by_size(x=x, begin=ba, size=sa)[name=string(\"a\")];\n";
    m += "        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,64])];\n";
    m += "        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp16, [1,64,1,64]> w = slice_by_size(x=x, begin=bw, size=sw)[name=string(\"w\")];\n";
    m += "        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,64,64])];\n";
    m += "        tensor<fp16, [1,1,64,64]> a2 = reshape(shape=ra, x=a)[name=string(\"a2\")];\n";
    m += "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n";
    m += "        tensor<fp16, [1,1,64,64]> a3 = transpose(perm=pm, x=a2)[name=string(\"a3\")];\n";
    m += "        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,64,64])];\n";
    m += "        tensor<fp16, [1,1,64,64]> W = reshape(shape=rw, x=w)[name=string(\"W\")];\n";
    m += "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n";
    // OUTPUT TYPE INT8 directly on the matmul declaration
    m += "        tensor<int8, [1,1,64,64]> yh = matmul(transpose_x=bF, transpose_y=bF, x=a3, y=W)[name=string(\"yh\")];\n";
    m += "        tensor<int8, [1,1,64,64]> yt = transpose(perm=pm, x=yh)[name=string(\"yt\")];\n";
    m += "        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<int8, [1,64,1,64]> y = reshape(shape=ro, x=yt)[name=string(\"y\")];\n";
    m += &mil::mil_footer("y");
    m
}

// P26: matmul with output_dtype attribute
fn mil_p26() -> String {
    let mut m = header_simple(64, 128);
    m += "        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n";
    m += "        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp16, [1,64,1,64]> a = slice_by_size(x=x, begin=ba, size=sa)[name=string(\"a\")];\n";
    m += "        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,64])];\n";
    m += "        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp16, [1,64,1,64]> w = slice_by_size(x=x, begin=bw, size=sw)[name=string(\"w\")];\n";
    m += "        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,64,64])];\n";
    m += "        tensor<fp16, [1,1,64,64]> a2 = reshape(shape=ra, x=a)[name=string(\"a2\")];\n";
    m += "        tensor<fp16, [1,1,64,64]> W = reshape(shape=ra, x=w)[name=string(\"W\")];\n";
    m += "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n";
    // matmul with output_dtype attribute (in brackets)
    m += "        tensor<int8, [1,1,64,64]> yh = matmul(transpose_x=bF, transpose_y=bF, x=a2, y=W)[name=string(\"yh\"), output_dtype=string(\"int8\")];\n";
    m += "        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<int8, [1,64,1,64]> y = reshape(shape=ro, x=yh)[name=string(\"y\")];\n";
    m += &mil::mil_footer("y");
    m
}

// P27: matmul → mul(1/256) → cast(int8) — extracts byte_1 if int8 cast does TRUE modular wrap on the high accumulator
fn mil_p27() -> String {
    let mut m = header_simple(64, 128);
    mil::gen_dyn_matmul(&mut m, "mm", 64, 64, 64, 0, 64, "x");
    // mul by 1/256 = 0.00390625
    m +=
        "        tensor<fp16, []> scale = const()[name=string(\"scale\"), val=fp16(0.00390625)];\n";
    m += "        tensor<fp16, [1,64,1,64]> scaled = mul(x=mm_y, y=scale)[name=string(\"scaled\")];\n";
    m += "        tensor<int8, [1,64,1,64]> y = cast(x=scaled, dtype=string(\"int8\"))[name=string(\"y\")];\n";
    m += &mil::mil_footer("y");
    m
}

// P28: matmul → mul(1/65536) → cast(int8)
fn mil_p28() -> String {
    let mut m = header_simple(64, 128);
    mil::gen_dyn_matmul(&mut m, "mm", 64, 64, 64, 0, 64, "x");
    m += "        tensor<fp16, []> scale = const()[name=string(\"scale\"), val=fp16(0.0000152587890625)];\n";
    m += "        tensor<fp16, [1,64,1,64]> scaled = mul(x=mm_y, y=scale)[name=string(\"scaled\")];\n";
    m += "        tensor<int8, [1,64,1,64]> y = cast(x=scaled, dtype=string(\"int8\"))[name=string(\"y\")];\n";
    m += &mil::mil_footer("y");
    m
}

// P5: 2 matmuls → concat (along channels) → reduce_sum (channel axis)
fn mil_p5() -> String {
    let mut m = header_simple(64, 128);
    mil::gen_dyn_matmul_chunk(&mut m, "c0", 0, 32, 64, 64, 64, "x");
    mil::gen_dyn_matmul_chunk(&mut m, "c1", 32, 32, 64, 64, 64, "x");
    // concat along channel axis: [1,64,1,64] + [1,64,1,64] → [1,128,1,64]
    m += "        tensor<fp16, [1,128,1,64]> cc = concat(values=(c0_y, c1_y), axis=int32(1), interleave=bool(false))[name=string(\"cc\")];\n";
    // reshape to [1,2,64,64] for reduce_sum
    m += "        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,2,64,64])];\n";
    m += "        tensor<fp16, [1,2,64,64]> r2 = reshape(shape=rsh, x=cc)[name=string(\"r2\")];\n";
    // reduce_sum along axis 1
    m += "        tensor<int32, [1]> axes = const()[name=string(\"axes\"), val=tensor<int32, [1]>([1])];\n";
    m += "        bool kd = const()[name=string(\"kd\"), val=bool(false)];\n";
    m += "        tensor<fp16, [1,64,64]> rs = reduce_sum(x=r2, axes=axes, keep_dims=kd)[name=string(\"rs\")];\n";
    m += "        tensor<int32, [4]> rsho = const()[name=string(\"rsho\"), val=tensor<int32, [4]>([1,64,1,64])];\n";
    m += "        tensor<fp16, [1,64,1,64]> y = reshape(shape=rsho, x=rs)[name=string(\"y\")];\n";
    m += "        tensor<fp32, [1,64,1,64]> yf = cast(x=y, dtype=string(\"fp32\"))[name=string(\"yf\")];\n";
    m += &mil::mil_footer("yf");
    m
}
