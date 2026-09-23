//! Probe constexpr_affine_dequantize syntax variants.
//!
//! Probes A-F: inline/BLOBFILE variants, scalar type forms, ios18.
//! Probes G-J: Apple exact syntax (tensor<T,[]> rank-0 forms, ios16/ios18).
//!
//! Key hypothesis from binary RE: real Apple models use tensor<T,[]>() for ALL
//! scalar parameters, not int32(0)/string("x") shorthand. Also BLOBFILE for all
//! weight data; no inline arrays.
//!
//! Run: cargo run -p rane --example quant_probe --release

use rane::{mil, Buffer, Program};

const BUILD_INFO: &str = concat!(
    "{{\"coremlc-component-MIL\", \"3510.2.1\"}, ",
    "{\"coremlc-version\", \"3505.4.1\"}, ",
    "{\"coremltools-component-milinternal\", \"\"}, ",
    "{\"coremltools-version\", \"9.0\"}}",
);

fn hdr(ic: usize, sp: usize) -> String {
    format!(
        "program(1.3)\n[buildInfo = dict<string, string>({info})]\n{{\n    func main<ios18>(tensor<fp16, [1, {ic}, 1, {sp}]> x) {{\n",
        info = BUILD_INFO,
        ic = ic,
        sp = sp,
    )
}

fn probe(label: &str, mil_text: &str, input_size: usize, output_size: usize) {
    let src = rane::Source {
        text: mil_text.to_string(),
        input_channels: 64,
        input_spatial: input_size,
        output_channels: 64,
        output_spatial: 64,
        output_dtype: rane::OutputDtype::Fp16,
    };
    match Program::compile(&src, &[]) {
        Ok(mut m) => match m.load() {
            Ok(()) => {
                let inp = Buffer::new(input_size * 2).unwrap();
                let out = Buffer::new(output_size).unwrap();
                inp.write(|d| {
                    for v in d.iter_mut() {
                        *v = rane::f32_to_fp16(1.0);
                    }
                });
                match m.run(&inp, &out) {
                    Ok(()) => {
                        let val = out.read(|d| rane::fp16_to_f32(d[0]));
                        println!("  [{}] ✓ COMPILED+RAN  first_output={:.1}", label, val);
                    }
                    Err(e) => println!(
                        "  [{}] compiled+loaded, RUN FAILED: {}",
                        label,
                        &format!("{e}")[..80.min(format!("{e}").len())]
                    ),
                }
            }
            Err(e) => println!(
                "  [{}] compiled, LOAD FAILED: {}",
                label,
                &format!("{e}")[..80.min(format!("{e}").len())]
            ),
        },
        Err(e) => {
            println!("  [{}] COMPILE FAILED:\n    {}", label, e);
        }
    }
}

fn main() {
    println!("constexpr_affine_dequantize syntax probe\n");

    // ic=oc=seq=64, input spatial = seq+oc = 128 (activation only, weight is constexpr)
    let ic = 64usize;
    let oc = 64usize;
    let seq = 64usize;
    let act_sp = seq; // activation spatial = seq tokens

    // Build the 64×64 weight values as a string of 1s
    let w_vals: String = (0..ic * oc).map(|_| "1").collect::<Vec<_>>().join(",");
    let sc_vals: String = (0..oc).map(|_| "1.0").collect::<Vec<_>>().join(",");
    let zp_vals: String = (0..oc).map(|_| "0").collect::<Vec<_>>().join(",");

    // --- Probe A: constexpr_affine_dequantize with ALL DATA as ATTRIBUTES ([] block) ---
    // Per-tensor scale, no axis → simplest case
    {
        let mut m = hdr(ic, act_sp);
        // Activation: reshape x from [1,ic,1,seq] → [1,1,ic,seq] → transpose → [1,1,seq,ic]
        m += &format!("        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,{ic},{seq}])];\n");
        m += &format!("        tensor<fp16, [1,1,{ic},{seq}]> a2 = reshape(shape=ra, x=x)[name=string(\"a2\")];\n");
        m += "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n";
        m += &format!("        tensor<fp16, [1,1,{seq},{ic}]> a3 = transpose(perm=pm, x=a2)[name=string(\"a3\")];\n");
        // Weight: constexpr_affine_dequantize with inline attributes, shape [ic,oc]
        m += &format!(
            "        tensor<fp16, [{ic},{oc}]> wf = constexpr_affine_dequantize()[\
name=string(\"wf\"), \
quantized_data=tensor<int8, [{ic},{oc}]>([{w_vals}]), \
zero_point=int8(0), \
scale=fp16(1.0)];\n"
        );
        // Reshape weight to [1,1,ic,oc]
        m += &format!("        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,{ic},{oc}])];\n");
        m += &format!("        tensor<fp16, [1,1,{ic},{oc}]> W = reshape(shape=rw, x=wf)[name=string(\"W\")];\n");
        m += "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n";
        m += &format!("        tensor<fp16, [1,1,{seq},{oc}]> yh = matmul(transpose_x=bF, transpose_y=bF, x=a3, y=W)[name=string(\"yh\")];\n");
        m += &format!("        tensor<fp16, [1,1,{oc},{seq}]> yt = transpose(perm=pm, x=yh)[name=string(\"yt\")];\n");
        m += &format!("        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,{oc},1,{seq}])];\n");
        m += &format!("        tensor<fp16, [1,{oc},1,{seq}]> y = reshape(shape=ro, x=yt)[name=string(\"y\")];\n");
        m += &mil::mil_footer("y");
        probe(
            "A: constexpr inline attrs per-tensor",
            &m,
            ic * act_sp,
            oc * seq * 2,
        );
    }

    // --- Probe B: per-channel scale (one scale per output channel) ---
    {
        let mut m = hdr(ic, act_sp);
        m += &format!("        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,{ic},{seq}])];\n");
        m += &format!("        tensor<fp16, [1,1,{ic},{seq}]> a2 = reshape(shape=ra, x=x)[name=string(\"a2\")];\n");
        m += "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n";
        m += &format!("        tensor<fp16, [1,1,{seq},{ic}]> a3 = transpose(perm=pm, x=a2)[name=string(\"a3\")];\n");
        // Per-channel: scale shape [oc], zero_point shape [oc], axis=0
        m += &format!(
            "        tensor<fp16, [{ic},{oc}]> wf = constexpr_affine_dequantize()[\
name=string(\"wf\"), \
quantized_data=tensor<int8, [{ic},{oc}]>([{w_vals}]), \
zero_point=tensor<int8, [{oc}]>([{zp_vals}]), \
scale=tensor<fp16, [{oc}]>([{sc_vals}]), \
axis=int32(0)];\n"
        );
        m += &format!("        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,{ic},{oc}])];\n");
        m += &format!("        tensor<fp16, [1,1,{ic},{oc}]> W = reshape(shape=rw, x=wf)[name=string(\"W\")];\n");
        m += "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n";
        m += &format!("        tensor<fp16, [1,1,{seq},{oc}]> yh = matmul(transpose_x=bF, transpose_y=bF, x=a3, y=W)[name=string(\"yh\")];\n");
        m += &format!("        tensor<fp16, [1,1,{oc},{seq}]> yt = transpose(perm=pm, x=yh)[name=string(\"yt\")];\n");
        m += &format!("        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,{oc},1,{seq}])];\n");
        m += &format!("        tensor<fp16, [1,{oc},1,{seq}]> y = reshape(shape=ro, x=yt)[name=string(\"y\")];\n");
        m += &mil::mil_footer("y");
        probe(
            "B: constexpr inline attrs per-channel",
            &m,
            ic * act_sp,
            oc * seq * 2,
        );
    }

    // --- Probe C: weight shape [oc,ic] with axis=1 (per-input-channel = likely FAILS per binary) ---
    {
        let w_vals_t: String = (0..ic * oc).map(|_| "1").collect::<Vec<_>>().join(",");
        let sc_ic: String = (0..ic).map(|_| "1.0").collect::<Vec<_>>().join(",");
        let zp_ic: String = (0..ic).map(|_| "0").collect::<Vec<_>>().join(",");
        let mut m = hdr(ic, act_sp);
        m += &format!("        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,{ic},{seq}])];\n");
        m += &format!("        tensor<fp16, [1,1,{ic},{seq}]> a2 = reshape(shape=ra, x=x)[name=string(\"a2\")];\n");
        m += "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n";
        m += &format!("        tensor<fp16, [1,1,{seq},{ic}]> a3 = transpose(perm=pm, x=a2)[name=string(\"a3\")];\n");
        // Weight shape [oc,ic], axis=1 (per input channel)
        m += &format!(
            "        tensor<fp16, [{oc},{ic}]> wf = constexpr_affine_dequantize()[\
name=string(\"wf\"), \
quantized_data=tensor<int8, [{oc},{ic}]>([{w_vals_t}]), \
zero_point=tensor<int8, [{ic}]>([{zp_ic}]), \
scale=tensor<fp16, [{ic}]>([{sc_ic}]), \
axis=int32(1)];\n"
        );
        // Transpose weight [oc,ic] → [ic,oc] then reshape to [1,1,ic,oc]
        m += &format!("        tensor<int32, [2]> pm2 = const()[name=string(\"pm2\"), val=tensor<int32, [2]>([1,0])];\n");
        m += &format!("        tensor<fp16, [{ic},{oc}]> wft = transpose(perm=pm2, x=wf)[name=string(\"wft\")];\n");
        m += &format!("        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,{ic},{oc}])];\n");
        m += &format!("        tensor<fp16, [1,1,{ic},{oc}]> W = reshape(shape=rw, x=wft)[name=string(\"W\")];\n");
        m += "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n";
        m += &format!("        tensor<fp16, [1,1,{seq},{oc}]> yh = matmul(transpose_x=bF, transpose_y=bF, x=a3, y=W)[name=string(\"yh\")];\n");
        m += &format!("        tensor<fp16, [1,1,{oc},{seq}]> yt = transpose(perm=pm, x=yh)[name=string(\"yt\")];\n");
        m += &format!("        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,{oc},1,{seq}])];\n");
        m += &format!("        tensor<fp16, [1,{oc},1,{seq}]> y = reshape(shape=ro, x=yt)[name=string(\"y\")];\n");
        m += &mil::mil_footer("y");
        probe(
            "C: per-input-channel (likely fails)",
            &m,
            ic * act_sp,
            oc * seq * 2,
        );
    }

    // --- Probe D: just constexpr_affine_dequantize + identity output (no matmul) ---
    // Test if constexpr_affine_dequantize itself compiles at all
    {
        let w_small: String = (0..16)
            .map(|i| (i as i8).to_string())
            .collect::<Vec<_>>()
            .join(",");
        let mut m = hdr(4, 4);
        m += &format!(
            "        tensor<fp16, [4,4]> wf = constexpr_affine_dequantize()[\
name=string(\"wf\"), \
quantized_data=tensor<int8, [4,4]>([{w_small}]), \
zero_point=int8(0), \
scale=fp16(1.0)];\n"
        );
        m += "        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,4,1,4])];\n";
        m += "        tensor<fp16, [1,4,1,4]> y = reshape(shape=ro, x=wf)[name=string(\"y\")];\n";
        m += &mil::mil_footer("y");
        probe(
            "D: standalone constexpr_affine_dequantize",
            &m,
            4 * 4 * 2,
            4 * 4 * 2,
        );
    }

    // --- Probe E: no reshape between dequant and matmul — direct connection ---
    // Maybe the reshape breaks the pattern match
    {
        let mut m = hdr(ic, act_sp);
        m += &format!("        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,{ic},{seq}])];\n");
        m += &format!("        tensor<fp16, [1,1,{ic},{seq}]> a2 = reshape(shape=ra, x=x)[name=string(\"a2\")];\n");
        m += "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n";
        m += &format!("        tensor<fp16, [1,1,{seq},{ic}]> a3 = transpose(perm=pm, x=a2)[name=string(\"a3\")];\n");
        // Weight directly in 4D form — can constexpr produce 4D?
        m += &format!(
            "        tensor<fp16, [1,1,{ic},{oc}]> W = constexpr_affine_dequantize()[\
name=string(\"W\"), \
quantized_data=tensor<int8, [1,1,{ic},{oc}]>([{w_vals}]), \
zero_point=int8(0), \
scale=fp16(1.0)];\n"
        );
        m += "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n";
        m += &format!("        tensor<fp16, [1,1,{seq},{oc}]> yh = matmul(transpose_x=bF, transpose_y=bF, x=a3, y=W)[name=string(\"yh\")];\n");
        m += &format!("        tensor<fp16, [1,1,{oc},{seq}]> yt = transpose(perm=pm, x=yh)[name=string(\"yt\")];\n");
        m += &format!("        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,{oc},1,{seq}])];\n");
        m += &format!("        tensor<fp16, [1,{oc},1,{seq}]> y = reshape(shape=ro, x=yt)[name=string(\"y\")];\n");
        m += &mil::mil_footer("y");
        probe(
            "E: constexpr 4D weight directly into matmul",
            &m,
            ic * act_sp,
            oc * seq * 2,
        );
    }

    // --- Probe F: use weight blob via weights parameter to Program::compile ---
    // constexpr_affine_dequantize with file_value (loaded from weight file)
    // This mimics how CoreML actually stores large weights in separate files
    probe_with_weight_file(ic, oc, seq);

    println!("\n--- Apple-exact syntax probes (G-J) ---\n");
    // G: tensor<T,[]> rank-0 forms + BLOBFILE + ios16 (exact VoiceActions style)
    probe_apple_exact(
        ic,
        oc,
        seq,
        "G: tensor<T,[]> BLOBFILE ios16",
        "ios16",
        true,
        true,
    );
    // H: scalar forms (int32(0)/string()) + BLOBFILE + ios16
    probe_apple_exact(
        ic,
        oc,
        seq,
        "H: scalar forms BLOBFILE ios16",
        "ios16",
        false,
        true,
    );
    // I: tensor<T,[]> forms + BLOBFILE + ios18
    probe_apple_exact(
        ic,
        oc,
        seq,
        "I: tensor<T,[]> BLOBFILE ios18",
        "ios18",
        true,
        true,
    );
    // J: no name attr + scalar + BLOBFILE + ios16 (TTS fastspeech2 style)
    probe_apple_exact(
        ic,
        oc,
        seq,
        "J: no-name scalar BLOBFILE ios16",
        "ios16",
        false,
        false,
    );

    println!("\ndone.");
}

/// Probe with Apple's exact constexpr_affine_dequantize syntax from compiled system models.
///
/// `tensor_forms`: use tensor<T,[]>() for all scalar params (true) or shorthand int32()/string() (false)
/// `include_name`: include name= attribute inside brackets (true) or omit it (false)
fn probe_apple_exact(
    ic: usize,
    oc: usize,
    seq: usize,
    label: &str,
    opset: &str,
    tensor_forms: bool,
    include_name: bool,
) {
    // Weight blob layout (axis=0 per-output-channel quantization):
    //   offset 64:          int8 quantized_data [ic × oc] = ic*oc bytes
    //   offset 64+ic*oc:    fp16 scale [oc]               = oc*2 bytes
    //   offset 64+ic*oc+oc*2: int8 zero_point [oc]        = oc bytes
    let data_off: u64 = 64;
    let scale_off: u64 = data_off + (ic * oc) as u64;
    let zp_off: u64 = scale_off + (oc * 2) as u64;
    let blob_len = (zp_off as usize) + oc;

    let mut blob = vec![0u8; blob_len];
    // fill int8 quantized_data with 1
    for b in &mut blob[data_off as usize..scale_off as usize] {
        *b = 1i8 as u8;
    }
    // fill fp16 scale with 1.0 = 0x3C00
    let scale_end = (scale_off + (oc * 2) as u64) as usize;
    let mut i = scale_off as usize;
    while i < scale_end {
        blob[i] = 0x00;
        blob[i + 1] = 0x3C;
        i += 2;
    }
    // zero_point stays 0

    let w_path = "@model_path/weights/W8.bin";

    // Helper closures to produce correct syntax forms
    let s_str = |s: &str| -> String {
        if tensor_forms {
            format!("tensor<string, []>(\"{s}\")")
        } else {
            format!("string(\"{s}\")")
        }
    };
    let s_u64 = |v: u64| -> String {
        if tensor_forms {
            format!("tensor<uint64, []>({v})")
        } else {
            format!("uint64({v})")
        }
    };
    let s_i32 = |v: i32| -> String {
        if tensor_forms {
            format!("tensor<int32, []>({v})")
        } else {
            format!("int32({v})")
        }
    };

    let name_attr = if include_name {
        format!("name = {}, ", s_str("wf"))
    } else {
        String::new()
    };

    let build_info =
        "{{\"coremlc-component-MIL\", \"4.28.2\"}, {\"coremlc-version\", \"1436.0.14.0.1\"}}";

    let mut m = format!(
        "program(1.3)\n[buildInfo = dict<string, string>({build_info})]\n{{\n    func main<{opset}>(tensor<fp16, [1, {ic}, 1, {seq}]> x) {{\n"
    );
    // Reshape + transpose activation
    m += &format!("        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,{ic},{seq}])];\n");
    m += &format!("        tensor<fp16, [1,1,{ic},{seq}]> a2 = reshape(shape=ra, x=x)[name=string(\"a2\")];\n");
    m += "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n";
    m += &format!("        tensor<fp16, [1,1,{seq},{ic}]> a3 = transpose(perm=pm, x=a2)[name=string(\"a3\")];\n");

    // constexpr_affine_dequantize with per-output-channel quantization
    let w_path_val = s_str(w_path);
    let data_offset_val = s_u64(data_off);
    let scale_offset_val = s_u64(scale_off);
    let zp_offset_val = s_u64(zp_off);
    let axis_val = s_i32(0);

    m += &format!(
        "        tensor<fp16, [{ic},{oc}]> wf = constexpr_affine_dequantize()[\
{name_attr}\
axis = {axis_val}, \
quantized_data = tensor<int8, [{ic},{oc}]>(BLOBFILE(path = {w_path_val}, offset = {data_offset_val})), \
scale = tensor<fp16, [{oc}]>(BLOBFILE(path = {w_path_val}, offset = {scale_offset_val})), \
zero_point = tensor<int8, [{oc}]>(BLOBFILE(path = {w_path_val}, offset = {zp_offset_val}))\
];\n"
    );
    // reshape weight to 4D, matmul, reshape output
    m += &format!("        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,{ic},{oc}])];\n");
    m += &format!(
        "        tensor<fp16, [1,1,{ic},{oc}]> W = reshape(shape=rw, x=wf)[name=string(\"W\")];\n"
    );
    m += "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n";
    m += &format!("        tensor<fp16, [1,1,{seq},{oc}]> yh = matmul(transpose_x=bF, transpose_y=bF, x=a3, y=W)[name=string(\"yh\")];\n");
    m += &format!("        tensor<fp16, [1,1,{oc},{seq}]> yt = transpose(perm=pm, x=yh)[name=string(\"yt\")];\n");
    m += &format!("        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,{oc},1,{seq}])];\n");
    m += &format!(
        "        tensor<fp16, [1,{oc},1,{seq}]> y = reshape(shape=ro, x=yt)[name=string(\"y\")];\n"
    );
    m += &rane::mil::mil_footer("y");

    let src = rane::Source {
        text: m,
        input_channels: ic,
        input_spatial: seq,
        output_channels: oc,
        output_spatial: seq,
        output_dtype: rane::OutputDtype::Fp16,
    };

    match Program::compile(&src, &[(w_path, &blob)]) {
        Ok(mut model) => match model.load() {
            Ok(()) => {
                let inp = Buffer::new(ic * seq * 2).unwrap();
                let out = Buffer::new(oc * seq * 2).unwrap();
                inp.write(|d| {
                    for v in d.iter_mut() {
                        *v = rane::f32_to_fp16(1.0);
                    }
                });
                match model.run(&inp, &out) {
                    Ok(()) => {
                        let val = out.read(|d| rane::fp16_to_f32(d[0]));
                        println!("  [{}] ✓ COMPILED+RAN  first_output={:.1}", label, val);
                    }
                    Err(e) => println!("  [{}] compiled+loaded, RUN FAILED: {}", label, e),
                }
            }
            Err(e) => println!("  [{}] compiled, LOAD FAILED: {}", label, e),
        },
        Err(e) => println!("  [{}] COMPILE FAILED:\n    {}", label, e),
    }
}

fn probe_with_weight_file(ic: usize, oc: usize, seq: usize) {
    // Build a weight blob: 128-byte header (magic 0xDEADBEEF) + int8 data
    let n = ic * oc;
    let mut weight_blob: Vec<u8> = vec![0u8; 128 + n];
    // rane weight blob header: magic at offset 0
    let magic: u32 = 0xDEAD_BEEF;
    weight_blob[0..4].copy_from_slice(&magic.to_le_bytes());
    // Fill data with 1s starting at offset 128
    for b in &mut weight_blob[128..] {
        *b = 1i8 as u8;
    }

    let mut m = hdr(ic, seq);
    m += &format!("        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,{ic},{seq}])];\n");
    m += &format!("        tensor<fp16, [1,1,{ic},{seq}]> a2 = reshape(shape=ra, x=x)[name=string(\"a2\")];\n");
    m += "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n";
    m += &format!("        tensor<fp16, [1,1,{seq},{ic}]> a3 = transpose(perm=pm, x=a2)[name=string(\"a3\")];\n");
    // file_value syntax — weight data loaded from a weight blob file
    m += &format!(
        "        tensor<fp16, [{ic},{oc}]> wf = constexpr_affine_dequantize()[\
name=string(\"wf\"), \
quantized_data=tensor<int8, [{ic},{oc}]>(BLOBFILE(path=string(\"weights/wq.bin\"), offset=uint64(128))), \
zero_point=int8(0), \
scale=fp16(1.0)];\n"
    );
    m += &format!("        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,{ic},{oc}])];\n");
    m += &format!(
        "        tensor<fp16, [1,1,{ic},{oc}]> W = reshape(shape=rw, x=wf)[name=string(\"W\")];\n"
    );
    m += "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n";
    m += &format!("        tensor<fp16, [1,1,{seq},{oc}]> yh = matmul(transpose_x=bF, transpose_y=bF, x=a3, y=W)[name=string(\"yh\")];\n");
    m += &format!("        tensor<fp16, [1,1,{oc},{seq}]> yt = transpose(perm=pm, x=yh)[name=string(\"yt\")];\n");
    m += &format!("        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,{oc},1,{seq}])];\n");
    m += &format!(
        "        tensor<fp16, [1,{oc},1,{seq}]> y = reshape(shape=ro, x=yt)[name=string(\"y\")];\n"
    );
    m += &mil::mil_footer("y");

    let src = rane::Source {
        text: m,
        input_channels: ic,
        input_spatial: seq,
        output_channels: oc,
        output_spatial: seq,
        output_dtype: rane::OutputDtype::Fp16,
    };

    match Program::compile(&src, &[("weights/wq.bin", &weight_blob)]) {
        Ok(mut model) => match model.load() {
            Ok(()) => {
                let inp = Buffer::new(ic * seq * 2).unwrap();
                let out = Buffer::new(oc * seq * 2).unwrap();
                inp.write(|d| {
                    for v in d.iter_mut() {
                        *v = rane::f32_to_fp16(1.0);
                    }
                });
                match model.run(&inp, &out) {
                    Ok(()) => {
                        let val = out.read(|d| rane::fp16_to_f32(d[0]));
                        println!(
                            "  [F: BLOBFILE weight] ✓ COMPILED+RAN  first_output={:.1}",
                            val
                        );
                    }
                    Err(e) => println!(
                        "  [F: BLOBFILE weight] compiled, RUN FAILED: {}",
                        &format!("{e}")[..80.min(format!("{e}").len())]
                    ),
                }
            }
            Err(e) => println!(
                "  [F: BLOBFILE weight] compiled, LOAD FAILED: {}",
                &format!("{e}")[..80.min(format!("{e}").len())]
            ),
        },
        Err(e) => {
            println!("  [F: BLOBFILE weight] COMPILE FAILED:\n    {}", e);
        }
    }
}
