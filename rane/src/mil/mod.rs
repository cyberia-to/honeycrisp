//! MIL (Model Intermediate Language) program builder for ANE
//!
//! Generates MIL text for all ANE kernels used in transformer training/inference.

const MIL_BUILD_INFO: &str = concat!(
    "{{\"coremlc-component-MIL\", \"3510.2.1\"}, ",
    "{\"coremlc-version\", \"3505.4.1\"}, ",
    "{\"coremltools-component-milinternal\", \"\"}, ",
    "{\"coremltools-version\", \"9.0\"}}"
);

/// Output element type for a compiled MIL program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputDtype {
    Fp16,  // 2 bytes — default ANE matmul output
    Fp32,  // 4 bytes — software cast at end (still capped at fp16 max internally)
    Int32, // 4 bytes — currently rejected by ANECompile
    Int8,  // 1 byte  — ANE supports as final output
    UInt8, // 1 byte  — ANE supports as final output
    Int16, // 2 bytes — ANE supports as internal/intermediate
}

impl OutputDtype {
    pub fn elem_bytes(self) -> usize {
        match self {
            OutputDtype::Fp16 => 2,
            OutputDtype::Fp32 => 4,
            OutputDtype::Int32 => 4,
            OutputDtype::Int8 => 1,
            OutputDtype::UInt8 => 1,
            OutputDtype::Int16 => 2,
        }
    }
    pub fn mil_name(self) -> &'static str {
        match self {
            OutputDtype::Fp16 => "fp16",
            OutputDtype::Fp32 => "fp32",
            OutputDtype::Int32 => "int32",
            OutputDtype::Int8 => "int8",
            OutputDtype::UInt8 => "uint8",
            OutputDtype::Int16 => "int16",
        }
    }
}

/// A MIL program ready for ANE compilation.
pub struct Source {
    pub text: String,
    pub input_channels: usize,
    pub input_spatial: usize,
    pub output_channels: usize,
    pub output_spatial: usize,
    pub output_dtype: OutputDtype,
}

impl Source {
    pub fn as_str(&self) -> &str {
        &self.text
    }
    pub fn input_shape(&self) -> (usize, usize) {
        (self.input_channels, self.input_spatial)
    }
    pub fn output_shape(&self) -> (usize, usize) {
        (self.output_channels, self.output_spatial)
    }
    pub fn input_size(&self) -> usize {
        self.input_channels * self.input_spatial * 2
    }
    pub fn output_size(&self) -> usize {
        self.output_channels * self.output_spatial * self.output_dtype.elem_bytes()
    }
}

/// Start a MIL program with header and function signature.
pub fn mil_header(ic: usize, sp: usize) -> String {
    format!(
        "program(1.3)\n[buildInfo = dict<string, string>({info})]\n{{\n    func main<ios18>(tensor<fp16, [1, {ic}, 1, {sp}]> x) {{\n",
        info=MIL_BUILD_INFO, ic=ic, sp=sp,
    )
}

/// Close a MIL function with output variable.
pub fn mil_footer(output_var: &str) -> String {
    format!("    }} -> ({});\n}}\n", output_var)
}

/// Generate a dynamic matmul block within a MIL function.
/// Slices activations and weights from input, reshapes, transposes, matmuls.
/// Returns the output variable name "{prefix}_y".
pub fn gen_dyn_matmul(
    m: &mut String,
    prefix: &str,
    ic: usize,
    oc: usize,
    seq: usize,
    act_sp_off: usize,
    w_sp_off: usize,
    input_var: &str,
) {
    let p = prefix;
    let iv = input_var;
    *m += &format!("        tensor<int32, [4]> {p}_ba = const()[name=string(\"{p}_ba\"), val=tensor<int32, [4]>([0,0,0,{act_sp_off}])];\n");
    *m += &format!("        tensor<int32, [4]> {p}_sa = const()[name=string(\"{p}_sa\"), val=tensor<int32, [4]>([1,{ic},1,{seq}])];\n");
    *m += &format!("        tensor<fp16, [1,{ic},1,{seq}]> {p}_act = slice_by_size(x={iv},begin={p}_ba,size={p}_sa)[name=string(\"{p}_act\")];\n");
    *m += &format!("        tensor<int32, [4]> {p}_bw = const()[name=string(\"{p}_bw\"), val=tensor<int32, [4]>([0,0,0,{w_sp_off}])];\n");
    *m += &format!("        tensor<int32, [4]> {p}_sw = const()[name=string(\"{p}_sw\"), val=tensor<int32, [4]>([1,{ic},1,{oc}])];\n");
    *m += &format!("        tensor<fp16, [1,{ic},1,{oc}]> {p}_wt = slice_by_size(x={iv},begin={p}_bw,size={p}_sw)[name=string(\"{p}_wt\")];\n");
    *m += &format!("        tensor<int32, [4]> {p}_ra = const()[name=string(\"{p}_ra\"), val=tensor<int32, [4]>([1,1,{ic},{seq}])];\n");
    *m += &format!("        tensor<fp16, [1,1,{ic},{seq}]> {p}_a2 = reshape(shape={p}_ra,x={p}_act)[name=string(\"{p}_a2\")];\n");
    *m += &format!("        tensor<int32, [4]> {p}_pm = const()[name=string(\"{p}_pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n");
    *m += &format!("        tensor<fp16, [1,1,{seq},{ic}]> {p}_a3 = transpose(perm={p}_pm,x={p}_a2)[name=string(\"{p}_a3\")];\n");
    *m += &format!("        tensor<int32, [4]> {p}_rw = const()[name=string(\"{p}_rw\"), val=tensor<int32, [4]>([1,1,{ic},{oc}])];\n");
    *m += &format!("        tensor<fp16, [1,1,{ic},{oc}]> {p}_W = reshape(shape={p}_rw,x={p}_wt)[name=string(\"{p}_W\")];\n");
    *m += &format!("        bool {p}_bF = const()[name=string(\"{p}_bF\"), val=bool(false)];\n");
    *m += &format!("        tensor<fp16, [1,1,{seq},{oc}]> {p}_yh = matmul(transpose_x={p}_bF,transpose_y={p}_bF,x={p}_a3,y={p}_W)[name=string(\"{p}_yh\")];\n");
    *m += &format!("        tensor<fp16, [1,1,{oc},{seq}]> {p}_yt = transpose(perm={p}_pm,x={p}_yh)[name=string(\"{p}_yt\")];\n");
    *m += &format!("        tensor<int32, [4]> {p}_ro = const()[name=string(\"{p}_ro\"), val=tensor<int32, [4]>([1,{oc},1,{seq}])];\n");
    *m += &format!("        tensor<fp16, [1,{oc},1,{seq}]> {p}_y = reshape(shape={p}_ro,x={p}_yt)[name=string(\"{p}_y\")];\n");
}

/// Append a cast after the fp16 matmul output — same MIL tensor layout.
/// `{prefix}_y_cast` is the output variable name.
///
/// Tries two syntaxes to maximise compatibility:
///   A (positional): `cast(x=y, dtype=string("fp32"))[name=...]`
///   B (attribute):  `cast(x=y)[name=..., dtype=string("fp32")]`
///   C (inferred):   `cast(x=y)[name=...]`  ← type inferred from lhs declaration
pub fn gen_dyn_matmul_cast(
    m: &mut String,
    prefix: &str,
    oc: usize,
    seq: usize,
    dtype: OutputDtype,
) {
    if dtype == OutputDtype::Fp16 {
        return;
    }
    let p = prefix;
    let t = dtype.mil_name();
    // positional dtype parameter — this is the CoreMLTools Python API convention
    *m += &format!(
        "        tensor<{t}, [1,{oc},1,{seq}]> {p}_y_cast = \
         cast(x={p}_y, dtype=string(\"{t}\"))[name=string(\"{p}_y_cast\")];\n"
    );
}

/// Build a simple dynamic matmul MIL: y = x @ W
pub fn matmul(ic: usize, oc: usize, seq: usize) -> Source {
    let sp = seq + oc;
    let mut m = mil_header(ic, sp);
    gen_dyn_matmul(&mut m, "mm", ic, oc, seq, 0, seq, "x");
    m += &mil_footer("mm_y");
    Source {
        text: m,
        input_channels: ic,
        input_spatial: sp,
        output_channels: oc,
        output_spatial: seq,
        output_dtype: OutputDtype::Fp16,
    }
}

/// Generate a dynamic matmul that slices a CHANNEL-RANGE [ic_start..ic_start+ic_len)
/// out of the input x[1,ic_total,1,sp].  Used to split a large-ic matmul into
/// chunks whose fp16 accumulator stays below ANE's empirical ~32768 overflow.
/// Returns the output variable name "{prefix}_y" (fp16, shape [1,oc,1,seq]).
pub fn gen_dyn_matmul_chunk(
    m: &mut String,
    prefix: &str,
    ic_start: usize,
    ic_len: usize,
    oc: usize,
    seq: usize,
    w_sp_off: usize,
    input_var: &str,
) {
    let p = prefix;
    let iv = input_var;
    // act: slice channels [ic_start..ic_start+ic_len), spatial [0..seq)
    *m += &format!("        tensor<int32, [4]> {p}_ba = const()[name=string(\"{p}_ba\"), val=tensor<int32, [4]>([0,{ic_start},0,0])];\n");
    *m += &format!("        tensor<int32, [4]> {p}_sa = const()[name=string(\"{p}_sa\"), val=tensor<int32, [4]>([1,{ic_len},1,{seq}])];\n");
    *m += &format!("        tensor<fp16, [1,{ic_len},1,{seq}]> {p}_act = slice_by_size(x={iv},begin={p}_ba,size={p}_sa)[name=string(\"{p}_act\")];\n");
    // wt: slice channels [ic_start..ic_start+ic_len), spatial [w_sp_off..w_sp_off+oc)
    *m += &format!("        tensor<int32, [4]> {p}_bw = const()[name=string(\"{p}_bw\"), val=tensor<int32, [4]>([0,{ic_start},0,{w_sp_off}])];\n");
    *m += &format!("        tensor<int32, [4]> {p}_sw = const()[name=string(\"{p}_sw\"), val=tensor<int32, [4]>([1,{ic_len},1,{oc}])];\n");
    *m += &format!("        tensor<fp16, [1,{ic_len},1,{oc}]> {p}_wt = slice_by_size(x={iv},begin={p}_bw,size={p}_sw)[name=string(\"{p}_wt\")];\n");
    // reshape + transpose + matmul
    *m += &format!("        tensor<int32, [4]> {p}_ra = const()[name=string(\"{p}_ra\"), val=tensor<int32, [4]>([1,1,{ic_len},{seq}])];\n");
    *m += &format!("        tensor<fp16, [1,1,{ic_len},{seq}]> {p}_a2 = reshape(shape={p}_ra,x={p}_act)[name=string(\"{p}_a2\")];\n");
    *m += &format!("        tensor<int32, [4]> {p}_pm = const()[name=string(\"{p}_pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n");
    *m += &format!("        tensor<fp16, [1,1,{seq},{ic_len}]> {p}_a3 = transpose(perm={p}_pm,x={p}_a2)[name=string(\"{p}_a3\")];\n");
    *m += &format!("        tensor<int32, [4]> {p}_rw = const()[name=string(\"{p}_rw\"), val=tensor<int32, [4]>([1,1,{ic_len},{oc}])];\n");
    *m += &format!("        tensor<fp16, [1,1,{ic_len},{oc}]> {p}_W = reshape(shape={p}_rw,x={p}_wt)[name=string(\"{p}_W\")];\n");
    *m += &format!("        bool {p}_bF = const()[name=string(\"{p}_bF\"), val=bool(false)];\n");
    *m += &format!("        tensor<fp16, [1,1,{seq},{oc}]> {p}_yh = matmul(transpose_x={p}_bF,transpose_y={p}_bF,x={p}_a3,y={p}_W)[name=string(\"{p}_yh\")];\n");
    *m += &format!("        tensor<fp16, [1,1,{oc},{seq}]> {p}_yt = transpose(perm={p}_pm,x={p}_yh)[name=string(\"{p}_yt\")];\n");
    *m += &format!("        tensor<int32, [4]> {p}_ro = const()[name=string(\"{p}_ro\"), val=tensor<int32, [4]>([1,{oc},1,{seq}])];\n");
    *m += &format!("        tensor<fp16, [1,{oc},1,{seq}]> {p}_y = reshape(shape={p}_ro,x={p}_yt)[name=string(\"{p}_y\")];\n");
}

/// Variant: split matmul, sum in FP16 (output stays fp16).
/// Tests whether the split structure compiles on ANE regardless of dtype.
pub fn matmul_split_fp16(ic: usize, oc: usize, seq: usize, n_chunks: usize) -> Source {
    assert!(n_chunks >= 1);
    let sp = seq + oc;
    let mut m = mil_header(ic, sp);
    let chunk_len = ic.div_ceil(n_chunks);
    let mut chunks: Vec<String> = Vec::new();
    let mut offset = 0;
    for i in 0..n_chunks {
        if offset >= ic {
            break;
        }
        let this_len = chunk_len.min(ic - offset);
        let prefix = format!("c{i}");
        gen_dyn_matmul_chunk(&mut m, &prefix, offset, this_len, oc, seq, seq, "x");
        chunks.push(format!("{prefix}_y"));
        offset += this_len;
    }
    let mut current = chunks[0].clone();
    for i in 1..chunks.len() {
        let next = &chunks[i];
        let out = format!("s{i}");
        m += &format!(
            "        tensor<fp16, [1,{oc},1,{seq}]> {out} = add(x={current}, y={next})[name=string(\"{out}\")];\n"
        );
        current = out;
    }
    m += &mil_footer(&current);
    Source {
        text: m,
        input_channels: ic,
        input_spatial: sp,
        output_channels: oc,
        output_spatial: seq,
        output_dtype: OutputDtype::Fp16,
    }
}

/// Variant: split + sum in fp16 + ONE final cast to fp32 (no fp32 add).
/// Tests whether terminal-only fp32 cast can produce values > fp16 max.
/// Note: fp16 intermediate sums must stay ≤ 65504 (else they saturate before cast).
pub fn matmul_split_fp16_cast(ic: usize, oc: usize, seq: usize, n_chunks: usize) -> Source {
    assert!(n_chunks >= 1);
    let sp = seq + oc;
    let mut m = mil_header(ic, sp);
    let chunk_len = ic.div_ceil(n_chunks);
    let mut chunks: Vec<String> = Vec::new();
    let mut offset = 0;
    for i in 0..n_chunks {
        if offset >= ic {
            break;
        }
        let this_len = chunk_len.min(ic - offset);
        let prefix = format!("c{i}");
        gen_dyn_matmul_chunk(&mut m, &prefix, offset, this_len, oc, seq, seq, "x");
        chunks.push(format!("{prefix}_y"));
        offset += this_len;
    }
    let mut current = chunks[0].clone();
    for i in 1..chunks.len() {
        let next = &chunks[i];
        let out = format!("s{i}");
        m += &format!(
            "        tensor<fp16, [1,{oc},1,{seq}]> {out} = add(x={current}, y={next})[name=string(\"{out}\")];\n"
        );
        current = out;
    }
    // Final cast to fp32 for output
    let out_var = "y_f32".to_string();
    m += &format!(
        "        tensor<fp32, [1,{oc},1,{seq}]> {out_var} = cast(x={current}, dtype=string(\"fp32\"))[name=string(\"{out_var}\")];\n"
    );
    m += &mil_footer(&out_var);
    Source {
        text: m,
        input_channels: ic,
        input_spatial: sp,
        output_channels: oc,
        output_spatial: seq,
        output_dtype: OutputDtype::Fp32,
    }
}

/// Variant: split + CONCAT chunks → reduce_sum → terminal fp32 cast.
/// Tests whether ANE's `reduce_sum` has an internal fp32 accumulator that bypasses
/// fp16 max (65504) even when the final cast is fp32.
pub fn matmul_split_reduce(ic: usize, oc: usize, seq: usize, n_chunks: usize) -> Source {
    assert!(n_chunks >= 1);
    let sp = seq + oc;
    let mut m = mil_header(ic, sp);
    let chunk_len = ic.div_ceil(n_chunks);
    let mut chunks: Vec<String> = Vec::new();
    let mut offset = 0;
    for i in 0..n_chunks {
        if offset >= ic {
            break;
        }
        let this_len = chunk_len.min(ic - offset);
        let prefix = format!("c{i}");
        gen_dyn_matmul_chunk(&mut m, &prefix, offset, this_len, oc, seq, seq, "x");
        chunks.push(format!("{prefix}_y"));
        offset += this_len;
    }
    let n_actual = chunks.len();
    // concat chunks along channel axis (axis=1): each is [1,oc,1,seq] → [1,n*oc,1,seq]
    let concat_args = chunks.join(", ");
    m += &format!(
        "        tensor<fp16, [1,{ch},1,{seq}]> cc = concat(values=({concat_args}), axis=int32(1), interleave=bool(false))[name=string(\"cc\")];\n",
        ch = n_actual * oc,
    );
    // reshape to [1,n_actual,oc,seq] so we can reduce along axis 1
    m += &format!(
        "        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,{n_actual},{oc},{seq}])];\n"
    );
    m += &format!(
        "        tensor<fp16, [1,{n_actual},{oc},{seq}]> r2 = reshape(shape=rsh, x=cc)[name=string(\"r2\")];\n"
    );
    m += "        tensor<int32, [1]> axes = const()[name=string(\"axes\"), val=tensor<int32, [1]>([1])];\n";
    m += "        bool kd = const()[name=string(\"kd\"), val=bool(false)];\n";
    m += &format!(
        "        tensor<fp16, [1,{oc},{seq}]> rs = reduce_sum(x=r2, axes=axes, keep_dims=kd)[name=string(\"rs\")];\n"
    );
    m += &format!(
        "        tensor<int32, [4]> rsho = const()[name=string(\"rsho\"), val=tensor<int32, [4]>([1,{oc},1,{seq}])];\n"
    );
    m += &format!(
        "        tensor<fp16, [1,{oc},1,{seq}]> rr = reshape(shape=rsho, x=rs)[name=string(\"rr\")];\n"
    );
    m += &format!(
        "        tensor<fp32, [1,{oc},1,{seq}]> y = cast(x=rr, dtype=string(\"fp32\"))[name=string(\"y\")];\n"
    );
    m += &mil_footer("y");
    Source {
        text: m,
        input_channels: ic,
        input_spatial: sp,
        output_channels: oc,
        output_spatial: seq,
        output_dtype: OutputDtype::Fp32,
    }
}

/// Build a SPLIT matmul: divide ic into `n_chunks` and sum chunk outputs in fp32.
///
/// Each chunk's fp16 tile must stay below ANE's accumulator limit (~32768).
/// Chunks are cast to fp32 and summed pairwise → final fp32 output can exceed 32768.
///
/// This is the workaround for ANE's fp16 matmul accumulator saturating below
/// the theoretical fp16 max — by splitting into smaller matmuls and summing in
/// fp32, the per-tile fp16 limit doesn't constrain the final result.
pub fn matmul_split(ic: usize, oc: usize, seq: usize, n_chunks: usize) -> Source {
    assert!(n_chunks >= 1);
    let sp = seq + oc;
    let mut m = mil_header(ic, sp);

    // chunk_len = ceil(ic / n_chunks), rounded up
    let chunk_len = ic.div_ceil(n_chunks);
    let mut chunks: Vec<(String, usize)> = Vec::new(); // (var_name, actual_chunk_len)
    let mut offset = 0;
    for i in 0..n_chunks {
        if offset >= ic {
            break;
        }
        let this_len = chunk_len.min(ic - offset);
        let prefix = format!("c{i}");
        gen_dyn_matmul_chunk(&mut m, &prefix, offset, this_len, oc, seq, seq, "x");
        // cast fp16 → fp32
        m += &format!(
            "        tensor<fp32, [1,{oc},1,{seq}]> {prefix}_f = cast(x={prefix}_y, dtype=string(\"fp32\"))[name=string(\"{prefix}_f\")];\n"
        );
        chunks.push((format!("{prefix}_f"), this_len));
        offset += this_len;
    }

    // Pairwise fp32 sum
    let mut current = chunks[0].0.clone();
    for i in 1..chunks.len() {
        let next = &chunks[i].0;
        let out = format!("s{i}");
        m += &format!(
            "        tensor<fp32, [1,{oc},1,{seq}]> {out} = add(x={current}, y={next})[name=string(\"{out}\")];\n"
        );
        current = out;
    }

    m += &mil_footer(&current);
    Source {
        text: m,
        input_channels: ic,
        input_spatial: sp,
        output_channels: oc,
        output_spatial: seq,
        output_dtype: OutputDtype::Fp32,
    }
}

/// Build a matmul MIL that casts the fp16 output to the requested dtype.
///
/// If ANE accumulates in fp32 internally, `Fp32` and `Int32` variants extract
/// the exact accumulator value — bypassing the fp16 quantization-to-memory step.
/// Use this to verify whether the ANE can produce exact integer results for
/// values that overflow fp16 (> 65504).
pub fn matmul_cast(ic: usize, oc: usize, seq: usize, dtype: OutputDtype) -> Source {
    if dtype == OutputDtype::Fp16 {
        return matmul(ic, oc, seq);
    }
    let sp = seq + oc;
    let mut m = mil_header(ic, sp);
    gen_dyn_matmul(&mut m, "mm", ic, oc, seq, 0, seq, "x");
    gen_dyn_matmul_cast(&mut m, "mm", oc, seq, dtype);
    m += &mil_footer("mm_y_cast");
    Source {
        text: m,
        input_channels: ic,
        input_spatial: sp,
        output_channels: oc,
        output_spatial: seq,
        output_dtype: dtype,
    }
}

/// Build an ANE weight blob: 128-byte header + fp16 data.
pub fn pack_weights(fp16_data: &[u16]) -> Vec<u8> {
    let weight_bytes = fp16_data.len() * 2;
    let total = 128 + weight_bytes;
    let mut blob = vec![0u8; total];
    blob[0] = 1;
    blob[4] = 2;
    blob[64] = 0xEF;
    blob[65] = 0xBE;
    blob[66] = 0xAD;
    blob[67] = 0xDE;
    blob[68] = 1;
    blob[72..76].copy_from_slice(&(weight_bytes as u32).to_le_bytes());
    blob[80..84].copy_from_slice(&128u32.to_le_bytes());
    for (i, &val) in fp16_data.iter().enumerate() {
        let off = 128 + i * 2;
        blob[off..off + 2].copy_from_slice(&val.to_le_bytes());
    }
    blob
}
