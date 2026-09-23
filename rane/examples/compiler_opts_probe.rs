//! Probe the compiler options and temp dir for both MIL and non-MIL paths.
//!
//! After successful compile: list ALL files (including hidden) in tmp_dir.
//! Also try: isMILModel=false with various expected file names.
//!
//! Run: cargo run -p rane --example compiler_opts_probe --release

use rane::ffi::*;
use std::ffi::{c_void, CStr, CString};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Compiler options probe ===\n");

    for name in &["AppleNeuralEngine", "ANECompiler", "ANEServices"] {
        let path = format!("/System/Library/PrivateFrameworks/{name}.framework/{name}");
        let c = CString::new(path).unwrap();
        unsafe {
            dlopen(c.as_ptr(), RTLD_NOW);
        }
    }

    // ─── A. Successful MIL compile — list all files ───────────────────────
    println!("=== A. Successful matmul compile — all tmp_dir files ===");
    let plain_mil = rane::mil::matmul(16, 16, 1).text;
    compile_and_list_files(&plain_mil, None, None);

    // ─── B. Get compiler options plist content ─────────────────────────────
    println!("\n=== B. compilerOptionsWithOptions: output ===");
    compile_and_dump_opts(&plain_mil);

    // ─── C. isMILModel=false with various file names ──────────────────────
    let ic = 16usize;
    let oc = 16usize;
    let data_off: u64 = 64;
    let qdata_size = ic * oc;
    let scale_size = ic * 2;
    let total = 64 + qdata_size + scale_size + ic;
    let mut wblob = vec![0u8; total];
    for b in &mut wblob[64..64 + qdata_size] {
        *b = 1;
    }
    for i in 0..ic {
        let off = 64 + qdata_size + i * 2;
        wblob[off + 1] = 0x3C;
    }
    let scale_off = data_off + qdata_size as u64;
    let zp_off = scale_off + scale_size as u64;

    let constexpr_mil = format!(
        concat!(
            "program(1, 0)\nfunc main<ios16>(tensor<fp16, [1, {ic}, 1, 1]> x)",
            " -> (tensor<fp16, [1, {oc}, 1, 1]>) {{\n  block0() {{\n",
            "    tensor<fp16, [{ic},{oc}]> wf = constexpr_affine_dequantize()",
            "[axis = int32(0),",
            " quantized_data = tensor<int8, [{ic},{oc}]>(BLOBFILE(path = string(\"@model_path/weights/weights.bin\"), offset = uint64({data_off}))),",
            " scale = tensor<fp16, [{ic}]>(BLOBFILE(path = string(\"@model_path/weights/weights.bin\"), offset = uint64({scale_off}))),",
            " zero_point = tensor<int8, [{ic}]>(BLOBFILE(path = string(\"@model_path/weights/weights.bin\"), offset = uint64({zp_off})))];\n",
            "    tensor<fp16, [1, {oc}, 1, 1]> y = linear()[alpha = fp32(1), beta = fp32(0), weight = wf, x = x];\n",
            "  }} -> (y)\n}}\n"
        ),
        ic = ic, oc = oc, data_off = data_off, scale_off = scale_off, zp_off = zp_off
    );
    let wblob_clone = wblob.clone();

    println!("\n=== C. isMILModel=false + net.plist ===");
    compile_non_mil_with_file(
        &constexpr_mil,
        &wblob,
        "net.plist",
        &constexpr_mil.as_bytes(),
    );

    println!("\n=== D. isMILModel=false + model.espresso.net ===");
    // Write a minimal Espresso plist (just the file, empty dict plist data)
    let espresso_plist = b"\
<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
<plist version=\"1.0\"><dict></dict></plist>\n";
    compile_non_mil_with_file(&constexpr_mil, &wblob, "model.espresso.net", espresso_plist);

    println!("\n=== E. isMILModel=false + coremldata.bin + model.mil ===");
    let coreml_bin = std::fs::read("/System/Library/PrivateFrameworks/VoiceActions.framework/Versions/A/Resources/aa_encoder_125141826.mlmodelc/coremldata.bin").unwrap_or_default();
    compile_non_mil_with_coreml_bin(&constexpr_mil, &wblob, &coreml_bin);

    Ok(())
}

fn compile_and_list_files(
    mil_text: &str,
    extra_file: Option<(&str, &[u8])>,
    weights: Option<&[u8]>,
) {
    unsafe {
        type DataFn = unsafe extern "C" fn(ObjcId, ObjcSel, *const u8, u64) -> ObjcId;
        type DictFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type DescFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, ObjcId) -> ObjcId;
        type ModelFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type StrFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type Utf8Fn = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const std::ffi::c_char;
        type CompileFn = unsafe extern "C" fn(ObjcId, ObjcSel, u32, ObjcId, *mut ObjcId) -> bool;

        let cls_data = cls("NSData");
        let cls_dict = cls("NSDictionary");
        let cls_desc = cls("_ANEInMemoryModelDescriptor");
        let cls_model = cls("_ANEInMemoryModel");

        let dataf: DataFn = std::mem::transmute(objc_msgSend as *const c_void);
        let dictf: DictFn = std::mem::transmute(objc_msgSend as *const c_void);
        let descf: DescFn = std::mem::transmute(objc_msgSend as *const c_void);
        let modelf: ModelFn = std::mem::transmute(objc_msgSend as *const c_void);
        let strf: StrFn = std::mem::transmute(objc_msgSend as *const c_void);
        let utf8f: Utf8Fn = std::mem::transmute(objc_msgSend as *const c_void);
        let compilef: CompileFn = std::mem::transmute(objc_msgSend as *const c_void);

        let bytes = mil_text.as_bytes();
        let ns_text = dataf(
            cls_data as ObjcId,
            sel("dataWithBytes:length:"),
            bytes.as_ptr(),
            bytes.len() as u64,
        );
        let empty = dictf(cls_dict as ObjcId, sel("dictionary"));

        let desc = descf(
            cls_desc as ObjcId,
            sel("modelWithMILText:weights:optionsPlist:"),
            ns_text,
            empty,
            std::ptr::null_mut(),
        );
        let model = modelf(
            cls_model as ObjcId,
            sel("inMemoryModelWithDescriptor:"),
            desc,
        );

        let hex_id = strf(model, sel("hexStringIdentifier"));
        let hex_str = {
            let c = utf8f(hex_id, sel("UTF8String"));
            CStr::from_ptr(c).to_string_lossy().into_owned()
        };
        let tmp_dir = std::env::temp_dir().join(&hex_str);
        let _ = std::fs::create_dir_all(tmp_dir.join("weights"));
        let _ = std::fs::write(tmp_dir.join("model.mil"), mil_text);

        if let Some((fname, fdata)) = extra_file {
            let _ = std::fs::write(tmp_dir.join(fname), fdata);
        }
        if let Some(w) = weights {
            let _ = std::fs::write(tmp_dir.join("weights").join("weights.bin"), w);
        }

        let mut err: ObjcId = std::ptr::null_mut();
        let ok = compilef(
            model,
            sel("compileWithQoS:options:error:"),
            21,
            empty,
            &mut err,
        );
        println!("  compile: {}", if ok { "OK" } else { "FAILED" });
        if !ok {
            println!("  error: {:?}", nserror_string(err));
        }

        println!("  tmp_dir: {}", tmp_dir.display());
        println!("  Files (including hidden):");
        print_dir_tree_all(tmp_dir.to_str().unwrap());
    }
}

fn compile_and_dump_opts(mil_text: &str) {
    unsafe {
        type DataFn = unsafe extern "C" fn(ObjcId, ObjcSel, *const u8, u64) -> ObjcId;
        type DictFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type DescFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, ObjcId) -> ObjcId;
        type ModelFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type ModelFn2 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, u8) -> ObjcId;
        type StrFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type Utf8Fn = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const std::ffi::c_char;
        type StrU8Fn = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const std::ffi::c_char;
        type LenFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> u64;
        type BytesFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const u8;

        let cls_data = cls("NSData");
        let cls_dict = cls("NSDictionary");
        let cls_desc = cls("_ANEInMemoryModelDescriptor");
        let cls_model = cls("_ANEInMemoryModel");

        let dataf: DataFn = std::mem::transmute(objc_msgSend as *const c_void);
        let dictf: DictFn = std::mem::transmute(objc_msgSend as *const c_void);
        let descf: DescFn = std::mem::transmute(objc_msgSend as *const c_void);
        let modelf: ModelFn = std::mem::transmute(objc_msgSend as *const c_void);
        let modelf2: ModelFn2 = std::mem::transmute(objc_msgSend as *const c_void);
        let strf: StrFn = std::mem::transmute(objc_msgSend as *const c_void);
        let utf8f: Utf8Fn = std::mem::transmute(objc_msgSend as *const c_void);
        let lenf: LenFn = std::mem::transmute(objc_msgSend as *const c_void);
        let bytesf: BytesFn = std::mem::transmute(objc_msgSend as *const c_void);

        let bytes = mil_text.as_bytes();
        let ns_text = dataf(
            cls_data as ObjcId,
            sel("dataWithBytes:length:"),
            bytes.as_ptr(),
            bytes.len() as u64,
        );
        let empty = dictf(cls_dict as ObjcId, sel("dictionary"));
        let desc = descf(
            cls_desc as ObjcId,
            sel("modelWithMILText:weights:optionsPlist:"),
            ns_text,
            empty,
            std::ptr::null_mut(),
        );
        let model = modelf(
            cls_model as ObjcId,
            sel("inMemoryModelWithDescriptor:"),
            desc,
        );
        if model.is_null() {
            println!("  model = null");
            return;
        }

        // Call compilerOptionsWithOptions:isCompiledModelCached:
        let opts = modelf2(
            model,
            sel("compilerOptionsWithOptions:isCompiledModelCached:"),
            empty,
            0,
        );
        if opts.is_null() {
            println!("  compilerOptions = null");
            return;
        }

        // It's likely an NSDictionary or NSData — check class
        let opts_cls = object_getClass(opts);
        let cn = if opts_cls.is_null() {
            "?".to_string()
        } else {
            let p = class_getName(opts_cls);
            if p.is_null() {
                "?".to_string()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        println!("  compilerOptions class: {cn}");

        // Try to get description
        let desc_obj = strf(opts, sel("description"));
        if !desc_obj.is_null() {
            let cstr = utf8f(desc_obj, sel("UTF8String"));
            if !cstr.is_null() {
                let s = CStr::from_ptr(cstr).to_string_lossy();
                println!("  compilerOptions description:\n{s}");
            }
        }

        // If it's NSData, dump bytes
        let len = lenf(opts, sel("length"));
        if len > 0 && len < 65536 {
            let ptr = bytesf(opts, sel("bytes"));
            if !ptr.is_null() {
                let data = std::slice::from_raw_parts(ptr, len as usize);
                let _ = std::fs::write("/tmp/ane_compiler_opts.bin", data);
                println!("  NSData: {} bytes → /tmp/ane_compiler_opts.bin", len);
                // Try to print as string
                if let Ok(s) = std::str::from_utf8(data) {
                    println!("  as string: {s}");
                }
            }
        }
    }
}

fn compile_non_mil_with_file(
    mil_text: &str,
    weights: &[u8],
    extra_filename: &str,
    extra_content: &[u8],
) {
    unsafe {
        type DataFn = unsafe extern "C" fn(ObjcId, ObjcSel, *const u8, u64) -> ObjcId;
        type DictFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type DescFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, ObjcId) -> ObjcId;
        type ModelFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type StrFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type Utf8Fn = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const std::ffi::c_char;
        type CompileFn = unsafe extern "C" fn(ObjcId, ObjcSel, u32, ObjcId, *mut ObjcId) -> bool;

        let cls_data = cls("NSData");
        let cls_dict = cls("NSDictionary");
        let cls_desc = cls("_ANEInMemoryModelDescriptor");
        let cls_model = cls("_ANEInMemoryModel");

        let dataf: DataFn = std::mem::transmute(objc_msgSend as *const c_void);
        let dictf: DictFn = std::mem::transmute(objc_msgSend as *const c_void);
        let descf: DescFn = std::mem::transmute(objc_msgSend as *const c_void);
        let modelf: ModelFn = std::mem::transmute(objc_msgSend as *const c_void);
        let strf: StrFn = std::mem::transmute(objc_msgSend as *const c_void);
        let utf8f: Utf8Fn = std::mem::transmute(objc_msgSend as *const c_void);
        let compilef: CompileFn = std::mem::transmute(objc_msgSend as *const c_void);

        let bytes = mil_text.as_bytes();
        let ns_text = dataf(
            cls_data as ObjcId,
            sel("dataWithBytes:length:"),
            bytes.as_ptr(),
            bytes.len() as u64,
        );
        let empty = dictf(cls_dict as ObjcId, sel("dictionary"));

        let desc = descf(
            cls_desc as ObjcId,
            sel("modelWithMILText:weights:optionsPlist:"),
            ns_text,
            empty,
            std::ptr::null_mut(),
        );
        let model = modelf(
            cls_model as ObjcId,
            sel("inMemoryModelWithDescriptor:"),
            desc,
        );

        // Flip _isMILModel at offset 8
        let p = (desc as *mut u8).add(8);
        *p = 0;

        let hex_id = strf(model, sel("hexStringIdentifier"));
        let hex_str = {
            let c = utf8f(hex_id, sel("UTF8String"));
            CStr::from_ptr(c).to_string_lossy().into_owned()
        };
        let tmp_dir = std::env::temp_dir().join(&hex_str);
        let _ = std::fs::create_dir_all(tmp_dir.join("weights"));
        let _ = std::fs::write(tmp_dir.join("model.mil"), mil_text);
        let _ = std::fs::write(tmp_dir.join(extra_filename), extra_content);
        let _ = std::fs::write(tmp_dir.join("weights").join("weights.bin"), weights);

        println!("  Files in tmp_dir before compile:");
        print_dir_tree_all(tmp_dir.to_str().unwrap());

        let mut err: ObjcId = std::ptr::null_mut();
        let ok = compilef(
            model,
            sel("compileWithQoS:options:error:"),
            21,
            empty,
            &mut err,
        );
        if ok {
            println!("  *** COMPILE SUCCESS with {extra_filename}! ***");
        } else {
            println!("  error: {:?}", nserror_string(err));
        }
    }
}

fn compile_non_mil_with_coreml_bin(mil_text: &str, weights: &[u8], coreml_bin: &[u8]) {
    unsafe {
        type DataFn = unsafe extern "C" fn(ObjcId, ObjcSel, *const u8, u64) -> ObjcId;
        type DictFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type DescFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, ObjcId) -> ObjcId;
        type ModelFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type StrFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type Utf8Fn = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const std::ffi::c_char;
        type CompileFn = unsafe extern "C" fn(ObjcId, ObjcSel, u32, ObjcId, *mut ObjcId) -> bool;

        let cls_data = cls("NSData");
        let cls_dict = cls("NSDictionary");
        let cls_desc = cls("_ANEInMemoryModelDescriptor");
        let cls_model = cls("_ANEInMemoryModel");

        let dataf: DataFn = std::mem::transmute(objc_msgSend as *const c_void);
        let dictf: DictFn = std::mem::transmute(objc_msgSend as *const c_void);
        let descf: DescFn = std::mem::transmute(objc_msgSend as *const c_void);
        let modelf: ModelFn = std::mem::transmute(objc_msgSend as *const c_void);
        let strf: StrFn = std::mem::transmute(objc_msgSend as *const c_void);
        let utf8f: Utf8Fn = std::mem::transmute(objc_msgSend as *const c_void);
        let compilef: CompileFn = std::mem::transmute(objc_msgSend as *const c_void);

        let bytes = mil_text.as_bytes();
        let ns_text = dataf(
            cls_data as ObjcId,
            sel("dataWithBytes:length:"),
            bytes.as_ptr(),
            bytes.len() as u64,
        );
        let empty = dictf(cls_dict as ObjcId, sel("dictionary"));

        // Use coremldata.bin as the networkDescription
        let ns_cml = dataf(
            cls_data as ObjcId,
            sel("dataWithBytes:length:"),
            coreml_bin.as_ptr(),
            coreml_bin.len() as u64,
        );

        // Try modelWithNetworkDescription: with coremldata.bin — set isMILModel=NO implicitly
        let desc = descf(
            cls_desc as ObjcId,
            sel("modelWithNetworkDescription:weights:optionsPlist:"),
            ns_cml,
            empty,
            std::ptr::null_mut(),
        );
        if desc.is_null() {
            println!("  descriptor = null");
            return;
        }

        let model = modelf(
            cls_model as ObjcId,
            sel("inMemoryModelWithDescriptor:"),
            desc,
        );
        if model.is_null() {
            println!("  model = null");
            return;
        }

        let hex_id = strf(model, sel("hexStringIdentifier"));
        let hex_str = {
            let c = utf8f(hex_id, sel("UTF8String"));
            CStr::from_ptr(c).to_string_lossy().into_owned()
        };
        let tmp_dir = std::env::temp_dir().join(&hex_str);
        let _ = std::fs::create_dir_all(tmp_dir.join("weights"));
        // Write BOTH coremldata.bin and model.mil
        let _ = std::fs::write(tmp_dir.join("coremldata.bin"), coreml_bin);
        let _ = std::fs::write(tmp_dir.join("model.mil"), mil_text);
        let _ = std::fs::write(tmp_dir.join("weights").join("weights.bin"), weights);
        // Also copy VoiceActions weights (needed since coremldata.bin refers to them)
        let _ = std::fs::copy(
            "/System/Library/PrivateFrameworks/VoiceActions.framework/Versions/A/Resources/aa_encoder_125141826.mlmodelc/weights/weight.bin",
            tmp_dir.join("weights").join("weight.bin"),
        );

        println!("  Files before compile:");
        print_dir_tree_all(tmp_dir.to_str().unwrap());

        let mut err: ObjcId = std::ptr::null_mut();
        let ok = compilef(
            model,
            sel("compileWithQoS:options:error:"),
            21,
            empty,
            &mut err,
        );
        if ok {
            println!("  *** COMPILE SUCCESS! ***");
        } else {
            println!("  error: {:?}", nserror_string(err));
        }
    }
}

fn print_dir_tree_all(dir: &str) {
    // Use shell find to include hidden files
    let out = std::process::Command::new("find")
        .args([dir, "-maxdepth", "3"])
        .output()
        .unwrap_or_else(|_| std::process::Output {
            status: std::process::ExitStatus::default(),
            stdout: vec![],
            stderr: vec![],
        });
    let s = String::from_utf8_lossy(&out.stdout);
    for line in s.lines() {
        let rel = line.trim_start_matches(dir);
        if rel.is_empty() {
            continue;
        }
        if let Ok(meta) = std::fs::metadata(line) {
            if meta.is_dir() {
                println!("    {rel}/");
            } else {
                println!("    {rel}: {}B", meta.len());
            }
        }
    }
}
