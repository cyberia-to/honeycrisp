//! Force kANEFModelType = kANEFModelCoreML in compile options.
//!
//! Discovery: compilerOptionsWithOptions: returns:
//!   { kANEFModelType = kANEFModelMIL; kANEFIsInMemoryModelTypeKey = "<hex>"; kANEFInMemoryModelIsCachedKey = 0 }
//!
//! Strategy: build that dict with kANEFModelType = kANEFModelCoreML instead of kANEFModelMIL.
//! Pass to compileWithQoS:options:error: → routes to _ANECoreMLModelCompiler in XPC service.
//!
//! Run: cargo run -p rane --example force_coreml_type --release

use rane::ffi::*;
use std::ffi::{c_void, CStr, CString};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Force kANEFModelType=kANEFModelCoreML probe ===\n");

    for name in &["AppleNeuralEngine", "ANECompiler", "ANEServices"] {
        let path = format!("/System/Library/PrivateFrameworks/{name}.framework/{name}");
        let c = CString::new(path).unwrap();
        unsafe {
            dlopen(c.as_ptr(), RTLD_NOW);
        }
    }

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
        wblob[off + 1] = 0x3C; // fp16(1.0)
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

    unsafe {
        type DataFn = unsafe extern "C" fn(ObjcId, ObjcSel, *const u8, u64) -> ObjcId;
        type DictFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type DescFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, ObjcId) -> ObjcId;
        type ModelFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type StrFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type StrFn2 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type Utf8Fn = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const std::ffi::c_char;
        type NumFn = unsafe extern "C" fn(ObjcId, ObjcSel, i64) -> ObjcId;
        type DictMakeFn =
            unsafe extern "C" fn(ObjcId, ObjcSel, *const ObjcId, *const ObjcId, u64) -> ObjcId;
        type CompileFn = unsafe extern "C" fn(ObjcId, ObjcSel, u32, ObjcId, *mut ObjcId) -> bool;
        type ModelFn2 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, u8) -> ObjcId;
        type Utf8Fn2 = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const std::ffi::c_char;

        let cls_data = cls("NSData");
        let cls_dict = cls("NSDictionary");
        let cls_mdict = cls("NSMutableDictionary");
        let cls_desc = cls("_ANEInMemoryModelDescriptor");
        let cls_model = cls("_ANEInMemoryModel");
        let cls_nsstr = cls("NSString");
        let cls_nsnum = cls("NSNumber");

        let dataf: DataFn = std::mem::transmute(objc_msgSend as *const c_void);
        let dictf: DictFn = std::mem::transmute(objc_msgSend as *const c_void);
        let descf: DescFn = std::mem::transmute(objc_msgSend as *const c_void);
        let modelf: ModelFn = std::mem::transmute(objc_msgSend as *const c_void);
        let strf: StrFn = std::mem::transmute(objc_msgSend as *const c_void);
        let strf2: StrFn2 = std::mem::transmute(objc_msgSend as *const c_void);
        let utf8f: Utf8Fn = std::mem::transmute(objc_msgSend as *const c_void);
        let numf: NumFn = std::mem::transmute(objc_msgSend as *const c_void);
        let compilef: CompileFn = std::mem::transmute(objc_msgSend as *const c_void);
        let modelf2: ModelFn2 = std::mem::transmute(objc_msgSend as *const c_void);

        let bytes = constexpr_mil.as_bytes();
        let ns_text = dataf(
            cls_data as ObjcId,
            sel("dataWithBytes:length:"),
            bytes.as_ptr(),
            bytes.len() as u64,
        );
        let empty = dictf(cls_dict as ObjcId, sel("dictionary"));

        // Create descriptor + model (normal MIL path)
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
            println!("model = null");
            return Ok(());
        }

        // Get hex identifier
        let hex_id = strf(model, sel("hexStringIdentifier"));
        let hex_str = {
            let c = utf8f(hex_id, sel("UTF8String"));
            CStr::from_ptr(c).to_string_lossy().into_owned()
        };
        println!("hex_id: {hex_str}");

        // Get the real compiler options first (to understand the exact keys/values)
        let real_opts = modelf2(
            model,
            sel("compilerOptionsWithOptions:isCompiledModelCached:"),
            empty,
            0,
        );
        println!("real options: {}", objc_description(real_opts));

        // Build NSString objects for keys/values
        let ns_key_type = nsstring(cls_nsstr as ObjcId, &strf2, "kANEFModelType");
        let ns_key_id = nsstring(cls_nsstr as ObjcId, &strf2, "kANEFIsInMemoryModelTypeKey");
        let ns_key_cached = nsstring(cls_nsstr as ObjcId, &strf2, "kANEFInMemoryModelIsCachedKey");

        // Try different model type values
        let model_types = [
            "kANEFModelCoreML",
            "kANEFModelPreCompiled",
            "kANEFModelANECIR",
            "kANEFModelMLIR",
            "kANEFModelLLIRBundle",
        ];

        // Setup tmp_dir
        let tmp_dir = format!("/tmp/{hex_str}");
        std::fs::create_dir_all(format!("{tmp_dir}/weights"))?;
        std::fs::write(format!("{tmp_dir}/model.mil"), &constexpr_mil)?;
        std::fs::write(format!("{tmp_dir}/weights/weights.bin"), &wblob)?;
        // Also copy VoiceActions coremldata.bin into the dir
        let _ = std::fs::copy(
            "/System/Library/PrivateFrameworks/VoiceActions.framework/Versions/A/Resources/aa_encoder_125141826.mlmodelc/coremldata.bin",
            format!("{tmp_dir}/coremldata.bin"),
        );

        for model_type_val in &model_types {
            println!("\n--- kANEFModelType = {model_type_val} ---");

            let ns_val_type = nsstring(cls_nsstr as ObjcId, &strf2, model_type_val);
            let ns_num_zero = numf(cls_nsnum as ObjcId, sel("numberWithInteger:"), 0);

            // Build NSDictionary with 3 keys
            let keys = [ns_key_type, ns_key_id, ns_key_cached];
            let vals = [ns_val_type, hex_id, ns_num_zero];

            type DictFromKVFn =
                unsafe extern "C" fn(ObjcId, ObjcSel, *const ObjcId, *const ObjcId, u64) -> ObjcId;
            let dkv: DictFromKVFn = std::mem::transmute(objc_msgSend as *const c_void);
            let opts_dict = dkv(
                cls_dict as ObjcId,
                sel("dictionaryWithObjects:forKeys:count:"),
                vals.as_ptr(),
                keys.as_ptr(),
                3u64,
            );
            println!("  opts: {}", objc_description(opts_dict));

            // Create fresh model for each attempt
            let desc_n = descf(
                cls_desc as ObjcId,
                sel("modelWithMILText:weights:optionsPlist:"),
                ns_text,
                empty,
                std::ptr::null_mut(),
            );
            let model_n = modelf(
                cls_model as ObjcId,
                sel("inMemoryModelWithDescriptor:"),
                desc_n,
            );
            if model_n.is_null() {
                println!("  model = null");
                continue;
            }

            let mut err: ObjcId = std::ptr::null_mut();
            let ok = compilef(
                model_n,
                sel("compileWithQoS:options:error:"),
                21,
                opts_dict,
                &mut err,
            );
            if ok {
                println!("  *** COMPILE SUCCESS with {model_type_val}! ***");
                // List what's in tmp_dir
                list_dir(&tmp_dir);
            } else {
                println!("  error: {:?}", nserror_string(err));
            }
        }

        // ─── Also try: pass options dict from compilerOptionsWithOptions: but with CoreML type ──
        println!("\n--- Using compilerOptionsWithOptions base + override kANEFModelType ---");
        // Get base opts and make mutable copy
        let base_opts = modelf2(
            model,
            sel("compilerOptionsWithOptions:isCompiledModelCached:"),
            empty,
            0,
        );
        let mut_opts = dictf(cls_mdict as ObjcId, sel("new"));
        // Copy base keys
        type SetKVFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId);
        let setkv: SetKVFn = std::mem::transmute(objc_msgSend as *const c_void);
        // Add base_opts to mut_opts (just add each key from base_opts)
        // Actually, use setDictionary: to copy all
        type SetDictFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId);
        let sd: SetDictFn = std::mem::transmute(objc_msgSend as *const c_void);
        sd(mut_opts, sel("setDictionary:"), base_opts);
        // Override model type
        let ns_coreml = nsstring(cls_nsstr as ObjcId, &strf2, "kANEFModelCoreML");
        setkv(mut_opts, sel("setObject:forKey:"), ns_coreml, ns_key_type);
        println!("  modified opts: {}", objc_description(mut_opts));

        let desc_x = descf(
            cls_desc as ObjcId,
            sel("modelWithMILText:weights:optionsPlist:"),
            ns_text,
            empty,
            std::ptr::null_mut(),
        );
        let model_x = modelf(
            cls_model as ObjcId,
            sel("inMemoryModelWithDescriptor:"),
            desc_x,
        );
        if model_x.is_null() {
            println!("  model_x = null");
            return Ok(());
        }

        let mut err_x: ObjcId = std::ptr::null_mut();
        let ok_x = compilef(
            model_x,
            sel("compileWithQoS:options:error:"),
            21,
            mut_opts,
            &mut err_x,
        );
        if ok_x {
            println!("  *** COMPILE SUCCESS with modified base opts! ***");
            list_dir(&tmp_dir);
        } else {
            println!("  error: {:?}", nserror_string(err_x));
        }
    }

    Ok(())
}

unsafe fn nsstring(
    cls: ObjcId,
    strf2: &(unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId),
    s: &str,
) -> ObjcId {
    let cs = CString::new(s).unwrap();
    strf2(cls, sel("stringWithUTF8String:"), cs.as_ptr() as ObjcId)
}

unsafe fn objc_description(obj: ObjcId) -> String {
    if obj.is_null() {
        return "(null)".into();
    }
    type StrFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
    type Utf8Fn = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const std::ffi::c_char;
    let sf: StrFn = std::mem::transmute(objc_msgSend as *const c_void);
    let uf: Utf8Fn = std::mem::transmute(objc_msgSend as *const c_void);
    let d = sf(obj, sel("description"));
    if d.is_null() {
        return "(no description)".into();
    }
    let c = uf(d, sel("UTF8String"));
    if c.is_null() {
        return "(null UTF8)".into();
    }
    CStr::from_ptr(c).to_string_lossy().into_owned()
}

fn list_dir(dir: &str) {
    let out = std::process::Command::new("find")
        .args([dir, "-maxdepth", "2", "-type", "f"])
        .output()
        .unwrap_or_else(|_| std::process::Output {
            status: unsafe { std::mem::zeroed() },
            stdout: vec![],
            stderr: vec![],
        });
    let s = String::from_utf8_lossy(&out.stdout);
    for line in s.lines() {
        let rel = line.trim_start_matches(dir);
        if let Ok(meta) = std::fs::metadata(line) {
            println!("    {rel}: {}B", meta.len());
        }
    }
}
