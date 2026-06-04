//! Check kANEFModelType when isMILModel=false.
//! Also: try every plausible file name for the non-MIL network source file.
//!
//! Run: cargo run -p rane --example non_mil_opts --release

use rane::ffi::*;
use std::ffi::{c_void, CStr, CString};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== non-MIL compiler options + file name probes ===\n");

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

    unsafe {
        type DataFn = unsafe extern "C" fn(ObjcId, ObjcSel, *const u8, u64) -> ObjcId;
        type DictFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type DescFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, ObjcId) -> ObjcId;
        type ModelFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type ModelFn2 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, u8) -> ObjcId;
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
        let modelf2: ModelFn2 = std::mem::transmute(objc_msgSend as *const c_void);
        let strf: StrFn = std::mem::transmute(objc_msgSend as *const c_void);
        let utf8f: Utf8Fn = std::mem::transmute(objc_msgSend as *const c_void);
        let compilef: CompileFn = std::mem::transmute(objc_msgSend as *const c_void);

        let bytes = constexpr_mil.as_bytes();
        let ns_text = dataf(
            cls_data as ObjcId,
            sel("dataWithBytes:length:"),
            bytes.as_ptr(),
            bytes.len() as u64,
        );
        let empty = dictf(cls_dict as ObjcId, sel("dictionary"));

        // ─── A. Compiler options with isMILModel=false ──────────────────────
        println!("=== A. compilerOptionsWithOptions when isMILModel=false ===");
        {
            let desc = descf(
                cls_desc as ObjcId,
                sel("modelWithMILText:weights:optionsPlist:"),
                ns_text,
                empty,
                std::ptr::null_mut(),
            );
            // Flip _isMILModel BEFORE creating model
            let p = (desc as *mut u8).add(8);
            *p = 0;
            let model = modelf(
                cls_model as ObjcId,
                sel("inMemoryModelWithDescriptor:"),
                desc,
            );
            if model.is_null() {
                println!("  model = null");
            } else {
                let opts = modelf2(
                    model,
                    sel("compilerOptionsWithOptions:isCompiledModelCached:"),
                    empty,
                    0,
                );
                if opts.is_null() {
                    println!("  opts = null");
                } else {
                    println!("  {}", objc_desc(opts));
                }
            }
        }

        // ─── B. File name probes for non-MIL compiler path ─────────────────
        println!("\n=== B. File name probes (isMILModel=false) ===");

        let candidates = [
            "model.espresso.net",
            "net.plist",
            "model.mlpackage",
            "coremldata.bin",
            "model.pb",
            "model.onnx",
            "model.mlir",
            "model.anec",
            "network.plist",
            "model.plist",
            "model.cvair",
            "model.csv",
        ];

        // Use constexpr MIL text for plist-ish content and binary for binary content
        let plist_content = constexpr_mil.as_bytes();
        let bin_content = &wblob[..16]; // just some bytes

        // Get hex_id for tmp_dir
        let desc_hex = descf(
            cls_desc as ObjcId,
            sel("modelWithMILText:weights:optionsPlist:"),
            ns_text,
            empty,
            std::ptr::null_mut(),
        );
        let model_hex = modelf(
            cls_model as ObjcId,
            sel("inMemoryModelWithDescriptor:"),
            desc_hex,
        );
        let hex_id = strf(model_hex, sel("hexStringIdentifier"));
        let hex_str = {
            let c = utf8f(hex_id, sel("UTF8String"));
            CStr::from_ptr(c).to_string_lossy().into_owned()
        };
        let tmp_dir = format!("/tmp/{hex_str}");
        std::fs::create_dir_all(format!("{tmp_dir}/weights")).unwrap();
        std::fs::write(format!("{tmp_dir}/model.mil"), &constexpr_mil).unwrap();
        std::fs::write(format!("{tmp_dir}/weights/weights.bin"), &wblob).unwrap();

        for filename in &candidates {
            // Write the file (try MIL text for text-ish, binary for binary-ish)
            let content: &[u8] = if filename.ends_with(".plist")
                || filename.ends_with(".net")
                || filename.ends_with(".mlir")
            {
                plist_content
            } else {
                bin_content
            };
            std::fs::write(format!("{tmp_dir}/{filename}"), content).unwrap();

            // Create fresh descriptor + model with flip
            let desc = descf(
                cls_desc as ObjcId,
                sel("modelWithMILText:weights:optionsPlist:"),
                ns_text,
                empty,
                std::ptr::null_mut(),
            );
            let p = (desc as *mut u8).add(8);
            *p = 0; // isMILModel=false

            let model = modelf(
                cls_model as ObjcId,
                sel("inMemoryModelWithDescriptor:"),
                desc,
            );
            if model.is_null() {
                println!("  [{filename}] model = null");
                std::fs::remove_file(format!("{tmp_dir}/{filename}")).ok();
                continue;
            }

            let mut err: ObjcId = std::ptr::null_mut();
            let ok = compilef(
                model,
                sel("compileWithQoS:options:error:"),
                21,
                empty,
                &mut err,
            );
            if ok {
                println!("  [{filename}] *** COMPILE SUCCESS! ***");
            } else {
                let e = nserror_string(err).unwrap_or_default();
                // Extract just the error code
                let code = extract_error_code(&e);
                println!("  [{filename}] {code}");
            }

            std::fs::remove_file(format!("{tmp_dir}/{filename}")).ok();
        }

        // ─── C. All candidates simultaneously ─────────────────────────────
        println!("\n=== C. All candidate files present simultaneously ===");
        for filename in &candidates {
            let content: &[u8] = if filename.ends_with(".plist")
                || filename.ends_with(".net")
                || filename.ends_with(".mlir")
            {
                plist_content
            } else {
                bin_content
            };
            std::fs::write(format!("{tmp_dir}/{filename}"), content).unwrap();
        }
        let desc = descf(
            cls_desc as ObjcId,
            sel("modelWithMILText:weights:optionsPlist:"),
            ns_text,
            empty,
            std::ptr::null_mut(),
        );
        let p = (desc as *mut u8).add(8);
        *p = 0;
        let model = modelf(
            cls_model as ObjcId,
            sel("inMemoryModelWithDescriptor:"),
            desc,
        );
        if !model.is_null() {
            let mut err: ObjcId = std::ptr::null_mut();
            let ok = compilef(
                model,
                sel("compileWithQoS:options:error:"),
                21,
                empty,
                &mut err,
            );
            if ok {
                println!("  *** COMPILE SUCCESS with all files! ***");
            } else {
                let e = nserror_string(err).unwrap_or_default();
                println!("  {}", extract_error_code(&e));
            }
        }
    }

    Ok(())
}

fn extract_error_code(e: &str) -> String {
    // Extract the innermost error code
    if let Some(idx) = e.rfind("err=(\n    ") {
        let rest = &e[idx + 9..];
        if let Some(end) = rest.find('\n') {
            return rest[..end].trim().to_string();
        }
    }
    if e.len() > 200 {
        e[..200].to_string()
    } else {
        e.to_string()
    }
}

unsafe fn objc_desc(obj: ObjcId) -> String {
    if obj.is_null() {
        return "(null)".into();
    }
    type StrFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
    type Utf8Fn = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const std::ffi::c_char;
    let sf: StrFn = std::mem::transmute(objc_msgSend as *const c_void);
    let uf: Utf8Fn = std::mem::transmute(objc_msgSend as *const c_void);
    let d = sf(obj, sel("description"));
    if d.is_null() {
        return "(no desc)".into();
    }
    let c = uf(d, sel("UTF8String"));
    if c.is_null() {
        return "(null utf8)".into();
    }
    CStr::from_ptr(c).to_string_lossy().into_owned()
}
