//! Probe for _ANECoreMLModelCompiler path.
//!
//! Root cause confirmed: `modelWithMILText:` uses `_ANEMILCompiler` which REJECTS
//! `constexpr_*` ops. The `_ANECoreMLModelCompiler` handles CoreML bundle loading
//! and DOES accept `constexpr_affine_dequantize`.
//!
//! Goals:
//!   1. Enumerate methods defined DIRECTLY on _ANEInMemoryModelDescriptor and _ANEInMemoryModel
//!      (not inherited NSObject noise)
//!   2. Try bundle-loading APIs only if selector exists on the class
//!   3. Build minimal .mlmodelc bundle and try loading it
//!
//! Run: cargo run -p rane --example coreml_path_probe --release

use rane::ffi::*;
use std::ffi::{c_void, CStr, CString};

extern "C" {
    fn free(p: *mut c_void);
    fn class_getClassMethod(cls: ObjcClass, sel: ObjcSel) -> *mut c_void;
    fn class_getInstanceMethod(cls: ObjcClass, sel: ObjcSel) -> *mut c_void;
    fn class_respondsToSelector(cls: ObjcClass, sel: ObjcSel) -> bool;
    fn method_getTypeEncoding(method: *mut c_void) -> *const std::ffi::c_char;
}

fn libc_free(p: *mut c_void) {
    unsafe {
        free(p);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ANE CoreML compiler path probe ===\n");

    for name in &["AppleNeuralEngine", "ANECompiler", "ANEServices"] {
        let path = format!("/System/Library/PrivateFrameworks/{name}.framework/{name}");
        let c = CString::new(path).unwrap();
        unsafe {
            dlopen(c.as_ptr(), RTLD_NOW);
        }
    }

    // ─── 1. Methods defined DIRECTLY on each class ──────────────────────────
    for class_name in &["_ANEInMemoryModelDescriptor", "_ANEInMemoryModel"] {
        let c = cls(class_name);
        if c.is_null() {
            println!("Class {class_name}: NOT FOUND\n");
            continue;
        }
        println!("╔══ {} (own methods only) ══╗", class_name);

        // Instance methods directly on this class
        println!("  ── Instance methods ──");
        print_own_methods(c, "-");

        // Class methods: get metaclass, then its own methods
        let meta = unsafe { object_getClass(c as ObjcId) };
        println!("  ── Class methods ──");
        print_own_methods(meta, "+");
        println!();
    }

    // ─── 2. Check which bundle-loading selectors actually exist ─────────────
    println!("╔══ Selector existence check ══╗");
    let candidates_class = [
        "modelWithCoreMLBundleAtPath:options:error:",
        "modelWithCoreMLBundleAtURL:options:error:",
        "modelWithCoreMLBundlePath:",
        "modelWithCoreMLBundlePath:options:",
        "modelWithCoreMLBundlePath:options:error:",
        "modelWithCoreMLBundleURL:",
        "modelWithCoreMLBundleURL:options:",
        "modelWithCoreMLBundleURL:options:error:",
        "modelWithPath:options:error:",
        "modelWithURL:options:error:",
        "modelWithBundlePath:options:error:",
        "modelWithBundleURL:options:error:",
        "modelWithCoreMLBundle:options:error:",
        "descriptorWithCoreMLBundlePath:",
        "descriptorWithCoreMLBundlePath:options:",
        "descriptorWithBundlePath:",
        "descriptorWithBundlePath:options:",
        "descriptorWithURL:",
        "descriptorWithURL:options:",
        "modelWithMILText:weights:optionsPlist:", // known existing — sanity check
    ];

    let desc_cls = cls("_ANEInMemoryModelDescriptor");
    let model_cls = cls("_ANEInMemoryModel");
    let desc_meta = unsafe { object_getClass(desc_cls as ObjcId) };
    let model_meta = unsafe { object_getClass(model_cls as ObjcId) };

    let mut found: Vec<(&str, bool)> = Vec::new(); // (selector, is_on_desc)
    for sel_name in &candidates_class {
        let s = sel(sel_name);
        let on_desc = unsafe { !class_getClassMethod(desc_cls, s).is_null() };
        let on_model = unsafe { !class_getClassMethod(model_cls, s).is_null() };
        let mark_d = if on_desc { "✓ Descriptor" } else { "" };
        let mark_m = if on_model { "✓ Model" } else { "" };
        if on_desc || on_model {
            println!("  [EXISTS] {sel_name}  {mark_d} {mark_m}");
            found.push((sel_name, on_desc));
        } else {
            println!("  [absent] {sel_name}");
        }
    }

    // Also check instance method variants
    let init_candidates = [
        "initWithCoreMLBundlePath:options:error:",
        "initWithCoreMLBundleURL:options:error:",
        "initWithBundlePath:options:error:",
        "initWithBundleURL:options:error:",
        "initWithPath:options:error:",
        "initWithURL:options:error:",
    ];
    for sel_name in &init_candidates {
        let s = sel(sel_name);
        let on_desc = unsafe { !class_getInstanceMethod(desc_cls, s).is_null() };
        let on_model = unsafe { !class_getInstanceMethod(model_cls, s).is_null() };
        let mark_d = if on_desc { "✓ Descriptor" } else { "" };
        let mark_m = if on_model { "✓ Model" } else { "" };
        if on_desc || on_model {
            println!("  [EXISTS instance] {sel_name}  {mark_d} {mark_m}");
        } else {
            println!("  [absent instance] {sel_name}");
        }
    }
    println!();

    // ─── 3. Try bundle load with VoiceActions .mlmodelc ─────────────────────
    let bundle_path =
        "/System/Library/PrivateFrameworks/VoiceActions.framework/Versions/A/Resources/aa_encoder_125141826.mlmodelc";
    println!("╔══ Bundle load: {} ══╗", bundle_path);

    if found.is_empty() {
        println!("  No bundle-loading selectors found on descriptor/model classes.");
        println!(
            "  Trying direct approach: pass bundle path to modelWithMILText: read from model.mil\n"
        );
        try_mil_from_bundle(bundle_path)?;
    } else {
        for (sel_name, _) in &found {
            println!("  Trying: {sel_name}");
            unsafe {
                try_bundle_sel(bundle_path, sel_name);
            }
        }
    }

    // ─── 4. Synthetic bundle ─────────────────────────────────────────────────
    println!("\n╔══ Synthetic .mlmodelc bundle attempt ══╗");
    let synth_dir = build_synthetic_bundle()?;
    println!("  Bundle: {synth_dir}");
    if found.is_empty() {
        try_mil_from_bundle(&synth_dir)?;
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Method enumeration — only methods defined directly on `cls` (not superclass)
// ─────────────────────────────────────────────────────────────────────────────

fn print_own_methods(cls: ObjcClass, prefix: &str) {
    if cls.is_null() {
        return;
    }
    unsafe {
        let mut count: u32 = 0;
        let list = class_copyMethodList(cls, &mut count);
        if list.is_null() || count == 0 {
            println!("    (none)");
            return;
        }
        let mut names: Vec<(String, String)> = Vec::new();
        for i in 0..count {
            let method = *list.add(i as usize);
            let s = method_getName(method);
            let enc = method_getTypeEncoding(method);
            if !s.is_null() {
                let np = sel_getName(s);
                if !np.is_null() {
                    let name = CStr::from_ptr(np).to_string_lossy().into_owned();
                    let type_enc = if enc.is_null() {
                        "?".into()
                    } else {
                        CStr::from_ptr(enc).to_string_lossy().into_owned()
                    };
                    names.push((name, type_enc));
                }
            }
        }
        libc_free(list as *mut c_void);
        names.sort_by(|a, b| a.0.cmp(&b.0));
        for (name, enc) in &names {
            println!("    {prefix} {name}  [{enc}]");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Try a known selector for bundle loading
// ─────────────────────────────────────────────────────────────────────────────

unsafe fn try_bundle_sel(bundle_path: &str, sel_name: &str) {
    type F4e = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, *mut ObjcId) -> ObjcId;
    type F3e = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, *mut ObjcId) -> ObjcId;
    type F2 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
    type Fb = unsafe extern "C" fn(ObjcId, ObjcSel, u32, ObjcId, *mut ObjcId) -> bool;

    let f4e: F4e = std::mem::transmute(objc_msgSend as *const c_void);
    let f3e: F3e = std::mem::transmute(objc_msgSend as *const c_void);
    let f2: F2 = std::mem::transmute(objc_msgSend as *const c_void);
    let fb: Fb = std::mem::transmute(objc_msgSend as *const c_void);

    let cls_desc = cls("_ANEInMemoryModelDescriptor");
    let cls_model = cls("_ANEInMemoryModel");
    let cls_dict = cls("NSDictionary");
    let cls_nsstr = cls("NSString");
    let cls_nsurl = cls("NSURL");

    let cs = CString::new(bundle_path).unwrap();
    let ns_path = f2(
        cls_nsstr as ObjcId,
        sel("stringWithUTF8String:"),
        cs.as_ptr() as ObjcId,
    );
    let ns_url = f2(cls_nsurl as ObjcId, sel("fileURLWithPath:"), ns_path);
    let empty = f2(cls_dict as ObjcId, sel("dictionary"), std::ptr::null_mut());

    let s = sel(sel_name);
    let nargs = sel_name.chars().filter(|&c| c == ':').count();

    let desc = if nargs == 3 {
        let mut err: ObjcId = std::ptr::null_mut();
        let arg = if sel_name.contains("URL") {
            ns_url
        } else {
            ns_path
        };
        f4e(cls_desc as ObjcId, s, arg, empty, &mut err)
    } else if nargs == 2 {
        let arg = if sel_name.contains("URL") {
            ns_url
        } else {
            ns_path
        };
        f3e(cls_desc as ObjcId, s, arg, std::ptr::null_mut())
    } else {
        let arg = if sel_name.contains("URL") {
            ns_url
        } else {
            ns_path
        };
        f2(cls_desc as ObjcId, s, arg)
    };

    if desc.is_null() {
        println!("    descriptor = null");
        return;
    }
    let desc_cls = object_getClass(desc);
    let cname = if desc_cls.is_null() {
        "(null)".to_string()
    } else {
        let p = class_getName(desc_cls);
        if p.is_null() {
            "(null)".to_string()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    };
    println!("    descriptor type: {cname}");

    let model = f2(
        cls_model as ObjcId,
        sel("inMemoryModelWithDescriptor:"),
        desc,
    );
    if model.is_null() {
        println!("    model = null");
        return;
    }

    let mut err2: ObjcId = std::ptr::null_mut();
    let ok = fb(
        model,
        sel("compileWithQoS:options:error:"),
        21,
        empty,
        &mut err2,
    );
    if ok {
        println!("    *** COMPILE SUCCESS ***");
    } else {
        let e = nserror_string(err2);
        println!("    compile failed: {:?}", e);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Try loading a bundle by reading its model.mil and feeding to modelWithMILText:
// (Direct path — won't solve constexpr but confirms bundle structure)
// ─────────────────────────────────────────────────────────────────────────────

fn try_mil_from_bundle(bundle_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mil_path = format!("{bundle_path}/model.mil");
    let mil_text = std::fs::read_to_string(&mil_path)?;
    println!("  model.mil: {} bytes", mil_text.len());
    println!("  Compiling via modelWithMILText: (will reject constexpr_affine_dequantize)...");

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
        let nsdata = dataf(
            cls_data as ObjcId,
            sel("dataWithBytes:length:"),
            bytes.as_ptr(),
            bytes.len() as u64,
        );
        let empty = dictf(cls_dict as ObjcId, sel("dictionary"));
        let desc = descf(
            cls_desc as ObjcId,
            sel("modelWithMILText:weights:optionsPlist:"),
            nsdata,
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
            let cstr = utf8f(hex_id, sel("UTF8String"));
            CStr::from_ptr(cstr).to_string_lossy().into_owned()
        };
        let tmp_dir = std::env::temp_dir().join(&hex_str);
        let _ = std::fs::create_dir_all(tmp_dir.join("weights"));
        let _ = std::fs::write(tmp_dir.join("model.mil"), &mil_text);

        // Copy weights from bundle
        let wt_src = format!("{bundle_path}/weights/weight.bin");
        let wt_dst = tmp_dir.join("weights").join("weight.bin");
        if let Ok(_) = std::fs::copy(&wt_src, &wt_dst) {
            println!("  Copied {} to weights/", wt_src);
        }
        let wt_src2 = format!("{bundle_path}/weights/weights.bin");
        let wt_dst2 = tmp_dir.join("weights").join("weights.bin");
        if let Ok(_) = std::fs::copy(&wt_src2, &wt_dst2) {
            println!("  Copied {} to weights/", wt_src2);
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
            println!("  *** COMPILE SUCCESS ***");
        } else {
            let e = nserror_string(err);
            println!("  compile error: {:?}", e);
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal synthetic .mlmodelc bundle
// ─────────────────────────────────────────────────────────────────────────────

fn build_synthetic_bundle() -> Result<String, Box<dyn std::error::Error>> {
    let ic = 16usize;
    let oc = 16usize;

    let dir = "/tmp/ane_quant_test.mlmodelc";
    std::fs::create_dir_all(format!("{dir}/weights"))?;

    // Weight blob: [64 header][ic*oc int8 quantized][ic fp16 scale][ic int8 zp]
    // axis=0 → scale dim = ic (input channels)
    let data_off: u64 = 64;
    let qdata_size = ic * oc;
    let scale_size = ic * 2; // fp16
    let zp_size = ic;
    let total = 64 + qdata_size + scale_size + zp_size;
    let mut wblob = vec![0u8; total];
    for b in &mut wblob[64..64 + qdata_size] {
        *b = 1;
    }
    for i in 0..ic {
        let off = 64 + qdata_size + i * 2;
        wblob[off] = 0x00;
        wblob[off + 1] = 0x3C; // fp16(1.0)
    }
    std::fs::write(format!("{dir}/weights/weights.bin"), &wblob)?;

    let scale_off = data_off + qdata_size as u64;
    let zp_off = scale_off + scale_size as u64;

    // ios16 syntax matching fastspeech2_encoder.mlmodelc exactly
    let mil = format!(
        "program(1, 0)\nfunc main<ios16>(tensor<fp16, [1, {ic}, 1, 1]> x) -> (tensor<fp16, [1, {oc}, 1, 1]>) {{\n  block0() {{\n    tensor<fp16, [{ic},{oc}]> wf = constexpr_affine_dequantize()[axis = int32(0), quantized_data = tensor<int8, [{ic},{oc}]>(BLOBFILE(path = string(\"@model_path/weights/weights.bin\"), offset = uint64({data_off}))), scale = tensor<fp16, [{ic}]>(BLOBFILE(path = string(\"@model_path/weights/weights.bin\"), offset = uint64({scale_off}))), zero_point = tensor<int8, [{ic}]>(BLOBFILE(path = string(\"@model_path/weights/weights.bin\"), offset = uint64({zp_off})))];
    tensor<fp16, [1, {oc}, 1, 1]> y = linear()[alpha = fp32(1), beta = fp32(0), weight = wf, x = x];
  }} -> (y)
}}\n"
    );
    std::fs::write(format!("{dir}/model.mil"), &mil)?;
    println!("  model.mil: {} bytes", mil.len());
    println!("  weights.bin: {total} bytes");
    Ok(dir.to_string())
}
