//! Pass NetworkSourceFileName / NetworkSourcePath in the options dict to compileWithQoS:.
//! These keys appear in ANECompilerService binary as expected XPC message fields.
//!
//! Run: cargo run -p rane --example anec_opts_inject --release

use rane::ffi::*;
use std::ffi::{c_void, CStr, CString};

const ESPRESSO_BUNDLE: &str = "/System/Library/DuetExpertCenter/Assets/Assets.bundle/AssetData/ATXActionValuationMLModel.mlmodelc";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== NetworkSourceFileName options injection ===\n");

    for name in &["AppleNeuralEngine", "ANECompiler", "ANEServices"] {
        let path = format!("/System/Library/PrivateFrameworks/{name}.framework/{name}");
        let c = CString::new(path).unwrap();
        unsafe {
            dlopen(c.as_ptr(), RTLD_NOW);
        }
    }

    let mil_text = "program(1, 0)\nfunc main<ios16>(tensor<fp16, [1,16,1,1]> x) -> (tensor<fp16, [1,16,1,1]>) {\n  block0() {\n    tensor<fp16, [1,16,1,1]> y = relu()[x = x];\n  } -> (y)\n}\n";

    unsafe {
        type DataFn = unsafe extern "C" fn(ObjcId, ObjcSel, *const u8, u64) -> ObjcId;
        type DictFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type DescFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, ObjcId) -> ObjcId;
        type ModelFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type StrFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type Utf8Fn = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const std::ffi::c_char;
        type ModelFn2 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, u8) -> ObjcId;
        type CompileFn = unsafe extern "C" fn(ObjcId, ObjcSel, u32, ObjcId, *mut ObjcId) -> bool;
        type StrFn2 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type DictMakeFn =
            unsafe extern "C" fn(ObjcId, ObjcSel, *const ObjcId, *const ObjcId, u64) -> ObjcId;
        type SetDictFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId);
        type SetKVFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId);

        let cls_data = cls("NSData");
        let cls_dict = cls("NSDictionary");
        let cls_mdict = cls("NSMutableDictionary");
        let cls_desc = cls("_ANEInMemoryModelDescriptor");
        let cls_model = cls("_ANEInMemoryModel");
        let cls_str = cls("NSString");

        let dataf: DataFn = std::mem::transmute(objc_msgSend as *const c_void);
        let dictf: DictFn = std::mem::transmute(objc_msgSend as *const c_void);
        let descf: DescFn = std::mem::transmute(objc_msgSend as *const c_void);
        let modelf: ModelFn = std::mem::transmute(objc_msgSend as *const c_void);
        let strf: StrFn = std::mem::transmute(objc_msgSend as *const c_void);
        let utf8f: Utf8Fn = std::mem::transmute(objc_msgSend as *const c_void);
        let modelf2: ModelFn2 = std::mem::transmute(objc_msgSend as *const c_void);
        let compilef: CompileFn = std::mem::transmute(objc_msgSend as *const c_void);
        let strf2: StrFn2 = std::mem::transmute(objc_msgSend as *const c_void);
        let dkv: DictMakeFn = std::mem::transmute(objc_msgSend as *const c_void);
        let setd: SetDictFn = std::mem::transmute(objc_msgSend as *const c_void);
        let setkv: SetKVFn = std::mem::transmute(objc_msgSend as *const c_void);

        let make_nsstr = |s: &str| -> ObjcId {
            let c = CString::new(s).unwrap();
            strf2(
                cls_str as ObjcId,
                sel("stringWithUTF8String:"),
                c.as_ptr() as ObjcId,
            )
        };

        let bytes = mil_text.as_bytes();
        let ns_text = dataf(
            cls_data as ObjcId,
            sel("dataWithBytes:length:"),
            bytes.as_ptr(),
            bytes.len() as u64,
        );
        let empty = dictf(cls_dict as ObjcId, sel("dictionary"));

        // Create flipped model (isMILModel=false → kANEFModelANECIR)
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
        if model.is_null() {
            println!("model=null");
            return Ok(());
        }

        let lp = nsstring_to_str(strf(model, sel("localModelPath")));
        println!("localModelPath: {lp}");

        // Get base opts
        let base_opts = modelf2(
            model,
            sel("compilerOptionsWithOptions:isCompiledModelCached:"),
            empty,
            0,
        );
        println!("base opts: {}", objc_desc(base_opts));

        // Write espresso files to localModelPath
        std::fs::create_dir_all(&lp)?;
        for fname in &[
            "model.espresso.net",
            "model.espresso.shape",
            "model.espresso.weights",
        ] {
            let src = format!("{ESPRESSO_BUNDLE}/{fname}");
            if let Ok(content) = std::fs::read(&src) {
                std::fs::write(format!("{lp}/{fname}"), &content)?;
                println!("wrote {fname}: {}B", content.len());
            }
        }

        // Build mutable options dict with NetworkSourceFileName/Path injected
        let key_names = [
            "NetworkSourceFileName",
            "NetworkSourcePath",
            "NetworkJITShapesName",
            "NetworkJITShapesPath",
        ];
        let val_names = ["model.espresso.net", &lp, "", ""];

        // Start from base opts and add extra keys
        let mut_opts = dictf(cls_mdict as ObjcId, sel("new"));
        setd(mut_opts, sel("setDictionary:"), base_opts);

        let mut injected: Vec<(String, String)> = Vec::new();

        for (k, v) in key_names.iter().zip(val_names.iter()) {
            if v.is_empty() {
                continue;
            }
            let ns_k = make_nsstr(k);
            let ns_v = make_nsstr(v);
            setkv(mut_opts, sel("setObject:forKey:"), ns_v, ns_k);
            injected.push((k.to_string(), v.to_string()));
        }
        println!("\ninjected opts: {}", objc_desc(mut_opts));

        // Also test with fresh model
        let desc2 = descf(
            cls_desc as ObjcId,
            sel("modelWithMILText:weights:optionsPlist:"),
            ns_text,
            empty,
            std::ptr::null_mut(),
        );
        let p2 = (desc2 as *mut u8).add(8);
        *p2 = 0;
        let model2 = modelf(
            cls_model as ObjcId,
            sel("inMemoryModelWithDescriptor:"),
            desc2,
        );
        if model2.is_null() {
            println!("model2=null");
            return Ok(());
        }

        let mut err: ObjcId = std::ptr::null_mut();
        let ok = compilef(
            model2,
            sel("compileWithQoS:options:error:"),
            21,
            mut_opts,
            &mut err,
        );
        if ok {
            println!("\n*** COMPILE SUCCESS! ***");
            list_dir(&lp);
        } else {
            let e = nserror_string(err).unwrap_or_default();
            println!("\nerror: {}", &e[..e.len().min(500)]);
        }

        // --- Try each key individually to find which one matters ---
        println!("\n=== Individual key probes ===");
        for (k, v) in &[
            ("NetworkSourceFileName", "model.espresso.net"),
            ("NetworkSourcePath", lp.as_str()),
            ("NetworkSourceFileName", "model.espresso.net"), // without path
        ] {
            let opts_single = dictf(cls_mdict as ObjcId, sel("new"));
            setd(opts_single, sel("setDictionary:"), base_opts);
            let ns_k = make_nsstr(k);
            let ns_v = make_nsstr(v);
            setkv(opts_single, sel("setObject:forKey:"), ns_v, ns_k);
            println!("  [{k}={v}]:");

            let d = descf(
                cls_desc as ObjcId,
                sel("modelWithMILText:weights:optionsPlist:"),
                ns_text,
                empty,
                std::ptr::null_mut(),
            );
            let pp = (d as *mut u8).add(8);
            *pp = 0;
            let m = modelf(cls_model as ObjcId, sel("inMemoryModelWithDescriptor:"), d);
            if m.is_null() {
                println!("    model=null");
                continue;
            }

            let mut e: ObjcId = std::ptr::null_mut();
            let ok = compilef(
                m,
                sel("compileWithQoS:options:error:"),
                21,
                opts_single,
                &mut e,
            );
            if ok {
                println!("    *** SUCCESS! ***");
                list_dir(&lp);
            } else {
                let es = nserror_string(e).unwrap_or_default();
                // Extract innermost error
                let code = inner_error(&es);
                println!("    {code}");
            }
        }

        // --- Also try: kANEFModelType = kANEFModelANECIR + NetworkSourceFileName ---
        println!("\n=== Full compile opts with NetworkSourceFileName ===");
        let opts_full = dictf(cls_mdict as ObjcId, sel("new"));
        setd(opts_full, sel("setDictionary:"), base_opts);
        setkv(
            opts_full,
            sel("setObject:forKey:"),
            make_nsstr("model.espresso.net"),
            make_nsstr("NetworkSourceFileName"),
        );
        setkv(
            opts_full,
            sel("setObject:forKey:"),
            make_nsstr(&lp),
            make_nsstr("NetworkSourcePath"),
        );

        let d3 = descf(
            cls_desc as ObjcId,
            sel("modelWithMILText:weights:optionsPlist:"),
            ns_text,
            empty,
            std::ptr::null_mut(),
        );
        let pp3 = (d3 as *mut u8).add(8);
        *pp3 = 0;
        let m3 = modelf(cls_model as ObjcId, sel("inMemoryModelWithDescriptor:"), d3);

        let mut e3: ObjcId = std::ptr::null_mut();
        let ok3 = compilef(
            m3,
            sel("compileWithQoS:options:error:"),
            21,
            opts_full,
            &mut e3,
        );
        if ok3 {
            println!("  *** SUCCESS! ***");
            list_dir(&lp);
        } else {
            let es3 = nserror_string(e3).unwrap_or_default();
            println!("  {}", &es3[..es3.len().min(500)]);
        }

        std::fs::remove_dir_all(&lp).ok();
    }
    Ok(())
}

fn inner_error(e: &str) -> String {
    if let Some(idx) = e.rfind("err=(\n    ") {
        let rest = &e[idx + 9..];
        if let Some(end) = rest.find('\n') {
            return rest[..end].trim().to_string();
        }
    }
    e[..e.len().min(250)].to_string()
}

unsafe fn nsstring_to_str(obj: ObjcId) -> String {
    if obj.is_null() {
        return "(null)".to_string();
    }
    type Utf8Fn = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const std::ffi::c_char;
    let uf: Utf8Fn = std::mem::transmute(objc_msgSend as *const c_void);
    let p = uf(obj, sel("UTF8String"));
    if p.is_null() {
        return "(null utf8)".to_string();
    }
    CStr::from_ptr(p).to_string_lossy().into_owned()
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

fn list_dir(dir: &str) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        println!("  dir:");
        for entry in entries.flatten() {
            let meta = std::fs::metadata(entry.path()).ok();
            let size = meta.map(|m| m.len()).unwrap_or(0);
            println!("    {}: {size}B", entry.file_name().to_string_lossy());
        }
    }
}
