//! Hypothesis: when isMILModel=false, _networkText (offset 0x28) is read as filename,
//! not as program text. Test: swap _networkText from NSData(MIL) to NSString("model.espresso.net").
//!
//! Also probe initWithNetworkText:...:isMILModel: with NSString arg.
//!
//! Run: cargo run -p rane --example anec_nettext_swap --release

use rane::ffi::*;
use std::ffi::{c_void, CStr, CString};

const ESPRESSO_BUNDLE: &str = "/System/Library/DuetExpertCenter/Assets/Assets.bundle/AssetData/ATXActionValuationMLModel.mlmodelc";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== _networkText swap probe ===\n");

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
        type RetainFn = unsafe extern "C" fn(ObjcId) -> ObjcId;

        let cls_data = cls("NSData");
        let cls_dict = cls("NSDictionary");
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

        // --- Test A: swap _networkText to NSString("model.espresso.net") at offset 0x28 ---
        println!("=== Test A: _networkText swap at offset 0x28 ===");
        let desc_a = descf(
            cls_desc as ObjcId,
            sel("modelWithMILText:weights:optionsPlist:"),
            ns_text,
            empty,
            std::ptr::null_mut(),
        );

        // Flip isMILModel at +8
        let p_mil = (desc_a as *mut u8).add(8);
        *p_mil = 0;

        // Swap _networkText at +0x28 (40) to NSString("model.espresso.net")
        // Must retain the new string; let old NSData leak (no way to release safely here)
        let ns_fname = make_nsstr("model.espresso.net");
        let p_text = (desc_a as *mut ObjcId).add(5); // offset 40 = 5 * 8 bytes
        *p_text = ns_fname;

        // Verify
        let model_a = modelf(
            cls_model as ObjcId,
            sel("inMemoryModelWithDescriptor:"),
            desc_a,
        );
        if model_a.is_null() {
            println!("  model_a=null");
        } else {
            let lp = strf(model_a, sel("localModelPath"));
            let lp_s = nsstring_to_str(lp);
            println!("  localModelPath: {lp_s}");

            let opts = modelf2(
                model_a,
                sel("compilerOptionsWithOptions:isCompiledModelCached:"),
                empty,
                0,
            );
            println!("  opts: {}", objc_desc(opts));

            // Write espresso files to localModelPath
            std::fs::create_dir_all(&lp_s)?;
            for fname in &[
                "model.espresso.net",
                "model.espresso.shape",
                "model.espresso.weights",
            ] {
                let src = format!("{ESPRESSO_BUNDLE}/{fname}");
                let dst = format!("{lp_s}/{fname}");
                // SIP-protected; try reading content directly
                if let Ok(content) = std::fs::read(&src) {
                    std::fs::write(&dst, &content)?;
                    println!("  wrote {fname}: {}B", content.len());
                } else {
                    println!("  FAILED read {fname}");
                }
            }

            let mut err: ObjcId = std::ptr::null_mut();
            let ok = compilef(
                model_a,
                sel("compileWithQoS:options:error:"),
                21,
                empty,
                &mut err,
            );
            if ok {
                println!("  *** COMPILE SUCCESS! ***");
                list_dir(&lp_s);
            } else {
                println!("  err: {:?}", nserror_string(err));
            }
            std::fs::remove_dir_all(&lp_s).ok();
        }

        // --- Test B: swap _networkText to NSData of espresso content ---
        println!("\n=== Test B: _networkText swap to NSData(espresso.net content) ===");
        let espresso_content = std::fs::read(format!("{ESPRESSO_BUNDLE}/model.espresso.net"))?;
        println!("  espresso.net content: {}B", espresso_content.len());

        let desc_b = descf(
            cls_desc as ObjcId,
            sel("modelWithMILText:weights:optionsPlist:"),
            ns_text,
            empty,
            std::ptr::null_mut(),
        );
        let p_mil_b = (desc_b as *mut u8).add(8);
        *p_mil_b = 0;

        // Replace _networkText with NSData of espresso content
        let ns_esp_data = dataf(
            cls_data as ObjcId,
            sel("dataWithBytes:length:"),
            espresso_content.as_ptr(),
            espresso_content.len() as u64,
        );
        let p_text_b = (desc_b as *mut ObjcId).add(5); // offset 40
        *p_text_b = ns_esp_data;

        let model_b = modelf(
            cls_model as ObjcId,
            sel("inMemoryModelWithDescriptor:"),
            desc_b,
        );
        if model_b.is_null() {
            println!("  model_b=null");
        } else {
            let lp_b = nsstring_to_str(strf(model_b, sel("localModelPath")));
            println!("  localModelPath: {lp_b}");
            let opts_b = modelf2(
                model_b,
                sel("compilerOptionsWithOptions:isCompiledModelCached:"),
                empty,
                0,
            );
            println!("  opts: {}", objc_desc(opts_b));

            // Write all espresso files
            std::fs::create_dir_all(&lp_b)?;
            for fname in &[
                "model.espresso.net",
                "model.espresso.shape",
                "model.espresso.weights",
            ] {
                let src = format!("{ESPRESSO_BUNDLE}/{fname}");
                if let Ok(content) = std::fs::read(&src) {
                    std::fs::write(format!("{lp_b}/{fname}"), &content)?;
                    println!("  wrote {fname}: {}B", content.len());
                }
            }

            let mut err: ObjcId = std::ptr::null_mut();
            let ok = compilef(
                model_b,
                sel("compileWithQoS:options:error:"),
                21,
                empty,
                &mut err,
            );
            if ok {
                println!("  *** COMPILE SUCCESS! ***");
                list_dir(&lp_b);
            } else {
                println!("  err: {:?}", nserror_string(err));
            }
            std::fs::remove_dir_all(&lp_b).ok();
        }

        // --- Test C: _networkText = NSString (full path to espresso.net) ---
        println!("\n=== Test C: _networkText = full path NSString ===");
        // Create a tmp dir, copy espresso files, use path as networkText
        let tmp_esp = format!("/tmp/espresso_test_{}", std::process::id());
        std::fs::create_dir_all(&tmp_esp)?;
        for fname in &[
            "model.espresso.net",
            "model.espresso.shape",
            "model.espresso.weights",
        ] {
            let src = format!("{ESPRESSO_BUNDLE}/{fname}");
            if let Ok(content) = std::fs::read(&src) {
                std::fs::write(format!("{tmp_esp}/{fname}"), &content)?;
            }
        }

        let desc_c = descf(
            cls_desc as ObjcId,
            sel("modelWithMILText:weights:optionsPlist:"),
            ns_text,
            empty,
            std::ptr::null_mut(),
        );
        let p_mil_c = (desc_c as *mut u8).add(8);
        *p_mil_c = 0;

        // Use the full path to the espresso.net file as networkText
        let full_path_str = make_nsstr(&format!("{tmp_esp}/model.espresso.net"));
        let p_text_c = (desc_c as *mut ObjcId).add(5);
        *p_text_c = full_path_str;

        let model_c = modelf(
            cls_model as ObjcId,
            sel("inMemoryModelWithDescriptor:"),
            desc_c,
        );
        if model_c.is_null() {
            println!("  model_c=null");
        } else {
            let lp_c = nsstring_to_str(strf(model_c, sel("localModelPath")));
            println!("  localModelPath: {lp_c}");
            let opts_c = modelf2(
                model_c,
                sel("compilerOptionsWithOptions:isCompiledModelCached:"),
                empty,
                0,
            );
            println!("  opts: {}", objc_desc(opts_c));

            // Also copy espresso files to the model's localModelPath
            std::fs::create_dir_all(&lp_c)?;
            for fname in &[
                "model.espresso.net",
                "model.espresso.shape",
                "model.espresso.weights",
            ] {
                let src = format!("{tmp_esp}/{fname}");
                if let Ok(content) = std::fs::read(&src) {
                    std::fs::write(format!("{lp_c}/{fname}"), &content)?;
                }
            }

            let mut err: ObjcId = std::ptr::null_mut();
            let ok = compilef(
                model_c,
                sel("compileWithQoS:options:error:"),
                21,
                empty,
                &mut err,
            );
            if ok {
                println!("  *** COMPILE SUCCESS! ***");
                list_dir(&lp_c);
            } else {
                println!("  err: {:?}", nserror_string(err));
            }
            std::fs::remove_dir_all(&lp_c).ok();
        }
        std::fs::remove_dir_all(&tmp_esp).ok();

        // --- Test D: read ivars of desc BEFORE isMILModel flip ---
        println!("\n=== Test D: ivar dump of fresh descriptor ===");
        let desc_d = descf(
            cls_desc as ObjcId,
            sel("modelWithMILText:weights:optionsPlist:"),
            ns_text,
            empty,
            std::ptr::null_mut(),
        );
        // Print raw pointer layout
        for i in 0..8usize {
            let ptr = (desc_d as *const ObjcId).add(i);
            let val = *ptr;
            println!("  +{:02x}:{:03} = {:p}", i * 8, i * 8, val);
            if !val.is_null() && i >= 2 {
                // Try to print class name
                let isa = *(val as *const ObjcId);
                if !isa.is_null() {
                    let cn = class_getName(isa as ObjcClass);
                    if !cn.is_null() {
                        println!("          [{}]", CStr::from_ptr(cn).to_str().unwrap_or("?"));
                    }
                }
            }
        }
    }

    Ok(())
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
