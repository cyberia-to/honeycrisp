//! Fix: use localModelPath (not /tmp/) to place model.espresso.net.
//! Use a real Espresso bundle from the system to test the ANECIR compile path.
//!
//! Run: cargo run -p rane --example anec_path_fix --release

use rane::ffi::*;
use std::ffi::{c_void, CStr, CString};

// Path to a real Espresso bundle on the system (small, non-encrypted)
const ESPRESSO_BUNDLE: &str = "/System/Library/DuetExpertCenter/Assets/Assets.bundle/AssetData/ATXActionValuationMLModel.mlmodelc";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ANEC IR path-fix probe ===\n");

    for name in &["AppleNeuralEngine", "ANECompiler", "ANEServices"] {
        let path = format!("/System/Library/PrivateFrameworks/{name}.framework/{name}");
        let c = CString::new(path).unwrap();
        unsafe {
            dlopen(c.as_ptr(), RTLD_NOW);
        }
    }

    // Minimal MIL text (for creating descriptor and getting hexId/localModelPath)
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

        // Create flipped descriptor + model
        let desc = descf(
            cls_desc as ObjcId,
            sel("modelWithMILText:weights:optionsPlist:"),
            ns_text,
            empty,
            std::ptr::null_mut(),
        );
        let p = (desc as *mut u8).add(8);
        *p = 0; // isMILModel=false → kANEFModelANECIR
        let model = modelf(
            cls_model as ObjcId,
            sel("inMemoryModelWithDescriptor:"),
            desc,
        );
        if model.is_null() {
            println!("model=null");
            return Ok(());
        }

        // Get localModelPath — this is where the XPC service looks for files
        let local_path_ns = strf(model, sel("localModelPath"));
        let local_path = {
            let c = utf8f(local_path_ns, sel("UTF8String"));
            CStr::from_ptr(c).to_string_lossy().into_owned()
        };
        println!("localModelPath: {local_path}");

        // Confirm model type
        let opts = modelf2(
            model,
            sel("compilerOptionsWithOptions:isCompiledModelCached:"),
            empty,
            0,
        );
        println!("compilerOpts: {}", objc_desc(opts));

        // --- Test 1: write model.espresso.{net,shape,weights} to localModelPath ---
        println!("\n=== Test 1: Real Espresso files at localModelPath ===");
        std::fs::create_dir_all(&local_path)?;

        for fname in &[
            "model.espresso.net",
            "model.espresso.shape",
            "model.espresso.weights",
        ] {
            let src = format!("{ESPRESSO_BUNDLE}/{fname}");
            let dst = format!("{local_path}/{fname}");
            match std::fs::copy(&src, &dst) {
                Ok(n) => println!("  copied {fname}: {n}B"),
                Err(e) => println!("  FAILED to copy {fname}: {e}"),
            }
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
            println!("  *** COMPILE SUCCESS! ***");
            list_dir(&local_path);
        } else {
            println!("  error: {:?}", nserror_string(err));
        }

        // Cleanup
        for fname in &[
            "model.espresso.net",
            "model.espresso.shape",
            "model.espresso.weights",
        ] {
            std::fs::remove_file(format!("{local_path}/{fname}")).ok();
        }

        // --- Test 2: full bundle copy (including coremldata.bin, metadata.json, model dir) ---
        println!("\n=== Test 2: Full bundle files at localModelPath ===");
        // Copy everything from ESPRESSO_BUNDLE into local_path
        for entry in std::fs::read_dir(ESPRESSO_BUNDLE)? {
            let entry = entry?;
            let fname = entry.file_name();
            let dst = format!("{local_path}/{}", fname.to_string_lossy());
            let src = entry.path();
            if src.is_file() {
                std::fs::copy(&src, &dst)?;
                println!("  copied {}", fname.to_string_lossy());
            }
        }

        // Create a fresh model (same descriptor, same hex, same local_path)
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
            println!("  model2=null");
            return Ok(());
        }

        let mut err2: ObjcId = std::ptr::null_mut();
        let ok2 = compilef(
            model2,
            sel("compileWithQoS:options:error:"),
            21,
            empty,
            &mut err2,
        );
        if ok2 {
            println!("  *** COMPILE SUCCESS! ***");
            list_dir(&local_path);
        } else {
            println!("  error: {:?}", nserror_string(err2));
        }

        // --- Test 3: try modelWithNetworkDescription: with NSString path ---
        println!("\n=== Test 3: modelWithNetworkDescription: with NSString path ===");
        let path_str = make_nsstr(&format!("{local_path}/model.espresso.net"));
        // descriptor3 via networkDescription=NSString(path)
        let desc3 = descf(
            cls_desc as ObjcId,
            sel("modelWithNetworkDescription:weights:optionsPlist:"),
            path_str,
            empty,
            std::ptr::null_mut(),
        );
        if desc3.is_null() {
            println!("  desc3=null (NSString path didn't work)");
        } else {
            let is_mil3 = *(desc3 as *const u8).add(8);
            println!("  desc3={desc3:p} isMILModel={is_mil3}");
            let model3 = modelf(
                cls_model as ObjcId,
                sel("inMemoryModelWithDescriptor:"),
                desc3,
            );
            if !model3.is_null() {
                let lp3 = strf(model3, sel("localModelPath"));
                println!("  localModelPath3 = {}", nsstring_to_str(lp3));
                let opts3 = modelf2(
                    model3,
                    sel("compilerOptionsWithOptions:isCompiledModelCached:"),
                    empty,
                    0,
                );
                println!("  opts3 = {}", objc_desc(opts3));
                let mut err3: ObjcId = std::ptr::null_mut();
                let ok3 = compilef(
                    model3,
                    sel("compileWithQoS:options:error:"),
                    21,
                    empty,
                    &mut err3,
                );
                if ok3 {
                    println!("  *** SUCCESS! ***");
                } else {
                    println!("  err3: {:?}", nserror_string(err3));
                }
            }
        }

        // --- Test 4: modelWithNetworkDescription: with NSData ---
        println!("\n=== Test 4: modelWithNetworkDescription: with NSData(espresso.net) ===");
        let content = std::fs::read(format!("{ESPRESSO_BUNDLE}/model.espresso.net"))?;
        let ns_data = dataf(
            cls_data as ObjcId,
            sel("dataWithBytes:length:"),
            content.as_ptr(),
            content.len() as u64,
        );
        let desc4 = descf(
            cls_desc as ObjcId,
            sel("modelWithNetworkDescription:weights:optionsPlist:"),
            ns_data,
            empty,
            std::ptr::null_mut(),
        );
        if desc4.is_null() {
            println!("  desc4=null (NSData didn't work)");
        } else {
            let is_mil4 = *(desc4 as *const u8).add(8);
            println!("  desc4={desc4:p} isMILModel={is_mil4}");
            let model4 = modelf(
                cls_model as ObjcId,
                sel("inMemoryModelWithDescriptor:"),
                desc4,
            );
            if !model4.is_null() {
                let lp4 = strf(model4, sel("localModelPath"));
                println!("  localModelPath4 = {}", nsstring_to_str(lp4));
                let opts4 = modelf2(
                    model4,
                    sel("compilerOptionsWithOptions:isCompiledModelCached:"),
                    empty,
                    0,
                );
                println!("  opts4 = {}", objc_desc(opts4));
                let hex4 = strf(model4, sel("hexStringIdentifier"));
                let local4 = nsstring_to_str(lp4);
                // Write espresso files at the new localModelPath
                std::fs::create_dir_all(&local4).ok();
                for fname in &[
                    "model.espresso.net",
                    "model.espresso.shape",
                    "model.espresso.weights",
                ] {
                    let src = format!("{ESPRESSO_BUNDLE}/{fname}");
                    let dst = format!("{local4}/{fname}");
                    std::fs::copy(&src, &dst).ok();
                }
                let mut err4: ObjcId = std::ptr::null_mut();
                let ok4 = compilef(
                    model4,
                    sel("compileWithQoS:options:error:"),
                    21,
                    empty,
                    &mut err4,
                );
                if ok4 {
                    println!("  *** SUCCESS! ***");
                    list_dir(&local4);
                } else {
                    println!("  err4: {:?}", nserror_string(err4));
                }
                std::fs::remove_dir_all(&local4).ok();
            }
        }

        std::fs::remove_dir_all(&local_path).ok();
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
        println!("  dir contents:");
        for entry in entries.flatten() {
            let meta = std::fs::metadata(entry.path()).ok();
            let size = meta.map(|m| m.len()).unwrap_or(0);
            println!("    {}: {size}B", entry.file_name().to_string_lossy());
        }
    }
}
