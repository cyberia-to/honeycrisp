//! dlopen ANECompilerService.xpc directly to access _ANECVAIRCompiler,
//! call defaultANECIRFileName, and read kANEFNetPlistFilenameKey.
//!
//! Run: cargo run -p rane --example anec_xpc_dlopen --release

use rane::ffi::*;
use std::ffi::{c_void, CStr, CString};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ANECompilerService.xpc dlopen probe ===\n");

    // Load standard frameworks
    for name in &["AppleNeuralEngine", "ANECompiler", "ANEServices"] {
        let path = format!("/System/Library/PrivateFrameworks/{name}.framework/{name}");
        let c = CString::new(path).unwrap();
        unsafe {
            dlopen(c.as_ptr(), RTLD_NOW);
        }
    }

    // dlopen the XPC service binary directly
    let xpc_path = "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/XPCServices/ANECompilerService.xpc/Contents/MacOS/ANECompilerService";
    let c = CString::new(xpc_path).unwrap();
    let handle = unsafe { dlopen(c.as_ptr(), RTLD_NOW) };
    if handle.is_null() {
        println!("dlopen ANECompilerService failed: {}", unsafe {
            let e = dlerror();
            if e.is_null() {
                "(unknown)".to_string()
            } else {
                CStr::from_ptr(e).to_string_lossy().into_owned()
            }
        });
    } else {
        println!("dlopen ANECompilerService: OK @ {handle:p}");
    }

    unsafe {
        type StrFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type Utf8Fn = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const std::ffi::c_char;
        type DictFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type StrFn2 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type DictMakeFn =
            unsafe extern "C" fn(ObjcId, ObjcSel, *const ObjcId, *const ObjcId, u64) -> ObjcId;
        type DataFn = unsafe extern "C" fn(ObjcId, ObjcSel, *const u8, u64) -> ObjcId;
        type DescFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, ObjcId) -> ObjcId;
        type ModelFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type ModelFn2 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, u8) -> ObjcId;
        type CompileFn = unsafe extern "C" fn(ObjcId, ObjcSel, u32, ObjcId, *mut ObjcId) -> bool;
        type SetKVFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId);
        type SetDictFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId);

        let dictf: DictFn = std::mem::transmute(objc_msgSend as *const c_void);
        let strf2: StrFn2 = std::mem::transmute(objc_msgSend as *const c_void);
        let dkv: DictMakeFn = std::mem::transmute(objc_msgSend as *const c_void);
        let dataf: DataFn = std::mem::transmute(objc_msgSend as *const c_void);
        let descf: DescFn = std::mem::transmute(objc_msgSend as *const c_void);
        let modelf: ModelFn = std::mem::transmute(objc_msgSend as *const c_void);
        let modelf2: ModelFn2 = std::mem::transmute(objc_msgSend as *const c_void);
        let compilef: CompileFn = std::mem::transmute(objc_msgSend as *const c_void);
        let strf: StrFn = std::mem::transmute(objc_msgSend as *const c_void);
        let setkv: SetKVFn = std::mem::transmute(objc_msgSend as *const c_void);
        let setd: SetDictFn = std::mem::transmute(objc_msgSend as *const c_void);

        let cls_str = cls("NSString");
        let cls_dict = cls("NSDictionary");
        let cls_mdict = cls("NSMutableDictionary");
        let cls_data = cls("NSData");
        let cls_desc = cls("_ANEInMemoryModelDescriptor");
        let cls_model = cls("_ANEInMemoryModel");

        let make_nsstr = |s: &str| -> ObjcId {
            let c = CString::new(s).unwrap();
            strf2(
                cls_str as ObjcId,
                sel("stringWithUTF8String:"),
                c.as_ptr() as ObjcId,
            )
        };

        // --- 1. Read all kANEF* constants now that XPC service is loaded ---
        println!("=== 1. kANEF* constants from XPC service ===");
        let const_names = [
            "kANEFModelANECIRValue",
            "kANEFModelCoreMLValue",
            "kANEFModelMILValue",
            "kANEFModelMLIRValue",
            "kANEFModelLLIRBundleValue",
            "kANEFModelPreCompiledValue",
            "kANEFIsInMemoryModelTypeKey",
            "kANEFInMemoryModelIsCachedKey",
            "kANEFEspressoFileResourcesKey",
            "kANEFModelDescriptionKey",
            "kANEFNetPlistFilenameKey",
            "kANEFCompilerOptionsFilenameKey",
            "kANEFModelType",
            "kANEFModelIdentityStrKey",
            "kANEFBaseModelIdentifierKey",
            "kANEFModelIsEncryptedKey",
            "kANEFModelHasCacheURLIdentifierKey",
            "kANEFModelCacheIdentifierUsingSourceURLKey",
            "kANEFRetainModelsWithoutSourceURLKey",
            "kANEFCompilationInitiatedByE5MLKey",
        ];
        let mut kaneF: std::collections::HashMap<String, ObjcId> = Default::default();
        for name in &const_names {
            let c = CString::new(*name).unwrap();
            let p = dlsym(std::ptr::null_mut(), c.as_ptr()) as *const ObjcId;
            if !p.is_null() && !(*p).is_null() {
                let val = *p;
                kaneF.insert(name.to_string(), val);
                println!("  {name} = \"{}\"", nsstring_to_str(val));
            } else {
                println!("  {name} = NOT FOUND");
            }
        }

        // --- 2. Enumerate _ANECVAIRCompiler methods ---
        println!("\n=== 2. _ANECVAIRCompiler methods ===");
        let cvair_cls = cls("_ANECVAIRCompiler");
        if cvair_cls.is_null() {
            println!("  NOT FOUND");
        } else {
            println!("  found: {cvair_cls:p}");
            let mut count: u32 = 0;
            let methods = class_copyMethodList(cvair_cls, &mut count);
            if !methods.is_null() {
                for i in 0..count as usize {
                    let m = *methods.add(i);
                    let s = method_getName(m);
                    if !s.is_null() {
                        let np = sel_getName(s);
                        if !np.is_null() {
                            let enc = method_getTypeEncoding(m);
                            let enc_s = if enc.is_null() {
                                "?"
                            } else {
                                CStr::from_ptr(enc).to_str().unwrap_or("?")
                            };
                            println!(
                                "    {} [{}]",
                                CStr::from_ptr(np).to_str().unwrap_or("?"),
                                enc_s
                            );
                        }
                    }
                }
                libc_free(methods as *mut c_void);
            }

            // Call defaultANECIRFileName
            let has_default = class_getClassMethod(cvair_cls, sel("defaultANECIRFileName"));
            let has_default_inst = class_getInstanceMethod(cvair_cls, sel("defaultANECIRFileName"));
            println!(
                "  defaultANECIRFileName: class_method={:p} inst={:p}",
                has_default, has_default_inst
            );
        }

        // --- 3. Enumerate _ANEEspressoIRTranslator methods ---
        println!("\n=== 3. _ANEEspressoIRTranslator methods ===");
        let esp_cls = cls("_ANEEspressoIRTranslator");
        if esp_cls.is_null() {
            println!("  NOT FOUND");
        } else {
            println!("  found: {esp_cls:p}");
            let mut count: u32 = 0;
            let methods = class_copyMethodList(esp_cls, &mut count);
            if !methods.is_null() {
                for i in 0..count as usize {
                    let m = *methods.add(i);
                    let s = method_getName(m);
                    if !s.is_null() {
                        let np = sel_getName(s);
                        if !np.is_null() {
                            let enc = method_getTypeEncoding(m);
                            let enc_s = if enc.is_null() {
                                "?"
                            } else {
                                CStr::from_ptr(enc).to_str().unwrap_or("?")
                            };
                            println!(
                                "    {} [{}]",
                                CStr::from_ptr(np).to_str().unwrap_or("?"),
                                enc_s
                            );
                        }
                    }
                }
                libc_free(methods as *mut c_void);
            }
        }

        // --- 4. Try inject kANEFNetPlistFilenameKey ---
        println!("\n=== 4. Inject kANEFNetPlistFilenameKey ===");
        let mil_text = "program(1, 0)\nfunc main<ios16>(tensor<fp16, [1,16,1,1]> x) -> (tensor<fp16, [1,16,1,1]>) {\n  block0() {\n    tensor<fp16, [1,16,1,1]> y = relu()[x = x];\n  } -> (y)\n}\n";
        let ns_text = dataf(
            cls_data as ObjcId,
            sel("dataWithBytes:length:"),
            mil_text.as_bytes().as_ptr(),
            mil_text.len() as u64,
        );
        let empty = dictf(cls_dict as ObjcId, sel("dictionary"));

        let desc = descf(
            cls_desc as ObjcId,
            sel("modelWithMILText:weights:optionsPlist:"),
            ns_text,
            empty,
            std::ptr::null_mut(),
        );
        let p = (desc as *mut u8).add(8);
        *p = 0; // flip isMILModel
        let model = modelf(
            cls_model as ObjcId,
            sel("inMemoryModelWithDescriptor:"),
            desc,
        );
        if model.is_null() {
            println!("  model=null");
            return Ok(());
        }

        let lp = nsstring_to_str(strf(model, sel("localModelPath")));
        println!("  localModelPath: {lp}");

        // Get base opts
        let base_opts = modelf2(
            model,
            sel("compilerOptionsWithOptions:isCompiledModelCached:"),
            empty,
            0,
        );

        // Write espresso files
        const ESPRESSO: &str = "/System/Library/DuetExpertCenter/Assets/Assets.bundle/AssetData/ATXActionValuationMLModel.mlmodelc";
        std::fs::create_dir_all(&lp)?;
        for fname in &[
            "model.espresso.net",
            "model.espresso.shape",
            "model.espresso.weights",
        ] {
            if let Ok(content) = std::fs::read(format!("{ESPRESSO}/{fname}")) {
                std::fs::write(format!("{lp}/{fname}"), &content)?;
                println!("  wrote {fname}: {}B", content.len());
            }
        }

        // Try with kANEFNetPlistFilenameKey if found
        if let Some(&net_plist_key) = kaneF.get("kANEFNetPlistFilenameKey") {
            let mut_opts = dictf(cls_mdict as ObjcId, sel("new"));
            setd(mut_opts, sel("setDictionary:"), base_opts);
            setkv(
                mut_opts,
                sel("setObject:forKey:"),
                make_nsstr("model.espresso.net"),
                net_plist_key,
            );
            println!(
                "  opts with kANEFNetPlistFilenameKey: {}",
                objc_desc(mut_opts)
            );

            let d2 = descf(
                cls_desc as ObjcId,
                sel("modelWithMILText:weights:optionsPlist:"),
                ns_text,
                empty,
                std::ptr::null_mut(),
            );
            let pp2 = (d2 as *mut u8).add(8);
            *pp2 = 0;
            let m2 = modelf(cls_model as ObjcId, sel("inMemoryModelWithDescriptor:"), d2);
            if !m2.is_null() {
                let mut err: ObjcId = std::ptr::null_mut();
                let ok = compilef(
                    m2,
                    sel("compileWithQoS:options:error:"),
                    21,
                    mut_opts,
                    &mut err,
                );
                if ok {
                    println!("  *** SUCCESS! ***");
                    list_dir(&lp);
                } else {
                    println!(
                        "  err: {}",
                        inner_error(&nserror_string(err).unwrap_or_default())
                    );
                }
            }
        } else {
            // kANEFNetPlistFilenameKey not found — inject raw string key
            let mut_opts = dictf(cls_mdict as ObjcId, sel("new"));
            setd(mut_opts, sel("setDictionary:"), base_opts);
            // Use raw key name from XPC binary
            setkv(
                mut_opts,
                sel("setObject:forKey:"),
                make_nsstr("model.espresso.net"),
                make_nsstr("kANEFNetPlistFilenameKey"),
            );
            let d2 = descf(
                cls_desc as ObjcId,
                sel("modelWithMILText:weights:optionsPlist:"),
                ns_text,
                empty,
                std::ptr::null_mut(),
            );
            let pp2 = (d2 as *mut u8).add(8);
            *pp2 = 0;
            let m2 = modelf(cls_model as ObjcId, sel("inMemoryModelWithDescriptor:"), d2);
            if !m2.is_null() {
                let mut err: ObjcId = std::ptr::null_mut();
                let ok = compilef(
                    m2,
                    sel("compileWithQoS:options:error:"),
                    21,
                    mut_opts,
                    &mut err,
                );
                if ok {
                    println!("  *** SUCCESS! ***");
                    list_dir(&lp);
                } else {
                    println!(
                        "  err: {}",
                        inner_error(&nserror_string(err).unwrap_or_default())
                    );
                }
            }
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
    e[..e.len().min(300)].to_string()
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

unsafe fn libc_free(p: *mut c_void) {
    extern "C" {
        fn free(p: *mut c_void);
    }
    free(p);
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
