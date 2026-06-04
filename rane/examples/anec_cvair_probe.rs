//! Load ANECompilerService.xpc, use dlsym(handle) for kANEF* constants,
//! enumerate _ANECVAIRCompiler methods (with superclass walk),
//! and try kANEFNetPlistFilenameKey with real value.
//!
//! Run: cargo run -p rane --example anec_cvair_probe --release

use rane::ffi::*;
use std::ffi::{c_void, CStr, CString};

const ESPRESSO_BUNDLE: &str = "/System/Library/DuetExpertCenter/Assets/Assets.bundle/AssetData/ATXActionValuationMLModel.mlmodelc";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== _ANECVAIRCompiler + kANEFNetPlistFilenameKey probe ===\n");

    for name in &["AppleNeuralEngine", "ANECompiler", "ANEServices"] {
        let path = format!("/System/Library/PrivateFrameworks/{name}.framework/{name}");
        let c = CString::new(path).unwrap();
        unsafe {
            dlopen(c.as_ptr(), RTLD_NOW);
        }
    }

    let xpc_path = CString::new("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/XPCServices/ANECompilerService.xpc/Contents/MacOS/ANECompilerService").unwrap();
    let xpc_handle = unsafe { dlopen(xpc_path.as_ptr(), RTLD_NOW | 0x8) }; // 0x8 = RTLD_GLOBAL on macOS
    if xpc_handle.is_null() {
        println!("dlopen XPC service failed");
    } else {
        println!("dlopen XPC service: OK @ {xpc_handle:p}");
    }

    unsafe {
        type DataFn = unsafe extern "C" fn(ObjcId, ObjcSel, *const u8, u64) -> ObjcId;
        type DictFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type DescFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, ObjcId) -> ObjcId;
        type ModelFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type StrFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type ModelFn2 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, u8) -> ObjcId;
        type CompileFn = unsafe extern "C" fn(ObjcId, ObjcSel, u32, ObjcId, *mut ObjcId) -> bool;
        type StrFn2 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type SetKVFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId);
        type SetDictFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId);

        let dictf: DictFn = std::mem::transmute(objc_msgSend as *const c_void);
        let strf2: StrFn2 = std::mem::transmute(objc_msgSend as *const c_void);
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

        // --- 1. Read kANEF* constants via dlsym(handle) ---
        println!("=== 1. kANEF* constants via dlsym(xpc_handle) ===");
        let const_names_c = [
            "kANEFModelANECIRValue",
            "kANEFModelCoreMLValue",
            "kANEFModelMILValue",
            "kANEFModelMLIRValue",
            "kANEFModelLLIRBundleValue",
            "kANEFModelPreCompiledValue",
            "kANEFIsInMemoryModelTypeKey",
            "kANEFInMemoryModelIsCachedKey",
            "kANEFEspressoFileResourcesKey",
            "kANEFNetPlistFilenameKey",
            "kANEFCompilerOptionsFilenameKey",
            "kANEFModelType",
            "kANEFModelIdentityStrKey",
            "kANEFBaseModelIdentifierKey",
            "kANEFModelDescriptionKey",
            "kANEFModelIsEncryptedKey",
            "kANEFModelHasCacheURLIdentifierKey",
            "kANEFModelCacheIdentifierUsingSourceURLKey",
            "kANEFRetainModelsWithoutSourceURLKey",
            "kANEFCompilationInitiatedByE5MLKey",
        ];
        let mut kanef: std::collections::HashMap<String, ObjcId> = Default::default();
        for name in &const_names_c {
            // Try NULL (global), then xpc_handle
            let sym_name = CString::new(*name).unwrap();
            let mut p = dlsym(std::ptr::null_mut(), sym_name.as_ptr()) as *const ObjcId;
            if p.is_null() && !xpc_handle.is_null() {
                p = dlsym(xpc_handle, sym_name.as_ptr()) as *const ObjcId;
            }
            if !p.is_null() && !(*p).is_null() {
                let val = *p;
                kanef.insert(name.to_string(), val);
                println!("  {name} = \"{}\"", nsstring_to_str(val));
            } else {
                println!("  {name} = NOT FOUND");
            }
        }

        // --- 2. Enumerate _ANECVAIRCompiler with superclass walk ---
        println!("\n=== 2. _ANECVAIRCompiler full method walk ===");
        let cvair_cls = cls("_ANECVAIRCompiler");
        if cvair_cls.is_null() {
            println!("  NOT FOUND");
        } else {
            let mut c = cvair_cls;
            while !c.is_null() {
                let cname = {
                    let p = class_getName(c);
                    if p.is_null() {
                        "?".to_string()
                    } else {
                        CStr::from_ptr(p).to_string_lossy().into_owned()
                    }
                };
                let mut count: u32 = 0;
                let methods = class_copyMethodList(c, &mut count);
                if !methods.is_null() && count > 0 {
                    println!("  [{cname}] ({count} methods):");
                    for i in 0..count as usize {
                        let m = *methods.add(i);
                        let s = method_getName(m);
                        let enc = method_getTypeEncoding(m);
                        let name = if s.is_null() {
                            "?".to_string()
                        } else {
                            let n = sel_getName(s);
                            if n.is_null() {
                                "?".to_string()
                            } else {
                                CStr::from_ptr(n).to_string_lossy().into_owned()
                            }
                        };
                        let enc_s = if enc.is_null() {
                            "?"
                        } else {
                            CStr::from_ptr(enc).to_str().unwrap_or("?")
                        };
                        println!("    {} [{}]", name, enc_s);
                    }
                    libc_free(methods as *mut c_void);
                } else {
                    println!("  [{cname}] 0 own methods");
                }
                c = class_getSuperclass(c);
            }
        }

        // --- 3. Enumerate _ANEEspressoIRTranslator ---
        println!("\n=== 3. _ANEEspressoIRTranslator full method walk ===");
        let esp_cls = cls("_ANEEspressoIRTranslator");
        if esp_cls.is_null() {
            println!("  NOT FOUND");
        } else {
            let mut c = esp_cls;
            let mut depth = 0;
            while !c.is_null() && depth < 3 {
                let cname = {
                    let p = class_getName(c);
                    if p.is_null() {
                        "?".to_string()
                    } else {
                        CStr::from_ptr(p).to_string_lossy().into_owned()
                    }
                };
                let mut count: u32 = 0;
                let methods = class_copyMethodList(c, &mut count);
                if !methods.is_null() && count > 0 {
                    println!("  [{cname}] ({count} methods):");
                    for i in 0..count as usize {
                        let m = *methods.add(i);
                        let s = method_getName(m);
                        let enc = method_getTypeEncoding(m);
                        let name = if s.is_null() {
                            "?".to_string()
                        } else {
                            let n = sel_getName(s);
                            if n.is_null() {
                                "?".to_string()
                            } else {
                                CStr::from_ptr(n).to_string_lossy().into_owned()
                            }
                        };
                        let enc_s = if enc.is_null() {
                            "?"
                        } else {
                            CStr::from_ptr(enc).to_str().unwrap_or("?")
                        };
                        println!("    {} [{}]", name, enc_s);
                    }
                    libc_free(methods as *mut c_void);
                }
                c = class_getSuperclass(c);
                depth += 1;
            }
        }

        // --- 4. Compile with kANEFNetPlistFilenameKey using real constant value ---
        println!("\n=== 4. Compile with kANEFNetPlistFilenameKey ===");
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
        *p = 0;
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
        println!("  localModelPath: {}", &lp[..lp.len().min(100)]);

        // Write espresso files
        std::fs::create_dir_all(&lp)?;
        for fname in &[
            "model.espresso.net",
            "model.espresso.shape",
            "model.espresso.weights",
        ] {
            if let Ok(content) = std::fs::read(format!("{ESPRESSO_BUNDLE}/{fname}")) {
                std::fs::write(format!("{lp}/{fname}"), &content)?;
                println!("  wrote {fname}: {}B", content.len());
            }
        }

        let base_opts = modelf2(
            model,
            sel("compilerOptionsWithOptions:isCompiledModelCached:"),
            empty,
            0,
        );
        let mut_opts = dictf(cls_mdict as ObjcId, sel("new"));
        setd(mut_opts, sel("setDictionary:"), base_opts);

        // Use real constant if found, otherwise try known string values
        let net_plist_key = kanef.get("kANEFNetPlistFilenameKey").copied().unwrap_or_else(|| {
            // Try known string values from binary analysis
            println!("  kANEFNetPlistFilenameKey not found via dlsym, using raw string \"kANEFNetPlistFilenameKey\"");
            make_nsstr("kANEFNetPlistFilenameKey")
        });
        let net_plist_key_str = nsstring_to_str(net_plist_key);
        println!("  using key: \"{net_plist_key_str}\"");
        setkv(
            mut_opts,
            sel("setObject:forKey:"),
            make_nsstr("model.espresso.net"),
            net_plist_key,
        );

        // Also inject kANEFCompilerOptionsFilenameKey
        if let Some(&co_key) = kanef.get("kANEFCompilerOptionsFilenameKey") {
            println!(
                "  kANEFCompilerOptionsFilenameKey = \"{}\"",
                nsstring_to_str(co_key)
            );
        }

        let desc2 = descf(
            cls_desc as ObjcId,
            sel("modelWithMILText:weights:optionsPlist:"),
            ns_text,
            empty,
            std::ptr::null_mut(),
        );
        let pp2 = (desc2 as *mut u8).add(8);
        *pp2 = 0;
        let model2 = modelf(
            cls_model as ObjcId,
            sel("inMemoryModelWithDescriptor:"),
            desc2,
        );
        if !model2.is_null() {
            let mut err: ObjcId = std::ptr::null_mut();
            let ok = compilef(
                model2,
                sel("compileWithQoS:options:error:"),
                21,
                mut_opts,
                &mut err,
            );
            if ok {
                println!("  *** COMPILE SUCCESS! ***");
                list_dir(&lp);
            } else {
                println!(
                    "  err: {}",
                    &nserror_string(err).unwrap_or_default()
                        [..300.min(nserror_string(err).unwrap_or_default().len())]
                );
            }
        }

        // --- 5. Try different known string values as the key ---
        println!("\n=== 5. Try known string key values ===");
        let candidate_keys = [
            "NetPlistFilename",
            "NetworkPlistFilename",
            "ModelPlistFilename",
            "NetworkSourcePlistFilename",
            "NetPlistFileURL",
            "kNetPlistFilename",
            "NetPlistPath",
            "EspressoFilePath",
        ];
        for key_val in &candidate_keys {
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
                continue;
            }

            let opts_t = dictf(cls_mdict as ObjcId, sel("new"));
            setd(opts_t, sel("setDictionary:"), base_opts);
            let ns_k = make_nsstr(key_val);
            let ns_v = make_nsstr("model.espresso.net");
            setkv(opts_t, sel("setObject:forKey:"), ns_v, ns_k);

            let mut e: ObjcId = std::ptr::null_mut();
            let ok = compilef(m, sel("compileWithQoS:options:error:"), 21, opts_t, &mut e);
            let result = if ok {
                "SUCCESS".to_string()
            } else {
                inner_error(&nserror_string(e).unwrap_or_default())
            };
            println!("  [{key_val}]: {result}");
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
    e[..e.len().min(200)].to_string()
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
