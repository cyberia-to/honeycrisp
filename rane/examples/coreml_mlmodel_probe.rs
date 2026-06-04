//! Use public CoreML.framework MLModel to load VoiceActions bundle.
//! Goal: confirm _ANECoreMLModelCompiler is used, then introspect internal state.
//!
//! Run: cargo run -p rane --example coreml_mlmodel_probe --release

use rane::ffi::*;
use std::ffi::{c_void, CStr, CString};

extern "C" {
    fn free(p: *mut c_void);
}
fn libc_free(p: *mut c_void) {
    unsafe {
        free(p);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== CoreML MLModel probe ===\n");

    for name in &["AppleNeuralEngine", "ANECompiler", "ANEServices"] {
        let path = format!("/System/Library/PrivateFrameworks/{name}.framework/{name}");
        let c = CString::new(path).unwrap();
        unsafe {
            dlopen(c.as_ptr(), RTLD_NOW);
        }
    }

    // Load CoreML.framework
    let cml = CString::new("/System/Library/Frameworks/CoreML.framework/CoreML").unwrap();
    unsafe {
        dlopen(cml.as_ptr(), RTLD_NOW);
    }

    let ml_model_cls = cls("MLModel");
    if ml_model_cls.is_null() {
        println!("MLModel class not found");
        return Ok(());
    }
    println!("MLModel class: found");

    let bundle_path = "/System/Library/PrivateFrameworks/VoiceActions.framework/Versions/A/Resources/aa_encoder_125141826.mlmodelc";

    unsafe {
        type StrFn2 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type ModelFn3e =
            unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, *mut ObjcId) -> ObjcId;
        type DictFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;

        let sf: StrFn2 = std::mem::transmute(objc_msgSend as *const c_void);
        let mf: ModelFn3e = std::mem::transmute(objc_msgSend as *const c_void);
        let df: DictFn = std::mem::transmute(objc_msgSend as *const c_void);

        let cls_nsstr = cls("NSString");
        let cls_nsurl = cls("NSURL");
        let cls_dict = cls("NSDictionary");
        let cls_config = cls("MLModelConfiguration");

        let cs = CString::new(bundle_path).unwrap();
        let ns_str = sf(
            cls_nsstr as ObjcId,
            sel("stringWithUTF8String:"),
            cs.as_ptr() as ObjcId,
        );
        let ns_url = sf(cls_nsurl as ObjcId, sel("fileURLWithPath:"), ns_str);

        // Build MLModelConfiguration
        let config = if !cls_config.is_null() {
            let cfg = df(cls_config as ObjcId, sel("new"));
            // Try to set compute units to ANE only
            // MLComputeUnits: .all=0, .cpuOnly=1, .cpuAndGPU=2, .cpuAndNeuralEngine=3
            type SetIntFn = unsafe extern "C" fn(ObjcId, ObjcSel, i64);
            let sif: SetIntFn = std::mem::transmute(objc_msgSend as *const c_void);
            sif(cfg, sel("setComputeUnits:"), 3); // cpuAndNeuralEngine
            println!("MLModelConfiguration: cpuAndNeuralEngine");
            cfg
        } else {
            std::ptr::null_mut()
        };

        let mut err: ObjcId = std::ptr::null_mut();
        println!("Loading VoiceActions bundle via [MLModel modelWithContentsOfURL:configuration:error:]...");
        let ml_model = mf(
            ml_model_cls as ObjcId,
            sel("modelWithContentsOfURL:configuration:error:"),
            ns_url,
            config,
            &mut err,
        );

        if ml_model.is_null() {
            println!("MLModel = null: {:?}", nserror_string(err));
            return Ok(());
        }
        println!("MLModel loaded OK: {ml_model:p}");

        // Enumerate all ivars/methods on MLModel
        println!("\n=== MLModel class hierarchy + own methods ===");
        let ml_cls = object_getClass(ml_model);
        walk_and_print(ml_cls, ml_model);

        // Look for _ANEInMemoryModel inside
        println!("\n=== Searching for _ANEInMemoryModel inside MLModel ===");
        find_ane_model(ml_model);
    }

    Ok(())
}

unsafe fn walk_and_print(cls: ObjcClass, obj: ObjcId) {
    let mut c = cls;
    while !c.is_null() {
        let cname = {
            let p = class_getName(c);
            if p.is_null() {
                "?".to_string()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };

        // Own ivars
        let mut icount: u32 = 0;
        let ilist = class_copyIvarList(c, &mut icount);
        if !ilist.is_null() && icount > 0 {
            println!("  [{cname}] ivars:");
            for i in 0..icount {
                let ivar = *ilist.add(i as usize);
                let np = ivar_getName(ivar);
                let tp = ivar_getTypeEncoding(ivar);
                let off = ivar_getOffset(ivar);
                let name = if np.is_null() {
                    "?".into()
                } else {
                    CStr::from_ptr(np).to_string_lossy().into_owned()
                };
                let ty = if tp.is_null() {
                    "?".into()
                } else {
                    CStr::from_ptr(tp).to_string_lossy().into_owned()
                };
                // Read ivar value
                let val = object_getIvar(obj, ivar);
                print!("    +{off:04x} {name}: {ty} = {val:p}");
                if ty.starts_with('@') && !val.is_null() {
                    let vcls = object_getClass(val);
                    if !vcls.is_null() {
                        let vp = class_getName(vcls);
                        if !vp.is_null() {
                            let vn = CStr::from_ptr(vp).to_string_lossy();
                            print!("  [{vn}]");
                        }
                    }
                }
                println!();
            }
            libc_free(ilist as *mut c_void);
        }

        // Own methods (filtered)
        let mut mcount: u32 = 0;
        let mlist = class_copyMethodList(c, &mut mcount);
        if !mlist.is_null() && mcount > 0 {
            let mut names: Vec<String> = Vec::new();
            for i in 0..mcount {
                let method = *mlist.add(i as usize);
                let s = method_getName(method);
                if !s.is_null() {
                    let np = sel_getName(s);
                    if !np.is_null() {
                        names.push(CStr::from_ptr(np).to_string_lossy().into_owned());
                    }
                }
            }
            libc_free(mlist as *mut c_void);
            names.sort();
            // Print ANE-related methods
            let interesting: Vec<_> = names
                .iter()
                .filter(|n| {
                    n.contains("ANE")
                        || n.contains("ane")
                        || n.contains("neural")
                        || n.contains("Neural")
                        || n.contains("compile")
                        || n.contains("Compile")
                        || n.contains("program")
                        || n.contains("Program")
                        || n.contains("backend")
                        || n.contains("Backend")
                        || n.contains("engine")
                        || n.contains("Engine")
                })
                .cloned()
                .collect();
            if !interesting.is_empty() {
                println!("  [{cname}] interesting methods:");
                for n in &interesting {
                    println!("    {n}");
                }
            }
        }

        c = class_getSuperclass(c);
    }
}

unsafe fn find_ane_model(obj: ObjcId) {
    // BFS through object graph looking for _ANEInMemoryModel
    let mut queue: Vec<(ObjcId, String)> = vec![(obj, "root".to_string())];
    let mut visited: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut depth = 0;

    while !queue.is_empty() && depth < 4 {
        let next_queue: Vec<(ObjcId, String)> = queue
            .drain(..)
            .flat_map(|(o, path)| {
                if o.is_null() || !visited.insert(o as usize) {
                    return vec![];
                }

                let cls = object_getClass(o);
                if cls.is_null() {
                    return vec![];
                }
                let cname = {
                    let p = class_getName(cls);
                    if p.is_null() {
                        return vec![];
                    }
                    CStr::from_ptr(p).to_string_lossy().into_owned()
                };

                if cname.contains("ANEInMemory") {
                    println!("  FOUND: {path} → {cname} @ {o:p}");
                    // Print its ivars
                    let mut ic: u32 = 0;
                    let il = class_copyIvarList(cls, &mut ic);
                    if !il.is_null() {
                        for i in 0..ic {
                            let iv = *il.add(i as usize);
                            let np = ivar_getName(iv);
                            let tp = ivar_getTypeEncoding(iv);
                            let off = ivar_getOffset(iv);
                            let name = if np.is_null() {
                                "?".into()
                            } else {
                                CStr::from_ptr(np).to_string_lossy().into_owned()
                            };
                            let ty = if tp.is_null() {
                                "?".into()
                            } else {
                                CStr::from_ptr(tp).to_string_lossy().into_owned()
                            };
                            let val = object_getIvar(o, iv);
                            println!("    +{off:04x} {name}: {ty} = {val:p}");
                        }
                        libc_free(il as *mut c_void);
                    }
                    return vec![];
                }

                // Recurse into ObjC object ivars
                let mut children = vec![];
                let mut c = cls;
                while !c.is_null() {
                    let mut ic: u32 = 0;
                    let il = class_copyIvarList(c, &mut ic);
                    if !il.is_null() {
                        for i in 0..ic {
                            let iv = *il.add(i as usize);
                            let tp = ivar_getTypeEncoding(iv);
                            let np = ivar_getName(iv);
                            if !tp.is_null() {
                                let ty = CStr::from_ptr(tp).to_string_lossy();
                                if ty.starts_with('@') {
                                    let child = object_getIvar(o, iv);
                                    if !child.is_null() {
                                        let iname = if np.is_null() {
                                            "?".into()
                                        } else {
                                            CStr::from_ptr(np).to_string_lossy().into_owned()
                                        };
                                        children.push((child, format!("{path}.{iname}")));
                                    }
                                }
                            }
                        }
                        libc_free(il as *mut c_void);
                    }
                    c = class_getSuperclass(c);
                }
                children
            })
            .collect();

        queue = next_queue;
        depth += 1;
    }
}
