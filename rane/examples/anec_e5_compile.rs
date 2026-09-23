//! Try compiling VAD_ANE.e5 model via kANEFModelANECIR + kANEFNetPlistFilenameKey.
//! This model has 285 layers with compute_path=1 (ANE ops).
//!
//! Run: cargo run -p rane --example anec_e5_compile --release

use rane::ffi::*;
use std::ffi::{c_void, CStr, CString};

const VAD: &str = "/System/Library/PrivateFrameworks/VoiceActions.framework/Versions/A/Resources/VAD_ANE.e5/model.bundle/universal.bundle/main/main_classic_cpu";
const ANE_AX: &str = "/System/Library/PrivateFrameworks/VoiceActions.framework/Versions/A/Resources/ANE_AX_FP16_20240430.e5/model.bundle/universal.bundle/main/main_classic_cpu";
const AVS: &str = "/System/Library/PrivateFrameworks/VoiceActions.framework/Versions/A/Resources/AVS_embedding_ztcurt8my3_40.e5/model.bundle/universal.bundle/main/main_classic_cpu";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== E5 bundle ANE compile attempt ===\n");

    for name in &["AppleNeuralEngine", "ANECompiler", "ANEServices"] {
        let path = format!("/System/Library/PrivateFrameworks/{name}.framework/{name}");
        let c = CString::new(path).unwrap();
        unsafe {
            dlopen(c.as_ptr(), RTLD_NOW);
        }
    }
    let xpc_path = CString::new("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/XPCServices/ANECompilerService.xpc/Contents/MacOS/ANECompilerService").unwrap();
    unsafe {
        dlopen(xpc_path.as_ptr(), RTLD_NOW | 0x8);
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

        let cls_data = cls("NSData");
        let cls_dict = cls("NSDictionary");
        let cls_mdict = cls("NSMutableDictionary");
        let cls_desc = cls("_ANEInMemoryModelDescriptor");
        let cls_model = cls("_ANEInMemoryModel");
        let cls_str = cls("NSString");

        let make_nsstr = |s: &str| -> ObjcId {
            let c = CString::new(s).unwrap();
            strf2(
                cls_str as ObjcId,
                sel("stringWithUTF8String:"),
                c.as_ptr() as ObjcId,
            )
        };

        // Read kANEFNetPlistFilenameKey constant
        let kanef_net_plist = read_const("kANEFNetPlistFilenameKey");
        println!(
            "kANEFNetPlistFilenameKey = \"{}\"",
            nsstring_to_str(kanef_net_plist)
        );

        let mil_text = "program(1, 0)\nfunc main<ios16>(tensor<fp16, [1,16,1,1]> x) -> (tensor<fp16, [1,16,1,1]>) {\n  block0() {\n    tensor<fp16, [1,16,1,1]> y = relu()[x = x];\n  } -> (y)\n}\n";
        let ns_text = dataf(
            cls_data as ObjcId,
            sel("dataWithBytes:length:"),
            mil_text.as_bytes().as_ptr(),
            mil_text.len() as u64,
        );
        let empty = dictf(cls_dict as ObjcId, sel("dictionary"));

        for (tag, bundle_path) in &[
            ("VAD_ANE", VAD),
            ("ANE_AX_FP16", ANE_AX),
            ("AVS_embedding", AVS),
        ] {
            println!("\n=== Trying {tag} ===");

            // Create flipped model + get localModelPath
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
                continue;
            }

            let lp = nsstring_to_str(strf(model, sel("localModelPath")));
            println!("  localModelPath: {}", &lp[..80.min(lp.len())]);

            // Write espresso files
            std::fs::create_dir_all(&lp)?;
            for fname in &[
                "model.espresso.net",
                "model.espresso.shape",
                "model.espresso.weights",
            ] {
                let src = format!("{bundle_path}/{fname}");
                if let Ok(content) = std::fs::read(&src) {
                    std::fs::write(format!("{lp}/{fname}"), &content)?;
                    println!("  wrote {fname}: {}B", content.len());
                } else {
                    println!("  MISSING: {fname}");
                }
            }

            let base_opts = modelf2(
                model,
                sel("compilerOptionsWithOptions:isCompiledModelCached:"),
                empty,
                0,
            );

            // Build opts with kANEFNetPlistFilenameKey
            let opts = dictf(cls_mdict as ObjcId, sel("new"));
            setd(opts, sel("setDictionary:"), base_opts);
            setkv(
                opts,
                sel("setObject:forKey:"),
                make_nsstr("model.espresso.net"),
                kanef_net_plist,
            );

            // Fresh model for compile
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
            if model2.is_null() {
                println!("  model2=null");
                continue;
            }

            let mut err: ObjcId = std::ptr::null_mut();
            let ok = compilef(
                model2,
                sel("compileWithQoS:options:error:"),
                21,
                opts,
                &mut err,
            );
            if ok {
                println!("  *** COMPILE SUCCESS! ***");
                list_dir(&lp);
            } else {
                let e = nserror_string(err).unwrap_or_default();
                println!("  err: {}", inner_error_full(&e));
            }

            std::fs::remove_dir_all(&lp).ok();
        }

        // --- Also try: use the model.espresso.net content as the _networkText in descriptor ---
        println!("\n=== Try: _networkText = espresso.net content as NSData ===");
        let vad_net = std::fs::read(format!("{VAD}/model.espresso.net"))?;
        let ns_vad_net = dataf(
            cls_data as ObjcId,
            sel("dataWithBytes:length:"),
            vad_net.as_ptr(),
            vad_net.len() as u64,
        );

        let desc3 = descf(
            cls_desc as ObjcId,
            sel("modelWithMILText:weights:optionsPlist:"),
            ns_vad_net,
            empty,
            std::ptr::null_mut(),
        );
        // Don't flip — this is MIL path with espresso content as "MIL text"
        let p3 = (desc3 as *mut u8).add(8);
        *p3 = 0; // flip
        let model3 = modelf(
            cls_model as ObjcId,
            sel("inMemoryModelWithDescriptor:"),
            desc3,
        );
        if !model3.is_null() {
            let lp3 = nsstring_to_str(strf(model3, sel("localModelPath")));
            println!("  localModelPath: {}", &lp3[..80.min(lp3.len())]);

            // Write espresso files at this path
            std::fs::create_dir_all(&lp3)?;
            for fname in &[
                "model.espresso.net",
                "model.espresso.shape",
                "model.espresso.weights",
            ] {
                if let Ok(content) = std::fs::read(format!("{VAD}/{fname}")) {
                    std::fs::write(format!("{lp3}/{fname}"), &content)?;
                    println!("  wrote {fname}: {}B", content.len());
                }
            }

            let base_opts3 = modelf2(
                model3,
                sel("compilerOptionsWithOptions:isCompiledModelCached:"),
                empty,
                0,
            );
            println!("  opts: {}", objc_desc(base_opts3));

            // Try compile with and without kANEFNetPlistFilenameKey
            let opts3a = dictf(cls_mdict as ObjcId, sel("new"));
            setd(opts3a, sel("setDictionary:"), base_opts3);
            setkv(
                opts3a,
                sel("setObject:forKey:"),
                make_nsstr("model.espresso.net"),
                kanef_net_plist,
            );

            let d3a = descf(
                cls_desc as ObjcId,
                sel("modelWithMILText:weights:optionsPlist:"),
                ns_vad_net,
                empty,
                std::ptr::null_mut(),
            );
            let pp3a = (d3a as *mut u8).add(8);
            *pp3a = 0;
            let m3a = modelf(
                cls_model as ObjcId,
                sel("inMemoryModelWithDescriptor:"),
                d3a,
            );
            if !m3a.is_null() {
                let mut e3a: ObjcId = std::ptr::null_mut();
                let ok = compilef(
                    m3a,
                    sel("compileWithQoS:options:error:"),
                    21,
                    opts3a,
                    &mut e3a,
                );
                if ok {
                    println!("  *** SUCCESS (vad as networkText + NetPlist key)! ***");
                    list_dir(&lp3);
                } else {
                    println!(
                        "  err: {}",
                        inner_error_full(&nserror_string(e3a).unwrap_or_default())
                    );
                }
            }
            std::fs::remove_dir_all(&lp3).ok();
        }
    }

    Ok(())
}

unsafe fn read_const(name: &str) -> ObjcId {
    let c = CString::new(name).unwrap();
    let p = dlsym(std::ptr::null_mut(), c.as_ptr()) as *const ObjcId;
    if !p.is_null() && !(*p).is_null() {
        return *p;
    }
    let xpc = CString::new("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/XPCServices/ANECompilerService.xpc/Contents/MacOS/ANECompilerService").unwrap();
    let h = dlopen(xpc.as_ptr(), 0x1 | 0x8);
    if !h.is_null() {
        let p2 = dlsym(h, c.as_ptr()) as *const ObjcId;
        if !p2.is_null() && !(*p2).is_null() {
            return *p2;
        }
    }
    std::ptr::null_mut()
}

fn inner_error_full(e: &str) -> String {
    if let Some(idx) = e.rfind("err=(\n    ") {
        let rest = &e[idx + 9..];
        if let Some(end) = rest.find('\n') {
            return format!("err=[{}]", rest[..end].trim());
        }
    }
    e[..e.len().min(400)].to_string()
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
        for entry in entries.flatten() {
            let meta = std::fs::metadata(entry.path()).ok();
            let size = meta.map(|m| m.len()).unwrap_or(0);
            println!("    {}: {size}B", entry.file_name().to_string_lossy());
        }
    }
}
