//! Two-step ANE compile via Espresso→ANEC IR translation then compile:
//! 1. _ANEEspressoIRTranslator::translateModelAt:key:outputPath:... → net.plist (ANEC IR)
//! 2. _ANEInMemoryModel (isMIL=false) + kANEFNetPlistFilenameKey="net.plist"
//!
//! Run: cargo run -p rane --example anec_e5_translate --release

use rane::ffi::*;
use std::ffi::{c_void, CStr, CString};

const VAD_NET: &str = "/System/Library/PrivateFrameworks/VoiceActions.framework/Versions/A/Resources/VAD_ANE.e5/model.bundle/universal.bundle/main/main_classic_cpu/model.espresso.net";
const VAD_DIR: &str = "/System/Library/PrivateFrameworks/VoiceActions.framework/Versions/A/Resources/VAD_ANE.e5/model.bundle/universal.bundle/main/main_classic_cpu";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== E5 → ANEC IR → ANE compile ===\n");

    // Load frameworks
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
        type StrFn2 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type DataFn = unsafe extern "C" fn(ObjcId, ObjcSel, *const u8, u64) -> ObjcId;
        type DictFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type DescFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, ObjcId) -> ObjcId;
        type ModelFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type ModelFn2 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, u8) -> ObjcId;
        type CompileFn = unsafe extern "C" fn(ObjcId, ObjcSel, u32, ObjcId, *mut ObjcId) -> bool;
        type SetKVFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId);
        type SetDictFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId);
        type StrFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type TranslateFn = unsafe extern "C" fn(
            ObjcId,
            ObjcSel,
            ObjcId,
            ObjcId,
            ObjcId,
            u8,
            ObjcId,
            *mut ObjcId,
        ) -> u8;

        let strf2: StrFn2 = std::mem::transmute(objc_msgSend as *const c_void);
        let dataf: DataFn = std::mem::transmute(objc_msgSend as *const c_void);
        let dictf: DictFn = std::mem::transmute(objc_msgSend as *const c_void);
        let descf: DescFn = std::mem::transmute(objc_msgSend as *const c_void);
        let modelf: ModelFn = std::mem::transmute(objc_msgSend as *const c_void);
        let modelf2: ModelFn2 = std::mem::transmute(objc_msgSend as *const c_void);
        let compilef: CompileFn = std::mem::transmute(objc_msgSend as *const c_void);
        let setkv: SetKVFn = std::mem::transmute(objc_msgSend as *const c_void);
        let setd: SetDictFn = std::mem::transmute(objc_msgSend as *const c_void);
        let strf: StrFn = std::mem::transmute(objc_msgSend as *const c_void);
        let translatef: TranslateFn = std::mem::transmute(objc_msgSend as *const c_void);

        let cls_data = cls("NSData");
        let cls_dict = cls("NSDictionary");
        let cls_mdict = cls("NSMutableDictionary");
        let cls_desc = cls("_ANEInMemoryModelDescriptor");
        let cls_model = cls("_ANEInMemoryModel");
        let cls_str = cls("NSString");
        let cls_trans = cls("_ANEEspressoIRTranslator");

        let make_nsstr = |s: &str| -> ObjcId {
            let c = CString::new(s).unwrap();
            strf2(
                cls_str as ObjcId,
                sel("stringWithUTF8String:"),
                c.as_ptr() as ObjcId,
            )
        };

        // Read kANEFNetPlistFilenameKey
        let kanef_net_plist = read_const("kANEFNetPlistFilenameKey");
        println!(
            "kANEFNetPlistFilenameKey = \"{}\"",
            nsstring_to_str(kanef_net_plist)
        );

        // Step 1: Translate Espresso → ANEC IR
        println!("\n--- Step 1: Espresso → ANEC IR translation ---");

        // Create output dir
        let trans_out = "/tmp/ane_anecir_out";
        std::fs::create_dir_all(trans_out)?;
        let ns_trans_out = make_nsstr(trans_out);
        let ns_net_path = make_nsstr(VAD_NET);
        let empty_dict = dictf(cls_dict as ObjcId, sel("dictionary"));

        let mut trans_err: ObjcId = std::ptr::null_mut();
        let ok_trans = translatef(
            cls_trans as ObjcId,
            sel("translateModelAt:key:outputPath:isEncryptedModel:translationOptions:error:"),
            ns_net_path,          // modelAt = path to .net FILE
            std::ptr::null_mut(), // key = nil
            ns_trans_out,         // outputPath = dir to write ANEC IR
            0,                    // isEncryptedModel = NO
            empty_dict,           // translationOptions = {}
            &mut trans_err,
        );

        if ok_trans == 0 {
            let e = nserror_string(trans_err).unwrap_or_default();
            println!("  translate FAILED: {e}");
            return Err("translation failed".into());
        }

        println!("  translate OK!");
        let files: Vec<_> = std::fs::read_dir(trans_out)?
            .filter_map(|e| e.ok())
            .collect();
        for f in &files {
            let meta = std::fs::metadata(f.path()).ok();
            let sz = meta.map(|m| m.len()).unwrap_or(0);
            println!("  {}: {}B", f.file_name().to_string_lossy(), sz);
        }

        // Step 2: Compile ANEC IR via _ANEInMemoryModel
        println!("\n--- Step 2: Compile ANEC IR ---");

        type ObjFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        let objf: ObjFn = std::mem::transmute(objc_msgSend as *const c_void);

        let mil_text = "program(1, 0)\nfunc main<ios16>(tensor<fp16, [1,16,1,1]> x) -> (tensor<fp16, [1,16,1,1]>) {\n  block0() {\n    tensor<fp16, [1,16,1,1]> y = relu()[x = x];\n  } -> (y)\n}\n";
        let ns_mil = dataf(
            cls_data as ObjcId,
            sel("dataWithBytes:length:"),
            mil_text.as_bytes().as_ptr(),
            mil_text.len() as u64,
        );

        // Create model to get localModelPath and base options
        let desc_probe = descf(
            cls_desc as ObjcId,
            sel("modelWithMILText:weights:optionsPlist:"),
            ns_mil,
            empty_dict,
            std::ptr::null_mut(),
        );
        let pp_probe = (desc_probe as *mut u8).add(8);
        *pp_probe = 0;
        let model_probe = modelf(
            cls_model as ObjcId,
            sel("inMemoryModelWithDescriptor:"),
            desc_probe,
        );
        if model_probe.is_null() {
            return Err("probe model=null".into());
        }

        // localModelPath is the SHORT path (first ~31 chars of SHA256)
        // ANECCompile uses the full triple-hash path, but net plist lookup may use SHORT path
        let lp_short = nsstring_to_str(strf(model_probe, sel("localModelPath")));
        let base_opts = modelf2(
            model_probe,
            sel("compilerOptionsWithOptions:isCompiledModelCached:"),
            empty_dict,
            0,
        );

        // Extract kANEFIsInMemoryModelTypeKey value to know full compile dir
        let k_type_key = make_nsstr("kANEFIsInMemoryModelTypeKey");
        let full_hash_ns = objf(base_opts, sel("objectForKey:"), k_type_key);
        let full_hash = nsstring_to_str(full_hash_ns);
        let parent = std::path::Path::new(&lp_short)
            .parent()
            .unwrap_or(std::path::Path::new("/tmp"))
            .to_str()
            .unwrap_or("/tmp");
        let lp_full = format!("{parent}/{full_hash}");

        println!("  localModelPath (short): {lp_short}");
        println!(
            "  compile dir (full):     {}...",
            &lp_full[..80.min(lp_full.len())]
        );

        // Write ANEC IR files to BOTH the short path AND the full path,
        // since we don't know which one ANECCompile reads the net plist from.
        for lp in &[&lp_short, &lp_full] {
            std::fs::create_dir_all(lp)?;
            for entry in std::fs::read_dir(trans_out)?.filter_map(|e| e.ok()) {
                let name = entry.file_name();
                let content = std::fs::read(entry.path())?;
                std::fs::write(format!("{}/{}", lp, name.to_string_lossy()), &content)?;
            }
            for fname in &["model.espresso.shape", "model.espresso.weights"] {
                let src = format!("{VAD_DIR}/{fname}");
                if let Ok(content) = std::fs::read(&src) {
                    std::fs::write(format!("{lp}/{fname}"), &content)?;
                }
            }
            println!("  wrote files to: {}...", &lp[..70.min(lp.len())]);
        }

        // Build options: inject kANEFNetPlistFilenameKey="net.plist"
        let opts = dictf(cls_mdict as ObjcId, sel("new"));
        setd(opts, sel("setDictionary:"), base_opts);
        setkv(
            opts,
            sel("setObject:forKey:"),
            make_nsstr("net.plist"),
            kanef_net_plist,
        );

        // Fresh model for actual compile
        let desc2 = descf(
            cls_desc as ObjcId,
            sel("modelWithMILText:weights:optionsPlist:"),
            ns_mil,
            empty_dict,
            std::ptr::null_mut(),
        );
        let pp = (desc2 as *mut u8).add(8);
        *pp = 0;
        let model2 = modelf(
            cls_model as ObjcId,
            sel("inMemoryModelWithDescriptor:"),
            desc2,
        );
        if model2.is_null() {
            println!("  model2=null");
            return Err("model2 creation failed".into());
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
            println!("\n*** COMPILE SUCCESS! ***");
            list_dir(&lp_full);
        } else {
            let e = nserror_string(err).unwrap_or_default();
            println!("  compile err: {}", &e[..e.len().min(600)]);
        }

        std::fs::remove_dir_all(&lp_short).ok();
        std::fs::remove_dir_all(&lp_full).ok();
        std::fs::remove_dir_all(trans_out).ok();
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
