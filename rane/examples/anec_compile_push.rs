//! Push past InvalidCompilationParam by adding more required keys.
//! Known: kANEFNetPlistFilenameKey injection gets past InvalidNetworkSourceFileName.
//! Now: add kANEFCompilerOptionsFilenameKey and kANEFEspressoFileResourcesKey.
//! Also try NearbyInteraction model (simpler).
//!
//! Run: cargo run -p rane --example anec_compile_push --release

use rane::ffi::*;
use std::ffi::{c_void, CStr, CString};

const DUET: &str = "/System/Library/DuetExpertCenter/Assets/Assets.bundle/AssetData/ATXActionValuationMLModel.mlmodelc";
const NEARBY: &str = "/System/Library/NearbyInteractionBundles/MotionBasedSpatialGesturesResources.bundle/Contents/Resources";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Push past InvalidCompilationParam ===\n");

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

        // Read key constants
        let kanef_net_plist = read_const("kANEFNetPlistFilenameKey");
        let kanef_opts_file = read_const("kANEFCompilerOptionsFilenameKey");
        let kanef_espresso_resources = read_const("kANEFEspressoFileResourcesKey");
        let kanef_espresso_opt = read_const("kANEFEnablePowerSavingKey");
        println!(
            "kANEFNetPlistFilenameKey = \"{}\"",
            nsstring_to_str(kanef_net_plist)
        );
        println!(
            "kANEFCompilerOptionsFilenameKey = \"{}\"",
            nsstring_to_str(kanef_opts_file)
        );
        println!(
            "kANEFEspressoFileResourcesKey = \"{}\"",
            nsstring_to_str(kanef_espresso_resources)
        );

        let mil_text = "program(1, 0)\nfunc main<ios16>(tensor<fp16, [1,16,1,1]> x) -> (tensor<fp16, [1,16,1,1]>) {\n  block0() {\n    tensor<fp16, [1,16,1,1]> y = relu()[x = x];\n  } -> (y)\n}\n";
        let ns_text = dataf(
            cls_data as ObjcId,
            sel("dataWithBytes:length:"),
            mil_text.as_bytes().as_ptr(),
            mil_text.len() as u64,
        );
        let empty = dictf(cls_dict as ObjcId, sel("dictionary"));

        // Setup model + localModelPath
        let desc0 = descf(
            cls_desc as ObjcId,
            sel("modelWithMILText:weights:optionsPlist:"),
            ns_text,
            empty,
            std::ptr::null_mut(),
        );
        let pp0 = (desc0 as *mut u8).add(8);
        *pp0 = 0;
        let model0 = modelf(
            cls_model as ObjcId,
            sel("inMemoryModelWithDescriptor:"),
            desc0,
        );
        if model0.is_null() {
            println!("model=null");
            return Ok(());
        }
        let lp = nsstring_to_str(strf(model0, sel("localModelPath")));
        println!("localModelPath: {}", &lp[..lp.len().min(80)]);

        let base_opts = modelf2(
            model0,
            sel("compilerOptionsWithOptions:isCompiledModelCached:"),
            empty,
            0,
        );

        // Helper: create model + compile with given opts
        let try_compile = |opts: ObjcId, tag: &str| {
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
                println!("  [{tag}] model=null");
                return;
            }
            let mut err: ObjcId = std::ptr::null_mut();
            let ok = compilef(m, sel("compileWithQoS:options:error:"), 21, opts, &mut err);
            if ok {
                println!("  [{tag}] *** COMPILE SUCCESS! ***");
                list_dir(&lp);
            } else {
                let e = nserror_string(err).unwrap_or_default();
                println!("  [{tag}] {}", inner_error_full(&e));
            }
        };

        // --- Write Duet espresso files ---
        println!("\n--- Writing Duet espresso files to localModelPath ---");
        std::fs::create_dir_all(&lp)?;
        for fname in &[
            "model.espresso.net",
            "model.espresso.shape",
            "model.espresso.weights",
        ] {
            if let Ok(content) = std::fs::read(format!("{DUET}/{fname}")) {
                std::fs::write(format!("{lp}/{fname}"), &content)?;
                println!("  wrote {fname}: {}B", content.len());
            }
        }

        // --- Test 1: kANEFNetPlistFilenameKey only ---
        println!("\n=== Test 1: kANEFNetPlistFilenameKey only ===");
        let opts1 = dictf(cls_mdict as ObjcId, sel("new"));
        setd(opts1, sel("setDictionary:"), base_opts);
        setkv(
            opts1,
            sel("setObject:forKey:"),
            make_nsstr("model.espresso.net"),
            kanef_net_plist,
        );
        try_compile(opts1, "NetPlist only");

        // --- Test 2: + kANEFCompilerOptionsFilenameKey ---
        println!("\n=== Test 2: + kANEFCompilerOptionsFilenameKey ===");
        let opts2 = dictf(cls_mdict as ObjcId, sel("new"));
        setd(opts2, sel("setDictionary:"), base_opts);
        setkv(
            opts2,
            sel("setObject:forKey:"),
            make_nsstr("model.espresso.net"),
            kanef_net_plist,
        );
        // Try empty opts file name
        setkv(
            opts2,
            sel("setObject:forKey:"),
            make_nsstr(""),
            kanef_opts_file,
        );
        try_compile(opts2, "NetPlist + empty OptsFile");

        // Try various opts filenames
        for opts_fname in &[
            "options.plist",
            "compiler_options.plist",
            "compile_options.plist",
            "model.plist",
        ] {
            let opts_x = dictf(cls_mdict as ObjcId, sel("new"));
            setd(opts_x, sel("setDictionary:"), base_opts);
            setkv(
                opts_x,
                sel("setObject:forKey:"),
                make_nsstr("model.espresso.net"),
                kanef_net_plist,
            );
            setkv(
                opts_x,
                sel("setObject:forKey:"),
                make_nsstr(opts_fname),
                kanef_opts_file,
            );
            try_compile(opts_x, opts_fname);
        }

        // --- Test 3: try NearbyInteraction model ---
        println!("\n=== Test 3: NearbyInteraction model ===");
        for fname in &[
            "model.espresso.net",
            "model.espresso.shape",
            "model.espresso.weights",
        ] {
            if let Ok(content) = std::fs::read(format!("{NEARBY}/{fname}")) {
                std::fs::write(format!("{lp}/{fname}"), &content)?;
                println!("  wrote {fname}: {}B (NearbyInteraction)", content.len());
            }
        }
        let opts3 = dictf(cls_mdict as ObjcId, sel("new"));
        setd(opts3, sel("setDictionary:"), base_opts);
        setkv(
            opts3,
            sel("setObject:forKey:"),
            make_nsstr("model.espresso.net"),
            kanef_net_plist,
        );
        try_compile(opts3, "NearbyInteraction");

        // --- Test 4: kANEFEspressoFileResourcesKey ---
        println!("\n=== Test 4: kANEFEspressoFileResourcesKey ===");
        // Restore Duet files
        for fname in &[
            "model.espresso.net",
            "model.espresso.shape",
            "model.espresso.weights",
        ] {
            if let Ok(content) = std::fs::read(format!("{DUET}/{fname}")) {
                std::fs::write(format!("{lp}/{fname}"), &content).ok();
            }
        }
        let opts4 = dictf(cls_mdict as ObjcId, sel("new"));
        setd(opts4, sel("setDictionary:"), base_opts);
        setkv(
            opts4,
            sel("setObject:forKey:"),
            make_nsstr("model.espresso.net"),
            kanef_net_plist,
        );
        // Build espresso resources dict: {"model.espresso.net": ..., "model.espresso.shape": ..., "model.espresso.weights": ...}
        let net_content = std::fs::read(format!("{DUET}/model.espresso.net")).unwrap_or_default();
        let shape_content =
            std::fs::read(format!("{DUET}/model.espresso.shape")).unwrap_or_default();
        let weights_content =
            std::fs::read(format!("{DUET}/model.espresso.weights")).unwrap_or_default();
        let ns_net = dataf(
            cls_data as ObjcId,
            sel("dataWithBytes:length:"),
            net_content.as_ptr(),
            net_content.len() as u64,
        );
        let ns_shape = dataf(
            cls_data as ObjcId,
            sel("dataWithBytes:length:"),
            shape_content.as_ptr(),
            shape_content.len() as u64,
        );
        let ns_weights = dataf(
            cls_data as ObjcId,
            sel("dataWithBytes:length:"),
            weights_content.as_ptr(),
            weights_content.len() as u64,
        );
        let esp_mdict = dictf(cls_mdict as ObjcId, sel("new"));
        setkv(
            esp_mdict,
            sel("setObject:forKey:"),
            ns_net,
            make_nsstr("model.espresso.net"),
        );
        setkv(
            esp_mdict,
            sel("setObject:forKey:"),
            ns_shape,
            make_nsstr("model.espresso.shape"),
        );
        setkv(
            esp_mdict,
            sel("setObject:forKey:"),
            ns_weights,
            make_nsstr("model.espresso.weights"),
        );
        setkv(
            opts4,
            sel("setObject:forKey:"),
            esp_mdict,
            kanef_espresso_resources,
        );
        try_compile(opts4, "EspressoFileResources dict");

        // --- Test 5: Try kANEFModelDescriptionKey with value "ANEFModelDescription" ---
        println!("\n=== Test 5: kANEFModelDescriptionKey ===");
        let kanef_desc_key = read_const("kANEFModelDescriptionKey");
        println!(
            "  kANEFModelDescriptionKey = \"{}\"",
            nsstring_to_str(kanef_desc_key)
        );
        let opts5 = dictf(cls_mdict as ObjcId, sel("new"));
        setd(opts5, sel("setDictionary:"), base_opts);
        setkv(
            opts5,
            sel("setObject:forKey:"),
            make_nsstr("model.espresso.net"),
            kanef_net_plist,
        );
        setkv(
            opts5,
            sel("setObject:forKey:"),
            make_nsstr("kANEFModelANECIR"),
            kanef_desc_key,
        );
        try_compile(opts5, "with kANEFModelDescriptionKey");

        // --- Test 6: All known kANEF keys together ---
        println!("\n=== Test 6: Kitchen sink opts ===");
        let opts6 = dictf(cls_mdict as ObjcId, sel("new"));
        setd(opts6, sel("setDictionary:"), base_opts);
        setkv(
            opts6,
            sel("setObject:forKey:"),
            make_nsstr("model.espresso.net"),
            kanef_net_plist,
        );
        // Try with empty compiler opts file
        setkv(
            opts6,
            sel("setObject:forKey:"),
            make_nsstr("kANEFModelANECIR"),
            kanef_desc_key,
        );
        setkv(
            opts6,
            sel("setObject:forKey:"),
            make_nsstr(&lp),
            make_nsstr("NetworkSourcePath"),
        );
        try_compile(opts6, "kitchen sink");

        std::fs::remove_dir_all(&lp).ok();
    }

    Ok(())
}

unsafe fn read_const(name: &str) -> ObjcId {
    let c = CString::new(name).unwrap();
    let p = dlsym(std::ptr::null_mut(), c.as_ptr()) as *const ObjcId;
    if !p.is_null() && !(*p).is_null() {
        return *p;
    }
    // Fallback: find handle for XPC service
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
    // Try to extract innermost error
    if let Some(idx) = e.rfind("err=(\n    ") {
        let rest = &e[idx + 9..];
        if let Some(end) = rest.find('\n') {
            return format!("err=[{}]", rest[..end].trim());
        }
    }
    // Try last error domain
    if let Some(idx) = e.rfind("\"ANECCompile(") {
        let chunk = &e[idx..];
        if let Some(end) = chunk.find("\"}}") {
            return format!("ANECCompile: ...{}", &chunk[..end.min(200)]);
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

fn list_dir(dir: &str) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let meta = std::fs::metadata(entry.path()).ok();
            let size = meta.map(|m| m.len()).unwrap_or(0);
            println!("    {}: {size}B", entry.file_name().to_string_lossy());
        }
    }
}
