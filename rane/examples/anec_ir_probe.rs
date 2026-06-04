//! Probe the ANEC IR / kANEFModelANECIR path in depth:
//! 1. Dump all kANEF* string constant values via dlsym
//! 2. Try modelWithNetworkDescription: with NetworkSourceFileName key
//! 3. Try compileWithQoS: after writing model.espresso.net with various content
//! 4. Probe compilerOptionsFileName on the model object
//!
//! Run: cargo run -p rane --example anec_ir_probe --release

use rane::ffi::*;
use std::ffi::{c_void, CStr, CString};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ANEC IR path deep probe ===\n");

    for name in &["AppleNeuralEngine", "ANECompiler", "ANEServices"] {
        let path = format!("/System/Library/PrivateFrameworks/{name}.framework/{name}");
        let c = CString::new(path).unwrap();
        unsafe {
            dlopen(c.as_ptr(), RTLD_NOW);
        }
    }

    // --- 1. Dump kANEF* constants ---
    println!("=== 1. kANEF* string constants ===");
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
        "kANEFModelIdentityStrKey",
        "kANEFModelHasCacheURLIdentifierKey",
        "kANEFModelCacheIdentifierUsingSourceURLKey",
        "kANEFModelInstanceParameters",
        "kANEFModelIsEncryptedKey",
        "kANEFBaseModelIdentifierKey",
        "kANEFKeepModelMemoryWiredKey",
        "kANEFModelInput16KAlignmentArrayKey",
        "kANEFModelInputSymbolIndexArrayKey",
        "kANEFModelInputSymbolsArrayKey",
        "kANEFModelOutputSymbolIndexArrayKey",
        "kANEFModelOutputSymbolsArrayKey",
        "kANEFModelOutput16KAlignmentArrayKey",
        "kANEFModelLoadPerformanceStatsKey",
        "kANEFModelType",
    ];
    unsafe {
        for name in &const_names {
            let c = CString::new(*name).unwrap();
            let p = dlsym(std::ptr::null_mut(), c.as_ptr()) as *const ObjcId;
            if !p.is_null() && !(*p).is_null() {
                println!("  {name} = \"{}\"", nsstring_to_str(*p));
            } else {
                println!("  {name} = NOT FOUND");
            }
        }
    }

    // --- Setup: build a small MIL program + weights ---
    let ic = 16usize;
    let oc = 16usize;
    let data_off: u64 = 64;
    let qdata_size = ic * oc;
    let scale_size = ic * 2;
    let total = 64 + qdata_size + scale_size + ic;
    let mut wblob = vec![0u8; total];
    for b in &mut wblob[64..64 + qdata_size] {
        *b = 1;
    }
    for i in 0..ic {
        let off = 64 + qdata_size + i * 2;
        wblob[off + 1] = 0x3C;
    }
    let scale_off = data_off + qdata_size as u64;
    let zp_off = scale_off + scale_size as u64;

    let mil_text = format!(
        concat!(
            "program(1, 0)\nfunc main<ios16>(tensor<fp16, [1, {ic}, 1, 1]> x)",
            " -> (tensor<fp16, [1, {oc}, 1, 1]>) {{\n  block0() {{\n",
            "    tensor<fp16, [{ic},{oc}]> wf = constexpr_affine_dequantize()",
            "[axis = int32(0),",
            " quantized_data = tensor<int8, [{ic},{oc}]>(BLOBFILE(path = string(\"@model_path/weights/weights.bin\"), offset = uint64({data_off}))),",
            " scale = tensor<fp16, [{ic}]>(BLOBFILE(path = string(\"@model_path/weights/weights.bin\"), offset = uint64({scale_off}))),",
            " zero_point = tensor<int8, [{ic}]>(BLOBFILE(path = string(\"@model_path/weights/weights.bin\"), offset = uint64({zp_off})))];\n",
            "    tensor<fp16, [1, {oc}, 1, 1]> y = linear()[alpha = fp32(1), beta = fp32(0), weight = wf, x = x];\n",
            "  }} -> (y)\n}}\n"
        ),
        ic = ic, oc = oc, data_off = data_off, scale_off = scale_off, zp_off = zp_off
    );

    unsafe {
        type DataFn = unsafe extern "C" fn(ObjcId, ObjcSel, *const u8, u64) -> ObjcId;
        type DictFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type DescFn3 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, ObjcId) -> ObjcId;
        type ModelFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type StrFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type Utf8Fn = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const std::ffi::c_char;
        type ModelFn2 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, u8) -> ObjcId;
        type CompileFn = unsafe extern "C" fn(ObjcId, ObjcSel, u32, ObjcId, *mut ObjcId) -> bool;
        type DictMakeFn =
            unsafe extern "C" fn(ObjcId, ObjcSel, *const ObjcId, *const ObjcId, u64) -> ObjcId;

        let cls_data = cls("NSData");
        let cls_dict = cls("NSDictionary");
        let cls_desc = cls("_ANEInMemoryModelDescriptor");
        let cls_model = cls("_ANEInMemoryModel");
        let cls_str = cls("NSString");

        let dataf: DataFn = std::mem::transmute(objc_msgSend as *const c_void);
        let dictf: DictFn = std::mem::transmute(objc_msgSend as *const c_void);
        let descf: DescFn3 = std::mem::transmute(objc_msgSend as *const c_void);
        let modelf: ModelFn = std::mem::transmute(objc_msgSend as *const c_void);
        let strf: StrFn = std::mem::transmute(objc_msgSend as *const c_void);
        let utf8f: Utf8Fn = std::mem::transmute(objc_msgSend as *const c_void);
        let modelf2: ModelFn2 = std::mem::transmute(objc_msgSend as *const c_void);
        let compilef: CompileFn = std::mem::transmute(objc_msgSend as *const c_void);
        let dkv: DictMakeFn = std::mem::transmute(objc_msgSend as *const c_void);

        let bytes = mil_text.as_bytes();
        let ns_text = dataf(
            cls_data as ObjcId,
            sel("dataWithBytes:length:"),
            bytes.as_ptr(),
            bytes.len() as u64,
        );
        let empty = dictf(cls_dict as ObjcId, sel("dictionary"));

        // Build tmp_dir from MIL model
        let desc_base = descf(
            cls_desc as ObjcId,
            sel("modelWithMILText:weights:optionsPlist:"),
            ns_text,
            empty,
            std::ptr::null_mut(),
        );
        let model_base = modelf(
            cls_model as ObjcId,
            sel("inMemoryModelWithDescriptor:"),
            desc_base,
        );
        let hex_id = strf(model_base, sel("hexStringIdentifier"));
        let hex_str = {
            let c = utf8f(hex_id, sel("UTF8String"));
            CStr::from_ptr(c).to_string_lossy().into_owned()
        };
        let tmp_dir = format!("/tmp/{hex_str}");
        println!("\ntmp_dir: {tmp_dir}");
        std::fs::create_dir_all(format!("{tmp_dir}/weights")).unwrap();
        std::fs::write(format!("{tmp_dir}/model.mil"), &mil_text).unwrap();
        std::fs::write(format!("{tmp_dir}/weights/weights.bin"), &wblob).unwrap();

        // --- 2. compilerOptionsFileName ---
        println!("\n=== 2. compilerOptionsFileName ===");
        let opts_fname = strf(model_base, sel("compilerOptionsFileName"));
        println!(
            "  MIL model compilerOptionsFileName = {}",
            nsstring_to_str(opts_fname)
        );

        // Flipped model
        let desc_flip = descf(
            cls_desc as ObjcId,
            sel("modelWithMILText:weights:optionsPlist:"),
            ns_text,
            empty,
            std::ptr::null_mut(),
        );
        let p = (desc_flip as *mut u8).add(8);
        *p = 0; // isMILModel=false
        let model_flip = modelf(
            cls_model as ObjcId,
            sel("inMemoryModelWithDescriptor:"),
            desc_flip,
        );
        let opts_fname_flip = strf(model_flip, sel("compilerOptionsFileName"));
        println!(
            "  ANECIR model compilerOptionsFileName = {}",
            nsstring_to_str(opts_fname_flip)
        );

        // localModelPath
        let local_path = strf(model_base, sel("localModelPath"));
        println!("  MIL localModelPath = {}", nsstring_to_str(local_path));
        let local_path_flip = strf(model_flip, sel("localModelPath"));
        println!(
            "  ANECIR localModelPath = {}",
            nsstring_to_str(local_path_flip)
        );

        // --- 3. modelWithNetworkDescription: probes ---
        println!("\n=== 3. modelWithNetworkDescription: probes ===");

        let make_nsstr = |s: &str| -> ObjcId {
            let c = CString::new(s).unwrap();
            type StrFn2 = unsafe extern "C" fn(ObjcId, ObjcSel, *const std::ffi::c_char) -> ObjcId;
            let sf2: StrFn2 = std::mem::transmute(objc_msgSend as *const c_void);
            sf2(cls_str as ObjcId, sel("stringWithUTF8String:"), c.as_ptr())
        };

        // Helper: build dict with 1 k/v pair
        let make_dict1 = |k: ObjcId, v: ObjcId| -> ObjcId {
            let keys = [k];
            let vals = [v];
            dkv(
                cls_dict as ObjcId,
                sel("dictionaryWithObjects:forKeys:count:"),
                vals.as_ptr(),
                keys.as_ptr(),
                1u64,
            )
        };

        // Try dict with various keys that might be the networkDescription format
        let candidate_keys = [
            "NetworkSourceFileName",
            "NetworkSourcePath",
            "kANEFModelDescriptionKey",
        ];

        for key_name in &candidate_keys {
            let k = make_nsstr(key_name);
            let v = make_nsstr("model.espresso.net");
            let nd = make_dict1(k, v);
            let desc_nd = descf(
                cls_desc as ObjcId,
                sel("modelWithNetworkDescription:weights:optionsPlist:"),
                nd,
                empty,
                std::ptr::null_mut(),
            );
            if desc_nd.is_null() {
                println!("  key={key_name}: desc=null");
                continue;
            }
            let is_mil = *(desc_nd as *const u8).add(8);
            let hex = strf(desc_nd, sel("hexStringIdentifier"));
            let hex_s = nsstring_to_str(hex);
            println!("  key={key_name}: desc={desc_nd:p} isMILModel={is_mil} hex={hex_s}");

            // Try to compile it
            let model_nd = modelf(
                cls_model as ObjcId,
                sel("inMemoryModelWithDescriptor:"),
                desc_nd,
            );
            if model_nd.is_null() {
                println!("    model=null");
                continue;
            }

            let opts = modelf2(
                model_nd,
                sel("compilerOptionsWithOptions:isCompiledModelCached:"),
                empty,
                0,
            );
            println!("    opts = {}", objc_desc(opts));

            // Write model.espresso.net to the correct tmp dir
            let hex_nd = strf(model_nd, sel("hexStringIdentifier"));
            let hex_nd_s = nsstring_to_str(hex_nd);
            let nd_dir = format!("/tmp/{hex_nd_s}");
            std::fs::create_dir_all(format!("{nd_dir}/weights")).ok();
            std::fs::write(format!("{nd_dir}/model.espresso.net"), b"{}").ok();
            std::fs::write(format!("{nd_dir}/weights/weights.bin"), &wblob).ok();

            let mut err: ObjcId = std::ptr::null_mut();
            let ok = compilef(
                model_nd,
                sel("compileWithQoS:options:error:"),
                21,
                empty,
                &mut err,
            );
            if ok {
                println!("    *** COMPILE SUCCESS! ***");
            } else {
                let e = nserror_string(err).unwrap_or_default();
                println!("    error: {}", &e[..e.len().min(300)]);
            }
            std::fs::remove_dir_all(&nd_dir).ok();
        }

        // --- 4. kANEFEspressoFileResourcesKey value probe ---
        println!("\n=== 4. kANEFEspressoFileResourcesKey probe ===");
        let espc = CString::new("kANEFEspressoFileResourcesKey").unwrap();
        let esp_ptr = dlsym(std::ptr::null_mut(), espc.as_ptr()) as *const ObjcId;
        if !esp_ptr.is_null() && !(*esp_ptr).is_null() {
            let esp_key = *esp_ptr;
            println!(
                "  kANEFEspressoFileResourcesKey = \"{}\"",
                nsstring_to_str(esp_key)
            );

            // Try using it as the networkDescription key
            let v_espresso = make_nsstr("model.espresso.net");
            let nd = make_dict1(esp_key, v_espresso);
            let desc_esp = descf(
                cls_desc as ObjcId,
                sel("modelWithNetworkDescription:weights:optionsPlist:"),
                nd,
                empty,
                std::ptr::null_mut(),
            );
            if desc_esp.is_null() {
                println!("  espressoKey desc=null");
            } else {
                let is_mil = *(desc_esp as *const u8).add(8);
                println!("  espressoKey desc={desc_esp:p} isMILModel={is_mil}");
                let model_esp = modelf(
                    cls_model as ObjcId,
                    sel("inMemoryModelWithDescriptor:"),
                    desc_esp,
                );
                if !model_esp.is_null() {
                    let opts = modelf2(
                        model_esp,
                        sel("compilerOptionsWithOptions:isCompiledModelCached:"),
                        empty,
                        0,
                    );
                    println!("  opts = {}", objc_desc(opts));
                }
            }
        } else {
            println!("  kANEFEspressoFileResourcesKey: not in main process — checking via ANEServices dlsym");
            // Try after loading ANEServices
            let svc_path =
                CString::new("/System/Library/PrivateFrameworks/ANEServices.framework/ANEServices")
                    .unwrap();
            dlopen(svc_path.as_ptr(), RTLD_NOW);
            let p2 = dlsym(std::ptr::null_mut(), espc.as_ptr()) as *const ObjcId;
            if !p2.is_null() && !(*p2).is_null() {
                println!(
                    "  found after ANEServices load: \"{}\"",
                    nsstring_to_str(*p2)
                );
            } else {
                println!("  still not found");
            }
        }

        // --- 5. Write model.espresso.net and try ANECIR compile ---
        println!("\n=== 5. Write model.espresso.net to flipped model dir ===");
        // The flipped model (isMILModel=false from modelWithMILText:) has same hex ID as model_flip
        // Write model.espresso.net there
        std::fs::write(format!("{tmp_dir}/model.espresso.net"), b"{}").ok();

        // The ANECVAIRCompiler in the XPC service has defaultANECIRFileName.
        // The NetworkSourceFile name comes from the descriptor. Let's try a descriptor
        // created via modelWithNetworkDescription: and see what hexId it gets.

        // Try modelWithNetworkDescription: with NSData as the description (maybe it IS the Espresso binary)
        println!("\n  Testing NSData as networkDescription...");
        type DescFnData = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, ObjcId) -> ObjcId;
        let descfd: DescFnData = std::mem::transmute(objc_msgSend as *const c_void);
        let dummy_data = dataf(
            cls_data as ObjcId,
            sel("dataWithBytes:length:"),
            b"{}".as_ptr(),
            2u64,
        );
        let desc_data = descfd(
            cls_desc as ObjcId,
            sel("modelWithNetworkDescription:weights:optionsPlist:"),
            dummy_data,
            empty,
            std::ptr::null_mut(),
        );
        if desc_data.is_null() {
            println!("  NSData desc=null (expected — it crashes or returns nil)");
        } else {
            let is_mil = *(desc_data as *const u8).add(8);
            println!("  NSData desc={desc_data:p} isMILModel={is_mil}");
            let model_data = modelf(
                cls_model as ObjcId,
                sel("inMemoryModelWithDescriptor:"),
                desc_data,
            );
            if !model_data.is_null() {
                let opts = modelf2(
                    model_data,
                    sel("compilerOptionsWithOptions:isCompiledModelCached:"),
                    empty,
                    0,
                );
                println!("  opts = {}", objc_desc(opts));
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
