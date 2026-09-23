//! Probe the non-MIL compiler path.
//!
//! Key discoveries so far:
//!   - _ANEInMemoryModelDescriptor has isMILModel BOOL ivar
//!   - initWithNetworkText:...:isMILModel:NO throws ObjC exception (can't call directly)
//!   - modelWithNetworkDescription: takes unknown "network description" object
//!
//! Strategy:
//!   A. Enumerate ivars on _ANEInMemoryModelDescriptor → find _isMILModel offset
//!   B. Create descriptor via modelWithMILText: (isMILModel=YES), then flip _isMILModel=NO
//!   C. Compile and observe: does the compile route to a different compiler?
//!   D. Try modelWithNetworkDescription: with NSData(coremldata.bin)
//!   E. Try CoreML.framework MLModel path to access _ANECoreMLModelCompiler
//!
//! Run: cargo run -p rane --example non_mil_probe --release

use rane::ffi::*;
use std::ffi::{c_void, CStr, CString};

extern "C" {
    fn free(p: *mut c_void);
    fn class_getInstanceVariable(cls: ObjcClass, name: *const std::ffi::c_char) -> ObjcIvar;
}
fn libc_free(p: *mut c_void) {
    unsafe {
        free(p);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== non-MIL path probe ===\n");

    for name in &["AppleNeuralEngine", "ANECompiler", "ANEServices"] {
        let path = format!("/System/Library/PrivateFrameworks/{name}.framework/{name}");
        let c = CString::new(path).unwrap();
        unsafe {
            dlopen(c.as_ptr(), RTLD_NOW);
        }
    }

    // ─── Setup ───────────────────────────────────────────────────────────────
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
        wblob[off + 1] = 0x3C; // fp16(1.0)
    }
    let scale_off = data_off + qdata_size as u64;
    let zp_off = scale_off + scale_size as u64;

    let dir = "/tmp/ane_quant_test.mlmodelc";
    std::fs::create_dir_all(format!("{dir}/weights"))?;
    std::fs::write(format!("{dir}/weights/weights.bin"), &wblob)?;

    let constexpr_mil = format!(
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
    std::fs::write(format!("{dir}/model.mil"), &constexpr_mil)?;

    // ─── A. Enumerate _ANEInMemoryModelDescriptor ivars ──────────────────────
    println!("=== A. _ANEInMemoryModelDescriptor ivars ===");
    let desc_cls = cls("_ANEInMemoryModelDescriptor");
    let isMILModel_offset = enumerate_ivars_and_find(desc_cls, "_isMILModel");
    println!("  _isMILModel ivar offset: {:?}", isMILModel_offset);
    println!();

    // ─── B. Flip _isMILModel after modelWithMILText: ─────────────────────────
    println!("=== B. modelWithMILText: then flip _isMILModel=false ===");
    probe_flip_is_mil(&constexpr_mil, &wblob, dir, isMILModel_offset);
    println!();

    // ─── C. modelWithNetworkDescription: with NSData(coremldata.bin) ─────────
    println!("=== C. modelWithNetworkDescription: NSData(coremldata.bin) ===");
    let coreml_bin_path = "/System/Library/PrivateFrameworks/VoiceActions.framework/Versions/A/Resources/aa_encoder_125141826.mlmodelc/coremldata.bin";
    if let Ok(coreml_bin) = std::fs::read(coreml_bin_path) {
        println!("  coremldata.bin: {} bytes", coreml_bin.len());
        // Try as networkDescription, with VoiceActions weights as weights arg
        let wa_weights = "/System/Library/PrivateFrameworks/VoiceActions.framework/Versions/A/Resources/aa_encoder_125141826.mlmodelc/weights/weight.bin";
        let wa_wblob = std::fs::read(wa_weights).unwrap_or_default();
        probe_network_desc_nsdata(&coreml_bin, &wa_wblob, dir);
    }
    println!();

    // ─── D. CoreML.framework path ─────────────────────────────────────────────
    println!("=== D. MLModel from VoiceActions bundle ===");
    probe_coreml_framework();
    println!();

    // ─── E. modelWithNetworkDescription: with NSString path ──────────────────
    println!("=== E. modelWithNetworkDescription: with NSString bundle path ===");
    probe_network_desc_string(
        "/System/Library/PrivateFrameworks/VoiceActions.framework/Versions/A/Resources/aa_encoder_125141826.mlmodelc",
        &wblob,
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────

fn enumerate_ivars_and_find(cls: ObjcClass, target: &str) -> Option<isize> {
    unsafe {
        let mut c = cls;
        while !c.is_null() {
            let cls_name = {
                let p = class_getName(c);
                if p.is_null() {
                    "(?)".to_string()
                } else {
                    CStr::from_ptr(p).to_string_lossy().into_owned()
                }
            };
            let mut count: u32 = 0;
            let list = class_copyIvarList(c, &mut count);
            if !list.is_null() {
                for i in 0..count {
                    let ivar = *list.add(i as usize);
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
                    println!("  [{cls_name}] +{off:04x} {name}: {ty}");
                }
                libc_free(list as *mut c_void);
            }
            c = class_getSuperclass(c);
        }

        // Find by exact ivar name
        let cn = CString::new(target).unwrap();
        let ivar = class_getInstanceVariable(cls, cn.as_ptr());
        if ivar.is_null() {
            // Try without underscore
            let cn2 = CString::new(target.trim_start_matches('_')).unwrap();
            let ivar2 = class_getInstanceVariable(cls, cn2.as_ptr());
            if !ivar2.is_null() {
                return Some(ivar_getOffset(ivar2));
            }
            return None;
        }
        Some(ivar_getOffset(ivar))
    }
}

fn probe_flip_is_mil(
    mil_text: &str,
    weights: &[u8],
    model_dir: &str,
    isMILModel_offset: Option<isize>,
) {
    unsafe {
        type DataFn = unsafe extern "C" fn(ObjcId, ObjcSel, *const u8, u64) -> ObjcId;
        type DictFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type DescFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, ObjcId) -> ObjcId;
        type ModelFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type StrFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type Utf8Fn = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const std::ffi::c_char;
        type BoolFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> bool;
        type CompileFn = unsafe extern "C" fn(ObjcId, ObjcSel, u32, ObjcId, *mut ObjcId) -> bool;

        let cls_data = cls("NSData");
        let cls_dict = cls("NSDictionary");
        let cls_desc = cls("_ANEInMemoryModelDescriptor");
        let cls_model = cls("_ANEInMemoryModel");

        let dataf: DataFn = std::mem::transmute(objc_msgSend as *const c_void);
        let dictf: DictFn = std::mem::transmute(objc_msgSend as *const c_void);
        let descf: DescFn = std::mem::transmute(objc_msgSend as *const c_void);
        let modelf: ModelFn = std::mem::transmute(objc_msgSend as *const c_void);
        let strf: StrFn = std::mem::transmute(objc_msgSend as *const c_void);
        let utf8f: Utf8Fn = std::mem::transmute(objc_msgSend as *const c_void);
        let boolf: BoolFn = std::mem::transmute(objc_msgSend as *const c_void);
        let compilef: CompileFn = std::mem::transmute(objc_msgSend as *const c_void);

        let bytes = mil_text.as_bytes();
        let ns_text = dataf(
            cls_data as ObjcId,
            sel("dataWithBytes:length:"),
            bytes.as_ptr(),
            bytes.len() as u64,
        );
        let ns_weights = dataf(
            cls_data as ObjcId,
            sel("dataWithBytes:length:"),
            weights.as_ptr(),
            weights.len() as u64,
        );
        let empty = dictf(cls_dict as ObjcId, sel("dictionary"));

        // Create via modelWithMILText: (isMILModel=YES)
        let desc = descf(
            cls_desc as ObjcId,
            sel("modelWithMILText:weights:optionsPlist:"),
            ns_text,
            ns_weights,
            empty,
        );
        if desc.is_null() {
            println!("  descriptor = null");
            return;
        }

        // Read isMILModel before flip
        let before = boolf(desc, sel("isMILModel"));
        println!("  isMILModel before: {before}");

        // Flip _isMILModel ivar if we found the offset
        if let Some(off) = isMILModel_offset {
            let ptr = (desc as *mut u8).add(off as usize) as *mut u8;
            *ptr = 0; // false
            let after = boolf(desc, sel("isMILModel"));
            println!("  isMILModel after flip: {after}");
        } else {
            // Try flipping via known ivar name alternatives
            let mut found = false;
            for name in &["_isMILModel", "isMILModel", "_isMIL", "_milModel"] {
                let cn = CString::new(*name).unwrap();
                let ivar = class_getInstanceVariable(cls_desc, cn.as_ptr());
                if !ivar.is_null() {
                    let off = ivar_getOffset(ivar);
                    let ptr = (desc as *mut u8).add(off as usize) as *mut u8;
                    *ptr = 0;
                    println!("  Flipped {name} at offset {off}");
                    found = true;
                    break;
                }
            }
            if !found {
                println!("  Could not find isMILModel ivar — skipping flip");
            }
        }

        let model = modelf(
            cls_model as ObjcId,
            sel("inMemoryModelWithDescriptor:"),
            desc,
        );
        if model.is_null() {
            println!("  model = null");
            return;
        }

        // Setup tmp_dir
        let hex_id = strf(model, sel("hexStringIdentifier"));
        let hex_str = {
            let c = utf8f(hex_id, sel("UTF8String"));
            CStr::from_ptr(c).to_string_lossy().into_owned()
        };
        let tmp_dir = std::env::temp_dir().join(&hex_str);
        let _ = std::fs::create_dir_all(tmp_dir.join("weights"));
        let _ = std::fs::write(tmp_dir.join("model.mil"), mil_text);
        let _ = std::fs::copy(
            format!("{model_dir}/weights/weights.bin"),
            tmp_dir.join("weights").join("weights.bin"),
        );
        println!("  tmp_dir: {}", tmp_dir.display());

        let mut err: ObjcId = std::ptr::null_mut();
        let ok = compilef(
            model,
            sel("compileWithQoS:options:error:"),
            21,
            empty,
            &mut err,
        );
        if ok {
            println!("  *** COMPILE SUCCESS after isMILModel flip! ***");
        } else {
            println!("  compile error: {:?}", nserror_string(err));
        }
        print_dir_tree(tmp_dir.to_str().unwrap());
    }
}

fn probe_network_desc_nsdata(data: &[u8], weights: &[u8], model_dir: &str) {
    unsafe {
        type DataFn = unsafe extern "C" fn(ObjcId, ObjcSel, *const u8, u64) -> ObjcId;
        type DictFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type DescFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, ObjcId) -> ObjcId;
        type ModelFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type StrFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type Utf8Fn = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const std::ffi::c_char;
        type BoolFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> bool;
        type CompileFn = unsafe extern "C" fn(ObjcId, ObjcSel, u32, ObjcId, *mut ObjcId) -> bool;

        let cls_data = cls("NSData");
        let cls_dict = cls("NSDictionary");
        let cls_desc = cls("_ANEInMemoryModelDescriptor");
        let cls_model = cls("_ANEInMemoryModel");

        let dataf: DataFn = std::mem::transmute(objc_msgSend as *const c_void);
        let dictf: DictFn = std::mem::transmute(objc_msgSend as *const c_void);
        let descf: DescFn = std::mem::transmute(objc_msgSend as *const c_void);
        let modelf: ModelFn = std::mem::transmute(objc_msgSend as *const c_void);
        let strf: StrFn = std::mem::transmute(objc_msgSend as *const c_void);
        let utf8f: Utf8Fn = std::mem::transmute(objc_msgSend as *const c_void);
        let boolf: BoolFn = std::mem::transmute(objc_msgSend as *const c_void);
        let compilef: CompileFn = std::mem::transmute(objc_msgSend as *const c_void);

        let ns_data = dataf(
            cls_data as ObjcId,
            sel("dataWithBytes:length:"),
            data.as_ptr(),
            data.len() as u64,
        );
        let ns_w = if !weights.is_empty() {
            dataf(
                cls_data as ObjcId,
                sel("dataWithBytes:length:"),
                weights.as_ptr(),
                weights.len() as u64,
            )
        } else {
            std::ptr::null_mut()
        };
        let empty = dictf(cls_dict as ObjcId, sel("dictionary"));

        let desc = descf(
            cls_desc as ObjcId,
            sel("modelWithNetworkDescription:weights:optionsPlist:"),
            ns_data,
            ns_w,
            empty,
        );
        if desc.is_null() {
            println!("  descriptor = null");
            return;
        }
        println!("  descriptor OK");
        println!("  isMILModel: {}", boolf(desc, sel("isMILModel")));

        let model = modelf(
            cls_model as ObjcId,
            sel("inMemoryModelWithDescriptor:"),
            desc,
        );
        if model.is_null() {
            println!("  model = null");
            return;
        }

        let hex_id = strf(model, sel("hexStringIdentifier"));
        let hex_str = {
            let c = utf8f(hex_id, sel("UTF8String"));
            CStr::from_ptr(c).to_string_lossy().into_owned()
        };
        let tmp_dir = std::env::temp_dir().join(&hex_str);
        let _ = std::fs::create_dir_all(tmp_dir.join("weights"));
        // Write coremldata.bin to see if compiler accepts it
        let _ = std::fs::write(tmp_dir.join("coremldata.bin"), data);
        // Also write the VoiceActions weights
        let wa_w = "/System/Library/PrivateFrameworks/VoiceActions.framework/Versions/A/Resources/aa_encoder_125141826.mlmodelc/weights/weight.bin";
        if let Ok(_) = std::fs::copy(wa_w, tmp_dir.join("weights").join("weight.bin")) {}
        // Also write model.mil from VoiceActions
        let va_mil = "/System/Library/PrivateFrameworks/VoiceActions.framework/Versions/A/Resources/aa_encoder_125141826.mlmodelc/model.mil";
        if let Ok(t) = std::fs::read_to_string(va_mil) {
            let _ = std::fs::write(tmp_dir.join("model.mil"), t);
        }
        println!("  tmp_dir: {}", tmp_dir.display());

        let mut err: ObjcId = std::ptr::null_mut();
        let ok = compilef(
            model,
            sel("compileWithQoS:options:error:"),
            21,
            empty,
            &mut err,
        );
        if ok {
            println!("  *** COMPILE SUCCESS with NSData(coremldata.bin)! ***");
            print_dir_tree(tmp_dir.to_str().unwrap());
        } else {
            println!("  compile error: {:?}", nserror_string(err));
        }
    }
}

fn probe_network_desc_string(bundle_path: &str, weights: &[u8]) {
    unsafe {
        type DictFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type DataFn = unsafe extern "C" fn(ObjcId, ObjcSel, *const u8, u64) -> ObjcId;
        type DescFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, ObjcId) -> ObjcId;
        type ModelFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type StrFn2 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type BoolFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> bool;
        type CompileFn = unsafe extern "C" fn(ObjcId, ObjcSel, u32, ObjcId, *mut ObjcId) -> bool;

        let cls_data = cls("NSData");
        let cls_dict = cls("NSDictionary");
        let cls_desc = cls("_ANEInMemoryModelDescriptor");
        let cls_model = cls("_ANEInMemoryModel");
        let cls_nsstr = cls("NSString");

        let dataf: DataFn = std::mem::transmute(objc_msgSend as *const c_void);
        let dictf: DictFn = std::mem::transmute(objc_msgSend as *const c_void);
        let descf: DescFn = std::mem::transmute(objc_msgSend as *const c_void);
        let modelf: ModelFn = std::mem::transmute(objc_msgSend as *const c_void);
        let strf2: StrFn2 = std::mem::transmute(objc_msgSend as *const c_void);
        let boolf: BoolFn = std::mem::transmute(objc_msgSend as *const c_void);
        let compilef: CompileFn = std::mem::transmute(objc_msgSend as *const c_void);

        let cs = CString::new(bundle_path).unwrap();
        let ns_str = strf2(
            cls_nsstr as ObjcId,
            sel("stringWithUTF8String:"),
            cs.as_ptr() as ObjcId,
        );
        let ns_w = if !weights.is_empty() {
            dataf(
                cls_data as ObjcId,
                sel("dataWithBytes:length:"),
                weights.as_ptr(),
                weights.len() as u64,
            )
        } else {
            std::ptr::null_mut()
        };
        let empty = dictf(cls_dict as ObjcId, sel("dictionary"));

        let desc = descf(
            cls_desc as ObjcId,
            sel("modelWithNetworkDescription:weights:optionsPlist:"),
            ns_str,
            ns_w,
            empty,
        );
        if desc.is_null() {
            println!("  descriptor = null with NSString");
            return;
        }
        println!("  descriptor OK with NSString");
        println!("  isMILModel: {}", boolf(desc, sel("isMILModel")));

        let model = modelf(
            cls_model as ObjcId,
            sel("inMemoryModelWithDescriptor:"),
            desc,
        );
        if model.is_null() {
            println!("  model = null");
            return;
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
            println!("  *** COMPILE SUCCESS with NSString path! ***");
        } else {
            println!("  compile error: {:?}", nserror_string(err));
        }
    }
}

fn probe_coreml_framework() {
    unsafe {
        // dlopen CoreML.framework (public)
        let cml_path = "/System/Library/Frameworks/CoreML.framework/CoreML";
        let c = CString::new(cml_path).unwrap();
        let handle = dlopen(c.as_ptr(), RTLD_NOW);
        if handle.is_null() {
            println!("  CoreML.framework: failed to load");
            return;
        }
        println!("  CoreML.framework: loaded");

        // Get MLModel class
        let ml_model_cls = cls("MLModel");
        if ml_model_cls.is_null() {
            println!("  MLModel class: not found");
            return;
        }
        println!("  MLModel class: found");

        // Build URL for VoiceActions bundle
        let bundle_path = "/System/Library/PrivateFrameworks/VoiceActions.framework/Versions/A/Resources/aa_encoder_125141826.mlmodelc";
        let cls_nsstr = cls("NSString");
        let cls_nsurl = cls("NSURL");
        let cls_nsdict = cls("NSDictionary");

        type StrFn2 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type DictFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type ModelFn3e =
            unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, *mut ObjcId) -> ObjcId;
        type CompileFn2 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, *mut ObjcId) -> ObjcId;

        let sf: StrFn2 = std::mem::transmute(objc_msgSend as *const c_void);
        let df: DictFn = std::mem::transmute(objc_msgSend as *const c_void);
        let mf3e: ModelFn3e = std::mem::transmute(objc_msgSend as *const c_void);

        let cs = CString::new(bundle_path).unwrap();
        let ns_str = sf(
            cls_nsstr as ObjcId,
            sel("stringWithUTF8String:"),
            cs.as_ptr() as ObjcId,
        );
        let ns_url = sf(cls_nsurl as ObjcId, sel("fileURLWithPath:"), ns_str);

        // [MLModel modelWithContentsOfURL:configuration:error:]
        let cls_config = cls("MLModelConfiguration");
        let config = if cls_config.is_null() {
            std::ptr::null_mut()
        } else {
            df(cls_config as ObjcId, sel("new"))
        };

        let mut err: ObjcId = std::ptr::null_mut();
        println!("  Loading VoiceActions bundle via MLModel...");
        let ml_model = mf3e(
            ml_model_cls as ObjcId,
            sel("modelWithContentsOfURL:configuration:error:"),
            ns_url,
            config,
            &mut err,
        );
        if ml_model.is_null() {
            println!("  MLModel = null: {:?}", nserror_string(err));
            return;
        }
        println!("  MLModel loaded: {ml_model:p}");

        // Enumerate MLModel ivars to find _ANEInMemoryModel inside
        type IvarFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        let ivf: IvarFn = std::mem::transmute(objc_msgSend as *const c_void);

        // Try common internal property names
        for ivar_name in &["_impl", "_engine", "_model", "_backend", "model", "impl"] {
            let inner = read_ivar_obj(ml_model, ivar_name);
            if !inner.is_null() {
                let cn = class_name(inner);
                println!("  MLModel.{ivar_name}: {cn}");
                if cn.contains("ANE") {
                    // Found ANE model inside MLModel
                    println!("  *** Found ANE model at MLModel.{ivar_name} ***");
                    let isMIL = ivf(inner, sel("isMILModel"));
                    let _ = isMIL;
                }
            }
        }
    }
}

unsafe fn read_ivar_obj(obj: ObjcId, name: &str) -> ObjcId {
    let cls = object_getClass(obj);
    let mut c = cls;
    while !c.is_null() {
        let mut count: u32 = 0;
        let list = class_copyIvarList(c, &mut count);
        if !list.is_null() {
            for i in 0..count {
                let ivar = *list.add(i as usize);
                let np = ivar_getName(ivar);
                if !np.is_null() {
                    let n = CStr::from_ptr(np).to_string_lossy();
                    if n == name {
                        libc_free(list as *mut c_void);
                        return object_getIvar(obj, ivar);
                    }
                }
            }
            libc_free(list as *mut c_void);
        }
        c = class_getSuperclass(c);
    }
    std::ptr::null_mut()
}

unsafe fn class_name(obj: ObjcId) -> String {
    let c = object_getClass(obj);
    if c.is_null() {
        return "(null)".into();
    }
    let p = class_getName(c);
    if p.is_null() {
        return "(null)".into();
    }
    CStr::from_ptr(p).to_string_lossy().into_owned()
}

fn print_dir_tree(dir: &str) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let name = e.file_name();
            let sz = e.metadata().map(|m| m.len()).unwrap_or(0);
            let is_dir = e.metadata().map(|m| m.is_dir()).unwrap_or(false);
            if is_dir {
                println!("    {name:?}/");
                let sub = format!("{}/{}", dir, name.to_string_lossy());
                if let Ok(sub_entries) = std::fs::read_dir(&sub) {
                    for se in sub_entries.flatten() {
                        let sn = se.file_name();
                        let ssz = se.metadata().map(|m| m.len()).unwrap_or(0);
                        println!("      {sn:?}: {ssz}B");
                    }
                }
            } else {
                println!("    {name:?}: {sz}B");
            }
        }
    }
}
