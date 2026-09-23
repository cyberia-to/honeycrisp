//! Load an int8-MIL .mlmodelc via public MLModel API and run on ANE.
//! Path: MLModelConfiguration.computeUnits = .cpuAndNeuralEngine
//!       → CoreML JIT-compiles MIL → ANE hwx → routes inference
//!
//! Run: cargo run -p rane --example mlmodel_int8_load --release

use rane::ffi::*;
use std::ffi::{c_void, CStr, CString};

const NSUTF8_ENCODING: u64 = 4;

// MLComputeUnits enum
const MLCOMPUTE_UNITS_ALL: i64 = 0;
const MLCOMPUTE_UNITS_CPU_AND_GPU: i64 = 1;
const MLCOMPUTE_UNITS_CPU_ONLY: i64 = 2;
const MLCOMPUTE_UNITS_CPU_AND_ANE: i64 = 3;

const FS_INT8: &str = "/System/Library/PrivateFrameworks/VoiceActions.framework/Versions/A/Resources/aa_encoder_125141826.mlmodelc";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MLModel int8 load + ANE routing ===\n");

    // Load CoreML.framework
    let cml_path = CString::new("/System/Library/Frameworks/CoreML.framework/CoreML").unwrap();
    unsafe { dlopen(cml_path.as_ptr(), RTLD_NOW | 0x8); }

    unsafe {
        type AllocFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type InitBytesFn = unsafe extern "C" fn(ObjcId, ObjcSel, *const u8, usize, u64) -> ObjcId;
        type UrlFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type DictInitFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type ModelWithUrlConfigFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, *mut ObjcId) -> ObjcId;
        type SetComputeUnitsFn = unsafe extern "C" fn(ObjcId, ObjcSel, i64);
        type GetComputeUnitsFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> i64;

        let allocf: AllocFn = std::mem::transmute(objc_msgSend as *const c_void);
        let initf: InitBytesFn = std::mem::transmute(objc_msgSend as *const c_void);
        let urlf: UrlFn = std::mem::transmute(objc_msgSend as *const c_void);
        let dict_init_f: DictInitFn = std::mem::transmute(objc_msgSend as *const c_void);
        let model_url_cfg: ModelWithUrlConfigFn = std::mem::transmute(objc_msgSend as *const c_void);
        let set_cu: SetComputeUnitsFn = std::mem::transmute(objc_msgSend as *const c_void);
        let get_cu: GetComputeUnitsFn = std::mem::transmute(objc_msgSend as *const c_void);

        let cls_str = cls("NSString");
        let cls_url = cls("NSURL");
        let cls_model = cls("MLModel");
        let cls_config = cls("MLModelConfiguration");

        if cls_model.is_null() {
            println!("ERROR: MLModel class not loaded — CoreML.framework dlopen failed");
            return Ok(());
        }
        if cls_config.is_null() {
            println!("ERROR: MLModelConfiguration class not loaded");
            return Ok(());
        }
        println!("MLModel class:                {:?}", cls_model);
        println!("MLModelConfiguration class:   {:?}", cls_config);

        let make_nsstr = |s: &str| -> ObjcId {
            let raw = allocf(cls_str as ObjcId, sel("alloc"));
            initf(raw, sel("initWithBytes:length:encoding:"),
                  s.as_ptr(), s.len(), NSUTF8_ENCODING)
        };
        let make_url = |path: &str| -> ObjcId {
            let ns = make_nsstr(path);
            urlf(cls_url as ObjcId, sel("fileURLWithPath:"), ns)
        };

        // Try loading at each compute target
        let model_url = make_url(FS_INT8);
        for (label, cu_val) in &[
            ("ALL (auto)",      MLCOMPUTE_UNITS_ALL),
            ("CPU_AND_ANE",     MLCOMPUTE_UNITS_CPU_AND_ANE),
            ("CPU_ONLY",        MLCOMPUTE_UNITS_CPU_ONLY),
            ("CPU_AND_GPU",     MLCOMPUTE_UNITS_CPU_AND_GPU),
        ] {
            println!("\n--- compute={} ({}) ---", label, cu_val);
            let cfg_raw = allocf(cls_config as ObjcId, sel("alloc"));
            let cfg = dict_init_f(cfg_raw, sel("init"));
            set_cu(cfg, sel("setComputeUnits:"), *cu_val);
            let actual = get_cu(cfg, sel("computeUnits"));
            println!("  cfg.computeUnits = {}", actual);

            let mut err: ObjcId = std::ptr::null_mut();
            let t0 = std::time::Instant::now();
            let model = model_url_cfg(
                cls_model as ObjcId,
                sel("modelWithContentsOfURL:configuration:error:"),
                model_url, cfg, &mut err);
            let dt = t0.elapsed();

            if !model.is_null() {
                println!("  loaded in {:?}", dt);
                // Dump model description
                type DescFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
                let descf: DescFn = std::mem::transmute(objc_msgSend as *const c_void);
                let desc = descf(model, sel("modelDescription"));
                if !desc.is_null() {
                    let dstr = descf(desc, sel("description"));
                    let s = nsstring_to_str(dstr);
                    let head = if s.len() > 400 { &s[..400] } else { &s[..] };
                    println!("  desc: {}", head);
                }
            } else {
                let e = nserror_string(err).unwrap_or_default();
                let s = if e.len() > 600 { &e[..600] } else { &e[..] };
                println!("  FAIL ({:?}): {}", dt, s);
            }
        }
    }

    Ok(())
}

unsafe fn nsstring_to_str(obj: ObjcId) -> String {
    if obj.is_null() { return "(null)".to_string(); }
    type Utf8Fn = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const std::ffi::c_char;
    let uf: Utf8Fn = std::mem::transmute(objc_msgSend as *const c_void);
    let p = uf(obj, sel("UTF8String"));
    if p.is_null() { return "(null utf8)".to_string(); }
    CStr::from_ptr(p).to_string_lossy().into_owned()
}
