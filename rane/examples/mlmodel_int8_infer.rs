//! Inference timing for int8 MIL model: CPU_ONLY vs CPU_AND_ANE.
//! Big delta = ANE actually runs the model.
//!
//! Run: cargo run -p rane --example mlmodel_int8_infer --release

use rane::ffi::*;
use std::ffi::{c_void, CStr, CString};

const NSUTF8_ENCODING: u64 = 4;
const MLCOMPUTE_UNITS_ALL: i64 = 0;
const MLCOMPUTE_UNITS_CPU_ONLY: i64 = 2;
const MLCOMPUTE_UNITS_CPU_AND_ANE: i64 = 3;

// MLMultiArrayDataType
const MLAT_FLOAT16: i64 = 0x10010; // = 65552
const MLAT_FLOAT32: i64 = 0x10020;

const MODEL: &str = "/System/Library/PrivateFrameworks/VoiceActions.framework/Versions/A/Resources/aa_encoder_125141826.mlmodelc";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== int8 MIL inference: CPU vs ANE timing ===\n");
    println!("model: aa_encoder_125141826.mlmodelc");
    println!("       (constexpr_affine_dequantize int8→fp16 → matmul)");
    println!("input: fp16 [198, 40]   output: fp16 [1, 48, 144]\n");

    unsafe { dlopen(CString::new("/System/Library/Frameworks/CoreML.framework/CoreML").unwrap().as_ptr(), RTLD_NOW | 0x8); }

    unsafe {
        type AllocFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type InitBytesFn = unsafe extern "C" fn(ObjcId, ObjcSel, *const u8, usize, u64) -> ObjcId;
        type UrlFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type DictInitFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type ModelWithUrlConfigFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, *mut ObjcId) -> ObjcId;
        type SetCuFn = unsafe extern "C" fn(ObjcId, ObjcSel, i64);
        type PredictFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, *mut ObjcId) -> ObjcId;
        type InitMaShapeFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, i64, *mut ObjcId) -> ObjcId;
        type ArrayWithObjsFn = unsafe extern "C" fn(ObjcId, ObjcSel, *const ObjcId, u64) -> ObjcId;
        type NumberWithIntFn = unsafe extern "C" fn(ObjcId, ObjcSel, i64) -> ObjcId;
        type FpInitFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, *mut ObjcId) -> ObjcId;
        type DataPtrFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> *mut c_void;

        let allocf: AllocFn = std::mem::transmute(objc_msgSend as *const c_void);
        let initf: InitBytesFn = std::mem::transmute(objc_msgSend as *const c_void);
        let urlf: UrlFn = std::mem::transmute(objc_msgSend as *const c_void);
        let dict_init_f: DictInitFn = std::mem::transmute(objc_msgSend as *const c_void);
        let model_url_cfg: ModelWithUrlConfigFn = std::mem::transmute(objc_msgSend as *const c_void);
        let set_cu: SetCuFn = std::mem::transmute(objc_msgSend as *const c_void);
        let predf: PredictFn = std::mem::transmute(objc_msgSend as *const c_void);
        let ma_init: InitMaShapeFn = std::mem::transmute(objc_msgSend as *const c_void);
        let arr_with: ArrayWithObjsFn = std::mem::transmute(objc_msgSend as *const c_void);
        let num_init: NumberWithIntFn = std::mem::transmute(objc_msgSend as *const c_void);
        let fp_init: FpInitFn = std::mem::transmute(objc_msgSend as *const c_void);
        let data_ptr: DataPtrFn = std::mem::transmute(objc_msgSend as *const c_void);

        let cls_str = cls("NSString");
        let cls_url = cls("NSURL");
        let cls_arr = cls("NSArray");
        let cls_num = cls("NSNumber");
        let cls_model = cls("MLModel");
        let cls_config = cls("MLModelConfiguration");
        let cls_ma = cls("MLMultiArray");
        let cls_fp = cls("MLDictionaryFeatureProvider");

        if cls_ma.is_null() || cls_fp.is_null() {
            println!("ERROR: MLMultiArray or MLDictionaryFeatureProvider class missing");
            return Ok(());
        }

        let make_nsstr = |s: &str| -> ObjcId {
            let raw = allocf(cls_str as ObjcId, sel("alloc"));
            initf(raw, sel("initWithBytes:length:encoding:"), s.as_ptr(), s.len(), NSUTF8_ENCODING)
        };
        let make_url = |path: &str| -> ObjcId {
            let ns = make_nsstr(path);
            urlf(cls_url as ObjcId, sel("fileURLWithPath:"), ns)
        };
        let make_num = |v: i64| -> ObjcId {
            num_init(cls_num as ObjcId, sel("numberWithLongLong:"), v)
        };

        // Build shape NSArray [198, 40]
        let s198 = make_num(198);
        let s40 = make_num(40);
        let shape_objs = [s198, s40];
        let shape_arr = arr_with(cls_arr as ObjcId, sel("arrayWithObjects:count:"), shape_objs.as_ptr(), 2);

        // Build MLMultiArray
        let mut err: ObjcId = std::ptr::null_mut();
        let ma_raw = allocf(cls_ma as ObjcId, sel("alloc"));
        let ma = ma_init(ma_raw, sel("initWithShape:dataType:error:"),
                         shape_arr, MLAT_FLOAT16, &mut err);
        if ma.is_null() {
            println!("MLMultiArray init failed: {}", nserror_string(err).unwrap_or_default());
            return Ok(());
        }
        // Fill with random fp16 noise
        let dp = data_ptr(ma, sel("dataPointer"));
        let n = 198 * 40;
        let buf = std::slice::from_raw_parts_mut(dp as *mut u16, n);
        let mut state: u64 = 0xdeadbeefcafef00d;
        for x in buf.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // small fp16 normal-ish value
            *x = (((state >> 48) as u16) & 0x3BFF) | 0x3800; // ~[-1..1]
        }
        println!("input MLMultiArray ready: fp16 [198, 40] = {n} elements\n");

        // Build feature provider dict {"input_wav": MLMultiArray}
        // MLDictionaryFeatureProvider initWithDictionary:error:
        type FpDictInitFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, *mut ObjcId) -> ObjcId;
        let fp_dict: FpDictInitFn = std::mem::transmute(objc_msgSend as *const c_void);
        // Build NSDictionary
        let cls_dict = cls("NSDictionary");
        type DictWithObjsKeysFn = unsafe extern "C" fn(ObjcId, ObjcSel, *const ObjcId, *const ObjcId, u64) -> ObjcId;
        let dict_with: DictWithObjsKeysFn = std::mem::transmute(objc_msgSend as *const c_void);
        let key = make_nsstr("input_wav");
        let objs = [ma];
        let keys = [key];
        let in_dict = dict_with(cls_dict as ObjcId, sel("dictionaryWithObjects:forKeys:count:"),
                                objs.as_ptr(), keys.as_ptr(), 1);
        let fp_raw = allocf(cls_fp as ObjcId, sel("alloc"));
        let mut fp_err: ObjcId = std::ptr::null_mut();
        let fp_in = fp_dict(fp_raw, sel("initWithDictionary:error:"), in_dict, &mut fp_err);
        if fp_in.is_null() {
            println!("Feature provider init failed: {}", nserror_string(fp_err).unwrap_or_default());
            return Ok(());
        }

        let _ = fp_init; // unused

        // For each compute target
        let model_url = make_url(MODEL);
        for (label, cu) in &[
            ("CPU_ONLY",     MLCOMPUTE_UNITS_CPU_ONLY),
            ("CPU_AND_ANE",  MLCOMPUTE_UNITS_CPU_AND_ANE),
            ("ALL",          MLCOMPUTE_UNITS_ALL),
        ] {
            let cfg_raw = allocf(cls_config as ObjcId, sel("alloc"));
            let cfg = dict_init_f(cfg_raw, sel("init"));
            set_cu(cfg, sel("setComputeUnits:"), *cu);
            let mut e: ObjcId = std::ptr::null_mut();
            let m = model_url_cfg(cls_model as ObjcId,
                sel("modelWithContentsOfURL:configuration:error:"),
                model_url, cfg, &mut e);
            if m.is_null() {
                println!("[{}] load failed: {}", label, nserror_string(e).unwrap_or_default());
                continue;
            }
            // Warmup
            let mut werr: ObjcId = std::ptr::null_mut();
            let _ = predf(m, sel("predictionFromFeatures:error:"), fp_in, &mut werr);
            // Time N iterations
            let n_iter = 50;
            let t0 = std::time::Instant::now();
            for _ in 0..n_iter {
                let mut perr: ObjcId = std::ptr::null_mut();
                let out = predf(m, sel("predictionFromFeatures:error:"), fp_in, &mut perr);
                if out.is_null() {
                    println!("[{}] predict failed: {}", label, nserror_string(perr).unwrap_or_default());
                    break;
                }
            }
            let dt = t0.elapsed();
            let per = dt / n_iter;
            println!("[{:<13}] {:?} per inference, {} iters in {:?}", label, per, n_iter, dt);
        }
    }

    Ok(())
}
