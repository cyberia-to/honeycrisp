//! Sustained-throughput measurement on Apple's text_md7 transformer encoder.
//!
//! Real transformer: 12 attention blocks, 64 MB palettized weights,
//! 144 softmaxes, 288 einsums, 25 LayerNorms, ~32M effective params.
//!
//! Measures ANE vs CPU per-forward throughput → projects to Llama-class tok/s.
//!
//! Run: cargo run -p rane --example text_md7_throughput --release

use rane::ffi::*;
use std::ffi::{c_void, CStr, CString};

const NSUTF8_ENCODING: u64 = 4;
const MLCU_ALL: i64 = 0;
const MLCU_CPU_ONLY: i64 = 2;
const MLCU_CPU_AND_ANE: i64 = 3;
const MLAT_FLOAT16: i64 = 0x10010;

const MODEL: &str = "/System/Library/PrivateFrameworks/CoreSceneUnderstanding.framework/Versions/A/Resources/SystemSearch/v7.0.0/text_md7_6bit_ctx_512_77.mlmodelc";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== text_md7 transformer throughput: CPU vs ANE ===\n");
    println!("Model: 12-layer transformer, 64 MB palettized weights, ~32M params\n");
    println!("Inputs: token_embed fp16 [1, 512, 256], indices fp16 [1]");
    println!("Outputs: spatial_embed [1, 512, 768], hidden [1, 768], text [1, 512]\n");

    unsafe {
        dlopen(CString::new("/System/Library/Frameworks/CoreML.framework/CoreML").unwrap().as_ptr(), RTLD_NOW | 0x8);

        type AllocFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type InitBytesFn = unsafe extern "C" fn(ObjcId, ObjcSel, *const u8, usize, u64) -> ObjcId;
        type UrlFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type DictInitFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type ModelUrlCfgFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, *mut ObjcId) -> ObjcId;
        type SetCuFn = unsafe extern "C" fn(ObjcId, ObjcSel, i64);
        type PredFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, *mut ObjcId) -> ObjcId;
        type MaInitFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, i64, *mut ObjcId) -> ObjcId;
        type ArrFn = unsafe extern "C" fn(ObjcId, ObjcSel, *const ObjcId, u64) -> ObjcId;
        type NumLLFn = unsafe extern "C" fn(ObjcId, ObjcSel, i64) -> ObjcId;
        type DataPtrFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> *mut c_void;
        type DictWithFn = unsafe extern "C" fn(ObjcId, ObjcSel, *const ObjcId, *const ObjcId, u64) -> ObjcId;
        type FpDictInitFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, *mut ObjcId) -> ObjcId;

        let allocf: AllocFn = std::mem::transmute(objc_msgSend as *const c_void);
        let initf: InitBytesFn = std::mem::transmute(objc_msgSend as *const c_void);
        let urlf: UrlFn = std::mem::transmute(objc_msgSend as *const c_void);
        let dict_init_f: DictInitFn = std::mem::transmute(objc_msgSend as *const c_void);
        let model_uc: ModelUrlCfgFn = std::mem::transmute(objc_msgSend as *const c_void);
        let set_cu: SetCuFn = std::mem::transmute(objc_msgSend as *const c_void);
        let predf: PredFn = std::mem::transmute(objc_msgSend as *const c_void);
        let ma_init: MaInitFn = std::mem::transmute(objc_msgSend as *const c_void);
        let arr_with: ArrFn = std::mem::transmute(objc_msgSend as *const c_void);
        let num_init: NumLLFn = std::mem::transmute(objc_msgSend as *const c_void);
        let data_ptr: DataPtrFn = std::mem::transmute(objc_msgSend as *const c_void);
        let dict_with: DictWithFn = std::mem::transmute(objc_msgSend as *const c_void);
        let fp_dict: FpDictInitFn = std::mem::transmute(objc_msgSend as *const c_void);

        let cls_str = cls("NSString");
        let cls_url = cls("NSURL");
        let cls_arr = cls("NSArray");
        let cls_num = cls("NSNumber");
        let cls_dict = cls("NSDictionary");
        let cls_model = cls("MLModel");
        let cls_cfg = cls("MLModelConfiguration");
        let cls_ma = cls("MLMultiArray");
        let cls_fp = cls("MLDictionaryFeatureProvider");

        let make_nsstr = |s: &str| -> ObjcId {
            let raw = allocf(cls_str as ObjcId, sel("alloc"));
            initf(raw, sel("initWithBytes:length:encoding:"), s.as_ptr(), s.len(), NSUTF8_ENCODING)
        };
        let make_url = |p: &str| -> ObjcId {
            urlf(cls_url as ObjcId, sel("fileURLWithPath:"), make_nsstr(p))
        };
        let make_num = |v: i64| -> ObjcId {
            num_init(cls_num as ObjcId, sel("numberWithLongLong:"), v)
        };

        // Build input MLMultiArray: token_embed [1, 512, 256] fp16 (main_ctx_512 function)
        let shape_objs = [make_num(1), make_num(512), make_num(256)];
        let shape = arr_with(cls_arr as ObjcId, sel("arrayWithObjects:count:"), shape_objs.as_ptr(), 3);
        let mut e: ObjcId = std::ptr::null_mut();
        let token_embed = ma_init(allocf(cls_ma as ObjcId, sel("alloc")), sel("initWithShape:dataType:error:"),
                                  shape, MLAT_FLOAT16, &mut e);
        let dp = data_ptr(token_embed, sel("dataPointer"));
        let buf = std::slice::from_raw_parts_mut(dp as *mut u16, 512 * 256);
        // Fill with small fp16 values (~0.1 random)
        let mut s: u64 = 0x1234567890abcdef;
        for x in buf.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *x = (((s >> 48) as u16) & 0x33FF) | 0x3000;
        }

        // indices: [1] fp16 = 0
        let one_shape = [make_num(1)];
        let oshape = arr_with(cls_arr as ObjcId, sel("arrayWithObjects:count:"), one_shape.as_ptr(), 1);
        let indices = ma_init(allocf(cls_ma as ObjcId, sel("alloc")), sel("initWithShape:dataType:error:"),
                              oshape, MLAT_FLOAT16, &mut e);
        let ip = data_ptr(indices, sel("dataPointer"));
        *(ip as *mut u16) = 0x3000; // ~0.1

        // Build feature provider with both inputs
        let key1 = make_nsstr("token_embed");
        let key2 = make_nsstr("indices");
        let objs = [token_embed, indices];
        let keys = [key1, key2];
        let in_dict = dict_with(cls_dict as ObjcId, sel("dictionaryWithObjects:forKeys:count:"),
                                objs.as_ptr(), keys.as_ptr(), 2);
        let mut fe: ObjcId = std::ptr::null_mut();
        let fp_in = fp_dict(allocf(cls_fp as ObjcId, sel("alloc")), sel("initWithDictionary:error:"), in_dict, &mut fe);
        if fp_in.is_null() {
            println!("FP init failed: {}", nserror_string(fe).unwrap_or_default());
            return Ok(());
        }

        let model_url = make_url(MODEL);

        // Per Pearl paper Section 7: weight matrices in inference are public,
        // so weight hash can be preprocessed. Measure inference rate only.
        // Each forward = ~32M params × 2 ops/param ≈ 64 MOps + attention overhead.
        // Rough estimate per forward: ~100 MOps (mostly attention is what dominates here)
        let est_ops_per_forward: f64 = 100e6;

        for &(label, cu) in &[
            ("CPU_ONLY    ", MLCU_CPU_ONLY),
            ("CPU_AND_ANE ", MLCU_CPU_AND_ANE),
            ("ALL         ", MLCU_ALL),
        ] {
            let cfg_raw = allocf(cls_cfg as ObjcId, sel("alloc"));
            let cfg = dict_init_f(cfg_raw, sel("init"));
            set_cu(cfg, sel("setComputeUnits:"), cu);

            let mut me: ObjcId = std::ptr::null_mut();
            let t_load = std::time::Instant::now();
            let model = model_uc(cls_model as ObjcId, sel("modelWithContentsOfURL:configuration:error:"),
                                  model_url, cfg, &mut me);
            let load_dt = t_load.elapsed();
            if model.is_null() {
                let s = nserror_string(me).unwrap_or_default();
                let head = if s.len() > 200 { &s[..200] } else { &s[..] };
                println!("[{label}] LOAD FAIL ({:?}): {}", load_dt, head);
                continue;
            }

            // Warmup
            let mut warmup_err: ObjcId = std::ptr::null_mut();
            let warm = predf(model, sel("predictionFromFeatures:error:"), fp_in, &mut warmup_err);
            if warm.is_null() {
                let s = nserror_string(warmup_err).unwrap_or_default();
                let head = if s.len() > 200 { &s[..200] } else { &s[..] };
                println!("[{label}] WARMUP FAIL: {}", head);
                continue;
            }

            // Sustained measurement
            let n_iter = 200;
            let t = std::time::Instant::now();
            for _ in 0..n_iter {
                let mut perr: ObjcId = std::ptr::null_mut();
                let _ = predf(model, sel("predictionFromFeatures:error:"), fp_in, &mut perr);
            }
            let dt = t.elapsed();
            let per = dt / n_iter;
            let per_sec = 1.0 / per.as_secs_f64();
            let gops = (per_sec * est_ops_per_forward) / 1e9;
            println!("[{label}] load={:>5.1}ms  per_inf={:>7.2}ms  rate={:>8.2}/s  ≈{:>5.2} GOps/s (rough)",
                     load_dt.as_secs_f64() * 1000.0,
                     per.as_secs_f64() * 1000.0,
                     per_sec, gops);
        }

        println!("\nLlama-3.1-8B extrapolation:");
        println!("  8B int8 params → ~16 GOps per token (2× params per forward)");
        println!("  If ANE sustains R inferences/s on text_md7 (~100 MOps each):");
        println!("  Total ANE TOPS ≈ R × 100 MOps");
        println!("  Llama tok/s ≈ ANE TOPS × 1e9 / 16e9 = ANE_TOPS / 16");
    }

    Ok(())
}
