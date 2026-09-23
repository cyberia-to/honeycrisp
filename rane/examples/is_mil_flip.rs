//! Flip _isMILModel=false then compile constexpr_affine_dequantize.
//!
//! Avoids calling isMILModel getter (which throws ObjC exception when inconsistent).
//! Direct memory patch: _isMILModel at ivar offset 8.
//! Then probe what compileWithQoS: does when isMILModel=false.
//!
//! Run: cargo run -p rane --example is_mil_flip --release

use rane::ffi::*;
use std::ffi::{c_void, CStr, CString};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== _isMILModel flip probe ===\n");

    for name in &["AppleNeuralEngine", "ANECompiler", "ANEServices"] {
        let path = format!("/System/Library/PrivateFrameworks/{name}.framework/{name}");
        let c = CString::new(path).unwrap();
        unsafe {
            dlopen(c.as_ptr(), RTLD_NOW);
        }
    }

    // Build weight blob
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

    let dir = "/tmp/is_mil_flip";
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

    unsafe {
        type DataFn = unsafe extern "C" fn(ObjcId, ObjcSel, *const u8, u64) -> ObjcId;
        type DictFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type DescFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, ObjcId) -> ObjcId;
        type ModelFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type StrFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type Utf8Fn = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const std::ffi::c_char;
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
        let compilef: CompileFn = std::mem::transmute(objc_msgSend as *const c_void);

        let bytes = constexpr_mil.as_bytes();
        let ns_text = dataf(
            cls_data as ObjcId,
            sel("dataWithBytes:length:"),
            bytes.as_ptr(),
            bytes.len() as u64,
        );
        // Weights arg = empty NSDictionary (BLOBFILE refs are in MIL text)
        let empty = dictf(cls_dict as ObjcId, sel("dictionary"));

        // Create descriptor with isMILModel=YES
        let desc = descf(
            cls_desc as ObjcId,
            sel("modelWithMILText:weights:optionsPlist:"),
            ns_text,
            empty, // weights = empty dict
            std::ptr::null_mut(),
        );
        if desc.is_null() {
            println!("descriptor = null");
            return Ok(());
        }

        // Read _isMILModel via direct memory access (offset 8, 1 byte)
        let is_mil_ptr = (desc as *const u8).add(8);
        let before = *is_mil_ptr;
        println!("_isMILModel before (direct): {before}");

        // ─── Test A: compile with isMILModel=true (baseline) ─────────────────
        println!("\n--- Test A: compile isMILModel=true (baseline, expect reject) ---");
        let model_a = modelf(
            cls_model as ObjcId,
            sel("inMemoryModelWithDescriptor:"),
            desc,
        );
        let hex_a = get_hex(model_a, &strf, &utf8f);
        let tmp_a = setup_tmp_dir(&hex_a, dir);
        let mut err_a: ObjcId = std::ptr::null_mut();
        let ok_a = compilef(
            model_a,
            sel("compileWithQoS:options:error:"),
            21,
            empty,
            &mut err_a,
        );
        if ok_a {
            println!("  SUCCESS (unexpected)");
        } else {
            println!("  error: {:?}", nserror_string(err_a));
        }

        // ─── Test B: flip _isMILModel=false, compile ────────────────────────
        println!("\n--- Test B: flip _isMILModel=false, compile ---");
        // Create a fresh descriptor
        let desc2 = descf(
            cls_desc as ObjcId,
            sel("modelWithMILText:weights:optionsPlist:"),
            ns_text,
            empty,
            std::ptr::null_mut(),
        );
        if desc2.is_null() {
            println!("  desc2 = null");
            return Ok(());
        }

        // Flip _isMILModel at offset 8 to false
        let is_mil_ptr2 = (desc2 as *mut u8).add(8);
        *is_mil_ptr2 = 0;
        println!("  _isMILModel flipped: {} → {}", before, *is_mil_ptr2);

        let model_b = modelf(
            cls_model as ObjcId,
            sel("inMemoryModelWithDescriptor:"),
            desc2,
        );
        if model_b.is_null() {
            println!("  model_b = null");
            return Ok(());
        }
        println!("  model created OK");

        let hex_b = get_hex(model_b, &strf, &utf8f);
        let tmp_b = setup_tmp_dir(&hex_b, dir);
        println!("  tmp_dir: {}", tmp_b);

        let mut err_b: ObjcId = std::ptr::null_mut();
        let ok_b = compilef(
            model_b,
            sel("compileWithQoS:options:error:"),
            21,
            empty,
            &mut err_b,
        );
        if ok_b {
            println!("  *** COMPILE SUCCESS after isMILModel flip! ***");
            print_dir_tree(&tmp_b);
        } else {
            println!("  compile error: {:?}", nserror_string(err_b));
            print_dir_tree(&tmp_b);
        }

        // ─── Test C: flip + write coremldata.bin to tmp_dir ─────────────────
        println!("\n--- Test C: flip + coremldata.bin in tmp_dir ---");
        let coreml_bin = std::fs::read("/System/Library/PrivateFrameworks/VoiceActions.framework/Versions/A/Resources/aa_encoder_125141826.mlmodelc/coremldata.bin").unwrap_or_default();
        if coreml_bin.is_empty() {
            println!("  coremldata.bin not found");
        } else {
            let desc3 = descf(
                cls_desc as ObjcId,
                sel("modelWithMILText:weights:optionsPlist:"),
                ns_text,
                empty,
                std::ptr::null_mut(),
            );
            let p3 = (desc3 as *mut u8).add(8);
            *p3 = 0; // isMILModel=false

            let model_c = modelf(
                cls_model as ObjcId,
                sel("inMemoryModelWithDescriptor:"),
                desc3,
            );
            if model_c.is_null() {
                println!("  model_c = null");
                return Ok(());
            }

            let hex_c = get_hex(model_c, &strf, &utf8f);
            let tmp_c = setup_tmp_dir(&hex_c, dir);
            // Write coremldata.bin + model.mil
            let _ = std::fs::write(format!("{tmp_c}/coremldata.bin"), &coreml_bin);
            // Also copy VoiceActions weights
            let _ = std::fs::copy(
                "/System/Library/PrivateFrameworks/VoiceActions.framework/Versions/A/Resources/aa_encoder_125141826.mlmodelc/weights/weight.bin",
                format!("{tmp_c}/weights/weight.bin"),
            );
            println!("  tmp_dir: {tmp_c}");
            print_dir_tree(&tmp_c);

            let mut err_c: ObjcId = std::ptr::null_mut();
            let ok_c = compilef(
                model_c,
                sel("compileWithQoS:options:error:"),
                21,
                empty,
                &mut err_c,
            );
            if ok_c {
                println!("  *** COMPILE SUCCESS with coremldata.bin! ***");
                print_dir_tree(&tmp_c);
            } else {
                println!("  compile error: {:?}", nserror_string(err_c));
            }
        }
    }

    Ok(())
}

unsafe fn get_hex(
    model: ObjcId,
    strf: &(unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId),
    utf8f: &(unsafe extern "C" fn(ObjcId, ObjcSel) -> *const std::ffi::c_char),
) -> String {
    let hex_id = strf(model, sel("hexStringIdentifier"));
    let c = utf8f(hex_id, sel("UTF8String"));
    CStr::from_ptr(c).to_string_lossy().into_owned()
}

fn setup_tmp_dir(hex: &str, src_dir: &str) -> String {
    let tmp = format!("/tmp/{hex}");
    let _ = std::fs::create_dir_all(format!("{tmp}/weights"));
    let _ = std::fs::write(
        format!("{tmp}/model.mil"),
        std::fs::read(format!("{src_dir}/model.mil")).unwrap_or_default(),
    );
    let _ = std::fs::copy(
        format!("{src_dir}/weights/weights.bin"),
        format!("{tmp}/weights/weights.bin"),
    );
    tmp
}

fn print_dir_tree(dir: &str) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut es: Vec<_> = entries.flatten().collect();
        es.sort_by_key(|e| e.file_name());
        for e in es {
            let name = e.file_name();
            let sz = e.metadata().map(|m| m.len()).unwrap_or(0);
            let is_dir = e.metadata().map(|m| m.is_dir()).unwrap_or(false);
            if is_dir {
                println!("    {name:?}/");
                let sub = format!("{}/{}", dir, name.to_string_lossy());
                if let Ok(sub_entries) = std::fs::read_dir(&sub) {
                    let mut ses: Vec<_> = sub_entries.flatten().collect();
                    ses.sort_by_key(|e| e.file_name());
                    for se in ses {
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
