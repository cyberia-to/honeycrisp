//! Compile an Espresso/E5 model bundle → model.hwx via _ANECoreMLModelCompiler.
//! This is the working path for ANE compilation from Espresso-format bundles.
//!
//! Run: cargo run -p rane --example anec_coreml_compile --release

use rane::ffi::*;
use std::ffi::{c_void, CStr, CString};

const NSUTF8_ENCODING: u64 = 4; // NSUTF8StringEncoding

const VAD_E5: &str =
    "/System/Library/PrivateFrameworks/VoiceActions.framework/Versions/A/Resources/VAD_ANE.e5";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ANECoreMLModelCompiler: Espresso → model.hwx ===\n");

    for name in &["AppleNeuralEngine", "ANECompiler", "ANEServices"] {
        let path = format!("/System/Library/PrivateFrameworks/{name}.framework/{name}");
        let c = CString::new(path).unwrap();
        unsafe {
            dlopen(c.as_ptr(), RTLD_NOW | 0x8);
        }
    }
    let xpc_path = CString::new("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/XPCServices/ANECompilerService.xpc/Contents/MacOS/ANECompilerService").unwrap();
    unsafe {
        dlopen(xpc_path.as_ptr(), RTLD_NOW | 0x8);
    }

    let out_dir = std::env::temp_dir().join("ane_hwx_out");
    let tmp_dir = std::env::temp_dir().join("ane_hwx_tmp");
    let save_dir = std::env::temp_dir().join("ane_hwx_save");
    std::fs::create_dir_all(&out_dir)?;
    std::fs::create_dir_all(&tmp_dir)?;
    std::fs::create_dir_all(&save_dir)?;

    unsafe {
        type AllocFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type InitBytesFn = unsafe extern "C" fn(ObjcId, ObjcSel, *const u8, usize, u64) -> ObjcId;
        type DictFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type UrlFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        type CMLFn = unsafe extern "C" fn(
            ObjcId,
            ObjcSel,
            ObjcId,
            ObjcId,
            ObjcId,
            ObjcId,
            ObjcId,
            ObjcId,
            ObjcId,
            ObjcId,
            u8,
            ObjcId,
            *mut u8,
            *mut ObjcId,
        ) -> ObjcId;

        let allocf: AllocFn = std::mem::transmute(objc_msgSend as *const c_void);
        let initf: InitBytesFn = std::mem::transmute(objc_msgSend as *const c_void);
        let _dictf: DictFn = std::mem::transmute(objc_msgSend as *const c_void);
        let urlf: UrlFn = std::mem::transmute(objc_msgSend as *const c_void);
        let cmlf: CMLFn = std::mem::transmute(objc_msgSend as *const c_void);

        let cls_str = cls("NSString");
        let cls_url = cls("NSURL");
        let cls_dict = cls("NSDictionary");
        let cls_cml = cls("_ANECoreMLModelCompiler");

        // alloc+init returns retained (+1) — never goes through autorelease pool.
        // stringWithUTF8String: returns autoreleased, which crashes without a pool.
        let make_nsstr = |s: &str| -> ObjcId {
            let raw = allocf(cls_str as ObjcId, sel("alloc"));
            initf(
                raw,
                sel("initWithBytes:length:encoding:"),
                s.as_ptr(),
                s.len(),
                NSUTF8_ENCODING,
            )
        };

        let make_url = |path: &str| -> ObjcId {
            let ns = make_nsstr(path);
            urlf(cls_url as ObjcId, sel("fileURLWithPath:"), ns)
        };

        println!("_ANECoreMLModelCompiler = {:?}", cls_cml as *const _);

        let model_url = make_url(VAD_E5);
        let out_url = make_url(out_dir.to_str().unwrap());
        let tmp_url = make_url(tmp_dir.to_str().unwrap());
        // saveSourceModelPath must be a non-nil NSString — macOS 26.4.1 changed
        // [NSURL fileURLWithPath:nil] from returning nil to throwing NSInvalidArgumentException.
        let save_path = make_nsstr(save_dir.to_str().unwrap());
        type DictInitFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        let dict_init_f: DictInitFn = std::mem::transmute(objc_msgSend as *const c_void);
        let empty_dict = {
            let raw = allocf(cls_dict as ObjcId, sel("alloc"));
            dict_init_f(raw, sel("init"))
        };

        // Check what paths the compiler finds in the bundle
        type PathsFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        let pathsf: PathsFn = std::mem::transmute(objc_msgSend as *const c_void);
        let paths = pathsf(cls_cml as ObjcId, sel("pathsForModelURL:"), model_url);
        println!("pathsForModelURL: {}", nsstring_to_str(paths));

        // Compile: _ANECoreMLModelCompiler::compileModelAt:csIdentity:key:optionsFilename:
        //          tempDirectory:outputURL:saveSourceModelPath:aotModelBinaryPath:
        //          isEncryptedModel:options:ok:error:
        println!("\nCompiling VAD_ANE.e5 → model.hwx ...");
        let mut ok: u8 = 0;
        let mut err: ObjcId = std::ptr::null_mut();

        let result = cmlf(
            cls_cml as ObjcId,
            sel("compileModelAt:csIdentity:key:optionsFilename:tempDirectory:outputURL:saveSourceModelPath:aotModelBinaryPath:isEncryptedModel:options:ok:error:"),
            model_url,                              // modelAt
            make_nsstr("com.apple.voiceactions"),   // csIdentity (bundle ID)
            make_nsstr("model.espresso.net"),        // key (Espresso net filename)
            std::ptr::null_mut(),                   // optionsFilename = nil
            tmp_url,                                // tempDirectory
            out_url,                                // outputURL
            save_path,                              // saveSourceModelPath (NSString, must be non-nil)
            std::ptr::null_mut(),                   // aotModelBinaryPath = nil
            0,                                      // isEncryptedModel = NO
            empty_dict,                             // options = {}
            &mut ok,
            &mut err,
        );

        if ok != 0 {
            println!("*** COMPILE SUCCESS! ***");
            println!("result: {:?}", result as *const _);
            let mut total_bytes = 0u64;
            for entry in std::fs::read_dir(&out_dir)?.filter_map(|e| e.ok()) {
                let sz = std::fs::metadata(entry.path())
                    .ok()
                    .map(|m| m.len())
                    .unwrap_or(0);
                total_bytes += sz;
                println!("  {}: {}B", entry.file_name().to_string_lossy(), sz);
            }
            println!("  total: {}KB", total_bytes / 1024);

            // Read the hwx for further analysis
            let hwx_path = out_dir.join("model.hwx");
            if hwx_path.exists() {
                let hwx = std::fs::read(&hwx_path)?;
                println!(
                    "\nhwx[0..4] = {:02x} {:02x} {:02x} {:02x}  (magic)",
                    hwx[0], hwx[1], hwx[2], hwx[3]
                );
                println!("hwx size: {} bytes", hwx.len());
            }
        } else {
            let e = nserror_string(err).unwrap_or_default();
            println!("FAILED: {}", &e[..e.len().min(400)]);
        }

        // Clean up
        std::fs::remove_dir_all(&out_dir).ok();
        std::fs::remove_dir_all(&tmp_dir).ok();
        std::fs::remove_dir_all(&save_dir).ok();
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
