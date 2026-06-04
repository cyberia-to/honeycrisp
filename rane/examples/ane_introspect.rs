//! Deep ANE reverse-engineering — find the bytecode buffer in _ANEInMemoryModel.
//!
//! Strategy:
//!   1. Compile a matmul → `_ANEInMemoryModel` ObjC object
//!   2. Walk the class hierarchy & enumerate ALL ivars
//!   3. For each ObjC ivar that is NSData (or has data bytes), dump it
//!   4. Try sending common selectors that might return data: `bytecode`, `compiledData`,
//!      `programData`, `binary`, `kernels`, etc.
//!   5. Also enumerate ALL methods on the class to find data-returning methods
//!   6. Dump everything to /tmp for offline binary diff
//!
//! Goal: locate the ANE bytecode in memory so we can patch the dtype field directly.
//!
//! Run: cargo run -p rane --example ane_introspect --release

use rane::ffi::*;
use rane::mil;
use std::ffi::{c_char, c_void, CStr};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("ANE deep introspection — finding the bytecode\n");
    introspect_compile()?;
    Ok(())
}

fn introspect_compile() -> Result<(), Box<dyn std::error::Error>> {
    // Load frameworks first
    for name in &["AppleNeuralEngine", "ANECompiler", "ANEServices"] {
        let path = format!("/System/Library/PrivateFrameworks/{name}.framework/{name}");
        let c = std::ffi::CString::new(path).unwrap();
        unsafe {
            dlopen(c.as_ptr(), RTLD_NOW);
        }
    }

    // Compile matmul to get a model pointer
    let mil_text = mil::matmul(64, 64, 64).text;
    let model_obj = unsafe { compile_and_get_model(&mil_text)? };

    println!("\n=== Class hierarchy of _ANEInMemoryModel ===");
    let cls = unsafe { object_getClass(model_obj) };
    walk_class_hierarchy(cls);

    println!("\n=== Enumerating ALL ivars (including superclasses) ===");
    enumerate_ivars(model_obj);

    println!("\n=== Methods on _ANEInMemoryModel ===");
    enumerate_methods(cls);

    // ── KEY DISCOVERY: _model ivar holds _ANEModel, which has the bytecode ──
    println!("\n=== Following _model (_ANEModel) ivar ===");
    let ane_model = unsafe { read_object_ivar(model_obj, "_model") };
    if !ane_model.is_null() {
        println!("  _ANEModel @ {ane_model:p}");
        let ane_model_cls = unsafe { object_getClass(ane_model) };
        println!("  Class hierarchy:");
        walk_class_hierarchy(ane_model_cls);
        println!("  ivars:");
        enumerate_ivars(ane_model);
        println!("  methods:");
        enumerate_methods(ane_model_cls);
    }

    // ── Try modelURL path — maybe the bytecode is in a file ──
    println!("\n=== _modelURL contents ===");
    let url = unsafe { read_object_ivar(model_obj, "_modelURL") };
    if !url.is_null() {
        unsafe {
            dump_nsurl(url);
        }
    }

    println!("\n=== _sharedConnection (_ANEClient) ===");
    let conn = unsafe { read_object_ivar(model_obj, "_sharedConnection") };
    if !conn.is_null() {
        let conn_cls = unsafe { object_getClass(conn) };
        println!("  Methods:");
        enumerate_methods(conn_cls);
    }

    // ── KEY: Load the model and see what new state appears ──
    println!("\n=== LOADING MODEL via loadWithQoS: ===");
    let cls_dict = rane::ffi::cls("NSDictionary");
    unsafe {
        type DictFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type LoadFn = unsafe extern "C" fn(ObjcId, ObjcSel, u32, ObjcId, *mut ObjcId) -> bool;
        let dictf: DictFn = std::mem::transmute(objc_msgSend as *const c_void);
        let loadf: LoadFn = std::mem::transmute(objc_msgSend as *const c_void);
        let empty = dictf(cls_dict as ObjcId, sel("dictionary"));
        let mut err: ObjcId = std::ptr::null_mut();
        let ok = loadf(
            model_obj,
            sel("loadWithQoS:options:error:"),
            21,
            empty,
            &mut err,
        );
        if !ok {
            println!("  load FAILED");
        } else {
            println!("  loaded OK\n");
            println!("=== Post-load ivar state ===");
            enumerate_ivars(model_obj);

            // Follow _model._program (_ANEProgramForEvaluation)
            let am = read_object_ivar(model_obj, "_model");
            if !am.is_null() {
                let prog = read_object_ivar(am, "_program");
                if !prog.is_null() {
                    println!("\n=== _ANEProgramForEvaluation @ {prog:p} ===");
                    let pc = object_getClass(prog);
                    println!("  Class hierarchy:");
                    walk_class_hierarchy(pc);
                    println!("  ivars:");
                    enumerate_ivars(prog);
                    println!("  methods:");
                    enumerate_methods(pc);
                }
            }

            // Try the outer model's _program too
            let prog2 = read_object_ivar(model_obj, "_program");
            if !prog2.is_null() {
                println!("\n=== Outer _program @ {prog2:p} ===");
                let pc = object_getClass(prog2);
                enumerate_ivars(prog2);
                enumerate_methods(pc);
            }
        }
    }

    Ok(())
}

unsafe fn read_object_ivar(obj: ObjcId, name: &str) -> ObjcId {
    let cls = object_getClass(obj);
    let mut c = cls;
    while !c.is_null() {
        let mut count: u32 = 0;
        let list = class_copyIvarList(c, &mut count);
        if !list.is_null() {
            for i in 0..count {
                let ivar = *list.add(i as usize);
                let n_ptr = ivar_getName(ivar);
                if !n_ptr.is_null() {
                    let n = CStr::from_ptr(n_ptr).to_string_lossy();
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

unsafe fn dump_nsurl(url: ObjcId) {
    type StrF = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
    type Utf8F = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const c_char;
    let sf: StrF = std::mem::transmute(objc_msgSend as *const c_void);
    let uf: Utf8F = std::mem::transmute(objc_msgSend as *const c_void);

    // absoluteString
    let abs = sf(url, sel("absoluteString"));
    if !abs.is_null() {
        let cstr = uf(abs, sel("UTF8String"));
        if !cstr.is_null() {
            println!(
                "  absoluteString: {}",
                CStr::from_ptr(cstr).to_string_lossy()
            );
        }
    }
    // path
    let p = sf(url, sel("path"));
    if !p.is_null() {
        let cstr = uf(p, sel("UTF8String"));
        if !cstr.is_null() {
            let path = CStr::from_ptr(cstr).to_string_lossy().into_owned();
            println!("  path: {path}");
            // If it's a real file, dump it
            if let Ok(meta) = std::fs::metadata(&path) {
                println!("    size: {} bytes, is_dir: {}", meta.len(), meta.is_dir());
                if meta.is_dir() {
                    if let Ok(entries) = std::fs::read_dir(&path) {
                        for e in entries.flatten() {
                            let name = e.file_name();
                            let sz = e.metadata().map(|m| m.len()).unwrap_or(0);
                            println!("      {name:?}: {sz} bytes");
                        }
                    }
                } else if meta.len() < 16 * 1024 * 1024 {
                    if let Ok(data) = std::fs::read(&path) {
                        let dst = format!("/tmp/ane_intro_modelURL.bin");
                        let _ = std::fs::write(&dst, &data);
                        println!("    saved {} bytes to {dst}", data.len());
                        let n = data.len().min(64);
                        print!("    head: ");
                        for &b in &data[..n] {
                            print!("{:02x} ", b);
                        }
                        println!();
                    }
                }
            }
        }
    }
}

unsafe fn compile_and_get_model(mil_text: &str) -> Result<ObjcId, Box<dyn std::error::Error>> {
    type DataFn = unsafe extern "C" fn(ObjcId, ObjcSel, *const u8, u64) -> ObjcId;
    type DictFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
    type DescFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, ObjcId) -> ObjcId;
    type ModelFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
    type CompileFn = unsafe extern "C" fn(ObjcId, ObjcSel, u32, ObjcId, *mut ObjcId) -> bool;

    let cls_data = cls("NSData");
    let cls_dict = cls("NSDictionary");
    let cls_desc = cls("_ANEInMemoryModelDescriptor");
    let cls_model = cls("_ANEInMemoryModel");

    let dataf: DataFn = std::mem::transmute(objc_msgSend as *const c_void);
    let dictf: DictFn = std::mem::transmute(objc_msgSend as *const c_void);
    let descf: DescFn = std::mem::transmute(objc_msgSend as *const c_void);
    let modelf: ModelFn = std::mem::transmute(objc_msgSend as *const c_void);
    let compilef: CompileFn = std::mem::transmute(objc_msgSend as *const c_void);

    let bytes = mil_text.as_bytes();
    let nsdata = dataf(
        cls_data as ObjcId,
        sel("dataWithBytes:length:"),
        bytes.as_ptr(),
        bytes.len() as u64,
    );
    let empty = dictf(cls_dict as ObjcId, sel("dictionary"));
    let descriptor = descf(
        cls_desc as ObjcId,
        sel("modelWithMILText:weights:optionsPlist:"),
        nsdata,
        empty,
        std::ptr::null_mut(),
    );
    let model = modelf(
        cls_model as ObjcId,
        sel("inMemoryModelWithDescriptor:"),
        descriptor,
    );

    // Set up tmp dir
    type StrF = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
    type Utf8F = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const c_char;
    let sf: StrF = std::mem::transmute(objc_msgSend as *const c_void);
    let uf: Utf8F = std::mem::transmute(objc_msgSend as *const c_void);
    let hex_id = sf(model, sel("hexStringIdentifier"));
    let hex_str = {
        let cstr = uf(hex_id, sel("UTF8String"));
        CStr::from_ptr(cstr).to_string_lossy().into_owned()
    };
    let tmp_dir = std::env::temp_dir().join(&hex_str);
    let _ = std::fs::create_dir_all(tmp_dir.join("weights"));
    let _ = std::fs::write(tmp_dir.join("model.mil"), mil_text);

    let mut err: ObjcId = std::ptr::null_mut();
    let ok = compilef(
        model,
        sel("compileWithQoS:options:error:"),
        21,
        empty,
        &mut err,
    );
    if !ok {
        return Err("compile failed".into());
    }
    Ok(model)
}

fn walk_class_hierarchy(cls: ObjcClass) {
    let mut c = cls;
    while !c.is_null() {
        unsafe {
            let name_ptr = class_getName(c);
            let name = if !name_ptr.is_null() {
                CStr::from_ptr(name_ptr).to_string_lossy().into_owned()
            } else {
                "(null)".into()
            };
            println!("  {:?} → {name}", c);
            c = class_getSuperclass(c);
        }
    }
}

fn enumerate_ivars(obj: ObjcId) {
    unsafe {
        let mut c = object_getClass(obj);
        while !c.is_null() {
            let cls_name = if !class_getName(c).is_null() {
                CStr::from_ptr(class_getName(c))
                    .to_string_lossy()
                    .into_owned()
            } else {
                "?".into()
            };

            let mut count: u32 = 0;
            let list = class_copyIvarList(c, &mut count);
            if !list.is_null() && count > 0 {
                println!("  --- {cls_name} ({count} ivars) ---");
                for i in 0..count {
                    let ivar = *list.add(i as usize);
                    let name_ptr = ivar_getName(ivar);
                    let type_ptr = ivar_getTypeEncoding(ivar);
                    let name = if !name_ptr.is_null() {
                        CStr::from_ptr(name_ptr).to_string_lossy().into_owned()
                    } else {
                        "?".into()
                    };
                    let ty = if !type_ptr.is_null() {
                        CStr::from_ptr(type_ptr).to_string_lossy().into_owned()
                    } else {
                        "?".into()
                    };
                    let off = ivar_getOffset(ivar);

                    // Get ivar value
                    let val = object_getIvar(obj, ivar);
                    print!("    [{off:04x}] {name}: {ty} = {val:p}");

                    // If object pointer, try to get class & some info
                    if ty.starts_with('@') && !val.is_null() {
                        let val_cls = object_getClass(val);
                        if !val_cls.is_null() {
                            let vc_name_p = class_getName(val_cls);
                            if !vc_name_p.is_null() {
                                let vc_name = CStr::from_ptr(vc_name_p).to_string_lossy();
                                print!("  [{vc_name}]");
                                // If NSData-like, dump it
                                if vc_name.contains("Data") || vc_name.contains("Buffer") {
                                    dump_nsdata(val, &format!("{cls_name}_{name}"));
                                }
                            }
                        }
                    }
                    println!();
                }
                libc_free(list as *mut c_void);
            }
            c = class_getSuperclass(c);
        }
    }
}

fn enumerate_methods(cls: ObjcClass) {
    unsafe {
        let mut count: u32 = 0;
        let list = class_copyMethodList(cls, &mut count);
        if !list.is_null() && count > 0 {
            println!("  Total methods on class: {count}");
            let mut names: Vec<String> = Vec::new();
            for i in 0..count {
                let method = *list.add(i as usize);
                let s = method_getName(method);
                if !s.is_null() {
                    let n_ptr = sel_getName(s);
                    if !n_ptr.is_null() {
                        let n = CStr::from_ptr(n_ptr).to_string_lossy().into_owned();
                        names.push(n);
                    }
                }
            }
            names.sort();
            // Show methods that look interesting
            for n in &names {
                let lower = n.to_lowercase();
                if lower.contains("data")
                    || lower.contains("byte")
                    || lower.contains("compile")
                    || lower.contains("kernel")
                    || lower.contains("file")
                    || lower.contains("url")
                    || lower.contains("path")
                    || lower.contains("binary")
                    || lower.contains("blob")
                    || lower.contains("size")
                    || lower.contains("length")
                {
                    println!("    {n}");
                }
            }
            libc_free(list as *mut c_void);
        }
    }
}

fn dump_nsdata(data: ObjcId, label: &str) {
    unsafe {
        type LenFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> u64;
        type BytesFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const u8;
        let lf: LenFn = std::mem::transmute(objc_msgSend as *const c_void);
        let bf: BytesFn = std::mem::transmute(objc_msgSend as *const c_void);
        let len = lf(data, sel("length"));
        let ptr = bf(data, sel("bytes"));
        if ptr.is_null() {
            println!("    bytes: <null>");
            return;
        }
        println!("    NSData length={len}");
        let slice = std::slice::from_raw_parts(ptr, len as usize);
        // Save to /tmp
        let safe_label = label.replace(['/', ' ', ':'], "_");
        let path = format!("/tmp/ane_intro_{safe_label}.bin");
        let _ = std::fs::write(&path, slice);
        println!("    saved to {path}");
        // Print first 64 hex bytes
        let n = slice.len().min(64);
        print!("    head: ");
        for &b in &slice[..n] {
            print!("{:02x} ", b);
        }
        println!();
    }
}

extern "C" {
    fn free(p: *mut c_void);
}
fn libc_free(p: *mut c_void) {
    unsafe {
        free(p);
    }
}
