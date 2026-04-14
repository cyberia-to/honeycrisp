//! DEXT Contiguous Allocation Client
//!
//! Connects to CybMemAllocDriver DEXT, maps its contiguous buffer into
//! userspace, inspects physical layout, writes test data, and attempts
//! to create an IOSurface backed by the same memory for ANE visibility.
//!
//! This is PATH 1 of the unimem architecture:
//!   DEXT allocates contiguous physical pages
//!     -> maps to userspace via IOConnectMapMemory64
//!     -> wrap as IOSurface for ANE/GPU visibility
//!
//! If the DEXT is not installed/running, falls back to a standalone mode
//! that demonstrates what the client *would* do, using a regular IOSurface
//! allocation for comparison measurements.

#![allow(
    non_camel_case_types,
    non_upper_case_globals,
    non_snake_case,
    dead_code
)]

use std::ffi::{c_char, c_void};
use std::ptr;
use std::time::Instant;

// ────────────────────────────────────────────────────────────────────
// Type aliases
// ────────────────────────────────────────────────────────────────────

type kern_return_t = i32;
type io_object_t = u32;
type io_service_t = io_object_t;
type io_connect_t = u32;
type io_iterator_t = u32;
type mach_port_t = u32;
type IOOptionBits = u32;
type CFMutableDictionaryRef = *mut c_void;
type CFDictionaryRef = *const c_void;
type CFStringRef = *const c_void;
type CFTypeRef = *const c_void;
type IOSurfaceRef = *mut c_void;

type mach_vm_address_t = u64;
type mach_vm_size_t = u64;
type vm_region_flavor_t = i32;
type vm_prot_t = i32;
type memory_object_name_t = mach_port_t;
type vm_region_info_t = *mut i32;

// ────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────

const KERN_SUCCESS: kern_return_t = 0;
const kIOMasterPortDefault: mach_port_t = 0;
const kCFStringEncodingUTF8: u32 = 0x08000100;
const kCFNumberSInt32Type: i32 = 3;
const kCFNumberSInt64Type: i32 = 4;

const PAGE_SIZE: usize = 16384; // Apple Silicon

// ExternalMethod selectors (must match DEXT)
const SELECTOR_GET_INFO: u32 = 0;
const SELECTOR_GET_SEGMENTS: u32 = 1;

// Memory type for IOConnectMapMemory64
const MEMORY_TYPE_BUFFER: u32 = 0;

// VM region constants
const VM_REGION_BASIC_INFO_64: vm_region_flavor_t = 9;
const VM_REGION_BASIC_INFO_COUNT_64: u32 = 9;

// mach_vm_page_info
const VM_PAGE_INFO_BASIC: i32 = 1;

// ────────────────────────────────────────────────────────────────────
// Structs (must match DEXT definitions)
// ────────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Default)]
struct CybAllocInfo {
    alloc_size: u64,
    segment_count: u64,
    flags: u64,
    reserved: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Default)]
struct CybSegmentEntry {
    phys_addr: u64,
    length: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Default)]
struct vm_region_basic_info_data_64_t {
    protection: vm_prot_t,
    max_protection: vm_prot_t,
    inheritance: u32,
    shared: u32,
    reserved: u32,
    offset: u64,
    behavior: i32,
    user_wired_count: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Default)]
struct vm_page_info_basic_data_t {
    disposition: i32,
    ref_count: i32,
    object_id: u64,
    offset: u64,
    depth: i32,
    _pad: i32,
}

// ────────────────────────────────────────────────────────────────────
// IOKit FFI
// ────────────────────────────────────────────────────────────────────

#[link(name = "IOKit", kind = "framework")]
extern "C" {
    fn IOServiceMatching(name: *const c_char) -> CFMutableDictionaryRef;
    fn IOServiceGetMatchingService(
        mainPort: mach_port_t,
        matching: CFMutableDictionaryRef,
    ) -> io_service_t;
    fn IOServiceGetMatchingServices(
        mainPort: mach_port_t,
        matching: CFMutableDictionaryRef,
        existing: *mut io_iterator_t,
    ) -> kern_return_t;
    fn IOServiceOpen(
        service: io_service_t,
        owningTask: mach_port_t,
        type_: u32,
        connect: *mut io_connect_t,
    ) -> kern_return_t;
    fn IOServiceClose(connect: io_connect_t) -> kern_return_t;
    fn IOConnectCallStructMethod(
        connection: io_connect_t,
        selector: u32,
        inputStruct: *const c_void,
        inputStructCnt: usize,
        outputStruct: *mut c_void,
        outputStructCnt: *mut usize,
    ) -> kern_return_t;
    fn IOConnectMapMemory64(
        connect: io_connect_t,
        memoryType: u32,
        intoTask: mach_port_t,
        address: *mut mach_vm_address_t,
        size: *mut mach_vm_size_t,
        options: IOOptionBits,
    ) -> kern_return_t;
    fn IOConnectUnmapMemory64(
        connect: io_connect_t,
        memoryType: u32,
        intoTask: mach_port_t,
        address: mach_vm_address_t,
    ) -> kern_return_t;
    fn IOObjectRelease(object: io_object_t) -> kern_return_t;
    fn IOIteratorNext(iterator: io_iterator_t) -> io_object_t;
}

extern "C" {
    fn mach_task_self() -> mach_port_t;
}

// ────────────────────────────────────────────────────────────────────
// CoreFoundation FFI
// ────────────────────────────────────────────────────────────────────

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFStringCreateWithCString(
        alloc: *const c_void,
        cStr: *const c_char,
        encoding: u32,
    ) -> CFStringRef;
    fn CFDictionaryCreateMutable(
        allocator: *const c_void,
        capacity: i64,
        keyCallBacks: *const c_void,
        valueCallBacks: *const c_void,
    ) -> CFMutableDictionaryRef;
    fn CFDictionarySetValue(
        dict: CFMutableDictionaryRef,
        key: *const c_void,
        value: *const c_void,
    );
    fn CFNumberCreate(
        allocator: *const c_void,
        theType: i32,
        valuePtr: *const c_void,
    ) -> *const c_void;
    fn CFRelease(cf: CFTypeRef);

    static kCFTypeDictionaryKeyCallBacks: c_void;
    static kCFTypeDictionaryValueCallBacks: c_void;
}

// ────────────────────────────────────────────────────────────────────
// IOSurface FFI
// ────────────────────────────────────────────────────────────────────

#[link(name = "IOSurface", kind = "framework")]
extern "C" {
    fn IOSurfaceCreate(properties: CFMutableDictionaryRef) -> IOSurfaceRef;
    fn IOSurfaceLock(surface: IOSurfaceRef, options: u32, seed: *mut u32) -> kern_return_t;
    fn IOSurfaceUnlock(surface: IOSurfaceRef, options: u32, seed: *mut u32) -> kern_return_t;
    fn IOSurfaceGetBaseAddress(surface: IOSurfaceRef) -> *mut c_void;
    fn IOSurfaceGetAllocSize(surface: IOSurfaceRef) -> usize;
    fn IOSurfaceGetID(surface: IOSurfaceRef) -> u32;
    fn IOSurfaceLookupFromMachPort(port: mach_port_t) -> IOSurfaceRef;
    fn IOSurfaceCreateMachPort(surface: IOSurfaceRef) -> mach_port_t;
}

// ────────────────────────────────────────────────────────────────────
// Mach VM FFI
// ────────────────────────────────────────────────────────────────────

extern "C" {
    fn mach_vm_region(
        target_task: mach_port_t,
        address: *mut mach_vm_address_t,
        size: *mut mach_vm_size_t,
        flavor: vm_region_flavor_t,
        info: vm_region_info_t,
        infoCnt: *mut u32,
        object_name: *mut memory_object_name_t,
    ) -> kern_return_t;

    fn mach_vm_page_info(
        target_task: mach_port_t,
        address: mach_vm_address_t,
        flavor: i32,
        info: *mut i32,
        infoCnt: *mut u32,
    ) -> kern_return_t;
}

// ────────────────────────────────────────────────────────────────────
// CF helpers
// ────────────────────────────────────────────────────────────────────

fn cf_str(s: &str) -> CFStringRef {
    unsafe {
        let c = std::ffi::CString::new(s).unwrap();
        CFStringCreateWithCString(ptr::null(), c.as_ptr(), kCFStringEncodingUTF8)
    }
}

fn cf_num_i32(v: i32) -> *const c_void {
    unsafe {
        CFNumberCreate(
            ptr::null(),
            kCFNumberSInt32Type,
            &v as *const i32 as *const c_void,
        )
    }
}

fn cf_num_i64(v: i64) -> *const c_void {
    unsafe {
        CFNumberCreate(
            ptr::null(),
            kCFNumberSInt64Type,
            &v as *const i64 as *const c_void,
        )
    }
}

fn kern_err(kr: kern_return_t) -> String {
    match kr {
        0 => "KERN_SUCCESS".into(),
        x if x as u32 == 0xe00002bc => "kIOReturnBadArgument".into(),
        x if x as u32 == 0xe00002c1 => "kIOReturnNotPrivileged".into(),
        x if x as u32 == 0xe00002d8 => "kIOReturnUnsupported".into(),
        x if x as u32 == 0xe00002be => "kIOReturnExclusiveAccess".into(),
        x if x as u32 == 0xe00002c0 => "kIOReturnNotReady".into(),
        x if x as u32 == 0xe00002c2 => "kIOReturnNotPermitted".into(),
        x if x as u32 == 0xe00002ed => "kIOReturnNotFound".into(),
        _ => format!("{:#010x}", kr as u32),
    }
}

fn prot_str(prot: vm_prot_t) -> String {
    let r = if prot & 1 != 0 { "r" } else { "-" };
    let w = if prot & 2 != 0 { "w" } else { "-" };
    let x = if prot & 4 != 0 { "x" } else { "-" };
    format!("{}{}{}", r, w, x)
}

fn fmt_bytes(b: usize) -> String {
    if b >= 1024 * 1024 * 1024 {
        format!("{:.1} GB", b as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if b >= 1024 * 1024 {
        format!("{:.1} MB", b as f64 / (1024.0 * 1024.0))
    } else if b >= 1024 {
        format!("{:.1} KB", b as f64 / 1024.0)
    } else {
        format!("{} B", b)
    }
}

// ────────────────────────────────────────────────────────────────────
// IOSurface creation (same pattern as iosurface_probe)
// ────────────────────────────────────────────────────────────────────

fn create_iosurface(bytes: usize) -> Option<IOSurfaceRef> {
    unsafe {
        let dict = CFDictionaryCreateMutable(
            ptr::null(),
            0,
            &kCFTypeDictionaryKeyCallBacks as *const c_void,
            &kCFTypeDictionaryValueCallBacks as *const c_void,
        );
        CFDictionarySetValue(dict, cf_str("IOSurfaceWidth") as _, cf_num_i64(bytes as i64));
        CFDictionarySetValue(dict, cf_str("IOSurfaceHeight") as _, cf_num_i64(1));
        CFDictionarySetValue(dict, cf_str("IOSurfaceBytesPerElement") as _, cf_num_i64(1));
        CFDictionarySetValue(dict, cf_str("IOSurfaceBytesPerRow") as _, cf_num_i64(bytes as i64));
        CFDictionarySetValue(dict, cf_str("IOSurfaceAllocSize") as _, cf_num_i64(bytes as i64));
        CFDictionarySetValue(dict, cf_str("IOSurfacePixelFormat") as _, cf_num_i64(0));

        let raw = IOSurfaceCreate(dict);
        CFRelease(dict as CFTypeRef);
        if raw.is_null() {
            None
        } else {
            Some(raw)
        }
    }
}

fn destroy_iosurface(surface: IOSurfaceRef) {
    unsafe {
        CFRelease(surface as CFTypeRef);
    }
}

// ────────────────────────────────────────────────────────────────────
// Benchmark helpers
// ────────────────────────────────────────────────────────────────────

fn bench<F: FnMut()>(iters: usize, mut f: F) -> (u128, u128, u128) {
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        times.push(t0.elapsed().as_nanos());
    }
    times.sort();
    let min = times[0];
    let median = times[times.len() / 2];
    let mean = times.iter().sum::<u128>() / times.len() as u128;
    (min, median, mean)
}

// ────────────────────────────────────────────────────────────────────
// VM region analysis for a memory range
// ────────────────────────────────────────────────────────────────────

fn analyze_vm_regions(base: u64, size: usize) {
    println!("  VM regions covering [{:#018x} .. {:#018x}]:", base, base + size as u64);

    unsafe {
        let task = mach_task_self();
        let mut addr: mach_vm_address_t = base;
        let end = base + size as u64;
        let mut region_count = 0u32;

        while addr < end {
            let mut region_size: mach_vm_size_t = 0;
            let mut info = vm_region_basic_info_data_64_t::default();
            let mut info_count: u32 = VM_REGION_BASIC_INFO_COUNT_64;
            let mut object_name: memory_object_name_t = 0;

            let kr = mach_vm_region(
                task,
                &mut addr,
                &mut region_size,
                VM_REGION_BASIC_INFO_64,
                &mut info as *mut _ as vm_region_info_t,
                &mut info_count,
                &mut object_name,
            );

            if kr != KERN_SUCCESS {
                println!("    mach_vm_region failed at {:#018x}: {}", addr, kern_err(kr));
                break;
            }

            if addr >= end {
                break;
            }

            let overlap_start = addr.max(base);
            let overlap_end = (addr + region_size).min(end);

            if overlap_start < overlap_end {
                println!(
                    "    [{:#018x} - {:#018x}] size={:<12} prot={} max={} shared={} offset={:#x}",
                    addr,
                    addr + region_size,
                    fmt_bytes(region_size as usize),
                    prot_str(info.protection),
                    prot_str(info.max_protection),
                    info.shared,
                    info.offset,
                );
                region_count += 1;
            }

            addr += region_size;
        }

        println!(
            "  total VM regions: {} (single contiguous VA = {})",
            region_count,
            if region_count <= 1 { "YES" } else { "NO" }
        );
    }
}

// ────────────────────────────────────────────────────────────────────
// Page-level disposition analysis
// ────────────────────────────────────────────────────────────────────

fn analyze_page_dispositions(base: u64, size: usize) {
    let pages = (size + PAGE_SIZE - 1) / PAGE_SIZE;
    let sample = pages.min(32);

    println!("  page disposition analysis ({} of {} pages):", sample, pages);

    unsafe {
        let task = mach_task_self();
        let mut prev_obj_id: Option<u64> = None;
        let mut obj_id_changes = 0u32;

        for p in 0..sample {
            let page_addr = base + (p * PAGE_SIZE) as u64;
            let mut info = vm_page_info_basic_data_t::default();
            let mut info_count: u32 =
                (std::mem::size_of::<vm_page_info_basic_data_t>() / std::mem::size_of::<i32>())
                    as u32;

            let kr = mach_vm_page_info(
                task,
                page_addr,
                VM_PAGE_INFO_BASIC,
                &mut info as *mut _ as *mut i32,
                &mut info_count,
            );

            if kr != KERN_SUCCESS {
                if p < 4 {
                    println!(
                        "    page {:>4} @ {:#018x}: FAILED ({})",
                        p, page_addr, kern_err(kr)
                    );
                }
                continue;
            }

            if p < 8 || p == sample - 1 {
                println!(
                    "    page {:>4} @ {:#018x}: disp={:#010x} ref={} obj={:#x} off={:#x} depth={}",
                    p, page_addr, info.disposition, info.ref_count,
                    info.object_id, info.offset, info.depth
                );
            } else if p == 8 {
                println!("    ...");
            }

            if let Some(prev) = prev_obj_id {
                if prev != info.object_id {
                    obj_id_changes += 1;
                }
            }
            prev_obj_id = Some(info.object_id);
        }

        println!(
            "  object ID uniformity: {} changes across {} pages ({})",
            obj_id_changes,
            sample,
            if obj_id_changes == 0 { "UNIFORM -- likely contiguous" } else { "VARIES -- fragmented" }
        );
    }
}

// ────────────────────────────────────────────────────────────────────
// Memory throughput/latency benchmark
// ────────────────────────────────────────────────────────────────────

unsafe fn benchmark_memory(base: *mut u8, size: usize, label: &str) {
    println!("\n  --- Memory Benchmark: {} ({}) ---", label, fmt_bytes(size));

    // Sequential write throughput
    let iters = if size <= 1024 * 1024 { 20 } else { 5 };
    let mut write_ns = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let ptr64 = base as *mut u64;
        let count64 = size / 8;
        let pattern: u64 = 0xDEAD_BEEF_CAFE_BABE;
        for i in 0..count64 {
            ptr64.add(i).write_volatile(pattern);
        }
        write_ns.push(t0.elapsed().as_nanos());
    }
    write_ns.sort();
    let write_med = write_ns[write_ns.len() / 2];
    let write_gbps =
        size as f64 / (write_med as f64 / 1_000_000_000.0) / (1024.0 * 1024.0 * 1024.0);
    println!(
        "    seq write (volatile u64): med={:.1}us ({:.2} GB/s)",
        write_med as f64 / 1000.0,
        write_gbps
    );

    // Sequential read throughput
    let mut read_ns = Vec::with_capacity(iters);
    let mut sink: u64 = 0;
    for _ in 0..iters {
        let t0 = Instant::now();
        let ptr64 = base as *const u64;
        let count64 = size / 8;
        let mut acc: u64 = 0;
        for i in 0..count64 {
            acc ^= ptr64.add(i).read_volatile();
        }
        sink ^= acc;
        read_ns.push(t0.elapsed().as_nanos());
    }
    read_ns.sort();
    let read_med = read_ns[read_ns.len() / 2];
    let read_gbps =
        size as f64 / (read_med as f64 / 1_000_000_000.0) / (1024.0 * 1024.0 * 1024.0);
    println!(
        "    seq read  (volatile u64): med={:.1}us ({:.2} GB/s) [sink={}]",
        read_med as f64 / 1000.0,
        read_gbps,
        sink
    );

    // Data integrity
    let ptr64 = base as *const u64;
    let count64 = size / 8;
    let pattern: u64 = 0xDEAD_BEEF_CAFE_BABE;
    let mut mismatches = 0usize;
    for i in 0..count64 {
        if ptr64.add(i).read_volatile() != pattern {
            mismatches += 1;
        }
    }
    println!(
        "    integrity: {}/{} u64s match ({})",
        count64 - mismatches,
        count64,
        if mismatches == 0 { "PASS" } else { "FAIL" }
    );

    // Random access latency
    let n_samples = 10000.min(size / 64);
    let mut indices: Vec<usize> = (0..n_samples).map(|i| (i * 64) % size).collect();
    let mut rng: u64 = 0xBADC0FFEE0DDF00D;
    for i in (1..indices.len()).rev() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let j = (rng as usize) % (i + 1);
        indices.swap(i, j);
    }

    let mut rnd_sink: u64 = 0;
    let t0 = Instant::now();
    for &idx in &indices {
        rnd_sink ^= (base.add(idx).read_volatile()) as u64;
    }
    let random_ns = t0.elapsed().as_nanos();
    println!(
        "    random read ({} samples): {:.1}us total, {:.1}ns/access [sink={}]",
        n_samples,
        random_ns as f64 / 1000.0,
        random_ns as f64 / n_samples as f64,
        rnd_sink
    );
}

// ────────────────────────────────────────────────────────────────────
// IOSurface wrapping experiment
// ────────────────────────────────────────────────────────────────────

fn try_iosurface_wrap(mapped_addr: u64, mapped_size: usize) -> bool {
    println!("\n==========================================================");
    println!("  Phase 3: IOSurface Wrapping Experiment");
    println!("==========================================================");

    // Strategy 1: Create a normal IOSurface and compare characteristics
    println!("\n  [3a] Creating reference IOSurface ({})...", fmt_bytes(mapped_size));
    let ref_surface = match create_iosurface(mapped_size) {
        Some(s) => s,
        None => {
            println!("    FAILED to create reference IOSurface");
            return false;
        }
    };

    let ref_size = unsafe { IOSurfaceGetAllocSize(ref_surface) };
    let ref_id = unsafe { IOSurfaceGetID(ref_surface) };
    println!("    reference IOSurface: id={} size={}", ref_id, fmt_bytes(ref_size));

    unsafe {
        IOSurfaceLock(ref_surface, 0, ptr::null_mut());
        let ref_base = IOSurfaceGetBaseAddress(ref_surface) as u64;
        println!("    reference base address: {:#018x}", ref_base);
        println!("    DEXT mapped address:    {:#018x}", mapped_addr);
        println!("    delta:                  {:#x}", {
            if ref_base > mapped_addr {
                ref_base - mapped_addr
            } else {
                mapped_addr - ref_base
            }
        });

        // Benchmark the reference surface
        benchmark_memory(
            ref_base as *mut u8,
            ref_size,
            "reference IOSurface",
        );

        IOSurfaceUnlock(ref_surface, 0, ptr::null_mut());
    }

    // Strategy 2: Try to create IOSurface with kIOSurfaceAddressOffset hint
    // This is speculative -- IOSurface may not honor arbitrary backing addresses
    println!("\n  [3b] Attempting IOSurface with custom address properties...");
    println!("    NOTE: IOSurface typically allocates its own backing memory.");
    println!("    There is no public API to create an IOSurface backed by");
    println!("    pre-existing physical pages from userspace.");
    println!();
    println!("    Known approaches for custom-backed IOSurface:");
    println!("    - Kernel kext: IOSurfaceRootUserClient::createSurface with custom pager");
    println!("    - DEXT: Create IOMemoryDescriptor, pass to IOSurface subsystem");
    println!("    - IOSurfaceLookupFromMachPort: if DEXT creates the surface kernel-side");
    println!();

    // Strategy 3: Create IOSurface at same size, mach port roundtrip
    println!("  [3c] IOSurface mach port roundtrip test...");
    unsafe {
        let port = IOSurfaceCreateMachPort(ref_surface);
        if port != 0 {
            let looked_up = IOSurfaceLookupFromMachPort(port);
            if !looked_up.is_null() {
                let lu_id = IOSurfaceGetID(looked_up);
                let lu_size = IOSurfaceGetAllocSize(looked_up);
                println!(
                    "    roundtrip OK: port={} -> surface id={} size={}",
                    port, lu_id, fmt_bytes(lu_size)
                );
                println!(
                    "    same surface: {}",
                    if lu_id == ref_id { "YES" } else { "NO" }
                );
                CFRelease(looked_up as CFTypeRef);
            } else {
                println!("    IOSurfaceLookupFromMachPort returned null");
            }
        } else {
            println!("    IOSurfaceCreateMachPort returned 0");
        }
    }

    // Analyze the reference surface VM structure for comparison
    unsafe {
        IOSurfaceLock(ref_surface, 0, ptr::null_mut());
        let ref_base = IOSurfaceGetBaseAddress(ref_surface) as u64;
        println!("\n  [3d] VM analysis of reference IOSurface:");
        analyze_vm_regions(ref_base, ref_size);
        analyze_page_dispositions(ref_base, ref_size);
        IOSurfaceUnlock(ref_surface, 0, ptr::null_mut());
    }

    destroy_iosurface(ref_surface);
    true
}

// ────────────────────────────────────────────────────────────────────
// ANE visibility test (from rane patterns)
// ────────────────────────────────────────────────────────────────────

fn try_ane_visibility(surface_size: usize) {
    println!("\n==========================================================");
    println!("  Phase 4: ANE Visibility Test");
    println!("==========================================================");

    println!("  Loading ANE frameworks...");
    let frameworks = ["AppleNeuralEngine", "ANECompiler", "ANEServices"];
    let mut all_loaded = true;

    for name in &frameworks {
        let path = format!(
            "/System/Library/PrivateFrameworks/{}.framework/{}",
            name, name
        );
        let c_path = std::ffi::CString::new(path.clone()).unwrap();
        let handle = unsafe { libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW) };
        if handle.is_null() {
            println!("    {} -- NOT FOUND (expected on some configs)", name);
            all_loaded = false;
        } else {
            println!("    {} -- loaded", name);
        }
    }

    if !all_loaded {
        println!("  ANE frameworks not fully available, skipping ANE test.");
        println!("  To test ANE integration, use ~/git/rane/ directly.");
        return;
    }

    // Create a test IOSurface in ANE-compatible format
    println!("\n  Creating ANE-format test surface (fp16, {} bytes)...", fmt_bytes(surface_size));
    let surface = match create_iosurface(surface_size) {
        Some(s) => s,
        None => {
            println!("  Failed to create test IOSurface");
            return;
        }
    };

    let actual_size = unsafe { IOSurfaceGetAllocSize(surface) };
    let surface_id = unsafe { IOSurfaceGetID(surface) };
    println!("  surface id={} size={}", surface_id, fmt_bytes(actual_size));

    // Fill with fp16 test pattern
    unsafe {
        IOSurfaceLock(surface, 0, ptr::null_mut());
        let base = IOSurfaceGetBaseAddress(surface) as *mut u16;
        let count = actual_size / 2;
        for i in 0..count {
            // fp16 encoding of 1.0 = 0x3C00
            base.add(i).write_volatile(0x3C00);
        }
        IOSurfaceUnlock(surface, 0, ptr::null_mut());
    }

    println!("  Wrote fp16(1.0) pattern to surface");
    println!("  To verify ANE can read this surface, use rane::Program with");
    println!("  this surface ID as input. See ~/git/rane/ for the full pipeline.");
    println!("  The key question: can we substitute DEXT-allocated contiguous");
    println!("  memory as the backing store and still have ANE accept it?");

    destroy_iosurface(surface);
}

// ────────────────────────────────────────────────────────────────────
// DEXT connection attempt
// ────────────────────────────────────────────────────────────────────

fn try_connect_dext() -> Option<(io_connect_t, io_service_t)> {
    unsafe {
        let name = std::ffi::CString::new("CybMemAllocDriver").unwrap();
        let matching = IOServiceMatching(name.as_ptr());
        if matching.is_null() {
            println!("  IOServiceMatching returned null");
            return None;
        }

        let service = IOServiceGetMatchingService(kIOMasterPortDefault, matching);
        if service == 0 {
            println!("  IOServiceGetMatchingService: DEXT not found in IORegistry");
            return None;
        }

        println!("  Found DEXT service: {}", service);

        let mut connect: io_connect_t = 0;
        let kr = IOServiceOpen(service, mach_task_self(), 0, &mut connect);
        if kr != KERN_SUCCESS {
            println!("  IOServiceOpen failed: {}", kern_err(kr));
            IOObjectRelease(service);
            return None;
        }

        println!("  Connected to DEXT: connect={}", connect);
        Some((connect, service))
    }
}

// ────────────────────────────────────────────────────────────────────
// Full DEXT experiment path
// ────────────────────────────────────────────────────────────────────

fn run_dext_experiment(connect: io_connect_t) {
    println!("\n==========================================================");
    println!("  Phase 1: DEXT Buffer Mapping");
    println!("==========================================================");

    // Get allocation info via ExternalMethod
    println!("\n  [1a] Getting allocation info from DEXT...");
    let mut info = CybAllocInfo::default();
    let mut info_size = std::mem::size_of::<CybAllocInfo>();

    unsafe {
        let kr = IOConnectCallStructMethod(
            connect,
            SELECTOR_GET_INFO,
            ptr::null(),
            0,
            &mut info as *mut _ as *mut c_void,
            &mut info_size,
        );

        if kr != KERN_SUCCESS {
            println!("    GetInfo failed: {} (DEXT may not support struct output)", kern_err(kr));
            println!("    Continuing with mapping attempt...");
        } else {
            println!("    allocSize:    {}", fmt_bytes(info.alloc_size as usize));
            println!("    segmentCount: {}", info.segment_count);
            println!("    flags:        {:#x}", info.flags);
        }
    }

    // Get physical segments
    println!("\n  [1b] Getting physical segment list from DEXT...");
    let mut segments = vec![CybSegmentEntry::default(); 4096];
    let mut seg_size = segments.len() * std::mem::size_of::<CybSegmentEntry>();

    unsafe {
        let kr = IOConnectCallStructMethod(
            connect,
            SELECTOR_GET_SEGMENTS,
            ptr::null(),
            0,
            segments.as_mut_ptr() as *mut c_void,
            &mut seg_size,
        );

        if kr != KERN_SUCCESS {
            println!("    GetSegments failed: {}", kern_err(kr));
        } else {
            let count = seg_size / std::mem::size_of::<CybSegmentEntry>();
            println!("    received {} segments:", count);
            for i in 0..count.min(16) {
                println!(
                    "      seg[{}]: PA={:#018x} len={}",
                    i,
                    segments[i].phys_addr,
                    fmt_bytes(segments[i].length as usize)
                );
            }
            if count > 16 {
                println!("      ... ({} more)", count - 16);
            }
            if count == 1 {
                println!("    RESULT: single segment -- PHYSICALLY CONTIGUOUS");
            } else if count > 1 {
                println!("    RESULT: {} segments -- FRAGMENTED", count);
            }
        }
    }

    // Map buffer into our address space
    println!("\n  [1c] Mapping DEXT buffer via IOConnectMapMemory64...");
    let mut mapped_addr: mach_vm_address_t = 0;
    let mut mapped_size: mach_vm_size_t = 0;

    unsafe {
        let kr = IOConnectMapMemory64(
            connect,
            MEMORY_TYPE_BUFFER,
            mach_task_self(),
            &mut mapped_addr,
            &mut mapped_size,
            0, // kIOMapAnywhere
        );

        if kr != KERN_SUCCESS {
            println!("    IOConnectMapMemory64 failed: {}", kern_err(kr));
            println!("    The DEXT may not have implemented CopyClientMemoryForType,");
            println!("    or the buffer allocation may have failed.");
            return;
        }

        println!("    mapped address: {:#018x}", mapped_addr);
        println!("    mapped size:    {}", fmt_bytes(mapped_size as usize));
    }

    // Analyze the mapped region
    println!("\n==========================================================");
    println!("  Phase 2: Mapped Buffer Analysis");
    println!("==========================================================");

    println!("\n  [2a] VM region analysis:");
    analyze_vm_regions(mapped_addr, mapped_size as usize);

    // Touch all pages
    println!("\n  [2b] Touching all pages...");
    let t0 = Instant::now();
    let pages = (mapped_size as usize + PAGE_SIZE - 1) / PAGE_SIZE;
    unsafe {
        let base = mapped_addr as *mut u8;
        for i in (0..(mapped_size as usize)).step_by(PAGE_SIZE) {
            base.add(i).write_volatile(0x42);
        }
    }
    let touch_ns = t0.elapsed().as_nanos();
    println!(
        "    touched {} pages in {:.1}us ({:.0}ns/page)",
        pages,
        touch_ns as f64 / 1000.0,
        touch_ns as f64 / pages as f64
    );

    // Page disposition analysis
    println!("\n  [2c] Page disposition analysis:");
    analyze_page_dispositions(mapped_addr, mapped_size as usize);

    // Throughput/latency benchmark
    unsafe {
        benchmark_memory(
            mapped_addr as *mut u8,
            mapped_size as usize,
            "DEXT contiguous buffer",
        );
    }

    // IOSurface wrapping
    try_iosurface_wrap(mapped_addr, mapped_size as usize);

    // ANE visibility
    try_ane_visibility(mapped_size as usize);

    // Cleanup
    println!("\n==========================================================");
    println!("  Cleanup");
    println!("==========================================================");

    unsafe {
        let kr = IOConnectUnmapMemory64(
            connect,
            MEMORY_TYPE_BUFFER,
            mach_task_self(),
            mapped_addr,
        );
        println!("  IOConnectUnmapMemory64: {}", kern_err(kr));
    }
}

// ────────────────────────────────────────────────────────────────────
// Standalone fallback (no DEXT)
// ────────────────────────────────────────────────────────────────────

fn run_standalone_experiment() {
    println!("\n  Running in STANDALONE mode (no DEXT available).");
    println!("  This demonstrates the client-side analysis using a regular");
    println!("  IOSurface allocation for comparison measurements.");
    println!("  Install and activate the DEXT for the full experiment.");

    let test_size: usize = 16 * 1024 * 1024; // 16 MB

    println!("\n==========================================================");
    println!("  Phase 2 (standalone): IOSurface Analysis ({})", fmt_bytes(test_size));
    println!("==========================================================");

    let surface = match create_iosurface(test_size) {
        Some(s) => s,
        None => {
            println!("  FAILED to create IOSurface");
            return;
        }
    };

    let actual_size = unsafe { IOSurfaceGetAllocSize(surface) };
    let surface_id = unsafe { IOSurfaceGetID(surface) };
    println!("  IOSurface created: id={} size={}", surface_id, fmt_bytes(actual_size));

    unsafe {
        IOSurfaceLock(surface, 0, ptr::null_mut());
        let base = IOSurfaceGetBaseAddress(surface) as u64;
        println!("  base address: {:#018x}", base);

        // Touch all pages
        let t0 = Instant::now();
        let pages = (actual_size + PAGE_SIZE - 1) / PAGE_SIZE;
        let ptr = base as *mut u8;
        for i in (0..actual_size).step_by(PAGE_SIZE) {
            ptr.add(i).write_volatile(0x42);
        }
        let touch_ns = t0.elapsed().as_nanos();
        println!(
            "  first-touch: {} pages in {:.1}us ({:.0}ns/page)",
            pages,
            touch_ns as f64 / 1000.0,
            touch_ns as f64 / pages as f64
        );

        // VM region analysis
        println!();
        analyze_vm_regions(base, actual_size);
        println!();
        analyze_page_dispositions(base, actual_size);

        // Throughput benchmark
        benchmark_memory(base as *mut u8, actual_size, "IOSurface (standard alloc)");

        IOSurfaceUnlock(surface, 0, ptr::null_mut());
    }

    // IOSurface wrapping test (compares with itself)
    unsafe {
        IOSurfaceLock(surface, 0, ptr::null_mut());
        let base = IOSurfaceGetBaseAddress(surface) as u64;
        try_iosurface_wrap(base, actual_size);
        IOSurfaceUnlock(surface, 0, ptr::null_mut());
    }

    // ANE visibility test
    try_ane_visibility(actual_size);

    destroy_iosurface(surface);
}

// ────────────────────────────────────────────────────────────────────
// Main
// ────────────────────────────────────────────────────────────────────

fn main() {
    println!("==========================================================");
    println!("  DEXT Contiguous Allocation Experiment");
    println!("  PATH 1: DEXT alloc contiguous -> IOSurface -> ANE");
    println!("==========================================================");
    println!("  page size:    {} bytes", PAGE_SIZE);
    println!("  target alloc: 16 MB (1024 pages)");
    println!();

    // Try to connect to the DEXT
    println!("==========================================================");
    println!("  Phase 0: DEXT Connection");
    println!("==========================================================");
    println!("\n  Searching for CybMemAllocDriver in IORegistry...");

    match try_connect_dext() {
        Some((connect, service)) => {
            println!("  SUCCESS: connected to DEXT");
            run_dext_experiment(connect);

            // Cleanup
            unsafe {
                IOServiceClose(connect);
                IOObjectRelease(service);
            }
        }
        None => {
            println!("\n  DEXT not available.");
            println!("  To install the DEXT:");
            println!("    1. Build: ./build_dext.sh");
            println!("    2. Enable developer mode: systemextensionsctl developer on");
            println!("    3. Install the .dext bundle to /Library/DriverExtensions/");
            println!("    4. Approve in System Settings > Privacy & Security");
            run_standalone_experiment();
        }
    }

    println!("\n==========================================================");
    println!("  Experiment Summary");
    println!("==========================================================");
    println!();
    println!("  KEY FINDINGS (to be updated after running):");
    println!();
    println!("  Q1: Does IOBufferMemoryDescriptor in DEXT give contiguous PA?");
    println!("      -> Check segment count from Phase 1b");
    println!("      -> kIOMemoryPhysicallyContiguous is KERNEL-ONLY");
    println!("      -> DEXT may get contiguous for small allocs (< ~2MB)");
    println!();
    println!("  Q2: Can we map DEXT memory into userspace?");
    println!("      -> Check Phase 1c IOConnectMapMemory64 result");
    println!();
    println!("  Q3: Can we wrap the mapped memory as IOSurface?");
    println!("      -> Check Phase 3 results");
    println!("      -> No public API to wrap arbitrary PA as IOSurface");
    println!("      -> Alternative: DEXT creates IOSurface kernel-side");
    println!();
    println!("  Q4: Would ANE accept such a surface?");
    println!("      -> Check Phase 4 results");
    println!("      -> ANE uses IOSurface IDs, not raw addresses");
    println!();
    println!("  NEXT STEPS:");
    println!("    - If DEXT contiguous works: try IOSurfaceClientCreate from DEXT");
    println!("    - If not: explore kext path or IODMACommand with contiguous hint");
    println!("    - Compare throughput: DEXT buffer vs standard IOSurface");
    println!();
    println!("==========================================================");
    println!("  Done.");
    println!("==========================================================");
}
