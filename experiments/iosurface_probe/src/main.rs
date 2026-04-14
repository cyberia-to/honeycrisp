//! IOSurface probe experiment
//!
//! Creates IOSurfaces at various sizes and measures:
//! - Allocation / deallocation latency
//! - Write / read access latency (first-touch and steady-state)
//! - Sequential read/write throughput
//! - Physical page information via mach_vm_page_info / mach_vm_region
//! - Memory contiguity analysis

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
type mach_port_t = u32;
type CFMutableDictionaryRef = *mut c_void;
type CFStringRef = *const c_void;
type CFTypeRef = *const c_void;
type IOSurfaceRef = *mut c_void;

type mach_vm_address_t = u64;
type mach_vm_size_t = u64;
type vm_region_flavor_t = i32;
type vm_prot_t = i32;
type memory_object_name_t = mach_port_t;
type vm_region_info_t = *mut i32;
type natural_t = u32;

const KERN_SUCCESS: kern_return_t = 0;
const kCFStringEncodingUTF8: u32 = 0x08000100;
const kCFNumberSInt64Type: i32 = 4;

// vm_region flavors
const VM_REGION_BASIC_INFO_64: vm_region_flavor_t = 9;
const VM_REGION_BASIC_INFO_COUNT_64: u32 = 9; // struct has 9 natural_t fields

// page info flavors (mach_vm_page_info)
const VM_PAGE_INFO_BASIC: i32 = 1;

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
    fn CFDictionarySetValue(dict: CFMutableDictionaryRef, key: *const c_void, value: *const c_void);
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
}

// ────────────────────────────────────────────────────────────────────
// Mach VM FFI
// ────────────────────────────────────────────────────────────────────

extern "C" {
    fn mach_task_self() -> mach_port_t;
}

// mach_vm_region for querying virtual memory region info
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
}

// mach_vm_page_info for querying per-page info
extern "C" {
    fn mach_vm_page_info(
        target_task: mach_port_t,
        address: mach_vm_address_t,
        flavor: i32,
        info: *mut i32,
        infoCnt: *mut u32,
    ) -> kern_return_t;
}

// ────────────────────────────────────────────────────────────────────
// vm_region_basic_info_data_64_t
// ────────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Default)]
struct vm_region_basic_info_data_64_t {
    protection: vm_prot_t,        // current protection
    max_protection: vm_prot_t,    // max protection
    inheritance: u32,             // inheritance
    shared: u32,                  // is it shared?
    reserved: u32,                // reserved
    offset: u64,                  // offset into object
    behavior: i32,                // behavior
    user_wired_count: u16,        // user wired count
}

// ────────────────────────────────────────────────────────────────────
// vm_page_info_basic_data_t
// ────────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Default)]
struct vm_page_info_basic_data_t {
    disposition: i32,
    ref_count: i32,
    object_id: u64,
    offset: u64,
    depth: i32,
    // padding to be safe
    _pad: i32,
}

// ────────────────────────────────────────────────────────────────────
// CF helpers (same pattern as rane)
// ────────────────────────────────────────────────────────────────────

fn cf_str(s: &str) -> CFStringRef {
    unsafe {
        let c = std::ffi::CString::new(s).unwrap();
        CFStringCreateWithCString(ptr::null(), c.as_ptr(), kCFStringEncodingUTF8)
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

// ────────────────────────────────────────────────────────────────────
// IOSurface creation / destruction
// ────────────────────────────────────────────────────────────────────

fn create_iosurface(bytes: usize) -> Option<IOSurfaceRef> {
    unsafe {
        let dict = CFDictionaryCreateMutable(
            ptr::null(),
            0,
            &kCFTypeDictionaryKeyCallBacks as *const c_void,
            &kCFTypeDictionaryValueCallBacks as *const c_void,
        );
        // Use i64 numbers so sizes > 2GB don't overflow
        CFDictionarySetValue(dict, cf_str("IOSurfaceWidth") as _, cf_num_i64(bytes as i64));
        CFDictionarySetValue(dict, cf_str("IOSurfaceHeight") as _, cf_num_i64(1));
        CFDictionarySetValue(dict, cf_str("IOSurfaceBytesPerElement") as _, cf_num_i64(1));
        CFDictionarySetValue(
            dict,
            cf_str("IOSurfaceBytesPerRow") as _,
            cf_num_i64(bytes as i64),
        );
        CFDictionarySetValue(
            dict,
            cf_str("IOSurfaceAllocSize") as _,
            cf_num_i64(bytes as i64),
        );
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

const PAGE_SIZE: usize = 16384; // Apple Silicon uses 16KB pages

/// Format bytes as human-readable string
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

/// Run a closure N times and return (min, median, mean) durations in nanoseconds
fn bench<F: FnMut() -> ()>(iters: usize, mut f: F) -> (u128, u128, u128) {
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
// Probe: allocation / deallocation latency
// ────────────────────────────────────────────────────────────────────

fn probe_alloc_dealloc(size: usize) {
    let label = fmt_bytes(size);
    println!("\n--- Allocation/Deallocation: {} ---", label);

    let iters = if size <= 1024 * 1024 { 100 } else { 20 };

    // Allocation
    let (min, med, mean) = bench(iters, || {
        let s = create_iosurface(size).expect("alloc failed");
        // prevent optimization: read the ID
        std::hint::black_box(unsafe { IOSurfaceGetID(s) });
        destroy_iosurface(s);
    });
    println!(
        "  alloc+dealloc (n={}): min={:.1}us  med={:.1}us  mean={:.1}us",
        iters,
        min as f64 / 1000.0,
        med as f64 / 1000.0,
        mean as f64 / 1000.0
    );

    // Allocation only
    let mut surfaces: Vec<IOSurfaceRef> = Vec::new();
    let (min, med, mean) = bench(iters, || {
        let s = create_iosurface(size).expect("alloc failed");
        std::hint::black_box(unsafe { IOSurfaceGetID(s) });
        surfaces.push(s);
    });
    println!(
        "  alloc only   (n={}): min={:.1}us  med={:.1}us  mean={:.1}us",
        iters,
        min as f64 / 1000.0,
        med as f64 / 1000.0,
        mean as f64 / 1000.0
    );

    // Deallocation only
    let (min, med, mean) = bench(surfaces.len(), || {
        let s = surfaces.pop().unwrap();
        destroy_iosurface(s);
    });
    println!(
        "  dealloc only (n={}): min={:.1}us  med={:.1}us  mean={:.1}us",
        iters,
        min as f64 / 1000.0,
        med as f64 / 1000.0,
        mean as f64 / 1000.0
    );
}

// ────────────────────────────────────────────────────────────────────
// Probe: read / write access latency and throughput
// ────────────────────────────────────────────────────────────────────

fn probe_access(size: usize) {
    let label = fmt_bytes(size);
    println!("\n--- Access Latency & Throughput: {} ---", label);

    let surface = create_iosurface(size).expect("alloc failed");
    let actual_size = unsafe { IOSurfaceGetAllocSize(surface) };
    println!(
        "  requested={} actual={} id={}",
        fmt_bytes(size),
        fmt_bytes(actual_size),
        unsafe { IOSurfaceGetID(surface) }
    );

    unsafe {
        // Lock for read/write
        IOSurfaceLock(surface, 0, ptr::null_mut());
        let base = IOSurfaceGetBaseAddress(surface) as *mut u8;
        let slice = std::slice::from_raw_parts_mut(base, actual_size);

        // ── First-touch write latency (page fault) ──
        let t0 = Instant::now();
        for i in (0..actual_size).step_by(PAGE_SIZE) {
            slice[i] = 0xAA;
        }
        let first_touch_ns = t0.elapsed().as_nanos();
        let pages = (actual_size + PAGE_SIZE - 1) / PAGE_SIZE;
        println!(
            "  first-touch write (1 byte per page, {} pages): {:.1}us total, {:.0}ns/page",
            pages,
            first_touch_ns as f64 / 1000.0,
            first_touch_ns as f64 / pages as f64
        );

        // ── Sequential write throughput ──
        let iters = if size <= 1024 * 1024 { 20 } else { 5 };
        let mut write_ns = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            // Write known pattern using 8-byte writes for throughput
            let ptr64 = base as *mut u64;
            let count64 = actual_size / 8;
            let pattern64: u64 = 0xDEAD_BEEF_CAFE_BABE;
            for i in 0..count64 {
                ptr64.add(i).write_volatile(pattern64);
            }
            write_ns.push(t0.elapsed().as_nanos());
        }
        write_ns.sort();
        let write_med = write_ns[write_ns.len() / 2];
        let write_gbps = actual_size as f64 / (write_med as f64 / 1_000_000_000.0) / (1024.0 * 1024.0 * 1024.0);
        println!(
            "  seq write (volatile u64): med={:.1}us  ({:.2} GB/s)",
            write_med as f64 / 1000.0,
            write_gbps
        );

        // ── Sequential read throughput ──
        let mut read_ns = Vec::with_capacity(iters);
        let mut sink: u64 = 0;
        for _ in 0..iters {
            let t0 = Instant::now();
            let ptr64 = base as *const u64;
            let count64 = actual_size / 8;
            let mut acc: u64 = 0;
            for i in 0..count64 {
                acc ^= ptr64.add(i).read_volatile();
            }
            sink ^= acc;
            read_ns.push(t0.elapsed().as_nanos());
        }
        read_ns.sort();
        let read_med = read_ns[read_ns.len() / 2];
        let read_gbps = actual_size as f64 / (read_med as f64 / 1_000_000_000.0) / (1024.0 * 1024.0 * 1024.0);
        println!(
            "  seq read  (volatile u64): med={:.1}us  ({:.2} GB/s)  [sink={}]",
            read_med as f64 / 1000.0,
            read_gbps,
            sink // prevent dead-code elimination
        );

        // ── Verify data integrity ──
        let ptr64 = base as *const u64;
        let count64 = actual_size / 8;
        let mut mismatches = 0usize;
        let pattern64: u64 = 0xDEAD_BEEF_CAFE_BABE;
        for i in 0..count64 {
            if ptr64.add(i).read_volatile() != pattern64 {
                mismatches += 1;
            }
        }
        println!(
            "  data integrity: {}/{} u64s match ({})",
            count64 - mismatches,
            count64,
            if mismatches == 0 { "PASS" } else { "FAIL" }
        );

        // ── Random access latency (64-byte stride, cache-line hopping) ──
        // Create a shuffled index array for random access
        let n_samples = 10000.min(actual_size / 64);
        let mut indices: Vec<usize> = (0..n_samples).map(|i| (i * 64) % actual_size).collect();
        // Simple pseudo-random shuffle (xorshift)
        let mut rng: u64 = 0xBADC0FFEE0DDF00D;
        for i in (1..indices.len()).rev() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let j = (rng as usize) % (i + 1);
            indices.swap(i, j);
        }

        let ptr8 = base as *const u8;
        let mut rnd_sink: u64 = 0;
        let t0 = Instant::now();
        for &idx in &indices {
            rnd_sink ^= (ptr8.add(idx).read_volatile()) as u64;
        }
        let random_ns = t0.elapsed().as_nanos();
        println!(
            "  random read ({} samples, 64B stride): {:.1}us total, {:.1}ns/access  [sink={}]",
            n_samples,
            random_ns as f64 / 1000.0,
            random_ns as f64 / n_samples as f64,
            rnd_sink
        );

        IOSurfaceUnlock(surface, 0, ptr::null_mut());
    }

    destroy_iosurface(surface);
}

// ────────────────────────────────────────────────────────────────────
// Probe: VM region info (contiguity analysis)
// ────────────────────────────────────────────────────────────────────

fn probe_vm_region(size: usize) {
    let label = fmt_bytes(size);
    println!("\n--- VM Region Analysis: {} ---", label);

    let surface = create_iosurface(size).expect("alloc failed");

    unsafe {
        IOSurfaceLock(surface, 0, ptr::null_mut());
        let base = IOSurfaceGetBaseAddress(surface) as u64;
        let actual_size = IOSurfaceGetAllocSize(surface);

        println!(
            "  base address: {:#018x}  size: {}",
            base,
            fmt_bytes(actual_size)
        );

        // Walk the VM regions covering this IOSurface
        let task = mach_task_self();
        let mut addr: mach_vm_address_t = base;
        let end = base + actual_size as u64;
        let mut region_count = 0u32;

        println!("  VM regions covering the surface:");
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
                println!("    mach_vm_region failed at {:#018x}: kr={}", addr, kr);
                break;
            }

            // Only print regions that overlap our surface
            if addr >= end {
                break;
            }

            let region_end = addr + region_size;
            let overlap_start = addr.max(base);
            let overlap_end = region_end.min(end);

            if overlap_start < overlap_end {
                println!(
                    "    [{:#018x} - {:#018x}] size={:<12} prot={} max_prot={} shared={} offset={:#x} behavior={}",
                    addr,
                    addr + region_size,
                    fmt_bytes(region_size as usize),
                    prot_str(info.protection),
                    prot_str(info.max_protection),
                    info.shared,
                    info.offset,
                    info.behavior,
                );
                region_count += 1;
            }

            addr += region_size;
        }

        println!(
            "  total regions: {} (contiguous={})",
            region_count,
            if region_count == 1 { "YES" } else { "NO" }
        );

        IOSurfaceUnlock(surface, 0, ptr::null_mut());
    }

    destroy_iosurface(surface);
}

fn prot_str(prot: vm_prot_t) -> String {
    let r = if prot & 1 != 0 { "r" } else { "-" };
    let w = if prot & 2 != 0 { "w" } else { "-" };
    let x = if prot & 4 != 0 { "x" } else { "-" };
    format!("{}{}{}", r, w, x)
}

// ────────────────────────────────────────────────────────────────────
// Probe: physical page info via mach_vm_page_info
// ────────────────────────────────────────────────────────────────────

fn probe_page_info(size: usize) {
    let label = fmt_bytes(size);
    println!("\n--- Physical Page Info: {} ---", label);

    let surface = create_iosurface(size).expect("alloc failed");

    unsafe {
        IOSurfaceLock(surface, 0, ptr::null_mut());
        let base = IOSurfaceGetBaseAddress(surface) as *mut u8;
        let actual_size = IOSurfaceGetAllocSize(surface);

        // Touch all pages first to ensure they're faulted in
        for i in (0..actual_size).step_by(PAGE_SIZE) {
            base.add(i).write_volatile(0x42);
        }

        let task = mach_task_self();
        let pages = (actual_size + PAGE_SIZE - 1) / PAGE_SIZE;
        let sample_pages = pages.min(32); // sample first N pages

        println!(
            "  sampling {} of {} pages (page_size={})",
            sample_pages, pages, PAGE_SIZE
        );

        let mut prev_disposition: Option<i32> = None;
        let mut disposition_changes = 0u32;

        for p in 0..sample_pages {
            let page_addr = (base as u64) + (p * PAGE_SIZE) as u64;
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
                println!(
                    "    page {:>4} @ {:#018x}: mach_vm_page_info FAILED (kr={})",
                    p, page_addr, kr
                );
                continue;
            }

            let disp = info.disposition;
            if p < 8 || p == sample_pages - 1 {
                println!(
                    "    page {:>4} @ {:#018x}: disposition={:#010x} ref_count={} obj_id={:#x} offset={:#x} depth={}",
                    p, page_addr, disp, info.ref_count, info.object_id, info.offset, info.depth
                );
            } else if p == 8 {
                println!("    ... (sampling {} more pages) ...", sample_pages - 9);
            }

            if let Some(prev) = prev_disposition {
                if prev != disp {
                    disposition_changes += 1;
                }
            }
            prev_disposition = Some(disp);
        }

        // Disposition bit analysis
        if let Some(d) = prev_disposition {
            println!("  disposition bits analysis (last page):");
            println!("    VM_PAGE_QUERY_PAGE_PRESENT     = {}", (d & (1 << 0)) != 0);
            println!("    VM_PAGE_QUERY_PAGE_FICTITIOUS   = {}", (d & (1 << 1)) != 0);
            println!("    VM_PAGE_QUERY_PAGE_REF          = {}", (d & (1 << 2)) != 0);
            println!("    VM_PAGE_QUERY_PAGE_DIRTY        = {}", (d & (1 << 3)) != 0);
            println!("    VM_PAGE_QUERY_PAGE_PAGED_OUT    = {}", (d & (1 << 4)) != 0);
            println!("    VM_PAGE_QUERY_PAGE_ENCRYPTED    = {}", (d & (1 << 5)) != 0);
            println!("    VM_PAGE_QUERY_PAGE_COMPRESSED   = {}", (d & (1 << 6)) != 0);
            // Additional useful bits
            println!(
                "    VM_PAGE_QUERY_PAGE_CS_VALIDATED = {}",
                (d & (1 << 9)) != 0
            );
            println!(
                "    VM_PAGE_QUERY_PAGE_CS_NX        = {}",
                (d & (1 << 11)) != 0
            );
        }

        println!(
            "  disposition uniformity: {} changes across {} pages ({})",
            disposition_changes,
            sample_pages,
            if disposition_changes == 0 {
                "UNIFORM"
            } else {
                "VARIES"
            }
        );

        IOSurfaceUnlock(surface, 0, ptr::null_mut());
    }

    destroy_iosurface(surface);
}

// ────────────────────────────────────────────────────────────────────
// Probe: lock/unlock overhead
// ────────────────────────────────────────────────────────────────────

fn probe_lock_unlock(size: usize) {
    let label = fmt_bytes(size);
    println!("\n--- Lock/Unlock Overhead: {} ---", label);

    let surface = create_iosurface(size).expect("alloc failed");
    let iters = 1000;

    let (min, med, mean) = bench(iters, || unsafe {
        IOSurfaceLock(surface, 0, ptr::null_mut());
        IOSurfaceUnlock(surface, 0, ptr::null_mut());
    });
    println!(
        "  lock+unlock (rw, n={}): min={:.0}ns  med={:.0}ns  mean={:.0}ns",
        iters, min, med, mean
    );

    let (min, med, mean) = bench(iters, || unsafe {
        IOSurfaceLock(surface, 1, ptr::null_mut()); // read-only
        IOSurfaceUnlock(surface, 1, ptr::null_mut());
    });
    println!(
        "  lock+unlock (ro, n={}): min={:.0}ns  med={:.0}ns  mean={:.0}ns",
        iters, min, med, mean
    );

    destroy_iosurface(surface);
}

// ────────────────────────────────────────────────────────────────────
// Main
// ────────────────────────────────────────────────────────────────────

fn main() {
    println!("==========================================================");
    println!("  IOSurface Probe Experiment");
    println!("==========================================================");
    println!("  page size: {} bytes", PAGE_SIZE);
    println!("  testing sizes: 4KB, 1MB, 16MB, 256MB");

    let sizes: &[(usize, &str)] = &[
        (4 * 1024, "4KB"),
        (1 * 1024 * 1024, "1MB"),
        (16 * 1024 * 1024, "16MB"),
        (256 * 1024 * 1024, "256MB"),
    ];

    // ── Phase 1: Allocation / deallocation latency ──
    println!("\n==========================================================");
    println!("  Phase 1: Allocation / Deallocation Latency");
    println!("==========================================================");
    for &(sz, _) in sizes {
        probe_alloc_dealloc(sz);
    }

    // ── Phase 2: Lock/unlock overhead ──
    println!("\n==========================================================");
    println!("  Phase 2: Lock/Unlock Overhead");
    println!("==========================================================");
    for &(sz, _) in sizes {
        probe_lock_unlock(sz);
    }

    // ── Phase 3: Access latency & throughput ──
    println!("\n==========================================================");
    println!("  Phase 3: Access Latency & Throughput");
    println!("==========================================================");
    for &(sz, _) in sizes {
        probe_access(sz);
    }

    // ── Phase 4: VM region analysis (contiguity) ──
    println!("\n==========================================================");
    println!("  Phase 4: VM Region Analysis (Contiguity)");
    println!("==========================================================");
    for &(sz, _) in sizes {
        probe_vm_region(sz);
    }

    // ── Phase 5: Physical page info ──
    println!("\n==========================================================");
    println!("  Phase 5: Physical Page Info");
    println!("==========================================================");
    for &(sz, _) in sizes {
        probe_page_info(sz);
    }

    println!("\n==========================================================");
    println!("  Done.");
    println!("==========================================================");
}
