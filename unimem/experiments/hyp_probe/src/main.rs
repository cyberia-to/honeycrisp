//! hyp_probe: Minimal Hypervisor.framework experiment on Apple Silicon (arm64).
//!
//! Tests hv_vm_create, hv_vm_map, hv_vm_protect, hv_vm_unmap, hv_vm_destroy
//! using raw FFI -- no wrapper crates.
//!
//! Measures latencies and verifies correctness at different memory sizes.
//!
//! IMPORTANT: Apple Silicon uses 16KB pages. All sizes and IPAs must be
//! 16KB-aligned for hv_vm_map to succeed.

use std::ptr;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Raw FFI to Hypervisor.framework (arm64 API)
//
// From SDK headers (MacOSX14.4):
//   hv_return_t       = mach_error_t = i32
//   hv_ipa_t          = uint64_t
//   hv_memory_flags_t = uint64_t
//   hv_vm_config_t    = opaque OS_OBJECT pointer (nullable)
//
// arm64 API signatures (from hv_vm.h):
//   hv_return_t hv_vm_create(hv_vm_config_t _Nullable config);
//   hv_return_t hv_vm_destroy(void);
//   hv_return_t hv_vm_map(void *addr, hv_ipa_t ipa, size_t size, hv_memory_flags_t flags);
//   hv_return_t hv_vm_unmap(hv_ipa_t ipa, size_t size);
//   hv_return_t hv_vm_protect(hv_ipa_t ipa, size_t size, hv_memory_flags_t flags);
// ---------------------------------------------------------------------------

type HvReturn = i32;
type HvIpa = u64;
type HvMemoryFlags = u64;

// hv_memory_flags_t constants (from arm64/hv/hv_kern_types.h)
const HV_MEMORY_READ: HvMemoryFlags = 1 << 0;
const HV_MEMORY_WRITE: HvMemoryFlags = 1 << 1;
const HV_MEMORY_EXEC: HvMemoryFlags = 1 << 2;

// hv_return_t error codes (from arm64/hv/hv_kern_types.h)
//   err_local = err_system(0x3e) = 0x3e << 26 = 0xF800_0000
//   err_sub_hypervisor = err_sub(0xba5) = 0xba5 << 14 = 0x02E9_4000
//   err_common_hypervisor = 0xF800_0000 | 0x02E9_4000 = 0xFAE9_4000
const HV_SUCCESS: HvReturn = 0;

const HV_ERROR_RAW: u32 = 0xFAE9_4001;
const HV_BUSY_RAW: u32 = 0xFAE9_4002;
const HV_BAD_ARGUMENT_RAW: u32 = 0xFAE9_4003;
const HV_ILLEGAL_GUEST_STATE_RAW: u32 = 0xFAE9_4004;
const HV_NO_RESOURCES_RAW: u32 = 0xFAE9_4005;
const HV_NO_DEVICE_RAW: u32 = 0xFAE9_4006;
const HV_DENIED_RAW: u32 = 0xFAE9_4007;
const HV_UNSUPPORTED_RAW: u32 = 0xFAE9_400F;

fn hv_error_name(ret: HvReturn) -> &'static str {
    if ret == HV_SUCCESS {
        return "HV_SUCCESS";
    }
    let raw = ret as u32;
    match raw {
        x if x == HV_ERROR_RAW => "HV_ERROR",
        x if x == HV_BUSY_RAW => "HV_BUSY",
        x if x == HV_BAD_ARGUMENT_RAW => "HV_BAD_ARGUMENT",
        x if x == HV_ILLEGAL_GUEST_STATE_RAW => "HV_ILLEGAL_GUEST_STATE",
        x if x == HV_NO_RESOURCES_RAW => "HV_NO_RESOURCES",
        x if x == HV_NO_DEVICE_RAW => "HV_NO_DEVICE",
        x if x == HV_DENIED_RAW => "HV_DENIED",
        x if x == HV_UNSUPPORTED_RAW => "HV_UNSUPPORTED",
        _ => "UNKNOWN",
    }
}

#[link(name = "Hypervisor", kind = "framework")]
extern "C" {
    fn hv_vm_create(config: *const std::ffi::c_void) -> HvReturn;
    fn hv_vm_destroy() -> HvReturn;
    fn hv_vm_map(
        addr: *mut std::ffi::c_void,
        ipa: HvIpa,
        size: usize,
        flags: HvMemoryFlags,
    ) -> HvReturn;
    fn hv_vm_unmap(ipa: HvIpa, size: usize) -> HvReturn;
    fn hv_vm_protect(ipa: HvIpa, size: usize, flags: HvMemoryFlags) -> HvReturn;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn check_hv(label: &str, ret: HvReturn) -> bool {
    if ret != HV_SUCCESS {
        eprintln!(
            "  FAIL: {} returned {} (0x{:08X} = {})",
            label,
            ret,
            ret as u32,
            hv_error_name(ret)
        );
        false
    } else {
        true
    }
}

fn size_label(size: usize) -> String {
    if size >= 1024 * 1024 {
        format!("{}MB", size / (1024 * 1024))
    } else {
        format!("{}KB", size / 1024)
    }
}

fn page_size() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

/// Allocate page-aligned memory via mmap and mlock it.
unsafe fn alloc_locked(size: usize) -> *mut u8 {
    let ptr = libc::mmap(
        ptr::null_mut(),
        size,
        libc::PROT_READ | libc::PROT_WRITE,
        libc::MAP_PRIVATE | libc::MAP_ANON,
        -1,
        0,
    );
    if ptr == libc::MAP_FAILED {
        eprintln!("  mmap failed for size {}", size_label(size));
        return ptr::null_mut();
    }
    // Touch every page to fault them in before mlock
    let pgsz = page_size();
    let slice = std::slice::from_raw_parts_mut(ptr as *mut u8, size);
    for i in (0..size).step_by(pgsz) {
        slice[i] = 0;
    }
    let ret = libc::mlock(ptr, size);
    if ret != 0 {
        let err = *libc::__error();
        eprintln!(
            "  mlock failed for size {} (errno={})",
            size_label(size),
            err
        );
    }
    ptr as *mut u8
}

unsafe fn free_locked(ptr: *mut u8, size: usize) {
    if !ptr.is_null() {
        libc::munlock(ptr as *mut libc::c_void, size);
        libc::munmap(ptr as *mut libc::c_void, size);
    }
}

// ---------------------------------------------------------------------------
// Latency measurement helpers
// ---------------------------------------------------------------------------

const LATENCY_ITERS: u64 = 1_000_000;

/// Measure average host-side read latency (volatile reads, strided by 64B).
unsafe fn measure_read_latency_ns(ptr: *const u8, size: usize) -> f64 {
    let mut sum: u64 = 0;
    let start = Instant::now();
    for i in 0..LATENCY_ITERS {
        let offset = ((i * 64) as usize) % size;
        sum = sum.wrapping_add(ptr::read_volatile(ptr.add(offset)) as u64);
    }
    let elapsed = start.elapsed();
    std::hint::black_box(sum);
    elapsed.as_nanos() as f64 / LATENCY_ITERS as f64
}

/// Measure average host-side write latency (volatile writes, strided by 64B).
unsafe fn measure_write_latency_ns(ptr: *mut u8, size: usize) -> f64 {
    let start = Instant::now();
    for i in 0..LATENCY_ITERS {
        let offset = ((i * 64) as usize) % size;
        ptr::write_volatile(ptr.add(offset), (i & 0xFF) as u8);
    }
    let elapsed = start.elapsed();
    elapsed.as_nanos() as f64 / LATENCY_ITERS as f64
}

/// Write a deterministic pattern: byte at offset i = (i & 0xFF).
unsafe fn write_pattern(ptr: *mut u8, size: usize) {
    let slice = std::slice::from_raw_parts_mut(ptr, size);
    for i in 0..size {
        slice[i] = (i & 0xFF) as u8;
    }
}

/// Verify the pattern written by write_pattern(). Returns true if OK.
unsafe fn verify_pattern(ptr: *const u8, size: usize) -> bool {
    let slice = std::slice::from_raw_parts(ptr, size);
    // Check first page and last page
    let pgsz = page_size();
    let check_len = std::cmp::min(pgsz, size);
    for i in 0..check_len {
        if slice[i] != (i & 0xFF) as u8 {
            eprintln!(
                "  DATA MISMATCH at offset {}: expected 0x{:02X}, got 0x{:02X}",
                i,
                (i & 0xFF) as u8,
                slice[i]
            );
            return false;
        }
    }
    if size > pgsz {
        for i in (size - check_len)..size {
            if slice[i] != (i & 0xFF) as u8 {
                eprintln!(
                    "  DATA MISMATCH at offset {}: expected 0x{:02X}, got 0x{:02X}",
                    i,
                    (i & 0xFF) as u8,
                    slice[i]
                );
                return false;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Main experiment
// ---------------------------------------------------------------------------

fn main() {
    let pgsz = page_size();

    println!("============================================================");
    println!("  hyp_probe: Hypervisor.framework Memory Mapping Experiment");
    println!("  Platform: Apple Silicon (arm64)");
    println!("  System page size: {} bytes ({})", pgsz, size_label(pgsz));
    println!("============================================================");
    println!();

    // -----------------------------------------------------------------------
    // Step 1: Create VM
    // -----------------------------------------------------------------------
    println!("[1] Creating VM via hv_vm_create(NULL)...");
    let t0 = Instant::now();
    let ret = unsafe { hv_vm_create(ptr::null()) };
    let create_us = t0.elapsed().as_micros();
    if !check_hv("hv_vm_create", ret) {
        eprintln!();
        eprintln!("*** hv_vm_create failed. Possible causes:");
        eprintln!("    - Missing entitlement: com.apple.security.hypervisor");
        eprintln!("    - Binary not codesigned with entitlement plist");
        eprintln!("    - Another VM already created in this process");
        eprintln!("    - SIP or MDM policy blocking hypervisor access");
        std::process::exit(1);
    }
    println!("  OK -- hv_vm_create succeeded in {} us", create_us);
    println!();

    // -----------------------------------------------------------------------
    // Step 2: Test sub-page size (4KB) -- expect failure on 16KB-page systems
    // -----------------------------------------------------------------------
    if pgsz > 4096 {
        println!("------------------------------------------------------------");
        println!("[Test 0] Size = 4KB (sub-page, expect HV_BAD_ARGUMENT)");
        println!("------------------------------------------------------------");
        let small_size = 4096usize;
        let small_ptr = unsafe { alloc_locked(small_size) };
        if !small_ptr.is_null() {
            let ret = unsafe {
                hv_vm_map(
                    small_ptr as *mut std::ffi::c_void,
                    0x1000_0000,
                    small_size,
                    HV_MEMORY_READ | HV_MEMORY_WRITE,
                )
            };
            if ret == HV_SUCCESS {
                println!("  UNEXPECTED: 4KB map succeeded on {}-byte page system", pgsz);
                unsafe { hv_vm_unmap(0x1000_0000, small_size) };
            } else {
                println!(
                    "  CONFIRMED: 4KB map fails with {} -- minimum mapping size = {} (page size)",
                    hv_error_name(ret),
                    size_label(pgsz)
                );
            }
            unsafe { free_locked(small_ptr, small_size) };
        }
        println!();
    }

    // -----------------------------------------------------------------------
    // Step 3-8: Test each valid memory size
    // -----------------------------------------------------------------------
    let sizes: &[(usize, &str)] = &[
        (16 * 1024, "16KB"),       // 1 page on Apple Silicon
        (1024 * 1024, "1MB"),      // 64 pages
        (16 * 1024 * 1024, "16MB"),    // 1024 pages
        (256 * 1024 * 1024, "256MB"),  // 16384 pages
    ];

    // IPA bases (well-separated, all page-aligned)
    let ipa_bases: &[HvIpa] = &[
        0x4000_0000,       // 1 GB
        0x8000_0000,       // 2 GB
        0xC000_0000,       // 3 GB
        0x1_0000_0000,     // 4 GB
    ];

    for (idx, &(size, label)) in sizes.iter().enumerate() {
        println!("------------------------------------------------------------");
        println!("[Test {}] Size = {} ({} pages)", idx + 1, label, size / pgsz);
        println!("------------------------------------------------------------");

        let ipa = ipa_bases[idx];

        // Allocate + mlock
        println!("  Allocating {} via mmap + mlock...", label);
        let ptr = unsafe { alloc_locked(size) };
        if ptr.is_null() {
            eprintln!("  SKIP: allocation failed for {}", label);
            println!();
            continue;
        }
        println!("  Allocated at host VA: {:p} (page-aligned: {})", ptr, (ptr as usize) % pgsz == 0);

        // Write pattern and verify it before anything else
        println!("  Writing deterministic pattern...");
        unsafe { write_pattern(ptr, size) };

        // Measure host access latency BEFORE hv_vm_map
        // (read latency measured on a COPY of pattern to not disturb it)
        let read_lat_before = unsafe { measure_read_latency_ns(ptr, size) };
        println!("  Host read  latency BEFORE hv_vm_map: {:.1} ns/op", read_lat_before);

        // Re-write pattern (read latency measurement did reads only, but be safe)
        unsafe { write_pattern(ptr, size) };
        let write_lat_before = unsafe { measure_write_latency_ns(ptr, size) };
        println!("  Host write latency BEFORE hv_vm_map: {:.1} ns/op", write_lat_before);

        // Re-write pattern since write latency test corrupted it
        unsafe { write_pattern(ptr, size) };

        // Map into guest IPA space
        println!("  Mapping into guest IPA 0x{:X} (RW)...", ipa);
        let t_map = Instant::now();
        let ret = unsafe {
            hv_vm_map(
                ptr as *mut std::ffi::c_void,
                ipa,
                size,
                HV_MEMORY_READ | HV_MEMORY_WRITE,
            )
        };
        let map_ns = t_map.elapsed().as_nanos();
        if !check_hv("hv_vm_map", ret) {
            eprintln!("  SKIP: hv_vm_map failed for {}", label);
            unsafe { free_locked(ptr, size) };
            println!();
            continue;
        }
        println!("  OK -- hv_vm_map took {} ns ({} us)", map_ns, map_ns / 1000);

        // Verify host-side data integrity is preserved after mapping
        println!("  Verifying host-side data integrity after map...");
        if unsafe { verify_pattern(ptr, size) } {
            println!("  OK -- data integrity verified (map does not corrupt host memory)");
        } else {
            eprintln!("  WARNING: data corruption detected after hv_vm_map!");
        }

        // Write NEW data from host side after mapping, verify readback
        println!("  Writing new pattern from host side (post-map)...");
        unsafe {
            let slice = std::slice::from_raw_parts_mut(ptr, size);
            for i in 0..size {
                slice[i] = ((i.wrapping_mul(7)) & 0xFF) as u8;
            }
        }
        let mut post_write_ok = true;
        unsafe {
            let slice = std::slice::from_raw_parts(ptr, size);
            for i in 0..std::cmp::min(pgsz, size) {
                if slice[i] != ((i.wrapping_mul(7)) & 0xFF) as u8 {
                    eprintln!("  POST-WRITE MISMATCH at offset {}", i);
                    post_write_ok = false;
                    break;
                }
            }
        }
        if post_write_ok {
            println!("  OK -- post-map host write + readback verified");
        }

        // Measure host access latency AFTER hv_vm_map
        let read_lat_after = unsafe { measure_read_latency_ns(ptr, size) };
        let write_lat_after = unsafe { measure_write_latency_ns(ptr, size) };
        println!("  Host read  latency AFTER  hv_vm_map: {:.1} ns/op", read_lat_after);
        println!("  Host write latency AFTER  hv_vm_map: {:.1} ns/op", write_lat_after);
        println!(
            "  Delta (after - before): read {:+.1} ns, write {:+.1} ns",
            read_lat_after - read_lat_before,
            write_lat_after - write_lat_before
        );

        // Test hv_vm_protect: RW -> R
        println!("  Testing hv_vm_protect (RW -> R)...");
        let t_prot = Instant::now();
        let ret = unsafe { hv_vm_protect(ipa, size, HV_MEMORY_READ) };
        let prot_ns = t_prot.elapsed().as_nanos();
        if check_hv("hv_vm_protect(R)", ret) {
            println!("  OK -- hv_vm_protect(R) took {} ns", prot_ns);
        }

        // Test hv_vm_protect: R -> RWX
        println!("  Testing hv_vm_protect (R -> RWX)...");
        let t_prot2 = Instant::now();
        let ret = unsafe {
            hv_vm_protect(ipa, size, HV_MEMORY_READ | HV_MEMORY_WRITE | HV_MEMORY_EXEC)
        };
        let prot2_ns = t_prot2.elapsed().as_nanos();
        if check_hv("hv_vm_protect(RWX)", ret) {
            println!("  OK -- hv_vm_protect(RWX) took {} ns", prot2_ns);
        }

        // Test hv_vm_protect: RWX -> NONE
        println!("  Testing hv_vm_protect (RWX -> NONE)...");
        let t_prot3 = Instant::now();
        let ret = unsafe { hv_vm_protect(ipa, size, 0) };
        let prot3_ns = t_prot3.elapsed().as_nanos();
        if check_hv("hv_vm_protect(NONE)", ret) {
            println!("  OK -- hv_vm_protect(NONE) took {} ns", prot3_ns);
        }

        // Host access after protect(NONE) -- guest perms only, host unaffected
        let read_lat_prot = unsafe { measure_read_latency_ns(ptr, size) };
        println!(
            "  Host read  latency AFTER  protect(NONE): {:.1} ns/op (guest perms only)",
            read_lat_prot
        );

        // Unmap
        println!("  Unmapping...");
        let t_unmap = Instant::now();
        let ret = unsafe { hv_vm_unmap(ipa, size) };
        let unmap_ns = t_unmap.elapsed().as_nanos();
        if check_hv("hv_vm_unmap", ret) {
            println!("  OK -- hv_vm_unmap took {} ns ({} us)", unmap_ns, unmap_ns / 1000);
        }

        // Host access after unmap -- memory is still ours
        let read_lat_unmap = unsafe { measure_read_latency_ns(ptr, size) };
        println!("  Host read  latency AFTER  hv_vm_unmap:  {:.1} ns/op", read_lat_unmap);

        // Double-unmap test
        println!("  Testing double-unmap (expect error)...");
        let ret = unsafe { hv_vm_unmap(ipa, size) };
        if ret == HV_SUCCESS {
            println!("  NOTE: double-unmap returned HV_SUCCESS (silent no-op)");
        } else {
            println!(
                "  OK -- double-unmap returned {} ({})",
                hv_error_name(ret),
                ret
            );
        }

        // Free host memory
        unsafe { free_locked(ptr, size) };

        // Summary
        println!();
        println!("  --- Summary for {} ({} pages) ---", label, size / pgsz);
        println!("  hv_vm_map      : {} ns ({} us)", map_ns, map_ns / 1000);
        println!("  hv_vm_protect  : {} / {} / {} ns (R / RWX / NONE)", prot_ns, prot2_ns, prot3_ns);
        println!("  hv_vm_unmap    : {} ns ({} us)", unmap_ns, unmap_ns / 1000);
        println!(
            "  Read  latency  : {:.1} -> {:.1} -> {:.1} ns (before/after-map/after-unmap)",
            read_lat_before, read_lat_after, read_lat_unmap
        );
        println!(
            "  Write latency  : {:.1} -> {:.1} ns (before/after-map)",
            write_lat_before, write_lat_after
        );
        println!();
    }

    // -----------------------------------------------------------------------
    // Micro-benchmark: repeated map/unmap of single page (16KB)
    // -----------------------------------------------------------------------
    println!("============================================================");
    println!("[Micro-benchmark] Repeated map/unmap of 1 page ({}) -- 1000 iterations", size_label(pgsz));
    println!("============================================================");

    let bench_size = pgsz;
    let bench_ipa_base: HvIpa = 0x2_0000_0000;
    let bench_ptr = unsafe { alloc_locked(bench_size) };
    if !bench_ptr.is_null() {
        let iters = 1000u64;
        let ipa_stride = pgsz as u64; // each mapping at next page-aligned IPA

        // Map 1000 pages at distinct IPAs
        let mut map_ok = 0u64;
        let t_bench_map = Instant::now();
        for i in 0..iters {
            let ret = unsafe {
                hv_vm_map(
                    bench_ptr as *mut std::ffi::c_void,
                    bench_ipa_base + (i * ipa_stride),
                    bench_size,
                    HV_MEMORY_READ | HV_MEMORY_WRITE,
                )
            };
            if ret != HV_SUCCESS {
                eprintln!("  map failed at iter {}: {}", i, hv_error_name(ret));
                break;
            }
            map_ok += 1;
        }
        let map_total_ns = t_bench_map.elapsed().as_nanos();

        // Unmap all that succeeded
        let t_bench_unmap = Instant::now();
        for i in 0..map_ok {
            let ret = unsafe { hv_vm_unmap(bench_ipa_base + (i * ipa_stride), bench_size) };
            if ret != HV_SUCCESS {
                eprintln!("  unmap failed at iter {}: {}", i, hv_error_name(ret));
                break;
            }
        }
        let unmap_total_ns = t_bench_unmap.elapsed().as_nanos();

        if map_ok > 0 {
            println!(
                "  map   {}x{}: {} ns total, {:.0} ns/op ({:.1} us/op)",
                map_ok,
                size_label(bench_size),
                map_total_ns,
                map_total_ns as f64 / map_ok as f64,
                map_total_ns as f64 / map_ok as f64 / 1000.0,
            );
            println!(
                "  unmap {}x{}: {} ns total, {:.0} ns/op ({:.1} us/op)",
                map_ok,
                size_label(bench_size),
                unmap_total_ns,
                unmap_total_ns as f64 / map_ok as f64,
                unmap_total_ns as f64 / map_ok as f64 / 1000.0,
            );
        }

        // Protect benchmark: map once, toggle permissions 1000 times
        println!();
        println!("  Repeated protect of 1 page -- 1000 iterations (toggling RW <-> R)...");
        let prot_ipa: HvIpa = 0x3_0000_0000;
        let ret = unsafe {
            hv_vm_map(
                bench_ptr as *mut std::ffi::c_void,
                prot_ipa,
                bench_size,
                HV_MEMORY_READ | HV_MEMORY_WRITE,
            )
        };
        if ret == HV_SUCCESS {
            let mut prot_ok = 0u64;
            let t_bench_prot = Instant::now();
            for i in 0..iters {
                let flags = if i % 2 == 0 {
                    HV_MEMORY_READ
                } else {
                    HV_MEMORY_READ | HV_MEMORY_WRITE
                };
                let ret = unsafe { hv_vm_protect(prot_ipa, bench_size, flags) };
                if ret != HV_SUCCESS {
                    eprintln!("  protect failed at iter {}: {}", i, hv_error_name(ret));
                    break;
                }
                prot_ok += 1;
            }
            let prot_total_ns = t_bench_prot.elapsed().as_nanos();
            if prot_ok > 0 {
                println!(
                    "  protect {}x{}: {} ns total, {:.0} ns/op ({:.1} us/op)",
                    prot_ok,
                    size_label(bench_size),
                    prot_total_ns,
                    prot_total_ns as f64 / prot_ok as f64,
                    prot_total_ns as f64 / prot_ok as f64 / 1000.0,
                );
            }
            unsafe { hv_vm_unmap(prot_ipa, bench_size) };
        }

        // Map/unmap same IPA repeatedly (measures re-use overhead)
        println!();
        println!("  Repeated map+unmap of SAME IPA -- 1000 iterations...");
        let same_ipa: HvIpa = 0x4_0000_0000;
        let mut same_ok = 0u64;
        let t_same = Instant::now();
        for _ in 0..iters {
            let ret = unsafe {
                hv_vm_map(
                    bench_ptr as *mut std::ffi::c_void,
                    same_ipa,
                    bench_size,
                    HV_MEMORY_READ | HV_MEMORY_WRITE,
                )
            };
            if ret != HV_SUCCESS {
                break;
            }
            let ret = unsafe { hv_vm_unmap(same_ipa, bench_size) };
            if ret != HV_SUCCESS {
                break;
            }
            same_ok += 1;
        }
        let same_total_ns = t_same.elapsed().as_nanos();
        if same_ok > 0 {
            println!(
                "  map+unmap {}x same IPA: {} ns total, {:.0} ns/pair ({:.1} us/pair)",
                same_ok,
                same_total_ns,
                same_total_ns as f64 / same_ok as f64,
                same_total_ns as f64 / same_ok as f64 / 1000.0,
            );
        }

        unsafe { free_locked(bench_ptr, bench_size) };
    }
    println!();

    // -----------------------------------------------------------------------
    // Step 9: Destroy VM
    // -----------------------------------------------------------------------
    println!("[9] Destroying VM via hv_vm_destroy...");
    let t_destroy = Instant::now();
    let ret = unsafe { hv_vm_destroy() };
    let destroy_ns = t_destroy.elapsed().as_nanos();
    if check_hv("hv_vm_destroy", ret) {
        println!("  OK -- hv_vm_destroy succeeded in {} ns ({} us)", destroy_ns, destroy_ns / 1000);
    }

    // Double-destroy test
    println!("  Testing double-destroy (expect error)...");
    let ret = unsafe { hv_vm_destroy() };
    if ret == HV_SUCCESS {
        println!("  UNEXPECTED: double-destroy succeeded");
    } else {
        println!(
            "  OK -- double-destroy correctly returned {} (0x{:08X})",
            hv_error_name(ret),
            ret as u32
        );
    }

    println!();
    println!("============================================================");
    println!("  hyp_probe complete.");
    println!("============================================================");
}
