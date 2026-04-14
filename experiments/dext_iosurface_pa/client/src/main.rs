//! CybMemDriver client -- IOSurface physical address resolution via DEXT
//!
//! PATH 2 experiment: IOSurface alloc -> DEXT reads physical addresses
//!
//! This client:
//!   1. Creates an IOSurface of configurable size
//!   2. Writes a known test pattern into the surface
//!   3. Connects to the CybMemDriver DEXT via IOServiceOpen
//!   4. Sends the IOSurface ID + size to the DEXT
//!   5. Receives back physical address segments
//!   6. Prints the PA map and verifies data integrity

#![allow(
    non_camel_case_types,
    non_upper_case_globals,
    non_snake_case,
    dead_code
)]

use std::ffi::{c_char, c_void, CStr};
use std::mem;
use std::ptr;

// ============================================================================
// Constants
// ============================================================================

const KERN_SUCCESS: i32 = 0;
const K_IO_MASTER_PORT_DEFAULT: u32 = 0;
const K_CF_STRING_ENCODING_UTF8: u32 = 0x08000100;
const K_CF_NUMBER_SINT32_TYPE: i32 = 3;

/// Page size on Apple Silicon (and modern Intel Macs)
const PAGE_SIZE: usize = 16384; // 16 KiB on arm64, 4 KiB on x86_64 -- we use 16K for safety

/// Test surface size: 4 pages = 64 KiB
const SURFACE_BYTES: usize = 4 * PAGE_SIZE;

/// ExternalMethod selector matching kCybMemDriverMethodGetPhysAddrs in the DEXT
const SELECTOR_GET_PHYS_ADDRS: u32 = 0;

/// Maximum segments returned by the DEXT (must match CYBMEM_MAX_SEGMENTS)
const MAX_SEGMENTS: usize = 32;

/// IOKit class name for our DEXT service
const DEXT_CLASS_NAME: &str = "CybMemDriver";

// ============================================================================
// FFI type aliases
// ============================================================================

type io_object_t = u32;
type io_service_t = io_object_t;
type io_connect_t = u32;
type io_iterator_t = u32;
type mach_port_t = u32;
type kern_return_t = i32;
type IOSurfaceRef = *mut c_void;
type CFMutableDictionaryRef = *mut c_void;
type CFStringRef = *const c_void;
type CFTypeRef = *const c_void;

// ============================================================================
// Structures matching the DEXT's wire format (must be identical layout)
// ============================================================================

/// Input to the DEXT: identifies the IOSurface and its size
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CybMemInput {
    surface_id: u32,
    _pad0: u32,
    byte_length: u64,
}

/// A single physical segment returned by the DEXT
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct CybMemPhysSegment {
    phys_addr: u64,
    length: u64,
}

/// Output from the DEXT: array of physical address segments
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CybMemOutput {
    num_segments: u32,
    _pad0: u32,
    total_length: u64,
    segments: [CybMemPhysSegment; MAX_SEGMENTS],
}

impl Default for CybMemOutput {
    fn default() -> Self {
        Self {
            num_segments: 0,
            _pad0: 0,
            total_length: 0,
            segments: [CybMemPhysSegment::default(); MAX_SEGMENTS],
        }
    }
}

// ============================================================================
// IOKit FFI bindings
// ============================================================================

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
    fn IOIteratorNext(iterator: io_iterator_t) -> io_object_t;
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
    fn IOObjectRelease(object: io_object_t) -> kern_return_t;
    fn IOObjectGetClass(object: io_object_t, class_name: *mut c_char) -> kern_return_t;
}

extern "C" {
    fn mach_task_self() -> mach_port_t;
}

// ============================================================================
// CoreFoundation FFI (for IOSurface property dictionary)
// ============================================================================

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

// ============================================================================
// IOSurface FFI
// ============================================================================

#[link(name = "IOSurface", kind = "framework")]
extern "C" {
    fn IOSurfaceCreate(properties: CFMutableDictionaryRef) -> IOSurfaceRef;
    fn IOSurfaceLock(surface: IOSurfaceRef, options: u32, seed: *mut u32) -> kern_return_t;
    fn IOSurfaceUnlock(surface: IOSurfaceRef, options: u32, seed: *mut u32) -> kern_return_t;
    fn IOSurfaceGetBaseAddress(surface: IOSurfaceRef) -> *mut c_void;
    fn IOSurfaceGetAllocSize(surface: IOSurfaceRef) -> usize;
    fn IOSurfaceGetID(surface: IOSurfaceRef) -> u32;
}

// ============================================================================
// CF helper functions (reused pattern from rane/src/ffi.rs)
// ============================================================================

fn cf_str(s: &str) -> CFStringRef {
    unsafe {
        let c = std::ffi::CString::new(s).unwrap();
        CFStringCreateWithCString(ptr::null(), c.as_ptr(), K_CF_STRING_ENCODING_UTF8)
    }
}

fn cf_num(v: i32) -> *const c_void {
    unsafe {
        CFNumberCreate(
            ptr::null(),
            K_CF_NUMBER_SINT32_TYPE,
            &v as *const i32 as *const c_void,
        )
    }
}

fn kern_err_str(kr: kern_return_t) -> String {
    match kr {
        0 => "KERN_SUCCESS".into(),
        _ => format!("{:#010x}", kr as u32),
    }
}

// ============================================================================
// IOSurface creation (pattern from rane/src/surface.rs)
// ============================================================================

/// Create an IOSurface of the given byte size.
/// Returns (IOSurfaceRef, actual_alloc_size).
fn create_iosurface(bytes: usize) -> Result<(IOSurfaceRef, usize), String> {
    unsafe {
        // Build the property dictionary that describes the surface layout.
        // We create a 1D surface: width=bytes, height=1, bytesPerElement=1.
        let dict = CFDictionaryCreateMutable(
            ptr::null(),
            0,
            &kCFTypeDictionaryKeyCallBacks as *const c_void,
            &kCFTypeDictionaryValueCallBacks as *const c_void,
        );

        // IOSurface properties -- flat 1D buffer
        CFDictionarySetValue(dict, cf_str("IOSurfaceWidth") as _, cf_num(bytes as i32));
        CFDictionarySetValue(dict, cf_str("IOSurfaceHeight") as _, cf_num(1));
        CFDictionarySetValue(dict, cf_str("IOSurfaceBytesPerElement") as _, cf_num(1));
        CFDictionarySetValue(
            dict,
            cf_str("IOSurfaceBytesPerRow") as _,
            cf_num(bytes as i32),
        );
        CFDictionarySetValue(
            dict,
            cf_str("IOSurfaceAllocSize") as _,
            cf_num(bytes as i32),
        );
        // PixelFormat 0 = unspecified / raw bytes
        CFDictionarySetValue(dict, cf_str("IOSurfacePixelFormat") as _, cf_num(0));

        let surface = IOSurfaceCreate(dict);
        // dict is consumed by IOSurfaceCreate, no need to release

        if surface.is_null() {
            return Err(format!("IOSurfaceCreate failed for {} bytes", bytes));
        }

        let actual_size = IOSurfaceGetAllocSize(surface);
        Ok((surface, actual_size))
    }
}

/// Write a known test pattern into the IOSurface.
/// Pattern: each byte = (page_index * 16 + offset_in_page) & 0xFF
/// This lets us verify which physical page contains which data.
fn write_test_pattern(surface: IOSurfaceRef, size: usize) {
    unsafe {
        IOSurfaceLock(surface, 0, ptr::null_mut());
        let base = IOSurfaceGetBaseAddress(surface) as *mut u8;

        for i in 0..size {
            let page_idx = i / PAGE_SIZE;
            let offset = i % PAGE_SIZE;
            // Distinctive pattern: page number in high nibble, rolling counter in low byte
            *base.add(i) = ((page_idx.wrapping_mul(37) ^ offset) & 0xFF) as u8;
        }

        IOSurfaceUnlock(surface, 0, ptr::null_mut());
    }
}

/// Read back and verify the test pattern is intact.
fn verify_test_pattern(surface: IOSurfaceRef, size: usize) -> bool {
    unsafe {
        IOSurfaceLock(surface, 1, ptr::null_mut()); // read-only lock
        let base = IOSurfaceGetBaseAddress(surface) as *const u8;

        let mut ok = true;
        let mut mismatches = 0u64;
        for i in 0..size {
            let page_idx = i / PAGE_SIZE;
            let offset = i % PAGE_SIZE;
            let expected = ((page_idx.wrapping_mul(37) ^ offset) & 0xFF) as u8;
            let actual = *base.add(i);
            if actual != expected {
                if mismatches < 5 {
                    eprintln!(
                        "  MISMATCH at offset {:#x}: expected {:#04x}, got {:#04x}",
                        i, expected, actual
                    );
                }
                mismatches += 1;
                ok = false;
            }
        }
        if mismatches > 5 {
            eprintln!("  ... and {} more mismatches", mismatches - 5);
        }

        IOSurfaceUnlock(surface, 1, ptr::null_mut());
        ok
    }
}

// ============================================================================
// IOKit service connection
// ============================================================================

/// Find and open a connection to the CybMemDriver DEXT.
/// Returns the io_connect_t handle on success.
fn open_dext_connection() -> Result<io_connect_t, String> {
    unsafe {
        // Create a matching dictionary for our DEXT class name
        let class_name = std::ffi::CString::new(DEXT_CLASS_NAME).unwrap();
        let matching = IOServiceMatching(class_name.as_ptr());
        if matching.is_null() {
            return Err("IOServiceMatching returned null".into());
        }

        // Find the service in the IOKit registry
        let service = IOServiceGetMatchingService(K_IO_MASTER_PORT_DEFAULT, matching);
        // matching dict is consumed by IOServiceGetMatchingService

        if service == 0 {
            return Err(format!(
                "IOServiceGetMatchingService found no '{}' service. \
                 Is the DEXT loaded? Check: sudo systemextensionsctl list",
                DEXT_CLASS_NAME
            ));
        }

        // Print the service class for debugging
        let mut class_buf = [0i8; 128];
        IOObjectGetClass(service, class_buf.as_mut_ptr());
        let class_str = CStr::from_ptr(class_buf.as_ptr());
        println!("[+] Found service: class={}", class_str.to_string_lossy());

        // Open a user client connection (type 0 = default)
        let mut connection: io_connect_t = 0;
        let kr = IOServiceOpen(service, mach_task_self(), 0, &mut connection);
        IOObjectRelease(service);

        if kr != KERN_SUCCESS {
            return Err(format!(
                "IOServiceOpen failed: {} ({})",
                kern_err_str(kr),
                kr
            ));
        }

        println!("[+] Opened connection: {}", connection);
        Ok(connection)
    }
}

/// Send the IOSurface info to the DEXT and receive physical addresses back.
fn get_physical_addresses(
    connection: io_connect_t,
    surface_id: u32,
    byte_length: u64,
) -> Result<CybMemOutput, String> {
    let input = CybMemInput {
        surface_id,
        _pad0: 0,
        byte_length,
    };

    let mut output = CybMemOutput::default();
    let mut output_size = mem::size_of::<CybMemOutput>();

    unsafe {
        let kr = IOConnectCallStructMethod(
            connection,
            SELECTOR_GET_PHYS_ADDRS,
            &input as *const CybMemInput as *const c_void,
            mem::size_of::<CybMemInput>(),
            &mut output as *mut CybMemOutput as *mut c_void,
            &mut output_size,
        );

        if kr != KERN_SUCCESS {
            return Err(format!(
                "IOConnectCallStructMethod failed: {} ({})",
                kern_err_str(kr),
                kr
            ));
        }

        println!(
            "[+] Received {} bytes of output ({} expected)",
            output_size,
            mem::size_of::<CybMemOutput>()
        );
    }

    Ok(output)
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("=== CybMemDriver Client -- IOSurface PA Resolution ===");
    println!("PATH 2: IOSurface alloc -> DEXT reads physical addresses");
    println!();

    // ---- Step 1: Create IOSurface ----
    println!("[*] Creating IOSurface ({} bytes = {} pages)...",
             SURFACE_BYTES, SURFACE_BYTES / PAGE_SIZE);

    let (surface, actual_size) = match create_iosurface(SURFACE_BYTES) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[!] Failed to create IOSurface: {}", e);
            std::process::exit(1);
        }
    };

    let surface_id = unsafe { IOSurfaceGetID(surface) };
    println!("[+] IOSurface created:");
    println!("    ID:         {}", surface_id);
    println!("    Alloc size: {} bytes ({} pages)", actual_size, actual_size / PAGE_SIZE);
    println!("    VA base:    {:p}", unsafe { IOSurfaceGetBaseAddress(surface) });
    println!();

    // ---- Step 2: Write test pattern ----
    println!("[*] Writing test pattern...");
    write_test_pattern(surface, actual_size);
    println!("[+] Test pattern written");
    println!();

    // ---- Step 3: Connect to DEXT ----
    println!("[*] Connecting to {} DEXT...", DEXT_CLASS_NAME);
    let connection = match open_dext_connection() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[!] {}", e);
            eprintln!();
            eprintln!("    The DEXT must be loaded before running this client.");
            eprintln!("    See run.sh or load manually:");
            eprintln!("      sudo systemextensionsctl activate <team-id> com.cyb.CybMemDriver");
            eprintln!();
            eprintln!("    For development without SIP, you can also use:");
            eprintln!("      sudo kmutil load -p ./CybMemDriver.dext");
            std::process::exit(1);
        }
    };
    println!();

    // ---- Step 4: Request physical addresses from DEXT ----
    println!("[*] Requesting physical address resolution...");
    println!("    Surface ID:  {}", surface_id);
    println!("    Byte length: {}", actual_size);

    let result = match get_physical_addresses(connection, surface_id, actual_size as u64) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[!] {}", e);
            unsafe { IOServiceClose(connection); }
            std::process::exit(1);
        }
    };
    println!();

    // ---- Step 5: Print physical address map ----
    println!("=== Physical Address Map ===");
    println!("Total length:   {} bytes", result.total_length);
    println!("Segment count:  {}", result.num_segments);
    println!();

    let n = result.num_segments as usize;
    if n == 0 {
        println!("[!] No segments returned -- DEXT may not have PA access");
    } else {
        println!("{:<8} {:>18} {:>12} {:>8}", "Seg#", "Phys Addr", "Length", "Pages");
        println!("{}", "-".repeat(50));

        let mut total_mapped = 0u64;
        for i in 0..n.min(MAX_SEGMENTS) {
            let seg = &result.segments[i];
            let pages = (seg.length + PAGE_SIZE as u64 - 1) / PAGE_SIZE as u64;
            println!(
                "{:<8} {:#018x} {:>10} B {:>6}",
                i, seg.phys_addr, seg.length, pages
            );
            total_mapped += seg.length;
        }

        println!("{}", "-".repeat(50));
        println!("Total mapped: {} bytes ({} pages)",
                 total_mapped, total_mapped / PAGE_SIZE as u64);

        // Check if mapping covers the full surface
        if total_mapped >= actual_size as u64 {
            println!("[+] Full surface coverage verified");
        } else {
            println!("[!] WARNING: mapped {} < surface {} -- partial coverage!",
                     total_mapped, actual_size);
        }
    }
    println!();

    // ---- Step 6: Verify data integrity ----
    println!("[*] Verifying IOSurface data integrity after DEXT interaction...");
    if verify_test_pattern(surface, actual_size) {
        println!("[+] Data integrity OK -- test pattern intact");
    } else {
        println!("[!] Data integrity FAILED -- surface was corrupted!");
    }
    println!();

    // ---- Cleanup ----
    unsafe {
        IOServiceClose(connection);
        CFRelease(surface as CFTypeRef);
    }
    println!("[+] Connection closed, surface released");
    println!("=== Experiment complete ===");
}
