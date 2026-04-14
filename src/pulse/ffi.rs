//! Dynamic loading of libkperf.dylib and kpc_* symbol resolution.
//!
//! Apple does not ship public headers for kpc, but the symbols are
//! available in `/usr/lib/libkperf.dylib` on macOS.  We resolve them
//! at runtime via `dlopen` / `dlsym` and cache the function pointers
//! in a `OnceLock` so the cost is paid at most once per process.

use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Fixed counters (cycles, instructions).
pub const KPC_CLASS_FIXED: u32 = 1;
/// Configurable PMU counters.
pub const KPC_CLASS_CONFIGURABLE: u32 = 2;

// ---------------------------------------------------------------------------
// FFI: dlopen / dlsym from libdl
// ---------------------------------------------------------------------------

extern "C" {
    fn dlopen(filename: *const u8, flags: i32) -> *mut u8;
    fn dlsym(handle: *mut u8, symbol: *const u8) -> *mut u8;
}

const RTLD_NOW: i32 = 2;

// ---------------------------------------------------------------------------
// Resolved function pointer table
// ---------------------------------------------------------------------------

type KpcSetConfig = unsafe extern "C" fn(classes: u32, config: *const u64) -> i32;
type KpcSetCounting = unsafe extern "C" fn(classes: u32) -> i32;
type KpcSetThreadCounting = unsafe extern "C" fn(classes: u32) -> i32;
type KpcGetThreadCounters64 = unsafe extern "C" fn(tid: u32, buf_count: u32, buf: *mut u64) -> i32;
type KpcGetCounterCount = unsafe extern "C" fn(classes: u32) -> u32;
type KpcGetConfigCount = unsafe extern "C" fn(classes: u32) -> u32;

pub(crate) struct KpcVtable {
    pub set_config: KpcSetConfig,
    pub set_counting: KpcSetCounting,
    pub set_thread_counting: KpcSetThreadCounting,
    pub get_thread_counters: KpcGetThreadCounters64,
    pub get_counter_count: KpcGetCounterCount,
    pub get_config_count: KpcGetConfigCount,
}

static VTABLE: OnceLock<Result<KpcVtable, &'static str>> = OnceLock::new();

/// Resolve all kpc_* symbols.  Returns a cached reference to the vtable,
/// or an error string if libkperf.dylib could not be loaded.
pub(crate) fn vtable() -> Result<&'static KpcVtable, &'static str> {
    VTABLE
        .get_or_init(|| unsafe {
            let handle = dlopen(c"/usr/lib/libkperf.dylib".as_ptr().cast(), RTLD_NOW);
            if handle.is_null() {
                return Err("dlopen libkperf.dylib failed");
            }

            let resolve = |name: &std::ffi::CStr| -> Result<*mut u8, &'static str> {
                let ptr = dlsym(handle, name.as_ptr().cast());
                if ptr.is_null() {
                    Err("dlsym failed for kpc symbol")
                } else {
                    Ok(ptr)
                }
            };

            Ok(KpcVtable {
                set_config: std::mem::transmute::<*mut u8, KpcSetConfig>(resolve(
                    c"kpc_set_config",
                )?),
                set_counting: std::mem::transmute::<*mut u8, KpcSetCounting>(resolve(
                    c"kpc_set_counting",
                )?),
                set_thread_counting: std::mem::transmute::<*mut u8, KpcSetThreadCounting>(resolve(
                    c"kpc_set_thread_counting",
                )?),
                get_thread_counters: std::mem::transmute::<*mut u8, KpcGetThreadCounters64>(
                    resolve(c"kpc_get_thread_counters64")?,
                ),
                get_counter_count: std::mem::transmute::<*mut u8, KpcGetCounterCount>(resolve(
                    c"kpc_get_counter_count",
                )?),
                get_config_count: std::mem::transmute::<*mut u8, KpcGetConfigCount>(resolve(
                    c"kpc_get_config_count",
                )?),
            })
        })
        .as_ref()
        .map_err(|e| *e)
}
