use std::ffi::c_void;
use unimem::ffi::*;

const _: i32 = kCFNumberSInt64Type;
const _: unsafe extern "C" fn(*const c_void, i32, *const c_void) -> *const c_void = CFNumberCreate;
const _: unsafe extern "C" fn(*const c_void, i64, *const c_void, *const c_void) -> CFMutableDictionaryRef = CFDictionaryCreateMutable;
const _: unsafe extern "C" fn(CFMutableDictionaryRef) -> IOSurfaceRef = IOSurfaceCreate;

// Retained roundtrip fixture: the old literal and the explicit bits are equal.
const _: () = assert!(3.14f32.to_bits() == 0x4048_f5c3);
