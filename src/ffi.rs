#![allow(
    non_camel_case_types,
    non_upper_case_globals,
    non_snake_case,
    dead_code
)]

use std::ffi::c_void;

pub type IOSurfaceRef = *mut c_void;
pub type CFTypeRef = *const c_void;
pub type CFStringRef = *const c_void;
pub type CFMutableDictionaryRef = *mut c_void;
pub type kern_return_t = i32;

pub const KERN_SUCCESS: kern_return_t = 0;
pub const kCFNumberSInt64Type: i32 = 4;

// IOSurface

#[link(name = "IOSurface", kind = "framework")]
extern "C" {
    pub fn IOSurfaceCreate(properties: CFMutableDictionaryRef) -> IOSurfaceRef;
    pub fn IOSurfaceLock(surface: IOSurfaceRef, options: u32, seed: *mut u32) -> kern_return_t;
    pub fn IOSurfaceUnlock(surface: IOSurfaceRef, options: u32, seed: *mut u32) -> kern_return_t;
    pub fn IOSurfaceGetBaseAddress(surface: IOSurfaceRef) -> *mut c_void;
    pub fn IOSurfaceGetAllocSize(surface: IOSurfaceRef) -> usize;
    pub fn IOSurfaceGetID(surface: IOSurfaceRef) -> u32;
}

// CoreFoundation

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    pub fn CFDictionaryCreateMutable(
        allocator: *const c_void,
        capacity: i64,
        keyCallBacks: *const c_void,
        valueCallBacks: *const c_void,
    ) -> CFMutableDictionaryRef;
    pub fn CFDictionarySetValue(
        dict: CFMutableDictionaryRef,
        key: *const c_void,
        value: *const c_void,
    );
    pub fn CFNumberCreate(
        allocator: *const c_void,
        theType: i32,
        valuePtr: *const c_void,
    ) -> *const c_void;
    pub fn CFStringCreateWithCString(
        alloc: *const c_void,
        cStr: *const i8,
        encoding: u32,
    ) -> CFStringRef;
    pub fn CFRelease(cf: CFTypeRef);

    pub static kCFTypeDictionaryKeyCallBacks: c_void;
    pub static kCFTypeDictionaryValueCallBacks: c_void;
}

pub const kCFStringEncodingUTF8: u32 = 0x08000100;

// Helpers

pub(crate) fn cf_str(s: &str) -> CFStringRef {
    unsafe {
        let c = std::ffi::CString::new(s).unwrap();
        CFStringCreateWithCString(std::ptr::null(), c.as_ptr(), kCFStringEncodingUTF8)
    }
}

pub(crate) fn cf_i64(v: i64) -> *const c_void {
    unsafe {
        CFNumberCreate(
            std::ptr::null(),
            kCFNumberSInt64Type,
            &v as *const i64 as *const c_void,
        )
    }
}
