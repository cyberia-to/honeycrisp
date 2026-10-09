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
pub type CFDictionaryRef = *const c_void;
pub type CFMutableDictionaryRef = *mut c_void;
/// SDK signed pointer-width index (`long` on the supported Apple LP64 target).
pub type CFIndex = isize;
pub type CFNumberType = CFIndex;
pub type kern_return_t = i32;

pub const KERN_SUCCESS: kern_return_t = 0;
pub const kCFNumberSInt64Type: CFNumberType = 4;

// IOSurface

#[link(name = "IOSurface", kind = "framework")]
extern "C" {
    pub fn IOSurfaceCreate(properties: CFDictionaryRef) -> IOSurfaceRef;
    pub fn IOSurfaceLock(surface: IOSurfaceRef, options: u32, seed: *mut u32) -> kern_return_t;
    pub fn IOSurfaceUnlock(surface: IOSurfaceRef, options: u32, seed: *mut u32) -> kern_return_t;
    pub fn IOSurfaceGetBaseAddress(surface: IOSurfaceRef) -> *mut c_void;
    pub fn IOSurfaceGetAllocSize(surface: IOSurfaceRef) -> usize;
    pub fn IOSurfaceGetID(surface: IOSurfaceRef) -> u32;
    pub fn IOSurfaceGetPropertyAlignment(property: CFStringRef) -> usize;
    pub fn IOSurfaceAlignProperty(property: CFStringRef, value: usize) -> usize;

    pub static kIOSurfaceWidth: CFStringRef;
    pub static kIOSurfaceHeight: CFStringRef;
    pub static kIOSurfaceBytesPerElement: CFStringRef;
    pub static kIOSurfaceBytesPerRow: CFStringRef;
    pub static kIOSurfaceAllocSize: CFStringRef;
    pub static kIOSurfacePixelFormat: CFStringRef;
}

// CoreFoundation

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    pub fn CFDictionaryCreate(
        allocator: *const c_void,
        keys: *const *const c_void,
        values: *const *const c_void,
        count: CFIndex,
        keyCallBacks: *const c_void,
        valueCallBacks: *const c_void,
    ) -> CFDictionaryRef;
    pub fn CFDictionaryCreateMutable(
        allocator: *const c_void,
        capacity: CFIndex,
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
        theType: CFNumberType,
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
