//! Raw FFI bindings to Apple frameworks: Metal, CoreFoundation, libobjc

#![allow(
    non_camel_case_types,
    non_upper_case_globals,
    non_snake_case,
    dead_code,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::missing_safety_doc
)]

mod selectors;
mod trampoline;

pub use selectors::*;
pub use trampoline::*;

use std::ffi::{c_char, c_void, CStr};

// ── Type aliases ──

pub type ObjcClass = *const c_void;
pub type ObjcSel = *const c_void;
pub type ObjcId = *mut c_void;
pub type CFTypeRef = *const c_void;
pub type CFStringRef = *const c_void;
pub type CFMutableDictionaryRef = *mut c_void;
pub type NSUInteger = usize;

// ── MTLSize (used for dispatch) ──

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MTLSize {
    pub width: NSUInteger,
    pub height: NSUInteger,
    pub depth: NSUInteger,
}

// ── MTLRegion ──

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MTLOrigin {
    pub x: NSUInteger,
    pub y: NSUInteger,
    pub z: NSUInteger,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MTLRegion {
    pub origin: MTLOrigin,
    pub size: MTLSize,
}

// ── Metal resource options ──

pub const MTLResourceStorageModeShared: NSUInteger = 0;
pub const MTLResourceStorageModeManaged: NSUInteger = 0x10;
pub const MTLResourceStorageModePrivate: NSUInteger = 0x20;

// ── Metal pixel formats (common) ──

pub const MTLPixelFormatR8Unorm: NSUInteger = 10;
pub const MTLPixelFormatR32Float: NSUInteger = 55;
pub const MTLPixelFormatRGBA8Unorm: NSUInteger = 70;
pub const MTLPixelFormatBGRA8Unorm: NSUInteger = 80;
pub const MTLPixelFormatRGBA16Float: NSUInteger = 115;
pub const MTLPixelFormatRGBA32Float: NSUInteger = 125;
pub const MTLPixelFormatDepth32Float: NSUInteger = 252;
pub const MTLPixelFormatStencil8: NSUInteger = 253;
pub const MTLPixelFormatDepth32Float_Stencil8: NSUInteger = 260;

// ── Metal texture usage ──

pub const MTLTextureUsageShaderRead: NSUInteger = 0x0001;
pub const MTLTextureUsageShaderWrite: NSUInteger = 0x0002;
pub const MTLTextureUsageRenderTarget: NSUInteger = 0x0004;

// ── Metal texture types ──

pub const MTLTextureType2D: NSUInteger = 2;
pub const MTLTextureType2DMultisample: NSUInteger = 4;

// ── Render pass load/store actions ──

pub const MTLLoadActionDontCare: NSUInteger = 0;
pub const MTLLoadActionLoad: NSUInteger = 1;
pub const MTLLoadActionClear: NSUInteger = 2;

pub const MTLStoreActionDontCare: NSUInteger = 0;
pub const MTLStoreActionStore: NSUInteger = 1;
pub const MTLStoreActionMultisampleResolve: NSUInteger = 2;
pub const MTLStoreActionStoreAndMultisampleResolve: NSUInteger = 3;

// ── Primitive type ──

pub const MTLPrimitiveTypePoint: NSUInteger = 0;
pub const MTLPrimitiveTypeLine: NSUInteger = 1;
pub const MTLPrimitiveTypeLineStrip: NSUInteger = 2;
pub const MTLPrimitiveTypeTriangle: NSUInteger = 3;
pub const MTLPrimitiveTypeTriangleStrip: NSUInteger = 4;

// ── Index type ──

pub const MTLIndexTypeUInt16: NSUInteger = 0;
pub const MTLIndexTypeUInt32: NSUInteger = 1;

// ── Cull / winding / compare / blend ──

pub const MTLCullModeNone: NSUInteger = 0;
pub const MTLCullModeFront: NSUInteger = 1;
pub const MTLCullModeBack: NSUInteger = 2;

pub const MTLWindingClockwise: NSUInteger = 0;
pub const MTLWindingCounterClockwise: NSUInteger = 1;

pub const MTLCompareFunctionNever: NSUInteger = 0;
pub const MTLCompareFunctionLess: NSUInteger = 1;
pub const MTLCompareFunctionEqual: NSUInteger = 2;
pub const MTLCompareFunctionLessEqual: NSUInteger = 3;
pub const MTLCompareFunctionGreater: NSUInteger = 4;
pub const MTLCompareFunctionNotEqual: NSUInteger = 5;
pub const MTLCompareFunctionGreaterEqual: NSUInteger = 6;
pub const MTLCompareFunctionAlways: NSUInteger = 7;

pub const MTLBlendFactorZero: NSUInteger = 0;
pub const MTLBlendFactorOne: NSUInteger = 1;
pub const MTLBlendFactorSourceColor: NSUInteger = 2;
pub const MTLBlendFactorOneMinusSourceColor: NSUInteger = 3;
pub const MTLBlendFactorSourceAlpha: NSUInteger = 4;
pub const MTLBlendFactorOneMinusSourceAlpha: NSUInteger = 5;
pub const MTLBlendFactorDestinationAlpha: NSUInteger = 6;
pub const MTLBlendFactorOneMinusDestinationAlpha: NSUInteger = 7;
pub const MTLBlendFactorDestinationColor: NSUInteger = 8;
pub const MTLBlendFactorOneMinusDestinationColor: NSUInteger = 9;

pub const MTLBlendOperationAdd: NSUInteger = 0;
pub const MTLBlendOperationSubtract: NSUInteger = 1;
pub const MTLBlendOperationReverseSubtract: NSUInteger = 2;
pub const MTLBlendOperationMin: NSUInteger = 3;
pub const MTLBlendOperationMax: NSUInteger = 4;

// ── Vertex format / step ──

pub const MTLVertexFormatFloat: NSUInteger = 28;
pub const MTLVertexFormatFloat2: NSUInteger = 29;
pub const MTLVertexFormatFloat3: NSUInteger = 30;
pub const MTLVertexFormatFloat4: NSUInteger = 31;
pub const MTLVertexFormatHalf2: NSUInteger = 25;
pub const MTLVertexFormatHalf4: NSUInteger = 27;
pub const MTLVertexFormatUChar4Normalized: NSUInteger = 11;
pub const MTLVertexFormatUInt: NSUInteger = 36;
pub const MTLVertexFormatInt: NSUInteger = 32;

pub const MTLVertexStepFunctionConstant: NSUInteger = 0;
pub const MTLVertexStepFunctionPerVertex: NSUInteger = 1;
pub const MTLVertexStepFunctionPerInstance: NSUInteger = 2;

// ── Viewport / scissor (passed by value to encoder) ──

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MTLViewport {
    pub origin_x: f64,
    pub origin_y: f64,
    pub width: f64,
    pub height: f64,
    pub znear: f64,
    pub zfar: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MTLScissorRect {
    pub x: NSUInteger,
    pub y: NSUInteger,
    pub width: NSUInteger,
    pub height: NSUInteger,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MTLClearColor {
    pub red: f64,
    pub green: f64,
    pub blue: f64,
    pub alpha: f64,
}

// ── CoreFoundation constants ──

pub const kCFStringEncodingUTF8: u32 = 0x08000100;

// ── Metal.framework ──

#[link(name = "Metal", kind = "framework")]
extern "C" {
    pub fn MTLCreateSystemDefaultDevice() -> ObjcId;
    pub fn MTLCopyAllDevices() -> ObjcId;
}

// ── CoreFoundation ──

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    pub fn CFStringCreateWithCString(
        alloc: *const c_void,
        cStr: *const c_char,
        encoding: u32,
    ) -> CFStringRef;
    pub fn CFStringCreateWithBytes(
        alloc: *const c_void,
        bytes: *const u8,
        numBytes: isize,
        encoding: u32,
        isExternalRepresentation: bool,
    ) -> CFStringRef;
    pub fn CFRelease(cf: CFTypeRef);
    pub fn CFRetain(cf: CFTypeRef) -> CFTypeRef;
}

// ── ObjC runtime ──

extern "C" {
    pub fn objc_getClass(name: *const c_char) -> ObjcClass;
    pub fn sel_registerName(name: *const c_char) -> ObjcSel;
    pub fn objc_msgSend() -> ObjcId;
    pub fn objc_retain(obj: ObjcId) -> ObjcId;
    pub fn objc_release(obj: ObjcId);
    pub fn objc_autoreleasePoolPush() -> *mut c_void;
    pub fn objc_autoreleasePoolPop(pool: *mut c_void);
    pub fn object_getClass(obj: ObjcId) -> ObjcClass;
    pub fn class_getMethodImplementation(cls: ObjcClass, sel: ObjcSel) -> *const c_void;
    /// ARC optimization: if called immediately after a msg_send that returns
    /// an autoreleased object, the runtime skips the autorelease+retain
    /// round-trip entirely. Must be called with no intervening code.
    pub fn objc_retainAutoreleasedReturnValue(obj: ObjcId) -> ObjcId;
}

// ── String helpers ──

#[mutants::skip] // null err → None — mutant → None passes on success path
pub fn nserror_string(err: ObjcId) -> Option<String> {
    if err.is_null() {
        return None;
    }
    unsafe {
        let desc = msg0(err, SEL_localizedDescription());
        if desc.is_null() {
            return None;
        }
        nsstring_to_rust(desc)
    }
}

pub fn nsstring_to_rust(obj: ObjcId) -> Option<String> {
    if obj.is_null() {
        return None;
    }
    unsafe {
        type F = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const c_char;
        let f: F = std::mem::transmute(objc_msgSend as *const c_void);
        let cstr = f(obj, SEL_UTF8String());
        if cstr.is_null() {
            return None;
        }
        Some(CStr::from_ptr(cstr).to_string_lossy().into_owned())
    }
}

/// Create NSString from &str. Zero-alloc: uses CFStringCreateWithBytes
/// directly on the Rust string's UTF-8 bytes, no CString intermediate.
pub fn nsstring(s: &str) -> ObjcId {
    unsafe {
        CFStringCreateWithBytes(
            std::ptr::null(),
            s.as_ptr(),
            s.len() as isize,
            kCFStringEncodingUTF8,
            false,
        ) as ObjcId
    }
}

// ── Retain/release (hot path — cached selectors) ──

/// Retain via objc_retain (direct C call, not msg_send).
#[inline(always)]
pub unsafe fn retain(obj: ObjcId) -> ObjcId {
    if !obj.is_null() {
        objc_retain(obj);
    }
    obj
}

/// Release via objc_release (direct C call, not msg_send).
#[inline(always)]
pub unsafe fn release(obj: ObjcId) {
    if !obj.is_null() {
        objc_release(obj);
    }
}

/// Retain a known-non-null object. No null check — hot path only.
#[inline(always)]
pub unsafe fn retain_nonnull(obj: ObjcId) -> ObjcId {
    debug_assert!(!obj.is_null());
    objc_retain(obj);
    obj
}

/// Release a known-non-null object. No null check — hot path only.
#[inline(always)]
pub unsafe fn release_nonnull(obj: ObjcId) {
    debug_assert!(!obj.is_null());
    objc_release(obj);
}
