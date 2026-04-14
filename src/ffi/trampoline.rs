//! Typed objc_msgSend trampolines
//!
//! The transmute is done once per call-site type signature.
//! Selectors are passed as pre-cached ObjcSel, not &str.

use super::{
    objc_msgSend, objc_retainAutoreleasedReturnValue, MTLSize, NSUInteger, ObjcId, ObjcSel,
};
use std::ffi::c_void;

/// Send a message with 0 args, returning ObjcId.
#[inline(always)]
pub unsafe fn msg0(target: ObjcId, sel: ObjcSel) -> ObjcId {
    type F = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
    let f: F = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel)
}

/// Send msg + immediately retain the autoreleased return value.
/// Uses ARC fast-path: runtime skips autorelease+retain round-trip
/// when retain is called immediately after msg_send.
#[inline(always)]
pub unsafe fn msg0_retained(target: ObjcId, sel: ObjcSel) -> ObjcId {
    type F = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
    let f: F = std::mem::transmute(objc_msgSend as *const c_void);
    let obj = f(target, sel);
    objc_retainAutoreleasedReturnValue(obj)
}

/// Send a message with 0 args, returning nothing.
#[inline(always)]
pub unsafe fn msg0_void(target: ObjcId, sel: ObjcSel) {
    type F = unsafe extern "C" fn(ObjcId, ObjcSel);
    let f: F = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel);
}

/// Send a message with 0 args, returning bool.
#[inline(always)]
pub unsafe fn msg0_bool(target: ObjcId, sel: ObjcSel) -> bool {
    type F = unsafe extern "C" fn(ObjcId, ObjcSel) -> bool;
    let f: F = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel)
}

/// Send a message with 0 args, returning usize.
#[inline(always)]
pub unsafe fn msg0_usize(target: ObjcId, sel: ObjcSel) -> NSUInteger {
    type F = unsafe extern "C" fn(ObjcId, ObjcSel) -> NSUInteger;
    let f: F = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel)
}

/// Send a message with 0 args, returning u64.
#[inline(always)]
pub unsafe fn msg0_u64(target: ObjcId, sel: ObjcSel) -> u64 {
    type F = unsafe extern "C" fn(ObjcId, ObjcSel) -> u64;
    let f: F = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel)
}

/// Send a message with 0 args, returning *mut c_void.
#[inline(always)]
pub unsafe fn msg0_ptr(target: ObjcId, sel: ObjcSel) -> *mut c_void {
    type F = unsafe extern "C" fn(ObjcId, ObjcSel) -> *mut c_void;
    let f: F = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel)
}

/// Send a message with 0 args, returning MTLSize.
#[inline(always)]
pub unsafe fn msg0_mtlsize(target: ObjcId, sel: ObjcSel) -> MTLSize {
    type F = unsafe extern "C" fn(ObjcId, ObjcSel) -> MTLSize;
    let f: F = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel)
}

/// Send a message with 1 arg.
#[inline(always)]
pub unsafe fn msg1<A>(target: ObjcId, sel: ObjcSel, a: A) -> ObjcId {
    type F<A> = unsafe extern "C" fn(ObjcId, ObjcSel, A) -> ObjcId;
    let f: F<A> = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel, a)
}

/// Send a message with 2 args.
#[inline(always)]
pub unsafe fn msg2<A, B>(target: ObjcId, sel: ObjcSel, a: A, b: B) -> ObjcId {
    type F<A, B> = unsafe extern "C" fn(ObjcId, ObjcSel, A, B) -> ObjcId;
    let f: F<A, B> = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel, a, b)
}

/// Send a message with 3 args.
#[inline(always)]
pub unsafe fn msg3<A, B, C>(target: ObjcId, sel: ObjcSel, a: A, b: B, c: C) -> ObjcId {
    type F<A, B, C> = unsafe extern "C" fn(ObjcId, ObjcSel, A, B, C) -> ObjcId;
    let f: F<A, B, C> = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel, a, b, c)
}

/// Send a message with 4 args.
#[inline(always)]
pub unsafe fn msg4<A, B, C, D>(target: ObjcId, sel: ObjcSel, a: A, b: B, c: C, d: D) -> ObjcId {
    type F<A, B, C, D> = unsafe extern "C" fn(ObjcId, ObjcSel, A, B, C, D) -> ObjcId;
    let f: F<A, B, C, D> = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel, a, b, c, d)
}

// ── Void msg_send helpers (encoder hot path) ──

/// Send a message with 1 ObjcId arg, returning nothing.
#[inline(always)]
pub unsafe fn msg1_void(target: ObjcId, sel: ObjcSel, a: ObjcId) {
    type F = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId);
    let f: F = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel, a);
}

/// Send a message with 3 args (ObjcId, usize, usize), returning nothing.
#[inline(always)]
pub unsafe fn msg3_void(target: ObjcId, sel: ObjcSel, a: ObjcId, b: NSUInteger, c: NSUInteger) {
    type F = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, NSUInteger, NSUInteger);
    let f: F = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel, a, b, c);
}

/// Send a message with (ptr, usize, usize) args, returning nothing.
#[inline(always)]
pub unsafe fn msg_bytes_void(
    target: ObjcId,
    sel: ObjcSel,
    ptr: *const c_void,
    len: NSUInteger,
    index: NSUInteger,
) {
    type F = unsafe extern "C" fn(ObjcId, ObjcSel, *const c_void, NSUInteger, NSUInteger);
    let f: F = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel, ptr, len, index);
}

/// Send a message with two MTLSize args, returning nothing.
#[inline(always)]
pub unsafe fn msg_dispatch_void(target: ObjcId, sel: ObjcSel, grid: MTLSize, group: MTLSize) {
    type F = unsafe extern "C" fn(ObjcId, ObjcSel, MTLSize, MTLSize);
    let f: F = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel, grid, group);
}
