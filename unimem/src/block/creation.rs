use super::{checked_size, Block, BlockPlan};
use crate::{ffi::*, MemError};
use std::ffi::c_void;
use std::ptr::{self, NonNull};

#[cfg(test)]
use super::tests::{self, Call, Event};

// Unit tests substitute the native failure return; production and tests then
// follow the same validation and ownership path. No seam exists in normal builds.
macro_rules! native_call {
    ($call:ident, $failure:expr, $native:expr) => {{
        #[cfg(test)]
        let result = if tests::fail(Call::$call) {
            $failure
        } else {
            $native
        };
        #[cfg(not(test))]
        let result = $native;
        result
    }};
}

/// One Create-rule reference. Dictionary callbacks own their separate retains.
struct OwnedCf(NonNull<c_void>);

impl OwnedCf {
    // Only native Create results enter here; a null result owns no reference.
    fn from_create(raw: CFTypeRef) -> Option<Self> {
        let raw = NonNull::new(raw.cast_mut())?;
        let owner = Self(raw);
        #[cfg(test)]
        tests::record(Event::Acquire(raw.as_ptr() as usize));
        Some(owner)
    }

    fn raw(&self) -> CFTypeRef {
        self.0.as_ptr()
    }
}

impl Drop for OwnedCf {
    fn drop(&mut self) {
        // SAFETY: This owner holds one native Create-rule reference.
        unsafe { release(self.raw()) }
    }
}

fn number(value: i64) -> Result<OwnedCf, MemError> {
    // SAFETY: SInt64Type matches the live i64 value; the default allocator is valid.
    OwnedCf::from_create(native_call!(Number, ptr::null(), unsafe {
        CFNumberCreate(
            ptr::null(),
            kCFNumberSInt64Type,
            ptr::from_ref(&value).cast(),
        )
    }))
    .ok_or(MemError::BlockPropertiesFailed)
}

fn properties(plan: &BlockPlan) -> Result<OwnedCf, MemError> {
    let numbers = [
        number(checked_size(plan.requested)?)?,
        number(1)?,
        number(1)?,
        number(checked_size(plan.row_bytes)?)?,
        number(checked_size(plan.allocation_size)?)?,
        number(0)?,
    ];
    // SAFETY: Read the borrowed SDK pointer constants, not the pointed-to strings.
    let keys = unsafe {
        [
            kIOSurfaceWidth,
            kIOSurfaceHeight,
            kIOSurfaceBytesPerElement,
            kIOSurfaceBytesPerRow,
            kIOSurfaceAllocSize,
            kIOSurfacePixelFormat,
        ]
    };
    let values = numbers.each_ref().map(OwnedCf::raw);
    // SAFETY: Six initialized entries remain live through the call. The SDK
    // callbacks retain entries and are passed by address as opaque structures.
    let dictionary = OwnedCf::from_create(native_call!(Dictionary, ptr::null(), unsafe {
        CFDictionaryCreate(
            ptr::null(),
            keys.as_ptr(),
            values.as_ptr(),
            6,
            ptr::addr_of!(kCFTypeDictionaryKeyCallBacks),
            ptr::addr_of!(kCFTypeDictionaryValueCallBacks),
        )
    }))
    .ok_or(MemError::BlockPropertiesFailed)?;
    // The dictionary retains its values; local Create references now release.
    Ok(dictionary)
}

struct Surface {
    owner: OwnedCf,
    locked: bool,
}

impl Surface {
    fn raw(&self) -> IOSurfaceRef {
        self.owner.raw().cast_mut()
    }

    fn into_raw(self) -> IOSurfaceRef {
        let raw = self.raw();
        // Block takes both this owned reference and the successful lock.
        std::mem::forget(self);
        raw
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        if self.locked {
            // SAFETY: locked becomes true only after IOSurfaceLock succeeds.
            unsafe { unlock(self.raw()) }
        }
        // The owned CF field releases after this destructor finishes.
    }
}

pub(super) fn open(plan: &BlockPlan) -> Result<Block, MemError> {
    let dictionary = properties(plan)?;
    // SAFETY: The complete immutable dictionary is retained across creation.
    let owner = OwnedCf::from_create(native_call!(Surface, ptr::null_mut(), unsafe {
        IOSurfaceCreate(dictionary.raw())
    }))
    .ok_or(MemError::BlockCreateFailed)?;
    let mut surface = Surface {
        owner,
        locked: false,
    };
    drop(dictionary);

    // SAFETY: This is a valid owned surface, queried before exposing a mapping.
    let actual = unsafe { IOSurfaceGetAllocSize(surface.raw()) };
    #[cfg(test)]
    let actual = tests::extent(actual);
    if actual != plan.allocation_size {
        return Err(MemError::BlockExtentMismatch {
            expected: plan.allocation_size,
            actual,
        });
    }

    // SAFETY: The surface is owned and currently unlocked; no seed is requested.
    let result = native_call!(Lock, tests::LOCK_ERROR, unsafe {
        IOSurfaceLock(surface.raw(), 0, ptr::null_mut())
    });
    if result != KERN_SUCCESS {
        return Err(MemError::BlockLockFailed(result));
    }
    surface.locked = true;
    #[cfg(test)]
    tests::record(Event::Locked);

    // SAFETY: Query the mapping only after successful CPU lock.
    let base = unsafe { IOSurfaceGetBaseAddress(surface.raw()).cast::<u8>() };
    #[cfg(test)]
    let base = tests::base(base);
    let va = NonNull::new(base).ok_or(MemError::BlockAddressInvalid)?;
    let address = base as usize;
    let alignment = std::mem::align_of::<f32>().max(std::mem::align_of::<u16>());
    if !address.is_multiple_of(alignment) || address.checked_add(actual).is_none() {
        return Err(MemError::BlockAddressInvalid);
    }
    #[cfg(test)]
    tests::record(Event::Call(Call::Id));
    // SAFETY: The owned, locked surface remains alive through this query.
    let id = unsafe { IOSurfaceGetID(surface.raw()) };
    Ok(Block {
        raw: surface.into_raw(),
        va,
        size: actual,
        id,
    })
}

unsafe fn release(raw: CFTypeRef) {
    // SAFETY: Caller transfers one owned native reference for release.
    unsafe { CFRelease(raw) }
    #[cfg(test)]
    tests::record(Event::Release(raw as usize));
}

unsafe fn unlock(raw: IOSurfaceRef) {
    // SAFETY: Caller holds one successful CPU lock on this live surface.
    unsafe { IOSurfaceUnlock(raw, 0, ptr::null_mut()) };
    #[cfg(test)]
    tests::record(Event::Unlocked);
}

pub(super) unsafe fn unlock_and_release(raw: IOSurfaceRef) {
    // SAFETY: Block transfers its successful lock and single owned reference.
    unsafe {
        unlock(raw);
        release(raw);
    }
}
