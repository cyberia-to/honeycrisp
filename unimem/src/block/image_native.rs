use super::creation::{native_call, number, OwnedCf, Surface};
use super::image::{ImageBlockPlan, NativeImageLayout, NativePlaneLayout};
use super::image_validation::{validate, ImageQuery, PlaneQuery};
#[cfg(test)]
use super::tests::{self, Call, Event};
use super::{checked_size, Block};
use crate::{ffi::*, MemError};
use core::ptr::{self, NonNull};

fn dictionary(keys: &[CFStringRef], values: &[CFTypeRef]) -> Result<OwnedCf, MemError> {
    debug_assert_eq!(keys.len(), values.len());
    // SAFETY: Equal complete key/value arrays stay live through the call. SDK
    // callbacks retain their own references. Length is bounded by our literals.
    OwnedCf::from_create(native_call!(Dictionary, ptr::null(), unsafe {
        CFDictionaryCreate(
            ptr::null(),
            keys.as_ptr(),
            values.as_ptr(),
            keys.len() as CFIndex,
            ptr::addr_of!(kCFTypeDictionaryKeyCallBacks),
            ptr::addr_of!(kCFTypeDictionaryValueCallBacks),
        )
    }))
    .ok_or(MemError::BlockPropertiesFailed)
}

fn plane_properties(plane: &NativePlaneLayout) -> Result<OwnedCf, MemError> {
    let values = [
        number(checked_size(plane.width())?)?,
        number(checked_size(plane.height())?)?,
        number(plane.element_bytes() as i64)?,
        number(checked_size(plane.stride())?)?,
        number(i64::try_from(plane.offset()).map_err(|_| MemError::SizeOverflow)?)?,
        number(checked_size(plane.row_extent())?)?,
        number(1)?,
        number(1)?,
    ];
    // SAFETY: Borrowed SDK key constants; no pointed-to values are mutated.
    let keys = unsafe {
        [
            kIOSurfacePlaneWidth,
            kIOSurfacePlaneHeight,
            kIOSurfacePlaneBytesPerElement,
            kIOSurfacePlaneBytesPerRow,
            kIOSurfacePlaneOffset,
            kIOSurfacePlaneSize,
            kIOSurfacePlaneElementWidth,
            kIOSurfacePlaneElementHeight,
        ]
    };
    dictionary(&keys, &values.each_ref().map(OwnedCf::raw))
}

fn properties(plan: &ImageBlockPlan) -> Result<OwnedCf, MemError> {
    let layout = plan.layout();
    let planes = [
        plane_properties(&layout.planes()[0])?,
        plane_properties(&layout.planes()[1])?,
    ];
    let pointers = planes.each_ref().map(OwnedCf::raw);
    // SAFETY: Two live owned dictionaries and SDK retaining callbacks. The
    // array retains its entries before local references are released.
    let array = OwnedCf::from_create(native_call!(Array, ptr::null(), unsafe {
        CFArrayCreate(
            ptr::null(),
            pointers.as_ptr(),
            2,
            ptr::addr_of!(kCFTypeArrayCallBacks),
        )
    }))
    .ok_or(MemError::BlockPropertiesFailed)?;
    let values = [
        number(checked_size(layout.width())?)?,
        number(checked_size(layout.height())?)?,
        number(checked_size(plan.allocation_size())?)?,
        number(i64::from(layout.format().fourcc()))?,
    ];
    // SAFETY: Borrowed SDK key constants.
    let keys = unsafe {
        [
            kIOSurfaceWidth,
            kIOSurfaceHeight,
            kIOSurfaceAllocSize,
            kIOSurfacePixelFormat,
            kIOSurfacePlaneInfo,
        ]
    };
    let [width, height, allocation, format] = values.each_ref().map(OwnedCf::raw);
    dictionary(&keys, &[width, height, allocation, format, array.raw()])
}

fn map(owner: OwnedCf, expected: Option<usize>) -> Result<Block, MemError> {
    let mut surface = Surface {
        owner,
        locked: false,
    };
    // SAFETY: Surface holds an owned live IOSurface reference.
    let size = unsafe { IOSurfaceGetAllocSize(surface.raw()) };
    #[cfg(test)]
    let size = tests::extent(size);
    if let Some(expected) = expected {
        if size != expected {
            return Err(MemError::BlockExtentMismatch {
                expected,
                actual: size,
            });
        }
    }
    checked_size(size)?;
    // SAFETY: Add one lock to this live owned surface; failure owns no lock.
    let result = native_call!(Lock, tests::LOCK_ERROR, unsafe {
        IOSurfaceLock(surface.raw(), 0, ptr::null_mut())
    });
    if result != KERN_SUCCESS {
        return Err(MemError::BlockLockFailed(result));
    }
    surface.locked = true;
    #[cfg(test)]
    tests::record(Event::Locked);
    // SAFETY: Query mapping after our successful CPU lock.
    let base = unsafe { IOSurfaceGetBaseAddress(surface.raw()).cast::<u8>() };
    #[cfg(test)]
    let base = tests::base(base);
    let va = NonNull::new(base).ok_or(MemError::BlockAddressInvalid)?;
    if !(base as usize).is_multiple_of(core::mem::align_of::<f32>())
        || (base as usize).checked_add(size).is_none()
    {
        return Err(MemError::BlockAddressInvalid);
    }
    // SAFETY: The locked owned surface remains alive for this query.
    let id = unsafe { IOSurfaceGetID(surface.raw()) };
    Ok(Block {
        raw: surface.into_raw(),
        va,
        size,
        id,
    })
}

pub(super) fn open(plan: &ImageBlockPlan) -> Result<Block, MemError> {
    let props = properties(plan)?;
    // SAFETY: The complete property dictionary stays live during creation.
    let owner = OwnedCf::from_create(native_call!(Surface, ptr::null_mut(), unsafe {
        IOSurfaceCreate(props.raw())
    }))
    .ok_or(MemError::BlockCreateFailed)?;
    drop(props);
    let block = map(owner, Some(plan.allocation_size()))?;
    if layout(&block)? != *plan.layout() {
        return Err(MemError::ImageLayoutInvalid);
    }
    Ok(block)
}

pub(super) unsafe fn retain(raw: IOSurfaceRef) -> Result<Block, MemError> {
    if raw.is_null() {
        return Err(MemError::BlockAddressInvalid);
    }
    // SAFETY: The caller supplies a live IOSurface. CFRetain adds exactly one
    // owned reference; OwnedCf releases it on any following returned error.
    let owner = OwnedCf::from_create(native_call!(Retain, ptr::null(), unsafe { CFRetain(raw) }))
        .ok_or(MemError::BlockRetainFailed)?;
    let block = map(owner, None)?;
    layout(&block)?;
    Ok(block)
}

pub(super) fn layout(block: &Block) -> Result<NativeImageLayout, MemError> {
    let raw = block.handle();
    // SAFETY: Block owns a live stable surface/mapping. Only metadata/pointers
    // are queried; no sample reference is formed, even for invalid metadata.
    let query = unsafe {
        let count = IOSurfaceGetPlaneCount(raw);
        let planes = [0, 1].map(|i| PlaneQuery {
            width: IOSurfaceGetWidthOfPlane(raw, i),
            height: IOSurfaceGetHeightOfPlane(raw, i),
            element_bytes: IOSurfaceGetBytesPerElementOfPlane(raw, i),
            element_width: IOSurfaceGetElementWidthOfPlane(raw, i),
            element_height: IOSurfaceGetElementHeightOfPlane(raw, i),
            stride: IOSurfaceGetBytesPerRowOfPlane(raw, i),
            base: IOSurfaceGetBaseAddressOfPlane(raw, i) as usize,
        });
        ImageQuery {
            format: IOSurfaceGetPixelFormat(raw),
            width: IOSurfaceGetWidth(raw),
            height: IOSurfaceGetHeight(raw),
            count,
            base: block.address() as usize,
            allocation_size: block.size(),
            planes,
        }
    };
    #[cfg(test)]
    let query = tests::image_query(query);
    validate(query)
}
