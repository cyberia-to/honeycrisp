use unimem::ffi::{
    kIOSurfaceAllocSize, kIOSurfaceBytesPerRow, IOSurfaceGetPropertyAlignment, IOSurfaceRef,
};
use unimem::{Block, MemError};

#[link(name = "IOSurface", kind = "framework")]
extern "C" {
    fn IOSurfaceGetWidth(surface: IOSurfaceRef) -> usize;
    fn IOSurfaceGetHeight(surface: IOSurfaceRef) -> usize;
    fn IOSurfaceGetBytesPerElement(surface: IOSurfaceRef) -> usize;
    fn IOSurfaceGetBytesPerRow(surface: IOSurfaceRef) -> usize;
    fn IOSurfaceGetPixelFormat(surface: IOSurfaceRef) -> u32;
}

extern "C" {
    fn getpagesize() -> i32;
}

#[test]
fn native_alignment_extent_and_raw_row_boundaries() {
    let row_alignment = unsafe { IOSurfaceGetPropertyAlignment(kIOSurfaceBytesPerRow) };
    let alloc_alignment = unsafe { IOSurfaceGetPropertyAlignment(kIOSurfaceAllocSize) };
    let page = usize::try_from(unsafe { getpagesize() }).unwrap();
    assert!(row_alignment > 0 && alloc_alignment > 0 && page > 1);
    for requested in [
        1,
        row_alignment,
        row_alignment + 1,
        page - 1,
        page,
        page + 1,
        4096,
        65537,
    ] {
        let plan = Block::plan(requested).unwrap();
        let row = requested.div_ceil(row_alignment) * row_alignment;
        let allocation = row.div_ceil(alloc_alignment) * alloc_alignment;
        assert_eq!(plan.requested_size(), requested);
        assert_eq!(plan.row_bytes(), row);
        assert_eq!(plan.allocation_size(), allocation);
        let block = plan.open().unwrap();
        assert_eq!(block.size(), allocation);
        assert_eq!(block.address() as usize % std::mem::align_of::<f32>(), 0);
        unsafe {
            assert_eq!(IOSurfaceGetWidth(block.handle()), requested);
            assert_eq!(IOSurfaceGetHeight(block.handle()), 1);
            assert_eq!(IOSurfaceGetBytesPerElement(block.handle()), 1);
            assert_eq!(IOSurfaceGetBytesPerRow(block.handle()), row);
            assert_eq!(IOSurfaceGetPixelFormat(block.handle()), 0);
            block.address().write_bytes(0xA5, block.size());
        }
        assert_eq!(block.as_bytes(), vec![0xA5; allocation]);
        assert_eq!(block.as_f32().len(), allocation / 4);
        assert_eq!(block.as_u16().len(), allocation / 2);
    }
}

#[test]
fn caller_can_account_before_open_and_reuse_numeric_plan() {
    let plan = Block::plan(1).unwrap();
    let mut available = 2 * plan.allocation_size();
    available = available.checked_sub(plan.allocation_size()).unwrap();
    let first = plan.open().unwrap();
    available = available.checked_sub(plan.allocation_size()).unwrap();
    let copied_plan = plan;
    let second = copied_plan.open().unwrap();
    assert_eq!(available, 0);
    assert_eq!(first.size() + second.size(), 2 * plan.allocation_size());
    assert_ne!(first.id(), second.id());
    assert_ne!(first.address(), second.address());
    unsafe {
        first.address().write(0x11);
        second.address().write(0x22);
        assert_eq!(first.address().read(), 0x11);
        assert_eq!(second.address().read(), 0x22);
    }
    drop(first);
    unsafe {
        assert_eq!(second.address().read(), 0x22);
    }
}

#[test]
fn large_extent_equality_and_initialized_endpoints() {
    let requested = 256 * 1024 * 1024;
    let plan = Block::plan(requested).unwrap();
    let block = plan.open().unwrap();
    assert_eq!(block.size(), plan.allocation_size());
    assert!(block.size() >= requested);
    unsafe {
        block.address().write(42);
        block.address().add(block.size() - 1).write(99);
        assert_eq!(block.address().read(), 42);
        assert_eq!(block.address().add(block.size() - 1).read(), 99);
    }
}

#[test]
fn public_invalid_requests_are_classified() {
    assert!(matches!(Block::plan(0), Err(MemError::ZeroSize)));
    assert!(matches!(Block::open(0), Err(MemError::ZeroSize)));
    for requested in [isize::MAX as usize + 1, usize::MAX] {
        assert!(matches!(
            Block::plan(requested),
            Err(MemError::SizeOverflow)
        ));
        assert!(matches!(
            Block::open(requested),
            Err(MemError::SizeOverflow)
        ));
    }
}

#[test]
fn precise_error_displays() {
    for (error, expected) in [
        (
            MemError::SizeOverflow,
            "allocation size exceeds integer or slice bounds",
        ),
        (
            MemError::BlockAlignmentInvalid,
            "invalid IOSurface property alignment",
        ),
        (
            MemError::BlockPropertiesFailed,
            "CoreFoundation property creation failed",
        ),
        (
            MemError::BlockExtentMismatch {
                expected: 128,
                actual: 129,
            },
            "IOSurface extent mismatch: expected 128, actual 129",
        ),
        (
            MemError::BlockAddressInvalid,
            "invalid IOSurface CPU mapping",
        ),
    ] {
        assert_eq!(error.to_string(), expected);
    }
}
