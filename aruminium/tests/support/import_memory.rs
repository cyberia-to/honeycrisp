//! Native fixture preflight, not a universal IOSurface geometry guarantee.

use aruminium::Block;
use std::ffi::c_int;
use std::mem::{align_of, offset_of, size_of};

// macOS SDK26.4.1: mach/vm_region.h short-info ABI under pack(4).
#[repr(C, packed(4))]
#[derive(Default)]
struct RegionInfo {
    protection: c_int,
    max_protection: c_int,
    inheritance: u32,
    offset: u64,
    user_tag: u32,
    ref_count: u32,
    shadow_depth: u16,
    external_pager: u8,
    share_mode: u8,
    is_submap: c_int,
    behavior: c_int,
    object_id: u32,
    user_wired_count: u16,
    flags: u16,
}

unsafe extern "C" {
    fn getpagesize() -> c_int;
    static mach_task_self_: u32;
    fn mach_vm_region_recurse(
        task: u32,
        address: *mut u64,
        size: *mut u64,
        depth: *mut u32,
        info: *mut c_int,
        count: *mut u32,
    ) -> c_int;
}

pub fn initialized_block(logical_bytes: usize) -> Block {
    // SAFETY: getpagesize has no pointer or initialization preconditions.
    let page = usize::try_from(unsafe { getpagesize() }).unwrap();
    assert!(page > 0);
    let request = logical_bytes
        .checked_add(page - 1)
        .unwrap()
        .checked_div(page)
        .unwrap()
        .checked_mul(page)
        .unwrap();
    let block = Block::open(request).unwrap();
    let pointer = block.address() as usize;
    assert!(block.size() >= logical_bytes);
    assert!(pointer.is_multiple_of(page));
    assert!(block.size().is_multiple_of(page));
    assert_eq!(size_of::<RegionInfo>(), 48);
    assert_eq!(align_of::<RegionInfo>(), 4);
    assert_eq!(offset_of!(RegionInfo, is_submap), 32);
    let mut start = u64::try_from(pointer).unwrap();
    let mut size = 0;
    let mut depth = u32::MAX;
    let mut info = RegionInfo::default();
    let mut count = 12;
    // SAFETY: Local writable ABI-sized outputs; inspect this task's live mapping.
    // The API returns no object-name send right. Maximum depth reaches a leaf.
    let status = unsafe {
        mach_vm_region_recurse(
            mach_task_self_,
            &mut start,
            &mut size,
            &mut depth,
            (&mut info as *mut RegionInfo).cast(),
            &mut count,
        )
    };
    assert_eq!(status, 0);
    assert_eq!(count, 12);
    assert_eq!(info.is_submap, 0);
    assert_eq!(info.protection & 3, 3); // VM_PROT_READ | WRITE: fixture needs both.
    let actual_start = u64::try_from(pointer).unwrap();
    let actual_end = actual_start
        .checked_add(u64::try_from(block.size()).unwrap())
        .unwrap();
    assert!(start <= actual_start && actual_end <= start.checked_add(size).unwrap());
    // SAFETY: Fresh unpublished mapping; initialize only its actual owned extent.
    unsafe { block.address().write_bytes(0, block.size()) };
    // Callers retain this owner through imports/commands/drop without remapping
    // or introducing external aliases. The query is only a current-mapping proof.
    block
}
