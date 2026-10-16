#![no_std]
#![no_main]
use core::alloc::{GlobalAlloc, Layout};
use core::panic::PanicInfo;
struct CompileOnly;
unsafe impl GlobalAlloc for CompileOnly {
    unsafe fn alloc(&self, _: Layout) -> *mut u8 { core::ptr::null_mut() }
    unsafe fn dealloc(&self, _: *mut u8, _: Layout) {}
}
#[global_allocator]
static ALLOCATOR: CompileOnly = CompileOnly;
#[panic_handler]
fn panic(_: &PanicInfo<'_>) -> ! { loop {} }
#[no_mangle]
pub extern "C" fn native_plan(size: usize) -> usize {
    unimem::Block::plan(size).map_or(0, |p| p.allocation_size())
}
