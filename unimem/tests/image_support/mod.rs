use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::ffi::c_void;
use std::ptr;
use unimem::ffi::*;
use unimem::{Block, NativeImageFormat};

thread_local! { static ALLOCS: Cell<Option<usize>> = const { Cell::new(None) }; }
struct Observer;
fn record() {
    let _ = ALLOCS.try_with(|c| {
        if let Some(n) = c.get() {
            c.set(Some(n + 1));
        }
    });
}
// SAFETY: All pointers/layouts pass through to System unchanged. Observation
// uses only a thread-local Cell and never allocates.
unsafe impl GlobalAlloc for Observer {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record();
        // SAFETY: The caller supplies a valid layout, forwarded to System.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        // SAFETY: The pointer and matching layout came from System.
        unsafe { System.dealloc(p, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record();
        // SAFETY: The caller supplies a valid layout, forwarded to System.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record();
        // SAFETY: System owns p and the caller supplies its layout/valid new size.
        unsafe { System.realloc(p, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: Observer = Observer;
pub fn observe<T>(f: impl FnOnce() -> T) -> (T, usize) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            ALLOCS.with(|c| c.set(None));
        }
    }
    ALLOCS.with(|c| c.set(Some(0)));
    let _reset = Reset;
    let result = f();
    (result, ALLOCS.with(|c| c.get().unwrap()))
}

type CVPixelBufferRef = *mut c_void;
#[link(name = "CoreVideo", kind = "framework")]
extern "C" {
    fn CVPixelBufferCreate(
        allocator: CFTypeRef,
        width: usize,
        height: usize,
        format: u32,
        attributes: CFDictionaryRef,
        output: *mut CVPixelBufferRef,
    ) -> i32;
    fn CVPixelBufferCreateWithIOSurface(
        allocator: CFTypeRef,
        surface: IOSurfaceRef,
        attributes: CFDictionaryRef,
        output: *mut CVPixelBufferRef,
    ) -> i32;
    fn CVPixelBufferGetIOSurface(buffer: CVPixelBufferRef) -> IOSurfaceRef;
    fn CVPixelBufferGetPixelFormatType(buffer: CVPixelBufferRef) -> u32;
    fn CVPixelBufferGetWidth(buffer: CVPixelBufferRef) -> usize;
    fn CVPixelBufferGetHeight(buffer: CVPixelBufferRef) -> usize;
    fn CVPixelBufferGetPlaneCount(buffer: CVPixelBufferRef) -> usize;
    fn CVPixelBufferGetWidthOfPlane(buffer: CVPixelBufferRef, plane: usize) -> usize;
    fn CVPixelBufferGetHeightOfPlane(buffer: CVPixelBufferRef, plane: usize) -> usize;
    fn CVPixelBufferGetBytesPerRowOfPlane(buffer: CVPixelBufferRef, plane: usize) -> usize;
    fn CVPixelBufferGetBaseAddressOfPlane(buffer: CVPixelBufferRef, plane: usize) -> *mut c_void;
    fn CVPixelBufferLockBaseAddress(buffer: CVPixelBufferRef, flags: u64) -> i32;
    fn CVPixelBufferUnlockBaseAddress(buffer: CVPixelBufferRef, flags: u64) -> i32;
    static kCVPixelBufferIOSurfacePropertiesKey: CFStringRef;
}

pub struct PixelBuffer(CVPixelBufferRef);
impl PixelBuffer {
    pub fn create(format: NativeImageFormat, width: usize, height: usize) -> Self {
        // SAFETY: SDK callbacks retain the live immutable property entries.
        // CoreVideo returns one owned buffer; CF properties are released below.
        unsafe {
            let empty = CFDictionaryCreate(
                ptr::null(),
                ptr::null(),
                ptr::null(),
                0,
                ptr::addr_of!(kCFTypeDictionaryKeyCallBacks),
                ptr::addr_of!(kCFTypeDictionaryValueCallBacks),
            );
            assert!(!empty.is_null());
            let attrs = CFDictionaryCreate(
                ptr::null(),
                [kCVPixelBufferIOSurfacePropertiesKey].as_ptr(),
                [empty].as_ptr(),
                1,
                ptr::addr_of!(kCFTypeDictionaryKeyCallBacks),
                ptr::addr_of!(kCFTypeDictionaryValueCallBacks),
            );
            assert!(!attrs.is_null());
            let mut raw = ptr::null_mut();
            let result =
                CVPixelBufferCreate(ptr::null(), width, height, format.fourcc(), attrs, &mut raw);
            CFRelease(attrs);
            CFRelease(empty);
            assert_eq!(result, 0, "CoreVideo create {format:?} {width}x{height}");
            assert!(!raw.is_null());
            Self(raw)
        }
    }
    pub fn wrap(block: &Block) -> Self {
        let mut raw = ptr::null_mut();
        // SAFETY: Block owns the live IOSurface. CoreVideo retains its own
        // reference; no CPU access or device work occurs during wrapping.
        let result = unsafe {
            CVPixelBufferCreateWithIOSurface(ptr::null(), block.handle(), ptr::null(), &mut raw)
        };
        assert_eq!(result, 0, "CoreVideo wrap {:?}", block.image_layout());
        assert!(!raw.is_null());
        Self(raw)
    }
    pub fn surface(&self) -> IOSurfaceRef {
        // SAFETY: self retains the buffer. The borrowed surface stays live.
        let raw = unsafe { CVPixelBufferGetIOSurface(self.0) };
        assert!(!raw.is_null());
        raw
    }
    pub fn surface_id(&self) -> u32 {
        // SAFETY: self retains the live IOSurface through its pixel buffer.
        unsafe { IOSurfaceGetID(self.surface()) }
    }
    pub fn check(&self, block: &Block) {
        let layout = block.image_layout().unwrap();
        assert_eq!(self.surface_id(), block.id());
        // SAFETY: Both owners retain the same native surface. This test does no
        // concurrent access; only metadata/pointers are inspected while locked.
        unsafe {
            assert_eq!(CVPixelBufferLockBaseAddress(self.0, 1), 0);
            assert_eq!(
                CVPixelBufferGetPixelFormatType(self.0),
                layout.format().fourcc()
            );
            assert_eq!(CVPixelBufferGetWidth(self.0), layout.width());
            assert_eq!(CVPixelBufferGetHeight(self.0), layout.height());
            assert_eq!(CVPixelBufferGetPlaneCount(self.0), 2);
            for (i, plane) in layout.planes().iter().enumerate() {
                assert_eq!(CVPixelBufferGetWidthOfPlane(self.0, i), plane.width());
                assert_eq!(CVPixelBufferGetHeightOfPlane(self.0, i), plane.height());
                assert_eq!(
                    CVPixelBufferGetBytesPerRowOfPlane(self.0, i),
                    plane.stride()
                );
                assert_eq!(
                    CVPixelBufferGetBaseAddressOfPlane(self.0, i).cast::<u8>(),
                    block.address().add(plane.offset())
                );
            }
            assert_eq!(CVPixelBufferUnlockBaseAddress(self.0, 1), 0);
        }
    }
    pub fn check_samples(&self, format: NativeImageFormat) {
        // SAFETY: fill initialized the whole backing earlier; this sole CPU
        // reader retains the pixel buffer and performs no conflicting access.
        unsafe {
            assert_eq!(CVPixelBufferLockBaseAddress(self.0, 1), 0);
            for i in 0..2 {
                let base = CVPixelBufferGetBaseAddressOfPlane(self.0, i).cast::<u8>();
                let stride = CVPixelBufferGetBytesPerRowOfPlane(self.0, i);
                let width = CVPixelBufferGetWidthOfPlane(self.0, i);
                let height = CVPixelBufferGetHeightOfPlane(self.0, i);
                let bytes = if format.fourcc() >> 24 == u32::from(b'x') {
                    2
                } else {
                    1
                };
                for y in 0..height {
                    let row =
                        std::slice::from_raw_parts(base.add(y * stride), width * bytes * (i + 1));
                    assert_pattern(row, bytes, i);
                }
            }
            assert_eq!(CVPixelBufferUnlockBaseAddress(self.0, 1), 0);
        }
    }
}
impl Drop for PixelBuffer {
    fn drop(&mut self) {
        // SAFETY: self owns one Create-rule reference.
        unsafe { CFRelease(self.0) }
    }
}

pub fn fill(block: &mut Block) {
    let layout = block.image_layout().unwrap();
    // SAFETY: The test has exclusive CPU access, no devices/imported views.
    // Raw initialization precedes any full-extent Rust slice, including padding.
    unsafe { block.address().write_bytes(0, block.size()) };
    // SAFETY: All bytes were initialized, and no conflicting view/work exists.
    let all = unsafe { block.as_bytes_mut() };
    for (i, plane) in layout.planes().iter().enumerate() {
        let lane_bytes = plane.element_bytes() / (i + 1);
        for y in 0..plane.height() {
            let row = plane.offset() + y * plane.stride();
            for x in 0..plane.width() * (i + 1) {
                if lane_bytes == 1 {
                    all[row + x] = 41 + i as u8;
                } else {
                    all[row + x * 2..row + x * 2 + 2]
                        .copy_from_slice(&((321 + i as u16) << 6).to_le_bytes());
                }
            }
        }
    }
}
fn assert_pattern(row: &[u8], lane_bytes: usize, index: usize) {
    if lane_bytes == 1 {
        assert!(row.iter().all(|b| *b == 41 + index as u8));
    } else {
        assert!(row
            .chunks_exact(2)
            .all(|b| u16::from_le_bytes([b[0], b[1]]) == (321 + index as u16) << 6));
    }
}
pub fn check_samples(block: &Block) {
    let layout = block.image_layout().unwrap();
    // SAFETY: fill initialized the complete mapping; every other owner/view has
    // been dropped or is idle, with no native device work in this test.
    let all = unsafe { block.as_bytes() };
    for (i, plane) in layout.planes().iter().enumerate() {
        for y in 0..plane.height() {
            let row = plane.offset() + y * plane.stride();
            let end = row + plane.width() * plane.element_bytes();
            assert_pattern(&all[row..end], plane.element_bytes() / (i + 1), i);
            assert!(all[end..row + plane.stride()].iter().all(|b| *b == 0));
        }
    }
    // Check the complete retained allocation, including gaps and final rounding.
    for (offset, byte) in all.iter().enumerate() {
        let sample = layout.planes().iter().any(|plane| {
            offset.checked_sub(plane.offset()).is_some_and(|position| {
                position < plane.row_extent()
                    && position % plane.stride() < plane.width() * plane.element_bytes()
            })
        });
        if !sample {
            assert_eq!(*byte, 0, "padding byte {offset}");
        }
    }
}
