use unimem::{Block, NativeImageFormat};

mod image_support;
use image_support::{observe, PixelBuffer};

const FORMATS: [NativeImageFormat; 4] = [
    NativeImageFormat::Nv12VideoRange,
    NativeImageFormat::Nv12FullRange,
    NativeImageFormat::P010VideoRange,
    NativeImageFormat::P010FullRange,
];

#[test]
fn planning_is_allocation_free_with_literal_odd_geometry() {
    for (format, lanes, code) in [
        (FORMATS[0], [1, 2], u32::from_be_bytes(*b"420v")),
        (FORMATS[1], [1, 2], u32::from_be_bytes(*b"420f")),
        (FORMATS[2], [2, 4], u32::from_be_bytes(*b"x420")),
        (FORMATS[3], [2, 4], u32::from_be_bytes(*b"xf20")),
    ] {
        let ((plan, invalid, overflow), allocations) = observe(|| {
            (
                Block::plan_image(format, 5, 3),
                Block::plan_image(format, 0, 3),
                Block::plan_image(format, usize::MAX, 3),
            )
        });
        assert_eq!(allocations, 0);
        assert!(invalid.is_err());
        assert!(overflow.is_err());
        let plan = plan.unwrap();
        let layout = plan.layout();
        assert_eq!((layout.width(), layout.height()), (5, 3));
        assert_eq!(layout.format().fourcc(), code);
        for (plane, (width, height), bytes) in layout
            .planes()
            .iter()
            .zip([(5, 3), (3, 2)])
            .zip(lanes)
            .map(|((p, s), b)| (p, s, b))
        {
            assert_eq!((plane.width(), plane.height()), (width, height));
            assert_eq!(plane.element_bytes(), bytes);
            assert!(plane.stride() >= width * bytes);
            assert_eq!(plane.row_extent(), plane.stride() * height);
            assert!(plane.offset() + plane.row_extent() <= plan.allocation_size());
        }
        let planes = layout.planes();
        assert!(planes[0].offset() + planes[0].row_extent() <= planes[1].offset());
    }
    let (control, allocations) = observe(|| std::hint::black_box(Box::new([7u8; 8])));
    assert_eq!(allocations, 1);
    drop(control);
}

#[test]
fn created_planes_wrap_in_corevideo_without_copying_and_survive_block_drop() {
    for format in FORMATS {
        for (width, height) in [(5, 3), (16, 8)] {
            let plan = Block::plan_image(format, width, height).unwrap();
            let mut block = plan.open().unwrap();
            assert_eq!(block.image_layout().unwrap(), *plan.layout());
            image_support::fill(&mut block);
            let pixel = PixelBuffer::wrap(&block);
            pixel.check(&block);
            let id = block.id();
            drop(block);
            assert_eq!(pixel.surface_id(), id);
            pixel.check_samples(format);
        }
    }
}

#[test]
fn corevideo_allocated_planes_retain_import_and_survive_original_owner_drop() {
    for format in FORMATS {
        for (width, height) in [(5, 3), (16, 8)] {
            let pixel = PixelBuffer::create(format, width, height);
            // SAFETY: PixelBuffer owns a live IOSurface with immutable geometry.
            // This test has no device work or competing access. Import alone
            // makes no initialization/readiness claim; fill initializes below.
            let mut block = unsafe { Block::retain_image(pixel.surface()) }.unwrap();
            assert_eq!(block.id(), pixel.surface_id());
            assert_eq!(block.image_layout().unwrap().format(), format);
            image_support::fill(&mut block);
            pixel.check(&block);
            drop(pixel);
            image_support::check_samples(&block);
            let wrapped = PixelBuffer::wrap(&block);
            wrapped.check(&block);
        }
    }
}

#[test]
fn raw_blocks_and_null_import_are_not_image_layouts() {
    let raw = Block::open(128).unwrap();
    assert!(raw.image_layout().is_err());
    // SAFETY: Null is explicitly rejected before native retain/query.
    assert!(unsafe { Block::retain_image(std::ptr::null_mut()) }.is_err());
    // SAFETY: Raw is a live owned surface. Unsupported metadata is rejected
    // while balancing only the constructor's additional reference/lock.
    assert!(unsafe { Block::retain_image(raw.handle()) }.is_err());
    assert!(raw.size() >= 128);
}
