use super::image::*;
use super::image_validation::{validate, ImageQuery, PlaneQuery};
use super::tests::{assert_balanced, traced, Call, Event, Fault};
use super::Block;
use crate::MemError;

#[test]
fn literal_numeric_plans_include_full_rows_and_native_gaps() {
    for (format, expected, total) in [
        (
            NativeImageFormat::Nv12VideoRange,
            [(5, 3, 1, 8, 0, 24), (3, 2, 2, 8, 32, 16)],
            64,
        ),
        (
            NativeImageFormat::P010FullRange,
            [(5, 3, 2, 16, 0, 48), (3, 2, 4, 16, 48, 32)],
            128,
        ),
    ] {
        let events = traced(Fault::None, || {
            let plan = ImageBlockPlan::checked(format, 5, 3, 8, 16, 64).unwrap();
            assert_eq!(plan.allocation_size(), total);
            for (p, tuple) in plan.layout().planes().iter().zip(expected) {
                assert_eq!(
                    (
                        p.width(),
                        p.height(),
                        p.element_bytes(),
                        p.stride(),
                        p.offset(),
                        p.row_extent()
                    ),
                    tuple
                );
            }
        });
        assert!(
            events.is_empty(),
            "numeric planning must not call CF or create native backing"
        );
    }
    for (width, height, row, offset, allocation) in [
        (0, 1, 1, 1, 1),
        (1, 0, 1, 1, 1),
        (1, 1, 0, 1, 1),
        (1, 1, 1, 0, 1),
        (1, 1, 1, 1, 0),
        (usize::MAX, 1, 1, 1, 1),
        (1, usize::MAX, 1, 1, 1),
        (isize::MAX as usize, 2, 1, 1, 1),
        (4, 3, usize::MAX, 1, 1),
        (4, 3, 1, usize::MAX, 1),
        (4, 3, 1, 1, usize::MAX),
    ] {
        assert!(ImageBlockPlan::checked(
            NativeImageFormat::P010VideoRange,
            width,
            height,
            row,
            offset,
            allocation
        )
        .is_err());
    }
}

fn query() -> ImageQuery {
    ImageQuery {
        format: u32::from_be_bytes(*b"420v"),
        width: 5,
        height: 3,
        count: 2,
        base: 0x1000,
        allocation_size: 48,
        planes: [
            PlaneQuery {
                width: 5,
                height: 3,
                element_bytes: 1,
                element_width: 1,
                element_height: 1,
                stride: 8,
                base: 0x1000,
            },
            PlaneQuery {
                width: 3,
                height: 2,
                element_bytes: 2,
                element_width: 1,
                element_height: 1,
                stride: 8,
                base: 0x1020,
            },
        ],
    }
}

#[test]
fn imported_geometry_checks_actual_extent_and_disjoint_full_planes() {
    let layout = validate(query()).unwrap();
    assert_eq!(layout.allocation_size(), 48);
    assert_eq!(layout.planes()[1].offset(), 32);
    for mutate in [
        |q: &mut ImageQuery| q.allocation_size = 47,
        |q: &mut ImageQuery| q.allocation_size = 0,
        |q: &mut ImageQuery| q.count = 0,
        |q: &mut ImageQuery| q.count = 1,
        |q: &mut ImageQuery| q.count = 3,
        |q: &mut ImageQuery| q.width = 0,
        |q: &mut ImageQuery| q.height = 0,
        |q: &mut ImageQuery| q.format = 0,
        |q: &mut ImageQuery| q.base = 0,
        |q: &mut ImageQuery| q.base += 1,
        |q: &mut ImageQuery| q.base = usize::MAX & !3,
        |q: &mut ImageQuery| q.planes[1].base = 0xff0,
        |q: &mut ImageQuery| q.planes[1].base = 0x1010,
        |q: &mut ImageQuery| q.planes[1].base = 0x1030,
        |q: &mut ImageQuery| q.planes[1].width = 2,
        |q: &mut ImageQuery| q.planes[1].height = 1,
        |q: &mut ImageQuery| q.planes[1].element_bytes = 1,
        |q: &mut ImageQuery| q.planes[1].element_width = 2,
        |q: &mut ImageQuery| q.planes[1].element_height = 2,
        |q: &mut ImageQuery| q.planes[1].stride = 5,
        |q: &mut ImageQuery| q.planes[1].stride = usize::MAX,
    ] {
        let mut malformed = query();
        mutate(&mut malformed);
        assert!(validate(malformed).is_err());
    }
    // Imports can have a leading gap and reversed disjoint plane placement.
    let mut reversed = query();
    reversed.allocation_size = 64;
    reversed.planes[0].base = 0x1020;
    reversed.planes[1].base = 0x1008;
    assert!(validate(reversed).is_ok());
}

fn balanced(events: &[Event]) {
    let acquired = events
        .iter()
        .filter(|e| matches!(e, Event::Acquire(_)))
        .count();
    assert_balanced(events, acquired);
    let locks = events.iter().filter(|e| matches!(e, Event::Locked)).count();
    let unlocks = events
        .iter()
        .filter(|e| matches!(e, Event::Unlocked))
        .count();
    assert_eq!(locks, unlocks);
}

#[test]
fn every_image_creation_stage_unwinds_its_own_references_and_locks() {
    let plan = Block::plan_image(NativeImageFormat::Nv12VideoRange, 5, 3).unwrap();
    let success = traced(Fault::None, || {
        drop(plan.open().unwrap());
    });
    assert_balanced(&success, 25);
    balanced(&success);
    for stage in [
        Call::Number,
        Call::Dictionary,
        Call::Array,
        Call::Surface,
        Call::Lock,
    ] {
        let count = success
            .iter()
            .filter(|e| matches!(e, Event::Call(call) if *call == stage))
            .count();
        assert!(count > 0);
        for occurrence in 1..=count {
            let events = traced(Fault::Fail(stage, occurrence), || {
                assert!(plan.open().is_err());
            });
            balanced(&events);
            assert_eq!(
                events
                    .iter()
                    .filter(|e| matches!(e, Event::Call(call) if *call == stage))
                    .count(),
                occurrence
            );
        }
    }
    for fault in [
        Fault::Extent(plan.allocation_size() - 1),
        Fault::Extent(plan.allocation_size() + 1),
        Fault::NullBase,
        Fault::MisalignedBase,
        Fault::WrappingBase,
        Fault::ImageCount(1),
        Fault::ImageFormat(u32::from_be_bytes(*b"420f")),
    ] {
        let events = traced(fault, || {
            assert!(plan.open().is_err());
        });
        balanced(&events);
    }
}

#[test]
fn failed_import_preserves_original_owner_and_balances_only_its_own_lock() {
    let original = Block::plan_image(NativeImageFormat::P010FullRange, 5, 3)
        .unwrap()
        .open()
        .unwrap();
    let identity = original.id();
    for fault in [
        Fault::None,
        Fault::Fail(Call::Retain, 1),
        Fault::Fail(Call::Lock, 1),
        Fault::NullBase,
        Fault::MisalignedBase,
        Fault::WrappingBase,
        Fault::ImageCount(1),
    ] {
        let events = traced(fault, || {
            // SAFETY: Original is live, stable and idle. No sample view is formed.
            let imported = unsafe { Block::retain_image(original.handle()) };
            if matches!(fault, Fault::None) {
                assert_eq!(imported.as_ref().unwrap().id(), identity);
            } else {
                assert!(imported.is_err());
            }
            drop(imported);
        });
        balanced(&events);
        assert_eq!(original.id(), identity);
        assert_eq!(
            original.image_layout().unwrap().format(),
            NativeImageFormat::P010FullRange
        );
    }
    let events = traced(Fault::None, || {
        // SAFETY: Null is rejected before any native access.
        assert!(matches!(
            unsafe { Block::retain_image(core::ptr::null_mut()) },
            Err(MemError::BlockAddressInvalid)
        ));
    });
    assert!(events.is_empty());
}
