use super::*;
use std::cell::RefCell;
use std::collections::HashMap;

pub(super) const LOCK_ERROR: i32 = 0x1234;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Call {
    Number,
    Dictionary,
    Surface,
    Extent,
    Lock,
    Base,
    Id,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Event {
    Call(Call),
    Acquire(usize),
    Release(usize),
    Locked,
    Unlocked,
}

#[derive(Clone, Copy)]
enum Fault {
    None,
    Fail(Call, usize),
    Extent(usize),
    NullBase,
    MisalignedBase,
    WrappingBase,
}

struct Trace {
    fault: Fault,
    events: Vec<Event>,
}

thread_local! {
    static TRACE: RefCell<Option<Trace>> = const { RefCell::new(None) };
}

pub(super) fn record(event: Event) {
    TRACE.with_borrow_mut(|trace| {
        if let Some(trace) = trace {
            trace.events.push(event);
        }
    });
}

pub(super) fn fail(call: Call) -> bool {
    record(Event::Call(call));
    TRACE.with_borrow(|trace| match trace {
        Some(Trace {
            fault: Fault::Fail(target, occurrence),
            events,
        }) => {
            *target == call
                && events.iter().filter(|e| **e == Event::Call(call)).count() == *occurrence
        }
        _ => false,
    })
}

pub(super) fn extent(actual: usize) -> usize {
    record(Event::Call(Call::Extent));
    TRACE.with_borrow(|trace| match trace {
        Some(Trace {
            fault: Fault::Extent(value),
            ..
        }) => *value,
        _ => actual,
    })
}

pub(super) fn base(actual: *mut u8) -> *mut u8 {
    record(Event::Call(Call::Base));
    TRACE.with_borrow(|trace| match trace.as_ref().map(|trace| trace.fault) {
        Some(Fault::NullBase) => std::ptr::null_mut(),
        Some(Fault::MisalignedBase) => actual.wrapping_add(1),
        Some(Fault::WrappingBase) => std::ptr::without_provenance_mut(usize::MAX & !3),
        _ => actual,
    })
}

fn traced(fault: Fault, f: impl FnOnce()) -> Vec<Event> {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            TRACE.with_borrow_mut(|trace| *trace = None);
        }
    }
    TRACE.with_borrow_mut(|trace| {
        assert!(trace.is_none());
        *trace = Some(Trace {
            fault,
            events: Vec::new(),
        });
    });
    let _reset = Reset;
    f();
    TRACE.with_borrow_mut(|trace| trace.take().unwrap().events)
}

fn assert_balanced(events: &[Event], acquisitions: usize) {
    let mut references = HashMap::<usize, usize>::new();
    let mut total = 0;
    for event in events {
        match *event {
            Event::Acquire(raw) => {
                *references.entry(raw).or_default() += 1;
                total += 1;
            }
            Event::Release(raw) => {
                let count = references
                    .get_mut(&raw)
                    .expect("release requires an acquired owner");
                assert!(*count > 0, "duplicate release: {events:?}");
                *count -= 1;
            }
            _ => {}
        }
    }
    assert_eq!(total, acquisitions);
    assert!(
        references.values().all(|count| *count == 0),
        "leaked reference: {events:?}"
    );
}

fn calls(events: &[Event]) -> Vec<Call> {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Call(call) => Some(*call),
            _ => None,
        })
        .collect()
}

#[test]
fn checked_numeric_boundaries_without_native_allocation() {
    assert!(matches!(
        BlockPlan::checked(0, 1, 1),
        Err(MemError::ZeroSize)
    ));
    for value in [isize::MAX as usize + 1, usize::MAX] {
        assert!(matches!(
            BlockPlan::checked(value, 1, 1),
            Err(MemError::SizeOverflow)
        ));
    }
    let max = isize::MAX as usize;
    assert_eq!(
        BlockPlan::checked(max, 1, 1).unwrap().allocation_size(),
        max
    );
    let last = max / 128 * 128;
    assert_eq!(
        BlockPlan::checked(last, 128, 1).unwrap().allocation_size(),
        last
    );
    assert!(matches!(
        BlockPlan::checked(last + 1, 128, 1),
        Err(MemError::SizeOverflow)
    ));
    assert!(BlockPlan::checked(last, 1, 128).is_ok());
    assert!(matches!(
        BlockPlan::checked(last + 1, 1, 128),
        Err(MemError::SizeOverflow)
    ));
    assert!(matches!(
        aligned_size(usize::MAX - 1, 4),
        Err(MemError::SizeOverflow)
    ));
    for (row, allocation) in [(0, 1), (1, 0)] {
        assert!(matches!(
            BlockPlan::checked(1, row, allocation),
            Err(MemError::BlockAlignmentInvalid)
        ));
    }
}

#[test]
fn checked_rounding_and_native_disagreement() {
    for (request, row_align, alloc_align, row, allocation) in [
        (1, 3, 5, 3, 5),
        (3, 3, 5, 3, 5),
        (4, 3, 5, 6, 10),
        (128, 128, 256, 128, 256),
        (256, 128, 256, 256, 256),
    ] {
        let plan = BlockPlan::checked(request, row_align, alloc_align).unwrap();
        assert_eq!(plan.requested_size(), request);
        assert_eq!(plan.row_bytes(), row);
        assert_eq!(plan.allocation_size(), allocation);
        assert!(plan.check_native_alignment(row, allocation).is_ok());
        for pair in [(row + 1, allocation), (row, allocation + 1)] {
            assert!(matches!(
                plan.check_native_alignment(pair.0, pair.1),
                Err(MemError::BlockAlignmentInvalid)
            ));
        }
    }
}

#[test]
fn rejected_sizes_and_planning_create_no_native_owners() {
    let events = traced(Fault::None, || {
        assert!(matches!(Block::open(0), Err(MemError::ZeroSize)));
        assert!(matches!(
            Block::open(usize::MAX),
            Err(MemError::SizeOverflow)
        ));
        let _ = Block::plan(1).unwrap();
    });
    assert!(events.is_empty());
}

#[test]
fn every_number_null_releases_only_preceding_create_references() {
    let plan = Block::plan(13).unwrap();
    for index in 1..=6 {
        let events = traced(Fault::Fail(Call::Number, index), || {
            assert!(matches!(plan.open(), Err(MemError::BlockPropertiesFailed)));
        });
        assert_balanced(&events, index - 1);
        assert_eq!(calls(&events), vec![Call::Number; index]);
    }
}

#[test]
fn dictionary_and_surface_null_release_properties() {
    let plan = Block::plan(13).unwrap();
    for (stage, count) in [(Call::Dictionary, 6), (Call::Surface, 7)] {
        let events = traced(Fault::Fail(stage, 1), || {
            let error = plan.open().err().unwrap();
            match stage {
                Call::Dictionary => assert!(matches!(error, MemError::BlockPropertiesFailed)),
                Call::Surface => assert!(matches!(error, MemError::BlockCreateFailed)),
                _ => unreachable!(),
            }
        });
        assert_balanced(&events, count);
        let mut expected = vec![Call::Number; 6];
        expected.push(Call::Dictionary);
        if stage == Call::Surface {
            expected.push(Call::Surface);
        }
        assert_eq!(calls(&events), expected);
        assert!(!events.contains(&Event::Unlocked));
    }
}

#[test]
fn extent_mismatch_is_fatal_before_lock() {
    let plan = Block::plan(13).unwrap();
    for actual in [
        0,
        plan.allocation_size() - 1,
        plan.allocation_size() + 1,
        usize::MAX,
    ] {
        let events = traced(Fault::Extent(actual), || match plan.open().err().unwrap() {
            MemError::BlockExtentMismatch {
                expected,
                actual: observed,
            } => {
                assert_eq!(expected, plan.allocation_size());
                assert_eq!(observed, actual);
            }
            other => panic!("unexpected error: {other:?}"),
        });
        assert_balanced(&events, 8);
        assert_eq!(calls(&events).last(), Some(&Call::Extent));
        assert!(!events.contains(&Event::Locked));
        assert!(!events.contains(&Event::Unlocked));
    }
}

#[test]
fn failed_lock_releases_without_unlock() {
    let plan = Block::plan(13).unwrap();
    let events = traced(Fault::Fail(Call::Lock, 1), || {
        assert!(matches!(
            plan.open(),
            Err(MemError::BlockLockFailed(LOCK_ERROR))
        ));
    });
    assert_balanced(&events, 8);
    assert_eq!(calls(&events).last(), Some(&Call::Lock));
    assert!(!events.contains(&Event::Locked));
    assert!(!events.contains(&Event::Unlocked));
}

fn assert_unlock_before_final_release(events: &[Event]) {
    assert_eq!(events.iter().filter(|e| **e == Event::Locked).count(), 1);
    assert_eq!(events.iter().filter(|e| **e == Event::Unlocked).count(), 1);
    assert_eq!(events[events.len() - 2], Event::Unlocked);
    assert!(matches!(events.last(), Some(Event::Release(_))));
}

#[test]
fn mapping_rejections_unlock_then_release() {
    let plan = Block::plan(13).unwrap();
    for fault in [Fault::NullBase, Fault::MisalignedBase, Fault::WrappingBase] {
        let events = traced(fault, || {
            assert!(matches!(plan.open(), Err(MemError::BlockAddressInvalid)));
        });
        assert_balanced(&events, 8);
        assert_eq!(calls(&events).last(), Some(&Call::Base));
        assert_unlock_before_final_release(&events);
    }
}

#[test]
fn success_transfers_one_locked_owner_to_block_drop() {
    let plan = Block::plan(13).unwrap();
    let events = traced(Fault::None, || {
        let block = plan.open().unwrap();
        TRACE.with_borrow(|trace| {
            assert!(!trace.as_ref().unwrap().events.contains(&Event::Unlocked));
        });
        assert_eq!(block.size(), plan.allocation_size());
        drop(block);
    });
    assert_balanced(&events, 8);
    assert_eq!(calls(&events).last(), Some(&Call::Id));
    assert_unlock_before_final_release(&events);
}
