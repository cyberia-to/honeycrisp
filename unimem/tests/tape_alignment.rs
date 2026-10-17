use std::panic::{catch_unwind, AssertUnwindSafe};
use unimem::Tape;

#[test]
fn alignment_unsatisfied_by_actual_base_is_respected_or_exhausted() {
    let tape = Tape::start(4096).unwrap();
    let base = tape.block().address() as usize;
    // This alignment deterministically fails for this actual native base.
    let align = 1usize
        .checked_shl(base.trailing_zeros() + 1)
        .expect("native mapping must admit a larger power-of-two alignment");
    assert_ne!(base % align, 0);
    let padding = align - base % align;
    let result = tape.take(1, align);
    if padding < tape.total() {
        let address = result.expect("aligned byte fits in this allocation") as usize;
        assert_eq!(address % align, 0);
        assert_eq!(address, base + padding);
        assert_eq!(tape.used(), padding + 1);
    } else {
        assert!(result.is_none(), "alignment padding exceeds the allocation");
        assert_eq!(tape.used(), 0);
    }
}

#[test]
fn returned_addresses_satisfy_alignment_through_one_mebibyte() {
    let tape = Tape::start(1 << 22).unwrap();
    let base = tape.block().address() as usize;
    let mut previous_end = base;
    for shift in 0..=20 {
        let align = 1usize << shift;
        let ptr = tape.take(1, align).unwrap();
        let address = ptr as usize;
        assert_eq!(address % align, 0, "align={align}");
        assert!(tape.owns(ptr));
        assert!(address >= previous_end);
        assert_eq!(tape.used(), address - base + 1);
        previous_end = address + 1;
    }
}

#[test]
fn invalid_alignment_panics_without_advancing_even_for_zero_size() {
    let tape = Tape::start(4096).unwrap();
    tape.take(1, 1).unwrap();
    let before = tape.used();
    for align in [0, 3, 6] {
        for size in [0, 1] {
            assert!(catch_unwind(AssertUnwindSafe(|| tape.take(size, align))).is_err());
            assert_eq!(tape.used(), before);
        }
    }
    assert!(tape.take(0, 8).is_none());
    assert_eq!(tape.used(), before);
}

#[test]
fn overflow_and_exhaustion_leave_the_cursor_unchanged() {
    let tape = Tape::start(4096).unwrap();
    tape.take(8, 1).unwrap();
    let before = tape.used();
    assert!(tape.take(usize::MAX, 1).is_none());
    assert_eq!(tape.used(), before);
    assert!(tape.take(1, 1usize << (usize::BITS - 1)).is_none());
    assert_eq!(tape.used(), before);
    tape.take(tape.free(), 1).unwrap();
    assert_eq!(tape.used(), tape.total());
    assert!(tape.take(1, 1).is_none());
    assert_eq!(tape.used(), tape.total());
}

#[test]
fn take_one_respects_overaligned_type_without_forming_a_reference() {
    #[repr(align(65536))]
    struct Aligned;
    // An aligned zero-sized type preserves the existing zero-size refusal.
    let tape = Tape::start(1 << 20).unwrap();
    assert!(tape.take_one::<Aligned>().is_none());
    assert_eq!(tape.used(), 0);
    #[repr(align(65536))]
    struct Storage {
        _byte: u8,
    }
    let ptr = tape.take_one::<Storage>().unwrap();
    assert_eq!(ptr as usize % core::mem::align_of::<Storage>(), 0);
    assert!(tape.owns(ptr.cast()));
}
