use unimem::{Block, Tape};

fn initialized(size: usize) -> Block {
    let block = Block::open(size).unwrap();
    // SAFETY: Fresh unpublished mapping; initialize its full actual extent.
    unsafe { block.address().write_bytes(0, block.size()) };
    block
}

#[test]
fn block_slice_lengths() {
    let mut b = initialized(4096);
    let actual = b.size();
    // SAFETY: All bytes initialized; each view ends before the next, no aliases.
    unsafe {
        assert_eq!(b.as_bytes().len(), actual);
        assert_eq!(b.as_bytes_mut().len(), actual);
        assert_eq!(b.as_f32().len(), actual / 4);
        assert_eq!(b.as_f32_mut().len(), actual / 4);
        assert_eq!(b.as_u16().len(), actual / 2);
        assert_eq!(b.as_u16_mut().len(), actual / 2);
    }
}

#[test]
fn block_slice_pointer() {
    let b = initialized(4096);
    // SAFETY: All bytes initialized; shared reads only, no device use.
    unsafe {
        assert_eq!(b.as_bytes().as_ptr(), b.address() as *const u8);
        assert_eq!(b.as_f32().as_ptr(), b.address() as *const f32);
        assert_eq!(b.as_u16().as_ptr(), b.address() as *const u16);
    }
}

#[test]
fn block_cross_view_f32_bytes() {
    let mut b = initialized(4096);
    let val = f32::from_bits(0x4048_f5c3); // Exact 3.14f32 byte-pattern fixture.
                                           // SAFETY: Full backing initialized; exclusive write ends before the read.
    unsafe {
        b.as_f32_mut()[0] = val;
        assert_eq!(&b.as_bytes()[..4], &val.to_ne_bytes());
    }
}

#[test]
fn block_cross_view_u16_bytes() {
    let mut b = initialized(4096);
    // SAFETY: Full backing initialized; exclusive write ends before the read.
    unsafe {
        b.as_u16_mut()[0] = 0xBEEF;
        assert_eq!(&b.as_bytes()[..2], &0xBEEFu16.to_ne_bytes());
    }
}

#[test]
fn awkward_extents_include_padding_and_preserve_lane_order() {
    for requested in [1, 13, std::mem::align_of::<f32>() + 1] {
        let mut block = initialized(requested);
        let actual = block.size();
        assert!(actual >= requested);
        // SAFETY: Full actual extent initialized, no aliases/device access;
        // each mutable view ends before any subsequent view is formed.
        unsafe {
            assert!(block.as_bytes().iter().all(|&byte| byte == 0));
            block.as_bytes_mut().fill(0xA5);
            assert_eq!(block.as_u16().len(), actual / 2);
            assert!(block.as_u16().iter().all(|&lane| lane == 0xA5A5));
            assert_eq!(block.as_f32().len(), actual / 4);
            assert!(block
                .as_f32()
                .iter()
                .all(|lane| lane.to_bits() == 0xA5A5_A5A5));
            block.as_u16_mut().fill(0xBEEF);
            for bytes in block.as_bytes().chunks_exact(2) {
                assert_eq!(bytes, 0xBEEFu16.to_ne_bytes());
            }
            block.as_f32_mut().fill(1.0);
            for bytes in block.as_bytes().chunks_exact(4) {
                assert_eq!(bytes, 1.0f32.to_ne_bytes());
            }
        }
    }
}

#[test]
fn warm_changes_only_fixed_stride_bytes_and_preserves_cursor() {
    for requested in [1, 16_384, 16_384 + 13, 32_768 + 1] {
        let mut tape = Tape::start(requested).unwrap();
        let actual = tape.total();
        // SAFETY: Fresh unpublished owner, no active CPU/device access.
        unsafe { tape.block().address().write_bytes(0xA5, actual) };
        let _ = tape.take(1, 1).unwrap();
        let before = (tape.used(), tape.free());
        // SAFETY: The raw take is unused; no references or external aliases.
        unsafe { tape.warm() };
        assert_eq!((tape.used(), tape.free()), before);
        // SAFETY: Full backing initialized before warm; no concurrent writer.
        for (offset, &byte) in unsafe { tape.block().as_bytes() }.iter().enumerate() {
            let expected = if offset % 16_384 == 0 { 0 } else { 0xA5 };
            assert_eq!(byte, expected, "requested={requested}, offset={offset}");
        }
    }
}

#[test]
fn initialized_owner_moves_and_shared_reads_finish_before_mutation() {
    let block = initialized(13);
    std::thread::scope(|scope| {
        let read = || {
            // SAFETY: Full initialization, shared readers only, no device uses.
            unsafe { block.as_bytes().iter().all(|&byte| byte == 0) }
        };
        let a = scope.spawn(read);
        let b = scope.spawn(read);
        assert!(a.join().unwrap());
        assert!(b.join().unwrap());
    });
    let block = std::thread::spawn(move || {
        let mut block = block;
        // SAFETY: Readers have joined; this thread exclusively owns the mapping.
        unsafe { block.as_bytes_mut().fill(0x37) };
        block
    })
    .join()
    .unwrap();
    // SAFETY: Writer joined; initialized backing now has only this reader.
    assert!(unsafe { block.as_bytes() }.iter().all(|&byte| byte == 0x37));
}
