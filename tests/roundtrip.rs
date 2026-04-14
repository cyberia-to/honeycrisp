use unimem::{Block, Grid, Layout, MemError, Tape};

#[test]
fn block_open_and_access() {
    let b = Block::open(4096).unwrap();
    assert!(b.size() >= 4096);
    assert!(!b.address().is_null());
    assert!(b.id() > 0);

    unsafe {
        let ptr = b.address();
        ptr.write_bytes(0xAB, 4096);
        assert_eq!(*ptr, 0xAB);
        assert_eq!(*ptr.add(4095), 0xAB);
    }
}

#[test]
fn block_zero_size_errors() {
    assert!(Block::open(0).is_err());
}

#[test]
fn block_large() {
    let b = Block::open(256 * 1024 * 1024).unwrap();
    assert!(b.size() >= 256 * 1024 * 1024);
    unsafe {
        b.address().write(42);
        b.address().add(b.size() - 1).write(99);
        assert_eq!(*b.address(), 42);
        assert_eq!(*b.address().add(b.size() - 1), 99);
    }
}

#[test]
fn tape_basic_take() {
    let tape = Tape::start(1024 * 1024).unwrap();
    assert_eq!(tape.used(), 0);

    let p1 = tape.take(256, 64).unwrap();
    assert_eq!(tape.used(), 256);
    assert!(tape.owns(p1));

    let p2 = tape.take(512, 64).unwrap();
    assert!(tape.used() >= 768);
    assert!(tape.owns(p2));
    assert_ne!(p1, p2);

    unsafe {
        p1.write_bytes(0x11, 256);
        p2.write_bytes(0x22, 512);
        assert_eq!(*p1, 0x11);
        assert_eq!(*p2, 0x22);
    }
}

#[test]
fn tape_clear() {
    let tape = Tape::start(4096).unwrap();
    let _ = tape.take(1000, 1).unwrap();
    assert!(tape.used() >= 1000);

    tape.clear();
    assert_eq!(tape.used(), 0);
    assert_eq!(tape.free(), tape.total());

    let p = tape.take(2000, 1).unwrap();
    assert!(!p.is_null());
}

#[test]
fn tape_full() {
    let tape = Tape::start(4096).unwrap();
    let cap = tape.total();
    let _ = tape.take(cap, 1).unwrap();
    assert_eq!(tape.free(), 0);
    assert!(tape.take(1, 1).is_none());
}

#[test]
fn tape_alignment() {
    let tape = Tape::start(1024 * 1024).unwrap();
    let _ = tape.take(1, 1).unwrap();
    let p = tape.take(64, 64).unwrap();
    assert_eq!(p as usize % 64, 0);

    let p2 = tape.take(4096, 4096).unwrap();
    assert_eq!(p2 as usize % 4096, 0);
}

#[test]
fn tape_owns() {
    let tape = Tape::start(4096).unwrap();
    let p = tape.take(100, 1).unwrap();
    assert!(tape.owns(p));
    assert!(tape.owns(unsafe { p.add(99) }));
    assert!(!tape.owns(std::ptr::null()));
    assert!(!tape.owns(0x1 as *const u8));
}

#[test]
fn tape_owns_boundary() {
    let tape = Tape::start(4096).unwrap();
    let base = tape.block().address() as usize;
    let total = tape.total();

    // First byte: inside
    assert!(tape.owns(base as *const u8));
    // Last byte: inside
    assert!(tape.owns((base + total - 1) as *const u8));
    // One past end: outside
    assert!(!tape.owns((base + total) as *const u8));
    // One before start: outside
    assert!(!tape.owns((base.wrapping_sub(1)) as *const u8));
}

#[test]
fn tape_take_one() {
    let tape = Tape::start(4096).unwrap();
    let p: *mut u64 = tape.take_one::<u64>().unwrap();
    assert_eq!(p as usize % std::mem::align_of::<u64>(), 0);
    unsafe {
        p.write(0xDEADBEEF_CAFEBABE);
        assert_eq!(p.read(), 0xDEADBEEF_CAFEBABE);
    }
}

#[test]
fn grid_take_give() {
    let grid: Grid<256, 4> = Grid::new().unwrap();
    assert_eq!(grid.free(), 4);
    assert_eq!(grid.total(), 4);

    let c0 = grid.take().unwrap();
    let c1 = grid.take().unwrap();
    assert_eq!(grid.free(), 2);
    assert_ne!(c0.address(), c1.address());

    unsafe {
        c0.address().write_bytes(0xAA, 256);
        c1.address().write_bytes(0xBB, 256);
        assert_eq!(*c0.address(), 0xAA);
        assert_eq!(*c1.address(), 0xBB);
    }

    grid.give(c0);
    assert_eq!(grid.free(), 3);
    grid.give(c1);
    assert_eq!(grid.free(), 4);
}

#[test]
fn grid_full() {
    let grid: Grid<64, 2> = Grid::new().unwrap();
    let _c0 = grid.take().unwrap();
    let _c1 = grid.take().unwrap();
    assert!(grid.take().is_none());
}

#[test]
fn grid_reuse() {
    let grid: Grid<64, 1> = Grid::new().unwrap();
    let cell = grid.take().unwrap();
    let addr = cell.address();
    grid.give(cell);

    let cell2 = grid.take().unwrap();
    assert_eq!(cell2.address(), addr);
    grid.give(cell2);
}

// ── Layout tests ──

#[test]
fn layout_new_total() {
    let w = 1024 * 1024;
    let s = 256 * 1024;
    let h = 256 * 1024;
    let lay = Layout::new(w, s, h).unwrap();
    assert_eq!(
        lay.total(),
        lay.weights().total() + lay.scratch().total() + lay.history().total()
    );
    assert!(lay.weights().total() >= w);
    assert!(lay.scratch().total() >= s);
    assert!(lay.history().total() >= h);
}

#[test]
fn layout_tape_accessors() {
    let lay = Layout::new(1024 * 1024, 256 * 1024, 256 * 1024).unwrap();

    let pw = lay.weights().take(128, 64).unwrap();
    assert!(lay.weights().used() >= 128);
    assert!(!pw.is_null());

    let ps = lay.scratch().take(64, 64).unwrap();
    assert!(lay.scratch().used() >= 64);
    assert!(!ps.is_null());

    let ph = lay.history().take(64, 64).unwrap();
    assert!(lay.history().used() >= 64);
    assert!(!ph.is_null());
}

#[test]
fn layout_clear_pass() {
    let lay = Layout::new(1024 * 1024, 256 * 1024, 256 * 1024).unwrap();

    let _ = lay.weights().take(512, 64).unwrap();
    let _ = lay.scratch().take(256, 64).unwrap();

    let w_used = lay.weights().used();
    assert!(w_used >= 512);

    lay.clear_pass();

    assert_eq!(lay.weights().used(), w_used);
    assert_eq!(lay.scratch().used(), 0);
}

#[test]
fn layout_clear_talk() {
    let lay = Layout::new(1024 * 1024, 256 * 1024, 256 * 1024).unwrap();

    let _ = lay.weights().take(512, 64).unwrap();
    let _ = lay.scratch().take(256, 64).unwrap();
    let _ = lay.history().take(128, 64).unwrap();

    let w_used = lay.weights().used();

    lay.clear_talk();

    assert_eq!(lay.weights().used(), w_used);
    assert_eq!(lay.scratch().used(), 0);
    assert_eq!(lay.history().used(), 0);
}

#[test]
fn layout_stat() {
    let lay = Layout::new(1024 * 1024, 256 * 1024, 256 * 1024).unwrap();

    let _ = lay.weights().take(1000, 64).unwrap();
    let _ = lay.scratch().take(500, 64).unwrap();
    let _ = lay.history().take(200, 64).unwrap();

    let st = lay.stat();
    assert!(st.weights_used >= 1000);
    assert!(st.weights_total >= 1024 * 1024);
    assert!(st.scratch_used >= 500);
    assert!(st.scratch_total >= 256 * 1024);
    assert!(st.history_used >= 200);
    assert!(st.history_total >= 256 * 1024);
}

#[test]
fn stat_display() {
    let lay = Layout::new(1024 * 1024, 256 * 1024, 256 * 1024).unwrap();
    let st = lay.stat();
    let s = format!("{}", st);
    assert!(!s.is_empty());
    assert!(s.contains("weights:"));
    assert!(s.contains("scratch:"));
    assert!(s.contains("history:"));
}

// ── Tape extended tests ──

#[test]
fn tape_start_warm() {
    let tape = Tape::start_warm(64 * 1024).unwrap();
    let p = tape.take(1024, 64).unwrap();
    assert!(!p.is_null());
    assert!(tape.used() >= 1024);
}

#[test]
fn tape_warm_after_start() {
    let tape = Tape::start(64 * 1024).unwrap();
    tape.warm();
    let p = tape.take(512, 64).unwrap();
    assert!(!p.is_null());
}

#[test]
fn tape_block_access() {
    let tape = Tape::start(4096).unwrap();
    let blk = tape.block();
    assert!(!blk.address().is_null());
    assert!(blk.size() >= 4096);

    let p = tape.take(64, 1).unwrap();
    let base = blk.address() as usize;
    let p_addr = p as usize;
    assert!(p_addr >= base && p_addr < base + blk.size());
}

// ── Grid extended tests ──

#[test]
fn grid_tape_access() {
    let grid: Grid<256, 4> = Grid::new().unwrap();
    let t = grid.tape();
    assert!(!t.block().address().is_null());
    assert!(t.total() >= 256 * 4);
}

#[test]
fn grid_address_spacing() {
    let grid: Grid<256, 4> = Grid::new().unwrap();
    let mut cells = Vec::new();
    for _ in 0..4 {
        cells.push(grid.take().unwrap());
    }
    let mut addrs: Vec<usize> = cells.iter().map(|c| c.address() as usize).collect();
    addrs.sort();
    for i in 1..addrs.len() {
        assert_eq!(addrs[i] - addrs[i - 1], 256);
    }
    for c in cells {
        grid.give(c);
    }
}

// ── Cell tests ──

#[test]
fn cell_bytes() {
    let grid: Grid<256, 1> = Grid::new().unwrap();
    let mut cell = grid.take().unwrap();
    unsafe {
        let slice = cell.bytes(256);
        slice[0] = 0xDE;
        slice[255] = 0xAD;
        assert_eq!(slice[0], 0xDE);
        assert_eq!(slice[255], 0xAD);
    }
    grid.give(cell);
}

#[test]
fn cell_id_distinct() {
    let grid: Grid<64, 4> = Grid::new().unwrap();
    let c0 = grid.take().unwrap();
    let c1 = grid.take().unwrap();
    let c2 = grid.take().unwrap();
    let c3 = grid.take().unwrap();

    let mut ids = vec![c0.id(), c1.id(), c2.id(), c3.id()];
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), 4);

    for &id in &ids {
        assert!(id < 4);
    }

    grid.give(c0);
    grid.give(c1);
    grid.give(c2);
    grid.give(c3);
}

// ── Block extended tests ──

#[test]
fn block_handle() {
    let b = Block::open(4096).unwrap();
    assert!(!b.handle().is_null());
}

#[test]
fn block_slice_lengths() {
    let size = 4096;
    let b = Block::open(size).unwrap();
    let actual = b.size();
    assert_eq!(b.as_bytes().len(), actual);
    assert_eq!(b.as_bytes_mut().len(), actual);
    assert_eq!(b.as_f32().len(), actual / 4);
    assert_eq!(b.as_f32_mut().len(), actual / 4);
    assert_eq!(b.as_u16().len(), actual / 2);
    assert_eq!(b.as_u16_mut().len(), actual / 2);
}

#[test]
fn block_slice_pointer() {
    let b = Block::open(4096).unwrap();
    assert_eq!(b.as_bytes().as_ptr(), b.address() as *const u8);
    assert_eq!(b.as_f32().as_ptr(), b.address() as *const f32);
    assert_eq!(b.as_u16().as_ptr(), b.address() as *const u16);
}

#[test]
fn block_cross_view_f32_bytes() {
    let b = Block::open(4096).unwrap();
    let val: f32 = 3.14;
    b.as_f32_mut()[0] = val;
    let expected = val.to_ne_bytes();
    assert_eq!(&b.as_bytes()[..4], &expected);
}

#[test]
fn block_cross_view_u16_bytes() {
    let b = Block::open(4096).unwrap();
    b.as_u16_mut()[0] = 0xBEEF;
    let expected = 0xBEEFu16.to_ne_bytes();
    assert_eq!(&b.as_bytes()[..2], &expected);
}

#[test]
fn block_id_unique() {
    let a = Block::open(4096).unwrap();
    let b = Block::open(4096).unwrap();
    assert_ne!(a.id(), b.id());
}

#[test]
fn block_drop_no_leak() {
    for _ in 0..1000 {
        let b = Block::open(4096).unwrap();
        assert!(b.size() >= 4096);
    }
}

// ── MemError Display tests ──

#[test]
fn mem_error_display_zero_size() {
    let e = MemError::ZeroSize;
    let s = format!("{}", e);
    assert!(!s.is_empty());
    assert!(s.contains("zero"));
}

#[test]
fn mem_error_display_block_create_failed() {
    let e = MemError::BlockCreateFailed;
    let s = format!("{}", e);
    assert!(!s.is_empty());
    assert!(s.contains("IOSurfaceCreate"));
}

#[test]
fn mem_error_display_block_lock_failed() {
    let e = MemError::BlockLockFailed(0x1234);
    let s = format!("{}", e);
    assert!(!s.is_empty());
    assert!(s.contains("IOSurfaceLock"));
    assert!(s.contains("0x1234"));
}
