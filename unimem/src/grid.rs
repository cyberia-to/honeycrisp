use crate::tape::Tape;
use crate::MemError;
use crossbeam_queue::SegQueue;
use std::marker::PhantomData;

/// Grid of fixed-size cells over a pinned tape.
///
/// Take a cell, use it, give it back. ~10ns cycle.
/// Like a spreadsheet: every cell is the same size.
pub struct Grid<const CELL_SIZE: usize, const CELLS: usize> {
    tape: Tape,
    free: SegQueue<usize>,
}

/// One cell from a grid. Cannot outlive the grid.
pub struct Cell<'a> {
    ptr: *mut u8,
    index: usize,
    _grid: PhantomData<&'a ()>,
}

impl<const CELL_SIZE: usize, const CELLS: usize> Grid<CELL_SIZE, CELLS> {
    /// Create a grid. CELL_SIZE must be a multiple of 64 (AMX alignment).
    pub fn new() -> Result<Self, MemError> {
        assert!(CELL_SIZE > 0, "CELL_SIZE must be > 0");
        assert!(
            CELL_SIZE.is_multiple_of(64),
            "CELL_SIZE must be multiple of 64"
        );
        assert!(CELLS > 0, "CELLS must be > 0");

        let tape = Tape::start(CELL_SIZE * CELLS)?;
        let free = SegQueue::new();

        for i in 0..CELLS {
            let _ = tape.take(CELL_SIZE, 64);
            free.push(i);
        }

        Ok(Grid { tape, free })
    }

    /// Take a cell. Returns None if all cells are in use.
    #[inline]
    pub fn take(&self) -> Option<Cell<'_>> {
        let index = self.free.pop()?;
        let base = self.tape.block().address();
        let ptr = unsafe { base.add(index * CELL_SIZE) };
        Some(Cell {
            ptr,
            index,
            _grid: PhantomData,
        })
    }

    /// Give a cell back.
    #[inline]
    pub fn give(&self, cell: Cell<'_>) {
        self.free.push(cell.index);
    }

    /// How many cells are free.
    pub fn free(&self) -> usize {
        self.free.len()
    }

    /// Total cells (compile-time).
    #[inline]
    pub const fn total(&self) -> usize {
        CELLS
    }

    /// Access the backing tape.
    #[inline]
    pub fn tape(&self) -> &Tape {
        &self.tape
    }
}

impl<'a> Cell<'a> {
    /// Memory address of this cell.
    #[inline(always)]
    pub fn address(&self) -> *mut u8 {
        self.ptr
    }

    /// Cell contents as a mutable byte slice.
    ///
    /// # Safety
    /// `len` must be <= CELL_SIZE. Caller must ensure exclusive access.
    #[inline]
    pub unsafe fn bytes(&mut self, len: usize) -> &mut [u8] {
        std::slice::from_raw_parts_mut(self.ptr, len)
    }

    /// Cell number.
    #[inline]
    pub fn id(&self) -> usize {
        self.index
    }
}
