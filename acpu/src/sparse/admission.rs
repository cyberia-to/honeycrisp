//! Private CSR evidence reused across operations on equally sized vectors.

pub(super) struct Csr<'a> {
    row_ptr: &'a [u32],
    col_idx: &'a [u32],
    values: &'a [f32],
    columns: usize,
}

impl<'a> Csr<'a> {
    pub(super) fn new(
        row_ptr: &'a [u32],
        col_idx: &'a [u32],
        values: &'a [f32],
        columns: usize,
    ) -> Self {
        assert!(!row_ptr.is_empty(), "CSR: row_ptr must be nonempty");
        assert_eq!(row_ptr[0], 0, "CSR: row_ptr must start at zero");
        assert_eq!(col_idx.len(), values.len(), "CSR: column/value lengths");
        assert!(
            row_ptr.windows(2).all(|pair| pair[0] <= pair[1]),
            "CSR: row_ptr must be nondecreasing"
        );
        assert_eq!(
            row_ptr[row_ptr.len() - 1] as usize,
            col_idx.len(),
            "CSR: terminal offset must equal entry count"
        );
        assert!(
            col_idx.iter().all(|&column| (column as usize) < columns),
            "CSR: column index exceeds input length"
        );
        Self {
            row_ptr,
            col_idx,
            values,
            columns,
        }
    }

    pub(super) fn accumulate(&self, x: &[f32], y: &mut [f32]) {
        self.apply(x, y, false);
    }

    pub(super) fn set(&self, x: &[f32], y: &mut [f32]) {
        self.apply(x, y, true);
    }

    fn apply(&self, x: &[f32], y: &mut [f32], set: bool) {
        assert_eq!(x.len(), self.columns, "CSR: input length changed");
        let rows = self.row_ptr.len() - 1;
        assert_eq!(y.len(), rows, "CSR: output length must equal row count");
        if set {
            y.fill(0.0);
        }
        // The immutable partition covers both entry slices exactly; columns
        // fit x, and every row fits y. These facts hold for all unchecked
        // loads/stores and prefetch pointer arithmetic in the existing kernel.
        super::inner(self.row_ptr, self.col_idx, self.values, x, y, rows);
    }
}
