//! Public precondition regressions. Malformed CSR cases must only run with the
//! complete admission guard present: the historical aarch64 code was unsound.

use acpu::field::{bit_reverse_permute, multilinear_eval, ntt_forward};
use acpu::{chebyshev_matvec, csr_matvec, csr_matvec_set};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn rejects(message: &str, f: impl FnOnce()) {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("input must be rejected");
    let text = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .expect("precondition panic has a message");
    assert!(text.contains(message), "expected {message:?}, got {text:?}");
}

type Matvec = fn(&[u32], &[u32], &[f32], &[f32], &mut [f32]);

#[test]
fn csr_rejects_every_malformed_shape_before_output_mutation() {
    // Partition, count, pointer, scalar-tail and late-row failures. Every case
    // runs through both public entry points, including the zeroing variant.
    let cases: &[(&[u32], &[u32], &[f32], usize, &str)] = &[
        (&[], &[], &[], 0, "row_ptr must be nonempty"),
        (&[1, 1], &[0], &[2.0], 1, "row_ptr must start at zero"),
        (&[0, 2, 1], &[0], &[2.0], 2, "row_ptr must be nondecreasing"),
        (&[0, 0], &[0], &[2.0], 1, "terminal offset"),
        (&[0, 2], &[0], &[2.0], 1, "terminal offset"),
        (&[0, 1], &[0], &[], 1, "column/value lengths"),
        (&[0, 1], &[], &[2.0], 1, "column/value lengths"),
        (&[0, 1], &[0], &[2.0], 0, "output length"),
        (&[0, 1], &[0], &[2.0], 2, "output length"),
        (&[0, 1], &[1], &[2.0], 1, "column index"),
        (&[0, 1], &[u32::MAX], &[2.0], 1, "column index"),
        (&[0, 1, 2], &[0, 1], &[2.0, 3.0], 2, "column index"),
        (
            &[0, 1, 2, 3, 4],
            &[0, 0, 0, 1],
            &[1.0; 4],
            4,
            "column index",
        ),
        (&[0, 4], &[0, 0, 0, 1], &[1.0; 4], 1, "column index"),
    ];
    for &(row, columns, values, output_len, message) in cases {
        for op in [csr_matvec as Matvec, csr_matvec_set] {
            let mut y: Vec<f32> = (0..output_len).map(|i| 17.0 + i as f32).collect();
            let before = y.clone();
            rejects(message, || op(row, columns, values, &[3.0], &mut y));
            assert_eq!(y, before, "output changed for {row:?}/{columns:?}");
        }
    }
    for op in [csr_matvec as Matvec, csr_matvec_set] {
        let mut y = [17.0];
        rejects("column index", || op(&[0, 1], &[0], &[1.0], &[], &mut y));
        assert_eq!(y, [17.0]);
    }
}

#[test]
fn csr_retains_rectangular_duplicates_empty_rows_and_kernel_paths() {
    // Three rows, five input columns; unsorted duplicates, scalar tail and a
    // long row exercising four-wide loads plus both prefetch loops.
    let row = [0, 0, 1, 18];
    let columns = [4, 3, 0, 3, 2, 1, 4, 2, 0, 1, 4, 3, 2, 1, 0, 4, 2, 0];
    let values: Vec<f32> = (0..columns.len()).map(|i| (i % 3 + 1) as f32).collect();
    let x = [1.0, 2.0, 4.0, 8.0, 16.0];
    let expected: Vec<f32> = row
        .windows(2)
        .map(|pair| {
            (pair[0]..pair[1])
                .map(|j| values[j] * x[columns[j] as usize])
                .sum()
        })
        .collect();
    let mut y = [10.0, 20.0, 30.0];
    csr_matvec(&row.map(|x| x as u32), &columns, &values, &x, &mut y);
    assert_eq!(
        y,
        [expected[0] + 10.0, expected[1] + 20.0, expected[2] + 30.0]
    );
    csr_matvec_set(&row.map(|x| x as u32), &columns, &values, &x, &mut y);
    assert_eq!(y.as_slice(), expected);

    for op in [csr_matvec as Matvec, csr_matvec_set] {
        op(&[0], &[], &[], &[], &mut []);
        op(&[0], &[], &[], &x, &mut []); // zero-by-five rectangular matrix
    }
    let mut empty_rows = [7.0, 9.0];
    csr_matvec(&[0, 0, 0], &[], &[], &[], &mut empty_rows);
    assert_eq!(empty_rows, [7.0, 9.0]);
    csr_matvec_set(&[0, 0, 0], &[], &[], &[], &mut empty_rows);
    assert_eq!(empty_rows, [0.0, 0.0]);
}

struct Heat {
    row: Vec<u32>,
    columns: Vec<u32>,
    values: Vec<f32>,
    diagonal: Vec<f32>,
    x: Vec<f32>,
    y: Vec<f32>,
    t0: Vec<f32>,
    t1: Vec<f32>,
}

impl Heat {
    fn valid() -> Self {
        Self {
            row: vec![0, 1, 2],
            columns: vec![1, 0],
            values: vec![1.0; 2],
            diagonal: vec![1.0; 2],
            x: vec![1.0, 0.0],
            y: vec![17.0, 19.0],
            t0: vec![23.0, 29.0],
            t1: vec![31.0, 37.0],
        }
    }

    fn run(&mut self, order: usize) {
        chebyshev_matvec(
            &self.row,
            &self.columns,
            &self.values,
            &self.diagonal,
            &self.x,
            &mut self.y,
            &mut self.t0,
            &mut self.t1,
            1.0,
            order,
        );
    }
}

#[test]
fn chebyshev_checks_all_dimensions_and_csr_before_allocation_or_mutation() {
    type Change = fn(&mut Heat);
    let cases: &[(Change, &str)] = &[
        (|h| h.y.clear(), "y length"),
        (|h| h.y.push(0.0), "y length"),
        (|h| h.t0.clear(), "t0 length"),
        (|h| h.t0.push(0.0), "t0 length"),
        (|h| h.t1.clear(), "t1 length"),
        (|h| h.t1.push(0.0), "t1 length"),
        (|h| h.diagonal.clear(), "diagonal length"),
        (|h| h.diagonal.push(0.0), "diagonal length"),
        (|h| h.x.clear(), "y length"),
        (|h| h.row.clear(), "row count"),
        (|h| h.row = vec![0, 2], "row count"),
        (|h| h.row.push(2), "row count"),
        (|h| h.row[0] = 1, "row_ptr must start at zero"),
        (|h| h.row = vec![0, 3, 2], "nondecreasing"),
        (|h| h.row[2] = 3, "terminal offset"),
        (|h| h.values.clear(), "column/value lengths"),
        (|h| h.columns[1] = 2, "column index"),
    ];
    for &(change, message) in cases {
        // The huge order would fail coefficient construction. Its shape panic
        // proves admission precedes that work, including the order-zero path.
        for order in [0, 20, usize::MAX] {
            let mut h = Heat::valid();
            change(&mut h);
            let before = (h.y.clone(), h.t0.clone(), h.t1.clone());
            rejects(message, || h.run(order));
            assert_eq!((h.y, h.t0, h.t1), before);
        }
    }
}

#[test]
fn chebyshev_reuses_validated_square_matrix_through_recurrence() {
    let mut h = Heat::valid();
    h.run(20);
    // Two-element normalized Laplacian [[1,-1],[-1,1]] has eigenvalues
    // zero and two. Its heat action on [1,0] is known without CSR recurrence.
    let decay = (-2.0_f32).exp();
    for (actual, expected) in h.y.iter().zip([(1.0 + decay) / 2.0, (1.0 - decay) / 2.0]) {
        assert!((actual - expected).abs() < 1e-6);
    }
    for order in [0, 1, 2, 20] {
        chebyshev_matvec(
            &[0],
            &[],
            &[],
            &[],
            &[],
            &mut [],
            &mut [],
            &mut [],
            1.0,
            order,
        );
    }
}

#[test]
fn field_shape_failures_are_consistent_and_preserve_permutation_inputs() {
    for len in [0, 3, 5, 6, 7, 9] {
        let mut values: Vec<u64> = (0..len).map(|i| i as u64 + 11).collect();
        let before = values.clone();
        rejects("bit_reverse_permute: length", || {
            bit_reverse_permute(&mut values)
        });
        assert_eq!(values, before);
        rejects("ntt_forward: length", || ntt_forward(&mut values, 1));
        assert_eq!(values, before);
        rejects("multilinear_eval: length", || {
            multilinear_eval(&values, &[0, 0, 0]);
        });
    }
    for len in [1_usize, 2, 4, 8, 16] {
        let evals = vec![7; len];
        let dimensions = len.trailing_zeros() as usize;
        for count in 0..=dimensions + 2 {
            if count != dimensions {
                rejects("coordinate count", || {
                    multilinear_eval(&evals, &vec![0; count]);
                });
            } else {
                assert_eq!(multilinear_eval(&evals, &vec![0; count]), 7);
            }
        }
    }
}
