#![cfg(target_arch = "aarch64")]

use acpu::gemm::{
    gemv_amx_asm::gemv_pure_asm,
    gemv_kern::gemv_asm,
    matvec::{matvec_f32, matvec_f32_set},
};

type Kernel = fn(&[f32], &[f32], &mut [f32], usize, usize);

// Integer products and sums stay exactly representable in f32. The oracle
// uses integer arithmetic, independent of the AMX/NEON/scalar implementations.
fn check(kernel: Kernel, name: &str, n: usize, k: usize, seed: usize, add: bool) {
    let a: Vec<i32> = (0..k).map(|i| ((i + seed) % 7) as i32 - 3).collect();
    let b: Vec<i32> = (0..k * n)
        .map(|i| ((i * 5 + seed * 3) % 11) as i32 - 5)
        .collect();
    let initial: Vec<i32> = (0..n).map(|j| 17 + (j % 13) as i32).collect();
    let expected: Vec<i32> = (0..n)
        .map(|j| {
            let sum: i32 = (0..k).map(|i| a[i] * b[i * n + j]).sum();
            sum + if add { initial[j] } else { 0 }
        })
        .collect();
    let a: Vec<f32> = a.into_iter().map(|v| v as f32).collect();
    let b: Vec<f32> = b.into_iter().map(|v| v as f32).collect();
    let mut output: Vec<f32> = initial.into_iter().map(|v| v as f32).collect();
    kernel(&a, &b, &mut output, n, k);
    for (j, (&actual, &expected)) in output.iter().zip(&expected).enumerate() {
        assert_eq!(
            actual, expected as f32,
            "{name}: n={n}, k={k}, seed={seed}, column={j}"
        );
    }
}

#[test]
fn asm_overwrite_has_no_previous_z_contents_or_missing_columns() {
    // Start each sweep by populating all Z rows. Subsequent odd tile counts,
    // different data and K remainders must initialize every used row anew.
    for seed in [1, 2, 5] {
        for n in [
            1024, 48, 64, 16, 17, 31, 32, 33, 1, 15, 1008, 1023, 1025, 2048, 2049, 0,
        ] {
            for k in [8, 16, 24, 0] {
                check(gemv_pure_asm, "pure asm", n, k, seed, false);
                check(gemv_asm, "Rust asm", n, k, seed + 1, false);
            }
            for k in [1, 7, 9, 17] {
                check(gemv_asm, "Rust asm K tail", n, k, seed, false);
            }
        }
    }
}

#[test]
fn public_matvec_preserves_accumulate_and_overwrite_at_dispatch_boundaries() {
    for seed in [1, 3] {
        for n in [
            0, 1, 15, 16, 17, 127, 128, 129, 1023, 1024, 1025, 2048, 2049,
        ] {
            for k in [0, 1, 7, 8, 17] {
                check(matvec_f32, "accumulate", n, k, seed, true);
                check(matvec_f32_set, "overwrite", n, k, seed + 1, false);
            }
        }
    }
}
