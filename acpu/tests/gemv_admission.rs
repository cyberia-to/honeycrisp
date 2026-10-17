#![cfg(target_arch = "aarch64")]

use acpu::gemm::{
    gemv_amx_asm::gemv_pure_asm,
    gemv_kern::gemv_asm,
    matvec::{matvec_f32, matvec_f32_set},
};
use std::panic::{catch_unwind, AssertUnwindSafe};

type Kernel = fn(&[f32], &[f32], &mut [f32], usize, usize);

fn kernels() -> [Kernel; 4] {
    [matvec_f32, matvec_f32_set, gemv_asm, gemv_pure_asm]
}

#[test]
fn malformed_dimensions_preserve_output() {
    for kernel in kernels() {
        for (a_len, b_len, c_len) in [(7, 64, 8), (8, 63, 8), (8, 64, 7)] {
            let a = vec![1.0; a_len];
            let b = vec![2.0; b_len];
            let mut c = vec![13.5; c_len];
            let before = c.clone();
            assert!(catch_unwind(AssertUnwindSafe(|| kernel(&a, &b, &mut c, 8, 8))).is_err());
            assert_eq!(c, before);
        }
    }
}

#[test]
fn overflowing_dimensions_preserve_output() {
    for kernel in kernels() {
        let mut c = [13.5; 3];
        let error = catch_unwind(AssertUnwindSafe(|| {
            kernel(&[1.0; 8], &[], &mut c, usize::MAX, 8);
        }))
        .expect_err("overflowing dimensions must fail before access");
        let message = error
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| error.downcast_ref::<&str>().copied())
            .unwrap_or("");
        assert!(message.contains("k*n overflow"), "{message}");
        assert_eq!(c, [13.5; 3]);
    }
}

#[test]
fn pure_assembly_retains_k_multiple_of_eight_precondition() {
    let mut c = [13.5; 8];
    assert!(catch_unwind(AssertUnwindSafe(|| {
        gemv_pure_asm(&[1.0; 7], &[2.0; 56], &mut c, 8, 7);
    }))
    .is_err());
    assert_eq!(c, [13.5; 8]);
}

#[test]
fn empty_dimensions_follow_accumulate_and_overwrite_contracts() {
    for kernel in kernels() {
        kernel(&[], &[], &mut [], 0, 0);
        kernel(&[1.0; 8], &[], &mut [], 0, 8);
    }
    for n in [1, 16, 128, 1024, 1025] {
        let mut c = vec![13.5; n];
        matvec_f32(&[], &[], &mut c, n, 0);
        assert_eq!(c, vec![13.5; n]);
        for kernel in [matvec_f32_set, gemv_asm, gemv_pure_asm] {
            c.fill(13.5);
            kernel(&[], &[], &mut c, n, 0);
            assert_eq!(c, vec![0.0; n]);
        }
    }
}
