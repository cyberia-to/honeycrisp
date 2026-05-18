//! Goldilocks field arithmetic — hardware-accelerated primitives.
//!
//! p = 2^64 - 2^32 + 1. Every element is one u64.
//! These are the raw asm-optimized kernels that nebu calls on aarch64.

pub mod goldilocks;
pub mod merkle;
pub mod ntt;
pub mod permute;
pub mod poly;

pub use goldilocks::{
    gl_add, gl_inv, gl_mul, gl_mul_batch, gl_pow7, gl_pow7_x16, gl_reduce128, gl_sub,
};
pub use merkle::{batch_inv, hash_node, merkle_root};
pub use ntt::{bit_reverse_permute, ntt_forward, pow_gl};
pub use permute::poseidon2_permute;
pub use poly::multilinear_eval;
