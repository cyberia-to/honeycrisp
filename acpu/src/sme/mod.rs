//! SME f32 matmul — FMOPA outer-product GEMM.
//!
//! Public entry points are `matmul_f32_sme` (C += A·B) and
//! `matmul_f32_sme_set` (C = A·B). Both gate on FEAT_SME and fall back
//! to NEON / AMX for sizes / chips where the SME path would lose.

pub mod asm;
pub mod gemm;
pub mod tile;
pub mod tile_i16;

pub use gemm::{matmul_f32_sme, matmul_f32_sme_set};
