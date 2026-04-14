//! Convenience re-exports for numeric type conversions.
//!
//! All conversion functions are available directly under `acpu::convert::*`
//! or via the top-level `acpu::cast_*` aliases.

pub use crate::numeric::bf16::{bf16_to_f32, cast_bf16_f32, cast_f32_bf16, f32_to_bf16};
pub use crate::numeric::fp16::{cast_f16_f32, cast_f32_f16, f32_to_fp16, fp16_to_f32};
pub use crate::numeric::quant::{cast_f32_i8, cast_i8_f32};
