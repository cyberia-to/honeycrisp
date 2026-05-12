//! acpu — pure Rust driver for Apple Silicon CPU compute
//!
//! direct access to every useful compute unit in M1–M4:
//! matrix coprocessor (AMX), vector engine (NEON), numeric
//! extensions, sync primitives, and performance counters.
//!
//! memory via [`unimem::Block`] — IOSurface-backed pinned pages
//! shared with GPU (aruminium) and ANE (rane). zero-copy by default.

pub mod convert;
pub mod crypto;
pub mod field;
pub mod gemm;
pub mod sparse;
pub mod matrix;
pub mod numeric;
pub mod probe;
pub mod pulse;
pub mod sync;
pub mod vector;

// Memory foundation — re-export for single-import convenience
pub use unimem::{Block, Layout, MemError, Tape};

pub use convert::{
    cast_bf16_f32, cast_f16_f32, cast_f32_bf16, cast_f32_f16, cast_f32_i8, cast_i8_f32,
};
pub use gemm::{matmul_bf16, matmul_f16, matmul_f32, matmul_f32_set, matmul_i8};
pub use sparse::{csr_matvec, csr_matvec_set};
pub use matrix::Matrix;
pub use numeric::{bf16, complex, fp16, quant};
pub use probe::{scan, Chip, Feature, Features};
pub use pulse::Counters;
pub use sync::{affinity, prefetch};

use std::fmt;

/// all acpu errors
#[derive(Debug)]
pub enum CpuError {
    /// running on non-Apple-Silicon hardware
    ChipNotSupported,
    /// AMX_SET instruction failed
    AmxSetFailed,
    /// AMX operation error
    AmxOpFailed(String),
    /// libkperf.dylib not found or kpc access denied
    PmuNotAvailable,
    /// PMU counter configuration rejected
    PmuConfigFailed(String),
    /// required CPU extension absent on this chip
    FeatureNotAvailable(Feature),
    /// QoS class change failed
    AffinityFailed(String),
    /// sysctl query failed
    SysctlFailed(String),
}

impl fmt::Display for CpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChipNotSupported => write!(f, "not Apple Silicon"),
            Self::AmxSetFailed => write!(f, "AMX_SET failed"),
            Self::AmxOpFailed(s) => write!(f, "AMX op failed: {s}"),
            Self::PmuNotAvailable => write!(f, "PMU not available (libkperf.dylib)"),
            Self::PmuConfigFailed(s) => write!(f, "PMU config failed: {s}"),
            Self::FeatureNotAvailable(feat) => write!(f, "feature not available: {feat:?}"),
            Self::AffinityFailed(s) => write!(f, "affinity failed: {s}"),
            Self::SysctlFailed(s) => write!(f, "sysctl failed: {s}"),
        }
    }
}

impl std::error::Error for CpuError {}

pub type Result<T> = std::result::Result<T, CpuError>;
