//! Bare-metal Rust drivers for Apple Silicon.
//!
//! Re-exports all four honeycrisp crates:
//! - [`unimem`] — zero-copy IOSurface memory
//! - [`acpu`] — CPU/AMX compute
//! - [`aruminium`] — Metal GPU
//! - [`rane`] — Apple Neural Engine

pub use acpu;
pub use aruminium;
pub use rane;
pub use unimem;
