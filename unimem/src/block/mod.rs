//! Pinned shared memory block — one allocation visible to every compute unit.
//!
//! Apple: IOSurface (CPU, Metal, AMX, ANE). Android: AHardwareBuffer, the
//! platform's own IOSurface — gralloc-backed, CPU-lockable, and importable
//! into Vulkan through `VK_ANDROID_external_memory_android_hardware_buffer`
//! so the GPU reads the very pages the CPU wrote. Other Unix: memfd + mmap,
//! same CPU semantics with an fd to export.

#[cfg(target_vendor = "apple")]
mod apple;
#[cfg(target_vendor = "apple")]
pub use apple::Block;

#[cfg(target_os = "android")]
mod ahardware;
#[cfg(target_os = "android")]
pub use ahardware::Block;

#[cfg(all(not(target_vendor = "apple"), not(target_os = "android")))]
mod memfd;
#[cfg(all(not(target_vendor = "apple"), not(target_os = "android")))]
pub use memfd::Block;
