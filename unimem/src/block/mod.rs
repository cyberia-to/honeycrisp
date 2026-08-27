//! Pinned shared memory block — one allocation visible to every compute unit.
//!
//! Apple: IOSurface (CPU, Metal, AMX, ANE). Android/Linux: memfd + mmap — the
//! CPU view is identical, and the fd is the handle a Vulkan device imports
//! (`VK_KHR_external_memory_fd` / host-pointer import) so the same block can
//! reach the GPU without a copy where the driver allows it.

#[cfg(target_vendor = "apple")]
mod apple;
#[cfg(target_vendor = "apple")]
pub use apple::Block;

#[cfg(not(target_vendor = "apple"))]
mod memfd;
#[cfg(not(target_vendor = "apple"))]
pub use memfd::Block;
