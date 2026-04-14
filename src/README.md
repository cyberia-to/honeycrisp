# src/

core library. zero external dependencies — only macOS system frameworks via FFI.

| file | purpose |
|------|---------|
| `lib.rs` | public API surface: re-exports, error types |
| `device.rs` | `Gpu` — GPU discovery, properties, factory methods |
| `buffer.rs` | `Buffer` — zero-copy shared CPU↔GPU memory |
| `command.rs` | `Queue`, `Commands` |
| `encoder.rs` | `Encoder`, `Copier` |
| `dispatch.rs` | `Dispatch`, `Batch`, `GpuFuture` |
| `shader.rs` | `ShaderLib`, `Shader` — MSL compilation |
| `pipeline.rs` | `Pipeline` |
| `texture.rs` | `Texture` |
| `sync.rs` | `Fence`, `Event`, `SharedEvent` |
| `fp16.rs` | fp16↔f32 conversion (NEON + software fallback) |
| `probe.rs` | `metal_probe` binary — 5-level GPU capability probe |
| `tests.rs` | unit tests |
| `ffi/` | raw FFI bindings — see [ffi/README.md](ffi/README.md) |
