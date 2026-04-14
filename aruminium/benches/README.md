# benches/

separate crate (`metal-benches`) with heavy dependencies isolated from the core library.

```bash
cargo run --release -p metal-benches --bin bench
cargo run --release -p metal-benches --bin compare
```

| file | purpose |
|------|---------|
| `Cargo.toml` | crate manifest — heavy deps live here, not in root |
| `bench.rs` | main benchmark suite — latency, throughput, dispatch overhead |
| `compare.rs` | head-to-head comparison with objc2-metal |
| `objc2.rs` | objc2-metal baseline implementations |
| `shaders.rs` | MSL shader sources shared between benchmarks |
| `aruminium.rs` | aruminium benchmark implementations |
