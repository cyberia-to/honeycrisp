# examples/

runnable demos. each one is self-contained and shows a real GPU workflow.

```bash
cargo run --example vecadd
cargo run --example matmul
```

| file | what it does |
|------|-------------|
| `vecadd.rs` | vector addition on GPU — simplest possible compute dispatch |
| `matmul.rs` | matrix multiplication — tiled compute with shared memory |
