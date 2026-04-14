# src/ffi/

raw FFI layer. direct bindings to Metal.framework, CoreFoundation and libobjc.

no ObjC, no Swift, no headers — everything goes through `objc_msgSend`.

| file | purpose |
|------|---------|
| `mod.rs` | types, constants, extern links, CFString helpers |
| `selectors.rs` | cached ObjC selectors and class lookups |
| `trampoline.rs` | typed `objc_msgSend` trampolines for each Metal method |
