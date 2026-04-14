# unimem: How and Why

## The problem in one picture

Every inference framework on macOS does this:

```
                  copy 1        copy 2        copy 3
NVMe → kernel buf → malloc buf → Metal buf → ANE buf → result
```

Four buffers. Three copies. Each copy burns ~5 GB/s of memory bandwidth and adds latency. On a machine with 200 GB/s total bandwidth, three copies of a 7B model's weights eat 15% of available bandwidth before any computation starts.

unimem does this:

```
NVMe DMA → Block → AMX / ANE / CPU → result
```

One buffer. Zero copies. The physical address is known at allocation time. Every hardware unit on the SoC reads from the same DRAM cells.

---

## Why Apple Silicon makes this possible

### Unified memory is real

Apple Silicon M1/M2/M3/M4 use a single DRAM pool shared by CPU, GPU, AMX (matrix coprocessor), ANE (neural engine), and the NVMe controller. There is no "CPU memory" vs "GPU memory" — the DRAM is physically one array, and every unit on the SoC has a path to it through the fabric.

Darwin (macOS) hides this behind virtual memory. `malloc` returns a virtual address. The MMU maps it to some physical page. You never know which one. The page might move (compaction). It might get swapped. Metal creates its own mapping. CoreML creates another. Each mapping is a potential copy.

### What we bypass

1. Virtual memory indirection — we allocate physically contiguous pages and track both addresses
2. Page faults — pages are pinned at allocation, never paged out
3. TLB pressure — Hypervisor stage-2 tables give us a private address space
4. Buffer copies — hardware units get the physical address directly, no staging buffers
5. Kernel memory pressure — pinned pages are invisible to the compressor

---

## The IOKit path

### IOMallocContiguous

IOKit exposes `IOMallocContiguous` — a kernel function that allocates physically contiguous memory and returns both the virtual and physical address. This is the same function IOKit drivers use internally to allocate DMA buffers for PCIe devices, Thunderbolt controllers, and USB host controllers.

From userspace, the path is:
1. Open an IOKit user client to a driver that exposes contiguous allocation
2. Call `IOConnectCallStructMethod` to trigger `IOMallocContiguous` in kernel space
3. Map the result into userspace via `IOConnectMapMemory64`

The result: a virtual pointer the CPU can use, and a physical address that hardware can use. The memory is wired (pinned) — it will never be paged out, never be moved by the compressor, never trigger a page fault.

### IOMemoryDescriptor

For existing virtual allocations where we need to discover the physical address, IOKit provides `IOMemoryDescriptor`. Create a descriptor wrapping a virtual range, prepare it (which wires the pages), then query `getPhysicalSegment` to get physical addresses.

This is the same mechanism that Metal uses internally — `MTLBuffer` backed by an `IOMemoryDescriptor` that gives the GPU a physical address. We cut out the Metal middleman.

---

## The Hypervisor path

### Why stage-2 page tables matter

Apple's Hypervisor.framework exposes ARM64 stage-2 page tables to userspace. Normally only the kernel touches page tables. The Hypervisor framework lets us create a guest physical address space and map our pages into it.

Why this matters for latency:

- Stage-1 (normal) TLB entries are shared with all of macOS — kernel, other apps, daemons. TLB pressure from other processes evicts our entries, causing translation stalls.
- Stage-2 (hypervisor) TLB entries are in a separate namespace. Our mappings don't compete with anyone. Translation latency becomes deterministic.

The cost: one extra level of address translation (~1-2ns). The gain: no TLB shootdowns from other processes, no eviction from system memory pressure, fully predictable access latency.

### The entitlement

`com.apple.security.hypervisor` is a code-signing entitlement. It does not require SIP disabled. It does not require kernel extensions. It does not require root. Any signed binary with this entitlement can create a VM and map pages. Apple designed it for virtualization (Parallels, UTM, Docker) but nothing prevents using it purely for page table control.

---

## The tape: why bump allocation is the right choice

### Inference memory patterns

LLM inference has a very specific memory pattern:

1. Load model weights — allocated once, never freed during inference
2. Allocate KV cache — grows with context, freed when context resets
3. Allocate per-layer activations — allocated and freed every forward pass
4. Allocate output buffer — small, reused

Patterns 2-4 are perfectly served by a bump allocator:
- Allocate forward through the tape during one inference pass
- Reset the cursor to zero when done
- Pages stay pinned — no re-allocation cost

This is why general-purpose allocators (malloc, mimalloc, jemalloc) are the wrong tool. They maintain free lists, size classes, thread caches — all overhead that buys nothing when every allocation is freed at once.

### The atomic trick

The tape cursor is an `AtomicUsize`. Allocation is a single `fetch_add`:

```rust
let offset = self.cursor.fetch_add(aligned_size, Ordering::Relaxed);
if offset + aligned_size > self.capacity {
    return None;
}
```

This is ~4ns on M-series chips. No lock. No syscall. No branching except the bounds check. Multiple threads can allocate simultaneously — `fetch_add` is a single ARM `ldadd` instruction.

### Physical address arithmetic

Because the tape sits on physically contiguous pages, converting a virtual pointer to a physical address is pure arithmetic:

```
pa = base_pa + (ptr - base_va)
```

No page table walk. No IOKit call. No kernel transition. This is what makes the full pipeline zero-copy — every pointer in the tape can be handed to hardware instantly.

---

## The grid: why fixed-size slots work

### Tensor shapes are predictable

In transformer inference, activation tensor shapes are determined by the model architecture:
- Attention QKV: `[batch, heads, seq_len, head_dim]`
- FFN intermediate: `[batch, seq_len, intermediate_dim]`
- Layer output: `[batch, seq_len, hidden_dim]`

These shapes are known at model load time. A grid of fixed-size slots matching these shapes eliminates all allocation logic during inference — acquire a slot (pop from queue), use it, release it (push to queue).

The lock-free queue (crossbeam `SegQueue`) makes acquire/release ~10ns with no contention.

---

## The DMA trait: hardware abstraction

### One interface, three hardware targets

The `DmaTarget` trait abstracts physical buffer submission:

```rust
fn submit(&self, pa: u64, size: usize, op: DmaOp) -> DmaToken;
```

Three implementations (in separate crates, not in unimem):

1. AMX — write the physical address to AMX control registers, trigger matrix op
2. ANE — submit a compiled ANE program with physical buffer addresses in the descriptor
3. NVMe — write a submission queue entry with physical buffer address, ring doorbell

Each hardware unit reads from the same physical DRAM. The `DmaTarget` trait is the interface that makes this explicit — you hand over a physical address and a size, the hardware does the rest.

### Why not async

The DMA interface uses polling (`poll` + `wait`) instead of async/await. Reasons:

1. Hardware completion latency for AMX ops is ~100ns-1us — shorter than an async task switch
2. ANE inference latency is ~1-10ms — short enough to spin-wait in a dedicated thread
3. NVMe completion is interrupt-driven at the kernel level — our poll checks a completion queue
4. async adds allocation (Future state machines), indirection (vtables), and unpredictable scheduling — exactly what this crate exists to eliminate

For pipeline orchestration across multiple hardware units, a higher-level crate (cyb-runtime) can use threads or async. unimem stays synchronous and predictable.

---

## Security model

### What we expose

unimem gives userspace code physical addresses. This is a capability normally restricted to the kernel. The security implications:

- A physical address can be used to program DMA hardware to read/write arbitrary memory
- On Apple Silicon, IOMMU (DART) restricts which physical addresses each hardware unit can access
- We operate within DART constraints — our physical pages are allocated through IOKit, which registers them with DART

### What we don't bypass

- DART/IOMMU — hardware units can only access physical addresses that IOKit has registered
- AMCC (memory controller access control) — prevents access to SecureROM, SEP memory
- PAC (pointer authentication) — our raw pointers don't carry PAC signatures, but we don't need them for data buffers
- PPL (page protection layer) — kernel page tables are still protected

The attack surface is: a bug in unimem could corrupt its own physical buffers (same as any unsafe Rust). It cannot corrupt other processes' memory because DART isolation is hardware-enforced.

---

## Relationship to the cyber stack

unimem is the memory foundation for the cyber hardware pipeline:

```
unimem      — physical allocation, tape, grid, DMA trait
  ↓
rane        — ANE inference engine (uses unimem for buffers)
aruminium   — AMX matrix ops (uses unimem for buffers)
cyb-nvme    — direct NVMe access (uses unimem for DMA buffers)
  ↓
cyb-runtime — orchestrates the full pipeline
  ↓
tru         — runs the tri-kernel on the cybergraph
```

The tri-kernel computation (diffusion + springs + heat) over the cybergraph is the workload. unimem ensures that the data flowing through this computation never gets copied between pipeline stages. Weights load from NVMe directly into pinned memory. AMX does matrix multiplications on the same buffer. ANE runs neural inference on the same buffer. The result is readable by the CPU at the same address. Zero copies, start to finish.

This is how you build a relevance machine that runs at hardware speed.
