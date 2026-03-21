# jude-map

_Named after St. Jude, patron of hopeless causes — like getting Node.js to beat Python at ML inference._

A native Node.js addon providing mmap-backed shared tensor memory with seqlock consistency, zero-copy reads, and a libuv-integrated async wait mechanism for GPU-ready inference pipelines.

---

## Why

Every high-throughput ML inference pipeline in Node.js eventually hits the same wall: moving tensor data between threads means either serializing through `postMessage` (structured clone, pays a copy tax on every frame) or fighting the V8 heap ceiling with `SharedArrayBuffer` (capped at 4GB, misaligned for CUDA DMA).

`jude-map` sidesteps both. It maps tensor memory directly via `mmap`/`MapViewOfFile`, outside the V8 heap, with a seqlock protecting consistency across writer and reader. For GPU workflows, `pin()` page-locks the mapping so CUDA can DMA directly from host memory without a staging copy. The data pointer is 256-byte aligned at `DATA_OFFSET`, satisfying `cudaMemcpyAsync`'s alignment requirement out of the box.

---

## Features

- **mmap-backed segments** outside the V8 heap — no 4GB ceiling, no GC pressure
- **Seqlock consistency** — writer-priority, reader-optimistic, zero kernel involvement in the fast path
- **Zero-copy reads** via external `ArrayBuffer` backed directly by the mapped region
- **Safe copy reads** for data that needs to outlive the next write
- **`readWait()` / `readCopyWait()`** — spin N times, then park on `uv_async`; the writer's commit wakes all parked readers via a single coalesced `uv_async_send`
- **`pin()` / `unpin()`** — page-locks the mapping for CUDA H2D zero-copy paths
- **Cross-platform** — Linux, macOS, Windows (MSVC)
- **Full TypeScript API** with `DType` enum, `TensorResult` interface, and declaration maps

---

## Requirements

- Node.js >= 18
- npm >= 11.5.1
- For building from source:
  - **Linux/macOS**: GCC or Clang with C++17 support, Python 3
  - **Windows**: Visual Studio 2019+ with "Desktop development with C++" workload, Python 3

---

## Installation

```bash
npm install jude-map
```

ARM architectures compile locally on install. x64 uses prebuilt binaries when available — no toolchain required.

---

## Quick Start

```ts
import { SharedTensorSegment, DType } from "jude-map";

const seg = new SharedTensorSegment(4 * 1024 * 1024); // 4 MB capacity

// Write a [2, 3] float32 tensor
const input = new Float32Array([1, 2, 3, 4, 5, 6]);
seg.write([2, 3], DType.FLOAT32, input);

// Zero-copy read — view directly into mmap memory
// Valid until the next write(). Do not retain across async boundaries.
const r = seg.read();
if (r) {
  console.log(r.shape); // [2, 3]
  console.log(r.dtype); // 0 (DType.FLOAT32)
  console.log(r.version); // seqlock sequence number at time of read
  console.log((r.data as Float32Array)[5]); // 6
}

// Safe copy read — owns its buffer, safe to retain and pass anywhere
const copy = seg.readCopy();

seg.destroy();
```

---

## Async reads

`readWait()` is the primary API for consumer threads that need to park until data is ready. Under the hood, it spins up to 16 times on the seqlock before registering a pending read. When the writer calls `write()`, it calls `uv_async_send()` after the seqlock commit. libuv coalesces concurrent sends into a single callback, which wakes all parked readers at once.

```ts
const seg = new SharedTensorSegment(4 * 1024 * 1024);

// Park until a writer commits
const result = await seg.readWait();
if (result) {
  console.log(result.shape);
  console.log((result.data as Float32Array)[0]);
}

// Or park for a safe copy
const copy = await seg.readCopyWait();
```

This means there is no polling, no `setInterval`, and no blocked event loop. Readers pay zero CPU while waiting. The coalescing property of `uv_async_send` means 256 parked readers are woken by a single writer commit with one callback delivery, not 256.

---

## GPU / CUDA pinning

Page-locking the mapping lets CUDA initiate H2D DMA directly from host memory, skipping the internal staging buffer that `cudaMemcpy` would otherwise allocate. Call `pin()` before handing the data pointer to `TF_NewTensor`.

```ts
const seg = new SharedTensorSegment(1024 * 1024 * 64); // 64 MB

const pinned = seg.pin();
if (!pinned) {
  // Pinning failed — insufficient RLIMIT_MEMLOCK (Linux) or working set quota (Windows).
  // The segment still works correctly for CPU-only inference; pin() failing is non-fatal.
  console.warn("Page pinning unavailable, falling back to staged DMA");
}

// Write pre-processed tensor data (e.g. from dspx)
seg.write([1, 224, 224, 3], DType.FLOAT32, preprocessedBuffer);

// Hand data_ptr directly to TF_NewTensor with TF_DONT_DEALLOCATE_CONTENTS.
// The gpu_private host threads can now DMA without a staging copy.
const dataPtr = /* seg.dataPtr — see C++ shim notes below */;

seg.unpin(); // Release when inference is done
seg.destroy();
```

The tensor data begins at `DATA_OFFSET = 256` bytes into the mapped region. This offset is enforced by a `static_assert` in `segment.h` and satisfies CUDA's async DMA alignment requirement.

On Linux, if `pin()` returns `false`:

```bash
ulimit -l unlimited
# or programmatically: setrlimit(RLIMIT_MEMLOCK, { rlim_cur: RLIM_INFINITY, ... })
```

On Windows, the default per-process `VirtualLock` quota is 64MB. Raise it via `SetProcessWorkingSetSizeEx` for large models.

---

## API

### `new SharedTensorSegment(maxBytes: number)`

Allocates an mmap-backed segment with `maxBytes` bytes of tensor capacity, plus a 256-byte internal header. The total mapped region is rounded up to the nearest page boundary.

### `write(shape: number[], dtype: DType, buffer: ArrayBuffer | TypedArray): void`

Commits a tensor under the seqlock. Shape rank is capped at 8 dimensions. Throws `RangeError` if `buffer.byteLength` exceeds `byteCapacity`. After commit, wakes any parked `readWait()` / `readCopyWait()` callers via `uv_async_send`.

### `read(): TensorResult | null`

Zero-copy read. Returns an external `ArrayBuffer` view directly into mmap memory. Valid until the next `write()` or `destroy()`. Returns `null` if no data has been written yet.

**Do not retain the returned `data` view across async boundaries without calling `readCopy()` instead.**

### `readCopy(): TensorResult | null`

Copies tensor data into a fresh `Buffer` before returning. Safe to retain indefinitely and pass across worker boundaries.

### `readWait(): Promise<TensorResult | null>`

Spins up to 16 times, then parks until a writer commit wakes it. Resolves with a zero-copy view. Rejects if `destroy()` is called while parked.

### `readCopyWait(): Promise<TensorResult | null>`

Same as `readWait()` but resolves with a copied buffer.

### `pin(): boolean`

Page-locks the entire mapped region. Returns `true` on success, `false` if the OS denies the lock (non-fatal). Idempotent — calling `pin()` on an already-pinned segment returns `true` immediately.

### `unpin(): void`

Releases the page lock. Safe to call when not pinned or after `destroy()`.

### `destroy(): void`

Unmaps the region and rejects all parked `readWait()` promises. Subsequent calls to `write()` or `pin()` will throw.

### `byteCapacity: number`

The usable tensor capacity in bytes (excludes the internal header).

### `isPinned: boolean`

Whether the mapping is currently page-locked.

---

## `DType`

```ts
enum DType {
  FLOAT32 = 0,
  FLOAT64 = 1,
  INT32 = 2,
  INT64 = 3,
  UINT8 = 4,
  INT8 = 5,
  UINT16 = 6,
  INT16 = 7,
  BOOL = 8,
}
```

The numeric enum is passed directly over the N-API bridge, avoiding string parsing on the hot path.

---

## `TensorResult`

```ts
interface TensorResult {
  shape: number[]; // dimension sizes
  dtype: DType; // element type
  data: TypedArray; // typed view into the buffer
  version: number; // seqlock sequence number at time of read
}
```

`version` increments by 2 on each successful write (seqlock counter is even when stable). You can use it to detect stale reads in a pipeline.

---

## Segment memory layout

```
Byte offset 0:
  [0..63]    Seqlock          — cache line 0 (sequence counter, alignas(64))
  [64..191]  TensorMeta       — cache lines 1-2 (ndim, dtype, shape[8], byte_length, pad)
  [192..255] _gpu_pad         — 64 bytes, pads header to 256 for CUDA DMA alignment
Byte offset 256:
  [256..]    Tensor data      — your bytes, 256-byte aligned
```

Each section occupies distinct cache lines, preventing false sharing between the seqlock counter, metadata reads, and tensor data writes.

---

## Seqlock failure modes

The reader spins up to `READ_SPIN_LIMIT = 16` times on a torn read (writer mid-commit). In practice, a commit is a `memcpy` plus two atomic increments — the spin resolves in nanoseconds. If the segment has no committed data yet, `read()` returns `null` and `readWait()` parks until the first `write()`.

---

## Building from source

```bash
git clone https://github.com/A-KGeorge/jude
cd jude-map
npm install
npm run build
npm test
```

Individual steps:

```bash
npm run build:ts      # TypeScript → dist/
npm run build:native  # C++ → build/Release/jude-map.node
npm test              # node --import tsx --test src/ts/__tests__/*.test.ts
```

Generating prebuilds for distribution:

```bash
npm run prebuildify
```

This produces binaries for Node 18, 20, 22, and 24 under `prebuilds/`. Commit this directory — it is explicitly not gitignored.

---

## License

Apache-2.0
