"use strict";

import nodeGypBuild from "node-gyp-build";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

type NativeSharedTensorCtor = new (maxBytes: number) => any;

let NativeSharedTensor: NativeSharedTensorCtor;
try {
  const addon = nodeGypBuild(join(__dirname, "..")) as any;
  NativeSharedTensor = addon.SharedTensor ?? addon;
} catch (e) {
  try {
    const addon = nodeGypBuild(join(__dirname, "..", "..")) as any;
    NativeSharedTensor = addon.SharedTensor ?? addon;
  } catch (err: any) {
    console.error("Failed to load native SharedTensor module.");
    console.error("Tried using both relative paths.");
    console.error(
      "Attempt 1 error (installed path ../):",
      (e as Error).message,
    );
    console.error("Attempt 2 error (local path ../../):", err.message);
    throw new Error(
      `Could not load native module. Is the build complete? Search paths tried: ${join(
        __dirname,
        "..",
      )} and ${join(__dirname, "..", "..")}`,
    );
  }
}

if (typeof NativeSharedTensor !== "function") {
  throw new Error(
    "Native module loaded, but SharedTensor constructor was not found.",
  );
}

/**
 * Numeric DType enum to match the C++ DType enum class exactly.
 * Passing integers over the N-API bridge is significantly faster than strings.
 */
export enum DType {
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

type TypedArray =
  | Float32Array
  | Float64Array
  | Int32Array
  | BigInt64Array
  | Uint8Array
  | Int8Array
  | Uint16Array
  | Int16Array;

const TYPED_ARRAY_CTORS: Record<number, any> = {
  [DType.FLOAT32]: Float32Array,
  [DType.FLOAT64]: Float64Array,
  [DType.INT32]: Int32Array,
  [DType.INT64]: BigInt64Array,
  [DType.UINT8]: Uint8Array,
  [DType.INT8]: Int8Array,
  [DType.UINT16]: Uint16Array,
  [DType.INT16]: Int16Array,
  [DType.BOOL]: Uint8Array,
};

export interface TensorResult {
  shape: number[];
  dtype: DType;
  data: TypedArray;
  version: number;
}

function wrap(result: any): TensorResult | null {
  if (!result) return null;

  const Ctor = TYPED_ARRAY_CTORS[result.dtype];
  if (!Ctor) throw new Error(`Unsupported DType ID: ${result.dtype}`);

  const buf = result.buffer;
  let ab: ArrayBufferLike;
  let byteOffset = 0;

  if (ArrayBuffer.isView(buf)) {
    ab = buf.buffer;
    byteOffset = buf.byteOffset;
  } else {
    ab = buf;
  }

  return {
    shape: result.shape,
    dtype: result.dtype,
    version: result.version,
    data: new Ctor(
      ab,
      byteOffset,
      result.buffer.byteLength / Ctor.BYTES_PER_ELEMENT,
    ),
  };
}

export class SharedTensorSegment {
  private _native: any;

  constructor(maxBytes: number) {
    this._native = new NativeSharedTensor(maxBytes);
  }

  get byteCapacity(): number {
    return this._native.byteCapacity;
  }

  get isPinned(): boolean {
    return Boolean(this._native.isPinned);
  }

  /**
   * Write a tensor from an existing JS buffer.
   * Subject to V8's ~2 GB TypedArray ceiling — use fill() for large tensors.
   */
  write(shape: number[], dtype: DType, buffer: ArrayBuffer | TypedArray): void {
    this._native.write(shape, dtype, buffer);
  }

  /**
   * Fill every element with a scalar value, entirely in C++.
   * Synchronous — blocks the calling thread until complete.
   *
   * Use on Worker threads where blocking the isolated event loop is fine.
   * Use fillAsync() on the main thread to keep the event loop alive.
   */
  fill(shape: number[], dtype: DType, value: number): void {
    this._native.fill(shape, dtype, value);
  }

  /**
   * Non-blocking variant of fill() for the main event loop thread.
   *
   * Pushes the C++ fill onto libuv's thread pool. The event loop stays free
   * to process I/O, timers, and readWait() callbacks while the fill runs.
   * Resolves when the seqlock commit is complete and readWait() callers
   * have been woken.
   *
   * @example
   * // Main thread — event loop stays alive during 10 GB fill
   * await seg.fillAsync([N_ELEMS], DType.FLOAT32, 0.0);
   */
  async fillAsync(shape: number[], dtype: DType, value: number): Promise<void> {
    return this._native.fillAsync(shape, dtype, value);
  }

  /**
   * Zero-copy read. Returns a view directly into shared memory.
   * Valid until the next write() or fill() or destroy().
   */
  read(): TensorResult | null {
    return wrap(this._native.read());
  }

  /**
   * Safe read. Copies data into a fresh buffer.
   */
  readCopy(): TensorResult | null {
    return wrap(this._native.readCopy());
  }

  /**
   * Promise-based zero-copy read.
   * Native code spins briefly then parks until a writer commit wakes it.
   */
  async readWait(): Promise<TensorResult | null> {
    return wrap(await this._native.readWait());
  }

  /**
   * Promise-based copy read with spin-then-park behaviour.
   */
  async readCopyWait(): Promise<TensorResult | null> {
    return wrap(await this._native.readCopyWait());
  }

  /**
   * Page-lock the mapped region for CUDA H2D zero-copy paths.
   * Returns true on success, false if the OS denies the lock (non-fatal).
   */
  pin(): boolean {
    return Boolean(this._native.pin());
  }

  unpin(): void {
    this._native.unpin();
  }

  /**
   * Synchronous destroy. Blocks the event loop during munmap.
   * Use on Worker threads. On the main thread prefer destroyAsync().
   */
  destroy(): void {
    this._native.destroy();
  }

  /**
   * Low-latency destroy for the main event loop thread.
   *
   * Resolves immediately after logical teardown (pointer nulled, pending
   * waiters rejected, future ops throw "Destroyed"). Physical munmap runs in
   * the background on the libuv worker pool.
   *
   * @example
   * await seg.destroyAsync(); // returns quickly after logical teardown
   */
  async destroyAsync(): Promise<void> {
    return this._native.destroyAsync();
  }

  /**
   * Strict async destroy: resolves only after full OS unmap completes.
   *
   * Use this when you need an explicit synchronization point for resource
   * release (e.g. deterministic teardown sequencing in benchmarks/tests).
   */
  async destroyAsyncWait(): Promise<void> {
    return this._native.destroyAsyncWait();
  }
}
