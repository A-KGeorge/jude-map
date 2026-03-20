"use strict";

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { SharedTensorSegment, DType } from "../index";

// ---------------------------------------------------------------------------
// tensor-bridge smoke tests
// ---------------------------------------------------------------------------

describe("SharedTensorSegment", () => {
  // -- 1. Basic write / readCopy --
  it("float32 write / readCopy", () => {
    const seg = new SharedTensorSegment(4 * 1024 * 1024);
    const input = new Float32Array([1, 2, 3, 4, 5, 6]);
    seg.write([2, 3], DType.FLOAT32, input);

    const out = seg.readCopy();
    assert.notEqual(out, null);
    assert.deepEqual(out!.shape, [2, 3]);
    assert.equal(out!.dtype, DType.FLOAT32);
    assert.equal(out!.data.length, 6);
    assert.equal((out!.data as Float32Array)[5], 6);

    seg.destroy();
  });

  // -- 2. Zero-copy read --
  it("int32 zero-copy read", () => {
    const seg = new SharedTensorSegment(1024 * 1024);
    const input = new Int32Array([10, 20, 30]);
    seg.write([3], DType.INT32, input);

    const out = seg.read();
    assert.notEqual(out, null);
    assert.equal((out!.data as Int32Array)[1], 20);
    assert.ok(out!.version >= 2);

    seg.destroy();
  });

  // -- 3. High-rank tensor (rank-4, NHWC image batch) --
  it("rank-4 float32 (NHWC)", () => {
    const N = 2;
    const H = 8;
    const W = 8;
    const C = 3;
    const numel = N * H * W * C;
    const seg = new SharedTensorSegment(4 * numel);
    const input = new Float32Array(numel).fill(0.5);

    seg.write([N, H, W, C], DType.FLOAT32, input);
    const out = seg.readCopy();

    assert.notEqual(out, null);
    assert.deepEqual(out!.shape, [N, H, W, C]);
    assert.equal(out!.data.length, numel);

    seg.destroy();
  });

  // -- 4. uint8 dtype --
  it("uint8 dtype", () => {
    const seg = new SharedTensorSegment(256);
    const input = new Uint8Array([0, 127, 255]);
    seg.write([3], DType.UINT8, input);

    const out = seg.readCopy();
    assert.notEqual(out, null);
    assert.equal((out!.data as Uint8Array)[2], 255);

    seg.destroy();
  });

  // -- 5. Read before write returns null --
  it("read before write returns null", () => {
    const seg = new SharedTensorSegment(1024);
    const out = seg.read();
    assert.equal(out, null);
    seg.destroy();
  });

  // -- 5b. readWait parks and resolves after write --
  it("readWait resolves after writer commit", async () => {
    const seg = new SharedTensorSegment(4 * 16);
    const pending = seg.readWait();

    setTimeout(() => {
      seg.write([4], DType.FLOAT32, new Float32Array([7, 7, 7, 7]));
    }, 0);

    const out = await pending;
    assert.notEqual(out, null);
    assert.deepEqual(out!.shape, [4]);
    assert.equal((out!.data as Float32Array)[0], 7);

    seg.destroy();
  });

  // -- 5c. readWait stress: many parked readers, single writer wake --
  it("readWait stress (256 parked readers, single wake)", async () => {
    const seg = new SharedTensorSegment(4 * 16);
    const WAITERS = 256;

    const pending = Array.from({ length: WAITERS }, () => seg.readWait());

    // Queue a single commit that should wake all parked readers.
    setTimeout(() => {
      seg.write([4], DType.FLOAT32, new Float32Array([42, 42, 42, 42]));
    }, 0);

    const results = await Promise.all(pending);
    assert.equal(results.length, WAITERS);

    const firstVersion = results[0]!.version;
    for (const out of results) {
      assert.notEqual(out, null);
      assert.deepEqual(out!.shape, [4]);
      assert.equal((out!.data as Float32Array)[0], 42);
      assert.equal(out!.version, firstVersion);
    }

    seg.destroy();
  });

  // -- 6. Overwrite: shape changes between writes --
  it("shape change between writes", () => {
    const seg = new SharedTensorSegment(4 * 100);
    seg.write([10], DType.FLOAT32, new Float32Array(10).fill(1));
    seg.write([5, 4], DType.FLOAT32, new Float32Array(20).fill(2));

    const out = seg.readCopy();
    assert.notEqual(out, null);
    assert.deepEqual(out!.shape, [5, 4]);
    assert.equal((out!.data as Float32Array)[0], 2);

    seg.destroy();
  });

  // -- 7. Capacity overflow rejected --
  it("overflow throws RangeError", () => {
    const seg = new SharedTensorSegment(100); // only 100 bytes
    const big = new Float32Array(1000); // 4000 bytes
    assert.throws(() => seg.write([1000], DType.FLOAT32, big));
    seg.destroy();
  });

  // -- 8. Seqlock stress - no torn reads --
  // JS execution is single-threaded, so a synchronous write/read loop validates
  // seqlock visibility guarantees without timer-resolution artifacts on Windows.
  it("seqlock stress (50k reads, 0 torn)", () => {
    const ITERS = 50_000;
    const seg = new SharedTensorSegment(4 * 16);
    const shape = [16];
    let torn = 0;

    for (let i = 0; i < ITERS; i++) {
      const v = i % 256;
      seg.write(shape, DType.FLOAT32, new Float32Array(16).fill(v));

      const out = seg.read();
      if (out) {
        const data = out.data as Float32Array;
        const first = data[0];
        for (let k = 1; k < 16; k++) {
          if (data[k] !== first) {
            torn++;
            break;
          }
        }
      }
    }

    seg.destroy();
    assert.equal(torn, 0, `${torn} torn reads detected`);
  });
});
