"use strict";

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { DType, SharedTensorSegment } from "../index.js";

describe("Segment Modes: seqlock, rcu, ring", () => {
  it("seqlock mode works", () => {
    const seg = SharedTensorSegment.createShared(1024);
    const input = new Float32Array([1.5, 2.5, 3.5]);
    seg.write([3], DType.FLOAT32, input);

    const out = seg.read();
    assert.ok(out);
    assert.deepEqual(out.shape, [3]);
    assert.equal(out.data[0], 1.5);
    assert.equal(out.data[1], 2.5);
    assert.equal(out.data[2], 3.5);

    seg.destroy();
  });

  it("rcu mode works", () => {
    const seg = SharedTensorSegment.createSharedRCU(1024);
    const input = new Int32Array([10, 20, 30, 40]);
    seg.write([4], DType.INT32, input);

    const out = seg.read();
    assert.ok(out);
    assert.deepEqual(out.shape, [4]);
    assert.equal(out.data[0], 10);
    assert.equal(out.data[1], 20);
    assert.equal(out.data[2], 30);
    assert.equal(out.data[3], 40);

    seg.destroy();
  });

  it("ring mode works", () => {
    // ring capacity must be a power of 2
    const seg = SharedTensorSegment.createRing(4, 1024);

    const input1 = new Float32Array([1.5]);
    const input2 = new Float32Array([2.5]);
    const input3 = new Float32Array([3.5]);

    seg.write([1], DType.FLOAT32, input1);
    seg.write([1], DType.FLOAT32, input2);
    seg.write([1], DType.FLOAT32, input3);

    // Read should pop items in order (FIFO)
    let out = seg.read();
    assert.ok(out);
    assert.equal(out.data[0], 1.5);

    out = seg.read();
    assert.ok(out);
    assert.equal(out.data[0], 2.5);

    out = seg.read();
    assert.ok(out);
    assert.equal(out.data[0], 3.5);

    // Ring should be empty now
    out = seg.read();
    assert.equal(out, null);

    seg.destroy();
  });
});
