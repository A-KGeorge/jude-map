"use strict";

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { DType, SharedTensorSegment } from "../index";

// Note: DATA_OFFSET alignment (256 bytes for CUDA DMA) is enforced by a
// static_assert in segment.h and is not observable from JS. These tests
// cover the JS-visible pin/unpin state machine only.

describe("CUDA pinning", () => {
  it("pin returns boolean and updates isPinned consistently", () => {
    const seg = new SharedTensorSegment(1024 * 1024);

    assert.equal(seg.isPinned, false);

    const pinned = seg.pin();
    assert.equal(typeof pinned, "boolean");
    assert.equal(seg.isPinned, pinned);

    seg.unpin();
    assert.equal(seg.isPinned, false);

    seg.destroy();
  });

  it("pin is idempotent and unpin is safe when already unpinned", () => {
    const seg = new SharedTensorSegment(1024 * 1024);

    seg.unpin();
    assert.equal(seg.isPinned, false);

    const first = seg.pin();
    const second = seg.pin();

    assert.equal(typeof first, "boolean");
    assert.equal(typeof second, "boolean");

    // If pinning is allowed, both calls should be true. If disallowed, both false.
    assert.equal(second, first);

    seg.unpin();
    seg.unpin();
    assert.equal(seg.isPinned, false);

    seg.destroy();
  });

  it("pinning does not break write/read flows", () => {
    const seg = new SharedTensorSegment(4 * 32);

    seg.pin();

    const input = new Float32Array(32).fill(3.25);
    seg.write([32], DType.FLOAT32, input);

    const out = seg.readCopy();
    assert.notEqual(out, null);
    assert.equal(out!.data.length, 32);
    assert.equal((out!.data as Float32Array)[0], 3.25);
    assert.equal((out!.data as Float32Array)[31], 3.25);

    seg.unpin();
    seg.destroy();
  });

  it("pin after destroy throws", () => {
    const seg = new SharedTensorSegment(4096);
    seg.destroy();
    assert.throws(() => seg.pin());
  });

  it("unpin after destroy is a safe no-op", () => {
    const seg = new SharedTensorSegment(4096);
    seg.pin();
    seg.destroy();
    assert.doesNotThrow(() => seg.unpin()); // must not crash
  });
});
