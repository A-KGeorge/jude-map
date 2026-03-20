"use strict";

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { DType, SharedTensorSegment } from "../index";

describe("fillAsync()", () => {
  it("resolves with correct data (float32)", async () => {
    const seg = new SharedTensorSegment(4 * 64);
    await seg.fillAsync([64], DType.FLOAT32, 9.5);

    const out = seg.readCopy()!;
    assert.deepEqual(out.shape, [64]);
    const data = out.data as Float32Array;
    for (let i = 0; i < 64; i++) assert.ok(Math.abs(data[i] - 9.5) < 1e-5);

    seg.destroy();
  });

  it("event loop remains alive during fill (timer fires before resolve)", async () => {
    const MB = 1024 * 1024;
    const seg = new SharedTensorSegment(256 * MB);

    let timerFired = false;
    // This timer must fire BEFORE fillAsync resolves if the event loop is free.
    const t = setTimeout(() => {
      timerFired = true;
    }, 0);

    await seg.fillAsync([(256 * MB) / 4], DType.FLOAT32, 1.0);

    // Timer should have fired during the fill since the event loop was free.
    assert.ok(
      timerFired,
      "event loop was blocked — timer did not fire during fillAsync",
    );

    clearTimeout(t);
    seg.destroy();
  });

  it("readWait resolves after fillAsync commit", async () => {
    const seg = new SharedTensorSegment(4 * 16);

    // Park a reader before the fill starts.
    const reader = seg.readWait();

    // Fill without await — let event loop interleave.
    const writer = seg.fillAsync([16], DType.FLOAT32, 55.0);

    const [out] = await Promise.all([reader, writer]);

    assert.notEqual(out, null);
    assert.deepEqual(out!.shape, [16]);
    assert.ok(Math.abs((out!.data as Float32Array)[0] - 55.0) < 1e-5);

    seg.destroy();
  });

  it("multiple concurrent fillAsync calls serialise via seqlock", async () => {
    // Two fillAsync calls in flight simultaneously — the seqlock ensures
    // readers never see a mix of values from both fills.
    const seg = new SharedTensorSegment(4 * 256);

    const p1 = seg.fillAsync([256], DType.FLOAT32, 1.0);
    const p2 = seg.fillAsync([256], DType.FLOAT32, 2.0);

    await Promise.all([p1, p2]);

    const out = seg.readCopy()!;
    const data = out.data as Float32Array;
    const first = data[0];

    // All elements must match — no torn state between the two fills.
    for (let i = 0; i < 256; i++)
      assert.equal(data[i], first, `torn read at index ${i}`);

    seg.destroy();
  });

  it("fillAsync after destroy rejects", async () => {
    const seg = new SharedTensorSegment(64);
    seg.destroy();
    await assert.rejects(() => seg.fillAsync([16], DType.FLOAT32, 0.0));
  });

  it("fillAsync overflow rejects", async () => {
    const seg = new SharedTensorSegment(100);
    await assert.rejects(
      () => seg.fillAsync([1000], DType.FLOAT32, 0.0),
      /exceeds/,
    );
    seg.destroy();
  });

  it("destroyAsync resolves quickly and object is logically destroyed", async () => {
    const MB = 1024 * 1024;
    const seg = new SharedTensorSegment(512 * MB);
    seg.fill([32], DType.FLOAT32, 1.0);

    const t0 = performance.now();
    await seg.destroyAsync();
    const dt = performance.now() - t0;

    // Must complete as low-latency logical teardown, not full unmap wait.
    assert.ok(dt < 100, `destroyAsync took ${dt.toFixed(2)}ms`);
    assert.equal(seg.read(), null);
    assert.throws(() => seg.write([1], DType.FLOAT32, new Float32Array([1])));
  });

  it("destroyAsyncWait resolves after full unmap completion", async () => {
    const MB = 1024 * 1024;
    const seg = new SharedTensorSegment(128 * MB);
    seg.fill([32], DType.FLOAT32, 1.0);

    await seg.destroyAsyncWait();
    assert.equal(seg.read(), null);
    assert.throws(() => seg.write([1], DType.FLOAT32, new Float32Array([1])));
  });
});
