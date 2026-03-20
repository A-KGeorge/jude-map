"use strict";

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { DType, SharedTensorSegment } from "../index";

describe("fill()", () => {
  // -- 1. Basic scalar fill, float32 --
  it("fills float32 tensor with scalar", () => {
    const seg = new SharedTensorSegment(4 * 16);
    seg.fill([16], DType.FLOAT32, 3.14);

    const out = seg.readCopy();
    assert.notEqual(out, null);
    assert.deepEqual(out!.shape, [16]);
    assert.equal(out!.dtype, DType.FLOAT32);
    assert.equal(out!.data.length, 16);

    const data = out!.data as Float32Array;
    for (let i = 0; i < 16; i++) {
      // float32 precision — compare within epsilon
      assert.ok(Math.abs(data[i] - 3.14) < 1e-5, `data[${i}] = ${data[i]}`);
    }

    seg.destroy();
  });

  // -- 2. fill matches write output for float64 --
  it("fill and write produce identical float64 tensors", () => {
    const N = 128;
    const VALUE = 2.718281828;

    const segFill = new SharedTensorSegment(8 * N);
    segFill.fill([N], DType.FLOAT64, VALUE);
    const fromFill = segFill.readCopy()!;
    segFill.destroy();

    const segWrite = new SharedTensorSegment(8 * N);
    const buf = new Float64Array(N).fill(VALUE);
    segWrite.write([N], DType.FLOAT64, buf);
    const fromWrite = segWrite.readCopy()!;
    segWrite.destroy();

    assert.deepEqual(fromFill.shape, fromWrite.shape);
    assert.equal(fromFill.dtype, fromWrite.dtype);
    const fd = fromFill.data as Float64Array;
    const wd = fromWrite.data as Float64Array;
    for (let i = 0; i < N; i++) {
      assert.equal(fd[i], wd[i], `mismatch at index ${i}`);
    }
  });

  // -- 3. All integer dtypes --
  it("fills int32 tensor correctly", () => {
    const seg = new SharedTensorSegment(4 * 8);
    seg.fill([8], DType.INT32, -42);
    const out = seg.readCopy()!;
    const data = out.data as Int32Array;
    for (let i = 0; i < 8; i++) assert.equal(data[i], -42);
    seg.destroy();
  });

  it("fills uint8 tensor correctly", () => {
    const seg = new SharedTensorSegment(64);
    seg.fill([64], DType.UINT8, 255);
    const out = seg.readCopy()!;
    const data = out.data as Uint8Array;
    for (let i = 0; i < 64; i++) assert.equal(data[i], 255);
    seg.destroy();
  });

  it("fills bool tensor with 0/1", () => {
    const seg = new SharedTensorSegment(16);
    seg.fill([16], DType.BOOL, 1);
    const out = seg.readCopy()!;
    const data = out.data as Uint8Array;
    for (let i = 0; i < 16; i++) assert.equal(data[i], 1);
    seg.destroy();
  });

  it("fills bool tensor with false (0)", () => {
    const seg = new SharedTensorSegment(16);
    seg.fill([16], DType.BOOL, 0);
    const out = seg.readCopy()!;
    const data = out.data as Uint8Array;
    for (let i = 0; i < 16; i++) assert.equal(data[i], 0);
    seg.destroy();
  });

  // -- 4. Rank-4 NHWC shape --
  it("fills rank-4 tensor and reports correct shape", () => {
    const [N, H, W, C] = [2, 8, 8, 3];
    const numel = N * H * W * C;
    const seg = new SharedTensorSegment(4 * numel);

    seg.fill([N, H, W, C], DType.FLOAT32, 1.0);
    const out = seg.readCopy()!;

    assert.deepEqual(out.shape, [N, H, W, C]);
    assert.equal(out.data.length, numel);
    seg.destroy();
  });

  // -- 5. fill then write replaces content --
  it("write after fill replaces tensor content", () => {
    const seg = new SharedTensorSegment(4 * 8);
    seg.fill([8], DType.FLOAT32, 99.0);
    seg.write([8], DType.FLOAT32, new Float32Array(8).fill(1.0));

    const out = seg.readCopy()!;
    const data = out.data as Float32Array;
    for (let i = 0; i < 8; i++) assert.equal(data[i], 1.0);
    seg.destroy();
  });

  // -- 6. Overflow rejected --
  it("fill throws RangeError when shape exceeds capacity", () => {
    const seg = new SharedTensorSegment(100);
    // 1000 float32 elements = 4000 bytes > 100
    assert.throws(() => seg.fill([1000], DType.FLOAT32, 0.0));
    seg.destroy();
  });

  // -- 7. fill after destroy throws --
  it("fill after destroy throws", () => {
    const seg = new SharedTensorSegment(64);
    seg.destroy();
    assert.throws(() => seg.fill([16], DType.FLOAT32, 0.0));
  });

  // -- 8. readWait resolves after fill --
  it("readWait resolves after fill commit", async () => {
    const seg = new SharedTensorSegment(4 * 16);
    const pending = seg.readWait();

    setTimeout(() => {
      seg.fill([16], DType.FLOAT32, 7.0);
    }, 0);

    const out = await pending;
    assert.notEqual(out, null);
    assert.deepEqual(out!.shape, [16]);
    assert.equal((out!.data as Float32Array)[0], 7.0);
    seg.destroy();
  });

  // -- 9. Large tensor: fill 512 MB without V8 buffer allocation --
  //
  // This is the primary motivation for fill() — bypassing the V8 TypedArray
  // ceiling. We use 512 MB here (CI-friendly) but the same path handles 10+ GB.
  // Spot-check first and last elements only — reading the whole 512 MB back
  // via readCopy() would itself hit the V8 ceiling.
  it("fills 512 MB float32 tensor without V8 buffer (spot check)", () => {
    const MB = 1024 * 1024;
    const BYTES = 512 * MB;
    const N_ELEMS = BYTES / 4;

    const seg = new SharedTensorSegment(BYTES);
    seg.fill([N_ELEMS], DType.FLOAT32, 42.0);

    // Zero-copy read — view into mmap, no copy, no V8 allocation
    const out = seg.read();
    assert.notEqual(out, null);
    assert.equal(out!.data.length, N_ELEMS);

    const data = out!.data as Float32Array;
    assert.equal(data[0], 42.0, "first element");
    assert.equal(data[N_ELEMS - 1], 42.0, "last element");

    seg.destroy();
  });
});
