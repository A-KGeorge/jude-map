"use strict";

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  Worker,
  isMainThread,
  parentPort,
  workerData,
} from "node:worker_threads";
import { fileURLToPath } from "node:url";
import { DATA_OFFSET, DType, SharedTensorSegment } from "../index.js";

// ─── Worker-side bootstrap ───────────────────────────────────────────────────
//
// When Node.js loads this test file as a Worker (for the cross-thread tests),
// the "worker mode" block runs instead of the test suite.

if (!isMainThread) {
  const { segSab, maxInputBytes, command } = workerData as {
    segSab: SharedArrayBuffer;
    maxInputBytes: number;
    command: string;
  };

  const seg = SharedTensorSegment.fromSharedBuffer(segSab, maxInputBytes);

  switch (command) {
    // Read whatever the main thread wrote and send it back.
    case "read": {
      const t = seg.read();
      parentPort!.postMessage({
        ok: t !== null,
        shape: t?.shape ?? [],
        dtype: t?.dtype ?? -1,
        // Send the first element only — avoids structured-clone cost for large buffers.
        first: t ? (t.data as Float32Array)[0] : null,
        last: t
          ? (t.data as Float32Array)[(t.data as Float32Array).length - 1]
          : null,
      });
      break;
    }

    // Write from the Worker side; main thread reads back.
    case "write": {
      const input = new Float32Array(16).fill(99.5);
      seg.write([16], DType.FLOAT32, input);
      parentPort!.postMessage({ ok: true });
      break;
    }

    // Verify sharedBuffer accessor works inside a Worker too.
    case "sharedBuffer": {
      let ok = false;
      try {
        const sab = seg.sharedBuffer;
        ok = sab instanceof SharedArrayBuffer && sab === segSab;
      } catch {
        /* noop */
      }
      parentPort!.postMessage({ ok });
      break;
    }

    default:
      parentPort!.postMessage({ error: `unknown command: ${command}` });
  }

  seg.destroy();
} else {
  // ─── Helpers ─────────────────────────────────────────────────────────────────

  function runWorker(
    segSab: SharedArrayBuffer,
    maxInputBytes: number,
    command: string,
  ): Promise<any> {
    return new Promise((resolve, reject) => {
      const workerUrl = new URL("./shared_buffer.worker.mjs", import.meta.url);
      const worker = new Worker(workerUrl, {
        workerData: { segSab, maxInputBytes, command },
      });
      worker.once("message", resolve);
      worker.once("error", reject);
      worker.once("exit", (code) => {
        if (code !== 0) reject(new Error(`Worker exited with code ${code}`));
      });
    });
  }

  // ─── Tests: createShared ──────────────────────────────────────────────────────

  describe("SharedTensorSegment.createShared", () => {
    it("returns a SharedTensorSegment with correct byteCapacity", () => {
      const maxBytes = 4 * 1024;
      const seg = SharedTensorSegment.createShared(maxBytes);
      assert.strictEqual(seg.byteCapacity, maxBytes);
      seg.destroy();
    });

    it("sharedBuffer is a SharedArrayBuffer of DATA_OFFSET + maxBytes", () => {
      const maxBytes = 1024;
      const seg = SharedTensorSegment.createShared(maxBytes);
      const sab = seg.sharedBuffer;
      assert.ok(
        sab instanceof SharedArrayBuffer,
        "sharedBuffer should be a SharedArrayBuffer",
      );
      assert.strictEqual(sab.byteLength, DATA_OFFSET + maxBytes);
      seg.destroy();
    });

    it("sharedBuffer returns the same SAB on every access", () => {
      const seg = SharedTensorSegment.createShared(512);
      assert.strictEqual(seg.sharedBuffer, seg.sharedBuffer);
      seg.destroy();
    });

    it("write + read work normally on a createShared segment", () => {
      const seg = SharedTensorSegment.createShared(4 * 16);
      seg.write([16], DType.FLOAT32, new Float32Array(16).fill(1.5));
      const t = seg.read();
      assert.ok(t !== null);
      assert.deepStrictEqual(t!.shape, [16]);
      assert.strictEqual((t!.data as Float32Array)[0], 1.5);
      seg.destroy();
    });

    it("throws for maxBytes <= 0", () => {
      assert.throws(
        () => SharedTensorSegment.createShared(0),
        /maxBytes must be/,
      );
      assert.throws(
        () => SharedTensorSegment.createShared(-1),
        /maxBytes must be/,
      );
    });

    it("sharedBuffer throws on mmap-backed segment", () => {
      const seg = new SharedTensorSegment(1024);
      assert.throws(
        () => seg.sharedBuffer,
        /createShared|fromSharedBuffer|cross-thread/,
      );
      seg.destroy();
    });
  });

  // ─── Tests: fromSharedBuffer ──────────────────────────────────────────────────

  describe("SharedTensorSegment.fromSharedBuffer", () => {
    it("reconstructed segment can read what was written before reconstruction", () => {
      const maxBytes = 4 * 8;
      const seg1 = SharedTensorSegment.createShared(maxBytes);
      seg1.write([8], DType.FLOAT32, new Float32Array(8).fill(42.0));

      // Simulate Worker reconstruction from the SAB.
      const seg2 = SharedTensorSegment.fromSharedBuffer(
        seg1.sharedBuffer,
        maxBytes,
      );
      const t = seg2.read();
      assert.ok(t !== null);
      assert.deepStrictEqual(t!.shape, [8]);
      assert.strictEqual((t!.data as Float32Array)[0], 42.0);

      seg1.destroy();
      seg2.destroy();
    });

    it("write on reconstructed segment is visible to the original", () => {
      const maxBytes = 4 * 8;
      const seg1 = SharedTensorSegment.createShared(maxBytes);
      const seg2 = SharedTensorSegment.fromSharedBuffer(
        seg1.sharedBuffer,
        maxBytes,
      );

      seg2.write([8], DType.FLOAT32, new Float32Array(8).fill(7.0));

      const t = seg1.read();
      assert.ok(t !== null);
      assert.strictEqual((t!.data as Float32Array)[3], 7.0);

      seg1.destroy();
      seg2.destroy();
    });

    it("byteCapacity matches maxBytes on reconstructed segment", () => {
      const maxBytes = 2048;
      const seg1 = SharedTensorSegment.createShared(maxBytes);
      const seg2 = SharedTensorSegment.fromSharedBuffer(
        seg1.sharedBuffer,
        maxBytes,
      );
      assert.strictEqual(seg2.byteCapacity, maxBytes);
      seg1.destroy();
      seg2.destroy();
    });

    it("throws for non-SAB argument", () => {
      assert.throws(
        () => SharedTensorSegment.fromSharedBuffer({} as any, 1024),
        /SharedArrayBuffer/,
      );
    });

    it("throws when SAB is too small", () => {
      const tiny = new SharedArrayBuffer(DATA_OFFSET - 1);
      assert.throws(
        () => SharedTensorSegment.fromSharedBuffer(tiny, 1),
        /DATA_OFFSET|at least/i,
      );
    });

    it("throws when maxBytes inconsistent with SAB size", () => {
      const sab = new SharedArrayBuffer(DATA_OFFSET + 512);
      assert.throws(
        () => SharedTensorSegment.fromSharedBuffer(sab, 1024), // 1024 > 512
        /inconsistent|maxBytes/,
      );
    });

    it("sharedBuffer on reconstructed segment returns the same SAB", () => {
      const maxBytes = 512;
      const seg1 = SharedTensorSegment.createShared(maxBytes);
      const sab = seg1.sharedBuffer;
      const seg2 = SharedTensorSegment.fromSharedBuffer(sab, maxBytes);
      assert.strictEqual(seg2.sharedBuffer, sab);
      seg1.destroy();
      seg2.destroy();
    });

    it("destroy on reconstructed segment does not affect original", () => {
      const maxBytes = 4 * 4;
      const seg1 = SharedTensorSegment.createShared(maxBytes);
      seg1.write([4], DType.FLOAT32, new Float32Array([1, 2, 3, 4]));

      const seg2 = SharedTensorSegment.fromSharedBuffer(
        seg1.sharedBuffer,
        maxBytes,
      );
      seg2.destroy(); // should not unmap the SAB

      // seg1 still works — SAB memory is owned by the SAB itself, not seg2.
      const t = seg1.read();
      assert.ok(
        t !== null,
        "seg1 should still be readable after seg2.destroy()",
      );
      assert.strictEqual((t!.data as Float32Array)[0], 1);
      seg1.destroy();
    });
  });

  // ─── Tests: DATA_OFFSET export ────────────────────────────────────────────────

  describe("DATA_OFFSET constant", () => {
    it("matches SegmentHeader size (256 bytes)", () => {
      // This mirrors the static_assert in segment.h.
      // If it ever changes, the cross-thread transport protocol breaks.
      assert.strictEqual(DATA_OFFSET, 256);
    });

    it("createShared SAB is exactly DATA_OFFSET + maxBytes", () => {
      const maxBytes = 1337;
      const seg = SharedTensorSegment.createShared(maxBytes);
      assert.strictEqual(seg.sharedBuffer.byteLength, DATA_OFFSET + maxBytes);
      seg.destroy();
    });
  });

  // ─── Tests: cross-Worker thread integration ───────────────────────────────────
  //
  // These tests spin up actual Worker threads to verify that the SAB transport
  // works across the thread boundary — the primary use case for createShared.

  describe("cross-Worker transport", () => {
    it("Worker reads data written by main thread (zero-copy transport)", async () => {
      const maxInputBytes = 4 * 64;
      const seg = SharedTensorSegment.createShared(maxInputBytes);

      const input = new Float32Array(64).fill(3.14);
      seg.write([64], DType.FLOAT32, input);

      const result = await runWorker(seg.sharedBuffer, maxInputBytes, "read");
      assert.ok(result.ok, "Worker should see a non-null TensorResult");
      assert.deepStrictEqual(result.shape, [64]);
      assert.strictEqual(result.dtype, DType.FLOAT32);
      assert.ok(
        Math.abs(result.first - 3.14) < 1e-5,
        `first element should be ~3.14, got ${result.first}`,
      );

      seg.destroy();
    });

    it("Worker writes data; main thread reads it back", async () => {
      const maxInputBytes = 4 * 16;
      const seg = SharedTensorSegment.createShared(maxInputBytes);

      // Worker writes [16] FLOAT32 all 99.5
      await runWorker(seg.sharedBuffer, maxInputBytes, "write");

      const t = seg.read();
      assert.ok(t !== null, "main thread should see Worker's write");
      assert.deepStrictEqual(t!.shape, [16]);
      assert.strictEqual((t!.data as Float32Array)[0], 99.5);

      seg.destroy();
    });

    it("sharedBuffer accessor works inside the Worker", async () => {
      const maxInputBytes = 512;
      const seg = SharedTensorSegment.createShared(maxInputBytes);
      const result = await runWorker(
        seg.sharedBuffer,
        maxInputBytes,
        "sharedBuffer",
      );
      assert.ok(result.ok, "sharedBuffer in Worker should return the same SAB");
      seg.destroy();
    });

    it("seqlock ensures no torn reads across threads", async () => {
      // Main writes a large float32 tensor; Worker reads it.
      // All elements should be identical (no torn read).
      const N = 1024;
      const maxInputBytes = 4 * N;
      const seg = SharedTensorSegment.createShared(maxInputBytes);

      seg.write([N], DType.FLOAT32, new Float32Array(N).fill(1.618));

      const result = await runWorker(seg.sharedBuffer, maxInputBytes, "read");
      assert.ok(result.ok);
      // first and last should both be 1.618 — if they differ the read was torn.
      assert.ok(
        Math.abs(result.first - 1.618) < 1e-4,
        `first=${result.first} should be ~1.618`,
      );
      assert.ok(
        Math.abs(result.last - 1.618) < 1e-4,
        `last=${result.last} should be ~1.618`,
      );

      seg.destroy();
    });
  });
}
