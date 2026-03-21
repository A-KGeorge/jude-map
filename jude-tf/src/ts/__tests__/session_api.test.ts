"use strict";

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

async function loadTFSessionOrSkip(t: { skip: (message?: string) => void }) {
  try {
    const mod = await import("../index");
    return mod.TFSession;
  } catch (err: any) {
    const msg = `native addon unavailable in test environment: ${err?.message ?? err}`;
    if (process.env.JUDE_TF_ALLOW_SKIP_NATIVE_TESTS === "1") {
      t.skip(msg);
      return null;
    }
    throw new Error(msg);
  }
}

describe("TFSession API", () => {
  it("exports TFSession static loaders", async (t) => {
    const TFSession = await loadTFSessionOrSkip(t);
    if (!TFSession) return;
    assert.equal(typeof TFSession.loadSavedModel, "function");
    assert.equal(typeof TFSession.loadFrozenGraph, "function");
  });

  it("loadFrozenGraph rejects for missing file", async (t) => {
    const TFSession = await loadTFSessionOrSkip(t);
    if (!TFSession) return;
    const missing = join(tmpdir(), `missing-graph-${Date.now()}.pb`);
    await assert.rejects(
      () => TFSession.loadFrozenGraph(missing),
      /Cannot open frozen graph|failed/i,
    );
  });

  it("loadSavedModel rejects for invalid directory", async (t) => {
    const TFSession = await loadTFSessionOrSkip(t);
    if (!TFSession) return;
    const missingDir = join(tmpdir(), `missing-saved-model-${Date.now()}`);
    await assert.rejects(
      () => TFSession.loadSavedModel(missingDir),
      /TF_LoadSessionFromSavedModel failed|failed/i,
    );
  });

  it("loadSavedModel rejects for empty model directory", async (t) => {
    const TFSession = await loadTFSessionOrSkip(t);
    if (!TFSession) return;
    const dir = mkdtempSync(join(tmpdir(), "jude-tf-empty-model-"));
    try {
      await assert.rejects(
        () => TFSession.loadSavedModel(dir),
        /TF_LoadSessionFromSavedModel failed|failed/i,
      );
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
