"use strict";

import nodeGypBuild from "node-gyp-build";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let addon: any;
const builtLayoutRoot = join(__dirname, "..");
const sourceLayoutRoot = join(__dirname, "..", "..");

function isNoNativeBuildError(err: unknown): boolean {
  const message = err instanceof Error ? err.message : String(err ?? "");
  return message.includes("No native build was found for");
}

try {
  // Built package layout: dist -> package root.
  addon = nodeGypBuild(builtLayoutRoot) as any;
} catch (firstErr) {
  // Fall back only when the built-layout path truly has no matching prebuild.
  // For other failures (e.g. missing dependent DLLs), keep the original error.
  if (!isNoNativeBuildError(firstErr)) {
    throw firstErr;
  }

  try {
    // Source test layout: src/ts -> package root.
    addon = nodeGypBuild(sourceLayoutRoot) as any;
  } catch (secondErr) {
    if (isNoNativeBuildError(secondErr)) {
      throw secondErr;
    }
    throw firstErr;
  }
}
const { TFSession: NativeTFSession } = addon;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface TensorInfo {
  name: string; // graph node name, e.g. "serving_default_x:0"
  dtype: number; // TF_DataType integer value
  shape: number[]; // dimension sizes (-1 = unknown)
}

export interface SignatureDef {
  inputs: Record<string, TensorInfo>;
  outputs: Record<string, TensorInfo>;
  methodName: string;
}

export interface TensorResult {
  dtype: number;
  shape: number[];
  data:
    | Float32Array
    | Float64Array
    | Int32Array
    | Uint8Array
    | Int8Array
    | Uint16Array
    | Int16Array
    | ArrayBuffer;
}

// Accepted input types for run()
type TensorInput =
  | Float32Array
  | Float64Array
  | Int32Array
  | BigInt64Array
  | Uint8Array
  | Int8Array
  | Uint16Array
  | Int16Array
  | { read(): { buffer: ArrayBuffer; shape: number[]; dtype: number } | null };

// ---------------------------------------------------------------------------
// TFSession
// ---------------------------------------------------------------------------

export class TFSession {
  private _native: any;

  private constructor(native: any) {
    this._native = native;
  }

  // -------------------------------------------------------------------------
  // Factory methods
  // -------------------------------------------------------------------------

  /**
   * Load a TensorFlow SavedModel from a directory.
   *
   * Parses saved_model.pb to auto-detect input/output names and shapes
   * from the SignatureDef. Defaults to the "serve" tag and "serving_default"
   * signature.
   *
   * @param dir   Path to the SavedModel directory (contains saved_model.pb)
   * @param tags  MetaGraph tags to load (default: ["serve"])
   */
  static async loadSavedModel(
    dir: string,
    tags?: string[],
  ): Promise<TFSession> {
    const native = await (tags
      ? NativeTFSession.loadSavedModel(dir, tags)
      : NativeTFSession.loadSavedModel(dir));
    return new TFSession(native);
  }

  /**
   * Load a frozen TensorFlow graph (.pb file).
   *
   * Inputs are inferred from Placeholder ops in the graph.
   * When calling run(), use the op names directly as input keys
   * (e.g. "x", "input_1") rather than SignatureDef keys.
   *
   * @param path  Path to the frozen .pb file
   */
  static async loadFrozenGraph(path: string): Promise<TFSession> {
    const native = await NativeTFSession.loadFrozenGraph(path);
    return new TFSession(native);
  }

  /**
   * Non-blocking inference. TF_SessionRun runs on the libuv thread pool —
   * the event loop stays free for I/O, timers, and other work during inference.
   *
   * Use this on the main thread. Use run() on Worker threads where blocking
   * the isolated event loop is acceptable.
   *
   * @example
   * // Event loop alive throughout — timers and I/O continue during inference
   * const result = await sess.runAsync({ inputs: segment });
   */
  async runAsync(
    inputs: Record<string, TensorInput>,
    outputKeys?: string[],
  ): Promise<Record<string, TensorResult>> {
    return outputKeys
      ? this._native.runAsync(inputs, outputKeys)
      : this._native.runAsync(inputs);
  }

  // -------------------------------------------------------------------------
  // Inference
  // -------------------------------------------------------------------------

  /**
   * Run inference.
   *
   * @param inputs      Map from input key to data source.
   *                    A SharedTensorSegment is passed zero-copy (mmap pointer
   *                    goes directly into TF_NewTensor with a noop deallocator).
   *                    A TypedArray is copied into a new TF_Tensor.
   *
   * @param outputKeys  Which outputs to compute. Defaults to all outputs
   *                    in the active signature.
   *
   * @returns Map from output key to { dtype, shape, data: TypedArray }.
   *
   * @example
   * // Zero-copy path — segment data goes directly to TF without copying
   * const result = await sess.run({ x: segment });
   *
   * // TypedArray path — copied into TF_Tensor
   * const result = await sess.run({ x: new Float32Array([1, 2, 3]) });
   */
  async run(
    inputs: Record<string, TensorInput>,
    outputKeys?: string[],
  ): Promise<Record<string, TensorResult>> {
    return outputKeys
      ? this._native.run(inputs, outputKeys)
      : this._native.run(inputs);
  }

  // -------------------------------------------------------------------------
  // Introspection
  // -------------------------------------------------------------------------

  /**
   * All SignatureDefs detected from saved_model.pb.
   * Empty for frozen graphs.
   *
   * @example
   * const sigs = sess.signatures;
   * console.log(sigs["serving_default"].inputs);
   * // { x: { name: "serving_default_x:0", dtype: 1, shape: [-1, 784] } }
   */
  get signatures(): Record<string, SignatureDef> {
    return this._native.signatures;
  }

  /**
   * Convenience: returns the active signature (serving_default or first found).
   */
  get activeSignature(): SignatureDef | null {
    const sigs = this.signatures;
    if (!sigs) return null;
    return sigs["serving_default"] ?? Object.values(sigs)[0] ?? null;
  }

  /** Inferred Placeholder op names for frozen graphs. Empty for SavedModels. */
  get inputs(): string[] {
    return this._native.inputs ?? [];
  }

  /** Inferred output op names for frozen graphs. Empty for SavedModels. */
  get outputs(): string[] {
    return this._native.outputs ?? [];
  }

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  /**
   * Close the TF session and release all associated C++ resources.
   * The TFSession object is unusable after this call.
   */
  destroy(): void {
    this._native.destroy();
  }
}
