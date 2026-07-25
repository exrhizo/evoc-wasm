// Worker side of scripts/wasm_threaded_prototype_zig.mjs (PROTOTYPE).
//
// Long-lived worker: instantiates its OWN private (non-shared) instance of
// uvoc_rp_tree_proto.wasm ONCE, then handles repeated postMessage requests,
// each building whichever tree indices it was assigned and returning each
// tree's packed leaf-array bytes to the main thread as a transferable
// ArrayBuffer (zero-copy).
//
// See wasm_api_rp_tree_proto.zig's module doc comment for why this uses
// private per-worker memory rather than one shared WebAssembly.Memory.

import { parentPort } from "node:worker_threads";
import { readFileSync } from "node:fs";

let exports = null;
let memory = null;
let loadedWasmPath = null;

async function ensureLoaded(wasmPath) {
  if (exports && loadedWasmPath === wasmPath) return;
  const wasmBytes = readFileSync(wasmPath);
  const mod = await WebAssembly.compile(wasmBytes);
  const instance = await WebAssembly.instantiate(mod, {});
  exports = instance.exports;
  memory = exports.memory;
  loadedWasmPath = wasmPath;
}

function buildOneTree(dataBuffer, nSamples, nFeatures, leafSize, maxDepth, rngState) {
  const dataBytes = nSamples * nFeatures * 4;
  const dataPtr = exports.alloc(dataBytes);
  if (!dataPtr) throw new Error("alloc failed for data");
  new Uint8Array(memory.buffer, dataPtr, dataBytes).set(new Uint8Array(dataBuffer));

  const outLenPtr = exports.alloc(4);
  if (!outLenPtr) throw new Error("alloc failed for out_len");

  const resultPtr = exports.build_rp_tree_leaf_array(
    dataPtr,
    nSamples,
    nFeatures,
    BigInt(rngState[0]),
    BigInt(rngState[1]),
    BigInt(rngState[2]),
    BigInt(leafSize),
    BigInt(maxDepth),
    outLenPtr,
  );
  if (!resultPtr) throw new Error("build_rp_tree_leaf_array failed (OOM)");

  const outLen = new Uint32Array(memory.buffer, outLenPtr, 1)[0];
  const bytes = memory.buffer.slice(resultPtr, resultPtr + outLen);

  exports.dealloc(dataPtr, dataBytes);
  exports.dealloc(outLenPtr, 4);
  exports.dealloc(resultPtr, outLen);

  return bytes;
}

parentPort.on("message", async (msg) => {
  const { wasmPath, dataBuffer, nSamples, nFeatures, leafSize, maxDepth, treeAssignments } = msg;
  await ensureLoaded(wasmPath);
  const results = treeAssignments.map(({ treeIndex, rngState }) => ({
    treeIndex,
    bytes: buildOneTree(dataBuffer, nSamples, nFeatures, leafSize, maxDepth, rngState),
  }));
  parentPort.postMessage(
    { results },
    results.map((r) => r.bytes),
  );
});
