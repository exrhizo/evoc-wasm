// PROTOTYPE worker: instantiates its OWN private wasm module (own private
// WebAssembly.Memory, not shared with the parent or any sibling worker) and
// builds the tree indices assigned to it via `workerData`, matching the
// design note in wasm_threaded_prototype.mjs.
import { parentPort, workerData } from "node:worker_threads";
import { readFileSync } from "node:fs";

async function main() {
  try {
    const { rustWasmPath, dataBuf, nFeatures, leafSize, maxDepth, treeIndices, rngStatesForIndices } =
      workerData;

    const bytes = readFileSync(rustWasmPath);
    const { instance } = await WebAssembly.instantiate(bytes, {});
    const exports = instance.exports;
    const memory = exports.memory;

    const data = new Float32Array(dataBuf);
    const dataBytes = data.length * 4;
    const dataPtr = exports.wasm_proto_alloc(dataBytes);
    new Float32Array(memory.buffer, dataPtr, data.length).set(data);

    const nSamples = data.length / nFeatures;
    const capacity = exports.wasm_proto_leaf_capacity(nSamples, leafSize);

    const trees = [];
    for (const rngStrs of rngStatesForIndices) {
      const outPtr = exports.wasm_proto_alloc(capacity * 4);
      const packed = exports.wasm_proto_build_one_tree(
        dataPtr,
        data.length,
        nFeatures,
        BigInt(rngStrs[0]),
        BigInt(rngStrs[1]),
        BigInt(rngStrs[2]),
        leafSize,
        BigInt(maxDepth),
        outPtr,
        capacity,
      );
      if (packed === -1n) {
        exports.wasm_proto_dealloc(outPtr, capacity * 4);
        throw new Error("wasm_proto_build_one_tree failed (buffer too small)");
      }
      const width = Number(packed >> 32n);
      const nLeaves = Number(packed & 0xffffffffn);
      const flat = Array.from(new Int32Array(memory.buffer, outPtr, nLeaves * width));
      exports.wasm_proto_dealloc(outPtr, capacity * 4);
      trees.push({ flat, width, nLeaves });
    }

    exports.wasm_proto_dealloc(dataPtr, dataBytes);
    parentPort.postMessage({ treeIndices, trees });
  } catch (err) {
    parentPort.postMessage({ error: err.stack || String(err) });
  }
}

main();
