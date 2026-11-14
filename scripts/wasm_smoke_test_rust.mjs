#!/usr/bin/env node
// Minimal smoke test for the Rust core's wasm32 build (rust/core/src/wasm_api.rs).
//
// Loads the release .wasm produced by:
//   cd rust && cargo build --target wasm32-unknown-unknown --release -p evoc-core
// and exercises two exported functions end to end through the alloc/dealloc
// buffer protocol described in WASM_API.md, checking results against
// hand-computed expected values. This proves the wasm build is actually
// callable (memory layout, calling convention, exports) -- not just that it
// compiles.
//
// Run: node scripts/wasm_smoke_test_rust.mjs

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const wasmPath = path.join(
  __dirname,
  "..",
  "rust",
  "target",
  "wasm32-unknown-unknown",
  "release",
  "evoc_core.wasm",
);

let failures = 0;

function check(label, cond) {
  if (cond) {
    console.log(`ok - ${label}`);
  } else {
    console.error(`FAIL - ${label}`);
    failures += 1;
  }
}

function approxEqual(a, b, eps = 1e-4) {
  return Math.abs(a - b) <= eps;
}

async function main() {
  const bytes = readFileSync(wasmPath);
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const exports = instance.exports;
  const memory = exports.memory;

  // -------------------------------------------------------------------
  // Test 1: wasm_fast_cosine
  //
  // fast_cosine(x, y) = -dot(x, y) if dot > 0 else -f32::MIN_POSITIVE.
  // x = [1, 2, 3], y = [4, 5, 6] => dot = 4 + 10 + 18 = 32 => expect -32.
  // -------------------------------------------------------------------
  {
    const x = new Float32Array([1, 2, 3]);
    const y = new Float32Array([4, 5, 6]);
    const nbytes = x.length * 4;

    const xPtr = exports.wasm_alloc(nbytes);
    const yPtr = exports.wasm_alloc(nbytes);
    new Float32Array(memory.buffer, xPtr, x.length).set(x);
    new Float32Array(memory.buffer, yPtr, y.length).set(y);

    const result = exports.wasm_fast_cosine(xPtr, yPtr, x.length);
    check(`wasm_fast_cosine([1,2,3],[4,5,6]) == -32 (got ${result})`, approxEqual(result, -32));

    exports.wasm_dealloc(xPtr, nbytes);
    exports.wasm_dealloc(yPtr, nbytes);
  }

  // -------------------------------------------------------------------
  // Test 2: wasm_boruvka_mst
  //
  // 3 collinear 1D points: 0, 1, 3 (distances |pi - pj|: 0-1=1, 1-2=2,
  // 0-2=3). With min_samples=1, core_distances are all 0, so the MST is
  // the plain nearest-neighbor MST: edges {0-1 (w=1), 1-2 (w=2)}, total
  // weight 3, and (n_samples - 1) = 2 edges.
  // -------------------------------------------------------------------
  {
    const data = new Float32Array([0, 1, 3]);
    const nSamples = 3;
    const nFeatures = 1;
    const dataBytes = data.length * 4;
    const outBytes = (nSamples - 1) * 3 * 4;

    const dataPtr = exports.wasm_alloc(dataBytes);
    new Float32Array(memory.buffer, dataPtr, data.length).set(data);
    const outPtr = exports.wasm_alloc(outBytes);

    const nEdges = exports.wasm_boruvka_mst(dataPtr, nSamples, nFeatures, 1, outPtr);
    check(`wasm_boruvka_mst returned n_samples-1 = 2 edges (got ${nEdges})`, nEdges === 2);

    const edges = new Float32Array(memory.buffer, outPtr, nEdges * 3);
    let totalWeight = 0;
    const pointsSeen = new Set();
    for (let i = 0; i < nEdges; i++) {
      const a = edges[i * 3];
      const b = edges[i * 3 + 1];
      const w = edges[i * 3 + 2];
      totalWeight += w;
      pointsSeen.add(a);
      pointsSeen.add(b);
    }
    check(`MST total weight == 3 (got ${totalWeight})`, approxEqual(totalWeight, 3));
    check(
      `MST edges touch all 3 points (got ${[...pointsSeen].sort().join(",")})`,
      pointsSeen.size === 3,
    );

    exports.wasm_dealloc(dataPtr, dataBytes);
    exports.wasm_dealloc(outPtr, outBytes);
  }

  // Two-dimensional data previously trapped when recursive PCA produced fewer
  // columns than the default internal embedding width.
  {
    const n = 75, d = 2, values = new Float32Array(n * d);
    for (let i = 0; i < n; i++) { values[2*i] = i/74; values[2*i+1] = 1-i/74; }
    const data = exports.wasm_alloc(values.byteLength), labels = exports.wasm_alloc(n*4), strengths = exports.wasm_alloc(n*4);
    new Float32Array(memory.buffer, data, values.length).set(values);
    const status = exports.wasm_evoc_fit_predict(data,n,d,42n,0.5,5n,15,5,50,1.0,0.2,10,20,0,labels,strengths);
    check(`low-dimensional fit returns a cluster layer (got ${status})`, status >= 1);
    check('low-dimensional fit returns finite membership strengths', [...new Float32Array(memory.buffer,strengths,n)].every(Number.isFinite));
    exports.wasm_dealloc(data,values.byteLength);exports.wasm_dealloc(labels,n*4);exports.wasm_dealloc(strengths,n*4);
  }

  if (failures > 0) {
    console.error(`\n${failures} check(s) failed`);
    process.exit(1);
  }
  console.log("\nall checks passed");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
