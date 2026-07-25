// Minimal smoke test for the Zig wasm32-freestanding build
// (zig/zig-out/bin/uvoc_core.wasm, built via `zig build wasm` from zig/).
// See zig/WASM_API.md for the full exported function list/contracts.
//
// Run: node scripts/wasm_smoke_test_zig.mjs
// (build first if needed: cd zig && zig build wasm)
//
// Import object: this module requires NO `env` imports. The build uses
// `std.heap.wasm_allocator`, which manages linear memory itself via
// `memory.grow` instructions with no host callback, and the freestanding
// panic handler traps (`unreachable`) rather than calling out to JS. So the
// import object passed to WebAssembly.instantiate is simply `{}` (empty) --
// see the `importObject` below, kept as an explicit empty object (rather
// than omitted) to document that this is a deliberate, verified fact about
// this build, not an oversight.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const wasmPath = path.resolve(__dirname, "../zig/zig-out/bin/uvoc_core.wasm");

function assertClose(actual, expected, tol, msg) {
  if (Math.abs(actual - expected) > tol) {
    throw new Error(`${msg}: expected ${expected}, got ${actual}`);
  }
  console.log(`  OK: ${msg} (${actual})`);
}

async function main() {
  const bytes = readFileSync(wasmPath);
  const importObject = {}; // no env imports required — see header comment
  const { instance } = await WebAssembly.instantiate(bytes, importObject);
  const { memory, alloc, dealloc, distance_fast_cosine, boruvka_mst_edge_count, boruvka_mst } =
    instance.exports;

  // --- Test 1: distance_fast_cosine ---
  // x = [1, 2, 3], y = [4, 5, 6] -> dot = 4 + 10 + 18 = 32 > 0 -> result = -32
  {
    const xPtr = alloc(3 * 4);
    const yPtr = alloc(3 * 4);
    new Float32Array(memory.buffer, xPtr, 3).set([1, 2, 3]);
    new Float32Array(memory.buffer, yPtr, 3).set([4, 5, 6]);
    const result = distance_fast_cosine(xPtr, yPtr, 3);
    assertClose(result, -32, 1e-5, "distance_fast_cosine([1,2,3],[4,5,6]) == -32");
    dealloc(xPtr, 3 * 4);
    dealloc(yPtr, 3 * 4);
  }

  // --- Test 2: boruvka_mst on 3 trivial 2D points on a line ---
  // Points: (0,0), (1,0), (10,0); min_samples=1 -> core distance 0, so
  // mutual-reachability distance == plain euclidean distance. The MST over
  // 3 collinear points must be the two "short" edges: (0,0)-(1,0) [len 1]
  // and (1,0)-(10,0) [len 9] (never the direct (0,0)-(10,0) length-10 edge,
  // since a 2-edge tree spanning 3 nodes summing to 10 with the shorter
  // edges is strictly cheaper / is the only MST here). Hand-computed
  // expected: edge count == 2, sum of distances == 10.0.
  {
    const n = 3;
    const nFeatures = 2;
    const data = new Float32Array([0, 0, 1, 0, 10, 0]);
    const dataPtr = alloc(data.length * 4);
    new Float32Array(memory.buffer, dataPtr, data.length).set(data);

    const edgeCount = boruvka_mst_edge_count(n);
    assertClose(edgeCount, 2, 0, "boruvka_mst_edge_count(3) == 2");

    const leftPtr = alloc(edgeCount * 4);
    const rightPtr = alloc(edgeCount * 4);
    const distPtr = alloc(edgeCount * 4);

    const status = boruvka_mst(
      dataPtr,
      n,
      nFeatures,
      /* min_samples */ 1,
      /* leaf_size */ 40n, // i64 param
      /* reproducible */ 0,
      leftPtr,
      rightPtr,
      distPtr,
    );
    assertClose(status, 0, 0, "boruvka_mst status == 0 (success)");

    const dists = new Float32Array(memory.buffer, distPtr, edgeCount);
    const sum = dists[0] + dists[1];
    assertClose(sum, 10.0, 1e-4, "boruvka_mst edge distances sum to 10.0");

    dealloc(dataPtr, data.length * 4);
    dealloc(leftPtr, edgeCount * 4);
    dealloc(rightPtr, edgeCount * 4);
    dealloc(distPtr, edgeCount * 4);
  }

  console.log("\nAll wasm smoke tests passed.");
}

main().catch((e) => {
  console.error("SMOKE TEST FAILED:", e);
  process.exit(1);
});
