#!/usr/bin/env node
// Wall-clock benchmark of the wasm32 builds of both ports, for
// comparison/results.md's "Execution time" table (VALIDATION_PLAN.md
// section 3, items 3-4). Runs the full evoc_fit_predict / evoc_clusters
// end-to-end pipeline through BOTH wasm binaries on a few fixture dataset
// sizes, plus a couple of cheaper standalone stage functions where both
// languages export them (currently just boruvka_mst -- cluster-tree and
// node_embedding standalone exports are Rust-only, see WASM_API.md files).
//
// Usage: node scripts/wasm_benchmark.mjs
// Prints a table to stdout; does not write files (paste numbers into
// comparison/results.md by hand).

import { readFileSync } from "node:fs";
import path from "node:path";
import { ROOT, loadFixture } from "./wasm_fixture_lib.mjs";

const RUST_WASM = path.join(ROOT, "rust", "target", "wasm32-unknown-unknown", "release", "evoc_core.wasm");
const ZIG_WASM = path.join(ROOT, "zig", "zig-out", "bin", "evoc_core.wasm");

const WARMUP = 2;
const ITERS = 10;

function stats(times) {
  const sorted = [...times].sort((a, b) => a - b);
  const n = sorted.length;
  const median = n % 2 === 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2 : sorted[(n - 1) / 2];
  return { median, min: sorted[0], max: sorted[n - 1] };
}

function fmt(ms) {
  return ms.toFixed(2).padStart(9) + " ms";
}

async function timeRuns(label, fn) {
  for (let i = 0; i < WARMUP; i++) fn();
  const times = [];
  for (let i = 0; i < ITERS; i++) {
    const t0 = performance.now();
    fn();
    const t1 = performance.now();
    times.push(t1 - t0);
  }
  const s = stats(times);
  console.log(`  ${label.padEnd(46)} median=${fmt(s.median)}  min=${fmt(s.min)}  max=${fmt(s.max)}`);
  return s;
}

// ---------------------------------------------------------------------------
// Rust adapter (mirrors scripts/wasm_fixture_runner.mjs's RustCore)
// ---------------------------------------------------------------------------

class RustCore {
  constructor(instance) {
    this.exports = instance.exports;
    this.memory = this.exports.memory;
  }
  alloc(len) {
    const ptr = this.exports.wasm_alloc(len);
    if (ptr === 0) throw new Error(`wasm_alloc(${len}) failed`);
    return ptr;
  }
  dealloc(ptr, len) {
    this.exports.wasm_dealloc(ptr, len);
  }
  writeF32(arr) {
    const ptr = this.alloc(arr.length * 4);
    new Float32Array(this.memory.buffer, ptr, arr.length).set(arr);
    return { ptr, len: arr.length * 4 };
  }
  readF32(ptr, count) {
    return Array.from(new Float32Array(this.memory.buffer, ptr, count));
  }
  free(...bufs) {
    for (const b of bufs) this.dealloc(b.ptr, b.len);
  }
}

async function loadRust() {
  const bytes = readFileSync(RUST_WASM);
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return new RustCore(instance);
}

function rustFitPredict(core, data, nSamples, nFeatures, seed) {
  const buf = core.writeF32(data);
  const outLabels = core.alloc(nSamples * 4);
  const outStrengths = core.alloc(nSamples * 4);
  const status = core.exports.wasm_evoc_fit_predict(
    buf.ptr, nSamples, nFeatures, BigInt(seed),
    0.5, 5n, 15, 5, 50, 1.0, 0.2, 10, 20, 0,
    outLabels, outStrengths,
  );
  if (status < 1) throw new Error(`wasm_evoc_fit_predict status ${status}`);
  core.free(buf, { ptr: outLabels, len: nSamples * 4 }, { ptr: outStrengths, len: nSamples * 4 });
}

function rustBoruvka(core, data, nSamples, nFeatures, minSamples) {
  const buf = core.writeF32(data);
  const nEdges = nSamples - 1;
  const outPtr = core.alloc(nEdges * 3 * 4);
  const status = core.exports.wasm_boruvka_mst(buf.ptr, nSamples, nFeatures, minSamples, outPtr);
  if (status !== nEdges) throw new Error(`wasm_boruvka_mst status ${status}`);
  core.free(buf, { ptr: outPtr, len: nEdges * 3 * 4 });
}

// ---------------------------------------------------------------------------
// Zig adapter (mirrors scripts/wasm_fixture_runner.mjs's ZigCore)
// ---------------------------------------------------------------------------

class ZigCore {
  constructor(instance) {
    this.exports = instance.exports;
    this.memory = this.exports.memory;
  }
  alloc(len) {
    const ptr = this.exports.alloc(len);
    if (ptr === 0) throw new Error(`alloc(${len}) failed`);
    return ptr;
  }
  dealloc(ptr, len) {
    this.exports.dealloc(ptr, len);
  }
  writeF32(arr) {
    const ptr = this.alloc(arr.length * 4);
    new Float32Array(this.memory.buffer, ptr, arr.length).set(arr);
    return { ptr, len: arr.length * 4 };
  }
  readI64(ptr, count) {
    return Array.from(new BigInt64Array(this.memory.buffer, ptr, count)).map((x) => Number(x));
  }
  free(...bufs) {
    for (const b of bufs) this.dealloc(b.ptr, b.len);
  }
}

async function loadZig() {
  const bytes = readFileSync(ZIG_WASM);
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return new ZigCore(instance);
}

function zigFitPredict(core, data, nSamples, nFeatures, seed) {
  const buf = core.writeF32(data);
  const outLabels = core.alloc(nSamples * 8);
  const status = core.exports.evoc_fit_predict(
    buf.ptr, nSamples, nFeatures, BigInt(seed),
    0.5, 5n, 15, 5, 50, 1.0, 0.2, 10, 20, 0,
    outLabels,
  );
  if (status !== 0) throw new Error(`evoc_fit_predict status ${status}`);
  core.free(buf, { ptr: outLabels, len: nSamples * 8 });
}

function zigBoruvka(core, data, nSamples, nFeatures, minSamples, leafSize) {
  const buf = core.writeF32(data);
  const nEdges = core.exports.boruvka_mst_edge_count(nSamples);
  const leftPtr = core.alloc(nEdges * 4);
  const rightPtr = core.alloc(nEdges * 4);
  const distPtr = core.alloc(nEdges * 4);
  const status = core.exports.boruvka_mst(
    buf.ptr, nSamples, nFeatures, minSamples, BigInt(leafSize), 1,
    leftPtr, rightPtr, distPtr,
  );
  if (status !== 0) throw new Error(`boruvka_mst status ${status}`);
  core.free(buf, { ptr: leftPtr, len: nEdges * 4 }, { ptr: rightPtr, len: nEdges * 4 }, { ptr: distPtr, len: nEdges * 4 });
}

// ---------------------------------------------------------------------------

function flattenFixture(fx) {
  return { data: fx.data.flat(), nSamples: fx.n_samples, nFeatures: fx.n_features };
}

async function main() {
  console.log(`Node ${process.version}; ${WARMUP} warmup + ${ITERS} timed iterations per cell\n`);

  const datasets = [
    { name: "panel_small_balanced (n=100)", fx: loadFixture("clustering", "panel_small_balanced") },
    { name: "evoc_clusters_e2e_ari_check (n=240)", fx: loadFixture("clustering", "evoc_clusters_e2e_ari_check") },
    { name: "panel_large_many_clusters (n=800)", fx: loadFixture("clustering", "panel_large_many_clusters") },
  ];

  const rust = await loadRust();
  const zig = await loadZig();

  console.log("=== End-to-end fit_predict/evoc_clusters ===\n");
  for (const { name, fx } of datasets) {
    const { data, nSamples, nFeatures } = flattenFixture(fx);
    console.log(`Dataset: ${name}`);
    await timeRuns(`rust wasm_evoc_fit_predict`, () => rustFitPredict(rust, data, nSamples, nFeatures, 0));
    await timeRuns(`zig  evoc_fit_predict`, () => zigFitPredict(zig, data, nSamples, nFeatures, 0));
    console.log();
  }

  console.log("=== Standalone stage: boruvka MST (mid-size fixture) ===\n");
  {
    const fx = loadFixture("boruvka", "parallel_boruvka_n80_d4_reproducible_true");
    const data = fx.inputs.data.flat();
    const nSamples = fx.inputs.data.length;
    const nFeatures = fx.inputs.data[0].length;
    console.log(`Dataset: boruvka/parallel_boruvka_n80_d4 (n=${nSamples}, d=${nFeatures})`);
    await timeRuns(`rust wasm_boruvka_mst (leaf_size=40 fixed)`, () =>
      rustBoruvka(rust, data, nSamples, nFeatures, fx.inputs.min_samples));
    await timeRuns(`zig  boruvka_mst (leaf_size=${fx.inputs.leaf_size ?? 40})`, () =>
      zigBoruvka(zig, data, nSamples, nFeatures, fx.inputs.min_samples, fx.inputs.leaf_size ?? 40));
  }

  console.log(
    "\nNote: Zig has no standalone wasm export for cluster-tree labels or\n" +
    "node_embedding (only reachable inside evoc_fit_predict) -- see zig/WASM_API.md\n" +
    "'Not separately exposed'. Per-stage comparison is therefore limited to\n" +
    "boruvka MST; end-to-end fit_predict timing above covers everything else.",
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
