#!/usr/bin/env node
// PROTOTYPE (additive, opt-in, not part of the default single-threaded
// path): benchmarks + bit-parity-verifies a wasm-threaded RP-tree forest
// build for Zig, using Node `worker_threads` + one private wasm instance
// per worker (see wasm_api_rp_tree_proto.zig for why NOT shared memory).
//
// Usage:
//   node scripts/wasm_threaded_prototype_zig.mjs [n1,n2,...] [--workers=N]
//
// Defaults to n = 100,500,2000,5000,10000,20000,30000, d=32, n_trees=4
// (evoc's default forest size), workers = min(n_trees, os.cpus - 1, 8).
//
// For each n: builds the SAME forest (same data, same rng_states) two
// ways -- (a) sequentially, all n_trees calls to
// `build_rp_tree_leaf_array` in one wasm instance in the main thread
// (this is bit-identical to what the existing single-threaded
// make_forest -> make_float_leaf_array_serial path already does, per
// fixture parity tests -- this script does not re-litigate that, only
// reuses the same underlying function), and (b) distributed across N
// worker_threads, each with its own private wasm instance. Verifies (a)
// and (b) produce byte-identical leaf arrays per tree (after the same
// pad-to-max-width + vstack merge `make_forest` does), then times both.

import { Worker } from "node:worker_threads";
import { readFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const WASM_PATH = path.join(ROOT, "zig", "zig-out", "bin", "uvoc_rp_tree_proto.wasm");
const WORKER_PATH = path.join(__dirname, "wasm_threaded_prototype_zig_worker.mjs");

const LEAF_SIZE = 30n;
const MAX_DEPTH = 200n;
const N_TREES = 4;
const N_FEATURES = 32;

// Deterministic xorshift-ish PRNG, JS-side only, purely for generating
// reproducible synthetic data + rng_state triples for THIS script's
// internal threaded-vs-single-threaded comparison. Not required to match
// numpy/Python bit-for-bit -- that parity is covered elsewhere (fixture
// tests); this script only needs internal (Zig-vs-Zig) consistency.
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function genData(n, d, seed) {
  const rnd = mulberry32(seed);
  const buf = new Float32Array(n * d);
  for (let i = 0; i < buf.length; i++) buf[i] = rnd() * 2 - 1;
  return buf;
}

function genRngStates(nTrees, seed) {
  const rnd = mulberry32(seed);
  const states = [];
  for (let i = 0; i < nTrees; i++) {
    states.push([
      Math.floor(rnd() * 2 ** 31) - 2 ** 30,
      Math.floor(rnd() * 2 ** 31) - 2 ** 30,
      Math.floor(rnd() * 2 ** 31) - 2 ** 30,
    ]);
  }
  return states;
}

async function loadWasmModule() {
  const bytes = readFileSync(WASM_PATH);
  return WebAssembly.compile(bytes);
}

async function buildForestSequential(mod, data, nSamples, nFeatures, rngStates) {
  const instance = await WebAssembly.instantiate(mod, {});
  const exports = instance.exports;
  const memory = exports.memory;

  const dataBytes = nSamples * nFeatures * 4;
  const dataPtr = exports.alloc(dataBytes);
  new Uint8Array(memory.buffer, dataPtr, dataBytes).set(new Uint8Array(data.buffer));

  const outLenPtr = exports.alloc(4);

  const perTree = [];
  for (const rngState of rngStates) {
    const resultPtr = exports.build_rp_tree_leaf_array(
      dataPtr,
      nSamples,
      nFeatures,
      BigInt(rngState[0]),
      BigInt(rngState[1]),
      BigInt(rngState[2]),
      LEAF_SIZE,
      MAX_DEPTH,
      outLenPtr,
    );
    if (!resultPtr) throw new Error("build_rp_tree_leaf_array failed (OOM)");
    const outLen = new Uint32Array(memory.buffer, outLenPtr, 1)[0];
    perTree.push(memory.buffer.slice(resultPtr, resultPtr + outLen));
    exports.dealloc(resultPtr, outLen);
  }
  exports.dealloc(dataPtr, dataBytes);
  exports.dealloc(outLenPtr, 4);
  return perTree;
}

// A persistent pool of long-lived worker_threads (one wasm instance each,
// created once and reused across calls/reps/sizes) -- spawning a fresh
// Worker per call is dominated by ~20ms of Node worker-thread + wasm
// instantiation startup cost, which would swamp any real RP-tree-build
// signal and make the threaded path look uniformly worse regardless of n.
// A real deployment (e.g. a long-lived browser tab) would also want a
// persistent worker pool rather than spawn-per-call, so this is the fair
// and realistic comparison.
class WorkerPool {
  constructor(n) {
    this.workers = Array.from({ length: n }, () => new Worker(WORKER_PATH));
  }
  runOne(worker, assignments, dataBuffer, nSamples, nFeatures) {
    return new Promise((resolve, reject) => {
      const onMessage = (msg) => {
        worker.off("error", onError);
        resolve(msg.results);
      };
      const onError = (err) => {
        worker.off("message", onMessage);
        reject(err);
      };
      worker.once("message", onMessage);
      worker.once("error", onError);
      worker.postMessage({
        wasmPath: WASM_PATH,
        dataBuffer,
        nSamples,
        nFeatures,
        leafSize: 30,
        maxDepth: 200,
        treeAssignments: assignments,
      });
    });
  }
  async terminate() {
    await Promise.all(this.workers.map((w) => w.terminate()));
  }
}

async function buildForestThreaded(pool, data, nSamples, nFeatures, rngStates) {
  const nWorkers = pool.workers.length;
  const assignmentsPerWorker = Array.from({ length: nWorkers }, () => []);
  rngStates.forEach((rngState, treeIndex) => {
    assignmentsPerWorker[treeIndex % nWorkers].push({ treeIndex, rngState });
  });
  // Each call re-slices a fresh copy of the data buffer per worker
  // (structured clone), matching what an in-browser caller would pay to
  // hand each worker its own private-memory copy of the input.
  const perWorkerResults = await Promise.all(
    assignmentsPerWorker.map((assignments, i) =>
      assignments.length === 0
        ? []
        : pool.runOne(pool.workers[i], assignments, data.buffer.slice(0), nSamples, nFeatures),
    ),
  );
  const perTree = new Array(rngStates.length);
  for (const workerResults of perWorkerResults) {
    for (const { treeIndex, bytes } of workerResults) {
      perTree[treeIndex] = bytes;
    }
  }
  return perTree;
}

// Parse one packed tree buffer (as produced by build_rp_tree_leaf_array):
// [u32 n_leaves][u32 width][i32 x n_leaves*width].
function parseTree(buf) {
  const u32 = new Uint32Array(buf, 0, 2);
  const n_leaves = u32[0];
  const width = u32[1];
  const body = new Int32Array(buf, 8, n_leaves * width);
  return { n_leaves, width, body };
}

// Replicates knn_graph.zig's make_forest merge: pad every tree's leaf rows
// to the max width across ALL trees (with -1), then vstack in tree order.
function mergeForest(perTreeBuffers) {
  const trees = perTreeBuffers.map(parseTree);
  const maxWidth = Math.max(...trees.map((t) => t.width));
  let totalLeaves = 0;
  for (const t of trees) totalLeaves += t.n_leaves;
  const out = new Int32Array(totalLeaves * maxWidth).fill(-1);
  let row = 0;
  for (const t of trees) {
    for (let i = 0; i < t.n_leaves; i++) {
      out.set(t.body.subarray(i * t.width, i * t.width + t.width), (row + i) * maxWidth);
    }
    row += t.n_leaves;
  }
  return { n_leaves: totalLeaves, width: maxWidth, data: out };
}

function forestsEqual(a, b) {
  if (a.n_leaves !== b.n_leaves || a.width !== b.width) return false;
  if (a.data.length !== b.data.length) return false;
  for (let i = 0; i < a.data.length; i++) if (a.data[i] !== b.data[i]) return false;
  return true;
}

async function timeIt(fn, reps) {
  const times = [];
  let result;
  for (let i = 0; i < reps; i++) {
    const t0 = performance.now();
    result = await fn();
    times.push(performance.now() - t0);
  }
  times.sort((a, b) => a - b);
  return { result, times, median: times[Math.floor(times.length / 2)] };
}

async function main() {
  const args = process.argv.slice(2);
  const sizesArg = args.find((a) => !a.startsWith("--"));
  const sizes = sizesArg ? sizesArg.split(",").map(Number) : [100, 500, 2000, 5000, 10000, 20000, 30000];
  const workersArg = args.find((a) => a.startsWith("--workers="));
  const nWorkers = workersArg
    ? Number(workersArg.split("=")[1])
    : Math.max(1, Math.min(N_TREES, os.cpus().length - 1, 8));

  console.log(`PROTOTYPE: wasm-threaded RP-tree forest build (Zig)`);
  console.log(`n_trees=${N_TREES} n_features=${N_FEATURES} n_workers=${nWorkers} host cpus=${os.cpus().length}`);
  console.log("");

  const mod = await loadWasmModule();
  const pool = new WorkerPool(nWorkers);
  // Warm up: instantiate each worker's wasm module once before timing, so
  // the "threaded" numbers reflect steady-state (persistent pool) cost,
  // not one-time worker-spawn + wasm-compile cost.
  await buildForestThreaded(pool, genData(8, N_FEATURES, 1), 8, N_FEATURES, genRngStates(N_TREES, 1));

  const rows = [];
  for (const n of sizes) {
    const data = genData(n, N_FEATURES, 1000 + n);
    const rngStates = genRngStates(N_TREES, 2000 + n);

    // Correctness first: single build of each, compare merged forests.
    const seqBuffers = await buildForestSequential(mod, data, n, N_FEATURES, rngStates);
    const threadedBuffers = await buildForestThreaded(pool, data, n, N_FEATURES, rngStates);
    const seqForest = mergeForest(seqBuffers);
    const threadedForest = mergeForest(threadedBuffers);
    const bitExact = forestsEqual(seqForest, threadedForest);

    const reps = n <= 2000 ? 5 : 3;
    const seqTiming = await timeIt(() => buildForestSequential(mod, data, n, N_FEATURES, rngStates), reps);
    const threadedTiming = await timeIt(
      () => buildForestThreaded(pool, data, n, N_FEATURES, rngStates),
      reps,
    );

    const row = {
      n,
      bitExact,
      seqMedianMs: seqTiming.median,
      threadedMedianMs: threadedTiming.median,
      speedup: seqTiming.median / threadedTiming.median,
    };
    rows.push(row);
    console.log(
      `n=${String(n).padStart(6)}  bit-exact=${bitExact ? "YES" : "NO "}  ` +
        `single=${row.seqMedianMs.toFixed(2)}ms  threaded=${row.threadedMedianMs.toFixed(2)}ms  ` +
        `speedup=${row.speedup.toFixed(2)}x`,
    );
    if (!bitExact) {
      console.error(`  !! MISMATCH at n=${n}: threaded output differs from single-threaded output.`);
    }
  }

  console.log("");
  console.log("Summary (single vs threaded RP-forest-build-only time; NOT full pipeline time):");
  console.log("n,bitExact,seqMedianMs,threadedMedianMs,speedup");
  for (const r of rows) {
    console.log(`${r.n},${r.bitExact},${r.seqMedianMs.toFixed(3)},${r.threadedMedianMs.toFixed(3)},${r.speedup.toFixed(3)}`);
  }

  const crossover = rows.find((r) => r.speedup > 1);
  if (crossover) {
    console.log(`\nFirst n with threaded speedup > 1x: n=${crossover.n} (speedup=${crossover.speedup.toFixed(2)}x)`);
  } else {
    console.log(`\nNo tested size showed threaded speedup > 1x (threaded RP-forest build was slower at every size tested).`);
  }

  await pool.terminate();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
