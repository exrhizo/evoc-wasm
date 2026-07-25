#!/usr/bin/env node
// PROTOTYPE harness for parallel RP-forest construction, see
// rust/core/src/rp_forest_threaded_prototype.rs's module doc comment for
// the safety argument (each tree in `make_forest` is built from read-only
// shared `data` plus a fully private per-tree `rng_state`, no shared
// mutable state across trees during construction).
//
// DESIGN NOTE (read before assuming this uses literal SharedArrayBuffer +
// Atomics coordination -- it deliberately does not, and here is why):
// this prototype spawns real OS threads via Node's `worker_threads` (one
// per logical worker) and each worker instantiates its OWN, independent
// copy of the wasm module against its OWN private (non-shared)
// `WebAssembly.Memory`. This still gets genuine multi-core wall-clock
// parallelism (worker_threads are real OS threads, not an event-loop
// simulation) while sidestepping a real, separate hazard: Rust's default
// wasm allocator (`dlmalloc`) is not thread-safe, so if multiple wasm
// *instances* shared ONE `WebAssembly.Memory` (the literal
// `shared: true` + cross-instance heap model the task sketched), every
// `Vec`/allocation `wasm_proto_build_one_tree` makes internally could race
// on the allocator's free-list bookkeeping across worker OS threads, since
// nothing in this codebase (or plain `dlmalloc`) makes that safe by
// default -- fixing that would mean writing/vendoring a thread-safe custom
// `#[global_allocator]` for the wasm target, which is its own multi-day
// yak-shave separate from "does wasm-threads-on-nightly work at all"
// (answered below: yes). Per-worker-private-memory avoids the hazard
// entirely: there is no shared mutable memory at all, so there is no race
// to reason about, and the merge step (padding/stacking leaf arrays) runs
// single-threaded on the main thread after every worker's result lands,
// identical in spirit to how `make_forest` merges per-tree results today.
//
// Toolchain finding (still using the nightly + atomics build, confirmed
// working): `rustup toolchain install nightly` + `rustup component add
// rust-src --toolchain nightly` + `RUSTFLAGS="-C target-feature=+atomics,
// +bulk-memory,+mutable-globals" cargo +nightly build --target
// wasm32-unknown-unknown --release -Z build-std=panic_abort,std` compiles
// cleanly on current nightly (rustc 1.99.0-nightly 2026-07-24) -- so a
// genuine SharedArrayBuffer-backed multi-instance design IS reachable from
// here, it would just also need the thread-safe-allocator work above.
// That nightly-only build was produced into `rust/target-threaded-proto/`
// (see comparison/results.md) and is NOT used by this harness (which uses
// the ordinary stable single-threaded release wasm binary, since this
// harness's per-worker-private-memory design needs nothing from the
// atomics feature).
//
// Usage:
//   node scripts/wasm_threaded_prototype.mjs [--verify] [--bench]
//   --verify  build the same forest single-threaded (n_workers=1, in this
//             process) and threaded (n_workers=N, via Workers), assert the
//             leaf arrays are byte-identical per tree.
//   --bench   time single-threaded vs threaded across a size sweep and
//             print a table.

import { Worker } from "node:worker_threads";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import os from "node:os";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, "..");
const RUST_WASM = path.join(
  ROOT,
  "rust",
  "target",
  "wasm32-unknown-unknown",
  "release",
  "uvoc_core.wasm",
);

// ---------------------------------------------------------------------------
// Deterministic synthetic data + rng_states generator (mirrors this file's
// sibling scripts' cosine-friendly orthogonal-cluster-center construction,
// kept self-contained/inline so this harness has no dependency on
// gitignored scale_data fixtures).
// ---------------------------------------------------------------------------

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
  const rand = mulberry32(seed);
  const nClusters = 8;
  const centers = [];
  for (let c = 0; c < nClusters; c++) {
    const v = new Float32Array(d);
    v[c % d] = 4.0;
    centers.push(v);
  }
  const data = new Float32Array(n * d);
  for (let i = 0; i < n; i++) {
    const c = centers[i % nClusters];
    for (let j = 0; j < d; j++) {
      data[i * d + j] = c[j] + (rand() - 0.5) * 0.6;
    }
  }
  return data;
}

// rng_states: one [i64,i64,i64] triple per tree, deterministic from a seed
// (mirrors the fixture convention of pre-drawn rng state; this harness
// doesn't need numpy-bit-identical values, only a fixed, reproducible
// per-tree seed so the SAME tree index always gets the SAME rng_state
// whichever worker builds it -- that determinism is exactly what makes
// per-tree bit-parity meaningful across thread-count/assignment changes).
function genRngStates(nTrees, seed) {
  const rand = mulberry32(seed ^ 0x9e3779b9);
  const states = [];
  for (let t = 0; t < nTrees; t++) {
    const s0 = BigInt(Math.floor(rand() * 1e9)) + 1n;
    const s1 = BigInt(Math.floor(rand() * 1e9)) + 1n;
    const s2 = BigInt(Math.floor(rand() * 1e9)) + 1n;
    states.push([s0, s1, s2]);
  }
  return states;
}

// ---------------------------------------------------------------------------
// Single-instance wasm wrapper around the new prototype exports.
// ---------------------------------------------------------------------------

async function loadCore() {
  const bytes = readFileSync(RUST_WASM);
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance.exports;
}

async function buildOneTree(exports, data, nFeatures, rngState, leafSize, maxDepth) {
  const memory = exports.memory;
  const dataBytes = data.length * 4;
  const dataPtr = exports.wasm_proto_alloc(dataBytes);
  new Float32Array(memory.buffer, dataPtr, data.length).set(data);

  const nSamples = data.length / nFeatures;
  const capacity = exports.wasm_proto_leaf_capacity(nSamples, leafSize);
  const outPtr = exports.wasm_proto_alloc(capacity * 4);

  const packed = exports.wasm_proto_build_one_tree(
    dataPtr,
    data.length,
    nFeatures,
    rngState[0],
    rngState[1],
    rngState[2],
    leafSize,
    BigInt(maxDepth),
    outPtr,
    capacity,
  );

  exports.wasm_proto_dealloc(dataPtr, dataBytes);

  if (packed === -1n || packed === -1) {
    exports.wasm_proto_dealloc(outPtr, capacity * 4);
    throw new Error("wasm_proto_build_one_tree: buffer too small or invalid tree");
  }

  const packedBig = BigInt(packed);
  const width = Number(packedBig >> 32n);
  const nLeaves = Number(packedBig & 0xffffffffn);
  const flat = Array.from(new Int32Array(memory.buffer, outPtr, nLeaves * width));
  exports.wasm_proto_dealloc(outPtr, capacity * 4);
  return { flat, width, nLeaves };
}

async function buildForestSingleThreaded(data, nFeatures, rngStates, leafSize, maxDepth) {
  const exports = await loadCore();
  const trees = [];
  for (const rs of rngStates) {
    trees.push(await buildOneTree(exports, data, nFeatures, rs, leafSize, maxDepth));
  }
  return trees;
}

// ---------------------------------------------------------------------------
// Threaded path: worker_threads, each worker builds a disjoint slice of
// tree indices, private wasm instance + private memory per worker.
// ---------------------------------------------------------------------------

function buildForestThreaded(data, nFeatures, rngStates, leafSize, maxDepth, nWorkers) {
  const nTrees = rngStates.length;
  const chunks = Array.from({ length: nWorkers }, () => []);
  // Deterministic, order-independent assignment: round-robin by tree index.
  // Which worker builds a given tree never affects that tree's own bytes
  // (each tree only depends on `data` + its own rng_state), so any
  // assignment scheme is valid; round-robin just balances load evenly.
  for (let i = 0; i < nTrees; i++) chunks[i % nWorkers].push(i);

  const dataBuf = data.buffer.slice(
    data.byteOffset,
    data.byteOffset + data.byteLength,
  );

  return new Promise((resolve, reject) => {
    const results = new Array(nTrees);
    let remaining = nWorkers;
    for (let w = 0; w < nWorkers; w++) {
      const treeIndices = chunks[w];
      if (treeIndices.length === 0) {
        remaining--;
        if (remaining === 0) resolve(results);
        continue;
      }
      const worker = new Worker(path.join(__dirname, "wasm_threaded_prototype_worker.mjs"), {
        workerData: {
          rustWasmPath: RUST_WASM,
          dataBuf,
          nFeatures,
          leafSize,
          maxDepth,
          treeIndices,
          rngStatesForIndices: treeIndices.map((i) => rngStates[i].map((x) => x.toString())),
        },
      });
      worker.on("message", (msg) => {
        if (msg.error) {
          reject(new Error(msg.error));
          return;
        }
        for (let k = 0; k < msg.treeIndices.length; k++) {
          results[msg.treeIndices[k]] = msg.trees[k];
        }
        remaining--;
        if (remaining === 0) resolve(results);
      });
      worker.on("error", reject);
    }
  });
}

// Empirical crossover (see comparison/results.md's "Threaded RP-tree
// prototype" section): measured on this 24-core machine with
// nTrees=16/nWorkers=8, threaded overhead (worker spawn + wasm
// instantiate-per-worker + postMessage copy) stops being worth paying
// somewhere between n=7000 (0.81x, still slower) and n=8000 (0.97x,
// roughly break-even); n=10000 is the first clearly-faster point (1.09x),
// widening steadily to 2.37x by n=30000. This constant is deliberately
// conservative (biased toward n=10000, not the noisier ~8000 break-even
// point) so the adaptive dispatch only pays worker-spawn overhead when a
// win is comfortably established.
const ADAPTIVE_THRESHOLD_N = 10000;

/**
 * Adaptive dispatch: build the RP-forest single-threaded below
 * ADAPTIVE_THRESHOLD_N, threaded above it. Both paths are bit-identical
 * per-tree (see --verify), so this choice only affects wall-clock time,
 * never output.
 */
async function buildForestAdaptive(data, nFeatures, rngStates, leafSize, maxDepth, nWorkers) {
  const n = data.length / nFeatures;
  if (n < ADAPTIVE_THRESHOLD_N) {
    return buildForestSingleThreaded(data, nFeatures, rngStates, leafSize, maxDepth);
  }
  return buildForestThreaded(data, nFeatures, rngStates, leafSize, maxDepth, nWorkers);
}

function mergeForest(perTree) {
  const maxWidth = Math.max(...perTree.map((t) => t.width));
  const rows = [];
  for (const { flat, width, nLeaves } of perTree) {
    for (let r = 0; r < nLeaves; r++) {
      const row = flat.slice(r * width, r * width + width);
      while (row.length < maxWidth) row.push(-1);
      rows.push(row);
    }
  }
  return rows;
}

function treesEqual(a, b) {
  if (a.width !== b.width || a.nLeaves !== b.nLeaves) return false;
  if (a.flat.length !== b.flat.length) return false;
  for (let i = 0; i < a.flat.length; i++) if (a.flat[i] !== b.flat[i]) return false;
  return true;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

async function verify() {
  const n = 4000;
  const d = 32;
  const nFeatures = d;
  const data = genData(n, d, 7);
  const nTrees = 8;
  const rngStates = genRngStates(nTrees, 7).map((s) => s.map((x) => x));
  const leafSize = 30;
  const maxDepth = 200;

  console.log(`verify: n=${n} d=${d} nTrees=${nTrees}`);

  const single = await buildForestSingleThreaded(data, nFeatures, rngStates, leafSize, maxDepth);

  for (const nWorkers of [2, 4]) {
    const threaded = await buildForestThreaded(data, nFeatures, rngStates, leafSize, maxDepth, nWorkers);
    let allMatch = true;
    for (let i = 0; i < nTrees; i++) {
      if (!treesEqual(single[i], threaded[i])) {
        allMatch = false;
        console.log(`  MISMATCH at tree ${i} with nWorkers=${nWorkers}`);
      }
    }
    console.log(`  nWorkers=${nWorkers}: bit-identical per-tree = ${allMatch}`);
  }
}

async function bench() {
  const sizes = [100, 500, 2000, 5000, 10000, 20000, 30000];
  const d = 32;
  const nTrees = Math.max(4, Math.min(16, os.cpus().length));
  const leafSize = 30;
  const maxDepth = 200;
  const nWorkers = Math.min(8, os.cpus().length);

  console.log(`bench: nTrees=${nTrees} nWorkers=${nWorkers} cpus=${os.cpus().length}`);
  console.log("n\tsingle_ms\tthreaded_ms\tspeedup");

  for (const n of sizes) {
    const data = genData(n, d, 1234);
    const rngStates = genRngStates(nTrees, 1234);

    const t0 = performance.now();
    await buildForestSingleThreaded(data, d, rngStates, leafSize, maxDepth);
    const t1 = performance.now();
    const singleMs = t1 - t0;

    const t2 = performance.now();
    await buildForestThreaded(data, d, rngStates, leafSize, maxDepth, nWorkers);
    const t3 = performance.now();
    const threadedMs = t3 - t2;

    console.log(
      `${n}\t${singleMs.toFixed(2)}\t${threadedMs.toFixed(2)}\t${(singleMs / threadedMs).toFixed(2)}x`,
    );
  }
}

const args = process.argv.slice(2);
if (args.includes("--verify")) await verify();
if (args.includes("--bench")) await bench();
if (!args.includes("--verify") && !args.includes("--bench")) {
  console.log("usage: node scripts/wasm_threaded_prototype.mjs [--verify] [--bench]");
}
