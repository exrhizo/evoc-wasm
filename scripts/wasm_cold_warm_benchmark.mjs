#!/usr/bin/env node
// Cold-vs-warm investigation for the Rust-vs-Zig wasm execution-time gap
// documented in comparison/results.md's "Rust wasm overhead" section.
//
// Hypothesis: the gap is dominated by a roughly fixed, n-independent
// ONE-TIME cost paid on the first evoc_fit_predict call of a freshly
// instantiated wasm module (e.g. allocator warmup / many small
// memory.grow calls to ratchet up linear memory from its tiny initial
// size), not a genuine per-element algorithmic cost -- as opposed to the
// earlier documented investigation (build-config levers: panic=abort,
// allocator swap, target-features, opt-level), which never isolated
// cold-instance vs warm-instance timing.
//
// Measures, for both Rust and Zig wasm builds, at n=100 and n=2000:
//   - cold: time for the FIRST evoc_fit_predict call on a freshly
//     instantiated wasm module (fresh WebAssembly.instantiate per sample)
//   - warm: time for calls 2..N on ONE already-instantiated module,
//     reusing memory/allocator state (mirrors wasm_memory_probe.mjs's
//     "call repeatedly on one instance" pattern)
//
// Usage: node scripts/wasm_cold_warm_benchmark.mjs
// Prints a table to stdout; does not write files.

import { readFileSync, existsSync } from "node:fs";
import path from "node:path";
import { ROOT, loadFixture } from "./wasm_fixture_lib.mjs";

const RUST_WASM = path.join(ROOT, "rust", "target", "wasm32-unknown-unknown", "release", "evoc_core.wasm");
const ZIG_WASM = path.join(ROOT, "zig", "zig-out", "bin", "evoc_core.wasm");
const DATA_DIR = path.join(ROOT, "scripts", "scale_data");

const COLD_SAMPLES = 8; // fresh instance per sample
const WARM_SAMPLES = 10; // reused instance, after 1 discarded warmup call

function fmt(ms) {
  return ms.toFixed(3).padStart(9) + " ms";
}
function stats(times) {
  const sorted = [...times].sort((a, b) => a - b);
  const n = sorted.length;
  const median = n % 2 === 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2 : sorted[(n - 1) / 2];
  return { median, min: sorted[0], max: sorted[n - 1] };
}

// --- Rust adapter ---
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
  free(...bufs) {
    for (const b of bufs) this.dealloc(b.ptr, b.len);
  }
}
async function loadRustBytes() {
  return readFileSync(RUST_WASM);
}
async function instantiateRust(bytes) {
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return new RustCore(instance);
}
function rustFitPredict(core, data, n, d, seed) {
  const buf = core.writeF32(data);
  const outLabels = core.alloc(n * 4);
  const outStrengths = core.alloc(n * 4);
  const status = core.exports.wasm_evoc_fit_predict(
    buf.ptr, n, d, BigInt(seed),
    0.5, 5n, 15, 5, 50, 1.0, 0.2, 10, 20, 0,
    outLabels, outStrengths,
  );
  if (status < 1) throw new Error(`wasm_evoc_fit_predict status ${status}`);
  core.free(buf, { ptr: outLabels, len: n * 4 }, { ptr: outStrengths, len: n * 4 });
}

// --- Zig adapter ---
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
  free(...bufs) {
    for (const b of bufs) this.dealloc(b.ptr, b.len);
  }
}
async function loadZigBytes() {
  return readFileSync(ZIG_WASM);
}
async function instantiateZig(bytes) {
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return new ZigCore(instance);
}
function zigFitPredict(core, data, n, d, seed) {
  const buf = core.writeF32(data);
  const outLabels = core.alloc(n * 8);
  const status = core.exports.evoc_fit_predict(
    buf.ptr, n, d, BigInt(seed),
    0.5, 5n, 15, 5, 50, 1.0, 0.2, 10, 20, 0,
    outLabels,
  );
  if (status !== 0) throw new Error(`evoc_fit_predict status ${status}`);
  core.free(buf, { ptr: outLabels, len: n * 8 });
}

function flattenFixture(fx) {
  return { data: fx.data.flat(), nSamples: fx.n_samples, nFeatures: fx.n_features };
}

function loadScaleData(n) {
  const p = path.join(DATA_DIR, `scale_n${n}_d32.f32`);
  if (!existsSync(p)) throw new Error(`missing ${p}; run: python3 scripts/gen_scale_test_data.py`);
  const buf = readFileSync(p);
  return { data: new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4), nSamples: n, nFeatures: 32 };
}

async function measure(label, wasmBytesLoader, instantiate, fitPredict, dataset) {
  const { data, nSamples, nFeatures } = dataset;
  const bytes = await wasmBytesLoader();

  // COLD: fresh instance per sample, time only the first call.
  const coldTimes = [];
  for (let i = 0; i < COLD_SAMPLES; i++) {
    const core = await instantiate(bytes);
    const t0 = performance.now();
    fitPredict(core, data, nSamples, nFeatures, i);
    const t1 = performance.now();
    coldTimes.push(t1 - t0);
  }

  // WARM: one instance, one discarded warmup call, then time calls 2..N+1.
  const warmCore = await instantiate(bytes);
  fitPredict(warmCore, data, nSamples, nFeatures, 999); // warmup call, discarded
  const warmTimes = [];
  for (let i = 0; i < WARM_SAMPLES; i++) {
    const t0 = performance.now();
    fitPredict(warmCore, data, nSamples, nFeatures, i);
    const t1 = performance.now();
    warmTimes.push(t1 - t0);
  }

  const cold = stats(coldTimes);
  const warm = stats(warmTimes);
  console.log(`  ${label.padEnd(28)} cold median=${fmt(cold.median)} (min=${fmt(cold.min)} max=${fmt(cold.max)})`);
  console.log(`  ${" ".repeat(28)} warm median=${fmt(warm.median)} (min=${fmt(warm.min)} max=${fmt(warm.max)})`);
  console.log(`  ${" ".repeat(28)} cold - warm (median) = ${fmt(cold.median - warm.median)}   ratio cold/warm = ${(cold.median / warm.median).toFixed(2)}x`);
  return { cold, warm };
}

async function main() {
  console.log(`Node ${process.version}; ${COLD_SAMPLES} cold samples (fresh instance each), ${WARM_SAMPLES} warm samples (1 discarded warmup + N timed on one instance)\n`);

  const n100 = flattenFixture(loadFixture("clustering", "panel_small_balanced")); // n=100
  const n2000 = loadScaleData(2000);

  const datasets = [
    { name: "n=100", ds: n100 },
    { name: "n=2000", ds: n2000 },
  ];

  for (const { name, ds } of datasets) {
    console.log(`=== ${name} ===`);
    await measure("rust wasm_evoc_fit_predict", loadRustBytes, instantiateRust, rustFitPredict, ds);
    await measure("zig  evoc_fit_predict", loadZigBytes, instantiateZig, zigFitPredict, ds);
    console.log();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
