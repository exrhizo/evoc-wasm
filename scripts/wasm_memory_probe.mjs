#!/usr/bin/env node
// Ad-hoc investigation script (not wired into check_all.sh): calls Zig's
// wasm evoc_fit_predict several times in a row on the same wasm instance
// and logs memory.buffer.byteLength after each call, to distinguish a
// one-time high-water-mark ratchet (plateaus after call 1) from a genuine
// per-call leak (keeps growing call after call).
//
// Usage: node scripts/wasm_memory_probe.mjs

import { readFileSync } from "node:fs";
import path from "node:path";
import { ROOT, loadFixture } from "./wasm_fixture_lib.mjs";

const ZIG_WASM = path.join(ROOT, "zig", "zig-out", "bin", "evoc_core.wasm");

function flattenFixture(fx) {
  return { data: fx.data.flat(), nSamples: fx.n_samples, nFeatures: fx.n_features };
}

async function loadZig() {
  const bytes = readFileSync(ZIG_WASM);
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance.exports;
}

function fitPredict(exports, data, nSamples, nFeatures, seed) {
  const memBefore = exports.memory.buffer.byteLength;
  const bufPtr = exports.alloc(data.length * 4);
  new Float32Array(exports.memory.buffer, bufPtr, data.length).set(data);
  const outLabels = exports.alloc(nSamples * 8);
  const status = exports.evoc_fit_predict(
    bufPtr, nSamples, nFeatures, BigInt(seed),
    0.5, 5n, 15, 5, 50, 1.0, 0.2, 10, 20, 0,
    outLabels,
  );
  if (status !== 0) throw new Error(`evoc_fit_predict status ${status}`);
  exports.dealloc(bufPtr, data.length * 4);
  exports.dealloc(outLabels, nSamples * 8);
  const memAfter = exports.memory.buffer.byteLength;
  return { memBefore, memAfter };
}

async function main() {
  const fx = loadFixture("clustering", "panel_large_many_clusters");
  const { data, nSamples, nFeatures } = flattenFixture(fx);

  const exports = await loadZig();
  console.log(`n_samples=${nSamples} n_features=${nFeatures}`);
  console.log(`initial memory.buffer.byteLength = ${exports.memory.buffer.byteLength} (${(exports.memory.buffer.byteLength / 1048576).toFixed(1)} MiB)`);

  const N = 6;
  const results = [];
  for (let i = 0; i < N; i++) {
    const { memBefore, memAfter } = fitPredict(exports, data, nSamples, nFeatures, i);
    results.push(memAfter);
    console.log(
      `call ${i + 1}: before=${memBefore} after=${memAfter} ` +
      `(${(memAfter / 1048576).toFixed(1)} MiB)` +
      (i > 0 ? `  delta_from_prev_call=${memAfter - results[i - 1]}` : ""),
    );
  }

  console.log("\nSame-dataset-different-seed repeated calls above.");
  console.log(
    results[N - 1] === results[1]
      ? "-> Memory PLATEAUS after call 1: consistent with a one-time wasm memory.grow high-water mark (expected wasm behavior), not a per-call leak."
      : "-> Memory KEEPS GROWING across calls: consistent with a real per-call leak, not just the one-way memory.grow ratchet.",
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
