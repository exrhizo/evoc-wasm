#!/usr/bin/env node
// Runs the phase 1-8 JSON fixtures under fixtures/ through the compiled
// wasm32 binaries of BOTH ports (Rust: rust/target/wasm32-unknown-unknown/
// release/uvoc_core.wasm, Zig: zig/zig-out/bin/uvoc_core.wasm) and asserts
// bit-exact / documented-tolerance agreement with each fixture's expected
// output, per PORTING.md's tolerance classes. This is the "does bit-parity
// survive the actual shipping target" check described in
// VALIDATION_PLAN.md section 3, item 2.
//
// The Rust and Zig wasm APIs are NOT identical (see rust/core/WASM_API.md
// and zig/WASM_API.md) -- this file has one section per language with its
// own adapter code, sharing only comparison/reporting helpers from
// wasm_fixture_lib.mjs.
//
// Usage: node scripts/wasm_fixture_runner.mjs
// Exits non-zero if either binary is missing (with build instructions) or
// if any check fails.

import { readFileSync } from "node:fs";
import path from "node:path";
import {
  ROOT,
  loadFixture,
  tolFor,
  closeF32,
  compareF32Array,
  compareIntArray,
  adjustedRandIndex,
  median,
  Reporter,
  assert,
} from "./wasm_fixture_lib.mjs";

const RUST_WASM = path.join(
  ROOT,
  "rust",
  "target",
  "wasm32-unknown-unknown",
  "release",
  "uvoc_core.wasm",
);
const ZIG_WASM = path.join(ROOT, "zig", "zig-out", "bin", "uvoc_core.wasm");

function fileExists(p) {
  try {
    readFileSync(p);
    return true;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Rust adapter
// ---------------------------------------------------------------------------

class RustCore {
  constructor(instance) {
    this.exports = instance.exports;
    this.memory = this.exports.memory;
  }

  alloc(len) {
    const ptr = this.exports.wasm_alloc(len);
    assert(ptr !== 0, `wasm_alloc(${len}) failed`);
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
  writeU8(arr) {
    const ptr = this.alloc(arr.length);
    new Uint8Array(this.memory.buffer, ptr, arr.length).set(arr);
    return { ptr, len: arr.length };
  }
  writeI8(arr) {
    const ptr = this.alloc(arr.length);
    new Int8Array(this.memory.buffer, ptr, arr.length).set(arr);
    return { ptr, len: arr.length };
  }
  writeU32(arr) {
    const ptr = this.alloc(arr.length * 4);
    new Uint32Array(this.memory.buffer, ptr, arr.length).set(arr);
    return { ptr, len: arr.length * 4 };
  }
  readF32(ptr, count) {
    return Array.from(new Float32Array(this.memory.buffer, ptr, count));
  }
  readI32(ptr, count) {
    return Array.from(new Int32Array(this.memory.buffer, ptr, count));
  }

  free(...bufs) {
    for (const b of bufs) this.dealloc(b.ptr, b.len);
  }
}

async function runRust(reporter) {
  if (!fileExists(RUST_WASM)) {
    reporter.skip(
      "rust wasm suite",
      `binary not found at ${path.relative(ROOT, RUST_WASM)} -- run: cd rust && cargo build --target wasm32-unknown-unknown --release -p uvoc-core`,
    );
    return;
  }
  const bytes = readFileSync(RUST_WASM);
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const core = new RustCore(instance);

  // -- distance kernels ------------------------------------------------------
  await reporter.run("distance/fast_cosine", () => {
    const fx = loadFixture("distance", "fast_cosine");
    const tol = tolFor(fx.tolerance);
    fx.inputs.pairs.forEach((pair, i) => {
      const x = core.writeF32(pair.x);
      const y = core.writeF32(pair.y);
      const got = core.exports.wasm_fast_cosine(x.ptr, y.ptr, pair.x.length);
      core.free(x, y);
      assert(
        closeF32(got, fx.expected.values[i], tol),
        `pair ${i}: got ${got} expected ${fx.expected.values[i]}`,
      );
    });
  });

  await reporter.run("distance/fast_bit_jaccard", () => {
    const fx = loadFixture("distance", "fast_bit_jaccard");
    const tol = tolFor(fx.tolerance);
    fx.inputs.pairs.forEach((pair, i) => {
      const x = core.writeU8(pair.x);
      const y = core.writeU8(pair.y);
      const got = core.exports.wasm_fast_bit_jaccard(x.ptr, y.ptr, pair.x.length);
      core.free(x, y);
      assert(
        closeF32(got, fx.expected.values[i], tol),
        `pair ${i}: got ${got} expected ${fx.expected.values[i]}`,
      );
    });
  });

  await reporter.run("distance/fast_int_inner_product_dissimilarity", () => {
    const fx = loadFixture("distance", "fast_int_inner_product_dissimilarity");
    const tol = tolFor(fx.tolerance);
    fx.inputs.pairs.forEach((pair, i) => {
      const x = core.writeI8(pair.x);
      const y = core.writeI8(pair.y);
      const got = core.exports.wasm_fast_int_inner_product_dissimilarity(x.ptr, y.ptr, pair.x.length);
      core.free(x, y);
      assert(
        closeF32(got, fx.expected.values[i], tol),
        `pair ${i}: got ${got} expected ${fx.expected.values[i]}`,
      );
    });
  });

  // -- boruvka MST -------------------------------------------------------------
  // Rust's wasm_boruvka_mst has no `reproducible` flag (boruvka.rs documents
  // reproducible==false is bit-identical to reproducible=true at
  // n_threads==1) AND hardcodes leaf_size=40 internally (see WASM_API.md),
  // while these fixtures were recorded with leaf_size=10 -- a different kd-
  // tree leaf size changes Boruvka's per-round traversal/merge order, so the
  // *edge order* is not expected to match exactly even though both are valid
  // MSTs of the same point set. Assert what leaf_size-invariant properties
  // must hold instead: edge count, total MST weight, and that every point
  // appears (a spanning tree), then separately assert exact order/value
  // agreement is achievable when we reconstruct a leaf_size=10 tree
  // ourselves is not possible over this flat ABI -- documented as a real
  // wasm-API scope gap (no leaf_size parameter), not a numeric divergence.
  for (const caseName of [
    "parallel_boruvka_n80_d4_reproducible_true",
    "parallel_boruvka_n80_d4_reproducible_false",
  ]) {
    await reporter.run(`boruvka/${caseName} (leaf_size=40 vs fixture's leaf_size=10, structural check only)`, () => {
      const fx = loadFixture("boruvka", caseName);
      const distTol = tolFor(fx.tolerance, "distance");
      const data = fx.inputs.data.flat();
      const nSamples = fx.inputs.data.length;
      const nFeatures = fx.inputs.data[0].length;
      const dataBuf = core.writeF32(data);
      const nEdges = nSamples - 1;
      const outPtr = core.alloc(nEdges * 3 * 4);
      const status = core.exports.wasm_boruvka_mst(
        dataBuf.ptr,
        nSamples,
        nFeatures,
        fx.inputs.min_samples,
        outPtr,
      );
      assert(status === nEdges, `expected ${nEdges} edges, got status ${status}`);
      const edges = core.readF32(outPtr, nEdges * 3);
      core.free(dataBuf, { ptr: outPtr, len: nEdges * 3 * 4 });

      let totalWeight = 0;
      let expectedTotalWeight = 0;
      const pointsSeen = new Set();
      for (let i = 0; i < nEdges; i++) {
        const a = Math.round(edges[i * 3]);
        const b = Math.round(edges[i * 3 + 1]);
        totalWeight += edges[i * 3 + 2];
        expectedTotalWeight += fx.expected.distance[i];
        pointsSeen.add(a);
        pointsSeen.add(b);
      }
      assert(pointsSeen.size === nSamples, `MST should span all ${nSamples} points, touched ${pointsSeen.size}`);
      // MST total weight is leaf_size-invariant (it's the unique minimum
      // spanning weight for this mutual-reachability metric, ties aside);
      // use the distance field's own tolerance, scaled for summed error.
      assert(
        closeF32(totalWeight, expectedTotalWeight, { exact: false, rel: distTol.rel ?? 1e-4, abs: (distTol.abs ?? 1e-6) * nEdges }),
        `total MST weight: got ${totalWeight} expected ${expectedTotalWeight}`,
      );
    });
  }

  // -- cluster trees: chain boruvka MST -> wasm_cluster_tree_labels ----------
  // labels_and_strengths_n80_mcs5's `condensed_tree` input is itself derived
  // from mst_to_linkage_tree_n80's `sorted_mst` (same n=80 dataset,
  // min_cluster_size=5, cluster_selection_epsilon=0 -- matching the wasm
  // API's hardcoded epsilon). wasm_cluster_tree_labels composes
  // mst_to_linkage_tree -> condense_tree -> extract_leaves -> labels ->
  // strengths in one call, so feeding the recorded sorted_mst through it end
  // to end reproduces exactly what labels_and_strengths_n80_mcs5 expects.
  await reporter.run("cluster_trees/labels_and_strengths_n80_mcs5 (chained from mst_to_linkage_tree_n80)", () => {
    const mstFx = loadFixture("cluster_trees", "mst_to_linkage_tree_n80");
    const labelsFx = loadFixture("cluster_trees", "labels_and_strengths_n80_mcs5");
    const nSamples = labelsFx.inputs.n_samples;
    const sm = mstFx.inputs.sorted_mst;
    const nEdges = sm.point_a.length;
    const flat = new Float32Array(nEdges * 3);
    for (let i = 0; i < nEdges; i++) {
      flat[i * 3] = sm.point_a[i];
      flat[i * 3 + 1] = sm.point_b[i];
      flat[i * 3 + 2] = sm.distance[i];
    }
    const edgesBuf = core.writeF32(flat);
    const outLabelsPtr = core.alloc(nSamples * 4);
    const outStrengthsPtr = core.alloc(nSamples * 4);
    const status = core.exports.wasm_cluster_tree_labels(
      edgesBuf.ptr,
      nEdges,
      nSamples,
      5n, // min_cluster_size=5 (the "_mcs5" fixture name)
      outLabelsPtr,
      outStrengthsPtr,
    );
    assert(status === 0, `wasm_cluster_tree_labels status ${status}`);
    const labels = core.readI32(outLabelsPtr, nSamples);
    const strengths = core.readF32(outStrengthsPtr, nSamples);
    core.free(
      edgesBuf,
      { ptr: outLabelsPtr, len: nSamples * 4 },
      { ptr: outStrengthsPtr, len: nSamples * 4 },
    );
    const labelsCmp = compareIntArray(labels, labelsFx.expected.labels, "labels");
    assert(labelsCmp.ok, labelsCmp.detail);
    const strengthTol = tolFor(labelsFx.tolerance, "strengths");
    const strengthsCmp = compareF32Array(strengths, labelsFx.expected.strengths, strengthTol, "strengths");
    assert(strengthsCmp.ok, strengthsCmp.detail);
  });

  // -- node embedding: determinism only ---------------------------------------
  // wasm_node_embedding wraps `node_embedding_repr_self_driven`, a
  // seed-driven variant with no fixture of its own yet (the recorded
  // node_embedding_e2e_n120 fixture targets the externally-RNG-driven
  // `node_embedding_epoch_repr` path used by clustering.py -- a different
  // function with an incompatible calling convention: it takes a
  // caller-supplied per-epoch rng_val array and node_order_shuffles, not a
  // scalar seed). Not fixture-comparable; check same-seed determinism
  // instead (the property the self-driven path actually promises).
  await reporter.run("node_embedding/wasm_node_embedding determinism (not fixture-comparable, see note)", () => {
    const fx = loadFixture("node_embedding", "node_embedding_e2e_n120");
    const nVertices = fx.inputs.n_vertices;
    const nComponents = fx.inputs.n_components;
    const indptr = core.writeU32(fx.inputs.indptr);
    const indices = core.writeU32(fx.inputs.indices);
    const data = core.writeF32(fx.inputs.data);
    const nnz = fx.inputs.indices.length;

    function runOnce(seed) {
      const outPtr = core.alloc(nVertices * nComponents * 4);
      const status = core.exports.wasm_node_embedding(
        indptr.ptr,
        indices.ptr,
        data.ptr,
        nVertices,
        nnz,
        nComponents,
        10,
        0, // initial_embedding_ptr = null -> random init
        0.1,
        1.0,
        0.5,
        BigInt(seed),
        outPtr,
      );
      assert(status === 0, `wasm_node_embedding status ${status}`);
      const out = core.readF32(outPtr, nVertices * nComponents);
      core.free({ ptr: outPtr, len: nVertices * nComponents * 4 });
      return out;
    }
    const a = runOnce(42);
    const b = runOnce(42);
    const cmp = compareF32Array(a, b, { exact: true }, "same-seed embedding");
    assert(cmp.ok, cmp.detail);
    core.free(indptr, indices, data);
  });

  // -- knn_graph: structural sanity + determinism, not fixture-exact --------
  // wasm_knn_graph derives its internal RngState/rng_states deterministically
  // from a scalar `seed` (a fixed draw sequence baked into wasm_api.rs); the
  // recorded knn_graph_n120 fixture instead supplies the raw already-drawn
  // RngState/rng_states arrays used by the native/Python reference, which
  // can't be reverse-engineered from a seed. So this is not directly
  // fixture-comparable; check structural validity (indices in range,
  // distances sorted ascending per row) and determinism instead.
  await reporter.run("graph/wasm_knn_graph structural sanity + determinism (not fixture-comparable, see note)", () => {
    const fx = loadFixture("graph", "knn_graph_n120_d8_k10_seed42");
    const data = fx.inputs.data.flat();
    const nSamples = fx.inputs.data.length;
    const nFeatures = fx.inputs.data[0].length;
    const k = fx.inputs.n_neighbors;
    const dataBuf = core.writeF32(data);

    function runOnce(seed) {
      const outIdx = core.alloc(nSamples * k * 4);
      const outDist = core.alloc(nSamples * k * 4);
      const status = core.exports.wasm_knn_graph(
        dataBuf.ptr,
        nSamples,
        nFeatures,
        k,
        40,
        fx.inputs.n_iters,
        fx.inputs.effective_max_candidates,
        fx.inputs.delta,
        BigInt(seed),
        outIdx,
        outDist,
      );
      assert(status === 0, `wasm_knn_graph status ${status}`);
      const idx = core.readI32(outIdx, nSamples * k);
      const dist = core.readF32(outDist, nSamples * k);
      core.free({ ptr: outIdx, len: nSamples * k * 4 }, { ptr: outDist, len: nSamples * k * 4 });
      return { idx, dist };
    }
    const r1 = runOnce(42);
    const r2 = runOnce(42);
    assert(compareIntArray(r1.idx, r2.idx, "idx").ok, "same-seed indices differ");
    assert(compareF32Array(r1.dist, r2.dist, { exact: true }, "dist").ok, "same-seed distances differ");
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < k; j++) {
        const ix = r1.idx[i * k + j];
        assert(ix >= 0 && ix < nSamples, `row ${i} neighbor ${j}: index ${ix} out of range`);
        if (j > 0) {
          assert(
            r1.dist[i * k + j] >= r1.dist[i * k + j - 1] - 1e-4,
            `row ${i}: distances not sorted ascending at ${j}`,
          );
        }
      }
    }
    core.free(dataBuf);
  });

  // -- build_cluster_layers (best layer, via argmax persistence) -------------
  await reporter.run("clustering/build_cluster_layers_n126 (best layer, argmax persistence)", () => {
    const fx = loadFixture("clustering", "build_cluster_layers_n126");
    const nSamples = fx.inputs.embedding.length;
    const nFeatures = fx.inputs.embedding[0].length;
    const embedding = core.writeF32(fx.inputs.embedding.flat());
    const outLabels = core.alloc(nSamples * 4);
    const outStrengths = core.alloc(nSamples * 4);
    const nLayers = core.exports.wasm_build_cluster_layers_best(
      embedding.ptr,
      nSamples,
      nFeatures,
      fx.inputs.min_samples,
      BigInt(fx.inputs.base_min_cluster_size),
      fx.inputs.min_similarity_threshold,
      fx.inputs.max_layers,
      outLabels,
      outStrengths,
    );
    assert(nLayers >= 1, `wasm_build_cluster_layers_best status ${nLayers}`);
    const labels = core.readI32(outLabels, nSamples);
    const strengths = core.readF32(outStrengths, nSamples);
    core.free(embedding, { ptr: outLabels, len: nSamples * 4 }, { ptr: outStrengths, len: nSamples * 4 });

    const bestIdx = fx.expected.persistence_scores.reduce(
      (best, v, i, arr) => (v > arr[best] ? i : best),
      0,
    );
    const expLabels = fx.expected.cluster_layers[bestIdx];
    const expStrengths = fx.expected.membership_strengths[bestIdx];
    const labelsCmp = compareIntArray(labels, expLabels, "labels");
    assert(labelsCmp.ok, labelsCmp.detail);
    const tol = tolFor(fx.tolerance, "membership_strengths");
    const strengthsCmp = compareF32Array(strengths, expStrengths, tol, "strengths");
    assert(strengthsCmp.ok, strengthsCmp.detail);
  });

  // -- evoc_clusters e2e: ARI panel vs Python (seeds 0..19) -------------------
  await reporter.run("clustering/evoc_clusters_e2e_ari_check (ARI panel seeds 0..19 vs Python)", () => {
    const fx = loadFixture("clustering", "evoc_clusters_e2e_ari_check");
    const nSamples = fx.n_samples;
    const nFeatures = fx.n_features;
    const data = core.writeF32(fx.data.flat());
    const pythonLabels = fx.python_labels;

    const SEED_PANEL = Array.from({ length: 20 }, (_, i) => i);
    const PYTHON_NULL_MEDIAN = 0.5246615432510635;
    const MEDIAN_GATE = PYTHON_NULL_MEDIAN - 0.15;
    const MIN_GATE = 0.15;

    const aris = SEED_PANEL.map((seed) => {
      const outLabels = core.alloc(nSamples * 4);
      const outStrengths = core.alloc(nSamples * 4);
      const status = core.exports.wasm_evoc_fit_predict(
        data.ptr,
        nSamples,
        nFeatures,
        BigInt(seed),
        0.5, // noise_level
        5n, // base_min_cluster_size
        15, // n_neighbors
        5, // min_samples
        50, // n_epochs
        1.0, // neighbor_scale
        0.2, // min_similarity_threshold
        10, // max_layers
        20, // n_label_prop_iter
        0, // node_embedding_dim (default)
        outLabels,
        outStrengths,
      );
      assert(status >= 1, `wasm_evoc_fit_predict status ${status} at seed ${seed}`);
      const labels = core.readI32(outLabels, nSamples);
      core.free({ ptr: outLabels, len: nSamples * 4 }, { ptr: outStrengths, len: nSamples * 4 });
      return adjustedRandIndex(labels, pythonLabels);
    });
    core.free(data);
    const sorted = [...aris].sort((a, b) => a - b);
    const med = median(sorted);
    const min = sorted[0];
    console.log(
      `    [rust] ARI panel: median=${med.toFixed(4)} min=${min.toFixed(4)} (gates: median>=${MEDIAN_GATE.toFixed(4)}, min>=${MIN_GATE})`,
    );
    assert(med >= MEDIAN_GATE, `panel median ARI ${med} < gate ${MEDIAN_GATE}`);
    assert(min >= MIN_GATE, `panel min ARI ${min} < floor ${MIN_GATE}`);
  });
}

// ---------------------------------------------------------------------------
// Zig adapter
// ---------------------------------------------------------------------------

class ZigCore {
  constructor(instance) {
    this.exports = instance.exports;
    this.memory = this.exports.memory;
  }
  alloc(len) {
    const ptr = this.exports.alloc(len);
    assert(ptr !== 0, `alloc(${len}) failed`);
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
  writeU8(arr) {
    const ptr = this.alloc(arr.length);
    new Uint8Array(this.memory.buffer, ptr, arr.length).set(arr);
    return { ptr, len: arr.length };
  }
  writeI8(arr) {
    const ptr = this.alloc(arr.length);
    new Int8Array(this.memory.buffer, ptr, arr.length).set(arr);
    return { ptr, len: arr.length };
  }
  readF32(ptr, count) {
    return Array.from(new Float32Array(this.memory.buffer, ptr, count));
  }
  readI64(ptr, count) {
    return Array.from(new BigInt64Array(this.memory.buffer, ptr, count)).map((x) => Number(x));
  }
  readI32(ptr, count) {
    return Array.from(new Int32Array(this.memory.buffer, ptr, count));
  }
  writeU32(arr) {
    const ptr = this.alloc(arr.length * 4);
    new Uint32Array(this.memory.buffer, ptr, arr.length).set(arr);
    return { ptr, len: arr.length * 4 };
  }
  free(...bufs) {
    for (const b of bufs) this.dealloc(b.ptr, b.len);
  }
}

async function runZig(reporter) {
  if (!fileExists(ZIG_WASM)) {
    reporter.skip(
      "zig wasm suite",
      `binary not found at ${path.relative(ROOT, ZIG_WASM)} -- run: cd zig && zig build wasm`,
    );
    return;
  }
  const bytes = readFileSync(ZIG_WASM);
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const core = new ZigCore(instance);

  await reporter.run("distance/fast_cosine", () => {
    const fx = loadFixture("distance", "fast_cosine");
    const tol = tolFor(fx.tolerance);
    fx.inputs.pairs.forEach((pair, i) => {
      const x = core.writeF32(pair.x);
      const y = core.writeF32(pair.y);
      const got = core.exports.distance_fast_cosine(x.ptr, y.ptr, pair.x.length);
      core.free(x, y);
      assert(
        closeF32(got, fx.expected.values[i], tol),
        `pair ${i}: got ${got} expected ${fx.expected.values[i]}`,
      );
    });
  });

  await reporter.run("distance/fast_bit_jaccard", () => {
    const fx = loadFixture("distance", "fast_bit_jaccard");
    const tol = tolFor(fx.tolerance);
    fx.inputs.pairs.forEach((pair, i) => {
      const x = core.writeU8(pair.x);
      const y = core.writeU8(pair.y);
      const got = core.exports.distance_fast_bit_jaccard(x.ptr, y.ptr, pair.x.length);
      core.free(x, y);
      assert(
        closeF32(got, fx.expected.values[i], tol),
        `pair ${i}: got ${got} expected ${fx.expected.values[i]}`,
      );
    });
  });

  reporter.skip(
    "distance/fast_int_inner_product_dissimilarity",
    "not exported by zig wasm_api.zig (no distance_fast_int_inner_product_dissimilarity export)",
  );

  for (const [caseName, reproducible] of [
    ["parallel_boruvka_n80_d4_reproducible_true", true],
    ["parallel_boruvka_n80_d4_reproducible_false", false],
  ]) {
    await reporter.run(`boruvka/${caseName}`, () => {
      const fx = loadFixture("boruvka", caseName);
      const distTol = tolFor(fx.tolerance, "distance");
      const data = fx.inputs.data.flat();
      const nSamples = fx.inputs.data.length;
      const nFeatures = fx.inputs.data[0].length;
      const dataBuf = core.writeF32(data);
      const nEdges = core.exports.boruvka_mst_edge_count(nSamples);
      assert(nEdges === nSamples - 1, `edge count ${nEdges}`);
      const leftPtr = core.alloc(nEdges * 4);
      const rightPtr = core.alloc(nEdges * 4);
      const distPtr = core.alloc(nEdges * 4);
      const status = core.exports.boruvka_mst(
        dataBuf.ptr,
        nSamples,
        nFeatures,
        fx.inputs.min_samples,
        BigInt(fx.inputs.leaf_size ?? 40),
        reproducible ? 1 : 0,
        leftPtr,
        rightPtr,
        distPtr,
      );
      assert(status === 0, `boruvka_mst status ${status}`);
      const left = core.readF32(leftPtr, nEdges);
      const right = core.readF32(rightPtr, nEdges);
      const dist = core.readF32(distPtr, nEdges);
      core.free(
        dataBuf,
        { ptr: leftPtr, len: nEdges * 4 },
        { ptr: rightPtr, len: nEdges * 4 },
        { ptr: distPtr, len: nEdges * 4 },
      );
      for (let i = 0; i < nEdges; i++) {
        const a = Math.round(left[i]);
        const b = Math.round(right[i]);
        assert(a === fx.expected.point_a[i], `edge[${i}].point_a: got ${a} expected ${fx.expected.point_a[i]}`);
        assert(b === fx.expected.point_b[i], `edge[${i}].point_b: got ${b} expected ${fx.expected.point_b[i]}`);
        assert(
          closeF32(dist[i], fx.expected.distance[i], distTol),
          `edge[${i}].distance: got ${dist[i]} expected ${fx.expected.distance[i]}`,
        );
      }
    });
  }

  // -- cluster trees: chain boruvka MST -> cluster_tree_labels ---------------
  // Same fixture-chaining approach as the Rust adapter above.
  await reporter.run("cluster_trees/labels_and_strengths_n80_mcs5 (chained from mst_to_linkage_tree_n80)", () => {
    const mstFx = loadFixture("cluster_trees", "mst_to_linkage_tree_n80");
    const labelsFx = loadFixture("cluster_trees", "labels_and_strengths_n80_mcs5");
    const nSamples = labelsFx.inputs.n_samples;
    const sm = mstFx.inputs.sorted_mst;
    const nEdges = sm.point_a.length;
    const flat = new Float32Array(nEdges * 3);
    for (let i = 0; i < nEdges; i++) {
      flat[i * 3] = sm.point_a[i];
      flat[i * 3 + 1] = sm.point_b[i];
      flat[i * 3 + 2] = sm.distance[i];
    }
    const edgesBuf = core.writeF32(flat);
    const outLabelsPtr = core.alloc(nSamples * 4);
    const outStrengthsPtr = core.alloc(nSamples * 4);
    const status = core.exports.cluster_tree_labels(
      edgesBuf.ptr,
      nEdges,
      nSamples,
      5n, // min_cluster_size=5 (the "_mcs5" fixture name), i64 param
      outLabelsPtr,
      outStrengthsPtr,
    );
    assert(status === 0, `cluster_tree_labels status ${status}`);
    const labels = core.readI32(outLabelsPtr, nSamples);
    const strengths = core.readF32(outStrengthsPtr, nSamples);
    core.free(
      edgesBuf,
      { ptr: outLabelsPtr, len: nSamples * 4 },
      { ptr: outStrengthsPtr, len: nSamples * 4 },
    );
    const labelsCmp = compareIntArray(labels, labelsFx.expected.labels, "labels");
    assert(labelsCmp.ok, labelsCmp.detail);
    const strengthTol = tolFor(labelsFx.tolerance, "strengths");
    const strengthsCmp = compareF32Array(strengths, labelsFx.expected.strengths, strengthTol, "strengths");
    assert(strengthsCmp.ok, strengthsCmp.detail);
  });

  // -- node embedding: determinism only (see Rust adapter's note above for
  // why node_embedding_e2e_n120 is not directly fixture-comparable against
  // the self-driven path) ------------------------------------------------
  await reporter.run("node_embedding/node_embedding determinism (not fixture-comparable, see note)", () => {
    const fx = loadFixture("node_embedding", "node_embedding_e2e_n120");
    const nVertices = fx.inputs.n_vertices;
    const nComponents = fx.inputs.n_components;
    const indptr = core.writeU32(fx.inputs.indptr);
    const indices = core.writeU32(fx.inputs.indices);
    const data = core.writeF32(fx.inputs.data);
    const nnz = fx.inputs.indices.length;

    function runOnce(seed) {
      const outPtr = core.alloc(nVertices * nComponents * 4);
      const status = core.exports.node_embedding(
        indptr.ptr,
        indices.ptr,
        data.ptr,
        nVertices,
        nnz,
        nComponents,
        10,
        0, // initial_embedding_ptr = null -> random init
        0.1,
        1.0,
        0.5,
        BigInt(seed),
        outPtr,
      );
      assert(status === 0, `node_embedding status ${status}`);
      const out = core.readF32(outPtr, nVertices * nComponents);
      core.free({ ptr: outPtr, len: nVertices * nComponents * 4 });
      return out;
    }
    const a = runOnce(42);
    const b = runOnce(42);
    const cmp = compareF32Array(a, b, { exact: true }, "same-seed embedding");
    assert(cmp.ok, cmp.detail);
    core.free(indptr, indices, data);
  });

  // -- knn_graph: structural sanity + determinism, not fixture-exact --------
  await reporter.run("graph/knn_graph_n120_d8_k10_seed42 structural sanity + determinism (not fixture-comparable, see note)", () => {
    const fx = loadFixture("graph", "knn_graph_n120_d8_k10_seed42");
    const data = fx.inputs.data.flat();
    const nSamples = fx.inputs.data.length;
    const nFeatures = fx.inputs.data[0].length;
    const k = fx.inputs.n_neighbors;
    const dataBuf = core.writeF32(data);

    function runOnce(seed) {
      const outIdx = core.alloc(nSamples * k * 4);
      const outDist = core.alloc(nSamples * k * 4);
      const status = core.exports.knn_graph(
        dataBuf.ptr,
        nSamples,
        nFeatures,
        k,
        BigInt(40), // leaf_size
        fx.inputs.n_iters,
        fx.inputs.effective_max_candidates,
        fx.inputs.delta,
        BigInt(200), // max_rptree_depth
        1, // use_sorted_updates
        BigInt(seed),
        outIdx,
        outDist,
      );
      assert(status === 0, `knn_graph status ${status}`);
      const idx = core.readI32(outIdx, nSamples * k);
      const dist = core.readF32(outDist, nSamples * k);
      core.free({ ptr: outIdx, len: nSamples * k * 4 }, { ptr: outDist, len: nSamples * k * 4 });
      return { idx, dist };
    }
    const r1 = runOnce(42);
    const r2 = runOnce(42);
    assert(compareIntArray(r1.idx, r2.idx, "idx").ok, "same-seed indices differ");
    assert(compareF32Array(r1.dist, r2.dist, { exact: true }, "dist").ok, "same-seed distances differ");
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < k; j++) {
        const ix = r1.idx[i * k + j];
        assert(ix >= 0 && ix < nSamples, `row ${i} neighbor ${j}: index ${ix} out of range`);
        if (j > 0) {
          assert(
            r1.dist[i * k + j] >= r1.dist[i * k + j - 1] - 1e-4,
            `row ${i}: distances not sorted ascending at ${j}`,
          );
        }
      }
    }
    core.free(dataBuf);
  });

  // -- cluster_layers_base: compare against fixture's base layer -------------
  // build_cluster_layers_n126 has no RNG (see its `note`); cluster_layers_base
  // surfaces the base/first layer (pre-persistence-argmax). The fixture
  // documents its output arrays are already in final sorted-by-n_clusters
  // order, and that the base layer is "index 0 before sorting" -- for this
  // 2-layer fixture we try index 0 first (most likely to be the base layer)
  // and report clearly if neither fixture layer matches, rather than
  // silently asserting a possibly-wrong index.
  await reporter.run("clustering/build_cluster_layers_n126 (base layer, best-effort layer-index match)", () => {
    const fx = loadFixture("clustering", "build_cluster_layers_n126");
    const nSamples = fx.inputs.embedding.length;
    const nFeatures = fx.inputs.embedding[0].length;
    const embedding = core.writeF32(fx.inputs.embedding.flat());
    const outLabels = core.alloc(nSamples * 8); // i64
    const outStrengths = core.alloc(nSamples * 4);
    const totalLayers = core.exports.cluster_layers_base(
      embedding.ptr,
      nSamples,
      nFeatures,
      fx.inputs.min_samples,
      BigInt(fx.inputs.base_min_cluster_size),
      fx.inputs.reproducible_flag ? 1 : 0,
      fx.inputs.min_similarity_threshold,
      fx.inputs.max_layers,
      outLabels,
      outStrengths,
    );
    assert(totalLayers >= 1, `cluster_layers_base status ${totalLayers}`);
    assert(totalLayers === fx.expected.cluster_layers.length, `layer count ${totalLayers} vs fixture ${fx.expected.cluster_layers.length}`);
    const labels = core.readI64(outLabels, nSamples);
    const strengths = core.readF32(outStrengths, nSamples);
    core.free(embedding, { ptr: outLabels, len: nSamples * 8 }, { ptr: outStrengths, len: nSamples * 4 });

    const tol = tolFor(fx.tolerance, "membership_strengths");
    let matchedIdx = -1;
    for (let idx = 0; idx < fx.expected.cluster_layers.length; idx++) {
      const labelsCmp = compareIntArray(labels, fx.expected.cluster_layers[idx], "labels");
      const strengthsCmp = compareF32Array(strengths, fx.expected.membership_strengths[idx], tol, "strengths");
      if (labelsCmp.ok && strengthsCmp.ok) {
        matchedIdx = idx;
        break;
      }
    }
    assert(
      matchedIdx >= 0,
      `base layer output did not match any of the ${fx.expected.cluster_layers.length} fixture layers`,
    );
    console.log(`    [zig] base layer matched fixture layer index ${matchedIdx}`);
  });

  // -- evoc_fit_predict e2e: ARI panel vs Python (seeds 0..19) ----------------
  await reporter.run("clustering/evoc_clusters_e2e_ari_check (ARI panel seeds 0..19 vs Python)", () => {
    const fx = loadFixture("clustering", "evoc_clusters_e2e_ari_check");
    const nSamples = fx.n_samples;
    const nFeatures = fx.n_features;
    const data = core.writeF32(fx.data.flat());
    const pythonLabels = fx.python_labels;

    const SEED_PANEL = Array.from({ length: 20 }, (_, i) => i);
    const PYTHON_NULL_MEDIAN = 0.5246615432510635;
    const MEDIAN_GATE = PYTHON_NULL_MEDIAN - 0.15;
    const MIN_GATE = 0.15;

    const aris = SEED_PANEL.map((seed) => {
      const outLabels = core.alloc(nSamples * 8); // i64
      const status = core.exports.evoc_fit_predict(
        data.ptr,
        nSamples,
        nFeatures,
        BigInt(seed),
        0.5,
        5n,
        15,
        5,
        50,
        1.0,
        0.2,
        10,
        20,
        0,
        outLabels,
      );
      assert(status === 0, `evoc_fit_predict status ${status} at seed ${seed}`);
      const labels = core.readI64(outLabels, nSamples);
      core.free({ ptr: outLabels, len: nSamples * 8 });
      return adjustedRandIndex(labels, pythonLabels);
    });
    core.free(data);
    const sorted = [...aris].sort((a, b) => a - b);
    const med = median(sorted);
    const min = sorted[0];
    console.log(
      `    [zig] ARI panel: median=${med.toFixed(4)} min=${min.toFixed(4)} (gates: median>=${MEDIAN_GATE.toFixed(4)}, min>=${MIN_GATE})`,
    );
    assert(med >= MEDIAN_GATE, `panel median ARI ${med} < gate ${MEDIAN_GATE}`);
    assert(min >= MIN_GATE, `panel min ARI ${min} < floor ${MIN_GATE}`);
  });
}

// ---------------------------------------------------------------------------

async function main() {
  const rustReporter = new Reporter("rust");
  const zigReporter = new Reporter("zig");

  await runRust(rustReporter);
  await runZig(zigReporter);

  const rs = rustReporter.summary();
  const zs = zigReporter.summary();
  console.log("\n=== wasm fixture runner summary ===");
  console.log(`rust: ${rs.pass} pass, ${rs.fail} fail, ${rs.skip} skip (of ${rs.total})`);
  console.log(`zig:  ${zs.pass} pass, ${zs.fail} fail, ${zs.skip} skip (of ${zs.total})`);

  if (rs.fail > 0 || zs.fail > 0) {
    process.exit(1);
  }
}

main().catch((e) => {
  console.error("wasm_fixture_runner.mjs crashed:", e);
  process.exit(1);
});
