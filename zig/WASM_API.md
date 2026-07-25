# Zig wasm32-freestanding API

This document describes the wasm-facing surface built from
`zig/src/wasm_api.zig`. It wraps the existing native phase 1-8 Zig core
(`root.zig` and friends) — no core algorithm logic lives in `wasm_api.zig`,
only marshalling.

See `PORTING.md` for the underlying module map and `VALIDATION_PLAN.md`
section 3 for why a wasm target exists at all.

## Build

```sh
cd zig
zig build wasm                      # ReleaseSmall (default), see build.zig
zig build wasm -Dwasm-optimize=Debug  # or ReleaseFast/ReleaseSafe
zig build wasm -Dwasm-simd=true       # opt-in SIMD128 target feature, see note below
```

Output: `zig/zig-out/bin/uvoc_core.wasm`. Target is always
`wasm32-freestanding` regardless of `-Dtarget`/`-Doptimize` passed for the
native `zig build test` step — the wasm step resolves its own fixed target
query and reads `-Dwasm-optimize` as a separate option (default
`ReleaseSmall`). ReleaseSmall build size at time of writing: **189,994
bytes** (~186 KiB), unstripped/pre-any-further-optimization (no `wasm-opt`
pass applied — that's left to downstream tooling if desired).

**`-Dwasm-optimize=ReleaseFast` and `-Dwasm-simd=true` were both evaluated
and rejected as the default** (see `comparison/results.md`'s "Wasm
build-config experiment", 2026-07-25): ReleaseFast is ~12x larger for only
a ~3-6% speed gain; SIMD128 is bit-exact-safe (verified via
`node scripts/wasm_fixture_runner.mjs`) but yields no measurable size or
speed change since no core algorithm file uses explicit `@Vector` types for
the auto-vectorizer to widen. `-Dwasm-simd=true` is kept as a documented
opt-in `build.zig` option for future work if hot loops are ever rewritten
with explicit vector types — **if you enable it, always rerun
`node scripts/wasm_fixture_runner.mjs` afterward** and confirm the zig
fixture checks still pass bit-exact, since SIMD can in general reorder
floating-point operations (this port's numeric-parity guarantee depends on
exact operation order, per `PORTING.md`).

The native `zig build test` step is completely unaffected by this addition;
it still builds/runs only for the host target as before.

## Memory model / allocator

Freestanding wasm has no OS allocator. `wasm_api.zig` uses
`std.heap.wasm_allocator` (Zig 0.13's `std.heap.WasmAllocator`), which grows
wasm linear memory itself via `memory.grow` — **no `env` import object is
required** for allocation. There is no exported `_start` (`entry = .disabled`
in build.zig) and no panic-handling host import either: Zig's default
freestanding panic handler traps (`unreachable`) rather than calling out to
JS.

**The import object passed to `WebAssembly.instantiate` can be `{}`
(empty).** Verified in `scripts/wasm_smoke_test_zig.mjs`.

Calling convention from JS:

1. `alloc(len)` → pointer into `memory.buffer` (or `0`/null on OOM).
2. Write input bytes at that pointer via a typed array view
   (`Float32Array`/`Uint8Array`/etc, per function's documented element
   type) over `memory.buffer`.
3. Call the exported function with pointer + length (+ scalar) args.
4. Read results back at the pointer(s) the function documents.
5. `dealloc(ptr, len)` every buffer you allocated once done.

**`i64`/`u64`-typed parameters must be passed as JS `BigInt`** (standard
wasm JS-API behavior); all other scalar params here are `i32`/`u32`/`f32`/
`f64` and take plain JS numbers.

## Exported functions

### `alloc(len: usize) -> ptr (u8*, or 0 on failure)`
Allocate `len` bytes in linear memory.

### `dealloc(ptr: u8*, len: usize) -> void`
Free a buffer previously returned by `alloc` (or any function below
documented as returning an owned buffer). `len` must match the original
allocation size.

### `distance_fast_cosine(x_ptr: f32*, y_ptr: f32*, len: usize) -> f32`
Port of `distance.fast_cosine`. `x`/`y` are `len`-element f32 vectors.
Returns `-dot(x,y)` if `dot > 0`, else `-f32_tiny` (`1.1754944e-38`).

### `distance_fast_bit_jaccard(x_ptr: u8*, y_ptr: u8*, len: usize) -> f32`
Port of `distance.fast_bit_jaccard`. `x`/`y` are `len`-byte packed-bit
vectors.

### `boruvka_mst_edge_count(n_samples: usize) -> usize`
Returns `n_samples - 1` (or `0` if `n_samples == 0`) — the MST edge count.
Call before `boruvka_mst` to size its output buffers.

### `boruvka_mst(data_ptr: f32*, n_samples: usize, n_features: usize, min_samples: usize, leaf_size: i64, reproducible: bool, out_left: f32*, out_right: f32*, out_dist: f32*) -> i32`
Builds a kdtree over `data` (row-major `n_samples x n_features`) and runs
`boruvka.parallel_boruvka` (single-threaded) to compute the
mutual-reachability MST, fusing `kdtree.build_kdtree` +
`parallel_boruvka` (the kdtree by itself has no useful flat-buffer
representation for a wasm caller, so it's not separately exposed).

Writes `n_samples - 1` edges into caller-allocated `out_left`/`out_right`
(f32 point indices, matching the native `Edges.point_a`/`point_b`
representation) and `out_dist` (f32, sqrt-transformed mutual-reachability
distance).

`leaf_size` is a `kdtree.build_kdtree` parameter (40 is a reasonable
default). `reproducible` selects the `boruvka_tree_query_reproducible` path,
matching `parallel_boruvka`'s flag.

Returns `0` on success, `-1` if `n_samples < 2`, `-2` on allocation/internal
failure.

### `cluster_layers_base(data_ptr: f32*, n_samples: usize, n_features: usize, min_samples: usize, base_min_cluster_size: i64, reproducible_flag: bool, min_similarity_threshold: f64, max_layers: usize, out_labels: i64*, out_strengths: f32*) -> i32`
Runs `clustering.build_cluster_layers` on a precomputed embedding and
writes only the base/first layer's labels (i64, `n_samples` elements) and
membership strengths (f32, `n_samples` elements) into the caller-allocated
output buffers. Returns the total number of layers computed (`>= 1`) on
success, or `-2` on failure.

Only the base layer is surfaced over this flat API — a caller wanting
persistence-argmax best-layer selection across ALL layers (as
`EVoC.fit_predict` does) should use `evoc_fit_predict` instead. Exposing the
full jagged `ClusterLayers` structure (variable-length array of
variable-length layers) over a flat C ABI was judged not worth the
complexity for this pass.

### `evoc_fit_predict(data_ptr: f32*, n_samples: usize, n_features: usize, seed: i64, noise_level: f32, base_min_cluster_size: i64, n_neighbors: usize, min_samples: usize, n_epochs: usize, neighbor_scale: f32, min_similarity_threshold: f64, max_layers: usize, n_label_prop_iter: usize, node_embedding_dim_or_zero: usize, out_labels: i64*) -> i32`
Full end-to-end pipeline: `knn_graph` → `graph_construction` →
`label_propagation_init_recursive` → `node_embedding_repr_self_driven` →
`build_cluster_layers` → `EVoC.fit_predict`'s persistence-argmax best-layer
selection. Matches `clustering.EVoC.fit_predict` exactly.

`data_ptr` is row-major `n_samples x n_features` f32. Options mirror
`clustering.EvocClustersOptions` field-for-field.
`node_embedding_dim_or_zero == 0` means "use the default derivation"
(`min(max(n_neighbors/4, 4), 15)`), matching `EvocClustersOptions
.node_embedding_dim = null` over a flat ABI that has no optional type.

Writes selected-layer labels (`-1` = noise) into caller-allocated
`out_labels` (i64, `n_samples` elements). Returns `0` on success, `-2` on
allocation/internal failure.

### `knn_graph(data_ptr: f32*, n_samples: usize, n_features: usize, n_neighbors: usize, leaf_size: i64, n_iters: usize, effective_max_candidates: usize, delta: f64, max_rptree_depth: i64, use_sorted_updates: bool, seed: i64, out_indices: i32*, out_dists: f32*) -> i32`
Standalone approximate kNN graph (`knn_graph.knn_graph`) over `data`
(row-major `n_samples x n_features` f32, assumed already unit-normalized).
Single-threaded (`n_threads = 1`).

`seed` drives every RNG draw (RP-forest construction + NN-descent candidate
sampling) via this project's own `tau_rand`-based `rng.State`, using the
same draw sequence `clustering.evoc_clusters` uses internally (`n_trees =
4`: one `rng.State` draw for nn_descent, then one per tree). This is a
different draw sequence than an externally-recorded `RngState`/`rng_states`
fixture would supply, so this function is **not bit-exact against fixtures
recorded from the Python/native reference's own RNG sequence** — it is
validated for structural correctness (indices in range, distances sorted
ascending per row) and same-seed determinism instead.

`out_indices`/`out_dists` must each point at a caller-allocated buffer of
`n_samples * n_neighbors` elements (i32 / f32 respectively). `leaf_size`
and `max_rptree_depth` mirror `make_forest`'s parameters (`40`/`200` are
reasonable defaults); `use_sorted_updates` mirrors `nn_descent`'s flag
(`true` is the reference default).

Returns `0` on success, `-2` on allocation/internal failure.

### `node_embedding(graph_indptr_ptr: u32*, graph_indices_ptr: u32*, graph_data_ptr: f32*, n_vertices: usize, nnz: usize, n_components: usize, n_epochs: usize, initial_embedding_ptr: f32* (or 0/null), initial_alpha: f32, negative_sample_rate: f32, noise_level: f32, seed: i64, out_embedding: f32*) -> i32`
Standalone self-driven node embedding
(`node_embedding.node_embedding_repr_self_driven`) over a CSR affinity
graph of `n_vertices` vertices: `graph_indptr` (`n_vertices + 1` u32),
`graph_indices`/`graph_data` (`nnz` u32/f32, `nnz = graph_indptr[n_vertices]`).

`initial_embedding_ptr` may be `0` (null) to request a random
(`tau_rand`-driven standard-normal, scale `0.25`) init seeded by `seed`, or
point at `n_vertices * n_components` f32 to seed the layout explicitly.
`out_embedding` must point at a caller-allocated buffer of the same size.

This is the reproducible/self-driven RNG path used internally by
`evoc_fit_predict` — **not bit-exact with the Python reference** (only a
numpy-MT replay could achieve that), and not directly comparable to
fixtures recorded against the externally-RNG-driven
`node_embedding_epoch_repr` path (which takes a caller-supplied per-epoch
`rng_val` array and `node_order` shuffles, not a scalar seed). Validated
for same-seed determinism instead.

Returns `0` on success, `-2` on allocation/internal failure.

### `cluster_tree_labels(mst_edges_ptr: f32*, n_edges: usize, n_samples: usize, min_cluster_size: i64, out_labels: i32*, out_strengths: f32*) -> i32`
Standalone MST → linkage tree → condensed tree → excess-of-mass labels +
membership strengths, in one call. Composes `cluster_trees.mst_to_linkage_tree`
→ `condense_tree` → `extract_leaves` → `get_cluster_label_vector` →
`get_point_membership_strength_vector` — the same chain
`clustering.build_cluster_layers`'s base layer uses, with
`cluster_selection_epsilon` hardcoded to `0.0`.

`mst_edges_ptr` points at `n_edges * 3` f32 elements: flat row-major
`(point_a, point_b, distance)` triples (matching `boruvka_mst`'s output
layout; `n_edges` is normally `n_samples - 1`). Edges are sorted by
distance internally, so the caller does not need to pre-sort.

`out_labels`/`out_strengths` must each point at a caller-allocated buffer
of `n_samples` elements (i32 / f32 respectively).

Returns `0` on success, `-2` on allocation/internal failure.

## Not separately exposed (and why)

`graph_construction` and `label_propagation` are exercised internally by
`evoc_fit_predict` but have no standalone wasm entry points: their native
signatures operate on intermediate CSR/init-embedding state with no
independently useful flat-buffer representation for a caller not already
holding a `knn_graph`/`node_embedding` result from the two exports above.
`evoc_fit_predict` remains the intended way to exercise the full pipeline
in one call; `knn_graph`, `node_embedding`, and `cluster_tree_labels` above
cover the standalone/inspectable intermediate stages.

## Freestanding-wasm gotchas encountered

- No `std.debug.print`/`std.time`/OS allocator usage exists anywhere in the
  non-test phase 1-8 core code (verified by grep across `zig/src/*.zig`
  excluding `test` blocks and `fixture.zig`/
  `dump_evoc_clusters_labels_for_cross_lang.zig`, both test/tooling-only and
  not part of the wasm build's root file). So no shims or gating were
  needed in this pass — the existing core compiled for
  `wasm32-freestanding` without modification.
- `usize` is 32-bit on `wasm32` vs 64-bit on the native host target; one
  call site in `wasm_api.zig` (`boruvka_mst`'s `leaf_size: i64` parameter
  feeding `kdtree.build_kdtree`'s `usize` parameter) needed an explicit
  `@intCast` that wasn't needed in the native-only code path.
- `fixture.zig` (test-only fixture loader, does file I/O via `std.fs`) is
  referenced from `root.zig`'s top-level `const fixture = @import(...)` but
  only actually used inside `root.zig`'s `test { ... }` block; Zig's lazy
  top-level analysis means it's never semantically analyzed when building
  `wasm_api.zig` (which imports `root.zig` for its `pub const` module
  re-exports but does not trigger the `test` block), so no `std.fs`-related
  freestanding-incompatibility issue arose in practice — flagged here in
  case a future change makes `fixture` referenced outside `test{}`.

## Smoke test

`scripts/wasm_smoke_test_zig.mjs` (run: `node scripts/wasm_smoke_test_zig.mjs`
from the repo root, after `zig build wasm`): loads the built `.wasm` with
Node's `WebAssembly` API using an empty `{}` import object, exercises
`distance_fast_cosine` and `boruvka_mst` against hand-computed expected
values, and asserts they match. Passing as of this writing.
