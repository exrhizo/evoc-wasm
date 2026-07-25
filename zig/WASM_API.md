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
```

Output: `zig/zig-out/bin/uvoc_core.wasm`. Target is always
`wasm32-freestanding` regardless of `-Dtarget`/`-Doptimize` passed for the
native `zig build test` step — the wasm step resolves its own fixed target
query and reads `-Dwasm-optimize` as a separate option (default
`ReleaseSmall`). ReleaseSmall build size at time of writing: **174,824
bytes** (~171 KiB), unstripped/pre-any-further-optimization (no `wasm-opt`
pass applied — that's left to downstream tooling if desired).

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

## Not separately exposed (and why)

`knn_graph`, `graph_construction`, `label_propagation`, and `node_embedding`
are exercised internally by `evoc_fit_predict` but have no standalone wasm
entry points in this pass: their native signatures operate on CSR graphs
and threaded RNG state (`rng.State`) that don't have a natural flat-buffer
representation on their own, and the task's "at minimum" bar is satisfied
by covering them through the end-to-end pipeline call. Adding standalone
wrappers for these is a reasonable follow-up if a caller needs to inspect
intermediate pipeline state (e.g. the knn graph before clustering), but was
out of scope here to avoid guessing at an ABI shape not yet exercised by
any caller.

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
