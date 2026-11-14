# `evoc-core` wasm32 API

`rust/core/src/wasm_api.rs` exposes a flat `extern "C"` API over the
`evoc-core` Rust crate for `wasm32-unknown-unknown`, with no wasm-bindgen
and no dependencies (the crate stays dependency-free for lean wasm builds,
per `rust/core/Cargo.toml`). The module is gated with
`#[cfg(target_arch = "wasm32")]` in `rust/core/src/lib.rs`, so it does not
exist in native builds and cannot affect `cargo test --workspace`.

## Build

```sh
rustup target add wasm32-unknown-unknown   # one-time
cd rust
cargo build --target wasm32-unknown-unknown --release -p evoc-core
# -> rust/target/wasm32-unknown-unknown/release/evoc_core.wasm
```

`Cargo.toml`'s `[lib]` section sets `crate-type = ["cdylib", "rlib"]`: the
`cdylib` target produces the standalone `.wasm` module with these exports;
`rlib` keeps native `cargo test` / other workspace crates working as a
normal Rust library dependency.

A minimal end-to-end smoke test (loads the built `.wasm` via Node's
`WebAssembly` API, calls a couple of exports, checks hand-computed
results) lives at `scripts/wasm_smoke_test_rust.mjs`:

```sh
node scripts/wasm_smoke_test_rust.mjs
```

## Calling convention

wasm functions can only pass/return numeric scalars (i32/i64/f32/f64) --
never arrays, slices, or structs. Every function below uses the same
caller-controlled buffer protocol:

1. JS calls `wasm_alloc(len)` to reserve `len` bytes of wasm linear memory
   and gets back a pointer (byte offset into the wasm `memory` export's
   `ArrayBuffer`).
2. JS writes input data into wasm memory at that offset via a typed-array
   view (e.g. `new Float32Array(memory.buffer, ptr, count).set(data)`).
3. JS calls the real exported function, passing pointers and lengths/counts
   instead of arrays.
4. For outputs, JS pre-allocates an output buffer the same way -- the
   required size is documented per function below as a simple function of
   the inputs, computable before the call -- and passes its pointer in; the
   Rust side writes results into it in place.
5. JS reads the output buffer back via a typed-array view over
   `memory.buffer`.
6. JS calls `wasm_dealloc(ptr, len)` on every pointer it allocated, with the
   *same* `len` used at allocation (the allocator reconstructs its layout
   from it).

Functions that can fail return an `i32` status/count instead of panicking
across the ABI boundary (`0` or a positive count = success; negative =
error). All float buffers are `f32` (the core's native float type
throughout the port); label/index buffers are `i32`.

wasm32-unknown-unknown has no threads, so every algorithm below runs single
threaded (matching this crate's existing "no threads, no filesystem, no
clock" design noted in `rust/core/src/lib.rs`'s module doc comment) -- no
code changes were needed to make the phase 1-8 core algorithms wasm-safe;
they were already freestanding-friendly.

## Memory management

- `wasm_alloc(len: usize) -> *mut u8` -- reserve `len` bytes, uninitialized.
- `wasm_dealloc(ptr: *mut u8, len: usize)` -- free memory from `wasm_alloc`;
  `len` must match the original call.

## Distance kernels (`distance.rs`)

- `wasm_fast_cosine(x_ptr: *const f32, y_ptr: *const f32, len: usize) -> f32`
- `wasm_fast_bit_jaccard(x_ptr: *const u8, y_ptr: *const u8, len: usize) -> f32`
- `wasm_fast_int_inner_product_dissimilarity(x_ptr: *const i8, y_ptr: *const i8, len: usize) -> f32`

Each computes the named dissimilarity between two length-`len` vectors
already written into wasm memory. No output buffer needed (scalar return).

## kNN graph (`knn_graph.rs`)

```
wasm_knn_graph(
    data_ptr: *const f32,        // n_samples * n_features, row-major
    n_samples: usize,
    n_features: usize,
    n_neighbors: usize,
    leaf_size: usize,
    n_iters: usize,
    effective_max_candidates: usize,
    delta: f64,
    seed: i64,
    out_indices_ptr: *mut i32,   // n_samples * n_neighbors
    out_dists_ptr: *mut f32,     // n_samples * n_neighbors
) -> i32   // 0 = success
```

Approximate kNN via NN-descent (float/cosine path). `seed` drives every RNG
draw (RP-forest construction + NN-descent candidate sampling).

## Boruvka MST (`boruvka.rs`)

```
wasm_boruvka_mst(
    data_ptr: *const f32,        // n_samples * n_features, row-major
    n_samples: usize,
    n_features: usize,
    min_samples: usize,
    out_edges_ptr: *mut f32,     // (n_samples - 1) * 3
) -> i32   // number of edges written (n_samples - 1), or < 0 on error
```

Builds a kd-tree internally (`leaf_size = 40`, the reference default) and
runs mutual-reachability Boruvka MST. Output is flat row-major
`(point_a, point_b, distance)` triples, unsorted.

## Cluster trees (`cluster_trees.rs`)

```
wasm_cluster_tree_labels(
    mst_edges_ptr: *const f32,   // n_edges * 3, e.g. wasm_boruvka_mst's output
    n_edges: usize,
    n_samples: usize,
    min_cluster_size: i64,
    out_labels_ptr: *mut i32,    // n_samples
    out_strengths_ptr: *mut f32, // n_samples
) -> i32   // 0 = success
```

Composes `mst_to_linkage_tree` -> `condense_tree` -> `extract_leaves` ->
`get_cluster_label_vector` -> `get_point_membership_strength_vector` in one
call. Sorts edges by weight internally, so `wasm_boruvka_mst`'s unsorted
output can be passed straight in.

## Node embedding (`node_embedding.rs`)

```
wasm_node_embedding(
    graph_indptr_ptr: *const u32,    // n_vertices + 1 (CSR)
    graph_indices_ptr: *const u32,   // nnz
    graph_data_ptr: *const f32,      // nnz
    n_vertices: usize,
    nnz: usize,
    n_components: usize,
    n_epochs: usize,
    initial_embedding_ptr: *const f32, // n_vertices * n_components, or null (0) for random init
    initial_alpha: f32,
    negative_sample_rate: f32,
    noise_level: f32,
    seed: i64,
    out_embedding_ptr: *mut f32,     // n_vertices * n_components
) -> i32   // 0 = success
```

SGD graph layout (`node_embedding_repr_self_driven`) over a CSR affinity
graph. Passing `initial_embedding_ptr = 0` (null) requests the reference's
random PCA-style init, driven by `seed`.

## Cluster layers, best layer only (`clustering.rs`, phase 8)

```
wasm_build_cluster_layers_best(
    data_ptr: *const f32,        // n_samples * n_features, row-major embedding
    n_samples: usize,
    n_features: usize,
    min_samples: usize,
    base_min_cluster_size: i64,
    min_similarity_threshold: f64,
    max_layers: usize,
    out_labels_ptr: *mut i32,    // n_samples
    out_strengths_ptr: *mut f32, // n_samples
) -> i32   // number of internal layers found (>= 1), or < 0 on error
```

Runs `build_cluster_layers` over an already-computed embedding, then
collapses the result to the single highest-persistence layer (same
`argmax` rule `EVoC::fit_predict` uses) -- the flat C ABI doesn't
describe a variable-length list of variable-length layers, so only the
best layer's labels/strengths are returned. The full per-layer data
(`cluster_layers`, `membership_strength_layers`, `persistence_scores`) is
available natively via `clustering::build_cluster_layers` for any
non-wasm caller that needs all layers.

## End-to-end `evoc_clusters` / `EVoC::fit_predict` (`clustering.rs`, phase 9)

```
wasm_evoc_fit_predict(
    data_ptr: *const f32,        // n_samples * n_features, row-major, raw (NOT pre-normalized)
    n_samples: usize,
    n_features: usize,
    seed: i64,
    noise_level: f32,
    base_min_cluster_size: i64,
    n_neighbors: usize,
    min_samples: usize,
    n_epochs: usize,
    neighbor_scale: f32,
    min_similarity_threshold: f64,
    max_layers: usize,
    n_label_prop_iter: usize,
    node_embedding_dim: usize,   // 0 = default clamp(n_neighbors/4, 4, 15)
    out_labels_ptr: *mut i32,    // n_samples
    out_strengths_ptr: *mut f32, // n_samples
) -> i32   // number of internal layers found (>= 1), or < 0 on error
```

The full pipeline: `knn_graph` -> `neighbor_graph_matrix` ->
`label_propagation_init` (recursive) -> `node_embedding` ->
`build_cluster_layers`, collapsed to the best layer -- i.e. this is
`EVoC::fit_predict`, not the raw multi-layer `evoc_clusters`. `seed` drives
every random draw in the pipeline (RP-tree construction, NN-descent
candidate sampling, label-propagation partition seeding + tie-breaks,
randomized-PCA base cases, node-embedding SGD): same seed -> same output,
cross-language reproducible against the Zig port, but not bit-exact with
the Python reference (impossible once PCA is involved -- see
`clustering.rs`'s module doc comment and `PCA_VALIDATION.md`).

## Not yet exposed

`label_propagation_init_recursive` (the standalone recursive
label-propagation init step) and the raw multi-layer `evoc_clusters` (all
layers, not just the best) are implemented in the Rust core but not given
their own flat wasm export yet -- `wasm_evoc_fit_predict` covers the
end-to-end path phase 1-9 fixtures need; a caller wanting the intermediate
per-layer or label-prop-only data should add a thin wrapper following the
same pattern as the functions above.
