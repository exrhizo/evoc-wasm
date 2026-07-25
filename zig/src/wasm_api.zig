//! Flat C-ABI wasm32-freestanding entry points for the Zig `evoc` port.
//!
//! This file is the ONLY thing built for wasm32-freestanding (see the
//! `wasm` step in build.zig). It wraps the existing phase 1-8 core modules
//! (root.zig / clustering.zig etc) -- no core algorithm logic is
//! duplicated or modified here, only marshalling between wasm linear-memory
//! flat buffers and the native Zig slice-based APIs.
//!
//! ## Memory model
//!
//! Freestanding wasm has no OS allocator. We use `std.heap.wasm_allocator`
//! (`std.heap.WasmAllocator`), which is available in Zig 0.13 for the
//! `wasm32-freestanding` target and manages the wasm linear memory itself
//! via `@wasmMemoryGrow` -- no imports are required from the JS host for
//! memory management. All `export fn`s below funnel through this single
//! global allocator instance.
//!
//! JS callers:
//!   1. call `alloc(len)` to get a pointer into wasm linear memory,
//!   2. write input bytes at that pointer (via a `Uint8Array`/`Float32Array`
//!      view over `memory.buffer`),
//!   3. call the relevant `export fn`, passing pointer+length args,
//!   4. read results back out of linear memory at the pointer(s) the
//!      function documents (either an out-param the caller allocated, or a
//!      pointer returned by the function that must later be `dealloc`ed),
//!   5. call `dealloc(ptr, len)` on every buffer once done with it.
//!
//! There is no exported `_start`/entry point (`-fno-entry`, see build.zig)
//! and no panic-handling import is required: freestanding Zig's default
//! panic handler for wasm calls `@trap` (an `unreachable` trap instruction)
//! which needs no host import. The only thing a JS host must supply is the
//! `env` import object documented per-function below -- currently NONE of
//! the exported functions require any `env` imports; the produced module
//! only exports `memory` and the functions in this file.
//!
//! ## Scope
//!
//! Exposes: alloc/dealloc, distance kernels, a full boruvka MST
//! (kdtree-build + parallel_boruvka fused into one call, since the kdtree
//! is not independently useful to a wasm caller), cluster-layers extraction
//! from a precomputed embedding (`build_cluster_layers`), and the full
//! end-to-end `evoc_clusters` / `EVoC.fit_predict` pipeline (which
//! internally exercises knn_graph, graph_construction, label_propagation
//! and node_embedding -- those are not separately exposed as standalone
//! wasm entry points because their native signatures take CSR graphs /
//! intermediate RNG state that has no useful flat-buffer representation on
//! its own; `evoc_fit_predict` is the intended way to exercise that part of
//! the pipeline from JS).
//!
//! See WASM_API.md at the repo root of zig/ for the full function list.

const std = @import("std");
const core = @import("root.zig");

const allocator = std.heap.wasm_allocator;

// ---------------------------------------------------------------------------
// Memory management
// ---------------------------------------------------------------------------

/// Allocate `len` bytes in wasm linear memory and return a pointer to them.
/// Returns null (address 0) on allocation failure.
export fn alloc(len: usize) ?[*]u8 {
    const mem = allocator.alloc(u8, len) catch return null;
    return mem.ptr;
}

/// Free a buffer previously returned by `alloc` (or by any function in this
/// file documented as returning an owned/allocated buffer). `len` must be
/// the same length originally allocated.
export fn dealloc(ptr: [*]u8, len: usize) void {
    allocator.free(ptr[0..len]);
}

// ---------------------------------------------------------------------------
// Distance kernels (distance.zig)
// ---------------------------------------------------------------------------

/// fast_cosine(x, y) over two f32 vectors of length `len` (in elements),
/// pointers into wasm linear memory. Matches distance.zig's `fast_cosine`
/// exactly (including its `-f32_tiny` fallback when dot <= 0).
export fn distance_fast_cosine(x_ptr: [*]const f32, y_ptr: [*]const f32, len: usize) f32 {
    return core.distance.fast_cosine(x_ptr[0..len], y_ptr[0..len]);
}

/// fast_bit_jaccard(x, y) over two u8-packed-bit vectors of length `len`
/// bytes each. Matches distance.zig's `fast_bit_jaccard`.
export fn distance_fast_bit_jaccard(x_ptr: [*]const u8, y_ptr: [*]const u8, len: usize) f32 {
    return core.distance.fast_bit_jaccard(x_ptr[0..len], y_ptr[0..len]);
}

// ---------------------------------------------------------------------------
// Boruvka mutual-reachability MST (kdtree.zig + boruvka.zig)
// ---------------------------------------------------------------------------

/// Number of MST edges `boruvka_mst` will write for `n_samples` points
/// (always `n_samples - 1`; call this first to size your output buffers
/// before calling `boruvka_mst`). Returns 0 if `n_samples == 0`.
export fn boruvka_mst_edge_count(n_samples: usize) usize {
    if (n_samples == 0) return 0;
    return n_samples - 1;
}

/// Builds a kdtree over `data` (row-major `n_samples x n_features`, f32) and
/// runs `parallel_boruvka` (single-threaded, `n_threads=1`) to compute the
/// mutual-reachability minimum spanning tree, matching
/// `boruvka.parallel_boruvka` fused with `kdtree.build_kdtree`.
///
/// Writes `n_samples - 1` edges into the three caller-provided output
/// buffers (`out_left`/`out_right` as f32 point indices -- matching the
/// native `Edges.point_a`/`point_b` representation used elsewhere in this
/// port -- and `out_dist` as f32 mutual-reachability distances, already
/// sqrt-transformed). Buffers must be pre-sized to
/// `boruvka_mst_edge_count(n_samples)` elements (4 bytes each).
///
/// `leaf_size` mirrors `kdtree.build_kdtree`'s leaf_size parameter (a
/// reasonable default is 40). `reproducible` selects
/// `boruvka_tree_query_reproducible` over the plain query path, matching
/// `parallel_boruvka`'s `reproducible` flag.
///
/// Returns 0 on success, or a negative error code:
///   -1: n_samples < 2 (no MST possible)
///   -2: allocation or internal failure
export fn boruvka_mst(
    data_ptr: [*]const f32,
    n_samples: usize,
    n_features: usize,
    min_samples: usize,
    leaf_size: i64,
    reproducible: bool,
    out_left: [*]f32,
    out_right: [*]f32,
    out_dist: [*]f32,
) i32 {
    if (n_samples < 2) return -1;
    const data = data_ptr[0 .. n_samples * n_features];

    var tree = core.kdtree.build_kdtree(allocator, data, n_features, @intCast(leaf_size)) catch return -2;
    defer tree.deinit();

    var edges = core.boruvka.parallel_boruvka(allocator, &tree, 1, min_samples, reproducible) catch return -2;
    defer edges.deinit();

    const m = edges.point_a.len;
    for (0..m) |i| {
        out_left[i] = @floatFromInt(edges.point_a[i]);
        out_right[i] = @floatFromInt(edges.point_b[i]);
        out_dist[i] = edges.distance[i];
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Cluster layers from a precomputed embedding (clustering.zig)
// ---------------------------------------------------------------------------

/// Runs `clustering.build_cluster_layers` on a precomputed embedding
/// (row-major `n_samples x n_features`, f32) and writes ONLY the best
/// layer's labels (the layer with the highest sort-key -- i.e. layer 0
/// after build_cluster_layers's own descending-by-cluster-count sort,
/// matching `EVoC.fit_predict`'s "first" convention when a caller wants a
/// single flat answer) into `out_labels` (caller-allocated, `n_samples`
/// x i64 = 8 bytes each). Also writes matching membership strengths (f32,
/// `n_samples` x 4 bytes) into `out_strengths`.
///
/// Returns the number of cluster layers computed on success (>= 1), or a
/// negative error code on failure (-2: allocation/internal failure).
///
/// NOTE: this only surfaces the base (layer-0) result; a caller wanting
/// ALL layers (as `EVoC.fit_predict`'s persistence-argmax selection does)
/// should use `evoc_fit_predict` instead, which does that selection
/// natively. Exposing the full jagged `ClusterLayers` structure over a flat
/// C ABI was judged not worth the complexity for this pass.
export fn cluster_layers_base(
    data_ptr: [*]const f32,
    n_samples: usize,
    n_features: usize,
    min_samples: usize,
    base_min_cluster_size: i64,
    reproducible_flag: bool,
    min_similarity_threshold: f64,
    max_layers: usize,
    out_labels: [*]i64,
    out_strengths: [*]f32,
) i32 {
    const data = data_ptr[0 .. n_samples * n_features];
    var layers = core.clustering.build_cluster_layers(
        allocator,
        data,
        n_features,
        min_samples,
        base_min_cluster_size,
        reproducible_flag,
        min_similarity_threshold,
        max_layers,
    ) catch return -2;
    defer layers.deinit();

    const base_labels = layers.cluster_layers[0];
    const base_strengths = layers.membership_strength_layers[0];
    for (0..n_samples) |i| {
        out_labels[i] = base_labels[i];
        out_strengths[i] = base_strengths[i];
    }
    return @intCast(layers.cluster_layers.len);
}

// ---------------------------------------------------------------------------
// End-to-end EVoC.fit_predict (clustering.zig)
// ---------------------------------------------------------------------------

/// Runs the full `evoc_clusters` pipeline end-to-end (knn_graph ->
/// graph_construction -> label_propagation_init_recursive ->
/// node_embedding_repr_self_driven -> build_cluster_layers) followed by
/// `EVoC.fit_predict`'s persistence-argmax best-layer selection, matching
/// `clustering.EVoC.fit_predict` exactly.
///
/// `data_ptr` is row-major `n_samples x n_features` f32. `out_labels` must
/// be caller-allocated to `n_samples` x i64 (8 bytes each) and receives the
/// selected layer's cluster labels (-1 = noise, matching the reference).
///
/// Option arguments mirror `clustering.EvocClustersOptions` field-for-field
/// (see that struct's doc comment for defaults / semantics);
/// `node_embedding_dim_or_zero == 0` means "use the default derivation"
/// (`EvocClustersOptions.node_embedding_dim = null`), matching Zig's
/// optional-as-sentinel convention over a flat C ABI.
///
/// Returns 0 on success, -2 on allocation/internal failure.
export fn evoc_fit_predict(
    data_ptr: [*]const f32,
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
    node_embedding_dim_or_zero: usize,
    out_labels: [*]i64,
) i32 {
    const data = data_ptr[0 .. n_samples * n_features];
    const opts = core.clustering.EvocClustersOptions{
        .noise_level = noise_level,
        .base_min_cluster_size = base_min_cluster_size,
        .n_neighbors = n_neighbors,
        .min_samples = min_samples,
        .n_epochs = n_epochs,
        .neighbor_scale = neighbor_scale,
        .min_similarity_threshold = min_similarity_threshold,
        .max_layers = max_layers,
        .n_label_prop_iter = n_label_prop_iter,
        .node_embedding_dim = if (node_embedding_dim_or_zero == 0) null else node_embedding_dim_or_zero,
    };

    var evoc = core.clustering.EVoC.init(allocator, seed, opts);
    defer evoc.deinit();

    const labels = evoc.fit_predict(data, n_features) catch return -2;
    for (0..n_samples) |i| out_labels[i] = labels[i];
    return 0;
}
