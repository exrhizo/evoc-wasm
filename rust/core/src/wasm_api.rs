//! Flat `extern "C"` wasm32 API surface over `evoc-core`.
//!
//! Only compiled for `target_arch = "wasm32"` (see the `#[cfg]` on the
//! module declaration in `lib.rs`) so native `cargo test --workspace` is
//! completely unaffected by anything in this file.
//!
//! # Calling convention
//!
//! wasm functions cannot take or return Rust slices/Vecs/structs directly --
//! only numeric scalars (i32/i64/f32/f64) cross the wasm ABI boundary. The
//! pattern used throughout this file is the standard "caller-controlled
//! buffer" protocol:
//!
//! 1. JS calls [`wasm_alloc`] to reserve `len` bytes of wasm linear memory
//!    and gets back a pointer (offset into the wasm memory `ArrayBuffer`).
//! 2. JS writes its input data (e.g. `Float32Array`) directly into wasm
//!    memory at that offset via a typed-array view.
//! 3. JS calls the real exported function, passing pointers (as `usize`/
//!    `u32`) and lengths instead of arrays.
//! 4. For outputs, JS pre-allocates an output buffer the same way (the
//!    required output size is documented per-function below -- it is always
//!    a simple function of the input sizes, so JS can compute it before
//!    calling) and passes its pointer in; the Rust side writes results into
//!    it in place. Functions that can fail return an `i32` status code (`0`
//!    = success, non-zero = error) rather than panicking across the ABI
//!    boundary.
//! 5. JS reads the output buffer back out via a typed-array view.
//! 6. JS calls [`wasm_dealloc`] on every pointer it allocated once done.
//!
//! All floating point element types are `f32` (matching the Rust core's
//! native float type throughout the port) unless documented otherwise.
//! Index/label buffers are `i32` (widened/narrowed from the core's
//! `i64`/`usize`/`i32` types at the boundary -- safe for the cluster/point
//! counts this library is designed for).
//!
//! No panics should cross the ABI boundary in normal operation; malformed
//! inputs (e.g. buffer-length mismatches) are the caller's responsibility,
//! matching the "trusted embedder" model used by this project's Zig target.

use std::alloc::{alloc, dealloc, Layout};

use crate::boruvka::parallel_boruvka;
use crate::cluster_trees::{
    condense_tree, extract_leaves, get_cluster_label_vector, get_point_membership_strength_vector,
    mst_to_linkage_tree,
};
use crate::clustering::{
    build_cluster_layers, evoc_clusters, BuildClusterLayersOptions, EvocClustersOptions,
};
use crate::distance::{fast_bit_jaccard, fast_cosine, fast_int_inner_product_dissimilarity};
use crate::kdtree::build_kdtree;
use crate::knn_graph::knn_graph as knn_graph_impl;
use crate::node_embedding::node_embedding_repr_self_driven;
use crate::rng::{tau_rand_int, RngState};

const ALIGN: usize = 16;

fn layout_for(len: usize) -> Layout {
    // ALIGN is a small power of two, so this never fails for realistic len.
    Layout::from_size_align(len.max(1), ALIGN).expect("valid alloc layout")
}

/// Reserve `len` bytes of wasm linear memory and return a pointer to it.
/// The memory is uninitialized -- callers must fully overwrite it (or only
/// read what they wrote) before use. Pair with [`wasm_dealloc`] using the
/// *same* `len`.
#[no_mangle]
pub extern "C" fn wasm_alloc(len: usize) -> *mut u8 {
    if len == 0 {
        return std::ptr::null_mut();
    }
    unsafe { alloc(layout_for(len)) }
}

/// Free memory previously returned by [`wasm_alloc`]. `len` MUST match the
/// `len` passed to the original `wasm_alloc` call (the allocator layout is
/// reconstructed from it, same as `Box`/`Vec`'s own drop glue).
///
/// # Safety
/// `ptr` must be a pointer previously returned by [`wasm_alloc`] (or null),
/// not already freed, and `len` must match the original `wasm_alloc` call.
#[no_mangle]
pub unsafe extern "C" fn wasm_dealloc(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    dealloc(ptr, layout_for(len))
}

// ---------------------------------------------------------------------------
// helpers: raw pointer <-> slice, not exported
// ---------------------------------------------------------------------------

unsafe fn slice_f32<'a>(ptr: *const f32, len: usize) -> &'a [f32] {
    if len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, len)
    }
}

unsafe fn slice_i8<'a>(ptr: *const i8, len: usize) -> &'a [i8] {
    if len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, len)
    }
}

unsafe fn slice_u8<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    if len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, len)
    }
}

unsafe fn slice_u32<'a>(ptr: *const u32, len: usize) -> &'a [u32] {
    if len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, len)
    }
}

unsafe fn write_f32(dst: *mut f32, data: &[f32]) {
    if !data.is_empty() {
        std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
    }
}

unsafe fn write_i32(dst: *mut i32, data: &[i32]) {
    if !data.is_empty() {
        std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
    }
}

// ---------------------------------------------------------------------------
// distance kernels (distance.rs)
// ---------------------------------------------------------------------------

/// Cosine dissimilarity between two length-`len` f32 vectors at `x_ptr`/
/// `y_ptr`. See `distance::fast_cosine`.
/// # Safety
/// Pointer arguments must be valid and point at buffers of at least the
/// documented length/count, per this module's calling convention (see the
/// module doc comment above).
#[no_mangle]
pub unsafe extern "C" fn wasm_fast_cosine(x_ptr: *const f32, y_ptr: *const f32, len: usize) -> f32 {
    fast_cosine(slice_f32(x_ptr, len), slice_f32(y_ptr, len))
}

/// Bit-Jaccard dissimilarity between two length-`len` u8 (bit-packed) vectors.
/// See `distance::fast_bit_jaccard`.
/// # Safety
/// Pointer arguments must be valid and point at buffers of at least the
/// documented length/count, per this module's calling convention (see the
/// module doc comment above).
#[no_mangle]
pub unsafe extern "C" fn wasm_fast_bit_jaccard(
    x_ptr: *const u8,
    y_ptr: *const u8,
    len: usize,
) -> f32 {
    fast_bit_jaccard(slice_u8(x_ptr, len), slice_u8(y_ptr, len))
}

/// Int8 inner-product dissimilarity between two length-`len` i8 vectors.
/// See `distance::fast_int_inner_product_dissimilarity`.
/// # Safety
/// Pointer arguments must be valid and point at buffers of at least the
/// documented length/count, per this module's calling convention (see the
/// module doc comment above).
#[no_mangle]
pub unsafe extern "C" fn wasm_fast_int_inner_product_dissimilarity(
    x_ptr: *const i8,
    y_ptr: *const i8,
    len: usize,
) -> f32 {
    fast_int_inner_product_dissimilarity(slice_i8(x_ptr, len), slice_i8(y_ptr, len))
}

// ---------------------------------------------------------------------------
// knn_graph (knn_graph.rs)
// ---------------------------------------------------------------------------

/// Approximate kNN graph over `n_samples` x `n_features` row-major f32 data
/// (cosine/float NN-descent path -- see `knn_graph::knn_graph`).
///
/// `out_indices_ptr`/`out_dists_ptr` must each point at a caller-allocated
/// buffer of `n_samples * n_neighbors` elements (i32 / f32 respectively).
/// `seed` drives every RNG draw (RP-forest construction + NN-descent
/// candidate sampling) via this project's own `tau_rand` RNG. Single
/// threaded (`n_threads = 1`) -- wasm32-unknown-unknown has no threads.
///
/// Returns `0` on success.
/// # Safety
/// Pointer arguments must be valid and point at buffers of at least the
/// documented length/count, per this module's calling convention (see the
/// module doc comment above).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn wasm_knn_graph(
    data_ptr: *const f32,
    n_samples: usize,
    n_features: usize,
    n_neighbors: usize,
    leaf_size: usize,
    n_iters: usize,
    effective_max_candidates: usize,
    delta: f64,
    seed: i64,
    out_indices_ptr: *mut i32,
    out_dists_ptr: *mut f32,
) -> i32 {
    let data = slice_f32(data_ptr, n_samples * n_features);

    let mut state: RngState = [0; 3];
    crate::rng::seed(&mut state, seed);
    let draw = |state: &mut RngState| -> RngState {
        [
            tau_rand_int(state) as i64,
            tau_rand_int(state) as i64,
            tau_rand_int(state) as i64,
        ]
    };
    let n_trees = 4usize;
    let rng_state = draw(&mut state);
    let mut rng_states: Vec<RngState> = (0..n_trees).map(|_| draw(&mut state)).collect();

    let (indices, dists) = knn_graph_impl(
        data,
        n_features,
        n_neighbors,
        &rng_state,
        &mut rng_states,
        leaf_size,
        200,
        effective_max_candidates,
        n_iters,
        delta,
        Some(delta),
        true,
        1,
        false,
    );

    write_i32(out_indices_ptr, &indices);
    write_f32(out_dists_ptr, &dists);
    0
}

// ---------------------------------------------------------------------------
// boruvka MST (boruvka.rs)
// ---------------------------------------------------------------------------

/// Mutual-reachability MST over `n_samples` x `n_features` row-major f32
/// data (builds a kd-tree internally, `leaf_size = 40` matching the
/// reference default). See `boruvka::parallel_boruvka`.
///
/// `out_edges_ptr` must point at a caller-allocated buffer of
/// `(n_samples - 1) * 3` f32 elements: flat row-major `(point_a, point_b,
/// distance)` triples. Returns the number of edges written (`n_samples -
/// 1`) on success, or a negative value on error (`n_samples < 2`).
/// # Safety
/// Pointer arguments must be valid and point at buffers of at least the
/// documented length/count, per this module's calling convention (see the
/// module doc comment above).
#[no_mangle]
pub unsafe extern "C" fn wasm_boruvka_mst(
    data_ptr: *const f32,
    n_samples: usize,
    n_features: usize,
    min_samples: usize,
    out_edges_ptr: *mut f32,
) -> i32 {
    if n_samples < 2 {
        return -1;
    }
    let data = slice_f32(data_ptr, n_samples * n_features);
    let tree = build_kdtree(data, n_features, 40);
    let edges = parallel_boruvka(&tree, 1, min_samples, false);
    let flat: Vec<f32> = edges.iter().flat_map(|e| e.iter().copied()).collect();
    write_f32(out_edges_ptr, &flat);
    edges.len() as i32
}

// ---------------------------------------------------------------------------
// cluster trees (cluster_trees.rs)
// ---------------------------------------------------------------------------

/// MST -> linkage tree -> condensed tree -> excess-of-mass labels +
/// membership strengths, in one call (composes `mst_to_linkage_tree`,
/// `condense_tree`, `extract_leaves`, `get_cluster_label_vector`,
/// `get_point_membership_strength_vector`).
///
/// `mst_edges_ptr` points at `n_edges * 3` f32 elements (same layout as
/// [`wasm_boruvka_mst`]'s output; `n_edges` is normally `n_samples - 1`,
/// but this function does not require it be sorted by weight -- it sorts
/// internally, matching the reference pipeline).
///
/// `out_labels_ptr` / `out_strengths_ptr` must each point at a
/// caller-allocated buffer of `n_samples` elements (i32 / f32
/// respectively). Returns `0` on success.
/// # Safety
/// Pointer arguments must be valid and point at buffers of at least the
/// documented length/count, per this module's calling convention (see the
/// module doc comment above).
#[no_mangle]
pub unsafe extern "C" fn wasm_cluster_tree_labels(
    mst_edges_ptr: *const f32,
    n_edges: usize,
    n_samples: usize,
    min_cluster_size: i64,
    out_labels_ptr: *mut i32,
    out_strengths_ptr: *mut f32,
) -> i32 {
    let flat = slice_f32(mst_edges_ptr, n_edges * 3);
    let mut edges: Vec<[f32; 3]> = flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    edges.sort_by(|a, b| a[2].partial_cmp(&b[2]).unwrap());

    let linkage = mst_to_linkage_tree(&edges);
    let condensed = condense_tree(&linkage, min_cluster_size);
    let leaves = extract_leaves(&condensed);
    let labels = get_cluster_label_vector(&condensed, &leaves, 0.0, n_samples);
    let strengths = get_point_membership_strength_vector(&condensed, &leaves, &labels);

    let labels_i32: Vec<i32> = labels.iter().map(|&x| x as i32).collect();
    write_i32(out_labels_ptr, &labels_i32);
    write_f32(out_strengths_ptr, &strengths);
    0
}

// ---------------------------------------------------------------------------
// node embedding (node_embedding.rs)
// ---------------------------------------------------------------------------

/// Self-driven node embedding (SGD layout, `node_embedding_repr_self_driven`)
/// over a CSR affinity graph of `n_vertices` vertices.
///
/// `graph_indptr_ptr` (`n_vertices + 1` u32), `graph_indices_ptr` /
/// `graph_data_ptr` (`nnz` u32 / f32, `nnz = graph_indptr[n_vertices]`).
/// `initial_embedding_ptr` may be null (0) to request a random PCA-style
/// init driven by `seed`, or point at `n_vertices * n_components` f32 to
/// seed the layout explicitly.
///
/// `out_embedding_ptr` must point at a caller-allocated buffer of
/// `n_vertices * n_components` f32. Returns `0` on success.
/// # Safety
/// Pointer arguments must be valid and point at buffers of at least the
/// documented length/count, per this module's calling convention (see the
/// module doc comment above).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn wasm_node_embedding(
    graph_indptr_ptr: *const u32,
    graph_indices_ptr: *const u32,
    graph_data_ptr: *const f32,
    n_vertices: usize,
    nnz: usize,
    n_components: usize,
    n_epochs: usize,
    initial_embedding_ptr: *const f32,
    initial_alpha: f32,
    negative_sample_rate: f32,
    noise_level: f32,
    seed: i64,
    out_embedding_ptr: *mut f32,
) -> i32 {
    let indptr = slice_u32(graph_indptr_ptr, n_vertices + 1);
    let indices = slice_u32(graph_indices_ptr, nnz);
    let data = slice_f32(graph_data_ptr, nnz);
    let initial: Option<&[f32]> = if initial_embedding_ptr.is_null() {
        None
    } else {
        Some(slice_f32(initial_embedding_ptr, n_vertices * n_components))
    };

    let mut state: RngState = [0; 3];
    crate::rng::seed(&mut state, seed);

    let embedding = node_embedding_repr_self_driven(
        indptr,
        indices,
        data,
        n_vertices,
        n_components,
        n_epochs,
        initial,
        initial_alpha,
        negative_sample_rate,
        noise_level,
        &mut state,
    );

    write_f32(out_embedding_ptr, &embedding);
    0
}

// ---------------------------------------------------------------------------
// build_cluster_layers (clustering.rs, phase 8) -- best layer only
// ---------------------------------------------------------------------------

/// [`build_cluster_layers`] over an already-computed `n_samples` x
/// `n_features` row-major f32 embedding, collapsed to the single
/// highest-persistence layer (same `argmax` rule `EVoC::fit_predict` uses)
/// so the flat C ABI doesn't need to describe a variable-length list of
/// variable-length layers.
///
/// `out_labels_ptr` / `out_strengths_ptr` must each point at a
/// caller-allocated buffer of `n_samples` elements (i32 / f32
/// respectively). Returns the number of layers `build_cluster_layers`
/// internally produced (>= 1) on success, or a negative value on error.
/// # Safety
/// Pointer arguments must be valid and point at buffers of at least the
/// documented length/count, per this module's calling convention (see the
/// module doc comment above).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn wasm_build_cluster_layers_best(
    data_ptr: *const f32,
    n_samples: usize,
    n_features: usize,
    min_samples: usize,
    base_min_cluster_size: i64,
    min_similarity_threshold: f64,
    max_layers: usize,
    out_labels_ptr: *mut i32,
    out_strengths_ptr: *mut f32,
) -> i32 {
    if n_samples < 2 {
        return -1;
    }
    let data = slice_f32(data_ptr, n_samples * n_features);
    let opts = BuildClusterLayersOptions {
        min_samples,
        base_min_cluster_size,
        base_n_clusters: None,
        reproducible_flag: false,
        min_similarity_threshold,
        max_layers,
    };
    let layers = build_cluster_layers(data, n_features, &opts);
    if layers.cluster_layers.is_empty() {
        return -2;
    }
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in layers.persistence_scores.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    let labels_i32: Vec<i32> = layers.cluster_layers[best]
        .iter()
        .map(|&x| x as i32)
        .collect();
    write_i32(out_labels_ptr, &labels_i32);
    write_f32(out_strengths_ptr, &layers.membership_strength_layers[best]);
    layers.cluster_layers.len() as i32
}

// ---------------------------------------------------------------------------
// evoc_clusters / EVoC::fit_predict, end to end (clustering.rs, phase 9)
// ---------------------------------------------------------------------------

/// Full `evoc_clusters` pipeline (knn_graph -> neighbor_graph_matrix ->
/// label_propagation_init -> node_embedding -> build_cluster_layers),
/// collapsed to the single highest-persistence layer -- i.e. this is
/// `EVoC::fit_predict`, not raw `evoc_clusters` (see `clustering.rs`'s
/// module doc for why layer selection is needed at all).
///
/// `data_ptr` is `n_samples * n_features` row-major f32 (raw, NOT
/// pre-normalized -- this function unit-normalizes internally, matching the
/// reference). `node_embedding_dim` of `0` means "use the default
/// `clamp(n_neighbors/4, 4, 15)` derivation" (the reference's `None`); pass
/// a positive value to override.
///
/// `out_labels_ptr` / `out_strengths_ptr` must each point at a
/// caller-allocated buffer of `n_samples` elements (i32 / f32
/// respectively). Returns the number of internal cluster layers produced
/// (>= 1, informational) on success, or a negative value on error.
/// # Safety
/// Pointer arguments must be valid and point at buffers of at least the
/// documented length/count, per this module's calling convention (see the
/// module doc comment above).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn wasm_evoc_fit_predict(
    data_ptr: *const f32,
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
    node_embedding_dim: usize,
    out_labels_ptr: *mut i32,
    out_strengths_ptr: *mut f32,
) -> i32 {
    if n_samples < 2 {
        return -1;
    }
    let data = slice_f32(data_ptr, n_samples * n_features);
    let opts = EvocClustersOptions {
        noise_level,
        base_min_cluster_size,
        n_neighbors,
        min_samples,
        n_epochs,
        neighbor_scale,
        min_similarity_threshold,
        max_layers,
        n_label_prop_iter,
        node_embedding_dim: if node_embedding_dim == 0 {
            None
        } else {
            Some(node_embedding_dim)
        },
    };

    let (layers, _nn_inds, _nn_dists) = evoc_clusters(data, n_features, seed, &opts);
    if layers.cluster_layers.is_empty() {
        return -2;
    }

    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in layers.persistence_scores.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }

    let labels_i32: Vec<i32> = layers.cluster_layers[best]
        .iter()
        .map(|&x| x as i32)
        .collect();
    write_i32(out_labels_ptr, &labels_i32);
    write_f32(out_strengths_ptr, &layers.membership_strength_layers[best]);
    layers.cluster_layers.len() as i32
}
