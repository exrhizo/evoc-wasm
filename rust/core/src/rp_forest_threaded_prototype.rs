//! PROTOTYPE, wasm32-only, NOT part of the default single-threaded pipeline.
//!
//! Additive, opt-in experiment: exposes a single-tree RP-forest builder so a
//! JS harness can build the `n_trees` independent RP-trees of `make_forest`
//! (see `knn_graph.rs::make_forest` / `nndescent_float.rs::make_float_forest`)
//! in parallel across `worker_threads` Workers, each with its own wasm
//! instance, then merge the per-tree leaf arrays exactly as `make_forest`
//! does today (pad to max leaf width, vstack).
//!
//! Why this is safe to parallelize (verified by reading `nndescent_float.rs`
//! before writing this file, not assumed): `make_float_forest_*` calls
//! `make_float_leaf_array_serial(data, n_features, &mut rng_states[i], ...)`
//! once per tree. `data` is read-only and shared; `rng_states[i]` is a
//! private `[i64; 3]` per tree with no interaction with any other tree's
//! state; `make_float_tree`'s recursion only ever touches `indices`/
//! `point_indices` local to that one call. There is no shared mutable state
//! across trees during construction -- the only cross-tree step is the
//! *post*-construction pad-and-stack in `make_forest`, which this file's
//! design defers back to the (single-threaded, unmodified) merge step.
//!
//! This module does NOT touch `wasm_api.rs`'s existing exports, does NOT
//! change `knn_graph.rs`/`nndescent_float.rs`, and is only compiled for
//! `target_arch = "wasm32"` (see the `#[cfg]` on its `lib.rs` declaration),
//! so `cargo test --workspace` (native) never sees it and the existing
//! single-threaded wasm build is unaffected unless a caller explicitly
//! invokes these new exports.

use std::alloc::{alloc, dealloc, Layout};

use crate::nndescent_float::make_float_leaf_array_serial;
use crate::rng::RngState;

const ALIGN: usize = 16;

fn layout_for(len: usize) -> Layout {
    Layout::from_size_align(len.max(1), ALIGN).expect("valid alloc layout")
}

/// Same alloc/dealloc protocol as `wasm_api.rs` (kept separate here so this
/// prototype has no dependency on that file). Reserve `len` bytes.
#[no_mangle]
pub extern "C" fn wasm_proto_alloc(len: usize) -> *mut u8 {
    if len == 0 {
        return std::ptr::null_mut();
    }
    unsafe { alloc(layout_for(len)) }
}

/// Free a buffer obtained from `wasm_proto_alloc` with the same `len`.
#[no_mangle]
pub unsafe extern "C" fn wasm_proto_dealloc(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    dealloc(ptr, layout_for(len));
}

/// Build exactly ONE RP-tree (one row of `make_forest`'s `rng_states`) and
/// write its flat, `-1`-padded `(n_leaves * leaf_width)` `i32` leaf array
/// into `out_ptr` (caller must size the buffer with
/// `wasm_proto_leaf_capacity(n_samples, leaf_size)` i32 elements -- an
/// upper bound, since the true leaf count/width isn't known until the tree
/// is built).
///
/// `rng_state` is passed as three `i64` words (matching `RngState = [i64;
/// 3]`) since wasm's ABI can't pass arrays; this mirrors one row of the
/// reference's `rng_states[i]`.
///
/// Returns `n_leaves * leaf_width_actual` packed into a single `i64`:
/// high 32 bits = `leaf_width_actual`, low 32 bits = `n_leaves`. (Kept as a
/// single return value since wasm `extern "C"` functions return one
/// scalar; JS unpacks it.) Bit-for-bit identical to what `make_forest`
/// would produce for this tree if run single-threaded with the same
/// `rng_state`, `data`, `n_features`, `leaf_size`, `max_depth` -- this
/// function is a thin wrapper around the exact same
/// `make_float_leaf_array_serial` the single-threaded path calls, just
/// exposed per-tree instead of per-forest.
#[no_mangle]
pub unsafe extern "C" fn wasm_proto_build_one_tree(
    data_ptr: *const f32,
    data_len: usize,
    n_features: usize,
    rng0: i64,
    rng1: i64,
    rng2: i64,
    leaf_size: usize,
    max_depth: i64,
    out_ptr: *mut i32,
    out_capacity: usize,
) -> i64 {
    let data = std::slice::from_raw_parts(data_ptr, data_len);
    let mut rng_state: RngState = [rng0, rng1, rng2];

    let (flat, width) =
        make_float_leaf_array_serial(data, n_features, &mut rng_state, leaf_size, max_depth);

    if width == 0 || flat.len() > out_capacity {
        return -1;
    }

    let n_leaves = flat.len() / width;
    let out = std::slice::from_raw_parts_mut(out_ptr, flat.len());
    out.copy_from_slice(&flat);

    ((width as i64) << 32) | (n_leaves as i64)
}

/// Upper bound (in i32 elements) on the buffer `wasm_proto_build_one_tree`
/// needs: worst case every point ends up in its own leaf array row with
/// width up to `leaf_size` (matches `make_float_leaf_array_impl`'s own
/// `max_leaf_size = max(leaf_size, max points in any single leaf)`, and
/// `n_leaves <= n_samples` since a leaf always holds >=1 point once
/// `max_depth` truncation is considered). We size generously
/// (`n_samples * leaf_size`) rather than trying to bound it tightly, since
/// this is a prototype and correctness (never under-sized) matters more
/// than exactness here.
#[no_mangle]
pub extern "C" fn wasm_proto_leaf_capacity(n_samples: usize, leaf_size: usize) -> usize {
    n_samples * leaf_size.max(1)
}
