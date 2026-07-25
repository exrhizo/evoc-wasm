//! Port of `evoc/float_nndescent.py`: RP-tree forest construction and
//! NN-descent for f32 data with (negative) cosine similarity.
//! `fast_cosine` lives in `distance.rs`; the heap machinery in
//! `nnd_heap.rs`; `tau_rand*` in `rng.rs`.
//!
//! PORT NOTE: every `numba.prange` is a plain sequential loop here, but all
//! `n_threads`-driven block/bucket structure is preserved exactly -- block
//! boundaries and per-thread buffer offsets determine push order and thus
//! output parity (fixtures use n_threads=1).
//!
//! PORT NOTE: as in `build_candidates`, `rng_state + idx` / `rng_state + i`
//! in the reference are numpy array-plus-scalar expressions that allocate a
//! COPY; draws from those copies never advance the caller's state. Only
//! `float_random_projection_split`'s direct draws (the two anchor picks and
//! the degenerate-case reassignment) mutate the passed-in state.

use crate::distance::fast_cosine;
use crate::nnd_heap::{
    apply_graph_update_array, apply_sorted_graph_updates, deheap_sort, flagged_heap_push,
    make_heap, NNDHeap,
};
use crate::rng::{tau_rand_int, RngState};

/// Used for a floating point "nearly zero" comparison (f64, as in the
/// reference where numba promotes the f32 operand to f64 for the compare).
const EPS: f64 = 1e-8;

#[inline]
fn data_row(data: &[f32], n_features: usize, i: usize) -> &[f32] {
    &data[i * n_features..(i + 1) * n_features]
}

/// Port of `float_random_projection_split`.
///
/// PORT NOTE: `tau_rand_int(...) % m` uses numba's Python-style modulo
/// (result takes the divisor's sign; verified against the reference numba
/// build) -- `rem_euclid` for the positive divisors used here.
pub fn float_random_projection_split(
    data: &[f32],
    n_features: usize,
    indices: &[i32],
    rng_state: &mut RngState,
) -> (Vec<i32>, Vec<i32>) {
    let dim = n_features;

    // Select two random points, set the hyperplane between them
    let indices_size = indices.len() as i32;
    let left_index = tau_rand_int(rng_state).rem_euclid(indices_size);
    let mut right_index = tau_rand_int(rng_state).rem_euclid(indices_size);
    right_index += (left_index == right_index) as i32;
    right_index = right_index.rem_euclid(indices_size);
    let left = indices[left_index as usize] as usize;
    let right = indices[right_index as usize] as usize;
    let left_data = data_row(data, dim, left);
    let right_data = data_row(data, dim, right);

    // Compute the normal vector to the hyperplane (the vector between
    // the two points)
    let mut hyperplane_vector = vec![0.0f32; dim];
    let mut hyperplane_norm: f32 = 0.0;

    for d in 0..dim {
        hyperplane_vector[d] = left_data[d] - right_data[d];
        hyperplane_norm += hyperplane_vector[d] * hyperplane_vector[d];
    }

    hyperplane_norm = hyperplane_norm.sqrt();
    if (hyperplane_norm as f64).abs() < EPS {
        hyperplane_norm = 1.0;
    }

    // Normalize in the same vector (avoiding second loop when possible)
    for d in 0..dim {
        hyperplane_vector[d] /= hyperplane_norm;
    }

    // Use temporary arrays sized for worst case, then trim
    let max_size = indices.len();
    let mut temp_left = vec![0i32; max_size];
    let mut temp_right = vec![0i32; max_size];
    let mut n_left = 0usize;
    let mut n_right = 0usize;

    // Single pass: classify points and directly populate result arrays
    for idx in 0..indices.len() {
        // per-point COPY of the rng state (lanes + idx); see module PORT NOTE
        let mut local_rng_state: RngState = [
            rng_state[0].wrapping_add(idx as i64),
            rng_state[1].wrapping_add(idx as i64),
            rng_state[2].wrapping_add(idx as i64),
        ];
        let point_idx = indices[idx];
        let test_data = data_row(data, dim, point_idx as usize);
        let mut margin: f32 = 0.0;

        // Compute margin (dot product with hyperplane normal)
        for d in 0..dim {
            margin += hyperplane_vector[d] * test_data[d];
        }

        // Classify point and directly assign to appropriate array
        let classification: i8 = if (margin as f64).abs() < EPS {
            tau_rand_int(&mut local_rng_state).rem_euclid(2) as i8
        } else if margin > 0.0 {
            0
        } else {
            1
        };

        if classification == 0 {
            temp_left[n_left] = point_idx;
            n_left += 1;
        } else {
            temp_right[n_right] = point_idx;
            n_right += 1;
        }
    }

    // Handle degenerate case where all points end up on one side
    if n_left == 0 || n_right == 0 {
        n_left = 0;
        n_right = 0;
        // Reassign randomly (this DOES advance the caller's rng_state)
        for idx in 0..indices.len() {
            let point_idx = indices[idx];
            let classification = tau_rand_int(rng_state).rem_euclid(2);
            if classification == 0 {
                temp_left[n_left] = point_idx;
                n_left += 1;
            } else {
                temp_right[n_right] = point_idx;
                n_right += 1;
            }
        }
    }

    // Create final arrays with exact sizes (copy only what we need)
    (temp_left[..n_left].to_vec(), temp_right[..n_right].to_vec())
}

/// Port of `make_float_tree`: recursively split, appending each leaf's
/// point indices to `point_indices` in depth-first left-to-right order.
pub fn make_float_tree(
    data: &[f32],
    n_features: usize,
    indices: &[i32],
    point_indices: &mut Vec<Vec<i32>>,
    rng_state: &mut RngState,
    leaf_size: usize,
    max_depth: i64,
) {
    if indices.len() > leaf_size && max_depth > 0 {
        let (left_indices, right_indices) =
            float_random_projection_split(data, n_features, indices, rng_state);

        make_float_tree(
            data,
            n_features,
            &left_indices,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        );

        make_float_tree(
            data,
            n_features,
            &right_indices,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        );
    } else {
        point_indices.push(indices.to_vec());
    }
}

fn make_float_leaf_array_impl(
    data: &[f32],
    n_features: usize,
    rng_state: &mut RngState,
    leaf_size: usize,
    max_depth: i64,
) -> (Vec<i32>, usize) {
    let n_samples = data.len() / n_features;
    let indices: Vec<i32> = (0..n_samples as i32).collect();

    let mut point_indices: Vec<Vec<i32>> = Vec::new();

    make_float_tree(
        data,
        n_features,
        &indices,
        &mut point_indices,
        rng_state,
        leaf_size,
        max_depth,
    );

    let n_leaves = point_indices.len();
    let mut max_leaf_size = leaf_size;

    for points in point_indices.iter() {
        max_leaf_size = std::cmp::max(max_leaf_size, points.len());
    }

    let mut result = vec![-1i32; n_leaves * max_leaf_size];
    for (i, points) in point_indices.iter().enumerate() {
        result[i * max_leaf_size..i * max_leaf_size + points.len()].copy_from_slice(points);
    }

    (result, max_leaf_size)
}

/// Port of `make_float_leaf_array_parallel`. Returns the flat row-major
/// (n_leaves x max_leaf_size) leaf array (padded with -1) plus the width.
///
/// PORT NOTE: the parallel and serial reference variants differ only in
/// `prange` vs `range` over independent leaves; sequentially they are
/// identical, so both delegate to one implementation here.
pub fn make_float_leaf_array_parallel(
    data: &[f32],
    n_features: usize,
    rng_state: &mut RngState,
    leaf_size: usize,
    max_depth: i64,
) -> (Vec<i32>, usize) {
    make_float_leaf_array_impl(data, n_features, rng_state, leaf_size, max_depth)
}

/// Port of `make_float_leaf_array_serial` (see the parallel variant's
/// PORT NOTE).
pub fn make_float_leaf_array_serial(
    data: &[f32],
    n_features: usize,
    rng_state: &mut RngState,
    leaf_size: usize,
    max_depth: i64,
) -> (Vec<i32>, usize) {
    make_float_leaf_array_impl(data, n_features, rng_state, leaf_size, max_depth)
}

/// Port of `make_float_forest_no_nested_parallelism`: one leaf array per
/// rng_states row (each row is mutated in place, as the reference's 2-D
/// rng_states rows are views).
pub fn make_float_forest_no_nested_parallelism(
    data: &[f32],
    n_features: usize,
    rng_states: &mut [RngState],
    leaf_size: usize,
    max_depth: i64,
) -> Vec<(Vec<i32>, usize)> {
    rng_states
        .iter_mut()
        .map(|rs| make_float_leaf_array_serial(data, n_features, rs, leaf_size, max_depth))
        .collect()
}

/// Port of `make_float_forest_with_nested_parallelism`.
pub fn make_float_forest_with_nested_parallelism(
    data: &[f32],
    n_features: usize,
    rng_states: &mut [RngState],
    leaf_size: usize,
    max_depth: i64,
) -> Vec<(Vec<i32>, usize)> {
    rng_states
        .iter_mut()
        .map(|rs| make_float_leaf_array_parallel(data, n_features, rs, leaf_size, max_depth))
        .collect()
}

/// Port of the `make_float_forest` dispatcher.
///
/// PORT NOTE: the reference dispatches on the ambient
/// `ENABLE_NESTED_PARALLELISM` (threading-layer detection); here it is an
/// explicit parameter. Both paths produce identical results
/// single-threaded.
pub fn make_float_forest(
    data: &[f32],
    n_features: usize,
    rng_states: &mut [RngState],
    leaf_size: usize,
    max_depth: i64,
    enable_nested_parallelism: bool,
) -> Vec<(Vec<i32>, usize)> {
    if enable_nested_parallelism {
        make_float_forest_with_nested_parallelism(
            data, n_features, rng_states, leaf_size, max_depth,
        )
    } else {
        make_float_forest_no_nested_parallelism(data, n_features, rng_states, leaf_size, max_depth)
    }
}

/// Port of `generate_leaf_updates_float`. `updates` is flat
/// (n_threads x updates_cap x 3); `leaf_block` flat (rows x leaf_width).
///
/// PORT NOTE: the reference never bounds `idx` against the updates buffer
/// (boundscheck=False); with large full leaves the buffer computed by
/// `init_rp_tree_float` can theoretically overflow (a latent reference
/// bug). Rust's slice indexing will panic instead of writing out of bounds;
/// no fixture hits this.
#[allow(clippy::too_many_arguments)]
pub fn generate_leaf_updates_float(
    updates: &mut [f32],
    updates_cap: usize,
    n_updates_per_thread: &mut [i32],
    leaf_block: &[i32],
    leaf_width: usize,
    dist_thresholds: &[f32],
    data: &[f32],
    n_features: usize,
    n_threads: usize,
) {
    let block_size = leaf_block.len() / leaf_width;
    let rows_per_thread = (block_size / n_threads) + 1;

    for t in 0..n_threads {
        let mut idx = 0usize;
        for r in 0..rows_per_thread {
            let n = t * rows_per_thread + r;
            if n >= block_size {
                break;
            }

            for i in 0..leaf_width {
                let p = leaf_block[n * leaf_width + i];
                if p < 0 {
                    break;
                }
                let data_p = data_row(data, n_features, p as usize);
                let base = (t * updates_cap + idx) * 3;
                updates[base] = p as f32;
                updates[base + 1] = p as f32;
                updates[base + 2] = -1.0;
                idx += 1;

                // Start from i+1 to skip self-comparison
                for j in (i + 1)..leaf_width {
                    let q = leaf_block[n * leaf_width + j];
                    if q < 0 {
                        break;
                    }

                    let d = fast_cosine(data_p, data_row(data, n_features, q as usize));
                    // Use max for better branch prediction than OR condition
                    let max_threshold =
                        dist_thresholds[p as usize].max(dist_thresholds[q as usize]);
                    if d < max_threshold {
                        let base = (t * updates_cap + idx) * 3;
                        updates[base] = p as f32;
                        updates[base + 1] = q as f32;
                        updates[base + 2] = d;
                        idx += 1;
                    }
                }
            }
        }

        n_updates_per_thread[t] = idx as i32;
    }
}

/// Port of `init_rp_tree_float`: seed the graph heap from RP-tree leaves
/// (including the (p, p, -1.0) self-updates the leaf generator emits).
pub fn init_rp_tree_float(
    data: &[f32],
    n_features: usize,
    current_graph: &mut NNDHeap,
    leaf_array: &[i32],
    leaf_width: usize,
    n_threads: usize,
) {
    let n_leaves = leaf_array.len() / leaf_width;
    let block_size = n_threads * 64;
    let n_blocks = n_leaves / block_size;

    let max_leaf_size = leaf_width;
    // Python: int(block_size * mls * (mls - 1) / (2 * n_threads)) + 1
    // (true division in f64, truncated)
    let updates_per_thread = ((block_size * max_leaf_size * (max_leaf_size - 1)) as f64
        / (2 * n_threads) as f64) as usize
        + 1;
    let mut updates = vec![0.0f32; n_threads * updates_per_thread * 3];
    let mut n_updates_per_thread = vec![0i32; n_threads];
    let n_vertices = current_graph.n_points;
    let vertex_block_size = n_vertices / n_threads + 1;
    let size = current_graph.size;

    for i in 0..(n_blocks + 1) {
        let block_start = i * block_size;
        let block_end = std::cmp::min(n_leaves, (i + 1) * block_size);

        let leaf_block = &leaf_array[block_start * leaf_width..block_end * leaf_width];
        // PORT NOTE: the reference's dist_thresholds is a live column view
        // (current_graph[1][:, 0]); a per-block snapshot copy is equivalent
        // because the generate step never mutates the graph.
        let dist_thresholds: Vec<f32> = (0..n_vertices)
            .map(|v| current_graph.distances[v * size])
            .collect();

        generate_leaf_updates_float(
            &mut updates,
            updates_per_thread,
            &mut n_updates_per_thread,
            leaf_block,
            leaf_width,
            &dist_thresholds,
            data,
            n_features,
            n_threads,
        );

        for t in 0..n_threads {
            let block_start = t * vertex_block_size;
            let block_end = std::cmp::min(block_start + vertex_block_size, n_vertices);

            for j in 0..n_threads {
                for k in 0..n_updates_per_thread[j] as usize {
                    let base = (j * updates_per_thread + k) * 3;
                    let p = updates[base] as i32;

                    if p == -1 {
                        continue;
                    }

                    let q = updates[base + 1] as i32;
                    let d = updates[base + 2];

                    if p >= block_start as i32 && p < block_end as i32 {
                        let r = (p as usize) * size..(p as usize + 1) * size;
                        flagged_heap_push(
                            &mut current_graph.distances[r.clone()],
                            &mut current_graph.indices[r.clone()],
                            &mut current_graph.flags[r],
                            d,
                            q,
                        );
                    }
                    if q >= block_start as i32 && q < block_end as i32 {
                        let r = (q as usize) * size..(q as usize + 1) * size;
                        flagged_heap_push(
                            &mut current_graph.distances[r.clone()],
                            &mut current_graph.indices[r.clone()],
                            &mut current_graph.flags[r],
                            d,
                            p,
                        );
                    }
                }
            }
        }
    }
}

/// Port of `init_random_float`: fill under-populated heap rows with random
/// neighbors. Uses per-row COPIES of `rng_state` (never advances it).
pub fn init_random_float(
    n_neighbors: usize,
    data: &[f32],
    n_features: usize,
    heap: &mut NNDHeap,
    rng_state: &RngState,
) {
    let n_samples = data.len() / n_features;
    let size = heap.size;
    for i in 0..n_samples {
        let mut local_rng_state: RngState = [
            rng_state[0].wrapping_add(i as i64),
            rng_state[1].wrapping_add(i as i64),
            rng_state[2].wrapping_add(i as i64),
        ];
        if heap.indices[i * size] < 0 {
            // Python: range(n_neighbors - np.sum(heap[0][i] >= 0.0)),
            // evaluated once before the loop.
            let n_present = heap.indices[i * size..(i + 1) * size]
                .iter()
                .filter(|&&x| x >= 0)
                .count();
            for _j in 0..(n_neighbors - n_present) {
                // np.abs wraps at i32::MIN; Python % is sign-of-divisor
                let idx = (tau_rand_int(&mut local_rng_state).wrapping_abs() as i64)
                    .rem_euclid(n_samples as i64) as i32;
                if heap.indices[i * size..(i + 1) * size].contains(&idx) {
                    continue;
                }
                let d = fast_cosine(
                    data_row(data, n_features, idx as usize),
                    data_row(data, n_features, i),
                );
                let r = i * size..(i + 1) * size;
                flagged_heap_push(
                    &mut heap.distances[r.clone()],
                    &mut heap.indices[r.clone()],
                    &mut heap.flags[r],
                    d,
                    idx,
                );
            }
        }
    }
}

/// Port of `generate_graph_update_array_float_basic` (kept for parity with
/// the reference; the drivers use the working-set variant below).
#[allow(clippy::too_many_arguments)]
pub fn generate_graph_update_array_float_basic(
    update_array: &mut [f32],
    max_updates: usize,
    n_updates_per_thread: &mut [i32],
    new_candidate_block: &[i32],
    old_candidate_block: &[i32],
    max_new_candidates: usize,
    max_old_candidates: usize,
    dist_thresholds: &[f32],
    data: &[f32],
    n_features: usize,
    n_threads: usize,
) {
    let block_size = new_candidate_block.len() / max_new_candidates;
    let rows_per_thread = (block_size / n_threads) + 1;

    for t in 0..n_threads {
        let mut idx = 0usize;

        for r in 0..rows_per_thread {
            let i = t * rows_per_thread + r;
            if i >= block_size || idx >= max_updates {
                break;
            }

            for j in 0..max_new_candidates {
                if idx >= max_updates {
                    break;
                }

                let p = new_candidate_block[i * max_new_candidates + j];
                if p < 0 {
                    continue;
                }
                let data_p = data_row(data, n_features, p as usize);
                let dist_thresh_p = dist_thresholds[p as usize];

                for k in (j + 1)..max_new_candidates {
                    if idx >= max_updates {
                        break;
                    }

                    let q = new_candidate_block[i * max_new_candidates + k];
                    if q < 0 {
                        continue;
                    }

                    // Compute distance once
                    let d = fast_cosine(data_p, data_row(data, n_features, q as usize));

                    let dist_thresh_q = dist_thresholds[q as usize];
                    let max_threshold = dist_thresh_p.max(dist_thresh_q);

                    if d <= max_threshold {
                        let base = (t * max_updates + idx) * 3;
                        update_array[base] = p as f32;
                        update_array[base + 1] = q as f32;
                        update_array[base + 2] = d;
                        idx += 1;
                    }
                }

                for k in 0..max_old_candidates {
                    if idx >= max_updates {
                        break;
                    }

                    let q = old_candidate_block[i * max_old_candidates + k];
                    if q < 0 {
                        continue;
                    }

                    let d = fast_cosine(data_p, data_row(data, n_features, q as usize));
                    let dist_thresh_q = dist_thresholds[q as usize];
                    let max_threshold = dist_thresh_p.max(dist_thresh_q);

                    if d <= max_threshold {
                        let base = (t * max_updates + idx) * 3;
                        update_array[base] = p as f32;
                        update_array[base + 1] = q as f32;
                        update_array[base + 2] = d;
                        idx += 1;
                    }
                }
            }
        }

        n_updates_per_thread[t] = idx as i32;
    }
}

/// Port of `generate_graph_update_array_float` (working-set variant used by
/// `nn_descent_float`). The chunked iteration order is parity-relevant.
#[allow(clippy::too_many_arguments)]
pub fn generate_graph_update_array_float(
    update_array: &mut [f32],
    max_updates: usize,
    n_updates_per_thread: &mut [i32],
    new_candidate_block: &[i32],
    old_candidate_block: &[i32],
    max_new_candidates: usize,
    max_old_candidates: usize,
    dist_thresholds: &[f32],
    data: &[f32],
    n_features: usize,
    n_threads: usize,
) {
    let block_size = new_candidate_block.len() / max_new_candidates;
    let rows_per_thread = (block_size / n_threads) + 1;

    // Working set size - process this many candidates at a time
    let working_set_size = 8usize;

    for t in 0..n_threads {
        let mut idx = 0usize;

        for r in 0..rows_per_thread {
            let i = t * rows_per_thread + r;
            if i >= block_size || idx >= max_updates {
                break;
            }

            // Process new candidates in working set chunks
            let mut new_start = 0usize;
            while new_start < max_new_candidates && idx < max_updates {
                let new_end = std::cmp::min(new_start + working_set_size, max_new_candidates);

                // Process pairs within this working set
                for j in new_start..new_end {
                    if idx >= max_updates {
                        break;
                    }

                    let p = new_candidate_block[i * max_new_candidates + j];
                    if p < 0 {
                        continue;
                    }

                    let data_p = data_row(data, n_features, p as usize);
                    let dist_thresh_p = dist_thresholds[p as usize];

                    // Compare with other candidates in the same working set
                    for k in (j + 1)..new_end {
                        if idx >= max_updates {
                            break;
                        }

                        let q = new_candidate_block[i * max_new_candidates + k];
                        if q < 0 {
                            continue;
                        }

                        let d = fast_cosine(data_p, data_row(data, n_features, q as usize));
                        let dist_thresh_q = dist_thresholds[q as usize];
                        let max_threshold = dist_thresh_p.max(dist_thresh_q);

                        if d <= max_threshold {
                            let base = (t * max_updates + idx) * 3;
                            update_array[base] = p as f32;
                            update_array[base + 1] = q as f32;
                            update_array[base + 2] = d;
                            idx += 1;
                        }
                    }

                    // Compare with candidates in future working sets
                    for k in new_end..max_new_candidates {
                        if idx >= max_updates {
                            break;
                        }

                        let q = new_candidate_block[i * max_new_candidates + k];
                        if q < 0 {
                            continue;
                        }

                        let d = fast_cosine(data_p, data_row(data, n_features, q as usize));
                        let dist_thresh_q = dist_thresholds[q as usize];
                        let max_threshold = dist_thresh_p.max(dist_thresh_q);

                        if d <= max_threshold {
                            let base = (t * max_updates + idx) * 3;
                            update_array[base] = p as f32;
                            update_array[base + 1] = q as f32;
                            update_array[base + 2] = d;
                            idx += 1;
                        }
                    }

                    // Compare with old candidates in working set chunks
                    let mut old_start = 0usize;
                    while old_start < max_old_candidates && idx < max_updates {
                        let old_end =
                            std::cmp::min(old_start + working_set_size, max_old_candidates);

                        for k in old_start..old_end {
                            if idx >= max_updates {
                                break;
                            }

                            let q = old_candidate_block[i * max_old_candidates + k];
                            if q < 0 {
                                continue;
                            }

                            let d = fast_cosine(data_p, data_row(data, n_features, q as usize));
                            let dist_thresh_q = dist_thresholds[q as usize];
                            let max_threshold = dist_thresh_p.max(dist_thresh_q);

                            if d <= max_threshold {
                                let base = (t * max_updates + idx) * 3;
                                update_array[base] = p as f32;
                                update_array[base + 1] = q as f32;
                                update_array[base + 2] = d;
                                idx += 1;
                            }
                        }

                        old_start = old_end;
                    }
                }

                new_start = new_end;
            }
        }

        n_updates_per_thread[t] = idx as i32;
    }
}

/// Port of `generate_sorted_graph_update_array_float`: updates bucketed by
/// target vertex block for `apply_sorted_graph_updates`.
/// `n_updates_per_block` is flat (n_threads x (n_threads + 1)), column 0
/// unused.
#[allow(clippy::too_many_arguments)]
pub fn generate_sorted_graph_update_array_float(
    update_array: &mut [f32],
    max_updates: usize,
    n_updates_per_block: &mut [i32],
    new_candidate_block: &[i32],
    old_candidate_block: &[i32],
    max_new_candidates: usize,
    max_old_candidates: usize,
    dist_thresholds: &[f32],
    data: &[f32],
    n_features: usize,
    n_threads: usize,
) {
    let block_size_candidates = new_candidate_block.len() / max_new_candidates;
    let rows_per_thread = (block_size_candidates / n_threads) + 1;

    let n_vertices = data.len() / n_features;
    let vertex_block_size = n_vertices / n_threads + 1;
    let max_updates_per_src_thread = max_updates / n_threads;

    // Reset update counts
    for b in 0..n_threads {
        for t in 0..(n_threads + 1) {
            n_updates_per_block[b * (n_threads + 1) + t] = 0;
        }
    }

    // Each thread generates updates and places them in appropriate buckets
    for t in 0..n_threads {
        // Thread-local counters for each bucket
        let mut local_counts = vec![0i32; n_threads];

        for r in 0..rows_per_thread {
            let i = t * rows_per_thread + r;
            if i >= block_size_candidates {
                break;
            }

            for j in 0..max_new_candidates {
                let p = new_candidate_block[i * max_new_candidates + j];
                if p < 0 {
                    continue;
                }

                let data_p = data_row(data, n_features, p as usize);
                let dist_thresh_p = dist_thresholds[p as usize];
                let mut p_block = p as usize / vertex_block_size;
                if p_block >= n_threads {
                    p_block = n_threads - 1;
                }

                // Compare with other new candidates
                for k in (j + 1)..max_new_candidates {
                    let q = new_candidate_block[i * max_new_candidates + k];
                    if q < 0 {
                        continue;
                    }

                    let d = fast_cosine(data_p, data_row(data, n_features, q as usize));
                    let dist_thresh_q = dist_thresholds[q as usize];
                    let max_threshold = dist_thresh_p.max(dist_thresh_q);

                    if d <= max_threshold {
                        let mut q_block = q as usize / vertex_block_size;
                        if q_block >= n_threads {
                            q_block = n_threads - 1;
                        }

                        // Place update in p's bucket
                        let bucket_idx = local_counts[p_block] as usize;
                        let write_idx = t * max_updates_per_src_thread + bucket_idx;
                        if write_idx < max_updates {
                            let base = (p_block * max_updates + write_idx) * 3;
                            update_array[base] = p as f32;
                            update_array[base + 1] = q as f32;
                            update_array[base + 2] = d;
                            local_counts[p_block] += 1;
                        }

                        // If q is in a different block, also place in q's bucket
                        if q_block != p_block {
                            let bucket_idx = local_counts[q_block] as usize;
                            let write_idx = t * max_updates_per_src_thread + bucket_idx;
                            if write_idx < max_updates {
                                let base = (q_block * max_updates + write_idx) * 3;
                                update_array[base] = p as f32;
                                update_array[base + 1] = q as f32;
                                update_array[base + 2] = d;
                                local_counts[q_block] += 1;
                            }
                        }
                    }
                }

                // Compare with old candidates
                for k in 0..max_old_candidates {
                    let q = old_candidate_block[i * max_old_candidates + k];
                    if q < 0 {
                        continue;
                    }

                    let d = fast_cosine(data_p, data_row(data, n_features, q as usize));
                    let dist_thresh_q = dist_thresholds[q as usize];
                    let max_threshold = dist_thresh_p.max(dist_thresh_q);

                    if d <= max_threshold {
                        let mut q_block = q as usize / vertex_block_size;
                        if q_block >= n_threads {
                            q_block = n_threads - 1;
                        }

                        // Place update in p's bucket
                        let bucket_idx = local_counts[p_block] as usize;
                        let write_idx = t * max_updates_per_src_thread + bucket_idx;
                        if write_idx < max_updates {
                            let base = (p_block * max_updates + write_idx) * 3;
                            update_array[base] = p as f32;
                            update_array[base + 1] = q as f32;
                            update_array[base + 2] = d;
                            local_counts[p_block] += 1;
                        }

                        // If q is in a different block, also place in q's bucket
                        if q_block != p_block {
                            let bucket_idx = local_counts[q_block] as usize;
                            let write_idx = t * max_updates_per_src_thread + bucket_idx;
                            if write_idx < max_updates {
                                let base = (q_block * max_updates + write_idx) * 3;
                                update_array[base] = p as f32;
                                update_array[base + 1] = q as f32;
                                update_array[base + 2] = d;
                                local_counts[q_block] += 1;
                            }
                        }
                    }
                }
            }
        }

        // Record total updates generated by this thread for each bucket
        for b in 0..n_threads {
            n_updates_per_block[b * (n_threads + 1) + t + 1] = local_counts[b];
        }
    }
}

/// Shared tail of both drivers: sort the heap and hand back its arrays.
fn finish(current_graph: NNDHeap) -> (Vec<i32>, Vec<f32>) {
    let size = current_graph.size;
    let mut indices = current_graph.indices;
    let mut distances = current_graph.distances;
    deheap_sort(&mut indices, &mut distances, size);
    (indices, distances)
}

/// Port of `nn_descent_float`. Returns `(indices, distances)` (row-major
/// n_samples x n_neighbors), distances being raw negative cosine
/// similarity, sorted ascending per row.
///
/// PORT NOTE: `verbose` printing is dropped (no I/O in core). The
/// `delta_improv` distance sum uses a sequential f64 accumulation where the
/// reference uses `np.sum` (pairwise f32); it only gates early termination
/// and the fixtures set `delta_improv = None`.
#[allow(clippy::too_many_arguments)]
pub fn nn_descent_float(
    data: &[f32],
    n_features: usize,
    n_neighbors: usize,
    rng_state: &RngState,
    max_candidates: usize,
    n_iters: usize,
    delta: f64,
    delta_improv: Option<f64>,
    leaf_array: &[i32],
    leaf_width: usize,
    n_threads: usize,
) -> (Vec<i32>, Vec<f32>) {
    let n_samples = data.len() / n_features;
    let mut current_graph = make_heap(n_samples, n_neighbors);
    init_rp_tree_float(
        data,
        n_features,
        &mut current_graph,
        leaf_array,
        leaf_width,
        n_threads,
    );
    init_random_float(n_neighbors, data, n_features, &mut current_graph, rng_state);

    let n_vertices = n_samples;
    let mut block_size = 65536 / n_threads;
    let mut n_blocks = n_vertices / block_size;

    // Python: int((mc**2 + mc*(mc-1)/2) * block_size) -- f64 then truncated
    let max_updates_per_thread = (((max_candidates * max_candidates) as f64
        + (max_candidates * (max_candidates - 1)) as f64 / 2.0)
        * block_size as f64) as usize;
    let mut update_array = vec![0.0f32; n_threads * max_updates_per_thread * 3];
    let mut n_updates_per_thread = vec![0i32; n_threads];

    // For distance-based termination
    let mut prev_sum_dist: Option<f64> = None;

    // Scratch buffers reused across all `n_iters` rounds; see the identical
    // comment in `nn_descent_float_sorted` above for why this is a safe,
    // value-preserving allocation-only change.
    let mut new_candidate_indices = vec![-1i32; n_samples * max_candidates];
    let mut new_candidate_priority = vec![f32::INFINITY; n_samples * max_candidates];
    let mut old_candidate_indices = vec![-1i32; n_samples * max_candidates];
    let mut old_candidate_priority = vec![f32::INFINITY; n_samples * max_candidates];
    let mut dist_thresholds = vec![0.0f32; n_samples];

    for _n in 0..n_iters {
        crate::nnd_heap::build_candidates_into(
            &mut current_graph,
            max_candidates,
            rng_state,
            n_threads,
            &mut new_candidate_indices,
            &mut new_candidate_priority,
            &mut old_candidate_indices,
            &mut old_candidate_priority,
        );
        let new_candidate_neighbors = &new_candidate_indices;
        let old_candidate_neighbors = &old_candidate_indices;

        let mut c: u64 = 0;
        let n_vertices = new_candidate_neighbors.len() / max_candidates;
        for i in 0..(n_blocks + 1) {
            let block_start = i * block_size;
            let block_end = std::cmp::min(n_vertices, (i + 1) * block_size);

            let new_candidate_block =
                &new_candidate_neighbors[block_start * max_candidates..block_end * max_candidates];
            let old_candidate_block =
                &old_candidate_neighbors[block_start * max_candidates..block_end * max_candidates];

            for (v, dst) in dist_thresholds.iter_mut().enumerate() {
                *dst = current_graph.distances[v * n_neighbors];
            }

            generate_graph_update_array_float(
                &mut update_array,
                max_updates_per_thread,
                &mut n_updates_per_thread,
                new_candidate_block,
                old_candidate_block,
                max_candidates,
                max_candidates,
                &dist_thresholds,
                data,
                n_features,
                n_threads,
            );

            c += apply_graph_update_array(
                &mut current_graph,
                &update_array,
                max_updates_per_thread,
                &n_updates_per_thread,
                n_threads,
            ) as u64;
        }

        // Check update count termination
        if c as f64 <= delta * n_neighbors as f64 * n_samples as f64 {
            return finish(current_graph);
        }

        // Check distance improvement termination (if enabled)
        if let Some(delta_improv) = delta_improv {
            let mut sum_dist = 0.0f64;
            for &d in current_graph.distances.iter() {
                if d < f32::MAX {
                    sum_dist += d as f64;
                }
            }

            if let Some(prev) = prev_sum_dist {
                let rel_improv = (sum_dist - prev).abs() / prev.abs();
                if rel_improv < delta_improv {
                    return finish(current_graph);
                }
            }

            prev_sum_dist = Some(sum_dist);
        }

        block_size = std::cmp::min(n_vertices, 2 * block_size);
        n_blocks = n_vertices / block_size;
    }

    finish(current_graph)
}

/// Port of `nn_descent_float_sorted` (bucketed-update variant; see
/// `nn_descent_float`'s PORT NOTEs, which apply here too).
#[allow(clippy::too_many_arguments)]
pub fn nn_descent_float_sorted(
    data: &[f32],
    n_features: usize,
    n_neighbors: usize,
    rng_state: &RngState,
    max_candidates: usize,
    n_iters: usize,
    delta: f64,
    delta_improv: Option<f64>,
    leaf_array: &[i32],
    leaf_width: usize,
    n_threads: usize,
) -> (Vec<i32>, Vec<f32>) {
    let n_samples = data.len() / n_features;
    let mut current_graph = make_heap(n_samples, n_neighbors);
    init_rp_tree_float(
        data,
        n_features,
        &mut current_graph,
        leaf_array,
        leaf_width,
        n_threads,
    );
    init_random_float(n_neighbors, data, n_features, &mut current_graph, rng_state);

    let n_vertices = n_samples;
    let mut block_size = 65536 / n_threads;
    let mut n_blocks = n_vertices / block_size;

    let max_updates_per_thread = (((max_candidates * max_candidates) as f64
        + (max_candidates * (max_candidates - 1)) as f64 / 2.0)
        * block_size as f64) as usize;
    // For sorted updates: shape is (n_threads, max_updates_per_block, 3);
    // each bucket (first dim) holds updates targeting that block
    let mut sorted_update_array = vec![0.0f32; n_threads * max_updates_per_thread * 3];
    // Track updates per block, with per-thread breakdown:
    // (n_threads, n_threads + 1); column 0 unused
    let mut n_updates_per_block = vec![0i32; n_threads * (n_threads + 1)];

    let mut prev_sum_dist: Option<f64> = None;

    // Scratch buffers for build_candidates_into, allocated once and reused
    // across all `n_iters` rounds (SAFE allocation-only change: same values
    // every round since build_candidates_into resets them at entry, see its
    // doc comment; avoids n_iters fresh allocations of 4 * n_vertices *
    // max_candidates elements each, which showed up as the dominant wasm
    // allocator cost in prior profiling, see comparison/results.md).
    let mut new_candidate_indices = vec![-1i32; n_samples * max_candidates];
    let mut new_candidate_priority = vec![f32::INFINITY; n_samples * max_candidates];
    let mut old_candidate_indices = vec![-1i32; n_samples * max_candidates];
    let mut old_candidate_priority = vec![f32::INFINITY; n_samples * max_candidates];
    // Reused across rounds/blocks too, same reasoning; refilled in full
    // every use (never partially stale).
    let mut dist_thresholds = vec![0.0f32; n_samples];

    for _n in 0..n_iters {
        crate::nnd_heap::build_candidates_into(
            &mut current_graph,
            max_candidates,
            rng_state,
            n_threads,
            &mut new_candidate_indices,
            &mut new_candidate_priority,
            &mut old_candidate_indices,
            &mut old_candidate_priority,
        );
        let new_candidate_neighbors = &new_candidate_indices;
        let old_candidate_neighbors = &old_candidate_indices;

        let mut c: u64 = 0;
        let n_vertices = new_candidate_neighbors.len() / max_candidates;
        for i in 0..(n_blocks + 1) {
            let block_start = i * block_size;
            let block_end = std::cmp::min(n_vertices, (i + 1) * block_size);

            let new_candidate_block =
                &new_candidate_neighbors[block_start * max_candidates..block_end * max_candidates];
            let old_candidate_block =
                &old_candidate_neighbors[block_start * max_candidates..block_end * max_candidates];

            for (v, dst) in dist_thresholds.iter_mut().enumerate() {
                *dst = current_graph.distances[v * n_neighbors];
            }

            // Reset update counts for this iteration
            n_updates_per_block.fill(0);

            generate_sorted_graph_update_array_float(
                &mut sorted_update_array,
                max_updates_per_thread,
                &mut n_updates_per_block,
                new_candidate_block,
                old_candidate_block,
                max_candidates,
                max_candidates,
                &dist_thresholds,
                data,
                n_features,
                n_threads,
            );

            c += apply_sorted_graph_updates(
                &mut current_graph,
                &sorted_update_array,
                max_updates_per_thread,
                &n_updates_per_block,
                n_threads,
            ) as u64;
        }

        if c as f64 <= delta * n_neighbors as f64 * n_samples as f64 {
            return finish(current_graph);
        }

        if let Some(delta_improv) = delta_improv {
            let mut sum_dist = 0.0f64;
            for &d in current_graph.distances.iter() {
                if d < f32::MAX {
                    sum_dist += d as f64;
                }
            }

            if let Some(prev) = prev_sum_dist {
                let rel_improv = (sum_dist - prev).abs() / prev.abs();
                if rel_improv < delta_improv {
                    return finish(current_graph);
                }
            }

            prev_sum_dist = Some(sum_dist);
        }

        block_size = std::cmp::min(n_vertices, 2 * block_size);
        n_blocks = n_vertices / block_size;
    }

    finish(current_graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::seed;

    fn normalized_data(rows: &[[f32; 2]]) -> Vec<f32> {
        rows.iter()
            .flat_map(|r| {
                let norm = (r[0] * r[0] + r[1] * r[1]).sqrt();
                [r[0] / norm, r[1] / norm]
            })
            .collect()
    }

    #[test]
    fn split_partitions_all_points() {
        let data: Vec<f32> = (0..20)
            .flat_map(|i| [i as f32, (i * 7 % 13) as f32])
            .collect();
        let indices: Vec<i32> = (0..20).collect();
        let mut rng: RngState = [0; 3];
        seed(&mut rng, 42);
        let (left, right) = float_random_projection_split(&data, 2, &indices, &mut rng);
        let mut all: Vec<i32> = left.iter().chain(right.iter()).copied().collect();
        all.sort_unstable();
        assert_eq!(all, indices);
    }

    #[test]
    fn split_identical_points_degenerate_reassigns_randomly() {
        // all-identical points: every margin is ~0, coin flips classify;
        // whatever happens, the result must still be a partition.
        let data: Vec<f32> = std::iter::repeat([1.0f32, 2.0])
            .take(10)
            .flatten()
            .collect();
        let indices: Vec<i32> = (0..10).collect();
        let mut rng: RngState = [0; 3];
        seed(&mut rng, 7);
        let (left, right) = float_random_projection_split(&data, 2, &indices, &mut rng);
        assert_eq!(left.len() + right.len(), 10);
        let mut all: Vec<i32> = left.iter().chain(right.iter()).copied().collect();
        all.sort_unstable();
        assert_eq!(all, indices);
    }

    #[test]
    fn leaf_array_covers_every_point_exactly_once() {
        let data: Vec<f32> = (0..96)
            .flat_map(|i| [(i % 17) as f32, (i % 5) as f32, (i * 3 % 11) as f32])
            .collect();
        let mut rng: RngState = [0; 3];
        seed(&mut rng, 123);
        let (leaf_array, width) = make_float_leaf_array_serial(&data, 3, &mut rng, 8, 200);
        assert!(width >= 8);
        let mut points: Vec<i32> = leaf_array.iter().copied().filter(|&x| x >= 0).collect();
        points.sort_unstable();
        assert_eq!(points, (0..96).collect::<Vec<_>>());
    }

    #[test]
    fn forest_variants_agree_and_advance_states_identically() {
        let data: Vec<f32> = (0..48)
            .flat_map(|i| [(i % 7) as f32, (i % 3) as f32])
            .collect();
        let mut states_a: Vec<RngState> = vec![[11, 22, 33], [44, 55, 66]];
        let mut states_b = states_a.clone();
        let fa = make_float_forest(&data, 2, &mut states_a, 5, 200, true);
        let fb = make_float_forest(&data, 2, &mut states_b, 5, 200, false);
        assert_eq!(fa.len(), 2);
        assert_eq!(fa, fb);
        assert_eq!(states_a, states_b);
    }

    #[test]
    fn nn_descent_tiny_self_is_first_neighbor() {
        // 4 normalized points; one leaf containing all of them seeds the
        // graph with all pairs plus (i, i, -1.0) self-updates, so after
        // sorting each row starts with the point itself at distance -1.
        let data = normalized_data(&[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-0.5, 0.8]]);
        let leaf_array: Vec<i32> = vec![0, 1, 2, 3];
        let mut rng: RngState = [0; 3];
        seed(&mut rng, 42);
        let (indices, distances) =
            nn_descent_float(&data, 2, 2, &rng, 4, 3, 0.001, None, &leaf_array, 4, 1);
        for i in 0..4 {
            assert_eq!(indices[i * 2], i as i32, "row {i} self first");
            assert!((distances[i * 2] - -1.0).abs() < 1e-6);
            // sorted ascending
            assert!(distances[i * 2] <= distances[i * 2 + 1]);
        }
    }

    #[test]
    fn nn_descent_sorted_variant_produces_valid_sorted_rows() {
        // NOTE: plain and sorted variants apply updates in different orders
        // so their outputs are not asserted equal (ties may resolve
        // differently); each must independently produce sorted rows with
        // self first.
        let data = normalized_data(&[
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [-0.5, 0.8],
            [0.3, -0.7],
            [0.8, 0.5],
        ]);
        let leaf_array: Vec<i32> = vec![0, 1, 2, -1, 3, 4, 5, -1];
        let mut rng: RngState = [0; 3];
        seed(&mut rng, 99);
        let (indices, distances) =
            nn_descent_float_sorted(&data, 2, 3, &rng, 4, 4, 0.001, None, &leaf_array, 4, 1);
        for i in 0..6 {
            assert_eq!(indices[i * 3], i as i32, "row {i} self first");
            assert!(distances[i * 3] <= distances[i * 3 + 1]);
            assert!(distances[i * 3 + 1] <= distances[i * 3 + 2]);
        }
    }
}
