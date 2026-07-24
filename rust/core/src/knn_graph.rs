//! Port of the float path of `evoc/knn_graph.py`: `make_forest` (float),
//! `nn_descent` (float branch, incl. the distance transform), and the
//! `knn_graph` orchestration.
//!
//! PORT NOTE (per PORT_NOTES.md): the reference draws `rng_state` and the
//! per-tree `rng_states` from numpy's Mersenne Twister
//! (`check_random_state`). We have no Mersenne Twister; the caller supplies
//! `rng_state` (for nn_descent, drawn first) and `rng_states` (one triple
//! per RP-tree, drawn second inside make_forest) verbatim -- draw order is
//! load-bearing and recorded in the fixture inputs. Only the float dtype
//! path is ported (int8/uint8 are later phases).

use crate::nndescent_float::{make_float_forest, nn_descent_float, nn_descent_float_sorted};
use crate::rng::RngState;

/// Port of the float branch of `make_forest`: build `n_trees` RP-trees
/// (one per `rng_states` row), then pad every tree's leaf array to the
/// widest leaf width across all trees (constant_values = -1) and vstack.
/// Returns flat row-major `(leaf_array, leaf_width)`.
///
/// PORT NOTE: `make_float_forest` mutates each `rng_states` row in place, as
/// in the reference (numba passes 2-D rows as views). `enable_nested_
/// parallelism` selects the (single-threaded-identical) forest variant.
pub fn make_forest(
    data: &[f32],
    n_features: usize,
    rng_states: &mut [RngState],
    leaf_size: usize,
    max_depth: i64,
    enable_nested_parallelism: bool,
) -> (Vec<i32>, usize) {
    let per_tree = make_float_forest(
        data,
        n_features,
        rng_states,
        leaf_size,
        max_depth,
        enable_nested_parallelism,
    );

    // different trees can end up with different max leaf sizes
    let max_leaf_size = per_tree.iter().map(|(_, w)| *w).max().unwrap_or(0);

    // pad each tree's leaf array out to max_leaf_size (constant_values=-1)
    // and stack the rows.
    let mut leaf_array: Vec<i32> = Vec::new();
    for (flat, width) in per_tree.iter() {
        let n_rows = flat.len() / width;
        for r in 0..n_rows {
            leaf_array.extend_from_slice(&flat[r * width..r * width + width]);
            leaf_array.extend(std::iter::repeat(-1).take(max_leaf_size - width));
        }
    }

    (leaf_array, max_leaf_size)
}

/// Port of the float branch of `nn_descent`: run NN-descent (sorted or
/// plain) and apply the float distance transform
/// `max(-log2(-d), 0.0)` elementwise. Returns `(indices, distances)` flat
/// row-major n_samples x n_neighbors.
#[allow(clippy::too_many_arguments)]
pub fn nn_descent(
    data: &[f32],
    n_features: usize,
    n_neighbors: usize,
    rng_state: &RngState,
    effective_max_candidates: usize,
    n_iters: usize,
    delta: f64,
    delta_improv: Option<f64>,
    leaf_array: &[i32],
    leaf_width: usize,
    n_threads: usize,
    use_sorted_updates: bool,
) -> (Vec<i32>, Vec<f32>) {
    let (indices, mut distances) = if use_sorted_updates {
        nn_descent_float_sorted(
            data,
            n_features,
            n_neighbors,
            rng_state,
            effective_max_candidates,
            n_iters,
            delta,
            delta_improv,
            leaf_array,
            leaf_width,
            n_threads,
        )
    } else {
        nn_descent_float(
            data,
            n_features,
            n_neighbors,
            rng_state,
            effective_max_candidates,
            n_iters,
            delta,
            delta_improv,
            leaf_array,
            leaf_width,
            n_threads,
        )
    };

    // Float distance transform: max(-log2(-d), 0.0)  (per PORT_NOTES)
    for d in distances.iter_mut() {
        *d = (-((-*d).log2())).max(0.0);
    }

    (indices, distances)
}

/// Port of the float path of `knn_graph`, consuming pre-drawn RNG state.
///
/// PORT NOTE: the reference derives `rng_state`/`rng_states`,
/// `n_trees`/`n_iters`/`effective_max_candidates`, and normalization from
/// `data`/`random_state`; here those are explicit inputs (the fixture bakes
/// the exact values a real `check_random_state(seed)` produced). The data
/// normalization path is assumed already applied by the caller (the fixture
/// data is pre-unit-normalized, so knn_graph's normalization is a verified
/// no-op). Returns `(nn_indices, nn_dists)` flat row-major.
#[allow(clippy::too_many_arguments)]
pub fn knn_graph(
    data: &[f32],
    n_features: usize,
    n_neighbors: usize,
    rng_state: &RngState,
    rng_states: &mut [RngState],
    leaf_size: usize,
    max_rptree_depth: i64,
    effective_max_candidates: usize,
    n_iters: usize,
    delta: f64,
    delta_improv: Option<f64>,
    use_sorted_updates: bool,
    n_threads: usize,
    enable_nested_parallelism: bool,
) -> (Vec<i32>, Vec<f32>) {
    let (leaf_array, leaf_width) = make_forest(
        data,
        n_features,
        rng_states,
        leaf_size,
        max_rptree_depth,
        enable_nested_parallelism,
    );

    nn_descent(
        data,
        n_features,
        n_neighbors,
        rng_state,
        effective_max_candidates,
        n_iters,
        delta,
        delta_improv,
        &leaf_array,
        leaf_width,
        n_threads,
        use_sorted_updates,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::seed;

    fn normalized(rows: &[[f32; 2]]) -> Vec<f32> {
        rows.iter()
            .flat_map(|r| {
                let n = (r[0] * r[0] + r[1] * r[1]).sqrt();
                [r[0] / n, r[1] / n]
            })
            .collect()
    }

    #[test]
    fn make_forest_pads_to_widest_tree() {
        let data: Vec<f32> = (0..40).flat_map(|i| [(i % 7) as f32, (i % 3) as f32]).collect();
        let mut rng_states: Vec<RngState> = vec![[1, 2, 3], [4, 5, 6]];
        let (leaf_array, width) = make_forest(&data, 2, &mut rng_states, 5, 200, false);
        // every row has the padded width; all real points 0..20 appear
        assert_eq!(leaf_array.len() % width, 0);
        let mut present: Vec<i32> = leaf_array.iter().copied().filter(|&x| x >= 0).collect();
        present.sort_unstable();
        present.dedup();
        // each tree covers all 40 points, so union is 0..40
        assert_eq!(present, (0..40).collect::<Vec<_>>());
    }

    #[test]
    fn float_distance_transform_clamps_at_zero() {
        // a raw distance of -1.0 (perfect cosine sim) -> -log2(1)=0 -> max(0,0)=0
        let data = normalized(&[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-0.3, 0.7]]);
        let leaf_array: Vec<i32> = vec![0, 1, 2, 3];
        let mut rng: RngState = [0; 3];
        seed(&mut rng, 7);
        let (indices, distances) =
            nn_descent(&data, 2, 2, &rng, 4, 3, 0.001, None, &leaf_array, 4, 1, true);
        for i in 0..4 {
            // self is first neighbor, transformed distance is exactly 0
            assert_eq!(indices[i * 2], i as i32);
            assert_eq!(distances[i * 2], 0.0);
            assert!(distances[i * 2 + 1] >= 0.0);
        }
    }

    #[test]
    fn knn_graph_runs_end_to_end_single_self_neighbor() {
        let data = normalized(&[[1.0, 0.0], [0.6, 0.8], [0.0, 1.0], [-0.8, 0.6], [0.5, -0.5]]);
        let rng_state: RngState = [11, 22, 33];
        let mut rng_states: Vec<RngState> = vec![[1, 2, 3], [4, 5, 6]];
        let (inds, dists) = knn_graph(
            &data, 2, 3, &rng_state, &mut rng_states, 3, 200, 4, 4, 0.001, Some(0.001), true, 1,
            false,
        );
        assert_eq!(inds.len(), 5 * 3);
        for i in 0..5 {
            assert_eq!(inds[i * 3], i as i32, "self first for row {i}");
            assert_eq!(dists[i * 3], 0.0);
        }
    }
}
