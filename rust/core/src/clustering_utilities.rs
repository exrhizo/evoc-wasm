//! Port of `evoc/clustering_utilities.py`: persistence-barcode utilities,
//! peak finding, diverse-peak selection, and cluster-tree/duplicate helpers
//! that sit on top of `cluster_trees.rs`.
//!
//! PORT NOTE: every function here is pure/deterministic given its inputs --
//! no `random_state`, no numba `prange` nondeterminism (see PORT_NOTES.md's
//! "clustering_utilities.py (phase 8)" section) -- so branch/iteration order
//! is the only real parity hazard, not RNG plumbing.

use crate::cluster_trees::{
    get_cluster_label_vector, get_point_membership_strength_vector, CondensedTree,
};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// find_peaks (ported near-verbatim from scipy's peak finder)
// ---------------------------------------------------------------------------

/// Port of `find_peaks`. First and last samples can never be peaks. A
/// plateau of equal consecutive values counts as one peak, with
/// `midpoint = (left_edge + right_edge) // 2` (integer floor division).
pub fn find_peaks(x: &[f32]) -> Vec<usize> {
    if x.len() < 2 {
        return Vec::new();
    }
    let mut midpoints = Vec::with_capacity(x.len() / 2);

    let i_max = x.len() - 1;
    let mut i = 1usize;
    while i < i_max {
        if x[i - 1] < x[i] {
            let mut i_ahead = i + 1;
            while i_ahead < i_max && x[i_ahead] == x[i] {
                i_ahead += 1;
            }
            if x[i_ahead] < x[i] {
                let left_edge = i;
                let right_edge = i_ahead - 1;
                midpoints.push((left_edge + right_edge) / 2);
                i = i_ahead;
            }
        }
        i += 1;
    }

    midpoints
}

// ---------------------------------------------------------------------------
// _binary_search_for_n_clusters
// ---------------------------------------------------------------------------

use crate::cluster_trees::{condense_tree, extract_leaves, LinkageTree};

/// Port of `_binary_search_for_n_clusters`. Returns `(leaves, clusters,
/// strengths)`.
///
/// PORT NOTE: bisects `min_cluster_size` in `[2, n_samples // 2]` using
/// Python's `round()` (banker's rounding on exact `.5`). Terminates the loop
/// when the bracket narrows to width 1, or early if `mid` lands exactly on
/// either bound (break without recomputing `mid_tree` for that iteration).
/// Final selection is a three-way branch: strictly-closer lower, strictly-
/// closer upper, or exact tie broken by whichever recomputed tree has more
/// non-noise points.
pub fn _binary_search_for_n_clusters(
    uncondensed_tree: &LinkageTree,
    approx_n_clusters: i64,
    n_samples: usize,
) -> (Vec<i64>, Vec<i64>, Vec<f32>) {
    let mut lower_bound: i64 = 2;
    let mut upper_bound: i64 = (n_samples / 2) as i64;

    let upper_tree = condense_tree(uncondensed_tree, upper_bound);
    let mut upper_n_clusters = extract_leaves(&upper_tree).len() as i64;

    let lower_tree = condense_tree(uncondensed_tree, lower_bound);
    let mut lower_n_clusters = extract_leaves(&lower_tree).len() as i64;

    while upper_bound - lower_bound > 1 {
        let mid = python_round_half_even((lower_bound + upper_bound) as f64 / 2.0);
        if mid == lower_bound || mid == upper_bound {
            break;
        }
        let mid_tree = condense_tree(uncondensed_tree, mid);
        let mid_n_clusters = extract_leaves(&mid_tree).len() as i64;

        if mid_n_clusters < approx_n_clusters {
            upper_bound = mid;
            upper_n_clusters = mid_n_clusters;
        } else {
            lower_bound = mid;
            lower_n_clusters = mid_n_clusters;
        }
    }

    let lower_diff = (lower_n_clusters - approx_n_clusters).abs();
    let upper_diff = (upper_n_clusters - approx_n_clusters).abs();

    if lower_diff < upper_diff {
        let lower_tree = condense_tree(uncondensed_tree, lower_bound);
        let leaves = extract_leaves(&lower_tree);
        let clusters = get_cluster_label_vector(&lower_tree, &leaves, 0.0, n_samples);
        let strengths = get_point_membership_strength_vector(&lower_tree, &leaves, &clusters);
        (leaves, clusters, strengths)
    } else if lower_diff > upper_diff {
        let upper_tree = condense_tree(uncondensed_tree, upper_bound);
        let leaves = extract_leaves(&upper_tree);
        let clusters = get_cluster_label_vector(&upper_tree, &leaves, 0.0, n_samples);
        let strengths = get_point_membership_strength_vector(&upper_tree, &leaves, &clusters);
        (leaves, clusters, strengths)
    } else {
        let lower_tree = condense_tree(uncondensed_tree, lower_bound);
        let lower_leaves = extract_leaves(&lower_tree);
        let lower_clusters = get_cluster_label_vector(&lower_tree, &lower_leaves, 0.0, n_samples);

        let upper_tree = condense_tree(uncondensed_tree, upper_bound);
        let upper_leaves = extract_leaves(&upper_tree);
        let upper_clusters = get_cluster_label_vector(&upper_tree, &upper_leaves, 0.0, n_samples);

        let lower_non_noise = lower_clusters.iter().filter(|&&c| c >= 0).count();
        let upper_non_noise = upper_clusters.iter().filter(|&&c| c >= 0).count();

        if lower_non_noise > upper_non_noise {
            let strengths =
                get_point_membership_strength_vector(&lower_tree, &lower_leaves, &lower_clusters);
            (lower_leaves, lower_clusters, strengths)
        } else {
            let strengths =
                get_point_membership_strength_vector(&upper_tree, &upper_leaves, &upper_clusters);
            (upper_leaves, upper_clusters, strengths)
        }
    }
}

/// Python's `round()` semantics: round-half-to-even on exact `.5`
/// boundaries (banker's rounding), unlike `f64::round` which rounds half
/// away from zero.
fn python_round_half_even(x: f64) -> i64 {
    let floor = x.floor();
    let diff = x - floor;
    let floor_i = floor as i64;
    if diff < 0.5 {
        floor_i
    } else if diff > 0.5 {
        floor_i + 1
    } else if floor_i % 2 == 0 {
        floor_i
    } else {
        floor_i + 1
    }
}

// ---------------------------------------------------------------------------
// min_cluster_size_barcode
// ---------------------------------------------------------------------------

/// Port of `min_cluster_size_barcode`.
///
/// PORT NOTE: `n_nodes = cluster_tree.child[-1] - n_points + 1`. Assumes
/// `cluster_tree` is non-empty and sorted with the root-adjacent pair last
/// (condense_tree's/mask_condensed_tree's natural output order). Iterates
/// row-pairs in reverse, stepping by 2. Index 0 (the tree root) is
/// special-cased before the loop.
pub fn min_cluster_size_barcode(
    cluster_tree: &CondensedTree,
    n_points: i64,
    min_size: f32,
) -> (Vec<f32>, Vec<f32>, Vec<i64>, Vec<f32>) {
    let n_nodes = (cluster_tree.child[cluster_tree.child.len() - 1] - n_points + 1) as usize;
    let mut parents = vec![0i64; n_nodes];
    let mut lambda_deaths = vec![0.0f32; n_nodes];
    let mut size_deaths = vec![0.0f32; n_nodes];
    let mut size_births = vec![min_size; n_nodes];

    lambda_deaths[0] = 0.0;
    size_deaths[0] = n_points as f32;
    parents[0] = n_points;

    let n_rows = cluster_tree.child.len();
    let mut idx = (n_rows as i64) - 1;
    while idx > 0 {
        let out_idx = (cluster_tree.child[idx as usize] - n_points) as usize;
        parents[out_idx - 1] = cluster_tree.parent[idx as usize];
        parents[out_idx] = cluster_tree.parent[idx as usize];
        let lambda_death = (-1.0 / cluster_tree.lambda_val[idx as usize]).exp();
        lambda_deaths[out_idx - 1] = lambda_death;
        lambda_deaths[out_idx] = lambda_death;

        let death_size = cluster_tree.child_size[(idx - 1) as usize]
            .min(cluster_tree.child_size[idx as usize]) as f32;
        size_deaths[out_idx - 1] = death_size;
        size_deaths[out_idx] = death_size;

        let parent_out = (cluster_tree.parent[idx as usize] - n_points) as usize;
        size_births[parent_out] = size_births[out_idx - 1]
            .max(size_births[out_idx])
            .max(death_size);

        idx -= 2;
    }

    (size_births, size_deaths, parents, lambda_deaths)
}

// ---------------------------------------------------------------------------
// compute_total_persistence
// ---------------------------------------------------------------------------

/// Port of `compute_total_persistence`.
///
/// PORT NOTE: loop starts at `i = 1` (root entry at index 0 is skipped,
/// "maintain left-open (birth, death] interval"). `sizes = unique(births)`
/// sorted+deduped. `birth_idx`/`death_idx` are literal first-match linear
/// scans (`death_idx` defaults to `sizes.len()` if none match). Entries
/// with `death <= birth` are skipped. Range-add `(death - birth) *
/// lambda_death` into every bucket in `[birth_idx, death_idx)`.
pub fn compute_total_persistence(
    births: &[f32],
    deaths: &[f32],
    lambda_deaths: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut sizes: Vec<f32> = births.to_vec();
    sizes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sizes.dedup();

    let mut total_persistence = vec![0.0f32; sizes.len()];

    for i in 1..births.len() {
        let birth = births[i];
        let death = deaths[i];
        let lambda_death = lambda_deaths[i];

        if death <= birth {
            continue;
        }

        let mut birth_idx = 0usize;
        for (j, &s) in sizes.iter().enumerate() {
            if s >= birth {
                birth_idx = j;
                break;
            }
        }

        let mut death_idx = sizes.len();
        for (j, &s) in sizes.iter().enumerate() {
            if s >= death {
                death_idx = j;
                break;
            }
        }

        for slot in total_persistence.iter_mut().take(death_idx).skip(birth_idx) {
            *slot += (death - birth) * lambda_death;
        }
    }

    (sizes, total_persistence)
}

// ---------------------------------------------------------------------------
// extract_clusters_by_id
// ---------------------------------------------------------------------------

/// Port of `extract_clusters_by_id`. `n_samples = condensed_tree.parent[0]`
/// (read off the tree, not a parameter); `cluster_selection_epsilon = 0.0`
/// is hardcoded.
pub fn extract_clusters_by_id(
    condensed_tree: &CondensedTree,
    selected_ids: &[i64],
) -> (Vec<i64>, Vec<f32>) {
    let n_samples = condensed_tree.parent[0] as usize;
    let labels = get_cluster_label_vector(condensed_tree, selected_ids, 0.0, n_samples);
    let strengths = get_point_membership_strength_vector(condensed_tree, selected_ids, &labels);
    (labels, strengths)
}

// ---------------------------------------------------------------------------
// jaccard_similarity
// ---------------------------------------------------------------------------

/// Port of `jaccard_similarity`.
///
/// PORT NOTE: `union_set` starts as `set(set_a_array)`, then `set_b_array`
/// is folded in element by element, counting `intersection_count` for every
/// b-item already in the (growing) union set -- replicate this exact
/// asymmetric construction (matters only with internal duplicates in the
/// input arrays). Empty/empty -> `union_count == 0` -> returns `0.0` exactly.
pub fn jaccard_similarity(set_a_array: &[i64], set_b_array: &[i64]) -> f64 {
    let mut intersection_count: i64 = 0;
    let mut union_set: HashSet<i64> = set_a_array.iter().copied().collect();

    for &item in set_b_array {
        if union_set.contains(&item) {
            intersection_count += 1;
        } else {
            union_set.insert(item);
        }
    }

    let union_count = union_set.len();
    if union_count > 0 {
        intersection_count as f64 / union_count as f64
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// estimate_cluster_similarity
// ---------------------------------------------------------------------------

/// Port of `estimate_cluster_similarity`. "Active at X" means `births[i] <=
/// X and deaths[i] > X` (half-open `[birth, death)` window read at a point
/// -- a different convention from `compute_total_persistence`'s documented
/// `(birth, death]` interval semantics).
pub fn estimate_cluster_similarity(births: &[f32], deaths: &[f32], birth_a: f32, birth_b: f32) -> f64 {
    let clusters_a: Vec<i64> = (0..births.len() as i64)
        .filter(|&i| births[i as usize] <= birth_a && deaths[i as usize] > birth_a)
        .collect();
    let clusters_b: Vec<i64> = (0..births.len() as i64)
        .filter(|&i| births[i as usize] <= birth_b && deaths[i as usize] > birth_b)
        .collect();

    jaccard_similarity(&clusters_a, &clusters_b)
}

// ---------------------------------------------------------------------------
// select_diverse_peaks
// ---------------------------------------------------------------------------

/// Port of `select_diverse_peaks`.
///
/// PORT NOTE: sorts candidate peaks by `total_persistence` descending via a
/// stable ascending sort followed by reversal (`np.argsort(...)[::-1]`),
/// so ties in persistence end up in descending original-index order among
/// themselves. Greedy selection rejects a peak on the first disqualifying
/// similarity comparison against an already-selected peak. `len(peaks) ==
/// 0` short-circuits before any allocation.
pub fn select_diverse_peaks(
    peaks: &[i64],
    total_persistence: &[f32],
    sizes: &[f32],
    births: &[f32],
    deaths: &[f32],
    min_similarity_threshold: f64,
    max_layers: usize,
) -> Vec<i64> {
    if peaks.is_empty() {
        return Vec::new();
    }

    // Stable ascending sort by persistence, then reverse -> ties end up in
    // descending original-index order (matches np.argsort(...)[::-1]).
    let mut order: Vec<usize> = (0..peaks.len()).collect();
    order.sort_by(|&a, &b| {
        total_persistence[peaks[a] as usize]
            .partial_cmp(&total_persistence[peaks[b] as usize])
            .unwrap()
    });
    order.reverse();
    let sorted_peaks: Vec<i64> = order.iter().map(|&i| peaks[i]).collect();

    let mut selected_peaks: Vec<i64> = Vec::with_capacity(max_layers);
    let mut selected_births: Vec<f32> = Vec::with_capacity(max_layers);

    for &peak in sorted_peaks.iter() {
        if selected_peaks.len() >= max_layers {
            break;
        }

        let birth_size = sizes[peak as usize];

        let mut is_diverse = true;
        for &selected_birth in selected_births.iter() {
            let similarity =
                estimate_cluster_similarity(births, deaths, birth_size, selected_birth);
            if similarity > min_similarity_threshold {
                is_diverse = false;
                break;
            }
        }

        if is_diverse {
            selected_peaks.push(peak);
            selected_births.push(birth_size);
        }
    }

    selected_peaks
}

// ---------------------------------------------------------------------------
// _build_cluster_tree / build_cluster_tree
// ---------------------------------------------------------------------------

/// Flat (parent_layer, parent_cluster, child_layer, child_cluster) edge, as
/// produced by `_build_cluster_tree`.
pub type ClusterTreeEdge = (usize, i64, usize, i64);

/// Port of `_build_cluster_tree`. `labels` is a list of label vectors,
/// index 0 = finest/base layer.
///
/// PORT NOTE: for each `upper_layer` from 1 upward, scans every
/// `lower_layer` from `upper_layer - 1` down to 0 (not just `upper_layer -
/// 1`), recording a `(parent_layer, parent_label, child_layer, child_label)`
/// edge the FIRST time that `(child_layer, child_label)` pair is seen
/// across the whole scan (first-seen-wins dedup per lower_layer, seeded
/// with `{-1}` so noise is pre-excluded). After the double loop, any
/// `(layer, label)` pair still unclaimed is attached under a synthetic
/// virtual root `(len(labels), 0)`.
pub fn _build_cluster_tree(labels: &[Vec<i64>]) -> Vec<ClusterTreeEdge> {
    let mut mapping: Vec<ClusterTreeEdge> = Vec::new();
    let mut found: Vec<HashSet<i64>> = (0..labels.len())
        .map(|_| HashSet::from([-1i64]))
        .collect();

    for upper_layer in 1..labels.len() {
        let mut upper_unique: Vec<i64> = labels[upper_layer].clone();
        upper_unique.sort_unstable();
        upper_unique.dedup();

        for lower_layer in (0..upper_layer).rev() {
            // Group labels[lower_layer] by upper-layer cluster via
            // argsort(upper_layer) + bincount/cumsum split, replicated
            // directly as a grouping by upper-layer label value.
            let mut groups: HashMap<i64, Vec<i64>> = HashMap::new();
            for (i, &upper_label) in labels[upper_layer].iter().enumerate() {
                groups
                    .entry(upper_label)
                    .or_default()
                    .push(labels[lower_layer][i]);
            }

            for &label in upper_unique.iter() {
                if label >= 0 {
                    if let Some(children) = groups.get(&label) {
                        for &child in children {
                            if child >= 0 && !found[lower_layer].contains(&child) {
                                mapping.push((upper_layer, label, lower_layer, child));
                                found[lower_layer].insert(child);
                            }
                        }
                    }
                }
            }
        }
    }

    for lower_layer in (0..labels.len()).rev() {
        let max_child = labels[lower_layer].iter().copied().max().unwrap_or(-1);
        if max_child >= 0 {
            for child in 0..=max_child {
                if !found[lower_layer].contains(&child) {
                    mapping.push((labels.len(), 0, lower_layer, child));
                }
            }
        }
    }

    mapping
}

/// Port of `build_cluster_tree`: turns the flat edge list into `dict[(layer,
/// cluster)] -> list[(layer, cluster)]`, child lists in edge-append order.
pub fn build_cluster_tree(labels: &[Vec<i64>]) -> HashMap<(usize, i64), Vec<(usize, i64)>> {
    let mut result: HashMap<(usize, i64), Vec<(usize, i64)>> = HashMap::new();
    for (parent_layer, parent_cluster, child_layer, child_cluster) in _build_cluster_tree(labels) {
        result
            .entry((parent_layer, parent_cluster))
            .or_default()
            .push((child_layer, child_cluster));
    }
    result
}

// ---------------------------------------------------------------------------
// find_duplicates
// ---------------------------------------------------------------------------

/// Port of `find_duplicates`. `knn_inds`/`knn_dists` are `n x k` flat
/// row-major matrices.
///
/// PORT NOTE: `duplicate_distance = max(knn_dists.T[0])` -- the single
/// largest first-nearest-neighbor distance across the WHOLE dataset sets a
/// GLOBAL threshold, replicated literally (not "fixed" to a per-point
/// threshold). Pairs canonicalized as `(min(i,k), max(i,k))`; `i == k` is
/// explicitly skipped.
pub fn find_duplicates(
    knn_inds: &[i64],
    knn_dists: &[f32],
    n: usize,
    k: usize,
) -> HashSet<(i64, i64)> {
    let duplicate_distance = (0..n)
        .map(|i| knn_dists[i * k])
        .fold(f32::NEG_INFINITY, f32::max);

    let mut duplicates: HashSet<(i64, i64)> = HashSet::new();
    for i in 0..n {
        for j in 0..k {
            if knn_dists[i * k + j] <= duplicate_distance {
                let kk = knn_inds[i * k + j];
                let ii = i as i64;
                if ii < kk {
                    duplicates.insert((ii, kk));
                } else if kk < ii {
                    duplicates.insert((kk, ii));
                }
                // ii == kk: skipped (continue)
            }
        }
    }
    duplicates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_peaks_empty_for_short_input() {
        assert_eq!(find_peaks(&[]), Vec::<usize>::new());
        assert_eq!(find_peaks(&[1.0]), Vec::<usize>::new());
    }

    #[test]
    fn find_peaks_monotone_has_no_peaks() {
        assert_eq!(find_peaks(&[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]), Vec::<usize>::new());
    }

    #[test]
    fn find_peaks_plateau_midpoint_rounds_left() {
        // plateau at indices 11..=12 (width 2) -> midpoint (11+12)//2 = 11.
        let x = [0.0, 1.0, 3.0, 2.0, 1.0, 0.0, 2.0, 5.0, 3.0, 1.0, 0.0, 4.0, 4.0, 1.0];
        assert_eq!(find_peaks(&x), vec![2, 7, 11]);
    }

    #[test]
    fn jaccard_empty_is_zero() {
        assert_eq!(jaccard_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn jaccard_basic() {
        assert_eq!(jaccard_similarity(&[1, 2, 3], &[2, 3, 4]), 0.5);
        assert_eq!(jaccard_similarity(&[1, 2, 3], &[1, 2, 3]), 1.0);
        assert_eq!(jaccard_similarity(&[1, 2, 3], &[4, 5, 6]), 0.0);
    }

    #[test]
    fn python_rounding_half_even() {
        assert_eq!(python_round_half_even(2.5), 2);
        assert_eq!(python_round_half_even(3.5), 4);
        assert_eq!(python_round_half_even(2.4), 2);
        assert_eq!(python_round_half_even(2.6), 3);
    }

    #[test]
    fn select_diverse_peaks_empty_short_circuits() {
        assert_eq!(
            select_diverse_peaks(&[], &[], &[], &[], &[], 0.2, 10),
            Vec::<i64>::new()
        );
    }

    #[test]
    fn find_duplicates_synthetic_pairs() {
        let knn_inds: Vec<i64> = vec![
            1, 2, 3, 0, 2, 3, 3, 0, 1, 2, 0, 1, 5, 6, 0, 4, 6, 0, 4, 5, 0,
        ];
        let knn_dists: Vec<f32> = vec![
            0.0, 0.5, 0.6, 0.0, 0.4, 0.5, 0.1, 0.5, 0.6, 0.1, 0.5, 0.6, 0.9, 0.95, 1.0, 0.9, 0.92,
            1.0, 0.9, 0.92, 1.0,
        ];
        let dups = find_duplicates(&knn_inds, &knn_dists, 7, 3);
        let mut sorted: Vec<(i64, i64)> = dups.into_iter().collect();
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (4, 5), (4, 6)]
        );
    }
}
