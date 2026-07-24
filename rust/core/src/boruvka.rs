//! Port of `evoc/boruvka.py`: Boruvka mutual-reachability minimum spanning
//! tree over a KD-tree.
//!
//! PORT NOTE (per PORT_NOTES.md): everything runs in SQUARED rdist space
//! until the very end, where `parallel_boruvka` applies one final `sqrt` to
//! the whole edge distance column. Every `numba.prange` becomes a plain
//! sequential loop (single-threaded wasm32 target); the block/component-
//! bound machinery is kept structurally intact with `n_threads` as a
//! parameter. At `n_threads == 1` the `reproducible` flag is bit-identical
//! to `reproducible=False` (it only bounds multithreaded races).

use crate::disjoint_set::{ds_find_rank, ds_rank_create, ds_union_by_rank, RankDisjointSet};
use crate::kdtree::{
    parallel_tree_query, point_to_node_lower_bound_rdist, rdist, NumbaKDTree,
};

/// Port of `merge_components`. `candidate_neighbors[i]` / `..._distances[i]`
/// hold, for point `i`, its best cross-component candidate; `point_components`
/// maps points to component ids. Returns the accepted MST edges as flat
/// row-major (k x 3) `(point_a, point_b, distance)` (still in rdist space).
///
/// PORT NOTE: the reference groups best edges per component in a numba typed
/// dict, keyed by component id, keeping the FIRST-improving edge; dict
/// iteration is insertion order. We reproduce that with an insertion-ordered
/// map (a Vec of (component -> slot) plus a lookup) so the merge order (and
/// therefore which edges survive union-find de-duplication) matches exactly.
pub fn merge_components(
    disjoint_set: &mut RankDisjointSet,
    candidate_neighbors: &[i32],
    candidate_neighbor_distances: &[f32],
    point_components: &[i64],
) -> Vec<[f32; 3]> {
    // component_edges: insertion-ordered map component_id -> (i, neighbor, dist)
    let mut order: Vec<i64> = Vec::new();
    let mut edges: std::collections::HashMap<i64, (i64, i64, f32)> =
        std::collections::HashMap::new();

    // Find the best edges from each component
    for i in 0..candidate_neighbors.len() {
        let from_component = point_components[i];
        if let Some(existing) = edges.get_mut(&from_component) {
            if candidate_neighbor_distances[i] < existing.2 {
                *existing = (
                    i as i64,
                    candidate_neighbors[i] as i64,
                    candidate_neighbor_distances[i],
                );
            }
        } else {
            order.push(from_component);
            edges.insert(
                from_component,
                (
                    i as i64,
                    candidate_neighbors[i] as i64,
                    candidate_neighbor_distances[i],
                ),
            );
        }
    }

    let mut result: Vec<[f32; 3]> = Vec::with_capacity(order.len());

    // Add the best edges to the edge set and merge the relevant components
    for comp in order.iter() {
        let edge = edges[comp];
        let from_component = ds_find_rank(disjoint_set, edge.0 as i32);
        let to_component = ds_find_rank(disjoint_set, edge.1 as i32);
        if from_component != to_component {
            result.push([edge.0 as f32, edge.1 as f32, edge.2]);
            ds_union_by_rank(disjoint_set, from_component, to_component);
        }
    }

    result
}

/// Port of `update_component_vectors`: refresh `point_components` from the
/// disjoint set, then propagate component labels up the tree (a node gets a
/// component only if all its points/children share it).
pub fn update_component_vectors(
    tree: &NumbaKDTree,
    disjoint_set: &mut RankDisjointSet,
    node_components: &mut [i64],
    point_components: &mut [i64],
) {
    for i in 0..point_components.len() {
        point_components[i] = ds_find_rank(disjoint_set, i as i32) as i64;
    }

    for i in (0..tree.idx_start.len()).rev() {
        let is_leaf = tree.is_leaf[i];
        let idx_start = tree.idx_start[i];
        let idx_end = tree.idx_end[i];

        if is_leaf {
            // every point in the node must share a component
            let candidate_component = point_components[tree.idx_array[idx_start]];
            let mut all_same = true;
            for j in (idx_start + 1)..idx_end {
                let idx = tree.idx_array[j];
                if point_components[idx] != candidate_component {
                    all_same = false;
                    break;
                }
            }
            if all_same {
                node_components[i] = candidate_component;
            }
        } else {
            let left = 2 * i + 1;
            let right = left + 1;
            if node_components[left] == node_components[right] {
                node_components[i] = node_components[left];
            }
        }
    }
}

/// Port of `component_aware_query_recursion`. `heap_p`/`heap_i` are the
/// single-slot best-candidate cell for the query point;
/// `component_nearest_neighbor_dist` is the single-slot component bound
/// (mutated in place). All distances are rdist (squared).
#[allow(clippy::too_many_arguments)]
fn component_aware_query_recursion(
    tree: &NumbaKDTree,
    node: usize,
    point: &[f32],
    heap_p: &mut f32,
    heap_i: &mut i32,
    current_core_distance: f32,
    core_distances: &[f32],
    current_component: i64,
    node_components: &[i64],
    point_components: &[i64],
    dist_lower_bound: f32,
    component_bound: &mut f32,
) {
    let is_leaf = tree.is_leaf[node];
    let idx_start = tree.idx_start[node];
    let idx_end = tree.idx_end[node];

    // Case 1a: query point is outside node radius -> trim
    if dist_lower_bound > *heap_p {
        return;
    }
    // Case 1b: can't improve on the best distance for this component -> trim
    else if dist_lower_bound > *component_bound || current_core_distance > *component_bound {
        return;
    }
    // Case 1c: node contains only points in same component as query -> trim
    else if node_components[node] == current_component {
        return;
    }
    // Case 2: leaf node -> update set of nearby points
    else if is_leaf {
        for i in idx_start..idx_end {
            let idx = tree.idx_array[i];
            if point_components[idx] != current_component
                && core_distances[idx] < *component_bound
            {
                let d = rdist(point, tree_data_row(tree, idx))
                    .max(current_core_distance)
                    .max(core_distances[idx]);
                if d < *heap_p {
                    *heap_p = d;
                    *heap_i = idx as i32;
                    if d < *component_bound {
                        *component_bound = d;
                    }
                }
            }
        }
    }
    // Case 3: internal node -> recurse into closer child first
    else {
        let left = 2 * node + 1;
        let right = left + 1;
        let dist_lower_bound_left = point_to_node_lower_bound_rdist(
            tree_bounds_lower(tree, left),
            tree_bounds_upper(tree, left),
            point,
        );
        let dist_lower_bound_right = point_to_node_lower_bound_rdist(
            tree_bounds_lower(tree, right),
            tree_bounds_upper(tree, right),
            point,
        );

        if dist_lower_bound_left <= dist_lower_bound_right {
            component_aware_query_recursion(
                tree, left, point, heap_p, heap_i, current_core_distance, core_distances,
                current_component, node_components, point_components, dist_lower_bound_left,
                component_bound,
            );
            component_aware_query_recursion(
                tree, right, point, heap_p, heap_i, current_core_distance, core_distances,
                current_component, node_components, point_components, dist_lower_bound_right,
                component_bound,
            );
        } else {
            component_aware_query_recursion(
                tree, right, point, heap_p, heap_i, current_core_distance, core_distances,
                current_component, node_components, point_components, dist_lower_bound_right,
                component_bound,
            );
            component_aware_query_recursion(
                tree, left, point, heap_p, heap_i, current_core_distance, core_distances,
                current_component, node_components, point_components, dist_lower_bound_left,
                component_bound,
            );
        }
    }
}

// -- small helpers mirroring NumbaKDTree's private row accessors ------------
// (NumbaKDTree stores data/bounds as flat row-major Vec<f32>; boruvka needs
// row slices so we recompute them here from the public fields.)
#[inline]
fn tree_data_row(tree: &NumbaKDTree, i: usize) -> &[f32] {
    &tree.data[i * tree.n_features..(i + 1) * tree.n_features]
}
#[inline]
fn tree_bounds_lower(tree: &NumbaKDTree, node: usize) -> &[f32] {
    &tree.node_bounds_lower[node * tree.n_features..(node + 1) * tree.n_features]
}
#[inline]
fn tree_bounds_upper(tree: &NumbaKDTree, node: usize) -> &[f32] {
    &tree.node_bounds_upper[node * tree.n_features..(node + 1) * tree.n_features]
}

/// Port of `boruvka_tree_query` (non-reproducible). Returns
/// `(candidate_distances, candidate_indices)` per point.
pub fn boruvka_tree_query(
    tree: &NumbaKDTree,
    node_components: &[i64],
    point_components: &[i64],
    core_distances: &[f32],
) -> (Vec<f32>, Vec<i32>) {
    let n = tree.n_samples();
    let mut candidate_distances = vec![f32::INFINITY; n];
    let mut candidate_indices = vec![-1i32; n];
    let mut component_nearest_neighbor_dist = vec![f32::INFINITY; n];

    for i in 0..n {
        let point = tree_data_row(tree, i);
        let distance_lower_bound = point_to_node_lower_bound_rdist(
            tree_bounds_lower(tree, 0),
            tree_bounds_upper(tree, 0),
            point,
        );
        let comp = point_components[i] as usize;
        // PORT NOTE: the reference's heap_p/heap_i are single-element views
        // into candidate_distances[i]/candidate_indices[i], and the
        // component bound is a live view into component_nearest_neighbor_dist
        // [comp] shared across ALL points of that component in this pass.
        // We pass a local `heap_p`/`heap_i` (distinct per point) but must
        // read/write the shared bound array in place so bound improvements
        // from earlier points in the same component prune later ones.
        let mut heap_p = candidate_distances[i];
        let mut heap_i = candidate_indices[i];
        let mut bound = component_nearest_neighbor_dist[comp];
        component_aware_query_recursion(
            tree,
            0,
            point,
            &mut heap_p,
            &mut heap_i,
            core_distances[i],
            core_distances,
            point_components[i],
            node_components,
            point_components,
            distance_lower_bound,
            &mut bound,
        );
        candidate_distances[i] = heap_p;
        candidate_indices[i] = heap_i;
        component_nearest_neighbor_dist[comp] = bound;
    }

    (candidate_distances, candidate_indices)
}

/// Port of `calculate_block_size`.
pub fn calculate_block_size(n_components: usize, n_points: usize, num_threads: usize) -> usize {
    let points_per_component = if n_components == 0 {
        n_points as f64
    } else {
        n_points as f64 / n_components as f64
    };

    let block_size = if points_per_component < 10.0 {
        num_threads * 512
    } else if points_per_component < 100.0 {
        num_threads * 128
    } else if points_per_component < 1000.0 {
        num_threads * 32
    } else {
        num_threads * 8
    };

    // Ensure reasonable bounds
    std::cmp::max(num_threads, std::cmp::min(block_size, n_points / 4 + 1))
}

/// Port of `update_component_bounds_from_block`.
fn update_component_bounds_from_block(
    component_nearest_neighbor_dist: &mut [f32],
    block_component_bounds: &[f32],
    point_components: &[i64],
    block_start: usize,
    block_end: usize,
) {
    for i in block_start..block_end {
        let component = point_components[i] as usize;
        let block_bound = block_component_bounds[i - block_start];
        if block_bound < component_nearest_neighbor_dist[component] {
            component_nearest_neighbor_dist[component] = block_bound;
        }
    }
}

/// Port of `boruvka_tree_query_reproducible`: block-based processing that,
/// under real multithreading, avoids races by only publishing per-component
/// bounds after each block. Sequentially it produces the same candidates as
/// `boruvka_tree_query`.
pub fn boruvka_tree_query_reproducible(
    tree: &NumbaKDTree,
    node_components: &[i64],
    point_components: &[i64],
    core_distances: &[f32],
    block_size: usize,
) -> (Vec<f32>, Vec<i32>) {
    let n = tree.n_samples();
    let mut candidate_distances = vec![f32::INFINITY; n];
    let mut candidate_indices = vec![-1i32; n];
    let mut component_nearest_neighbor_dist = vec![f32::INFINITY; n];

    let mut max_block_component_bounds = vec![f32::INFINITY; block_size];

    let mut block_start = 0;
    while block_start < n {
        let block_end = std::cmp::min(block_start + block_size, n);
        let block_size_actual = block_end - block_start;

        for b in max_block_component_bounds[..block_size_actual].iter_mut() {
            *b = f32::INFINITY;
        }

        for i in block_start..block_end {
            let point = tree_data_row(tree, i);
            let distance_lower_bound = point_to_node_lower_bound_rdist(
                tree_bounds_lower(tree, 0),
                tree_bounds_upper(tree, 0),
                point,
            );
            let current_component = point_components[i] as usize;
            // use current global bound for this component
            let mut local_component_bound = component_nearest_neighbor_dist[current_component];
            let mut heap_p = candidate_distances[i];
            let mut heap_i = candidate_indices[i];

            component_aware_query_recursion(
                tree,
                0,
                point,
                &mut heap_p,
                &mut heap_i,
                core_distances[i],
                core_distances,
                point_components[i],
                node_components,
                point_components,
                distance_lower_bound,
                &mut local_component_bound,
            );

            candidate_distances[i] = heap_p;
            candidate_indices[i] = heap_i;
            // store the potentially updated bound for this point
            max_block_component_bounds[i - block_start] = local_component_bound;
        }

        update_component_bounds_from_block(
            &mut component_nearest_neighbor_dist,
            &max_block_component_bounds,
            point_components,
            block_start,
            block_end,
        );

        block_start += block_size;
    }

    (candidate_distances, candidate_indices)
}

/// Port of `initialize_boruvka_from_knn`. `knn_indices`/`knn_distances` are
/// flat row-major n_points x k (k = min_samples + 1). Edge weights are in
/// rdist space (NOT sqrt'd). Returns flat-ish `Vec<[f32; 3]>`.
///
/// PORT NOTE: for each point, take the FIRST neighbor `j` (from j = 1,
/// skipping self at j = 0) with `core_distances[i] >= core_distances[k]`,
/// edge weight `max(core_distances[i], knn_distances[i, j])`, and break --
/// first qualifying, not best. Edge candidates are computed in f64 in the
/// reference then cast to f32 on return; the endpoints are exact integers
/// and the weight is a max of f32 values, so the f64 round-trip is exact.
pub fn initialize_boruvka_from_knn(
    knn_indices: &[i32],
    knn_distances: &[f32],
    core_distances: &[f32],
    k: usize,
    disjoint_set: &mut RankDisjointSet,
) -> Vec<[f32; 3]> {
    let n_points = knn_indices.len() / k;
    // component_edges[i] = Some((i, k, weight)) or None (unset / edge[0] < 0)
    let mut component_edges: Vec<Option<(i64, i64, f64)>> = vec![None; n_points];

    for i in 0..n_points {
        for j in 1..k {
            let kk = knn_indices[i * k + j];
            if core_distances[i] >= core_distances[kk as usize] {
                let edge_weight = core_distances[i].max(knn_distances[i * k + j]);
                component_edges[i] = Some((i as i64, kk as i64, edge_weight as f64));
                break;
            }
        }
    }

    let mut result: Vec<[f32; 3]> = Vec::new();

    for edge in component_edges.into_iter() {
        let edge = match edge {
            Some(e) => e,
            None => continue,
        };
        let from_component = ds_find_rank(disjoint_set, edge.0 as i32);
        let to_component = ds_find_rank(disjoint_set, edge.1 as i32);
        if from_component != to_component {
            result.push([edge.0 as f32, edge.1 as f32, edge.2 as f32]);
            ds_union_by_rank(disjoint_set, from_component, to_component);
        }
    }

    result
}

/// Port of `parallel_boruvka`. Builds the MST over `tree`, returning flat
/// row-major (n-1) x 3 edges `(point_a, point_b, distance)` with the
/// distance column sqrt-transformed at the very end.
pub fn parallel_boruvka(
    tree: &NumbaKDTree,
    n_threads: usize,
    min_samples: usize,
    reproducible: bool,
) -> Vec<[f32; 3]> {
    let n = tree.n_samples();
    let mut components_disjoint_set = ds_rank_create(n);
    let mut point_components: Vec<i64> = (0..n as i64).collect();
    let mut node_components: Vec<i64> = vec![-1; tree.idx_start.len()];

    let core_distances: Vec<f32>;
    let mut all_edges: Vec<[f32; 3]> = Vec::with_capacity(n.saturating_sub(1));

    if min_samples > 1 {
        let k = min_samples + 1;
        let (distances, neighbors) = parallel_tree_query(tree, &tree.data, k, true);
        // core_distances = distances.T[-1] = last column of each row
        core_distances = (0..n).map(|i| distances[i * k + (k - 1)]).collect();
        let initial_edges = initialize_boruvka_from_knn(
            &neighbors,
            &distances,
            &core_distances,
            k,
            &mut components_disjoint_set,
        );
        update_component_vectors(
            tree,
            &mut components_disjoint_set,
            &mut node_components,
            &mut point_components,
        );
        all_edges.extend_from_slice(&initial_edges);
    } else {
        core_distances = vec![0.0f32; n];
        let k = 2;
        let (distances, neighbors) = parallel_tree_query(tree, &tree.data, k, true);
        let initial_edges = initialize_boruvka_from_knn(
            &neighbors,
            &distances,
            &core_distances,
            k,
            &mut components_disjoint_set,
        );
        update_component_vectors(
            tree,
            &mut components_disjoint_set,
            &mut node_components,
            &mut point_components,
        );
        all_edges.extend_from_slice(&initial_edges);
    }

    // Count initial components after initialization
    let mut n_components = unique_count(&point_components);

    while n_components > 1 {
        let (candidate_distances, candidate_indices) = if reproducible {
            let block_size = calculate_block_size(n_components, n, n_threads);
            boruvka_tree_query_reproducible(
                tree,
                &node_components,
                &point_components,
                &core_distances,
                block_size,
            )
        } else {
            boruvka_tree_query(tree, &node_components, &point_components, &core_distances)
        };

        let new_edges = merge_components(
            &mut components_disjoint_set,
            &candidate_indices,
            &candidate_distances,
            &point_components,
        );

        n_components -= new_edges.len();

        update_component_vectors(
            tree,
            &mut components_disjoint_set,
            &mut node_components,
            &mut point_components,
        );

        all_edges.extend_from_slice(&new_edges);
    }

    // final sqrt of the whole distance column
    for e in all_edges.iter_mut() {
        e[2] = e[2].sqrt();
    }

    all_edges
}

/// `len(np.unique(...))` for the component labels.
fn unique_count(components: &[i64]) -> usize {
    let mut v = components.to_vec();
    v.sort_unstable();
    v.dedup();
    v.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kdtree::build_kdtree;

    /// Flatten Vec<[f32;3]> edges' endpoints as i32 pairs (rounded).
    fn edge_endpoints(edges: &[[f32; 3]]) -> Vec<(i32, i32)> {
        edges.iter().map(|e| (e[0] as i32, e[1] as i32)).collect()
    }

    #[test]
    fn tiny_collinear_mst_min_samples_1() {
        // 4 points on a line at x=0,1,2,3 -> MST is the chain of unit edges.
        let data = [0.0f32, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        let tree = build_kdtree(&data, 2, 40);
        let edges = parallel_boruvka(&tree, 1, 1, false);
        assert_eq!(edges.len(), 3);
        // total MST weight = 1 + 1 + 1 = 3 (euclidean)
        let total: f32 = edges.iter().map(|e| e[2]).sum();
        assert!((total - 3.0).abs() < 1e-4, "total={total}");
        // every edge connects adjacent-or-valid points; connectivity: union of
        // endpoints covers all 4 points
        let mut seen = std::collections::HashSet::new();
        for (a, b) in edge_endpoints(&edges) {
            seen.insert(a);
            seen.insert(b);
        }
        assert_eq!(seen.len(), 4);
    }

    #[test]
    fn all_identical_points_mst_is_zero_length() {
        let n = 8;
        let data: Vec<f32> = std::iter::repeat([2.0f32, -1.0]).take(n).flatten().collect();
        let tree = build_kdtree(&data, 2, 40);
        let edges = parallel_boruvka(&tree, 1, 1, false);
        assert_eq!(edges.len(), n - 1);
        for e in &edges {
            assert_eq!(e[2], 0.0);
        }
    }

    #[test]
    fn two_clusters_mst_connectivity() {
        // two tight clusters far apart; MST must span all points (n-1 edges).
        let mut data: Vec<f32> = Vec::new();
        for i in 0..6 {
            data.push((i % 2) as f32 * 0.01);
            data.push((i / 2) as f32 * 0.01);
        }
        for i in 0..6 {
            data.push(100.0 + (i % 2) as f32 * 0.01);
            data.push(100.0 + (i / 2) as f32 * 0.01);
        }
        let tree = build_kdtree(&data, 2, 40);
        let edges = parallel_boruvka(&tree, 1, 1, false);
        assert_eq!(edges.len(), 11);
        // exactly one long bridging edge (>~ 100) joins the two clusters
        let long_edges = edges.iter().filter(|e| e[2] > 50.0).count();
        assert_eq!(long_edges, 1);
    }

    #[test]
    fn reproducible_matches_nonreproducible_single_thread() {
        let data: Vec<f32> = (0..40)
            .flat_map(|i| [(i as f32 * 1.3).sin(), (i as f32 * 0.7).cos()])
            .collect();
        let tree = build_kdtree(&data, 2, 10);
        let a = parallel_boruvka(&tree, 1, 5, false);
        let b = parallel_boruvka(&tree, 1, 5, true);
        assert_eq!(a.len(), b.len());
        for (ea, eb) in a.iter().zip(b.iter()) {
            assert_eq!(ea[0], eb[0]);
            assert_eq!(ea[1], eb[1]);
            assert_eq!(ea[2].to_bits(), eb[2].to_bits());
        }
    }

    #[test]
    fn calculate_block_size_bounds() {
        // 80 points, many components -> points_per_component < 10 -> 512*nt
        // but clamped to max(nt, min(512, 80/4+1=21)) = 21
        assert_eq!(calculate_block_size(80, 80, 1), 21);
        // single component, 80 points -> ppc=80 (<100) -> 128, clamp to 21
        assert_eq!(calculate_block_size(1, 80, 1), 21);
    }
}
