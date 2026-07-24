//! Port of `evoc/cluster_trees.py`: MST -> linkage tree -> condensed tree
//! -> excess-of-mass cluster extraction -> point labels / membership
//! strengths.
//!
//! This is the most branch-order-sensitive module in the port. The
//! condense_tree five-way branch, the BFS node ordering, and the typed-dict
//! insertion orders are all reproduced faithfully (see the individual
//! PORT NOTEs). `numba.prange` becomes a plain sequential loop.
//!
//! PORT NOTE: `mst_to_linkage_tree` output is f64 (numpy default) even
//! though MST distances are f32; the linkage row convention puts the LARGER
//! component id in the left slot. CondensedTree fields are parent:i64,
//! child:i64, lambda_val:f32, child_size:i64.

use crate::disjoint_set::{ds_find_rank, ds_rank_create, ds_union_by_rank};
use std::collections::{HashMap, HashSet};

/// Condensed tree as struct-of-arrays (matching the reference namedtuple
/// `CondensedTree(parent, child, lambda_val, child_size)`).
#[derive(Debug, Clone, PartialEq)]
pub struct CondensedTree {
    pub parent: Vec<i64>,
    pub child: Vec<i64>,
    pub lambda_val: Vec<f32>,
    pub child_size: Vec<i64>,
}

/// Linkage tree: n-1 rows of `(left, right, delta, size)`, delta/size stored
/// as f64 to match numpy's default dtype for the `result` array.
#[derive(Debug, Clone)]
pub struct LinkageTree {
    /// Flat row-major (n-1) x 4, f64.
    pub rows: Vec<[f64; 4]>,
}

// ---------------------------------------------------------------------------
// Linkage merge union-find (its own scheme, distinct from disjoint_set.rs)
// ---------------------------------------------------------------------------

struct LinkageMergeData {
    parent: Vec<i64>,
    size: Vec<i64>,
    next: i64,
}

fn create_linkage_merge_data(base_size: usize) -> LinkageMergeData {
    let mut size = vec![1i64; base_size];
    size.extend(std::iter::repeat(0i64).take(base_size - 1));
    LinkageMergeData {
        parent: vec![-1; 2 * base_size - 1],
        size,
        next: base_size as i64,
    }
}

/// Port of `linkage_merge_find` (find with full path relabel to root).
fn linkage_merge_find(lm: &mut LinkageMergeData, node: i64) -> i64 {
    let mut node = node;
    let relabel_start = node;
    while lm.parent[node as usize] != -1 && lm.parent[node as usize] != node {
        node = lm.parent[node as usize];
    }
    lm.parent[node as usize] = node;

    // label up to the root
    let mut relabel = relabel_start;
    while lm.parent[relabel as usize] != node {
        let next_relabel = lm.parent[relabel as usize];
        lm.parent[relabel as usize] = node;
        relabel = next_relabel;
    }

    node
}

fn linkage_merge_join(lm: &mut LinkageMergeData, left: i64, right: i64) {
    lm.size[lm.next as usize] = lm.size[left as usize] + lm.size[right as usize];
    lm.parent[left as usize] = lm.next;
    lm.parent[right as usize] = lm.next;
    lm.next += 1;
}

/// Port of `mst_to_linkage_tree`. `sorted_mst` is flat row-major m x 3
/// `(left, right, delta)`; returns a [`LinkageTree`] (m rows x 4, f64).
pub fn mst_to_linkage_tree(sorted_mst: &[[f32; 3]]) -> LinkageTree {
    let m = sorted_mst.len();
    let mut rows = vec![[0.0f64; 4]; m];

    let n_samples = m + 1;
    let mut lm = create_linkage_merge_data(n_samples);

    for index in 0..m {
        let left = sorted_mst[index][0] as i64;
        let right = sorted_mst[index][1] as i64;
        let delta = sorted_mst[index][2];

        let left_component = linkage_merge_find(&mut lm, left);
        let right_component = linkage_merge_find(&mut lm, right);

        if left_component > right_component {
            rows[index][0] = left_component as f64;
            rows[index][1] = right_component as f64;
        } else {
            rows[index][1] = left_component as f64;
            rows[index][0] = right_component as f64;
        }

        rows[index][2] = delta as f64;
        rows[index][3] =
            (lm.size[left_component as usize] + lm.size[right_component as usize]) as f64;

        linkage_merge_join(&mut lm, left_component, right_component);
    }

    LinkageTree { rows }
}

// ---------------------------------------------------------------------------
// Condense
// ---------------------------------------------------------------------------

/// Port of `bfs_from_hierarchy`: breadth-first node id list from a linkage
/// hierarchy rooted at `bfs_root`. `hierarchy` is the linkage rows.
///
/// PORT NOTE: the level-by-level append order (left child then right child,
/// per node, per level) determines the `node_list` iteration order in
/// `condense_tree` and hence node relabeling -- reproduced exactly.
pub fn bfs_from_hierarchy(hierarchy: &LinkageTree, bfs_root: i64, num_points: i64) -> Vec<i64> {
    let mut to_process = vec![bfs_root];
    let mut result: Vec<i64> = Vec::new();

    while !to_process.is_empty() {
        result.extend_from_slice(&to_process);
        let mut next_to_process: Vec<i64> = Vec::new();
        for &n in to_process.iter() {
            if n >= num_points {
                let i = (n - num_points) as usize;
                next_to_process.push(hierarchy.rows[i][0] as i64);
                next_to_process.push(hierarchy.rows[i][1] as i64);
            }
        }
        to_process = next_to_process;
    }

    result
}

/// Port of `eliminate_branch`. Writes point children of `branch_node` under
/// `parent_node`, flattening internal descendants and marking them
/// `ignore`. Returns the updated write cursor `idx`.
#[allow(clippy::too_many_arguments)]
fn eliminate_branch(
    branch_node: i64,
    parent_node: i64,
    lambda_value: f32,
    parents: &mut [i64],
    children: &mut [i64],
    lambdas: &mut [f32],
    idx: usize,
    ignore: &mut [bool],
    hierarchy: &LinkageTree,
    num_points: i64,
) -> usize {
    let mut idx = idx;
    if branch_node < num_points {
        parents[idx] = parent_node;
        children[idx] = branch_node;
        lambdas[idx] = lambda_value;
        idx += 1;
    } else {
        for sub_node in bfs_from_hierarchy(hierarchy, branch_node, num_points) {
            if sub_node < num_points {
                children[idx] = sub_node;
                parents[idx] = parent_node;
                lambdas[idx] = lambda_value;
                idx += 1;
            } else {
                ignore[sub_node as usize] = true;
            }
        }
    }
    idx
}

/// Port of `condense_tree`.
///
/// PORT NOTE: the five-way branch is deliberately ordered for the common
/// (left singleton) case first; the exact branch ORDER (not just the
/// logical union) is load-bearing because it drives `next_label` assignment
/// and the write order into the output arrays. Reproduced verbatim.
pub fn condense_tree(hierarchy: &LinkageTree, min_cluster_size: i64) -> CondensedTree {
    let m = hierarchy.rows.len();
    let root = (2 * m) as i64;
    let num_points = (m + 1) as i64;
    let mut next_label = num_points + 1;

    let node_list = bfs_from_hierarchy(hierarchy, root, num_points);

    let mut relabel = vec![0i64; (root + 1) as usize];
    relabel[root as usize] = num_points;

    let cap = root as usize;
    let mut parents = vec![1i64; cap];
    let mut children = vec![0i64; cap];
    let mut lambdas = vec![0.0f32; cap];
    let mut sizes = vec![1i64; cap];
    let mut ignore = vec![false; (root + 1) as usize];

    let mut idx = 0usize;

    for &node in node_list.iter() {
        if ignore[node as usize] || node < num_points {
            continue;
        }

        let parent_node = relabel[node as usize];
        let row = &hierarchy.rows[(node - num_points) as usize];
        let left = row[0] as i64;
        let right = row[1] as i64;
        let d = row[2];
        let lambda_value: f32 = if d > 0.0 {
            (1.0 / d) as f32
        } else {
            f32::INFINITY
        };

        let left_count = if left >= num_points {
            hierarchy.rows[(left - num_points) as usize][3] as i64
        } else {
            1
        };
        let right_count = if right >= num_points {
            hierarchy.rows[(right - num_points) as usize][3] as i64
        } else {
            1
        };

        if left < num_points && right_count >= min_cluster_size {
            relabel[right as usize] = parent_node;
            parents[idx] = parent_node;
            children[idx] = left;
            lambdas[idx] = lambda_value;
            idx += 1;
        } else if left_count < min_cluster_size && right_count >= min_cluster_size {
            relabel[right as usize] = parent_node;
            idx = eliminate_branch(
                left,
                parent_node,
                lambda_value,
                &mut parents,
                &mut children,
                &mut lambdas,
                idx,
                &mut ignore,
                hierarchy,
                num_points,
            );
        } else if left_count >= min_cluster_size && right_count < min_cluster_size {
            relabel[left as usize] = parent_node;
            idx = eliminate_branch(
                right,
                parent_node,
                lambda_value,
                &mut parents,
                &mut children,
                &mut lambdas,
                idx,
                &mut ignore,
                hierarchy,
                num_points,
            );
        } else if left_count < min_cluster_size && right_count < min_cluster_size {
            idx = eliminate_branch(
                left,
                parent_node,
                lambda_value,
                &mut parents,
                &mut children,
                &mut lambdas,
                idx,
                &mut ignore,
                hierarchy,
                num_points,
            );
            idx = eliminate_branch(
                right,
                parent_node,
                lambda_value,
                &mut parents,
                &mut children,
                &mut lambdas,
                idx,
                &mut ignore,
                hierarchy,
                num_points,
            );
        } else {
            relabel[left as usize] = next_label;
            parents[idx] = parent_node;
            children[idx] = next_label;
            lambdas[idx] = lambda_value;
            sizes[idx] = left_count;
            next_label += 1;
            idx += 1;

            relabel[right as usize] = next_label;
            parents[idx] = parent_node;
            children[idx] = next_label;
            lambdas[idx] = lambda_value;
            sizes[idx] = right_count;
            next_label += 1;
            idx += 1;
        }
    }

    CondensedTree {
        parent: parents[..idx].to_vec(),
        child: children[..idx].to_vec(),
        lambda_val: lambdas[..idx].to_vec(),
        child_size: sizes[..idx].to_vec(),
    }
}

// ---------------------------------------------------------------------------
// Leaves / scoring / cluster tree
// ---------------------------------------------------------------------------

/// Port of `extract_leaves`. Leaf = node id (>= n_points) that never appears
/// as a parent with `child_size > 1`.
///
/// PORT NOTE: `n_points = parent.min()`, `n_nodes = parent.max() + 1`.
pub fn extract_leaves(condensed_tree: &CondensedTree) -> Vec<i64> {
    if condensed_tree.parent.is_empty() {
        return Vec::new();
    }
    let n_nodes = *condensed_tree.parent.iter().max().unwrap() + 1;
    let n_points = *condensed_tree.parent.iter().min().unwrap();
    let mut leaf_indicator = vec![true; n_nodes as usize];
    for x in leaf_indicator[..n_points as usize].iter_mut() {
        *x = false;
    }

    for (&parent, &child_size) in condensed_tree
        .parent
        .iter()
        .zip(condensed_tree.child_size.iter())
    {
        if child_size > 1 {
            leaf_indicator[parent as usize] = false;
        }
    }

    (0..n_nodes)
        .filter(|&i| leaf_indicator[i as usize])
        .collect()
}

/// Port of `score_condensed_tree_nodes`. Returns `(order, scores)` where
/// `order` preserves the reference typed-dict insertion order (needed
/// nowhere for parity but kept for fidelity) and `scores` maps node -> stability.
fn score_condensed_tree_nodes(condensed_tree: &CondensedTree) -> HashMap<i64, f32> {
    let mut result: HashMap<i64, f32> = HashMap::new();

    for i in 0..condensed_tree.parent.len() {
        let parent = condensed_tree.parent[i];
        let contribution = condensed_tree.lambda_val[i] * condensed_tree.child_size[i] as f32;
        *result.entry(parent).or_insert(0.0) += contribution;

        if condensed_tree.child_size[i] > 1 {
            let child = condensed_tree.child[i];
            *result.entry(child).or_insert(0.0) -= contribution;
        }
    }

    result
}

/// Port of `cluster_tree_from_condensed_tree` (mask child_size > 1).
pub fn cluster_tree_from_condensed_tree(condensed_tree: &CondensedTree) -> CondensedTree {
    let mut ct = CondensedTree {
        parent: Vec::new(),
        child: Vec::new(),
        lambda_val: Vec::new(),
        child_size: Vec::new(),
    };
    for i in 0..condensed_tree.parent.len() {
        if condensed_tree.child_size[i] > 1 {
            ct.parent.push(condensed_tree.parent[i]);
            ct.child.push(condensed_tree.child[i]);
            ct.lambda_val.push(condensed_tree.lambda_val[i]);
            ct.child_size.push(condensed_tree.child_size[i]);
        }
    }
    ct
}

// ---------------------------------------------------------------------------
// Excess-of-mass extraction
// ---------------------------------------------------------------------------

/// Children of `node` in the cluster tree, in row order.
fn children_of(cluster_tree: &CondensedTree, node: i64) -> Vec<i64> {
    cluster_tree
        .parent
        .iter()
        .zip(cluster_tree.child.iter())
        .filter_map(|(&p, &c)| if p == node { Some(c) } else { None })
        .collect()
}

/// Port of `unselect_below_node`.
fn unselect_below_node(node: i64, cluster_tree: &CondensedTree, selected: &mut HashMap<i64, bool>) {
    for child in children_of(cluster_tree, node) {
        unselect_below_node(child, cluster_tree, selected);
        selected.insert(child, false);
    }
}

/// Port of `eom_recursion`.
fn eom_recursion(
    node: i64,
    cluster_tree: &CondensedTree,
    node_scores: &HashMap<i64, f32>,
    selected: &mut HashMap<i64, bool>,
) -> f32 {
    let current_score = *node_scores.get(&node).unwrap_or(&0.0);

    let children = children_of(cluster_tree, node);
    let mut child_score_total = 0.0f32;
    for child_node in children {
        child_score_total += eom_recursion(child_node, cluster_tree, node_scores, selected);
    }

    if child_score_total > current_score {
        child_score_total
    } else {
        selected.insert(node, true);
        unselect_below_node(node, cluster_tree, selected);
        current_score
    }
}

/// Port of `extract_eom_clusters`. Returns the selected cluster node ids.
///
/// PORT NOTE: the reference builds the result from an unordered numba
/// typed-dict iteration, so order is not significant -- callers compare as
/// sorted sets (the fixture stores them sorted).
pub fn extract_eom_clusters(
    condensed_tree: &CondensedTree,
    cluster_tree: &CondensedTree,
    allow_single_cluster: bool,
) -> Vec<i64> {
    let node_scores = score_condensed_tree_nodes(condensed_tree);
    // selected_clusters keyed by every node appearing in node_scores
    let mut selected: HashMap<i64, bool> = node_scores.keys().map(|&k| (k, false)).collect();

    if cluster_tree.parent.is_empty() {
        return Vec::new();
    }

    let cluster_tree_root = *cluster_tree.parent.iter().min().unwrap();

    if allow_single_cluster {
        eom_recursion(cluster_tree_root, cluster_tree, &node_scores, &mut selected);
    } else if node_scores.len() > 1 {
        let root_children = children_of(cluster_tree, cluster_tree_root);
        for child_node in root_children {
            eom_recursion(child_node, cluster_tree, &node_scores, &mut selected);
        }
    }

    selected
        .iter()
        .filter_map(|(&node, &sel)| if sel { Some(node) } else { None })
        .collect()
}

// ---------------------------------------------------------------------------
// Labels / strengths
// ---------------------------------------------------------------------------

/// Port of `get_single_cluster_label_vector` (single-cluster path).
///
/// PORT NOTE: not exercised by the n80 fixture (which yields 4 clusters);
/// ported for completeness and covered by a colocated unit test.
pub fn get_single_cluster_label_vector(
    tree: &CondensedTree,
    cluster: i64,
    cluster_selection_epsilon: f32,
    n_samples: usize,
) -> Vec<i64> {
    if tree.parent.is_empty() {
        return vec![-1; n_samples];
    }

    let mut result = vec![-1i64; n_samples];
    // max_lambda = tree.lambda_val[tree.parent == cluster].max()
    let max_lambda = tree
        .parent
        .iter()
        .zip(tree.lambda_val.iter())
        .filter_map(|(&p, &l)| if p == cluster { Some(l) } else { None })
        .fold(f32::NEG_INFINITY, f32::max);

    for i in 0..tree.child.len() {
        let n = tree.child[i] as usize;
        let cur_lambda = tree.lambda_val[i];
        if cluster_selection_epsilon > 0.0 {
            if cur_lambda >= 1.0 / cluster_selection_epsilon {
                result[n] = 0;
            } else {
                result[n] = -1;
            }
        } else if cur_lambda >= max_lambda {
            result[n] = 0;
        }
    }

    result
}

/// Port of `get_cluster_label_vector`.
pub fn get_cluster_label_vector(
    tree: &CondensedTree,
    clusters: &[i64],
    cluster_selection_epsilon: f32,
    n_samples: usize,
) -> Vec<i64> {
    if clusters.len() == 1 {
        return get_single_cluster_label_vector(
            tree,
            clusters[0],
            cluster_selection_epsilon,
            n_samples,
        );
    }

    if tree.parent.is_empty() {
        return vec![-1; n_samples];
    }
    let root_cluster = *tree.parent.iter().min().unwrap();
    let mut result = vec![-1i64; n_samples];

    // cluster_label_map = {c: n for n, c in enumerate(np.sort(clusters))}
    let mut sorted_clusters = clusters.to_vec();
    sorted_clusters.sort_unstable();
    let cluster_label_map: HashMap<i64, i64> = sorted_clusters
        .iter()
        .enumerate()
        .map(|(n, &c)| (c, n as i64))
        .collect();

    let ds_size = std::cmp::max(
        *tree.parent.iter().max().unwrap() + 1,
        *tree.child.iter().max().unwrap() + 1,
    );
    let mut ds = ds_rank_create(ds_size as usize);
    let cluster_set: HashSet<i64> = clusters.iter().copied().collect();

    for n in 0..tree.parent.len() {
        let child = tree.child[n];
        let parent = tree.parent[n];
        if !cluster_set.contains(&child) {
            ds_union_by_rank(&mut ds, parent as i32, child as i32);
        }
    }

    for n in 0..n_samples {
        let cluster = ds_find_rank(&mut ds, n as i32) as i64;
        if cluster <= root_cluster {
            result[n] = -1;
        } else {
            result[n] = cluster_label_map[&cluster];
        }
    }

    result
}

/// Port of `mask_condensed_tree`.
pub fn mask_condensed_tree(condensed_tree: &CondensedTree, mask: &[bool]) -> CondensedTree {
    let mut ct = CondensedTree {
        parent: Vec::new(),
        child: Vec::new(),
        lambda_val: Vec::new(),
        child_size: Vec::new(),
    };
    for (i, &m) in mask.iter().enumerate() {
        if m {
            ct.parent.push(condensed_tree.parent[i]);
            ct.child.push(condensed_tree.child[i]);
            ct.lambda_val.push(condensed_tree.lambda_val[i]);
            ct.child_size.push(condensed_tree.child_size[i]);
        }
    }
    ct
}

/// Port of `traverse_upwards` (cluster-epsilon merge helper).
pub fn traverse_upwards(
    cluster_tree: &CondensedTree,
    min_persistence: f32,
    root: i64,
    segment: i64,
) -> i64 {
    // parent = cluster_tree.parent[cluster_tree.child == segment][0]
    let parent = cluster_tree
        .child
        .iter()
        .position(|&c| c == segment)
        .map(|i| cluster_tree.parent[i])
        .expect("segment has no parent");
    if parent == root {
        return root;
    }
    let parent_eps = 1.0
        / cluster_tree
            .child
            .iter()
            .position(|&c| c == parent)
            .map(|i| cluster_tree.lambda_val[i])
            .expect("parent not found");
    if parent_eps >= min_persistence {
        parent
    } else {
        traverse_upwards(cluster_tree, min_persistence, root, parent)
    }
}

/// Port of `in_set_parallel`.
pub fn in_set_parallel(values: &[i64], targets: &HashSet<i64>) -> Vec<bool> {
    values.iter().map(|v| targets.contains(v)).collect()
}

/// Port of `segments_in_branch`.
pub fn segments_in_branch(cluster_tree: &CondensedTree, segment: i64) -> HashSet<i64> {
    let mut result: HashSet<i64> = HashSet::new();
    let mut to_process: HashSet<i64> = HashSet::from([segment]);

    while !to_process.is_empty() {
        result.extend(to_process.iter().copied());
        let mask = in_set_parallel(&cluster_tree.parent, &to_process);
        to_process = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m { Some(cluster_tree.child[i]) } else { None })
            .collect();
    }

    result
}

/// Port of `cluster_epsilon_search`.
///
/// PORT NOTE: `selected` is a list (insertion order preserved); `processed`
/// a set. The reference iterates `clusters` in the given order.
pub fn cluster_epsilon_search(
    clusters: &[i64],
    cluster_tree: &CondensedTree,
    min_persistence: f32,
) -> Vec<i64> {
    let mut selected: Vec<i64> = Vec::new();
    let mut processed: HashSet<i64> = HashSet::new();

    let root = *cluster_tree.parent.iter().min().unwrap();
    for &cluster in clusters {
        // eps = 1 / cluster_tree.lambda_val[cluster_tree.child == cluster][0]
        let eps = 1.0
            / cluster_tree
                .child
                .iter()
                .position(|&c| c == cluster)
                .map(|i| cluster_tree.lambda_val[i])
                .expect("cluster not found in cluster tree");
        if eps < min_persistence {
            if !processed.contains(&cluster) {
                let parent = traverse_upwards(cluster_tree, min_persistence, root, cluster);
                selected.push(parent);
                processed.extend(segments_in_branch(cluster_tree, parent));
            }
        } else {
            selected.push(cluster);
        }
    }
    selected
}

/// Port of `get_cluster_labelling_at_cut`. `linkage_tree` rows are
/// `(left, right, delta, size)`; returns per-point cluster labels.
pub fn get_cluster_labelling_at_cut(
    linkage_tree: &LinkageTree,
    cut: f64,
    min_cluster_size: i64,
) -> Vec<i64> {
    let m = linkage_tree.rows.len();
    let root = (2 * m) as i64;
    let num_points = m + 1;
    let mut result = vec![0i64; num_points];
    let mut ds = ds_rank_create((root + 1) as usize);

    let mut cluster = num_points as i64;
    for i in 0..m {
        if linkage_tree.rows[i][2] < cut {
            ds_union_by_rank(&mut ds, linkage_tree.rows[i][0] as i32, cluster as i32);
            ds_union_by_rank(&mut ds, linkage_tree.rows[i][1] as i32, cluster as i32);
        }
        cluster += 1;
    }

    let mut cluster_size = vec![0i64; cluster as usize];
    for n in 0..num_points {
        let c = ds_find_rank(&mut ds, n as i32) as i64;
        cluster_size[c as usize] += 1;
        result[n] = c;
    }

    // unique labels in ascending order
    let mut unique_labels = result.clone();
    unique_labels.sort_unstable();
    unique_labels.dedup();

    let mut cluster_label_map: HashMap<i64, i64> = HashMap::from([(-1, -1)]);
    let mut cluster_label = 0i64;
    for c in unique_labels {
        if cluster_size[c as usize] < min_cluster_size {
            cluster_label_map.insert(c, -1);
        } else {
            cluster_label_map.insert(c, cluster_label);
            cluster_label += 1;
        }
    }

    for n in 0..num_points {
        result[n] = cluster_label_map[&result[n]];
    }

    result
}

/// Port of `max_lambdas`. Returns cluster -> max singleton lambda.
fn max_lambdas(tree: &CondensedTree, clusters: &HashSet<i64>) -> HashMap<i64, f32> {
    let mut result: HashMap<i64, f32> = clusters.iter().map(|&c| (c, 0.0)).collect();

    for n in 0..tree.parent.len() {
        let cluster = tree.parent[n];
        if clusters.contains(&cluster) && tree.child_size[n] == 1 {
            let e = result.get_mut(&cluster).unwrap();
            *e = e.max(tree.lambda_val[n]);
        }
    }

    result
}

/// Port of `get_point_membership_strength_vector`.
pub fn get_point_membership_strength_vector(
    tree: &CondensedTree,
    clusters: &[i64],
    labels: &[i64],
) -> Vec<f32> {
    let mut result = vec![0.0f32; labels.len()];
    let cluster_set: HashSet<i64> = clusters.iter().copied().collect();
    let deaths = max_lambdas(tree, &cluster_set);
    let root_cluster = *tree.parent.iter().min().unwrap();

    // cluster_index_map = {n: c for n, c in enumerate(np.sort(clusters))}
    let mut sorted_clusters = clusters.to_vec();
    sorted_clusters.sort_unstable();
    let cluster_index_map: HashMap<i64, i64> = sorted_clusters
        .iter()
        .enumerate()
        .map(|(n, &c)| (n as i64, c))
        .collect();

    for n in 0..tree.child.len() {
        let point = tree.child[n];
        if point >= root_cluster || labels[point as usize] < 0 {
            continue;
        }

        let cluster = cluster_index_map[&labels[point as usize]];
        let max_lambda = deaths[&cluster];
        if max_lambda == 0.0 || !tree.lambda_val[n].is_finite() {
            result[point as usize] = 1.0;
        } else {
            let lambda_val = tree.lambda_val[n].min(max_lambda);
            result[point as usize] = lambda_val / max_lambda;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_linkage_three_points() {
        // MST edges (already sorted by delta): 0-1 @1.0, 1-2 @2.0
        // n_samples = 3. First merge joins {0},{1} -> node 3; second joins
        // node 3 with {2} -> node 4.
        let mst = [[0.0f32, 1.0, 1.0], [1.0, 2.0, 2.0]];
        let lt = mst_to_linkage_tree(&mst);
        assert_eq!(lt.rows.len(), 2);
        // row0: components 1 and 0 -> larger first (1,0), delta 1, size 2
        assert_eq!(lt.rows[0][0] as i64, 1);
        assert_eq!(lt.rows[0][1] as i64, 0);
        assert_eq!(lt.rows[0][2], 1.0);
        assert_eq!(lt.rows[0][3] as i64, 2);
        // row1: joins node 3 (the merged pair) and point 2 -> larger first
        assert_eq!(lt.rows[1][0] as i64, 3);
        assert_eq!(lt.rows[1][1] as i64, 2);
        assert_eq!(lt.rows[1][3] as i64, 3);
    }

    #[test]
    fn condense_empty_and_degenerate() {
        // A linkage that condenses to nothing meaningful: all clusters below
        // min_cluster_size get eliminated to point children. With a 3-point
        // chain and min_cluster_size huge, everything is small.
        let mst = [[0.0f32, 1.0, 1.0], [1.0, 2.0, 2.0]];
        let lt = mst_to_linkage_tree(&mst);
        let ct = condense_tree(&lt, 10);
        // root splits into a small-left/small-right or singleton cases; all
        // children are points (child_size 1). No sub-cluster survives.
        assert!(ct.parent.iter().all(|&p| p == 3)); // single root cluster id = num_points = 3
        assert!(ct.child_size.iter().all(|&s| s == 1));
        // extract_leaves: the only cluster node is the root -> it's a leaf
        let leaves = extract_leaves(&ct);
        assert_eq!(leaves, vec![3]);
    }

    #[test]
    fn extract_leaves_empty_tree() {
        let ct = CondensedTree {
            parent: vec![],
            child: vec![],
            lambda_val: vec![],
            child_size: vec![],
        };
        assert_eq!(extract_leaves(&ct), Vec::<i64>::new());
    }

    #[test]
    fn single_cluster_label_path() {
        // Build a condensed tree with a single surviving cluster (the root),
        // and points as children. get_cluster_label_vector with clusters of
        // len 1 must route to the single-cluster path and label points with
        // max lambda as cluster 0.
        let ct = CondensedTree {
            parent: vec![3, 3, 3],
            child: vec![0, 1, 2],
            lambda_val: vec![1.0, 1.0, 0.5],
            child_size: vec![1, 1, 1],
        };
        let labels = get_cluster_label_vector(&ct, &[3], 0.0, 3);
        // max lambda among parent==3 is 1.0; points 0,1 (lambda 1.0) get 0,
        // point 2 (lambda 0.5 < 1.0) stays -1.
        assert_eq!(labels, vec![0, 0, -1]);
    }

    #[test]
    fn cluster_tree_masks_singletons() {
        let ct = CondensedTree {
            parent: vec![3, 3, 4, 4],
            child: vec![4, 0, 1, 2],
            lambda_val: vec![1.0, 1.0, 2.0, 2.0],
            child_size: vec![3, 1, 1, 1],
        };
        let cut = cluster_tree_from_condensed_tree(&ct);
        // only the child_size>1 row (3->4) survives
        assert_eq!(cut.parent, vec![3]);
        assert_eq!(cut.child, vec![4]);
        assert_eq!(cut.child_size, vec![3]);
    }

    // ---------------------------------------------------------------------
    // Deferred epsilon-search / cut family. No phase-6/8 JSON fixture
    // exercises these directly (cluster_epsilon_search / traverse_upwards /
    // segments_in_branch / in_set_parallel / get_cluster_labelling_at_cut are
    // dead code with zero call sites anywhere in the upstream reference
    // package outside cluster_trees.py itself; mask_condensed_tree IS used,
    // by clustering.py's build_cluster_layers, and is covered end-to-end by
    // the clustering::build_cluster_layers_n126 fixture test). Ground truth
    // below was independently computed by re-implementing each function
    // verbatim in plain numpy (no numba) and running it -- see
    // scratchpad/verify_ct.py / verify_cut.py used during porting.

    fn synthetic_cluster_tree() -> CondensedTree {
        // root = 5; 5 -> 6 (eps 1.0), 5 -> 7 (eps 1.0); 6 -> 8 (eps 0.25),
        // 6 -> 9 (eps 0.5).
        CondensedTree {
            parent: vec![5, 5, 6, 6],
            child: vec![6, 7, 8, 9],
            lambda_val: vec![1.0, 1.0, 4.0, 2.0],
            child_size: vec![3, 2, 2, 1],
        }
    }

    #[test]
    fn mask_condensed_tree_filters_rows() {
        let ct = synthetic_cluster_tree();
        let mask: Vec<bool> = ct.child.iter().map(|&c| c >= 7).collect();
        let masked = mask_condensed_tree(&ct, &mask);
        assert_eq!(masked.parent, vec![5, 6, 6]);
        assert_eq!(masked.child, vec![7, 8, 9]);
        assert_eq!(masked.lambda_val, vec![1.0, 4.0, 2.0]);
        assert_eq!(masked.child_size, vec![2, 2, 1]);
    }

    #[test]
    fn traverse_upwards_climbs_to_first_qualifying_ancestor() {
        let ct = synthetic_cluster_tree();
        // parent(8) = 6, eps(6) = 1/1.0 = 1.0 >= 0.3 -> returns 6 directly.
        assert_eq!(traverse_upwards(&ct, 0.3, 5, 8), 6);
    }

    #[test]
    fn segments_in_branch_collects_whole_subtree() {
        let ct = synthetic_cluster_tree();
        let mut got: Vec<i64> = segments_in_branch(&ct, 6).into_iter().collect();
        got.sort_unstable();
        assert_eq!(got, vec![6, 8, 9]);
    }

    #[test]
    fn in_set_parallel_membership() {
        let ct = synthetic_cluster_tree();
        let targets: HashSet<i64> = [5i64, 6].into_iter().collect();
        assert_eq!(
            in_set_parallel(&ct.parent, &targets),
            vec![true, true, true, true]
        );
    }

    #[test]
    fn cluster_epsilon_search_merges_low_persistence_segments() {
        let ct = synthetic_cluster_tree();
        // cluster 8: eps=0.25 < 0.3 -> traverse_upwards -> 6, mark {6,8,9} processed.
        // cluster 9: eps=0.5, not < 0.3 -> appended directly (branch order is
        // per-cluster, not gated by `processed` outside the eps<threshold arm).
        // cluster 7: eps=1.0, not < 0.3 -> appended directly.
        let result = cluster_epsilon_search(&[8, 9, 7], &ct, 0.3);
        assert_eq!(result, vec![6, 9, 7]);
    }

    #[test]
    fn get_cluster_labelling_at_cut_basic() {
        // 5 points, 4 linkage rows: (left,right,delta,size).
        let lt = LinkageTree {
            rows: vec![
                [0.0, 1.0, 1.0, 2.0],
                [2.0, 3.0, 1.0, 2.0],
                [5.0, 6.0, 2.0, 4.0],
                [7.0, 4.0, 3.0, 5.0],
            ],
        };
        assert_eq!(
            get_cluster_labelling_at_cut(&lt, 1.5, 2),
            vec![0, 0, 1, 1, -1]
        );
        assert_eq!(
            get_cluster_labelling_at_cut(&lt, 2.5, 1),
            vec![0, 0, 0, 0, 1]
        );
    }
}
