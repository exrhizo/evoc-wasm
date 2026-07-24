//! Port of `evoc/numba_kdtree.py` (KD-tree build + k-NN query).
//!
//! `kdtree_to_numba` (sklearn interop glue) is intentionally not ported.
//!
//! Data layout: the reference's 2-D `float32[:, ::1]` arrays are stored as
//! flat row-major `Vec<f32>` plus an `n_features` stride. The reference's
//! 3-D `node_bounds` (2 x n_nodes x n_features) is stored as two flat
//! vectors, `node_bounds_lower` (== `node_bounds[0]`) and
//! `node_bounds_upper` (== `node_bounds[1]`), matching the fixtures'
//! `node_bounds_lower` / `node_bounds_upper` keys.
//!
//! PORT NOTE: the reference stores tree indices as `intp` (pointer-sized
//! signed). Index values are all small non-negative array positions, so we
//! use `usize`; loop variables that the algorithms drive below zero
//! (insertion-sort/partition scans) locally use `isize` to preserve the
//! reference's signed loop-exit conditions.

/// Port of the `NumbaKDTree` namedtuple.
pub struct NumbaKDTree {
    /// Row-major n_samples x n_features.
    pub data: Vec<f32>,
    pub n_features: usize,
    pub idx_array: Vec<usize>,
    pub idx_start: Vec<usize>,
    pub idx_end: Vec<usize>,
    pub radius: Vec<f32>,
    pub is_leaf: Vec<bool>,
    /// `node_bounds[0]` in the reference: row-major n_nodes x n_features.
    pub node_bounds_lower: Vec<f32>,
    /// `node_bounds[1]` in the reference: row-major n_nodes x n_features.
    pub node_bounds_upper: Vec<f32>,
}

impl NumbaKDTree {
    pub fn n_samples(&self) -> usize {
        self.data.len() / self.n_features
    }

    pub fn n_nodes(&self) -> usize {
        self.is_leaf.len()
    }

    fn data_row(&self, i: usize) -> &[f32] {
        &self.data[i * self.n_features..(i + 1) * self.n_features]
    }

    fn bounds_lower(&self, node: usize) -> &[f32] {
        &self.node_bounds_lower[node * self.n_features..(node + 1) * self.n_features]
    }

    fn bounds_upper(&self, node: usize) -> &[f32] {
        &self.node_bounds_upper[node * self.n_features..(node + 1) * self.n_features]
    }
}

/// Port of `_init_node`: compute node bounds (sequential min/max) and
/// radius (f32 accumulation of squared half-extents, then sqrt).
#[allow(clippy::too_many_arguments)]
fn _init_node(
    data: &[f32],
    n_features: usize,
    node_bounds_lower: &mut [f32],
    node_bounds_upper: &mut [f32],
    idx_array: &[usize],
    idx_start_array: &mut [usize],
    idx_end_array: &mut [usize],
    radius_array: &mut [f32],
    node: usize,
    idx_start: usize,
    idx_end: usize,
) {
    let lower_bounds = &mut node_bounds_lower[node * n_features..(node + 1) * n_features];
    let upper_bounds = &mut node_bounds_upper[node * n_features..(node + 1) * n_features];

    // determine Node bounds
    for j in 0..n_features {
        lower_bounds[j] = f32::INFINITY;
        upper_bounds[j] = f32::NEG_INFINITY;
    }

    for i in idx_start..idx_end {
        let data_row = &data[idx_array[i] * n_features..(idx_array[i] + 1) * n_features];
        for j in 0..n_features {
            lower_bounds[j] = lower_bounds[j].min(data_row[j]);
            upper_bounds[j] = upper_bounds[j].max(data_row[j]);
        }
    }

    let mut radius: f32 = 0.0;
    for j in 0..n_features {
        let diff: f32 = (upper_bounds[j] - lower_bounds[j]).abs() * 0.5;
        radius += diff * diff;
    }

    idx_start_array[node] = idx_start;
    idx_end_array[node] = idx_end;

    radius_array[node] = radius.sqrt();
}

/// Port of `_find_node_split_dim`: dimension of maximum spread.
fn _find_node_split_dim(
    data: &[f32],
    n_features: usize,
    idx_array: &[usize],
    idx_start: usize,
    idx_end: usize,
) -> usize {
    let mut result: usize = 0;
    let mut max_spread: f32 = 0.0;

    for j in 0..n_features {
        let mut max_val = data[idx_array[idx_start] * n_features + j];
        let mut min_val = max_val;
        for i in (idx_start + 1)..idx_end {
            let val = data[idx_array[i] * n_features + j];
            max_val = max_val.max(val);
            min_val = min_val.min(val);
        }

        let spread = max_val - min_val;

        if spread > max_spread {
            max_spread = spread;
            result = j;
        }
    }

    result
}

/// Port of `_compare_indices`: compare data values on `axis`, breaking ties
/// by original index (like sklearn).
fn _compare_indices(data: &[f32], n_features: usize, axis: usize, idx1: usize, idx2: usize) -> i8 {
    let val1 = data[idx1 * n_features + axis];
    let val2 = data[idx2 * n_features + axis];

    if val1 < val2 {
        -1
    } else if val1 > val2 {
        1
    } else {
        // Break ties using original index values (like sklearn)
        #[allow(clippy::comparison_chain)]
        if idx1 < idx2 {
            -1
        } else if idx1 > idx2 {
            1
        } else {
            0
        }
    }
}

/// Port of `_insertion_sort_indices`.
fn _insertion_sort_indices(
    data: &[f32],
    n_features: usize,
    idx_array: &mut [usize],
    axis: usize,
    left: usize,
    right: usize,
) {
    for i in (left + 1)..right {
        let key_idx = idx_array[i];
        let mut j = i as isize - 1;

        while j >= left as isize
            && _compare_indices(data, n_features, axis, idx_array[j as usize], key_idx) > 0
        {
            idx_array[(j + 1) as usize] = idx_array[j as usize];
            j -= 1;
        }

        idx_array[(j + 1) as usize] = key_idx;
    }
}

/// Port of `_sift_down_indices`.
fn _sift_down_indices(
    data: &[f32],
    n_features: usize,
    idx_array: &mut [usize],
    axis: usize,
    offset: usize,
    start: usize,
    end: usize,
) {
    let mut root = start;

    while root * 2 + 1 < end {
        let child = root * 2 + 1;
        let mut swap = root;

        if _compare_indices(
            data,
            n_features,
            axis,
            idx_array[offset + swap],
            idx_array[offset + child],
        ) < 0
        {
            swap = child;
        }

        if child + 1 < end
            && _compare_indices(
                data,
                n_features,
                axis,
                idx_array[offset + swap],
                idx_array[offset + child + 1],
            ) < 0
        {
            swap = child + 1;
        }

        if swap == root {
            return;
        }

        idx_array.swap(offset + root, offset + swap);
        root = swap;
    }
}

/// Port of `_heapsort_indices`.
fn _heapsort_indices(
    data: &[f32],
    n_features: usize,
    idx_array: &mut [usize],
    axis: usize,
    left: usize,
    right: usize,
) {
    let size = right - left;

    // Build heap
    let mut i = size as isize / 2 - 1;
    while i >= 0 {
        _sift_down_indices(data, n_features, idx_array, axis, left, i as usize, size);
        i -= 1;
    }

    // Extract elements
    for i in (1..size).rev() {
        idx_array.swap(left, left + i);
        _sift_down_indices(data, n_features, idx_array, axis, left, 0, i);
    }
}

/// Port of `_median_of_three_pivot`.
fn _median_of_three_pivot(
    data: &[f32],
    n_features: usize,
    idx_array: &mut [usize],
    axis: usize,
    left: usize,
    right: usize,
) -> usize {
    let mid = (left + right - 1) / 2;

    let mut idx_left = idx_array[left];
    let mut idx_mid = idx_array[mid];
    let mut idx_right = idx_array[right - 1];

    // Sort the three candidates
    if _compare_indices(data, n_features, axis, idx_left, idx_mid) > 0 {
        idx_array.swap(left, mid);
        std::mem::swap(&mut idx_left, &mut idx_mid);
    }

    if _compare_indices(data, n_features, axis, idx_mid, idx_right) > 0 {
        idx_array.swap(mid, right - 1);
        std::mem::swap(&mut idx_mid, &mut idx_right);

        if _compare_indices(data, n_features, axis, idx_left, idx_mid) > 0 {
            idx_array.swap(left, mid);
        }
    }

    mid
}

/// Port of `_partition_indices` (Hoare-style partition around a pivot moved
/// to the end; comparisons via `_compare_indices` against the pivot's
/// original index, so ties resolve deterministically).
fn _partition_indices(
    data: &[f32],
    n_features: usize,
    idx_array: &mut [usize],
    axis: usize,
    left: usize,
    right: usize,
    pivot_idx: usize,
) -> usize {
    // Move pivot to end
    idx_array.swap(pivot_idx, right - 1);
    // (the reference also computes an unused local `pivot_value` here)
    let pivot_original_idx = idx_array[right - 1];

    let mut i = left as isize;
    let mut j = right as isize - 2;

    loop {
        // Find element from left that should be on right
        while i <= j
            && _compare_indices(data, n_features, axis, idx_array[i as usize], pivot_original_idx)
                < 0
        {
            i += 1;
        }

        // Find element from right that should be on left
        while i <= j
            && _compare_indices(data, n_features, axis, idx_array[j as usize], pivot_original_idx)
                >= 0
        {
            j -= 1;
        }

        if i >= j {
            break;
        }

        // Swap elements
        idx_array.swap(i as usize, j as usize);
        i += 1;
        j -= 1;
    }

    // Move pivot to final position
    idx_array.swap(i as usize, right - 1);
    i as usize
}

/// Port of `_introselect_impl`.
#[allow(clippy::comparison_chain)]
fn _introselect_impl(
    data: &[f32],
    n_features: usize,
    idx_array: &mut [usize],
    axis: usize,
    mut left: usize,
    mut right: usize,
    nth: usize,
    mut depth_limit: usize,
) {
    while right - left > 16 {
        if depth_limit == 0 {
            // Fall back to heapsort when recursion gets too deep
            _heapsort_indices(data, n_features, idx_array, axis, left, right);
            return;
        }

        depth_limit -= 1;

        // Choose pivot using median-of-three
        let pivot_idx = _median_of_three_pivot(data, n_features, idx_array, axis, left, right);

        // Partition around pivot
        let pivot_pos =
            _partition_indices(data, n_features, idx_array, axis, left, right, pivot_idx);

        // Recurse on the appropriate side
        if nth < pivot_pos {
            right = pivot_pos;
        } else if nth > pivot_pos {
            left = pivot_pos + 1;
        } else {
            // Found the nth element
            return;
        }
    }

    // Use insertion sort for small subarrays
    _insertion_sort_indices(data, n_features, idx_array, axis, left, right);
}

/// Port of `_introselect`.
fn _introselect(
    data: &[f32],
    n_features: usize,
    idx_array: &mut [usize],
    axis: usize,
    left: usize,
    right: usize,
    nth: usize,
) {
    let size = right - left;

    // Use insertion sort for small arrays
    if size <= 16 {
        _insertion_sort_indices(data, n_features, idx_array, axis, left, right);
        return;
    }

    // Calculate maximum recursion depth (2 * log2(size))
    let max_depth = 2 * ((size as f64).log2() as usize);
    _introselect_impl(data, n_features, idx_array, axis, left, right, nth, max_depth);
}

/// Port of `_recursive_build_tree`.
#[allow(clippy::too_many_arguments)]
fn _recursive_build_tree(
    data: &[f32],
    n_features: usize,
    idx_array: &mut [usize],
    idx_start_array: &mut [usize],
    idx_end_array: &mut [usize],
    radius_array: &mut [f32],
    is_leaf_array: &mut [bool],
    node_bounds_lower: &mut [f32],
    node_bounds_upper: &mut [f32],
    idx_start: usize,
    idx_end: usize,
    node: usize,
) {
    let n_points = idx_end - idx_start;
    let n_mid = n_points / 2;

    _init_node(
        data,
        n_features,
        node_bounds_lower,
        node_bounds_upper,
        idx_array,
        idx_start_array,
        idx_end_array,
        radius_array,
        node,
        idx_start,
        idx_end,
    );

    if 2 * node + 1 >= is_leaf_array.len() {
        is_leaf_array[node] = true;
    } else if idx_end - idx_start < 2 {
        is_leaf_array[node] = true;
    } else {
        is_leaf_array[node] = false;
        let axis = _find_node_split_dim(data, n_features, idx_array, idx_start, idx_end);
        _introselect(
            data,
            n_features,
            idx_array,
            axis,
            idx_start,
            idx_end,
            idx_start + n_mid,
        );
        _recursive_build_tree(
            data,
            n_features,
            idx_array,
            idx_start_array,
            idx_end_array,
            radius_array,
            is_leaf_array,
            node_bounds_lower,
            node_bounds_upper,
            idx_start,
            idx_start + n_mid,
            2 * node + 1,
        );
        _recursive_build_tree(
            data,
            n_features,
            idx_array,
            idx_start_array,
            idx_end_array,
            radius_array,
            is_leaf_array,
            node_bounds_lower,
            node_bounds_upper,
            idx_start + n_mid,
            idx_end,
            2 * node + 2,
        );
    }
}

/// Port of `build_kdtree`. `data` is row-major n_samples x n_features.
pub fn build_kdtree(data: &[f32], n_features: usize, leaf_size: usize) -> NumbaKDTree {
    assert!(n_features > 0 && data.len() % n_features == 0);
    let n_samples = data.len() / n_features;

    assert!(leaf_size >= 1, "leaf_size must be greater than or equal to 1");

    // determine number of levels in the tree, and from this
    // the number of nodes in the tree.  This results in leaf nodes
    // with numbers of points between leaf_size and 2 * leaf_size
    //
    // PORT NOTE: `(n_samples - 1) / leaf_size` is Python 3 true division
    // (f64), and `int(...)` truncates toward zero -- computed here exactly
    // as written, in f64, then truncated.
    let ratio = (n_samples as f64 - 1.0) / leaf_size as f64;
    let n_levels = ((if ratio > 1.0 { ratio } else { 1.0 }).log2() + 1.0) as i64;
    let n_nodes = ((1i64 << n_levels) - 1) as usize;

    // allocate arrays for storage
    let mut idx_array: Vec<usize> = (0..n_samples).collect();
    let mut idx_start_array = vec![0usize; n_nodes];
    let mut idx_end_array = vec![0usize; n_nodes];
    let mut radius_array = vec![0.0f32; n_nodes];
    let mut is_leaf_array = vec![false; n_nodes];
    let mut node_bounds_lower = vec![0.0f32; n_nodes * n_features];
    let mut node_bounds_upper = vec![0.0f32; n_nodes * n_features];

    _recursive_build_tree(
        data,
        n_features,
        &mut idx_array,
        &mut idx_start_array,
        &mut idx_end_array,
        &mut radius_array,
        &mut is_leaf_array,
        &mut node_bounds_lower,
        &mut node_bounds_upper,
        0,
        n_samples,
        0,
    );

    NumbaKDTree {
        data: data.to_vec(),
        n_features,
        idx_array,
        idx_start: idx_start_array,
        idx_end: idx_end_array,
        radius: radius_array,
        is_leaf: is_leaf_array,
        node_bounds_lower,
        node_bounds_upper,
    }
}

/// Port of `rdist`: squared euclidean distance, sequential f32 accumulation.
pub fn rdist(x: &[f32], y: &[f32]) -> f32 {
    let mut result: f32 = 0.0;
    let dim = x.len();
    for i in 0..dim {
        let diff: f32 = x[i] - y[i];
        result += diff * diff;
    }

    result
}

/// Port of `point_to_node_lower_bound_rdist`.
///
/// PORT NOTE: the reference's parameter names are swapped relative to what
/// callers pass: every call site passes `node_bounds[0]` (the LOWER bounds)
/// as the parameter named `upper`, and `node_bounds[1]` (the UPPER bounds)
/// as the parameter named `lower`. The body is written for that swapped
/// convention (`upper[i] > pt[i]` measures distance below the lower bound),
/// so the math is correct at the call sites. Ported verbatim, names and
/// all: call this as `point_to_node_lower_bound_rdist(lower_bounds,
/// upper_bounds, pt)` exactly like the reference call sites do.
pub fn point_to_node_lower_bound_rdist(upper: &[f32], lower: &[f32], pt: &[f32]) -> f32 {
    let mut result: f32 = 0.0;
    let dim = pt.len();
    for i in 0..dim {
        let d_lo: f32 = if upper[i] > pt[i] { upper[i] - pt[i] } else { 0.0 };
        let d_hi: f32 = if pt[i] > lower[i] { pt[i] - lower[i] } else { 0.0 };
        let d: f32 = d_lo + d_hi;
        result += d * d;
    }

    result
}

/// Port of `simple_heap_push`: push (p, n) onto a fixed-size max-heap of
/// priorities, returning 1 if inserted and 0 if rejected.
///
/// PORT NOTE: the reference's numba `locals` declare the loop indices as
/// u16, which is behaviorally irrelevant for heap sizes < 32768 (evoc uses
/// small k); we use usize.
pub fn simple_heap_push(priorities: &mut [f32], indices: &mut [i32], p: f32, n: i32) -> i32 {
    if p >= priorities[0] {
        return 0;
    }

    let size = priorities.len();

    // insert val at position zero
    priorities[0] = p;
    indices[0] = n;

    // descend the heap, swapping values until the max heap criterion is met
    let mut i = 0usize;
    loop {
        let ic1 = 2 * i + 1;
        let ic2 = ic1 + 1;

        let i_swap;
        if ic1 >= size {
            break;
        } else if ic2 >= size {
            if priorities[ic1] > p {
                i_swap = ic1;
            } else {
                break;
            }
        } else if priorities[ic1] >= priorities[ic2] {
            if p < priorities[ic1] {
                i_swap = ic1;
            } else {
                break;
            }
        } else if p < priorities[ic2] {
            i_swap = ic2;
        } else {
            break;
        }

        priorities[i] = priorities[i_swap];
        indices[i] = indices[i_swap];

        i = i_swap;
    }

    priorities[i] = p;
    indices[i] = n;

    1
}

/// Port of `siftdown` (max-heap sift-down over paired arrays).
pub fn siftdown(heap1: &mut [f32], heap2: &mut [i32], elt: usize) {
    let mut elt = elt;
    while elt * 2 + 1 < heap1.len() {
        let left_child = elt * 2 + 1;
        let right_child = left_child + 1;
        let mut swap = elt;

        if heap1[swap] < heap1[left_child] {
            swap = left_child;
        }

        if right_child < heap1.len() && heap1[swap] < heap1[right_child] {
            swap = right_child;
        }

        if swap == elt {
            break;
        } else {
            heap1.swap(elt, swap);
            heap2.swap(elt, swap);
            elt = swap;
        }
    }
}

/// Port of `deheap_sort`: row-wise heapsort of paired (distances, indices)
/// matrices, each row a max-heap; `n_cols` is the row width (`k`).
///
/// PORT NOTE: the reference iterates rows with `numba.prange`
/// (parallel-for); the core targets single-threaded wasm32, so this is a
/// plain sequential loop -- iteration order is irrelevant since rows are
/// independent.
pub fn deheap_sort(distances: &mut [f32], indices: &mut [i32], n_cols: usize) {
    let n_rows = indices.len() / n_cols;
    for i in 0..n_rows {
        let row_d = &mut distances[i * n_cols..(i + 1) * n_cols];
        let row_i = &mut indices[i * n_cols..(i + 1) * n_cols];
        // starting from the end of the array and moving back
        for j in (1..n_cols).rev() {
            row_i.swap(0, j);
            row_d.swap(0, j);

            siftdown(&mut row_d[..j], &mut row_i[..j], 0);
        }
    }
}

/// Port of `tree_query_recursion`.
pub fn tree_query_recursion(
    tree: &NumbaKDTree,
    node: usize,
    point: &[f32],
    heap_p: &mut [f32],
    heap_i: &mut [i32],
    dist_lower_bound: f32,
) {
    // Get node information
    let idx_start = tree.idx_start[node];
    let idx_end = tree.idx_end[node];
    let is_leaf = tree.is_leaf[node];

    // ------------------------------------------------------------
    // Case 1: query point is outside node radius:
    //         trim it from the query
    if dist_lower_bound > heap_p[0] {
        return;
    }
    // ------------------------------------------------------------
    // Case 2: this is a leaf node.  Update set of nearby points
    else if is_leaf {
        for i in idx_start..idx_end {
            let idx = tree.idx_array[i];
            let d = rdist(point, tree.data_row(idx));
            if d < heap_p[0] {
                simple_heap_push(heap_p, heap_i, d, idx as i32);
            }
        }
    }
    // ------------------------------------------------------------
    // Case 3: Node is not a leaf.  Recursively query subnodes
    //         starting with the closest
    else {
        let left = 2 * node + 1;
        let right = left + 1;
        let dist_lower_bound_left =
            point_to_node_lower_bound_rdist(tree.bounds_lower(left), tree.bounds_upper(left), point);
        let dist_lower_bound_right = point_to_node_lower_bound_rdist(
            tree.bounds_lower(right),
            tree.bounds_upper(right),
            point,
        );

        // recursively query subnodes
        if dist_lower_bound_left <= dist_lower_bound_right {
            tree_query_recursion(tree, left, point, heap_p, heap_i, dist_lower_bound_left);
            tree_query_recursion(tree, right, point, heap_p, heap_i, dist_lower_bound_right);
        } else {
            tree_query_recursion(tree, right, point, heap_p, heap_i, dist_lower_bound_right);
            tree_query_recursion(tree, left, point, heap_p, heap_i, dist_lower_bound_left);
        }
    }
}

/// Port of `parallel_tree_query`: k-NN query for each row of `data`
/// (row-major n_queries x tree.n_features). Returns flat row-major
/// (distances, indices), each n_queries x k, sorted ascending per row.
///
/// If `output_rdist` is true the distances are squared euclidean (rdist);
/// otherwise `np.sqrt` is applied to the whole heap BEFORE `deheap_sort`,
/// exactly as the reference does.
///
/// PORT NOTE: the reference's `numba.prange` over query rows is a plain
/// sequential loop here (single-threaded wasm32 target); rows are
/// independent so results are identical.
pub fn parallel_tree_query(
    tree: &NumbaKDTree,
    data: &[f32],
    k: usize,
    output_rdist: bool,
) -> (Vec<f32>, Vec<i32>) {
    let n_features = tree.n_features;
    let n_queries = data.len() / n_features;

    let mut result_p = vec![f32::INFINITY; n_queries * k];
    let mut result_i = vec![-1i32; n_queries * k];

    for i in 0..n_queries {
        let point = &data[i * n_features..(i + 1) * n_features];
        let distance_lower_bound =
            point_to_node_lower_bound_rdist(tree.bounds_lower(0), tree.bounds_upper(0), point);
        let heap_priorities = &mut result_p[i * k..(i + 1) * k];
        let heap_indices = &mut result_i[i * k..(i + 1) * k];
        tree_query_recursion(
            tree,
            0,
            point,
            heap_priorities,
            heap_indices,
            distance_lower_bound,
        );
    }

    if !output_rdist {
        for v in result_p.iter_mut() {
            *v = v.sqrt();
        }
    }
    deheap_sort(&mut result_p, &mut result_i, k);
    (result_p, result_i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_point_tree() {
        let data = [1.0f32, 2.0];
        let tree = build_kdtree(&data, 2, 40);
        assert_eq!(tree.n_nodes(), 1);
        assert!(tree.is_leaf[0]);
        assert_eq!(tree.idx_array, vec![0]);
        assert_eq!(tree.idx_start, vec![0]);
        assert_eq!(tree.idx_end, vec![1]);
        assert_eq!(tree.radius, vec![0.0]);
        assert_eq!(tree.node_bounds_lower, vec![1.0, 2.0]);
        assert_eq!(tree.node_bounds_upper, vec![1.0, 2.0]);
    }

    #[test]
    fn all_identical_points_tree_queries_correctly() {
        // 40 identical points with leaf_size 5 forces multiple split levels
        // where every spread is zero (split dim falls back to 0, introselect
        // ties break on index).
        let n = 40;
        let data: Vec<f32> = std::iter::repeat([3.0f32, -1.0]).take(n).flatten().collect();
        let tree = build_kdtree(&data, 2, 5);

        // idx_array must remain a permutation of 0..n
        let mut sorted = tree.idx_array.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>());
        assert_eq!(tree.radius[0], 0.0);

        let (dists, idxs) = parallel_tree_query(&tree, &[3.0, -1.0], 3, true);
        assert_eq!(dists, vec![0.0, 0.0, 0.0]);
        for &i in &idxs {
            assert!((0..n as i32).contains(&i));
        }
    }

    #[test]
    fn query_k_greater_than_n_pads_with_inf_and_minus_one() {
        let data = [0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0];
        let tree = build_kdtree(&data, 2, 40);
        let (dists, idxs) = parallel_tree_query(&tree, &[0.0, 0.0], 5, true);
        assert_eq!(&dists[..3], &[0.0, 1.0, 1.0]);
        assert_eq!(dists[3], f32::INFINITY);
        assert_eq!(dists[4], f32::INFINITY);
        assert_eq!(idxs[0], 0);
        assert_eq!(&idxs[3..], &[-1, -1]);
    }

    #[test]
    fn query_rdist_false_applies_sqrt() {
        let data = [0.0f32, 0.0, 3.0, 4.0];
        let tree = build_kdtree(&data, 2, 40);
        let (dists, idxs) = parallel_tree_query(&tree, &[0.0, 0.0], 2, false);
        assert_eq!(dists, vec![0.0, 5.0]);
        assert_eq!(idxs, vec![0, 1]);
    }

    #[test]
    fn simple_heap_push_rejects_when_full_of_smaller() {
        let mut p = vec![1.0f32, 0.5, 0.25];
        let mut i = vec![10, 11, 12];
        assert_eq!(simple_heap_push(&mut p, &mut i, 2.0, 99), 0);
        assert_eq!(p, vec![1.0, 0.5, 0.25]);
        assert_eq!(i, vec![10, 11, 12]);
        assert_eq!(simple_heap_push(&mut p, &mut i, 0.1, 99), 1);
        // max at root
        assert_eq!(p[0], 0.5);
    }

    #[test]
    fn deheap_sort_single_row() {
        let mut d = vec![5.0f32, 3.0, 4.0, 1.0, 2.0];
        let mut i = vec![50, 30, 40, 10, 20];
        deheap_sort(&mut d, &mut i, 5);
        assert_eq!(d, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(i, vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn rdist_identical_points_is_zero() {
        let x = [1.5f32, -2.5, 3.0];
        assert_eq!(rdist(&x, &x), 0.0);
    }

    #[test]
    fn point_inside_node_has_zero_lower_bound() {
        // reference call convention: (lower, upper, pt)
        let lower = [0.0f32, 0.0];
        let upper = [1.0f32, 1.0];
        assert_eq!(point_to_node_lower_bound_rdist(&lower, &upper, &[0.5, 0.5]), 0.0);
        // outside below dim 0 by 0.5 -> 0.25
        assert_eq!(point_to_node_lower_bound_rdist(&lower, &upper, &[-0.5, 0.5]), 0.25);
        // outside above dim 1 by 2 -> 4
        assert_eq!(point_to_node_lower_bound_rdist(&lower, &upper, &[0.5, 3.0]), 4.0);
    }
}
