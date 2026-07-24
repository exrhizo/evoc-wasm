//! Port of evoc/numba_kdtree.py (kdtree_to_numba sklearn glue skipped).
//! Keep the reference function names so the codebases diff side by side.
//! See PORTING.md.
//!
//! PORT NOTE: the reference stores `data` and `node_bounds` as numpy 2-D/3-D
//! arrays. Here matrices are flat row-major slices with an explicit
//! `n_features` passed alongside (data[i, j] == data[i * n_features + j]),
//! and `node_bounds` (shape (2, n_nodes, n_features)) becomes two flat
//! slices, node_bounds_lower / node_bounds_upper.
//!
//! PORT NOTE: the reference uses numpy intp (signed word-size) for
//! idx_array / idx_start / idx_end and index locals. All values are
//! non-negative array indices, so we use usize; the few loop variables that
//! legitimately go to -1 in the reference (insertion sort, heapsort,
//! partition scans) use isize locally to keep the exact loop structure.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Port of the NumbaKDTree namedtuple (data, idx_array, idx_start, idx_end,
/// radius, is_leaf, node_bounds). `data` is borrowed from the caller (the
/// reference shares the input array, too); everything else is owned.
pub const NumbaKDTree = struct {
    data: []const f32,
    n_features: usize,
    idx_array: []usize,
    idx_start: []usize,
    idx_end: []usize,
    radius: []f32,
    is_leaf: []bool,
    node_bounds_lower: []f32,
    node_bounds_upper: []f32,
    allocator: Allocator,

    pub fn n_nodes(self: *const NumbaKDTree) usize {
        return self.idx_start.len;
    }

    pub fn boundsLowerRow(self: *const NumbaKDTree, node: usize) []f32 {
        return self.node_bounds_lower[node * self.n_features ..][0..self.n_features];
    }

    pub fn boundsUpperRow(self: *const NumbaKDTree, node: usize) []f32 {
        return self.node_bounds_upper[node * self.n_features ..][0..self.n_features];
    }

    pub fn dataRow(self: *const NumbaKDTree, i: usize) []const f32 {
        return self.data[i * self.n_features ..][0..self.n_features];
    }

    pub fn deinit(self: *NumbaKDTree) void {
        self.allocator.free(self.idx_array);
        self.allocator.free(self.idx_start);
        self.allocator.free(self.idx_end);
        self.allocator.free(self.radius);
        self.allocator.free(self.is_leaf);
        self.allocator.free(self.node_bounds_lower);
        self.allocator.free(self.node_bounds_upper);
        self.* = undefined;
    }
};

/// Port of _init_node. `is_leaf_array` is unused in the reference body too;
/// kept for signature parity.
pub fn _init_node(
    data: []const f32,
    n_features: usize,
    node_bounds_lower: []f32,
    node_bounds_upper: []f32,
    idx_array: []const usize,
    idx_start_array: []usize,
    idx_end_array: []usize,
    radius_array: []f32,
    is_leaf_array: []bool,
    node: usize,
    idx_start: usize,
    idx_end: usize,
) void {
    _ = is_leaf_array;

    const lower_bounds = node_bounds_lower[node * n_features ..][0..n_features];
    const upper_bounds = node_bounds_upper[node * n_features ..][0..n_features];

    // determine Node bounds
    for (0..n_features) |j| {
        lower_bounds[j] = std.math.inf(f32);
        upper_bounds[j] = -std.math.inf(f32);
    }

    for (idx_start..idx_end) |i| {
        const data_row = data[idx_array[i] * n_features ..][0..n_features];
        for (0..n_features) |j| {
            lower_bounds[j] = @min(lower_bounds[j], data_row[j]);
            upper_bounds[j] = @max(upper_bounds[j], data_row[j]);
        }
    }

    var radius: f32 = 0.0;
    for (0..n_features) |j| {
        const diff: f32 = @abs(upper_bounds[j] - lower_bounds[j]) * 0.5;
        radius += diff * diff;
    }

    idx_start_array[node] = idx_start;
    idx_end_array[node] = idx_end;

    radius_array[node] = @sqrt(radius);
}

/// Port of _find_node_split_dim.
pub fn _find_node_split_dim(
    data: []const f32,
    n_features: usize,
    idx_array: []const usize,
    idx_start: usize,
    idx_end: usize,
) usize {
    var result: usize = 0;
    var max_spread: f32 = 0;

    for (0..n_features) |j| {
        var max_val: f32 = data[idx_array[idx_start] * n_features + j];
        var min_val: f32 = max_val;
        for (idx_start + 1..idx_end) |i| {
            const val: f32 = data[idx_array[i] * n_features + j];
            max_val = @max(max_val, val);
            min_val = @min(min_val, val);
        }

        const spread = max_val - min_val;

        if (spread > max_spread) {
            max_spread = spread;
            result = j;
        }
    }

    return result;
}

/// Port of _compare_indices: compare data values along `axis`, breaking ties
/// with the original index values (like sklearn).
pub fn _compare_indices(data: []const f32, n_features: usize, axis: usize, idx1: usize, idx2: usize) i8 {
    const val1: f32 = data[idx1 * n_features + axis];
    const val2: f32 = data[idx2 * n_features + axis];

    if (val1 < val2) {
        return -1;
    } else if (val1 > val2) {
        return 1;
    } else {
        // Break ties using original index values (like sklearn)
        if (idx1 < idx2) {
            return -1;
        } else if (idx1 > idx2) {
            return 1;
        } else {
            return 0;
        }
    }
}

/// Port of _insertion_sort_indices.
pub fn _insertion_sort_indices(
    data: []const f32,
    n_features: usize,
    idx_array: []usize,
    axis: usize,
    left: usize,
    right: usize,
) void {
    var i = left + 1;
    while (i < right) : (i += 1) {
        const key_idx = idx_array[i];
        var j: isize = @as(isize, @intCast(i)) - 1;

        while (j >= @as(isize, @intCast(left)) and
            _compare_indices(data, n_features, axis, idx_array[@intCast(j)], key_idx) > 0)
        {
            idx_array[@intCast(j + 1)] = idx_array[@intCast(j)];
            j -= 1;
        }

        idx_array[@intCast(j + 1)] = key_idx;
    }
}

/// Port of _sift_down_indices.
pub fn _sift_down_indices(
    data: []const f32,
    n_features: usize,
    idx_array: []usize,
    axis: usize,
    offset: usize,
    start: usize,
    end: usize,
) void {
    var root = start;

    while (root * 2 + 1 < end) {
        const child = root * 2 + 1;
        var swap = root;

        if (_compare_indices(data, n_features, axis, idx_array[offset + swap], idx_array[offset + child]) < 0) {
            swap = child;
        }

        if (child + 1 < end and
            _compare_indices(data, n_features, axis, idx_array[offset + swap], idx_array[offset + child + 1]) < 0)
        {
            swap = child + 1;
        }

        if (swap == root) {
            return;
        }

        const tmp = idx_array[offset + root];
        idx_array[offset + root] = idx_array[offset + swap];
        idx_array[offset + swap] = tmp;
        root = swap;
    }
}

/// Port of _heapsort_indices.
pub fn _heapsort_indices(
    data: []const f32,
    n_features: usize,
    idx_array: []usize,
    axis: usize,
    left: usize,
    right: usize,
) void {
    const size = right - left;

    // Build heap
    var i: isize = @as(isize, @intCast(size / 2)) - 1;
    while (i >= 0) : (i -= 1) {
        _sift_down_indices(data, n_features, idx_array, axis, left, @intCast(i), size);
    }

    // Extract elements
    i = @as(isize, @intCast(size)) - 1;
    while (i > 0) : (i -= 1) {
        const ui: usize = @intCast(i);
        const tmp = idx_array[left];
        idx_array[left] = idx_array[left + ui];
        idx_array[left + ui] = tmp;
        _sift_down_indices(data, n_features, idx_array, axis, left, 0, ui);
    }
}

/// Port of _median_of_three_pivot. Returns `mid` (position, not value).
pub fn _median_of_three_pivot(
    data: []const f32,
    n_features: usize,
    idx_array: []usize,
    axis: usize,
    left: usize,
    right: usize,
) usize {
    const mid = (left + right - 1) / 2;

    var idx_left = idx_array[left];
    var idx_mid = idx_array[mid];
    var idx_right = idx_array[right - 1];

    // Sort the three candidates
    if (_compare_indices(data, n_features, axis, idx_left, idx_mid) > 0) {
        idx_array[left] = idx_mid;
        idx_array[mid] = idx_left;
        const tmp = idx_left;
        idx_left = idx_mid;
        idx_mid = tmp;
    }

    if (_compare_indices(data, n_features, axis, idx_mid, idx_right) > 0) {
        idx_array[mid] = idx_right;
        idx_array[right - 1] = idx_mid;
        const tmp = idx_mid;
        idx_mid = idx_right;
        idx_right = tmp;

        if (_compare_indices(data, n_features, axis, idx_left, idx_mid) > 0) {
            idx_array[left] = idx_mid;
            idx_array[mid] = idx_left;
        }
    }

    return mid;
}

/// Port of _partition_indices (Hoare-style scan with index tie-breaks).
pub fn _partition_indices(
    data: []const f32,
    n_features: usize,
    idx_array: []usize,
    axis: usize,
    left: usize,
    right: usize,
    pivot_idx: usize,
) usize {
    // Move pivot to end
    {
        const tmp = idx_array[pivot_idx];
        idx_array[pivot_idx] = idx_array[right - 1];
        idx_array[right - 1] = tmp;
    }
    const pivot_original_idx = idx_array[right - 1];

    // PORT NOTE: i/j are intp in the reference; j can legitimately scan down
    // to left-1 (i.e. -1 when left == 0), so both are isize here.
    var i: isize = @intCast(left);
    var j: isize = @as(isize, @intCast(right)) - 2;

    while (true) {
        // Find element from left that should be on right
        while (i <= j and
            _compare_indices(data, n_features, axis, idx_array[@intCast(i)], pivot_original_idx) < 0)
        {
            i += 1;
        }

        // Find element from right that should be on left
        while (i <= j and
            _compare_indices(data, n_features, axis, idx_array[@intCast(j)], pivot_original_idx) >= 0)
        {
            j -= 1;
        }

        if (i >= j) {
            break;
        }

        // Swap elements
        const tmp = idx_array[@intCast(i)];
        idx_array[@intCast(i)] = idx_array[@intCast(j)];
        idx_array[@intCast(j)] = tmp;
        i += 1;
        j -= 1;
    }

    // Move pivot to final position
    const ui: usize = @intCast(i);
    const tmp = idx_array[ui];
    idx_array[ui] = idx_array[right - 1];
    idx_array[right - 1] = tmp;
    return ui;
}

/// Port of _introselect_impl.
pub fn _introselect_impl(
    data: []const f32,
    n_features: usize,
    idx_array: []usize,
    axis: usize,
    left_in: usize,
    right_in: usize,
    nth: usize,
    depth_limit_in: usize,
) void {
    var left = left_in;
    var right = right_in;
    var depth_limit = depth_limit_in;

    while (right - left > 16) {
        if (depth_limit == 0) {
            // Fall back to heapsort when recursion gets too deep
            _heapsort_indices(data, n_features, idx_array, axis, left, right);
            return;
        }

        depth_limit -= 1;

        // Choose pivot using median-of-three
        const pivot_idx = _median_of_three_pivot(data, n_features, idx_array, axis, left, right);

        // Partition around pivot
        const pivot_pos = _partition_indices(data, n_features, idx_array, axis, left, right, pivot_idx);

        // Recurse on the appropriate side
        if (nth < pivot_pos) {
            right = pivot_pos;
        } else if (nth > pivot_pos) {
            left = pivot_pos + 1;
        } else {
            // Found the nth element
            return;
        }
    }

    // Use insertion sort for small subarrays
    _insertion_sort_indices(data, n_features, idx_array, axis, left, right);
}

/// Port of _introselect.
pub fn _introselect(
    data: []const f32,
    n_features: usize,
    idx_array: []usize,
    axis: usize,
    left: usize,
    right: usize,
    nth: usize,
) void {
    const size = right - left;

    // Use heapsort for small arrays or when recursion depth is too high
    if (size <= 16) {
        _insertion_sort_indices(data, n_features, idx_array, axis, left, right);
        return;
    }

    // Calculate maximum recursion depth (2 * log2(size))
    const max_depth: usize = 2 * @as(usize, @intFromFloat(@log2(@as(f64, @floatFromInt(size)))));
    _introselect_impl(data, n_features, idx_array, axis, left, right, nth, max_depth);
}

/// Port of _recursive_build_tree.
pub fn _recursive_build_tree(
    data: []const f32,
    n_features: usize,
    idx_array: []usize,
    idx_start_array: []usize,
    idx_end_array: []usize,
    radius_array: []f32,
    is_leaf_array: []bool,
    node_bounds_lower: []f32,
    node_bounds_upper: []f32,
    idx_start: usize,
    idx_end: usize,
    node: usize,
) void {
    const n_points = idx_end - idx_start;
    const n_mid = n_points / 2;

    _init_node(
        data,
        n_features,
        node_bounds_lower,
        node_bounds_upper,
        idx_array,
        idx_start_array,
        idx_end_array,
        radius_array,
        is_leaf_array,
        node,
        idx_start,
        idx_end,
    );

    if (2 * node + 1 >= is_leaf_array.len) {
        is_leaf_array[node] = true;
    } else if (idx_end - idx_start < 2) {
        is_leaf_array[node] = true;
    } else {
        is_leaf_array[node] = false;
        const axis = _find_node_split_dim(data, n_features, idx_array, idx_start, idx_end);
        _introselect(data, n_features, idx_array, axis, idx_start, idx_end, idx_start + n_mid);
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

/// Port of build_kdtree. `data` is n_samples x n_features row-major and is
/// borrowed (must outlive the returned tree).
pub fn build_kdtree(allocator: Allocator, data: []const f32, n_features: usize, leaf_size: usize) !NumbaKDTree {
    std.debug.assert(n_features > 0);
    std.debug.assert(data.len % n_features == 0);
    // Reference raises ValueError for leaf_size < 1.
    std.debug.assert(leaf_size >= 1);

    const n_samples = data.len / n_features;

    // determine number of levels in the tree, and from this
    // the number of nodes in the tree.  This results in leaf nodes
    // with numbers of points between leaf_size and 2 * leaf_size
    // PORT NOTE: (n_samples - 1) / leaf_size is Python FLOAT division;
    // computed in f64 exactly as written, then int() truncation.
    const ratio: f64 = (@as(f64, @floatFromInt(n_samples)) - 1.0) / @as(f64, @floatFromInt(leaf_size));
    const n_levels: usize = @intFromFloat(@log2(@max(1.0, ratio)) + 1.0);
    const n_nodes: usize = (@as(usize, 1) << @intCast(n_levels)) - 1;

    // allocate arrays for storage
    const idx_array = try allocator.alloc(usize, n_samples);
    errdefer allocator.free(idx_array);
    for (idx_array, 0..) |*v, i| v.* = i;

    const idx_start_array = try allocator.alloc(usize, n_nodes);
    errdefer allocator.free(idx_start_array);
    @memset(idx_start_array, 0);

    const idx_end_array = try allocator.alloc(usize, n_nodes);
    errdefer allocator.free(idx_end_array);
    @memset(idx_end_array, 0);

    const radius_array = try allocator.alloc(f32, n_nodes);
    errdefer allocator.free(radius_array);
    @memset(radius_array, 0);

    const is_leaf_array = try allocator.alloc(bool, n_nodes);
    errdefer allocator.free(is_leaf_array);
    @memset(is_leaf_array, false);

    const node_bounds_lower = try allocator.alloc(f32, n_nodes * n_features);
    errdefer allocator.free(node_bounds_lower);
    @memset(node_bounds_lower, 0);

    const node_bounds_upper = try allocator.alloc(f32, n_nodes * n_features);
    errdefer allocator.free(node_bounds_upper);
    @memset(node_bounds_upper, 0);

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
        0,
        n_samples,
        0,
    );

    return NumbaKDTree{
        .data = data,
        .n_features = n_features,
        .idx_array = idx_array,
        .idx_start = idx_start_array,
        .idx_end = idx_end_array,
        .radius = radius_array,
        .is_leaf = is_leaf_array,
        .node_bounds_lower = node_bounds_lower,
        .node_bounds_upper = node_bounds_upper,
        .allocator = allocator,
    };
}

/// Port of rdist: squared euclidean distance, sequential f32 accumulation.
pub fn rdist(x: []const f32, y: []const f32) f32 {
    var result: f32 = 0.0;
    const dim = x.len;
    for (0..dim) |i| {
        const diff: f32 = x[i] - y[i];
        result += diff * diff;
    }

    return result;
}

/// Port of point_to_node_lower_bound_rdist.
///
/// PORT NOTE: the reference's parameter names are swapped relative to what
/// callers pass: every call site passes node_bounds[0] (the LOWER bounds) as
/// the parameter named `upper`, and node_bounds[1] (the UPPER bounds) as the
/// parameter named `lower`. The body is consistent with that convention
/// (d_lo measures how far the point is below the lower bound, d_hi how far
/// above the upper bound), so the math is correct — only the names are
/// misleading. Ported verbatim, names included.
pub fn point_to_node_lower_bound_rdist(upper: []const f32, lower: []const f32, pt: []const f32) f32 {
    var result: f32 = 0.0;
    const dim = pt.len;
    for (0..dim) |i| {
        const d_lo: f32 = if (upper[i] > pt[i]) upper[i] - pt[i] else 0.0;
        const d_hi: f32 = if (pt[i] > lower[i]) pt[i] - lower[i] else 0.0;
        const d: f32 = d_lo + d_hi;
        result += d * d;
    }

    return result;
}

/// Port of simple_heap_push: push (p, n) into a max-heap keyed on
/// priorities; returns 1 if the heap changed, 0 otherwise.
///
/// PORT NOTE: the reference's locals declare u16 loop indices (i, ic1, ic2,
/// i_swap); behaviorally irrelevant for any k < 32768, so usize is used.
pub fn simple_heap_push(priorities: []f32, indices: []i32, p: f32, n: i32) i32 {
    if (p >= priorities[0]) {
        return 0;
    }

    const size = priorities.len;

    // insert val at position zero
    priorities[0] = p;
    indices[0] = n;

    // descend the heap, swapping values until the max heap criterion is met
    var i: usize = 0;
    while (true) {
        const ic1 = 2 * i + 1;
        const ic2 = ic1 + 1;
        var i_swap: usize = undefined;

        if (ic1 >= size) {
            break;
        } else if (ic2 >= size) {
            if (priorities[ic1] > p) {
                i_swap = ic1;
            } else {
                break;
            }
        } else if (priorities[ic1] >= priorities[ic2]) {
            if (p < priorities[ic1]) {
                i_swap = ic1;
            } else {
                break;
            }
        } else {
            if (p < priorities[ic2]) {
                i_swap = ic2;
            } else {
                break;
            }
        }

        priorities[i] = priorities[i_swap];
        indices[i] = indices[i_swap];

        i = i_swap;
    }

    priorities[i] = p;
    indices[i] = n;

    return 1;
}

/// Port of siftdown (max-heap restore for the paired heap1/heap2 rows).
pub fn siftdown(heap1: []f32, heap2: []i32, elt_in: usize) void {
    var elt = elt_in;
    while (elt * 2 + 1 < heap1.len) {
        const left_child = elt * 2 + 1;
        const right_child = left_child + 1;
        var swap = elt;

        if (heap1[swap] < heap1[left_child]) {
            swap = left_child;
        }

        if (right_child < heap1.len and heap1[swap] < heap1[right_child]) {
            swap = right_child;
        }

        if (swap == elt) {
            break;
        } else {
            const tmp1 = heap1[elt];
            heap1[elt] = heap1[swap];
            heap1[swap] = tmp1;
            const tmp2 = heap2[elt];
            heap2[elt] = heap2[swap];
            heap2[swap] = tmp2;
            elt = swap;
        }
    }
}

/// Port of deheap_sort: row-wise heap extraction into ascending order.
/// `distances`/`indices` are n_rows x k row-major flat arrays.
///
/// PORT NOTE: the reference pranges over rows; single-threaded WASM target,
/// so this is a plain sequential loop (identical result — rows are
/// independent).
pub fn deheap_sort(distances: []f32, indices: []i32, k: usize) void {
    const n_rows = indices.len / k;
    for (0..n_rows) |i| {
        const dist_row = distances[i * k ..][0..k];
        const idx_row = indices[i * k ..][0..k];
        // starting from the end of the array and moving back
        var j: usize = k - 1;
        while (j > 0) : (j -= 1) {
            const tmp_i = idx_row[0];
            idx_row[0] = idx_row[j];
            idx_row[j] = tmp_i;
            const tmp_d = dist_row[0];
            dist_row[0] = dist_row[j];
            dist_row[j] = tmp_d;

            siftdown(dist_row[0..j], idx_row[0..j], 0);
        }
    }
}

/// Port of tree_query_recursion.
pub fn tree_query_recursion(
    tree: *const NumbaKDTree,
    node: usize,
    point: []const f32,
    heap_p: []f32,
    heap_i: []i32,
    dist_lower_bound: f32,
) void {
    // Get node information
    const idx_start = tree.idx_start[node];
    const idx_end = tree.idx_end[node];
    const is_leaf = tree.is_leaf[node];

    // ------------------------------------------------------------
    // Case 1: query point is outside node radius:
    //         trim it from the query
    if (dist_lower_bound > heap_p[0]) {
        return;
    }

    // ------------------------------------------------------------
    // Case 2: this is a leaf node.  Update set of nearby points
    else if (is_leaf) {
        for (idx_start..idx_end) |i| {
            const idx = tree.idx_array[i];
            const d = rdist(point, tree.dataRow(idx));
            if (d < heap_p[0]) {
                _ = simple_heap_push(heap_p, heap_i, d, @intCast(idx));
            }
        }
    }

    // ------------------------------------------------------------
    // Case 3: Node is not a leaf.  Recursively query subnodes
    //         starting with the closest
    else {
        const left = 2 * node + 1;
        const right = left + 1;
        const dist_lower_bound_left = point_to_node_lower_bound_rdist(
            tree.boundsLowerRow(left),
            tree.boundsUpperRow(left),
            point,
        );
        const dist_lower_bound_right = point_to_node_lower_bound_rdist(
            tree.boundsLowerRow(right),
            tree.boundsUpperRow(right),
            point,
        );

        // recursively query subnodes
        if (dist_lower_bound_left <= dist_lower_bound_right) {
            tree_query_recursion(tree, left, point, heap_p, heap_i, dist_lower_bound_left);
            tree_query_recursion(tree, right, point, heap_p, heap_i, dist_lower_bound_right);
        } else {
            tree_query_recursion(tree, right, point, heap_p, heap_i, dist_lower_bound_right);
            tree_query_recursion(tree, left, point, heap_p, heap_i, dist_lower_bound_left);
        }
    }
}

/// Result of parallel_tree_query: n_queries x k row-major matrices.
pub const QueryResult = struct {
    distances: []f32,
    indices: []i32,
    k: usize,
    allocator: Allocator,

    pub fn deinit(self: *QueryResult) void {
        self.allocator.free(self.distances);
        self.allocator.free(self.indices);
        self.* = undefined;
    }
};

/// Port of parallel_tree_query. `queries` is n_queries x n_features
/// row-major.
///
/// PORT NOTE: the reference pranges over query rows; single-threaded WASM
/// target, so this is a plain sequential loop (rows are independent, result
/// is identical). Also, the reference's output_rdist=False branch allocates
/// np.sqrt(result[0]) before deheap_sort; here sqrt is applied in place —
/// same values, and sqrt is monotonic so the heap invariant is preserved.
pub fn parallel_tree_query(
    allocator: Allocator,
    tree: *const NumbaKDTree,
    queries: []const f32,
    k: usize,
    output_rdist: bool,
) !QueryResult {
    std.debug.assert(queries.len % tree.n_features == 0);
    const n_queries = queries.len / tree.n_features;

    const distances = try allocator.alloc(f32, n_queries * k);
    errdefer allocator.free(distances);
    @memset(distances, std.math.inf(f32));

    const indices = try allocator.alloc(i32, n_queries * k);
    errdefer allocator.free(indices);
    @memset(indices, -1);

    for (0..n_queries) |i| {
        const point = queries[i * tree.n_features ..][0..tree.n_features];
        const distance_lower_bound = point_to_node_lower_bound_rdist(
            tree.boundsLowerRow(0),
            tree.boundsUpperRow(0),
            point,
        );
        const heap_priorities = distances[i * k ..][0..k];
        const heap_indices = indices[i * k ..][0..k];
        tree_query_recursion(tree, 0, point, heap_priorities, heap_indices, distance_lower_bound);
    }

    if (!output_rdist) {
        for (distances) |*d| d.* = @sqrt(d.*);
    }
    deheap_sort(distances, indices, k);

    return QueryResult{ .distances = distances, .indices = indices, .k = k, .allocator = allocator };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const fixture = @import("fixture.zig");

fn readMatrixF32(alloc: Allocator, arr: std.json.Array) !struct { data: []f32, rows: usize, cols: usize } {
    const rows = arr.items.len;
    const cols = if (rows > 0) arr.items[0].array.items.len else 0;
    const data = try alloc.alloc(f32, rows * cols);
    errdefer alloc.free(data);
    for (arr.items, 0..) |row, i| {
        for (row.array.items, 0..) |v, j| {
            data[i * cols + j] = @floatCast(fixture.numberAsF64(v));
        }
    }
    return .{ .data = data, .rows = rows, .cols = cols };
}

fn readVecF32(alloc: Allocator, arr: std.json.Array) ![]f32 {
    const out = try alloc.alloc(f32, arr.items.len);
    for (arr.items, 0..) |v, i| out[i] = @floatCast(fixture.numberAsF64(v));
    return out;
}

test "single-point tree: one leaf node containing the point" {
    const alloc = std.testing.allocator;
    const data = [_]f32{ 1.5, -2.0 };
    var tree = try build_kdtree(alloc, &data, 2, 40);
    defer tree.deinit();

    try std.testing.expectEqual(@as(usize, 1), tree.n_nodes());
    try std.testing.expect(tree.is_leaf[0]);
    try std.testing.expectEqual(@as(usize, 0), tree.idx_array[0]);
    try std.testing.expectEqual(@as(usize, 0), tree.idx_start[0]);
    try std.testing.expectEqual(@as(usize, 1), tree.idx_end[0]);
    try std.testing.expectEqual(@as(f32, 0.0), tree.radius[0]);
    try std.testing.expectEqual(@as(f32, 1.5), tree.node_bounds_lower[0]);
    try std.testing.expectEqual(@as(f32, 1.5), tree.node_bounds_upper[0]);
    try std.testing.expectEqual(@as(f32, -2.0), tree.node_bounds_lower[1]);
    try std.testing.expectEqual(@as(f32, -2.0), tree.node_bounds_upper[1]);
}

test "all-identical points: builds, radius 0, query finds distance 0" {
    const alloc = std.testing.allocator;
    const n = 20;
    var data: [n * 2]f32 = undefined;
    for (0..n) |i| {
        data[i * 2] = 3.0;
        data[i * 2 + 1] = 3.0;
    }
    var tree = try build_kdtree(alloc, &data, 2, 5);
    defer tree.deinit();

    // 19/5 = 3.8 -> log2 = 1.926 -> +1 -> int = 2 levels -> 3 nodes.
    try std.testing.expectEqual(@as(usize, 3), tree.n_nodes());
    for (tree.radius) |r| try std.testing.expectEqual(@as(f32, 0.0), r);

    // idx_array must remain a permutation of 0..n.
    var seen = [_]bool{false} ** n;
    for (tree.idx_array) |idx| {
        try std.testing.expect(!seen[idx]);
        seen[idx] = true;
    }

    const query = [_]f32{ 3.0, 3.0 };
    var result = try parallel_tree_query(alloc, &tree, &query, 3, true);
    defer result.deinit();
    for (result.distances) |d| try std.testing.expectEqual(@as(f32, 0.0), d);
    for (result.indices) |idx| try std.testing.expect(idx >= 0);
}

test "k > n query: tail entries stay inf/-1" {
    const alloc = std.testing.allocator;
    const data = [_]f32{ 0.0, 0.0, 1.0, 0.0, 0.0, 1.0 }; // 3 points in 2-D
    var tree = try build_kdtree(alloc, &data, 2, 40);
    defer tree.deinit();

    const query = [_]f32{ 0.0, 0.0 };
    const k = 5;
    var result = try parallel_tree_query(alloc, &tree, &query, k, true);
    defer result.deinit();

    // 3 real neighbors sorted ascending, then inf/-1 padding.
    try std.testing.expectEqual(@as(f32, 0.0), result.distances[0]);
    try std.testing.expectEqual(@as(i32, 0), result.indices[0]);
    try std.testing.expectEqual(@as(f32, 1.0), result.distances[1]);
    try std.testing.expectEqual(@as(f32, 1.0), result.distances[2]);
    try std.testing.expect(std.math.isInf(result.distances[3]));
    try std.testing.expect(std.math.isInf(result.distances[4]));
    try std.testing.expectEqual(@as(i32, -1), result.indices[3]);
    try std.testing.expectEqual(@as(i32, -1), result.indices[4]);

    // Same but with output_rdist=false: sqrt(inf) stays inf.
    var result2 = try parallel_tree_query(alloc, &tree, &query, k, false);
    defer result2.deinit();
    try std.testing.expectEqual(@as(f32, 0.0), result2.distances[0]);
    try std.testing.expect(std.math.isInf(result2.distances[4]));
}

test "rdist: zero and identical vectors" {
    const x = [_]f32{ 1.0, 2.0, 3.0 };
    try std.testing.expectEqual(@as(f32, 0.0), rdist(&x, &x));
    const z = [_]f32{ 0.0, 0.0, 0.0 };
    try std.testing.expectEqual(@as(f32, 14.0), rdist(&x, &z));
}

test "point_to_node_lower_bound_rdist: point inside bounds -> 0" {
    // Call convention: fn(bounds_lower, bounds_upper, pt).
    const lo = [_]f32{ 0.0, 0.0 };
    const hi = [_]f32{ 1.0, 1.0 };
    const inside = [_]f32{ 0.5, 0.5 };
    try std.testing.expectEqual(@as(f32, 0.0), point_to_node_lower_bound_rdist(&lo, &hi, &inside));
    const below = [_]f32{ -1.0, 0.5 }; // 1 below lower bound in dim 0
    try std.testing.expectEqual(@as(f32, 1.0), point_to_node_lower_bound_rdist(&lo, &hi, &below));
    const above = [_]f32{ 0.5, 3.0 }; // 2 above upper bound in dim 1
    try std.testing.expectEqual(@as(f32, 4.0), point_to_node_lower_bound_rdist(&lo, &hi, &above));
}

test "simple_heap_push: rejects p >= max, replaces max otherwise" {
    var priorities = [_]f32{ std.math.inf(f32), std.math.inf(f32), std.math.inf(f32) };
    var indices = [_]i32{ -1, -1, -1 };
    try std.testing.expectEqual(@as(i32, 1), simple_heap_push(&priorities, &indices, 5.0, 10));
    try std.testing.expectEqual(@as(i32, 1), simple_heap_push(&priorities, &indices, 3.0, 11));
    try std.testing.expectEqual(@as(i32, 1), simple_heap_push(&priorities, &indices, 4.0, 12));
    // Heap is full with {5,3,4}; max is 5 at the root.
    try std.testing.expectEqual(@as(f32, 5.0), priorities[0]);
    try std.testing.expectEqual(@as(i32, 0), simple_heap_push(&priorities, &indices, 6.0, 13));
    try std.testing.expectEqual(@as(i32, 0), simple_heap_push(&priorities, &indices, 5.0, 14));
    try std.testing.expectEqual(@as(i32, 1), simple_heap_push(&priorities, &indices, 1.0, 15));
    try std.testing.expectEqual(@as(f32, 4.0), priorities[0]);
}

test "deheap_sort: k=1 row is a no-op" {
    var d = [_]f32{2.5};
    var idx = [_]i32{7};
    deheap_sort(&d, &idx, 1);
    try std.testing.expectEqual(@as(f32, 2.5), d[0]);
    try std.testing.expectEqual(@as(i32, 7), idx[0]);
}

fn checkBuildFixture(case: []const u8) !void {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "kdtree", case);
    defer fx.deinit();

    const root = fx.value.object;
    const inputs = root.get("inputs").?.object;
    const tol_value = root.get("tolerance").?;

    const m = try readMatrixF32(alloc, inputs.get("data").?.array);
    defer alloc.free(m.data);
    const leaf_size: usize = @intCast(inputs.get("leaf_size").?.integer);

    var tree = try build_kdtree(alloc, m.data, m.cols, leaf_size);
    defer tree.deinit();

    const expected = root.get("expected").?.object;

    const exp_idx_array = expected.get("idx_array").?.array;
    try std.testing.expectEqual(exp_idx_array.items.len, tree.idx_array.len);
    for (exp_idx_array.items, 0..) |v, i| {
        try std.testing.expectEqual(@as(usize, @intCast(v.integer)), tree.idx_array[i]);
    }

    const exp_idx_start = expected.get("idx_start").?.array;
    const exp_idx_end = expected.get("idx_end").?.array;
    const exp_is_leaf = expected.get("is_leaf").?.array;
    const exp_radius = expected.get("radius").?.array;
    try std.testing.expectEqual(exp_idx_start.items.len, tree.n_nodes());
    for (0..tree.n_nodes()) |node| {
        try std.testing.expectEqual(@as(usize, @intCast(exp_idx_start.items[node].integer)), tree.idx_start[node]);
        try std.testing.expectEqual(@as(usize, @intCast(exp_idx_end.items[node].integer)), tree.idx_end[node]);
        try std.testing.expectEqual(exp_is_leaf.items[node].bool, tree.is_leaf[node]);
        const radius_tol = fixture.toleranceForField(tol_value, "radius");
        try fixture.expectF32(radius_tol, tree.radius[node], fixture.numberAsF64(exp_radius.items[node]));
    }

    const bounds_tol = fixture.toleranceForField(tol_value, "node_bounds_lower");
    const exp_lower = expected.get("node_bounds_lower").?.array;
    const exp_upper = expected.get("node_bounds_upper").?.array;
    for (0..tree.n_nodes()) |node| {
        const lower_row = exp_lower.items[node].array;
        const upper_row = exp_upper.items[node].array;
        for (0..tree.n_features) |j| {
            try fixture.expectF32(bounds_tol, tree.node_bounds_lower[node * tree.n_features + j], fixture.numberAsF64(lower_row.items[j]));
            try fixture.expectF32(bounds_tol, tree.node_bounds_upper[node * tree.n_features + j], fixture.numberAsF64(upper_row.items[j]));
        }
    }
}

test "fixture parity: build_n100_d4_leaf10" {
    try checkBuildFixture("build_n100_d4_leaf10");
}

test "fixture parity: build_n37_d3_leaf5" {
    try checkBuildFixture("build_n37_d3_leaf5");
}

test "fixture parity: build_n20_d2_leaf40_single_node" {
    try checkBuildFixture("build_n20_d2_leaf40_single_node");
}

test "fixture parity: rdist" {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "kdtree", "rdist");
    defer fx.deinit();

    const root = fx.value.object;
    const tol = fixture.toleranceForField(root.get("tolerance").?, "values");
    const pairs = root.get("inputs").?.object.get("pairs").?.array;
    const values = root.get("expected").?.object.get("values").?.array;

    for (pairs.items, 0..) |pair_val, i| {
        const pair = pair_val.object;
        const x = try readVecF32(alloc, pair.get("x").?.array);
        defer alloc.free(x);
        const y = try readVecF32(alloc, pair.get("y").?.array);
        defer alloc.free(y);
        try fixture.expectF32(tol, rdist(x, y), fixture.numberAsF64(values.items[i]));
    }
}

test "fixture parity: point_to_node_lower_bound_rdist" {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "kdtree", "point_to_node_lower_bound_rdist");
    defer fx.deinit();

    const root = fx.value.object;
    const tol = fixture.toleranceForField(root.get("tolerance").?, "values");
    const cases = root.get("inputs").?.object.get("cases").?.array;
    const values = root.get("expected").?.object.get("values").?.array;

    for (cases.items, 0..) |case_val, i| {
        const c = case_val.object;
        const bounds_lower = try readVecF32(alloc, c.get("bounds_lower").?.array);
        defer alloc.free(bounds_lower);
        const bounds_upper = try readVecF32(alloc, c.get("bounds_upper").?.array);
        defer alloc.free(bounds_upper);
        const pt = try readVecF32(alloc, c.get("pt").?.array);
        defer alloc.free(pt);

        // Per inputs.note: call as fn(bounds_lower, bounds_upper, pt) — the
        // reference's parameter names are swapped vs the caller.
        const got = point_to_node_lower_bound_rdist(bounds_lower, bounds_upper, pt);
        try fixture.expectF32(tol, got, fixture.numberAsF64(values.items[i]));
    }
}

test "fixture parity: heap_push_deheap_sort" {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "kdtree", "heap_push_deheap_sort");
    defer fx.deinit();

    const root = fx.value.object;
    const tol = fixture.toleranceForField(root.get("tolerance").?, "heap_priorities");
    const inputs = root.get("inputs").?.object;
    const k: usize = @intCast(inputs.get("k").?.integer);
    const pushes = inputs.get("pushes").?.array;

    const priorities = try alloc.alloc(f32, k);
    defer alloc.free(priorities);
    @memset(priorities, std.math.inf(f32));
    const indices = try alloc.alloc(i32, k);
    defer alloc.free(indices);
    @memset(indices, -1);

    const expected = root.get("expected").?.object;
    const push_returns = expected.get("push_returns").?.array;

    for (pushes.items, 0..) |push_val, i| {
        const push = push_val.object;
        const p: f32 = @floatCast(fixture.numberAsF64(push.get("p").?));
        const n: i32 = @intCast(push.get("n").?.integer);
        const ret = simple_heap_push(priorities, indices, p, n);
        try std.testing.expectEqual(@as(i32, @intCast(push_returns.items[i].integer)), ret);
    }

    const exp_heap_p = expected.get("heap_priorities").?.array;
    const exp_heap_i = expected.get("heap_indices").?.array;
    for (0..k) |i| {
        try fixture.expectF32(tol, priorities[i], fixture.numberAsF64(exp_heap_p.items[i]));
        try std.testing.expectEqual(@as(i32, @intCast(exp_heap_i.items[i].integer)), indices[i]);
    }

    // deheap_sort the heap as a 1-row matrix.
    deheap_sort(priorities, indices, k);
    const exp_sorted_p = expected.get("sorted_priorities").?.array;
    const exp_sorted_i = expected.get("sorted_indices").?.array;
    for (0..k) |i| {
        try fixture.expectF32(tol, priorities[i], fixture.numberAsF64(exp_sorted_p.items[i]));
        try std.testing.expectEqual(@as(i32, @intCast(exp_sorted_i.items[i].integer)), indices[i]);
    }
}

fn checkQueryFixture(case: []const u8) !void {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "kdtree", case);
    defer fx.deinit();

    const root = fx.value.object;
    const tol_value = root.get("tolerance").?;
    const inputs = root.get("inputs").?.object;

    const m = try readMatrixF32(alloc, inputs.get("data").?.array);
    defer alloc.free(m.data);
    const leaf_size: usize = @intCast(inputs.get("leaf_size").?.integer);
    const q = try readMatrixF32(alloc, inputs.get("queries").?.array);
    defer alloc.free(q.data);
    const k: usize = @intCast(inputs.get("k").?.integer);
    const output_rdist = inputs.get("output_rdist").?.bool;

    var tree = try build_kdtree(alloc, m.data, m.cols, leaf_size);
    defer tree.deinit();

    var result = try parallel_tree_query(alloc, &tree, q.data, k, output_rdist);
    defer result.deinit();

    const expected = root.get("expected").?.object;
    const exp_dist = expected.get("distances").?.array;
    const exp_idx = expected.get("indices").?.array;
    const dist_tol = fixture.toleranceForField(tol_value, "distances");
    const idx_tol = fixture.toleranceForField(tol_value, "indices");
    try std.testing.expect(idx_tol == .exact);

    for (0..q.rows) |i| {
        const dist_row = exp_dist.items[i].array;
        const idx_row = exp_idx.items[i].array;
        for (0..k) |j| {
            try fixture.expectF32(dist_tol, result.distances[i * k + j], fixture.numberAsF64(dist_row.items[j]));
            try std.testing.expectEqual(@as(i32, @intCast(idx_row.items[j].integer)), result.indices[i * k + j]);
        }
    }
}

test "fixture parity: parallel_tree_query_k5_rdist_true" {
    try checkQueryFixture("parallel_tree_query_k5_rdist_true");
}

test "fixture parity: parallel_tree_query_k5_rdist_false" {
    try checkQueryFixture("parallel_tree_query_k5_rdist_false");
}
