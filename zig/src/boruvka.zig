//! Port of evoc/boruvka.py: mutual-reachability MST via Boruvka's algorithm.
//! Reuses kdtree.zig (NumbaKDTree, rdist, point_to_node_lower_bound_rdist,
//! parallel_tree_query) and disjoint_set.zig (RankDisjointSet, ds_find,
//! ds_union_by_rank). See PORTING.md / PORT_NOTES.md.
//!
//! PORT NOTE: everything runs in SQUARED rdist space until the very end,
//! where parallel_boruvka applies one final sqrt to the whole edge distance
//! column. initialize_boruvka_from_knn is naturally pre-sqrt.
//!
//! PORT NOTE: all numba prange loops become sequential loops; the block /
//! component-bound machinery is kept structurally intact with n_threads as a
//! parameter (fixtures are n_threads=1). reproducible=True only bounds
//! multithreaded races (adaptive block size); at n_threads=1 it is
//! bit-identical to reproducible=False.

const std = @import("std");
const Allocator = std.mem.Allocator;
const kdtree = @import("kdtree.zig");
const ds = @import("disjoint_set.zig");

const NumbaKDTree = kdtree.NumbaKDTree;
const rdist = kdtree.rdist;
const point_to_node_lower_bound_rdist = kdtree.point_to_node_lower_bound_rdist;
const RankDisjointSet = ds.RankDisjointSet;
const ds_find = ds.ds_find;
const ds_union_by_rank = ds.ds_union_by_rank;

/// Edge list: n_edges rows of (point_a, point_b, distance). Allocator-owned.
pub const Edges = struct {
    point_a: []i32,
    point_b: []i32,
    distance: []f32,
    allocator: Allocator,

    pub fn deinit(self: *Edges) void {
        self.allocator.free(self.point_a);
        self.allocator.free(self.point_b);
        self.allocator.free(self.distance);
        self.* = undefined;
    }
};

/// Port of merge_components. `candidate_neighbors`/`candidate_neighbor_distances`
/// are per-point best out-of-component candidates; `point_components` maps each
/// point to its component id. Returns the newly added MST edges (the ones that
/// actually merged two distinct components).
///
/// PORT NOTE: the reference uses a numba typed dict keyed by component id with
/// value tuple (i64 from_point, i64 to_point, f32 dist), keeping the FIRST-
/// improving edge per component and iterating in insertion (first-seen) order.
/// Reproduced with a per-component slot array plus an insertion-order list so
/// the union order matches exactly.
pub fn merge_components(
    allocator: Allocator,
    disjoint_set: *RankDisjointSet,
    candidate_neighbors: []const i32,
    candidate_neighbor_distances: []const f32,
    point_components: []const i64,
) !Edges {
    const n = candidate_neighbors.len;

    // Per-component best edge (from_point, to_point, dist) + presence flag and
    // an insertion-order list of component ids.
    const edge_from = try allocator.alloc(i64, n);
    defer allocator.free(edge_from);
    const edge_to = try allocator.alloc(i64, n);
    defer allocator.free(edge_to);
    const edge_dist = try allocator.alloc(f32, n);
    defer allocator.free(edge_dist);
    const present = try allocator.alloc(bool, n);
    defer allocator.free(present);
    @memset(present, false);
    const order = try allocator.alloc(usize, n);
    defer allocator.free(order);
    var n_order: usize = 0;

    // Find the best edges from each component.
    for (0..n) |i| {
        const from_component: usize = @intCast(point_components[i]);
        if (present[from_component]) {
            if (candidate_neighbor_distances[i] < edge_dist[from_component]) {
                edge_from[from_component] = @intCast(i);
                edge_to[from_component] = candidate_neighbors[i];
                edge_dist[from_component] = candidate_neighbor_distances[i];
            }
        } else {
            present[from_component] = true;
            edge_from[from_component] = @intCast(i);
            edge_to[from_component] = candidate_neighbors[i];
            edge_dist[from_component] = candidate_neighbor_distances[i];
            order[n_order] = from_component;
            n_order += 1;
        }
    }

    var point_a = try allocator.alloc(i32, n_order);
    errdefer allocator.free(point_a);
    var point_b = try allocator.alloc(i32, n_order);
    errdefer allocator.free(point_b);
    var distance = try allocator.alloc(f32, n_order);
    errdefer allocator.free(distance);
    var result_idx: usize = 0;

    // Add the best edges to the edge set and merge the relevant components.
    for (0..n_order) |o| {
        const comp = order[o];
        const ef = edge_from[comp];
        const et = edge_to[comp];
        const ed = edge_dist[comp];

        const from_component = ds_find(disjoint_set, @intCast(ef));
        const to_component = ds_find(disjoint_set, @intCast(et));
        if (from_component != to_component) {
            // The reference stores edge[0]/edge[1] (the raw point indices) as
            // float32 then re-reads them; for point ids well under 2^24 this
            // round-trips exactly.
            point_a[result_idx] = @intCast(ef);
            point_b[result_idx] = @intCast(et);
            distance[result_idx] = ed;
            result_idx += 1;

            ds_union_by_rank(disjoint_set, from_component, to_component);
        }
    }

    point_a = try allocator.realloc(point_a, result_idx);
    point_b = try allocator.realloc(point_b, result_idx);
    distance = try allocator.realloc(distance, result_idx);
    return Edges{ .point_a = point_a, .point_b = point_b, .distance = distance, .allocator = allocator };
}

/// Port of update_component_vectors: refresh point_components via ds_find, then
/// propagate node_components bottom-up over the kdtree nodes.
///
/// PORT NOTE: the leaf case uses Python's for/else — node_components[i] is set
/// only if the inner loop completes WITHOUT break (all points same component).
pub fn update_component_vectors(
    tree: *const NumbaKDTree,
    disjoint_set: *RankDisjointSet,
    node_components: []i64,
    point_components: []i64,
) void {
    for (0..point_components.len) |i| { // prange, sequential here
        point_components[i] = ds_find(disjoint_set, @intCast(i));
    }

    var i: isize = @as(isize, @intCast(tree.idx_start.len)) - 1;
    while (i >= 0) : (i -= 1) {
        const ui: usize = @intCast(i);
        const is_leaf = tree.is_leaf[ui];
        const idx_start = tree.idx_start[ui];
        const idx_end = tree.idx_end[ui];

        if (is_leaf) {
            const candidate_component = point_components[tree.idx_array[idx_start]];
            var all_same = true;
            for (idx_start + 1..idx_end) |j| {
                const idx = tree.idx_array[j];
                if (point_components[idx] != candidate_component) {
                    all_same = false;
                    break;
                }
            }
            if (all_same) {
                node_components[ui] = candidate_component;
            }
        } else {
            const left = 2 * ui + 1;
            const right = left + 1;
            if (node_components[left] == node_components[right]) {
                node_components[ui] = node_components[left];
            }
        }
    }
}

/// Port of component_aware_query_recursion. heap_p/heap_i are single-element
/// slices (this point's best candidate); component_nearest_neighbor_dist is a
/// single-element slice for the query point's component.
pub fn component_aware_query_recursion(
    tree: *const NumbaKDTree,
    node: usize,
    point: []const f32,
    heap_p: []f32,
    heap_i: []i32,
    current_core_distance: f32,
    core_distances: []const f32,
    current_component: i64,
    node_components: []const i64,
    point_components: []const i64,
    dist_lower_bound: f32,
    component_nearest_neighbor_dist: []f32,
) void {
    const is_leaf = tree.is_leaf[node];
    const idx_start = tree.idx_start[node];
    const idx_end = tree.idx_end[node];

    // Case 1a: query point is outside node radius -> trim.
    if (dist_lower_bound > heap_p[0]) {
        return;
    }
    // Case 1b: can't improve on best distance for this component -> trim.
    else if (dist_lower_bound > component_nearest_neighbor_dist[0] or
        current_core_distance > component_nearest_neighbor_dist[0])
    {
        return;
    }
    // Case 1c: node contains only points in the query's component -> trim.
    else if (node_components[node] == current_component) {
        return;
    }
    // Case 2: leaf node -> update the nearby-point set.
    else if (is_leaf) {
        for (idx_start..idx_end) |i| {
            const idx = tree.idx_array[i];
            if (point_components[idx] != current_component and
                core_distances[idx] < component_nearest_neighbor_dist[0])
            {
                const d = @max(
                    @max(rdist(point, tree.dataRow(idx)), current_core_distance),
                    core_distances[idx],
                );
                if (d < heap_p[0]) {
                    heap_p[0] = d;
                    heap_i[0] = @intCast(idx);
                    if (d < component_nearest_neighbor_dist[0]) {
                        component_nearest_neighbor_dist[0] = d;
                    }
                }
            }
        }
    }
    // Case 3: interior node -> recurse into the closer child first.
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

        if (dist_lower_bound_left <= dist_lower_bound_right) {
            component_aware_query_recursion(tree, left, point, heap_p, heap_i, current_core_distance, core_distances, current_component, node_components, point_components, dist_lower_bound_left, component_nearest_neighbor_dist);
            component_aware_query_recursion(tree, right, point, heap_p, heap_i, current_core_distance, core_distances, current_component, node_components, point_components, dist_lower_bound_right, component_nearest_neighbor_dist);
        } else {
            component_aware_query_recursion(tree, right, point, heap_p, heap_i, current_core_distance, core_distances, current_component, node_components, point_components, dist_lower_bound_right, component_nearest_neighbor_dist);
            component_aware_query_recursion(tree, left, point, heap_p, heap_i, current_core_distance, core_distances, current_component, node_components, point_components, dist_lower_bound_left, component_nearest_neighbor_dist);
        }
    }
}

/// Per-point candidate distances/indices from a Boruvka query pass.
pub const QueryCandidates = struct {
    distances: []f32,
    indices: []i32,
    allocator: Allocator,

    pub fn deinit(self: *QueryCandidates) void {
        self.allocator.free(self.distances);
        self.allocator.free(self.indices);
        self.* = undefined;
    }
};

/// Port of boruvka_tree_query (non-reproducible).
pub fn boruvka_tree_query(
    allocator: Allocator,
    tree: *const NumbaKDTree,
    node_components: []const i64,
    point_components: []const i64,
    core_distances: []const f32,
) !QueryCandidates {
    const n = tree.data.len / tree.n_features;

    const candidate_distances = try allocator.alloc(f32, n);
    errdefer allocator.free(candidate_distances);
    @memset(candidate_distances, std.math.inf(f32));
    const candidate_indices = try allocator.alloc(i32, n);
    errdefer allocator.free(candidate_indices);
    @memset(candidate_indices, -1);
    const component_nearest_neighbor_dist = try allocator.alloc(f32, n);
    defer allocator.free(component_nearest_neighbor_dist);
    @memset(component_nearest_neighbor_dist, std.math.inf(f32));

    for (0..n) |i| { // prange, sequential here
        const point = tree.dataRow(i);
        const distance_lower_bound = point_to_node_lower_bound_rdist(
            tree.boundsLowerRow(0),
            tree.boundsUpperRow(0),
            point,
        );
        const comp: usize = @intCast(point_components[i]);
        component_aware_query_recursion(
            tree,
            0,
            point,
            candidate_distances[i .. i + 1],
            candidate_indices[i .. i + 1],
            core_distances[i],
            core_distances,
            point_components[i],
            node_components,
            point_components,
            distance_lower_bound,
            component_nearest_neighbor_dist[comp .. comp + 1],
        );
    }

    return QueryCandidates{ .distances = candidate_distances, .indices = candidate_indices, .allocator = allocator };
}

/// Port of calculate_block_size (adaptive block sizing for the reproducible
/// query). Only meaningfully affects multithreaded runs.
pub fn calculate_block_size(n_components: usize, n_points: usize, num_threads: usize) usize {
    var points_per_component: f64 = undefined;
    if (n_components == 0) {
        points_per_component = @floatFromInt(n_points);
    } else {
        points_per_component = @as(f64, @floatFromInt(n_points)) / @as(f64, @floatFromInt(n_components));
    }

    var block_size: usize = undefined;
    if (points_per_component < 10) {
        block_size = num_threads * 512;
    } else if (points_per_component < 100) {
        block_size = num_threads * 128;
    } else if (points_per_component < 1000) {
        block_size = num_threads * 32;
    } else {
        block_size = num_threads * 8;
    }

    block_size = @max(num_threads, @min(block_size, n_points / 4 + 1));
    return block_size;
}

/// Port of update_component_bounds_from_block.
pub fn update_component_bounds_from_block(
    component_nearest_neighbor_dist: []f32,
    block_component_bounds: []const f32,
    point_components: []const i64,
    block_start: usize,
    block_end: usize,
) void {
    for (block_start..block_end) |i| {
        const component: usize = @intCast(point_components[i]);
        const block_bound = block_component_bounds[i - block_start];
        if (block_bound < component_nearest_neighbor_dist[component]) {
            component_nearest_neighbor_dist[component] = block_bound;
        }
    }
}

/// Port of boruvka_tree_query_reproducible (block-based).
pub fn boruvka_tree_query_reproducible(
    allocator: Allocator,
    tree: *const NumbaKDTree,
    node_components: []const i64,
    point_components: []const i64,
    core_distances: []const f32,
    block_size: usize,
) !QueryCandidates {
    const n = tree.data.len / tree.n_features;

    const candidate_distances = try allocator.alloc(f32, n);
    errdefer allocator.free(candidate_distances);
    @memset(candidate_distances, std.math.inf(f32));
    const candidate_indices = try allocator.alloc(i32, n);
    errdefer allocator.free(candidate_indices);
    @memset(candidate_indices, -1);
    const component_nearest_neighbor_dist = try allocator.alloc(f32, n);
    defer allocator.free(component_nearest_neighbor_dist);
    @memset(component_nearest_neighbor_dist, std.math.inf(f32));

    const max_block_component_bounds = try allocator.alloc(f32, block_size);
    defer allocator.free(max_block_component_bounds);
    @memset(max_block_component_bounds, std.math.inf(f32));

    var block_start: usize = 0;
    while (block_start < n) : (block_start += block_size) {
        const block_end = @min(block_start + block_size, n);
        const block_size_actual = block_end - block_start;

        for (0..block_size_actual) |b| max_block_component_bounds[b] = std.math.inf(f32);

        for (block_start..block_end) |i| { // prange, sequential here
            const point = tree.dataRow(i);
            const distance_lower_bound = point_to_node_lower_bound_rdist(
                tree.boundsLowerRow(0),
                tree.boundsUpperRow(0),
                point,
            );
            const current_component: usize = @intCast(point_components[i]);
            const local_component_bound = component_nearest_neighbor_dist[current_component .. current_component + 1];

            component_aware_query_recursion(
                tree,
                0,
                point,
                candidate_distances[i .. i + 1],
                candidate_indices[i .. i + 1],
                core_distances[i],
                core_distances,
                point_components[i],
                node_components,
                point_components,
                distance_lower_bound,
                local_component_bound,
            );

            max_block_component_bounds[i - block_start] = local_component_bound[0];
        }

        update_component_bounds_from_block(
            component_nearest_neighbor_dist,
            max_block_component_bounds,
            point_components,
            block_start,
            block_end,
        );
    }

    return QueryCandidates{ .distances = candidate_distances, .indices = candidate_indices, .allocator = allocator };
}

/// Port of initialize_boruvka_from_knn. `knn_indices`/`knn_distances` are
/// n_points x k row-major; the edge weight for point i is
/// max(core_distances[i], knn_distances[i,j]) at the FIRST j>=1 whose neighbor
/// k satisfies core_distances[i] >= core_distances[k] (break on first hit).
///
/// PORT NOTE: the reference accumulates in float64 (component_edges dtype is
/// float64), casting to float32 only on return; we compute the max in f32
/// (matching the eventual f32 storage) — the values are pure copies/max of
/// already-f32 quantities, so this is exact.
pub fn initialize_boruvka_from_knn(
    allocator: Allocator,
    knn_indices: []const i32,
    knn_distances: []const f32,
    k: usize,
    core_distances: []const f32,
    disjoint_set: *RankDisjointSet,
) !Edges {
    const n = knn_indices.len / k;

    // component_edges: one candidate edge per point (or none).
    const cand_to = try allocator.alloc(i32, n);
    defer allocator.free(cand_to);
    const cand_w = try allocator.alloc(f32, n);
    defer allocator.free(cand_w);
    const has_edge = try allocator.alloc(bool, n);
    defer allocator.free(has_edge);
    @memset(has_edge, false);

    for (0..n) |i| { // prange, sequential here
        for (1..k) |j| {
            const kk = knn_indices[i * k + j];
            if (core_distances[i] >= core_distances[@intCast(kk)]) {
                const edge_weight = @max(core_distances[i], knn_distances[i * k + j]);
                cand_to[i] = kk;
                cand_w[i] = edge_weight;
                has_edge[i] = true;
                break;
            }
        }
    }

    var point_a = try allocator.alloc(i32, n);
    errdefer allocator.free(point_a);
    var point_b = try allocator.alloc(i32, n);
    errdefer allocator.free(point_b);
    var distance = try allocator.alloc(f32, n);
    errdefer allocator.free(distance);
    var result_idx: usize = 0;

    for (0..n) |i| {
        if (!has_edge[i]) continue;
        const from_component = ds_find(disjoint_set, @intCast(i));
        const to_component = ds_find(disjoint_set, cand_to[i]);
        if (from_component != to_component) {
            point_a[result_idx] = @intCast(i);
            point_b[result_idx] = cand_to[i];
            distance[result_idx] = cand_w[i];
            result_idx += 1;
            ds_union_by_rank(disjoint_set, from_component, to_component);
        }
    }

    point_a = try allocator.realloc(point_a, result_idx);
    point_b = try allocator.realloc(point_b, result_idx);
    distance = try allocator.realloc(distance, result_idx);
    return Edges{ .point_a = point_a, .point_b = point_b, .distance = distance, .allocator = allocator };
}

fn countUnique(allocator: Allocator, point_components: []const i64, n: usize) !usize {
    // np.unique(point_components).shape[0]
    const seen = try allocator.alloc(bool, n);
    defer allocator.free(seen);
    @memset(seen, false);
    var count: usize = 0;
    for (point_components) |c| {
        const uc: usize = @intCast(c);
        if (!seen[uc]) {
            seen[uc] = true;
            count += 1;
        }
    }
    return count;
}

/// Port of parallel_boruvka. Builds the mutual-reachability MST over `tree`.
/// Returns n-1 edges (point_a, point_b, distance) with the distance column
/// sqrt-transformed once at the end. `tree` must be a kdtree built over the
/// same data.
pub fn parallel_boruvka(
    allocator: Allocator,
    tree: *const NumbaKDTree,
    n_threads: usize,
    min_samples: usize,
    reproducible: bool,
) !Edges {
    const n = tree.data.len / tree.n_features;

    var components_disjoint_set = try ds.ds_rank_create(allocator, n);
    defer components_disjoint_set.deinit();

    const point_components = try allocator.alloc(i64, n);
    defer allocator.free(point_components);
    for (0..n) |i| point_components[i] = @intCast(i);

    const node_components = try allocator.alloc(i64, tree.idx_start.len);
    defer allocator.free(node_components);
    @memset(node_components, -1);

    // Core distances via the kdtree kNN query (k = min_samples + 1) in rdist
    // space; core_distances[i] = last (largest) column of row i.
    var core_distances = try allocator.alloc(f32, n);
    defer allocator.free(core_distances);

    var initial_edges: Edges = undefined;
    if (min_samples > 1) {
        const kq = min_samples + 1;
        var q = try kdtree.parallel_tree_query(allocator, tree, tree.data, kq, true);
        defer q.deinit();
        for (0..n) |i| core_distances[i] = q.distances[i * kq + (kq - 1)];
        initial_edges = try initialize_boruvka_from_knn(allocator, q.indices, q.distances, kq, core_distances, &components_disjoint_set);
    } else {
        @memset(core_distances, 0);
        const kq = 2;
        var q = try kdtree.parallel_tree_query(allocator, tree, tree.data, kq, true);
        defer q.deinit();
        initial_edges = try initialize_boruvka_from_knn(allocator, q.indices, q.distances, kq, core_distances, &components_disjoint_set);
    }
    defer initial_edges.deinit();

    update_component_vectors(tree, &components_disjoint_set, node_components, point_components);

    // n_components = len(np.unique(point_components))
    var n_components = try countUnique(allocator, point_components, n);

    // Accumulate edges into a fixed n-1 buffer.
    const max_edges = n - 1;
    const all_a = try allocator.alloc(i32, max_edges);
    errdefer allocator.free(all_a);
    const all_b = try allocator.alloc(i32, max_edges);
    errdefer allocator.free(all_b);
    const all_d = try allocator.alloc(f32, max_edges);
    errdefer allocator.free(all_d);

    var n_edges: usize = initial_edges.point_a.len;
    @memcpy(all_a[0..n_edges], initial_edges.point_a);
    @memcpy(all_b[0..n_edges], initial_edges.point_b);
    @memcpy(all_d[0..n_edges], initial_edges.distance);

    while (n_components > 1) {
        var candidates: QueryCandidates = undefined;
        if (reproducible) {
            const block_size = calculate_block_size(n_components, n, n_threads);
            candidates = try boruvka_tree_query_reproducible(allocator, tree, node_components, point_components, core_distances, block_size);
        } else {
            candidates = try boruvka_tree_query(allocator, tree, node_components, point_components, core_distances);
        }
        defer candidates.deinit();

        var new_edges = try merge_components(allocator, &components_disjoint_set, candidates.indices, candidates.distances, point_components);
        defer new_edges.deinit();

        n_components -= new_edges.point_a.len;

        update_component_vectors(tree, &components_disjoint_set, node_components, point_components);

        if (new_edges.point_a.len > 0) {
            @memcpy(all_a[n_edges .. n_edges + new_edges.point_a.len], new_edges.point_a);
            @memcpy(all_b[n_edges .. n_edges + new_edges.point_b.len], new_edges.point_b);
            @memcpy(all_d[n_edges .. n_edges + new_edges.distance.len], new_edges.distance);
            n_edges += new_edges.point_a.len;
        }
    }

    // Final sqrt over the whole distance column (rdist -> euclidean).
    for (all_d) |*v| v.* = @sqrt(v.*);

    return Edges{ .point_a = all_a, .point_b = all_b, .distance = all_d, .allocator = allocator };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const fixture = @import("fixture.zig");

fn readMatrixF32(alloc: Allocator, arr: std.json.Array) !struct { data: []f32, rows: usize, cols: usize } {
    const rows = arr.items.len;
    const cols = if (rows > 0) arr.items[0].array.items.len else 0;
    const out = try alloc.alloc(f32, rows * cols);
    errdefer alloc.free(out);
    for (arr.items, 0..) |row, i| {
        for (row.array.items, 0..) |v, j| out[i * cols + j] = @floatCast(fixture.numberAsF64(v));
    }
    return .{ .data = out, .rows = rows, .cols = cols };
}

fn readMatrixI32(alloc: Allocator, arr: std.json.Array) !struct { data: []i32, rows: usize, cols: usize } {
    const rows = arr.items.len;
    const cols = if (rows > 0) arr.items[0].array.items.len else 0;
    const out = try alloc.alloc(i32, rows * cols);
    errdefer alloc.free(out);
    for (arr.items, 0..) |row, i| {
        for (row.array.items, 0..) |v, j| out[i * cols + j] = @intCast(v.integer);
    }
    return .{ .data = out, .rows = rows, .cols = cols };
}

fn readVecF32(alloc: Allocator, arr: std.json.Array) ![]f32 {
    const out = try alloc.alloc(f32, arr.items.len);
    for (arr.items, 0..) |v, i| out[i] = @floatCast(fixture.numberAsF64(v));
    return out;
}

test "tiny 3-point MST (min_samples=1): edges span all points" {
    const alloc = std.testing.allocator;
    // colinear points at x=0,1,3; MST edges: 0-1 (len 1) and 1-2 (len 2).
    const data = [_]f32{ 0, 0, 1, 0, 3, 0 };
    var tree = try kdtree.build_kdtree(alloc, &data, 2, 40);
    defer tree.deinit();

    var edges = try parallel_boruvka(alloc, &tree, 1, 1, false);
    defer edges.deinit();

    try std.testing.expectEqual(@as(usize, 2), edges.point_a.len);
    // total tree weight should be 1 + 2 = 3 (euclidean, min_samples=1 -> core 0)
    var total: f32 = 0;
    for (edges.distance) |d| total += d;
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), total, 1e-4);
}

test "all-identical points: MST has n-1 zero-length edges" {
    const alloc = std.testing.allocator;
    const n = 8;
    var data: [n * 2]f32 = undefined;
    for (0..n) |i| {
        data[i * 2] = 2.0;
        data[i * 2 + 1] = 2.0;
    }
    var tree = try kdtree.build_kdtree(alloc, &data, 2, 40);
    defer tree.deinit();

    var edges = try parallel_boruvka(alloc, &tree, 1, 1, false);
    defer edges.deinit();
    try std.testing.expectEqual(@as(usize, n - 1), edges.point_a.len);
    for (edges.distance) |d| try std.testing.expectApproxEqAbs(@as(f32, 0.0), d, 1e-6);
}

test "two clusters: MST connects them with exactly one bridging edge" {
    const alloc = std.testing.allocator;
    // cluster A near origin, cluster B near (10,10); 6 points total.
    const data = [_]f32{
        0.0,  0.0,
        0.2,  0.1,
        0.1,  0.3,
        10.0, 10.0,
        10.2, 9.9,
        9.9,  10.1,
    };
    var tree = try kdtree.build_kdtree(alloc, &data, 2, 40);
    defer tree.deinit();

    var edges = try parallel_boruvka(alloc, &tree, 1, 1, false);
    defer edges.deinit();
    try std.testing.expectEqual(@as(usize, 5), edges.point_a.len);

    // Exactly one edge should be "long" (the bridge between clusters).
    var long_edges: usize = 0;
    for (edges.distance) |d| {
        if (d > 5.0) long_edges += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), long_edges);
}

test "reproducible == non-reproducible at n_threads=1" {
    const alloc = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(77);
    const n = 40;
    const d = 3;
    var data: [n * d]f32 = undefined;
    for (&data) |*v| v.* = prng.random().float(f32) * 4 - 2;
    var tree = try kdtree.build_kdtree(alloc, &data, d, 10);
    defer tree.deinit();

    var e_false = try parallel_boruvka(alloc, &tree, 1, 5, false);
    defer e_false.deinit();
    var e_true = try parallel_boruvka(alloc, &tree, 1, 5, true);
    defer e_true.deinit();

    try std.testing.expectEqualSlices(i32, e_false.point_a, e_true.point_a);
    try std.testing.expectEqualSlices(i32, e_false.point_b, e_true.point_b);
    try std.testing.expectEqualSlices(f32, e_false.distance, e_true.distance);
}

test "fixture parity: initialize_boruvka_from_knn_n80_d4" {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "boruvka", "initialize_boruvka_from_knn_n80_d4");
    defer fx.deinit();

    const root = fx.value.object;
    const tol_value = root.get("tolerance").?;
    const inputs = root.get("inputs").?.object;

    const knn = try readMatrixI32(alloc, inputs.get("knn_indices").?.array);
    defer alloc.free(knn.data);
    const dists = try readMatrixF32(alloc, inputs.get("knn_distances_rdist").?.array);
    defer alloc.free(dists.data);
    const core = try readVecF32(alloc, inputs.get("core_distances").?.array);
    defer alloc.free(core);
    const n_points: usize = @intCast(inputs.get("n_points").?.integer);

    var disjoint_set = try ds.ds_rank_create(alloc, n_points);
    defer disjoint_set.deinit();

    var edges = try initialize_boruvka_from_knn(alloc, knn.data, dists.data, knn.cols, core, &disjoint_set);
    defer edges.deinit();

    const expected = root.get("expected").?.object;
    const exp_a = expected.get("point_a").?.array;
    const exp_b = expected.get("point_b").?.array;
    const exp_d = expected.get("distance").?.array;
    const dist_tol = fixture.toleranceForField(tol_value, "distance");

    try std.testing.expectEqual(exp_a.items.len, edges.point_a.len);
    for (0..edges.point_a.len) |i| {
        try std.testing.expectEqual(@as(i32, @intCast(exp_a.items[i].integer)), edges.point_a[i]);
        try std.testing.expectEqual(@as(i32, @intCast(exp_b.items[i].integer)), edges.point_b[i]);
        try fixture.expectF32(dist_tol, edges.distance[i], fixture.numberAsF64(exp_d.items[i]));
    }
}

fn checkParallelBoruvkaFixture(case: []const u8, reproducible: bool) !void {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "boruvka", case);
    defer fx.deinit();

    const root = fx.value.object;
    const tol_value = root.get("tolerance").?;
    const inputs = root.get("inputs").?.object;

    const m = try readMatrixF32(alloc, inputs.get("data").?.array);
    defer alloc.free(m.data);
    const leaf_size: usize = @intCast(inputs.get("leaf_size").?.integer);
    const n_threads: usize = @intCast(inputs.get("n_threads").?.integer);
    const min_samples: usize = @intCast(inputs.get("min_samples").?.integer);

    var tree = try kdtree.build_kdtree(alloc, m.data, m.cols, leaf_size);
    defer tree.deinit();

    var edges = try parallel_boruvka(alloc, &tree, n_threads, min_samples, reproducible);
    defer edges.deinit();

    const expected = root.get("expected").?.object;
    const exp_a = expected.get("point_a").?.array;
    const exp_b = expected.get("point_b").?.array;
    const exp_d = expected.get("distance").?.array;
    const dist_tol = fixture.toleranceForField(tol_value, "distance");

    try std.testing.expectEqual(exp_a.items.len, edges.point_a.len);
    for (0..edges.point_a.len) |i| {
        try std.testing.expectEqual(@as(i32, @intCast(exp_a.items[i].integer)), edges.point_a[i]);
        try std.testing.expectEqual(@as(i32, @intCast(exp_b.items[i].integer)), edges.point_b[i]);
        try fixture.expectF32(dist_tol, edges.distance[i], fixture.numberAsF64(exp_d.items[i]));
    }
}

test "fixture parity: parallel_boruvka_n80_d4_reproducible_false" {
    try checkParallelBoruvkaFixture("parallel_boruvka_n80_d4_reproducible_false", false);
}

test "fixture parity: parallel_boruvka_n80_d4_reproducible_true" {
    try checkParallelBoruvkaFixture("parallel_boruvka_n80_d4_reproducible_true", true);
}
