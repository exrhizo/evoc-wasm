//! Port of the float path of evoc/knn_graph.py: make_forest orchestration,
//! nn_descent's float distance transform, and the knn_graph driver.
//!
//! PORT NOTE: the reference draws rng_state (for nn_descent) and rng_states
//! (per RP-tree, inside make_forest) from numpy's Mersenne Twister, which we
//! cannot reproduce. Per PORT_NOTES the port takes rng_state + rng_states as
//! inputs and calls the internals (make_float_forest + nn_descent_float*)
//! directly. Normalization is assumed done by the caller (the fixture data
//! is pre-unit-normalized; knn_graph's norm path is a no-op there).

const std = @import("std");
const Allocator = std.mem.Allocator;
const rng = @import("rng.zig");
const nndf = @import("nndescent_float.zig");
const nnd = @import("nnd_heap.zig");

/// Result of the kNN graph: n_samples x n_neighbors row-major matrices.
/// `distances` carry the float distance transform (see nn_descent below).
pub const KnnResult = struct {
    indices: []i32,
    distances: []f32,
    n_samples: usize,
    n_neighbors: usize,
    allocator: Allocator,

    pub fn deinit(self: *KnnResult) void {
        self.allocator.free(self.indices);
        self.allocator.free(self.distances);
        self.* = undefined;
    }
};

/// A padded leaf array stacked over all trees: total_leaves x width,
/// rows padded with -1.
pub const LeafArray = struct {
    data: []i32,
    n_leaves: usize,
    width: usize,
    allocator: Allocator,

    pub fn deinit(self: *LeafArray) void {
        self.allocator.free(self.data);
        self.* = undefined;
    }
};

/// Port of make_forest (float path): build a forest of RP-trees with the
/// given per-tree rng_states, then vstack the per-tree leaf arrays padded to
/// the max leaf width across all trees (matching np.pad + np.vstack).
pub fn make_forest(
    allocator: Allocator,
    data: []const f32,
    n_features: usize,
    rng_states: []rng.State,
    leaf_size: i64,
    max_depth: i64,
) !LeafArray {
    var forest = try nndf.make_float_forest(allocator, data, n_features, rng_states, leaf_size, max_depth);
    defer forest.deinit();

    var max_leaf_size: usize = 0;
    var total_leaves: usize = 0;
    for (forest.arrays) |la| {
        max_leaf_size = @max(max_leaf_size, la.width);
        total_leaves += la.n_leaves;
    }

    const out = try allocator.alloc(i32, total_leaves * max_leaf_size);
    errdefer allocator.free(out);
    @memset(out, -1);

    var row: usize = 0;
    for (forest.arrays) |la| {
        for (0..la.n_leaves) |i| {
            // copy la's row (width la.width) into the padded row
            @memcpy(
                out[(row + i) * max_leaf_size ..][0..la.width],
                la.data[i * la.width ..][0..la.width],
            );
        }
        row += la.n_leaves;
    }

    return LeafArray{ .data = out, .n_leaves = total_leaves, .width = max_leaf_size, .allocator = allocator };
}

/// Float distance transform applied by nn_descent for float input:
/// d <- max(-log2(-d), 0.0).
///
/// PORT NOTE (from PORT_NOTES): float path clamps at 0. int8 uses 1/(-d),
/// uint8 uses unclamped -log2(-d) — those are for later phases.
pub fn floatDistanceTransform(distances: []f32) void {
    for (distances) |*d| {
        const t: f32 = @floatCast(-std.math.log2(-@as(f64, d.*)));
        d.* = @max(t, 0.0);
    }
}

/// Port of nn_descent (float path). Runs the sorted or unsorted descent then
/// applies the float distance transform in place. Returns the deheap-sorted
/// graph as a KnnResult (indices + transformed distances).
pub fn nn_descent(
    allocator: Allocator,
    data: []const f32,
    n_features: usize,
    n_neighbors: usize,
    rng_state: *rng.State,
    effective_max_candidates: usize,
    n_iters: usize,
    delta: f64,
    delta_improv: ?f64,
    leaf_array: []const i32,
    leaf_width: usize,
    use_sorted_updates: bool,
    n_threads: usize,
) !KnnResult {
    const n_samples = data.len / n_features;

    var graph = if (use_sorted_updates)
        try nndf.nn_descent_float_sorted(
            allocator,
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
    else
        try nndf.nn_descent_float(
            allocator,
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
        );
    // graph owns indices/distances; hand them to KnnResult and neutralize
    // graph so its deinit doesn't free the slices we keep.
    floatDistanceTransform(graph.distances);

    const indices = graph.indices;
    const distances = graph.distances;
    allocator.free(graph.flags);
    graph.indices = &.{};
    graph.distances = &.{};
    graph.flags = &.{};

    return KnnResult{
        .indices = indices,
        .distances = distances,
        .n_samples = n_samples,
        .n_neighbors = n_neighbors,
        .allocator = allocator,
    };
}

/// Port of the knn_graph driver (float path). `data` is assumed already unit-
/// normalized (the reference's normalization is a no-op for such input).
/// `rng_state` seeds nn_descent; `rng_states` (one row per tree) seed the RP
/// forest — order matches make_forest's per-tree draw order.
pub fn knn_graph(
    allocator: Allocator,
    data: []const f32,
    n_features: usize,
    n_neighbors: usize,
    rng_state: *rng.State,
    rng_states: []rng.State,
    leaf_size: i64,
    n_iters: usize,
    effective_max_candidates: usize,
    delta: f64,
    delta_improv: ?f64,
    max_rptree_depth: i64,
    use_sorted_updates: bool,
    n_threads: usize,
) !KnnResult {
    var leaf = try make_forest(allocator, data, n_features, rng_states, leaf_size, max_rptree_depth);
    defer leaf.deinit();

    return nn_descent(
        allocator,
        data,
        n_features,
        n_neighbors,
        rng_state,
        effective_max_candidates,
        n_iters,
        delta,
        delta_improv,
        leaf.data,
        leaf.width,
        use_sorted_updates,
        n_threads,
    );
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

test "floatDistanceTransform: -1 -> 0, negatives map through -log2(-d) clamped" {
    var d = [_]f32{ -1.0, -0.5, -0.25 };
    floatDistanceTransform(&d);
    // -log2(1)=0; -log2(0.5)=1; -log2(0.25)=2
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), d[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), d[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), d[2], 1e-6);
}

test "floatDistanceTransform clamps positive log to 0" {
    // -log2(-d) < 0 when -d > 1, i.e. d < -1 (cosine sim just below -1 by fp)
    var d = [_]f32{-1.0000001};
    floatDistanceTransform(&d);
    try std.testing.expectEqual(@as(f32, 0.0), d[0]);
}

test "make_forest stacks and pads leaf arrays to a common width" {
    const alloc = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(3);
    const n = 30;
    const d = 4;
    var data: [n * d]f32 = undefined;
    for (&data) |*v| v.* = prng.random().float(f32) - 0.5;

    var rng_states = [_]rng.State{
        .{ 111, 222, 333 },
        .{ 444, 555, 666 },
    };
    var leaf = try make_forest(alloc, &data, d, &rng_states, 8, 200);
    defer leaf.deinit();

    // every non-negative index is a valid point id; padding only trails rows.
    for (0..leaf.n_leaves) |i| {
        var seen_neg = false;
        for (0..leaf.width) |j| {
            const v = leaf.data[i * leaf.width + j];
            if (v < 0) {
                seen_neg = true;
            } else {
                try std.testing.expect(!seen_neg);
                try std.testing.expect(v < n);
            }
        }
    }
}

test "fixture parity: knn_graph_n120_d8_k10_seed42" {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "graph", "knn_graph_n120_d8_k10_seed42");
    defer fx.deinit();

    const root = fx.value.object;
    const tol_value = root.get("tolerance").?;
    const inputs = root.get("inputs").?.object;

    const m = try readMatrixF32(alloc, inputs.get("data").?.array);
    defer alloc.free(m.data);
    const n_neighbors: usize = @intCast(inputs.get("n_neighbors").?.integer);
    const leaf_size: i64 = inputs.get("leaf_size").?.integer;
    const n_iters: usize = @intCast(inputs.get("n_iters").?.integer);
    const emc: usize = @intCast(inputs.get("effective_max_candidates").?.integer);
    const delta = fixture.numberAsF64(inputs.get("delta").?);
    const delta_improv = fixture.numberAsF64(inputs.get("delta_improv").?);
    const use_sorted = inputs.get("use_sorted_updates").?.bool;

    var rng_state: rng.State = undefined;
    const rs_in = inputs.get("rng_state").?.array;
    for (0..3) |i| rng_state[i] = rs_in.items[i].integer;

    const rss_in = inputs.get("rng_states").?.array;
    const rng_states = try alloc.alloc(rng.State, rss_in.items.len);
    defer alloc.free(rng_states);
    for (rss_in.items, 0..) |row, t| {
        for (0..3) |j| rng_states[t][j] = row.array.items[j].integer;
    }

    var result = try knn_graph(
        alloc,
        m.data,
        m.cols,
        n_neighbors,
        &rng_state,
        rng_states,
        leaf_size,
        n_iters,
        emc,
        delta,
        delta_improv,
        200,
        use_sorted,
        1, // fixtures generated with NUMBA_NUM_THREADS=1
    );
    defer result.deinit();

    const expected = root.get("expected").?.object;
    const exp_inds = expected.get("nn_inds").?.array;
    const exp_dists = expected.get("nn_dists").?.array;
    const dist_tol = fixture.toleranceForField(tol_value, "nn_dists");

    for (0..m.rows) |i| {
        const idx_row = exp_inds.items[i].array;
        const dist_row = exp_dists.items[i].array;
        for (0..n_neighbors) |j| {
            try std.testing.expectEqual(
                @as(i32, @intCast(idx_row.items[j].integer)),
                result.indices[i * n_neighbors + j],
            );
            try fixture.expectF32(dist_tol, result.distances[i * n_neighbors + j], fixture.numberAsF64(dist_row.items[j]));
        }
    }
}
