//! Port of evoc/clustering.py: `build_cluster_layers` only.
//! `evoc_clusters`/`EVoC.fit_predict` are explicitly out of scope (see
//! PORT_NOTES.md "clustering.py (phase 8)" — they depend on unportable
//! sklearn PCA internals reached through `label_propagation_init`'s
//! recursive path). `build_cluster_layers` takes a pre-computed embedding as
//! input and has no random_state of its own, which is why it is the phase-8
//! capstone fixture target instead.

const std = @import("std");
const Allocator = std.mem.Allocator;
const ct = @import("cluster_trees.zig");
const cu = @import("clustering_utilities.zig");
const kdtree = @import("kdtree.zig");
const boruvka = @import("boruvka.zig");

pub const ClusterLayers = struct {
    cluster_layers: [][]i64,
    membership_strength_layers: [][]f32,
    persistence_scores: []f64,
    allocator: Allocator,

    pub fn deinit(self: *ClusterLayers) void {
        for (self.cluster_layers) |l| self.allocator.free(l);
        self.allocator.free(self.cluster_layers);
        for (self.membership_strength_layers) |l| self.allocator.free(l);
        self.allocator.free(self.membership_strength_layers);
        self.allocator.free(self.persistence_scores);
        self.* = undefined;
    }
};

fn maxLabel(layer: []const i64) i64 {
    var m: i64 = layer[0];
    for (layer) |v| {
        if (v > m) m = v;
    }
    return m;
}

/// Port of build_cluster_layers.
///
/// PORT NOTE: only the `base_n_clusters=None` default path is fixtured (see
/// PORT_NOTES.md); the `base_n_clusters is not None` binary-search branch is
/// a known coverage gap and not implemented here (callers wanting it should
/// compose clustering_utilities._binary_search_for_n_clusters directly, as
/// PORT_NOTES.md suggests).
pub fn build_cluster_layers(
    allocator: Allocator,
    data: []const f32,
    n_features: usize,
    min_samples: usize,
    base_min_cluster_size: i64,
    reproducible_flag: bool,
    min_similarity_threshold: f64,
    max_layers: usize,
) !ClusterLayers {
    const n_samples = data.len / n_features;
    const min_cluster_size = base_min_cluster_size;

    var tree = try kdtree.build_kdtree(allocator, data, n_features, 40);
    defer tree.deinit();

    var edges = try boruvka.parallel_boruvka(allocator, &tree, 1, min_samples, reproducible_flag);
    defer edges.deinit();

    // sorted_mst = edges[argsort(edges.T[2])] — numpy's actual unstable
    // quicksort tie-break, replicated via cu.argsort_f32 (see
    // clustering_utilities.zig for why: ties in mutual-reachability
    // distances are common and a stable-sort substitution can produce a
    // structurally different tree).
    const m = edges.point_a.len;
    const order = try cu.argsort_f32(allocator, edges.distance);
    defer allocator.free(order);

    const mst_left = try allocator.alloc(f32, m);
    defer allocator.free(mst_left);
    const mst_right = try allocator.alloc(f32, m);
    defer allocator.free(mst_right);
    const mst_dist = try allocator.alloc(f32, m);
    defer allocator.free(mst_dist);
    for (0..m) |i| {
        mst_left[i] = @floatFromInt(edges.point_a[order[i]]);
        mst_right[i] = @floatFromInt(edges.point_b[order[i]]);
        mst_dist[i] = edges.distance[order[i]];
    }

    var linkage = try ct.mst_to_linkage_tree(allocator, mst_left, mst_right, mst_dist);
    defer linkage.deinit();

    // base_n_clusters=None default path: plain condense_tree at
    // base_min_cluster_size.
    var condensed_tree = try ct.condense_tree(allocator, linkage.left, linkage.right, linkage.delta, linkage.size, base_min_cluster_size);
    defer condensed_tree.deinit();
    const base_leaves = try ct.extract_leaves(allocator, &condensed_tree);
    defer allocator.free(base_leaves);
    const base_clusters = try ct.get_cluster_label_vector(allocator, &condensed_tree, base_leaves, 0.0, n_samples);
    errdefer allocator.free(base_clusters);
    const base_strengths = try ct.get_point_membership_strength_vector(allocator, &condensed_tree, base_leaves, base_clusters);
    errdefer allocator.free(base_strengths);

    // mask = condensed_tree.child >= n_samples (internal-node-only view —
    // NOT the child_size > 1 mask used elsewhere in the reference).
    const mask = try allocator.alloc(bool, condensed_tree.child.len);
    defer allocator.free(mask);
    for (0..condensed_tree.child.len) |i| mask[i] = condensed_tree.child[i] >= @as(i64, @intCast(n_samples));
    var cluster_tree = try ct.mask_condensed_tree(allocator, &condensed_tree, mask);
    defer cluster_tree.deinit();

    var births: []f32 = &.{};
    var deaths: []f32 = &.{};
    var lambda_deaths_arr: []f32 = &.{};
    var sizes: []f64 = &.{};
    var total_persistence: []f32 = &.{};
    var peaks: []i64 = &.{};
    var owns_barcode = false;
    var owns_persistence = false;
    var owns_peaks = false;
    defer if (owns_barcode) {
        allocator.free(births);
        allocator.free(deaths);
        allocator.free(lambda_deaths_arr);
    };
    defer if (owns_persistence) {
        allocator.free(sizes);
        allocator.free(total_persistence);
    };
    defer if (owns_peaks) allocator.free(peaks);

    // Guard: an empty or degenerate cluster tree skips the whole
    // barcode/persistence/peaks computation (legitimate output shape, not an
    // error — must not crash on empty-array indexing).
    if (cluster_tree.child.len > 0 and cluster_tree.child[cluster_tree.child.len - 1] >= @as(i64, @intCast(n_samples))) {
        const bc = try cu.min_cluster_size_barcode(allocator, &cluster_tree, @intCast(n_samples), min_cluster_size);
        allocator.free(bc.parents);
        births = bc.size_births;
        deaths = bc.size_deaths;
        lambda_deaths_arr = bc.lambda_deaths;
        owns_barcode = true;

        // births/deaths are f32 in min_cluster_size_barcode but the barcode
        // fields are conceptually f64-compatible sizes; widen for the
        // persistence/peak-selection helpers, which operate in f64.
        const births_f64 = try allocator.alloc(f64, births.len);
        defer allocator.free(births_f64);
        for (births, 0..) |b, i| births_f64[i] = b;
        const deaths_f64 = try allocator.alloc(f64, deaths.len);
        defer allocator.free(deaths_f64);
        for (deaths, 0..) |d, i| deaths_f64[i] = d;

        const pres = try cu.compute_total_persistence(allocator, births_f64, deaths_f64, lambda_deaths_arr);
        sizes = pres.sizes;
        total_persistence = pres.total_persistence;
        owns_persistence = true;

        peaks = try cu.find_peaks(f32, allocator, total_persistence);
        owns_peaks = true;
    }

    // births/deaths as f64 for select_diverse_peaks / estimate_cluster_similarity.
    const births_f64_full = try allocator.alloc(f64, births.len);
    defer allocator.free(births_f64_full);
    for (births, 0..) |b, i| births_f64_full[i] = b;
    const deaths_f64_full = try allocator.alloc(f64, deaths.len);
    defer allocator.free(deaths_f64_full);
    for (deaths, 0..) |d, i| deaths_f64_full[i] = d;
    const sizes_f64 = try allocator.alloc(f64, sizes.len);
    defer allocator.free(sizes_f64);
    for (sizes, 0..) |s, i| sizes_f64[i] = s;

    // Always include the base layer first.
    var cluster_layers = std.ArrayList([]i64).init(allocator);
    defer cluster_layers.deinit();
    var membership_strength_layers = std.ArrayList([]f32).init(allocator);
    defer membership_strength_layers.deinit();
    var persistence_scores = std.ArrayList(f64).init(allocator);
    defer persistence_scores.deinit();

    try cluster_layers.append(base_clusters);
    try membership_strength_layers.append(base_strengths);
    try persistence_scores.append(0.0); // Base layer persistence score is hardcoded 0.0.

    const max_peak_layers = if (max_layers > 0) max_layers - 1 else 0;
    const selected_peaks = try cu.select_diverse_peaks(allocator, peaks, total_persistence, sizes_f64, births_f64_full, deaths_f64_full, min_similarity_threshold, max_peak_layers);
    defer allocator.free(selected_peaks);

    for (selected_peaks) |peak| {
        const best_birth = sizes_f64[@intCast(peak)];
        const persistence: f64 = total_persistence[@intCast(peak)];

        var selected_clusters = std.ArrayList(i64).init(allocator);
        defer selected_clusters.deinit();
        for (0..births_f64_full.len) |i| {
            if (births_f64_full[i] <= best_birth and deaths_f64_full[i] > best_birth) {
                try selected_clusters.append(@as(i64, @intCast(i)) + @as(i64, @intCast(n_samples)));
            }
        }

        const res = try cu.extract_clusters_by_id(allocator, &condensed_tree, selected_clusters.items);
        try cluster_layers.append(res.labels);
        try membership_strength_layers.append(res.strengths);
        try persistence_scores.append(persistence);
    }

    // Sort layers by n_clusters_per_layer (layer.max()+1) descending, stable
    // (argsort(...)[::-1] on a stable ascending sort).
    const n_layers = cluster_layers.items.len;
    const sort_order = try allocator.alloc(usize, n_layers);
    defer allocator.free(sort_order);
    for (0..n_layers) |i| sort_order[i] = i;
    const layer_counts = try allocator.alloc(i64, n_layers);
    defer allocator.free(layer_counts);
    for (0..n_layers) |i| layer_counts[i] = maxLabel(cluster_layers.items[i]) + 1;

    const SortCtx = struct {
        counts: []const i64,
        fn lessThan(ctx: @This(), a: usize, b: usize) bool {
            return ctx.counts[a] < ctx.counts[b];
        }
    };
    std.mem.sort(usize, sort_order, SortCtx{ .counts = layer_counts }, SortCtx.lessThan);
    std.mem.reverse(usize, sort_order);

    const out_clusters = try allocator.alloc([]i64, n_layers);
    const out_strengths = try allocator.alloc([]f32, n_layers);
    const out_persistence = try allocator.alloc(f64, n_layers);
    for (sort_order, 0..) |src, dst| {
        out_clusters[dst] = cluster_layers.items[src];
        out_strengths[dst] = membership_strength_layers.items[src];
        out_persistence[dst] = persistence_scores.items[src];
    }

    return ClusterLayers{
        .cluster_layers = out_clusters,
        .membership_strength_layers = out_strengths,
        .persistence_scores = out_persistence,
        .allocator = allocator,
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const std_testing = std.testing;
const fixture = @import("fixture.zig");

fn readMatrixF32(alloc: Allocator, arr: std.json.Array) !struct { data: []f32, rows: usize, cols: usize } {
    const rows = arr.items.len;
    const cols = if (rows > 0) arr.items[0].array.items.len else 0;
    const out = try alloc.alloc(f32, rows * cols);
    for (arr.items, 0..) |row, i| {
        for (row.array.items, 0..) |v, j| out[i * cols + j] = @floatCast(fixture.numberAsF64(v));
    }
    return .{ .data = out, .rows = rows, .cols = cols };
}

test "fixture parity: build_cluster_layers_n126" {
    const alloc = std_testing.allocator;
    const fx = try fixture.readFixture(alloc, "clustering", "build_cluster_layers_n126");
    defer fx.deinit();
    const root = fx.value.object;
    const tol_value = root.get("tolerance").?;
    const inputs = root.get("inputs").?.object;

    const m = try readMatrixF32(alloc, inputs.get("embedding").?.array);
    defer alloc.free(m.data);
    const min_samples: usize = @intCast(inputs.get("min_samples").?.integer);
    const base_min_cluster_size = inputs.get("base_min_cluster_size").?.integer;
    const reproducible_flag = inputs.get("reproducible_flag").?.bool;
    const min_similarity_threshold = fixture.numberAsF64(inputs.get("min_similarity_threshold").?);
    const max_layers: usize = @intCast(inputs.get("max_layers").?.integer);

    var result = try build_cluster_layers(alloc, m.data, m.cols, min_samples, base_min_cluster_size, reproducible_flag, min_similarity_threshold, max_layers);
    defer result.deinit();

    const expected = root.get("expected").?.object;
    const exp_layers = expected.get("cluster_layers").?.array;
    const exp_strengths = expected.get("membership_strengths").?.array;
    const exp_persistence = expected.get("persistence_scores").?.array;
    const strength_tol = fixture.toleranceForField(tol_value, "membership_strengths");
    const persistence_tol = fixture.toleranceForField(tol_value, "persistence_scores");

    try std_testing.expectEqual(exp_layers.items.len, result.cluster_layers.len);
    for (0..result.cluster_layers.len) |li| {
        const exp_layer = exp_layers.items[li].array;
        const exp_strength_layer = exp_strengths.items[li].array;
        try std_testing.expectEqual(exp_layer.items.len, result.cluster_layers[li].len);
        for (0..result.cluster_layers[li].len) |i| {
            try std_testing.expectEqual(@as(i64, exp_layer.items[i].integer), result.cluster_layers[li][i]);
            try fixture.expectF32(strength_tol, result.membership_strength_layers[li][i], fixture.numberAsF64(exp_strength_layer.items[i]));
        }
        try std_testing.expectApproxEqAbs(fixture.numberAsF64(exp_persistence.items[li]), result.persistence_scores[li], if (persistence_tol == .approx) persistence_tol.approx.abs + persistence_tol.approx.rel * @abs(fixture.numberAsF64(exp_persistence.items[li])) else 1e-9);
    }
}
