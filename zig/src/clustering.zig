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
const gc = @import("graph_construction.zig");
const knn_graph_mod = @import("knn_graph.zig");
const label_propagation = @import("label_propagation.zig");
const node_embedding = @import("node_embedding.zig");
const rng = @import("rng.zig");

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

/// Options for `evoc_clusters`, mirroring the reference's keyword arguments
/// (all have the reference's defaults except `reproducible_flag`, which is a
/// no-op here -- this port always uses the reproducible/`_repr` node-
/// embedding kernel, matching build_cluster_layers's PORT NOTE).
pub const EvocClustersOptions = struct {
    noise_level: f32 = 0.5,
    base_min_cluster_size: i64 = 5,
    n_neighbors: usize = 15,
    min_samples: usize = 5,
    n_epochs: usize = 50,
    neighbor_scale: f32 = 1.0,
    min_similarity_threshold: f64 = 0.2,
    max_layers: usize = 10,
    n_label_prop_iter: usize = 20,
    /// `min(max(n_neighbors/4, 4), 15)` if `null`, matching the reference's
    /// `node_embedding_dim` default derivation.
    node_embedding_dim: ?usize = null,
};

pub const EvocClustersResult = struct {
    layers: ClusterLayers,
    nn_inds: []i32,
    nn_dists: []f32,
    allocator: Allocator,

    pub fn deinit(self: *EvocClustersResult) void {
        self.layers.deinit();
        self.allocator.free(self.nn_inds);
        self.allocator.free(self.nn_dists);
        self.* = undefined;
    }
};

/// Port of `evoc_clusters` (module-level function). `data` is a flat
/// row-major `n_samples x n_features` array of float embeddings (the int8/
/// uint8 quantized paths are out of scope, matching the rest of this port --
/// see PORTING.md's module map: `nndescent_int8`/`nndescent_uint8` are listed
/// "later"). `seed` drives every random draw in the pipeline (RP-tree
/// construction, NN-descent candidate sampling, label-propagation partition
/// seeding + tie-breaks, randomized-PCA base cases, node-embedding SGD) via
/// this project's own `tau_rand` RNG -- same seed -> same output,
/// cross-language reproducible against the Rust port, but not bit-exact with
/// the Python reference (see the module PORT NOTE on `evoc_clusters` scope).
///
/// Returns `ClusterLayers` plus the raw knn indices/distances (flat
/// row-major `n_samples x n_neighbors`), matching the reference's return
/// shape when `approx_n_clusters` is not requested (the
/// `approx_n_clusters`/`binary_search_for_n_clusters` single-layer branch and
/// `return_duplicates` are not ported here -- out of scope for this pass, see
/// STATUS.md).
pub fn evoc_clusters(
    allocator: Allocator,
    data: []const f32,
    n_features: usize,
    seed: i64,
    opts: EvocClustersOptions,
) !EvocClustersResult {
    const n_samples = data.len / n_features;

    // Unit-normalize rows (cosine distance), matching knn_graph's float path.
    const normalized = try allocator.dupe(f32, data);
    defer allocator.free(normalized);
    for (0..n_samples) |i| {
        const row = normalized[i * n_features .. i * n_features + n_features];
        var norm_sq: f32 = 0.0;
        for (row) |v| norm_sq += v * v;
        const norm = @sqrt(norm_sq);
        if (norm != 0.0) {
            for (row) |*v| v.* /= norm;
        }
    }

    var state: rng.State = undefined;
    rng.seed(&state, seed);

    const drawRngState = struct {
        fn call(s: *rng.State) rng.State {
            return .{
                @as(i64, rng.tau_rand_int(s)),
                @as(i64, rng.tau_rand_int(s)),
                @as(i64, rng.tau_rand_int(s)),
            };
        }
    }.call;

    // knn_graph parameters, single-threaded defaults (see PORT_NOTES.md).
    const n_trees: usize = 4;
    const leaf_size: i64 = @intCast(@max(10, opts.n_neighbors));
    const n_iters: usize = @max(5, @as(usize, @intFromFloat(@round(std.math.log2(@as(f64, @floatFromInt(n_samples)))))));
    const effective_max_candidates: usize = @min(60, @as(usize, @intFromFloat(@as(f64, @floatFromInt(opts.n_neighbors)) * 1.5)));

    var rng_state = drawRngState(&state);
    const rng_states = try allocator.alloc(rng.State, n_trees);
    defer allocator.free(rng_states);
    for (rng_states) |*rs| rs.* = drawRngState(&state);

    var knn = try knn_graph_mod.knn_graph(
        allocator,
        normalized,
        n_features,
        opts.n_neighbors,
        &rng_state,
        rng_states,
        leaf_size,
        n_iters,
        effective_max_candidates,
        0.001,
        0.001,
        200,
        true,
        1,
    );
    defer knn.deinit();

    const effective_n_neighbors: f32 = opts.neighbor_scale * @as(f32, @floatFromInt(opts.n_neighbors));
    var csr = try gc.neighbor_graph_matrix(
        allocator,
        effective_n_neighbors,
        knn.indices,
        knn.distances,
        n_samples,
        opts.n_neighbors,
        true,
    );
    defer csr.deinit();

    const n_components = opts.node_embedding_dim orelse @max(4, @min(15, opts.n_neighbors / 4));

    const approx_n_parts: usize = @intCast(std.math.clamp(
        @as(i64, @intFromFloat(8.0 * @sqrt(@as(f64, @floatFromInt(n_samples))))),
        256,
        16384,
    ));

    const init_embedding = try label_propagation.label_propagation_init_recursive(
        allocator,
        csr.indptr,
        csr.indices,
        csr.data,
        n_samples,
        .{
            .n_label_prop_iter = opts.n_label_prop_iter,
            .n_embedding_epochs = opts.n_epochs,
            .approx_n_parts = approx_n_parts,
            .n_components = n_components,
            .scaling = 0.5,
            .noise_level = opts.noise_level,
            .recursive_init = true,
            .base_init_threshold = 64,
            .data_full = normalized,
            .n_features = n_features,
        },
        &state,
    );
    defer allocator.free(init_embedding);

    const indptr_u32 = try allocator.alloc(u32, csr.indptr.len);
    defer allocator.free(indptr_u32);
    for (csr.indptr, 0..) |v, i| indptr_u32[i] = @intCast(v);
    const indices_u32 = try allocator.alloc(u32, csr.indices.len);
    defer allocator.free(indices_u32);
    for (csr.indices, 0..) |v, i| indices_u32[i] = @intCast(v);

    const embedding = try node_embedding.node_embedding_repr_self_driven(
        allocator,
        indptr_u32,
        indices_u32,
        csr.data,
        n_samples,
        n_components,
        opts.n_epochs,
        init_embedding,
        0.1, // initial_alpha
        1.0, // negative_sample_rate
        opts.noise_level,
        &state,
    );
    defer allocator.free(embedding);

    const layers = try build_cluster_layers(
        allocator,
        embedding,
        n_components,
        opts.min_samples,
        opts.base_min_cluster_size,
        false, // reproducible_flag: bit-identical to true at n_threads=1
        opts.min_similarity_threshold,
        opts.max_layers,
    );

    return EvocClustersResult{
        .layers = layers,
        .nn_inds = try allocator.dupe(i32, knn.indices),
        .nn_dists = try allocator.dupe(f32, knn.distances),
        .allocator = allocator,
    };
}

/// Faithful port of `EVoC.fit_predict`'s single-best-layer selection: when
/// `evoc_clusters` (with `approx_n_clusters=null`, the only mode ported
/// here) returns multiple layers, `fit_predict` picks the layer with the
/// highest persistence score (`np.argmax`, first-max-wins on ties).
pub const EVoC = struct {
    opts: EvocClustersOptions,
    seed: i64,
    labels: []i64 = &.{},
    membership_strengths: []f32 = &.{},
    result: ?EvocClustersResult = null,
    allocator: Allocator,

    pub fn init(allocator: Allocator, seed: i64, opts: EvocClustersOptions) EVoC {
        return .{ .opts = opts, .seed = seed, .allocator = allocator };
    }

    pub fn deinit(self: *EVoC) void {
        if (self.result) |*r| {
            var mr = r.*;
            mr.deinit();
        }
        self.* = undefined;
    }

    /// Port of `EVoC.fit_predict`. Returns `labels_`, and stores the full
    /// `EvocClustersResult` (`self.result`) for callers who want
    /// `nn_inds`/`nn_dists`/per-layer data (minus `duplicates_`, out of
    /// scope for this pass). `self.labels`/`self.membership_strengths`
    /// alias into `self.result`'s owned layer slices -- do not free them
    /// separately.
    pub fn fit_predict(self: *EVoC, data: []const f32, n_features: usize) ![]const i64 {
        if (self.result) |*r| {
            var mr = r.*;
            mr.deinit();
            self.result = null;
        }

        const res = try evoc_clusters(self.allocator, data, n_features, self.seed, self.opts);
        self.result = res;

        const layers = self.result.?.layers;
        const best_layer: usize = if (layers.cluster_layers.len == 1) 0 else blk: {
            var best: usize = 0;
            var best_val: f64 = -std.math.inf(f64);
            for (layers.persistence_scores, 0..) |v, i| {
                if (v > best_val) {
                    best_val = v;
                    best = i;
                }
            }
            break :blk best;
        };

        self.labels = layers.cluster_layers[best_layer];
        self.membership_strengths = layers.membership_strength_layers[best_layer];
        return self.labels;
    }
};

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

// ---------------------------------------------------------------------------
// End-to-end ARI-based validation for evoc_clusters/EVoC.fit_predict (phase
// 9). NOT a bit-exact fixture test like the rest of this port --
// label_propagation_init_recursive's base case reaches randomized PCA, which
// is intentionally NOT bit-parity with sklearn's internal randomized-SVD
// draws (see PORTING.md / PORT_NOTES.md / rust/core/PCA_VALIDATION.md).
// Instead this compares this Zig port's cluster labels against the same
// cached real `evoc.EVoC(random_state=7).fit_predict(X)` run the Rust port
// uses (fixtures/clustering/evoc_clusters_e2e_ari_check.json, generated by
// scripts/gen_evoc_clusters_ari_fixture.py), via Adjusted Rand Index, with
// the same pragmatic bar of ARI > 0.8 used on the Rust side (so the two
// numbers are directly comparable).
// ---------------------------------------------------------------------------

/// `n choose 2`, for non-negative `n`.
fn nChoose2(n: i64) f64 {
    const nf: f64 = @floatFromInt(n);
    return nf * (nf - 1.0) / 2.0;
}

/// Adjusted Rand Index between two integer label assignments of the same
/// length (matches `sklearn.metrics.adjusted_rand_score`'s definition,
/// including its treatment of any label value -- e.g. `-1` for noise -- as
/// just another ordinary cluster id, not special-cased).
fn adjustedRandIndex(allocator: Allocator, a: []const i64, b: []const i64) !f64 {
    std.debug.assert(a.len == b.len);
    const n = a.len;

    const a_sorted = try allocator.dupe(i64, a);
    defer allocator.free(a_sorted);
    std.mem.sort(i64, a_sorted, {}, std.sort.asc(i64));
    var a_labels = std.ArrayList(i64).init(allocator);
    defer a_labels.deinit();
    for (a_sorted, 0..) |v, i| {
        if (i == 0 or v != a_sorted[i - 1]) try a_labels.append(v);
    }

    const b_sorted = try allocator.dupe(i64, b);
    defer allocator.free(b_sorted);
    std.mem.sort(i64, b_sorted, {}, std.sort.asc(i64));
    var b_labels = std.ArrayList(i64).init(allocator);
    defer b_labels.deinit();
    for (b_sorted, 0..) |v, i| {
        if (i == 0 or v != b_sorted[i - 1]) try b_labels.append(v);
    }

    var a_index = std.AutoHashMap(i64, usize).init(allocator);
    defer a_index.deinit();
    for (a_labels.items, 0..) |l, i| try a_index.put(l, i);
    var b_index = std.AutoHashMap(i64, usize).init(allocator);
    defer b_index.deinit();
    for (b_labels.items, 0..) |l, i| try b_index.put(l, i);

    const na = a_labels.items.len;
    const nb = b_labels.items.len;
    const contingency = try allocator.alloc(i64, na * nb);
    defer allocator.free(contingency);
    @memset(contingency, 0);
    for (0..n) |i| {
        const ai = a_index.get(a[i]).?;
        const bi = b_index.get(b[i]).?;
        contingency[ai * nb + bi] += 1;
    }

    var sum_comb_c: f64 = 0.0;
    for (contingency) |c| sum_comb_c += nChoose2(c);

    const row_sums = try allocator.alloc(i64, na);
    defer allocator.free(row_sums);
    @memset(row_sums, 0);
    const col_sums = try allocator.alloc(i64, nb);
    defer allocator.free(col_sums);
    @memset(col_sums, 0);
    for (0..na) |i| {
        for (0..nb) |j| {
            row_sums[i] += contingency[i * nb + j];
            col_sums[j] += contingency[i * nb + j];
        }
    }

    var sum_comb_row: f64 = 0.0;
    for (row_sums) |r| sum_comb_row += nChoose2(r);
    var sum_comb_col: f64 = 0.0;
    for (col_sums) |c| sum_comb_col += nChoose2(c);

    const total_comb = nChoose2(@intCast(n));
    const expected_index = sum_comb_row * sum_comb_col / total_comb;
    const max_index = 0.5 * (sum_comb_row + sum_comb_col);

    if (@abs(max_index - expected_index) < 1e-12) {
        // Degenerate case (e.g. every point its own cluster in both, or a
        // single shared cluster): sklearn returns 1.0 here.
        return 1.0;
    }

    return (sum_comb_c - expected_index) / (max_index - expected_index);
}

test "ari sanity checks" {
    const alloc = std_testing.allocator;
    const a = [_]i64{ 0, 0, 0, 1, 1, 1 };
    try std_testing.expectApproxEqAbs(@as(f64, 1.0), try adjustedRandIndex(alloc, &a, &a), 1e-9);

    const b = [_]i64{ 0, 1, 0, 1, 0, 1 }; // effectively random wrt a
    const ari_random = try adjustedRandIndex(alloc, &a, &b);
    try std_testing.expect(ari_random < 1.0);

    // permuting labels doesn't change ARI
    const c = [_]i64{ 5, 5, 5, 9, 9, 9 };
    try std_testing.expectApproxEqAbs(@as(f64, 1.0), try adjustedRandIndex(alloc, &a, &c), 1e-9);
}

test "evoc_clusters e2e ARI against Python evoc.EVoC" {
    const alloc = std_testing.allocator;
    const fx = try fixture.readFixture(alloc, "clustering", "evoc_clusters_e2e_ari_check");
    defer fx.deinit();
    const root = fx.value.object;
    const n_samples: usize = @intCast(root.get("n_samples").?.integer);
    const n_features: usize = @intCast(root.get("n_features").?.integer);

    const m = try readMatrixF32(alloc, root.get("data").?.array);
    defer alloc.free(m.data);
    try std_testing.expectEqual(n_samples * n_features, m.data.len);

    const python_labels_arr = root.get("python_labels").?.array;
    const python_labels = try alloc.alloc(i64, python_labels_arr.items.len);
    defer alloc.free(python_labels);
    for (python_labels_arr.items, 0..) |v, i| python_labels[i] = v.integer;
    try std_testing.expectEqual(n_samples, python_labels.len);

    // evoc.EVoC()'s own defaults, matching the fixture generator script.
    const opts = EvocClustersOptions{
        .noise_level = 0.5,
        .base_min_cluster_size = 5,
        .n_neighbors = 15,
        .min_samples = 5,
        .n_epochs = 50,
        .neighbor_scale = 1.0,
        .min_similarity_threshold = 0.2,
        .max_layers = 10,
        .n_label_prop_iter = 20,
        .node_embedding_dim = null,
    };

    var evoc = EVoC.init(alloc, 29, opts);
    defer evoc.deinit();
    const zig_labels = try evoc.fit_predict(m.data, n_features);

    const ari = try adjustedRandIndex(alloc, python_labels, zig_labels);
    std.debug.print("evoc_clusters e2e ARI against Python evoc.EVoC: {d:.4}\n", .{ari});

    try std_testing.expect(ari > 0.8);
}
