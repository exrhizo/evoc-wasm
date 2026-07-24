//! Port of evoc/cluster_trees.py: MST -> linkage tree -> condensed tree ->
//! excess-of-mass cluster extraction -> point labels & membership strengths.
//! Reuses disjoint_set.zig. See PORTING.md / PORT_NOTES.md.
//!
//! This is the most branch-order-sensitive module in the port: condense_tree's
//! five-way branch and the BFS/relabel bookkeeping must match the reference
//! exactly, not merely produce a logically-equivalent tree.

const std = @import("std");
const Allocator = std.mem.Allocator;
const ds = @import("disjoint_set.zig");

// ---------------------------------------------------------------------------
// Linkage tree (mst_to_linkage_tree)
// ---------------------------------------------------------------------------

/// LinkageMergeData: union-find with size tracking + a running `next` label.
const LinkageMergeData = struct {
    parent: []i64,
    size: []i64,
    next: i64,
};

/// Port of create_linkage_merge_data.
fn create_linkage_merge_data(allocator: Allocator, base_size: usize) !LinkageMergeData {
    const parent = try allocator.alloc(i64, 2 * base_size - 1);
    @memset(parent, -1);
    const size = try allocator.alloc(i64, 2 * base_size - 1);
    for (0..base_size) |i| size[i] = 1;
    for (base_size..size.len) |i| size[i] = 0;
    return LinkageMergeData{ .parent = parent, .size = size, .next = @intCast(base_size) };
}

/// Port of linkage_merge_find (path compression up to the root).
fn linkage_merge_find(lm: *LinkageMergeData, node_in: i64) i64 {
    var node = node_in;
    const relabel_start = node_in;
    while (lm.parent[@intCast(node)] != -1 and lm.parent[@intCast(node)] != node) {
        node = lm.parent[@intCast(node)];
    }
    lm.parent[@intCast(node)] = node;

    // label up to the root
    var relabel = relabel_start;
    while (lm.parent[@intCast(relabel)] != node) {
        const next_relabel = lm.parent[@intCast(relabel)];
        lm.parent[@intCast(relabel)] = node;
        relabel = next_relabel;
    }
    return node;
}

/// Port of linkage_merge_join.
fn linkage_merge_join(lm: *LinkageMergeData, left: i64, right: i64) void {
    const nxt: usize = @intCast(lm.next);
    lm.size[nxt] = lm.size[@intCast(left)] + lm.size[@intCast(right)];
    lm.parent[@intCast(left)] = lm.next;
    lm.parent[@intCast(right)] = lm.next;
    lm.next += 1;
}

/// Linkage tree: n_rows x 4 (left, right, delta, size). Stored as f64 columns
/// (numpy default) — the endpoints and size are integral but carried as f64.
pub const LinkageTree = struct {
    left: []f64,
    right: []f64,
    delta: []f64,
    size: []f64,
    allocator: Allocator,

    pub fn n_rows(self: *const LinkageTree) usize {
        return self.left.len;
    }

    pub fn deinit(self: *LinkageTree) void {
        self.allocator.free(self.left);
        self.allocator.free(self.right);
        self.allocator.free(self.delta);
        self.allocator.free(self.size);
        self.* = undefined;
    }
};

/// Port of mst_to_linkage_tree. `mst_*` are the sorted MST edge columns
/// (length n-1). Result is f64.
///
/// PORT NOTE: result is f64 (numpy default) even though MST distances are f32;
/// row convention puts the larger component id in the left slot
/// (left_component > right_component).
pub fn mst_to_linkage_tree(
    allocator: Allocator,
    mst_left: []const f32,
    mst_right: []const f32,
    mst_dist: []const f32,
) !LinkageTree {
    const m = mst_left.len;
    const left = try allocator.alloc(f64, m);
    errdefer allocator.free(left);
    const right = try allocator.alloc(f64, m);
    errdefer allocator.free(right);
    const delta = try allocator.alloc(f64, m);
    errdefer allocator.free(delta);
    const size = try allocator.alloc(f64, m);
    errdefer allocator.free(size);

    const n_samples = m + 1;
    var lm = try create_linkage_merge_data(allocator, n_samples);
    defer allocator.free(lm.parent);
    defer allocator.free(lm.size);

    for (0..m) |index| {
        const l: i64 = @intFromFloat(mst_left[index]);
        const r: i64 = @intFromFloat(mst_right[index]);
        const d: f64 = mst_dist[index];

        const left_component = linkage_merge_find(&lm, l);
        const right_component = linkage_merge_find(&lm, r);

        if (left_component > right_component) {
            left[index] = @floatFromInt(left_component);
            right[index] = @floatFromInt(right_component);
        } else {
            right[index] = @floatFromInt(left_component);
            left[index] = @floatFromInt(right_component);
        }

        delta[index] = d;
        size[index] = @floatFromInt(lm.size[@intCast(left_component)] + lm.size[@intCast(right_component)]);

        linkage_merge_join(&lm, left_component, right_component);
    }

    return LinkageTree{ .left = left, .right = right, .delta = delta, .size = size, .allocator = allocator };
}

// ---------------------------------------------------------------------------
// Condensed tree (condense_tree)
// ---------------------------------------------------------------------------

/// Port of bfs_from_hierarchy. Appends the discovered node ids to `out`.
/// `hierarchy` is accessed via left/right columns; internal nodes (>=
/// num_points) enqueue their two children.
fn bfs_from_hierarchy(
    allocator: Allocator,
    hier_left: []const f64,
    hier_right: []const f64,
    bfs_root: i64,
    num_points: i64,
    out: *std.ArrayList(i64),
) !void {
    var to_process = std.ArrayList(i64).init(allocator);
    defer to_process.deinit();
    var next_to_process = std.ArrayList(i64).init(allocator);
    defer next_to_process.deinit();

    try to_process.append(bfs_root);

    while (to_process.items.len > 0) {
        try out.appendSlice(to_process.items);
        next_to_process.clearRetainingCapacity();
        for (to_process.items) |n| {
            if (n >= num_points) {
                const i: usize = @intCast(n - num_points);
                try next_to_process.append(@intFromFloat(hier_left[i]));
                try next_to_process.append(@intFromFloat(hier_right[i]));
            }
        }
        // swap to_process <- next_to_process
        std.mem.swap(std.ArrayList(i64), &to_process, &next_to_process);
    }
}

/// CondensedTree fields, struct-of-arrays. parent:i64, child:i64,
/// lambda_val:f32, child_size:i64.
pub const CondensedTree = struct {
    parent: []i64,
    child: []i64,
    lambda_val: []f32,
    child_size: []i64,
    allocator: Allocator,

    pub fn len(self: *const CondensedTree) usize {
        return self.parent.len;
    }

    pub fn deinit(self: *CondensedTree) void {
        self.allocator.free(self.parent);
        self.allocator.free(self.child);
        self.allocator.free(self.lambda_val);
        self.allocator.free(self.child_size);
        self.* = undefined;
    }
};

const EliminateCtx = struct {
    parents: []i64,
    children: []i64,
    lambdas: []f32,
    ignore: []bool,
    hier_left: []const f64,
    hier_right: []const f64,
    num_points: i64,
};

/// Port of eliminate_branch. Flattens `branch_node`'s descendants under
/// `parent_node` at `lambda_value`; marks internal descendants ignore=True.
fn eliminate_branch(
    allocator: Allocator,
    ctx: *EliminateCtx,
    branch_node: i64,
    parent_node: i64,
    lambda_value: f32,
    idx_in: usize,
) !usize {
    var idx = idx_in;
    if (branch_node < ctx.num_points) {
        ctx.parents[idx] = parent_node;
        ctx.children[idx] = branch_node;
        ctx.lambdas[idx] = lambda_value;
        idx += 1;
    } else {
        var sub_nodes = std.ArrayList(i64).init(allocator);
        defer sub_nodes.deinit();
        try bfs_from_hierarchy(allocator, ctx.hier_left, ctx.hier_right, branch_node, ctx.num_points, &sub_nodes);
        for (sub_nodes.items) |sub_node| {
            if (sub_node < ctx.num_points) {
                ctx.children[idx] = sub_node;
                ctx.parents[idx] = parent_node;
                ctx.lambdas[idx] = lambda_value;
                idx += 1;
            } else {
                ctx.ignore[@intCast(sub_node)] = true;
            }
        }
    }
    return idx;
}

/// Port of condense_tree. `hierarchy` is the linkage tree (columns
/// left/right/delta/size, length n-1). Returns the condensed tree.
///
/// PORT NOTE: lambda = 1/d if d>0 else inf. The five-way branch is ordered for
/// the common (left singleton) case first and must be replicated verbatim —
/// the branch chosen affects relabel/ignore state, not just the emitted edge.
pub fn condense_tree(
    allocator: Allocator,
    hier_left: []const f64,
    hier_right: []const f64,
    hier_delta: []const f64,
    hier_size: []const f64,
    min_cluster_size: i64,
) !CondensedTree {
    const n_hier = hier_left.len;
    const root: i64 = @intCast(2 * n_hier);
    const num_points: i64 = @intCast(n_hier + 1);
    var next_label: i64 = num_points + 1;

    var node_list = std.ArrayList(i64).init(allocator);
    defer node_list.deinit();
    try bfs_from_hierarchy(allocator, hier_left, hier_right, root, num_points, &node_list);

    const relabel = try allocator.alloc(i64, @intCast(root + 1));
    defer allocator.free(relabel);
    @memset(relabel, 0);
    relabel[@intCast(root)] = num_points;

    const rsz: usize = @intCast(root);
    const parents = try allocator.alloc(i64, rsz);
    defer allocator.free(parents);
    @memset(parents, 1);
    const children = try allocator.alloc(i64, rsz);
    defer allocator.free(children);
    const lambdas = try allocator.alloc(f32, rsz);
    defer allocator.free(lambdas);
    const sizes = try allocator.alloc(i64, rsz);
    defer allocator.free(sizes);
    @memset(sizes, 1);
    const ignore = try allocator.alloc(bool, @intCast(root + 1));
    defer allocator.free(ignore);
    @memset(ignore, false);

    var ctx = EliminateCtx{
        .parents = parents,
        .children = children,
        .lambdas = lambdas,
        .ignore = ignore,
        .hier_left = hier_left,
        .hier_right = hier_right,
        .num_points = num_points,
    };

    var idx: usize = 0;

    for (node_list.items) |node| {
        if (ignore[@intCast(node)] or node < num_points) {
            continue;
        }

        const parent_node = relabel[@intCast(node)];
        const hidx: usize = @intCast(node - num_points);
        const left: i64 = @intFromFloat(hier_left[hidx]);
        const right: i64 = @intFromFloat(hier_right[hidx]);
        const d = hier_delta[hidx];
        const lambda_value: f32 = if (d > 0.0) @floatCast(1.0 / d) else std.math.inf(f32);

        const left_count: i64 = if (left >= num_points)
            @intFromFloat(hier_size[@intCast(left - num_points)])
        else
            1;
        const right_count: i64 = if (right >= num_points)
            @intFromFloat(hier_size[@intCast(right - num_points)])
        else
            1;

        // Branch order matches the reference exactly.
        if (left < num_points and right_count >= min_cluster_size) {
            relabel[@intCast(right)] = parent_node;
            parents[idx] = parent_node;
            children[idx] = left;
            lambdas[idx] = lambda_value;
            idx += 1;
        } else if (left_count < min_cluster_size and right_count >= min_cluster_size) {
            relabel[@intCast(right)] = parent_node;
            idx = try eliminate_branch(allocator, &ctx, left, parent_node, lambda_value, idx);
        } else if (left_count >= min_cluster_size and right_count < min_cluster_size) {
            relabel[@intCast(left)] = parent_node;
            idx = try eliminate_branch(allocator, &ctx, right, parent_node, lambda_value, idx);
        } else if (left_count < min_cluster_size and right_count < min_cluster_size) {
            idx = try eliminate_branch(allocator, &ctx, left, parent_node, lambda_value, idx);
            idx = try eliminate_branch(allocator, &ctx, right, parent_node, lambda_value, idx);
        } else {
            relabel[@intCast(left)] = next_label;
            parents[idx] = parent_node;
            children[idx] = next_label;
            lambdas[idx] = lambda_value;
            sizes[idx] = left_count;
            next_label += 1;
            idx += 1;

            relabel[@intCast(right)] = next_label;
            parents[idx] = parent_node;
            children[idx] = next_label;
            lambdas[idx] = lambda_value;
            sizes[idx] = right_count;
            next_label += 1;
            idx += 1;
        }
    }

    const out_parent = try allocator.dupe(i64, parents[0..idx]);
    errdefer allocator.free(out_parent);
    const out_child = try allocator.dupe(i64, children[0..idx]);
    errdefer allocator.free(out_child);
    const out_lambda = try allocator.dupe(f32, lambdas[0..idx]);
    errdefer allocator.free(out_lambda);
    const out_size = try allocator.dupe(i64, sizes[0..idx]);
    errdefer allocator.free(out_size);

    return CondensedTree{
        .parent = out_parent,
        .child = out_child,
        .lambda_val = out_lambda,
        .child_size = out_size,
        .allocator = allocator,
    };
}

// ---------------------------------------------------------------------------
// Leaf extraction & scoring
// ---------------------------------------------------------------------------

/// Port of extract_leaves. leaf = node id never appearing as a parent with a
/// child_size>1 entry (and >= n_points). Returns the sorted leaf ids.
///
/// PORT NOTE: n_points = parent.min(), n_nodes = parent.max()+1.
pub fn extract_leaves(allocator: Allocator, tree: *const CondensedTree) ![]i64 {
    if (tree.parent.len == 0) {
        return allocator.alloc(i64, 0);
    }
    var pmin: i64 = tree.parent[0];
    var pmax: i64 = tree.parent[0];
    for (tree.parent) |p| {
        if (p < pmin) pmin = p;
        if (p > pmax) pmax = p;
    }
    const n_nodes: usize = @intCast(pmax + 1);
    const n_points: usize = @intCast(pmin);

    const leaf_indicator = try allocator.alloc(bool, n_nodes);
    defer allocator.free(leaf_indicator);
    @memset(leaf_indicator, true);
    for (0..@min(n_points, n_nodes)) |i| leaf_indicator[i] = false;

    for (tree.parent, tree.child_size) |parent, child_size| {
        if (child_size > 1) {
            leaf_indicator[@intCast(parent)] = false;
        }
    }

    var count: usize = 0;
    for (leaf_indicator) |v| {
        if (v) count += 1;
    }
    const result = try allocator.alloc(i64, count);
    var w: usize = 0;
    for (leaf_indicator, 0..) |v, i| {
        if (v) {
            result[w] = @intCast(i);
            w += 1;
        }
    }
    return result;
}

/// Node id -> score map. Keys are cluster/point node ids (small, dense-ish),
/// so a hashmap keyed by i64 is used.
const ScoreMap = std.AutoHashMap(i64, f64);

/// Port of score_condensed_tree_nodes.
fn score_condensed_tree_nodes(allocator: Allocator, tree: *const CondensedTree) !ScoreMap {
    var result = ScoreMap.init(allocator);
    errdefer result.deinit();

    for (0..tree.parent.len) |i| {
        const parent = tree.parent[i];
        const contrib = @as(f64, tree.lambda_val[i]) * @as(f64, @floatFromInt(tree.child_size[i]));
        if (result.getPtr(parent)) |p| {
            p.* += contrib;
        } else {
            try result.put(parent, contrib);
        }

        if (tree.child_size[i] > 1) {
            const child = tree.child[i];
            if (result.getPtr(child)) |p| {
                p.* -= contrib;
            } else {
                try result.put(child, -contrib);
            }
        }
    }
    return result;
}

/// Port of cluster_tree_from_condensed_tree (mask child_size > 1).
pub fn cluster_tree_from_condensed_tree(allocator: Allocator, tree: *const CondensedTree) !CondensedTree {
    var count: usize = 0;
    for (tree.child_size) |cs| {
        if (cs > 1) count += 1;
    }
    const parent = try allocator.alloc(i64, count);
    errdefer allocator.free(parent);
    const child = try allocator.alloc(i64, count);
    errdefer allocator.free(child);
    const lambda_val = try allocator.alloc(f32, count);
    errdefer allocator.free(lambda_val);
    const child_size = try allocator.alloc(i64, count);
    errdefer allocator.free(child_size);

    var w: usize = 0;
    for (0..tree.parent.len) |i| {
        if (tree.child_size[i] > 1) {
            parent[w] = tree.parent[i];
            child[w] = tree.child[i];
            lambda_val[w] = tree.lambda_val[i];
            child_size[w] = tree.child_size[i];
            w += 1;
        }
    }
    return CondensedTree{ .parent = parent, .child = child, .lambda_val = lambda_val, .child_size = child_size, .allocator = allocator };
}

// ---------------------------------------------------------------------------
// EOM cluster extraction
// ---------------------------------------------------------------------------

const SelectedMap = std.AutoHashMap(i64, bool);

/// Port of unselect_below_node (recursive).
fn unselect_below_node(node: i64, cluster_tree: *const CondensedTree, selected: *SelectedMap) void {
    for (0..cluster_tree.parent.len) |i| {
        if (cluster_tree.parent[i] == node) {
            const child = cluster_tree.child[i];
            unselect_below_node(child, cluster_tree, selected);
            selected.put(child, false) catch {};
        }
    }
}

/// Port of eom_recursion.
fn eom_recursion(
    node: i64,
    cluster_tree: *const CondensedTree,
    node_scores: *ScoreMap,
    selected: *SelectedMap,
) f64 {
    const current_score = node_scores.get(node) orelse 0.0;

    var child_score_total: f64 = 0.0;
    for (0..cluster_tree.parent.len) |i| {
        if (cluster_tree.parent[i] == node) {
            child_score_total += eom_recursion(cluster_tree.child[i], cluster_tree, node_scores, selected);
        }
    }

    if (child_score_total > current_score) {
        return child_score_total;
    } else {
        selected.put(node, true) catch {};
        unselect_below_node(node, cluster_tree, selected);
        return current_score;
    }
}

/// Port of extract_eom_clusters. Returns selected cluster node ids (order is
/// incidental in the reference; callers should compare sorted).
pub fn extract_eom_clusters(
    allocator: Allocator,
    condensed_tree: *const CondensedTree,
    cluster_tree: *const CondensedTree,
    allow_single_cluster: bool,
) ![]i64 {
    var node_scores = try score_condensed_tree_nodes(allocator, condensed_tree);
    defer node_scores.deinit();

    var selected = SelectedMap.init(allocator);
    defer selected.deinit();
    // selected_clusters = {node: False for node in node_scores}
    {
        var it = node_scores.keyIterator();
        while (it.next()) |k| try selected.put(k.*, false);
    }

    if (cluster_tree.parent.len == 0) {
        return allocator.alloc(i64, 0);
    }

    var cluster_tree_root: i64 = cluster_tree.parent[0];
    for (cluster_tree.parent) |p| {
        if (p < cluster_tree_root) cluster_tree_root = p;
    }

    if (allow_single_cluster) {
        _ = eom_recursion(cluster_tree_root, cluster_tree, &node_scores, &selected);
    } else if (node_scores.count() > 1) {
        // for child in cluster_tree.child[cluster_tree.parent == root]
        for (0..cluster_tree.parent.len) |i| {
            if (cluster_tree.parent[i] == cluster_tree_root) {
                _ = eom_recursion(cluster_tree.child[i], cluster_tree, &node_scores, &selected);
            }
        }
    }

    var count: usize = 0;
    {
        var it = selected.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.*) count += 1;
        }
    }
    const result = try allocator.alloc(i64, count);
    var w: usize = 0;
    {
        var it = selected.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.*) {
                result[w] = e.key_ptr.*;
                w += 1;
            }
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Labels & membership strengths
// ---------------------------------------------------------------------------

/// Port of get_single_cluster_label_vector (single-cluster fast path).
///
/// PORT NOTE: this path is NOT exercised by the n80 fixture (which yields 4
/// clusters); it is covered by a dedicated unit test.
pub fn get_single_cluster_label_vector(
    allocator: Allocator,
    tree: *const CondensedTree,
    cluster: i64,
    cluster_selection_epsilon: f32,
    n_samples: usize,
) ![]i64 {
    const result = try allocator.alloc(i64, n_samples);
    @memset(result, -1);
    if (tree.parent.len == 0) {
        return result;
    }

    // max over lambda_val where parent == cluster
    var max_lambda: f32 = -std.math.inf(f32);
    var any = false;
    for (0..tree.parent.len) |i| {
        if (tree.parent[i] == cluster) {
            if (!any or tree.lambda_val[i] > max_lambda) max_lambda = tree.lambda_val[i];
            any = true;
        }
    }

    for (0..tree.child.len) |i| {
        const n = tree.child[i];
        const cur_lambda = tree.lambda_val[i];
        if (cluster_selection_epsilon > 0.0) {
            if (cur_lambda >= 1.0 / cluster_selection_epsilon) {
                result[@intCast(n)] = 0;
            } else {
                result[@intCast(n)] = -1;
            }
        } else if (cur_lambda >= max_lambda) {
            result[@intCast(n)] = 0;
        }
    }
    return result;
}

/// Port of get_cluster_label_vector (multi-cluster path with single-cluster
/// fast-path dispatch). `clusters` are the selected cluster node ids.
pub fn get_cluster_label_vector(
    allocator: Allocator,
    tree: *const CondensedTree,
    clusters: []const i64,
    cluster_selection_epsilon: f32,
    n_samples: usize,
) ![]i64 {
    if (clusters.len == 1) {
        return get_single_cluster_label_vector(allocator, tree, clusters[0], cluster_selection_epsilon, n_samples);
    }

    if (tree.parent.len == 0) {
        const result = try allocator.alloc(i64, n_samples);
        @memset(result, -1);
        return result;
    }

    var root_cluster: i64 = tree.parent[0];
    var pmax: i64 = tree.parent[0];
    for (tree.parent) |p| {
        if (p < root_cluster) root_cluster = p;
        if (p > pmax) pmax = p;
    }
    var cmax: i64 = tree.child[0];
    for (tree.child) |c| {
        if (c > cmax) cmax = c;
    }

    const result = try allocator.alloc(i64, n_samples);
    errdefer allocator.free(result);
    @memset(result, -1);

    // cluster_label_map = {c: n for n, c in enumerate(np.sort(clusters))}
    const sorted_clusters = try allocator.dupe(i64, clusters);
    defer allocator.free(sorted_clusters);
    std.mem.sort(i64, sorted_clusters, {}, std.sort.asc(i64));
    var cluster_label_map = std.AutoHashMap(i64, i64).init(allocator);
    defer cluster_label_map.deinit();
    for (sorted_clusters, 0..) |c, n| try cluster_label_map.put(c, @intCast(n));

    // clusters = set(clusters)
    var cluster_set = std.AutoHashMap(i64, void).init(allocator);
    defer cluster_set.deinit();
    for (clusters) |c| try cluster_set.put(c, {});

    const dsize: usize = @intCast(@max(pmax + 1, cmax + 1));
    var disjoint_set = try ds.ds_rank_create(allocator, dsize);
    defer disjoint_set.deinit();

    for (0..tree.parent.len) |n| {
        const child = tree.child[n];
        const parent = tree.parent[n];
        if (!cluster_set.contains(child)) {
            ds.ds_union_by_rank(&disjoint_set, @intCast(parent), @intCast(child));
        }
    }

    for (0..n_samples) |n| {
        const cluster = ds.ds_find(&disjoint_set, @intCast(n));
        if (cluster <= root_cluster) {
            result[n] = -1;
        } else {
            result[n] = cluster_label_map.get(cluster).?;
        }
    }

    return result;
}

/// Port of max_lambdas. Returns a map cluster -> death lambda.
fn max_lambdas(
    allocator: Allocator,
    tree: *const CondensedTree,
    clusters: *const std.AutoHashMap(i64, void),
) !std.AutoHashMap(i64, f32) {
    var result = std.AutoHashMap(i64, f32).init(allocator);
    errdefer result.deinit();
    var it = clusters.keyIterator();
    while (it.next()) |c| try result.put(c.*, 0.0);

    for (0..tree.parent.len) |n| {
        const cluster = tree.parent[n];
        if (clusters.contains(cluster) and tree.child_size[n] == 1) {
            const cur = result.get(cluster).?;
            try result.put(cluster, @max(cur, tree.lambda_val[n]));
        }
    }
    return result;
}

/// Port of get_point_membership_strength_vector.
pub fn get_point_membership_strength_vector(
    allocator: Allocator,
    tree: *const CondensedTree,
    clusters: []const i64,
    labels: []const i64,
) ![]f32 {
    const result = try allocator.alloc(f32, labels.len);
    errdefer allocator.free(result);
    @memset(result, 0.0);

    var cluster_set = std.AutoHashMap(i64, void).init(allocator);
    defer cluster_set.deinit();
    for (clusters) |c| try cluster_set.put(c, {});

    var deaths = try max_lambdas(allocator, tree, &cluster_set);
    defer deaths.deinit();

    var root_cluster: i64 = tree.parent[0];
    for (tree.parent) |p| {
        if (p < root_cluster) root_cluster = p;
    }

    // cluster_index_map = {n: c for n, c in enumerate(np.sort(clusters))}
    const sorted_clusters = try allocator.dupe(i64, clusters);
    defer allocator.free(sorted_clusters);
    std.mem.sort(i64, sorted_clusters, {}, std.sort.asc(i64));

    for (0..tree.child.len) |n| {
        const point = tree.child[n];
        if (point >= root_cluster or labels[@intCast(point)] < 0) {
            continue;
        }

        const cluster = sorted_clusters[@intCast(labels[@intCast(point)])];
        const max_lambda = deaths.get(cluster).?;
        if (max_lambda == 0.0 or !std.math.isFinite(tree.lambda_val[n])) {
            result[@intCast(point)] = 1.0;
        } else {
            const lambda_val = @min(tree.lambda_val[n], max_lambda);
            result[@intCast(point)] = lambda_val / max_lambda;
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const fixture = @import("fixture.zig");

fn readVecI64(alloc: Allocator, arr: std.json.Array) ![]i64 {
    const out = try alloc.alloc(i64, arr.items.len);
    for (arr.items, 0..) |v, i| out[i] = v.integer;
    return out;
}

fn readVecF32(alloc: Allocator, arr: std.json.Array) ![]f32 {
    const out = try alloc.alloc(f32, arr.items.len);
    for (arr.items, 0..) |v, i| out[i] = @floatCast(fixture.numberAsF64(v));
    return out;
}

fn readCondensedTree(alloc: Allocator, obj: std.json.ObjectMap) !CondensedTree {
    const parent = try readVecI64(alloc, obj.get("parent").?.array);
    errdefer alloc.free(parent);
    const child = try readVecI64(alloc, obj.get("child").?.array);
    errdefer alloc.free(child);
    const lambda_val = try readVecF32(alloc, obj.get("lambda_val").?.array);
    errdefer alloc.free(lambda_val);
    const child_size = try readVecI64(alloc, obj.get("child_size").?.array);
    errdefer alloc.free(child_size);
    return CondensedTree{ .parent = parent, .child = child, .lambda_val = lambda_val, .child_size = child_size, .allocator = alloc };
}

test "tiny linkage: 3 points, two merges, larger-id-first convention" {
    const alloc = std.testing.allocator;
    // MST: edge 0-1 @ 1.0, edge 1-2 @ 2.0. points 0,1,2; internal 3,4.
    const l = [_]f32{ 0, 1 };
    const r = [_]f32{ 1, 2 };
    const dd = [_]f32{ 1.0, 2.0 };
    var tree = try mst_to_linkage_tree(alloc, &l, &r, &dd);
    defer tree.deinit();

    // row 0 merges comps 0 and 1 -> larger (1) in left slot
    try std.testing.expectEqual(@as(f64, 1.0), tree.left[0]);
    try std.testing.expectEqual(@as(f64, 0.0), tree.right[0]);
    try std.testing.expectEqual(@as(f64, 2.0), tree.size[0]);
    // row 1 merges comp(2)=2 with comp(1)=3 (the new node) -> larger 3 left
    try std.testing.expectEqual(@as(f64, 3.0), tree.left[1]);
    try std.testing.expectEqual(@as(f64, 2.0), tree.right[1]);
    try std.testing.expectEqual(@as(f64, 3.0), tree.size[1]);
}

test "extract_leaves: empty tree returns empty" {
    const alloc = std.testing.allocator;
    var tree = CondensedTree{
        .parent = &.{},
        .child = &.{},
        .lambda_val = &.{},
        .child_size = &.{},
        .allocator = alloc,
    };
    const leaves = try extract_leaves(alloc, &tree);
    defer alloc.free(leaves);
    try std.testing.expectEqual(@as(usize, 0), leaves.len);
}

test "get_single_cluster_label_vector: assigns cluster 0 to max-lambda children" {
    const alloc = std.testing.allocator;
    // one cluster node id 3 over 3 points (0,1,2). All children are points.
    var tree = CondensedTree{
        .parent = @constCast(&[_]i64{ 3, 3, 3 }),
        .child = @constCast(&[_]i64{ 0, 1, 2 }),
        .lambda_val = @constCast(&[_]f32{ 1.0, 2.0, 2.0 }),
        .child_size = @constCast(&[_]i64{ 1, 1, 1 }),
        .allocator = alloc,
    };
    const labels = try get_single_cluster_label_vector(alloc, &tree, 3, 0.0, 3);
    defer alloc.free(labels);
    // max_lambda = 2.0; children with lambda >= 2.0 -> label 0, else -1
    try std.testing.expectEqual(@as(i64, -1), labels[0]);
    try std.testing.expectEqual(@as(i64, 0), labels[1]);
    try std.testing.expectEqual(@as(i64, 0), labels[2]);
}

test "get_cluster_label_vector dispatches to single-cluster path for len 1" {
    const alloc = std.testing.allocator;
    var tree = CondensedTree{
        .parent = @constCast(&[_]i64{ 3, 3, 3 }),
        .child = @constCast(&[_]i64{ 0, 1, 2 }),
        .lambda_val = @constCast(&[_]f32{ 2.0, 2.0, 2.0 }),
        .child_size = @constCast(&[_]i64{ 1, 1, 1 }),
        .allocator = alloc,
    };
    const clusters = [_]i64{3};
    const labels = try get_cluster_label_vector(alloc, &tree, &clusters, 0.0, 3);
    defer alloc.free(labels);
    for (labels) |v| try std.testing.expectEqual(@as(i64, 0), v);
}

test "fixture parity: mst_to_linkage_tree_n80" {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "cluster_trees", "mst_to_linkage_tree_n80");
    defer fx.deinit();
    const root = fx.value.object;
    const tol_value = root.get("tolerance").?;
    const mst = root.get("inputs").?.object.get("sorted_mst").?.object;

    const pa = mst.get("point_a").?.array;
    const pb = mst.get("point_b").?.array;
    const dist = mst.get("distance").?.array;
    const m = pa.items.len;
    const mst_left = try alloc.alloc(f32, m);
    defer alloc.free(mst_left);
    const mst_right = try alloc.alloc(f32, m);
    defer alloc.free(mst_right);
    const mst_dist = try alloc.alloc(f32, m);
    defer alloc.free(mst_dist);
    for (0..m) |i| {
        mst_left[i] = @floatFromInt(pa.items[i].integer);
        mst_right[i] = @floatFromInt(pb.items[i].integer);
        mst_dist[i] = @floatCast(fixture.numberAsF64(dist.items[i]));
    }

    var tree = try mst_to_linkage_tree(alloc, mst_left, mst_right, mst_dist);
    defer tree.deinit();

    const expected = root.get("expected").?.object;
    const exp_left = expected.get("left").?.array;
    const exp_right = expected.get("right").?.array;
    const exp_delta = expected.get("delta").?.array;
    const exp_size = expected.get("size").?.array;
    const delta_tol = fixture.toleranceForField(tol_value, "delta");

    for (0..m) |i| {
        try std.testing.expectEqual(@as(i64, exp_left.items[i].integer), @as(i64, @intFromFloat(tree.left[i])));
        try std.testing.expectEqual(@as(i64, exp_right.items[i].integer), @as(i64, @intFromFloat(tree.right[i])));
        try std.testing.expectEqual(@as(i64, exp_size.items[i].integer), @as(i64, @intFromFloat(tree.size[i])));
        // delta tolerance check in f32 (fixture stores f64, but the value is a
        // copied f32 MST distance)
        try fixture.expectF32(delta_tol, @floatCast(tree.delta[i]), fixture.numberAsF64(exp_delta.items[i]));
    }
}

test "fixture parity: condense_tree_n80_mcs5" {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "cluster_trees", "condense_tree_n80_mcs5");
    defer fx.deinit();
    const root = fx.value.object;
    const tol_value = root.get("tolerance").?;
    const inputs = root.get("inputs").?.object;
    const hier = inputs.get("hierarchy").?.object;

    const hl_a = hier.get("left").?.array;
    const hr_a = hier.get("right").?.array;
    const hd_a = hier.get("delta").?.array;
    const hs_a = hier.get("size").?.array;
    const m = hl_a.items.len;
    const hier_left = try alloc.alloc(f64, m);
    defer alloc.free(hier_left);
    const hier_right = try alloc.alloc(f64, m);
    defer alloc.free(hier_right);
    const hier_delta = try alloc.alloc(f64, m);
    defer alloc.free(hier_delta);
    const hier_size = try alloc.alloc(f64, m);
    defer alloc.free(hier_size);
    for (0..m) |i| {
        hier_left[i] = @floatFromInt(hl_a.items[i].integer);
        hier_right[i] = @floatFromInt(hr_a.items[i].integer);
        hier_delta[i] = fixture.numberAsF64(hd_a.items[i]);
        hier_size[i] = @floatFromInt(hs_a.items[i].integer);
    }
    const mcs: i64 = inputs.get("min_cluster_size").?.integer;

    var ctree = try condense_tree(alloc, hier_left, hier_right, hier_delta, hier_size, mcs);
    defer ctree.deinit();

    const expected = root.get("expected").?.object;
    const exp_parent = expected.get("parent").?.array;
    const exp_child = expected.get("child").?.array;
    const exp_lambda = expected.get("lambda_val").?.array;
    const exp_size = expected.get("child_size").?.array;
    const lambda_tol = fixture.toleranceForField(tol_value, "lambda_val");

    try std.testing.expectEqual(exp_parent.items.len, ctree.parent.len);
    for (0..ctree.parent.len) |i| {
        try std.testing.expectEqual(@as(i64, exp_parent.items[i].integer), ctree.parent[i]);
        try std.testing.expectEqual(@as(i64, exp_child.items[i].integer), ctree.child[i]);
        try std.testing.expectEqual(@as(i64, exp_size.items[i].integer), ctree.child_size[i]);
        try fixture.expectF32(lambda_tol, ctree.lambda_val[i], fixture.numberAsF64(exp_lambda.items[i]));
    }
}

test "fixture parity: labels_and_strengths_n80_mcs5" {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "cluster_trees", "labels_and_strengths_n80_mcs5");
    defer fx.deinit();
    const root = fx.value.object;
    const tol_value = root.get("tolerance").?;
    const inputs = root.get("inputs").?.object;

    var ctree = try readCondensedTree(alloc, inputs.get("condensed_tree").?.object);
    defer ctree.deinit();
    const eps: f32 = @floatCast(fixture.numberAsF64(inputs.get("cluster_selection_epsilon").?));
    const n_samples: usize = @intCast(inputs.get("n_samples").?.integer);

    const leaves = try extract_leaves(alloc, &ctree);
    defer alloc.free(leaves);
    const labels = try get_cluster_label_vector(alloc, &ctree, leaves, eps, n_samples);
    defer alloc.free(labels);
    const strengths = try get_point_membership_strength_vector(alloc, &ctree, leaves, labels);
    defer alloc.free(strengths);

    const expected = root.get("expected").?.object;
    const exp_leaves = expected.get("leaves").?.array;
    const exp_labels = expected.get("labels").?.array;
    const exp_strengths = expected.get("strengths").?.array;
    const strength_tol = fixture.toleranceForField(tol_value, "strengths");

    try std.testing.expectEqual(exp_leaves.items.len, leaves.len);
    for (0..leaves.len) |i| {
        try std.testing.expectEqual(@as(i64, exp_leaves.items[i].integer), leaves[i]);
    }
    for (0..n_samples) |i| {
        try std.testing.expectEqual(@as(i64, exp_labels.items[i].integer), labels[i]);
        try fixture.expectF32(strength_tol, strengths[i], fixture.numberAsF64(exp_strengths.items[i]));
    }
}

test "fixture parity: extract_eom_clusters_n80_mcs5 (compare sorted)" {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "cluster_trees", "extract_eom_clusters_n80_mcs5");
    defer fx.deinit();
    const root = fx.value.object;
    const inputs = root.get("inputs").?.object;

    var ctree = try readCondensedTree(alloc, inputs.get("condensed_tree").?.object);
    defer ctree.deinit();
    const allow_single = inputs.get("allow_single_cluster").?.bool;

    var cluster_tree = try cluster_tree_from_condensed_tree(alloc, &ctree);
    defer cluster_tree.deinit();

    // Verify the recomputed cluster_tree matches the fixture's record.
    const expected = root.get("expected").?.object;
    const exp_ct_parent = expected.get("cluster_tree_parent").?.array;
    const exp_ct_child = expected.get("cluster_tree_child").?.array;
    const exp_ct_size = expected.get("cluster_tree_child_size").?.array;
    try std.testing.expectEqual(exp_ct_parent.items.len, cluster_tree.parent.len);
    for (0..cluster_tree.parent.len) |i| {
        try std.testing.expectEqual(@as(i64, exp_ct_parent.items[i].integer), cluster_tree.parent[i]);
        try std.testing.expectEqual(@as(i64, exp_ct_child.items[i].integer), cluster_tree.child[i]);
        try std.testing.expectEqual(@as(i64, exp_ct_size.items[i].integer), cluster_tree.child_size[i]);
    }

    const eom = try extract_eom_clusters(alloc, &ctree, &cluster_tree, allow_single);
    defer alloc.free(eom);
    std.mem.sort(i64, eom, {}, std.sort.asc(i64));

    const exp_eom = expected.get("eom_clusters").?.array;
    try std.testing.expectEqual(exp_eom.items.len, eom.len);
    for (0..eom.len) |i| {
        try std.testing.expectEqual(@as(i64, exp_eom.items[i].integer), eom[i]);
    }
}
