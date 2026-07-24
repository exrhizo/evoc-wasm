//! Port of evoc/label_propagation.py: label_prop_iteration, label_outliers,
//! remap_labels, label_prop_loop, label_propagation_init. See PORTING.md /
//! PORT_NOTES.md.
//!
//! PORT NOTE: label_prop_loop / label_propagation_init draw from numpy's
//! Mersenne Twister (random_state.randint / .shuffle / .normal), which is
//! not portably reproducible. All such draws are recorded verbatim in the
//! fixtures (rng_state, seed_position_draws, node_embedding_init_normal,
//! node_embedding_rng_val, node_embedding_shuffles, ...) and replayed here
//! rather than re-derived from a PRNG. tau_rand/tau_rand_int (the *evoc*
//! internal RNG, distinct from numpy's) IS reproduced bit-for-bit via
//! rng.zig's tau_rand/tau_rand_int, seeded per-row as `rng_state + i`
//! (elementwise on the 3-lane state), a fresh local copy per row -- not an
//! advancing shared state across rows.

const std = @import("std");
const Allocator = std.mem.Allocator;
const rng = @import("rng.zig");
const node_embedding = @import("node_embedding.zig");

fn localState(base: rng.State, i: i64) rng.State {
    return .{ base[0] + i, base[1] + i, base[2] + i };
}

/// Port of label_prop_iteration. `indptr`/`indices` are CSR adjacency
/// arrays (i32, matching the graph_construction CSR convention); `data`
/// is the edge weight per CSR entry; `labels` is int64 (>=0 assigned,
/// -1 unassigned). Returns a freshly allocated copy (`result = labels.copy()`
/// in the reference, mutated only at unassigned rows).
///
/// PORT NOTE: `votes` is a numba typed dict, which preserves *insertion*
/// order for iteration -- replicated here via parallel arrays (vote_labels,
/// vote_vals) scanned linearly (row degree is small, so this is cheap and
/// avoids depending on any particular hash-map iteration order).
pub fn label_prop_iteration(
    allocator: Allocator,
    indptr: []const i32,
    indices: []const i32,
    data: []const f32,
    labels: []const i64,
    rng_state: rng.State,
) ![]i64 {
    const n_rows = indptr.len - 1;
    const result = try allocator.dupe(i64, labels);
    errdefer allocator.free(result);

    var vote_labels = std.ArrayList(i64).init(allocator);
    defer vote_labels.deinit();
    var vote_vals = std.ArrayList(f32).init(allocator);
    defer vote_vals.deinit();

    for (0..n_rows) |i| {
        const current_l = labels[i];
        if (current_l >= 0) continue;

        var local_rng_state = localState(rng_state, @intCast(i));

        vote_labels.clearRetainingCapacity();
        vote_vals.clearRetainingCapacity();

        const row_start: usize = @intCast(indptr[i]);
        const row_end: usize = @intCast(indptr[i + 1]);
        for (row_start..row_end) |k| {
            const j: usize = @intCast(indices[k]);
            const l = labels[j];
            var found = false;
            for (vote_labels.items, 0..) |vl, idx| {
                if (vl == l) {
                    vote_vals.items[idx] += data[k];
                    found = true;
                    break;
                }
            }
            if (!found) {
                try vote_labels.append(l);
                try vote_vals.append(data[k]);
            }
        }

        var max_vote: f32 = 1.0;
        var tie_count: i32 = 1;
        for (vote_labels.items, 0..) |l, idx| {
            if (l == -1) continue;
            const v = vote_vals.items[idx];
            if (v > max_vote) {
                max_vote = v;
                result[i] = l;
                tie_count = 1;
            } else if (v == max_vote) {
                tie_count += 1;
                if (current_l == -1) {
                    result[i] = l;
                } else if (rng.tau_rand(&local_rng_state) < 1.0 / @as(f32, @floatFromInt(tie_count))) {
                    result[i] = l;
                }
            }
        }
    }

    return result;
}

/// Port of label_outliers. Mutates `labels` in place (also returned, matching
/// the reference's `return labels`). `max_label = labels.max()` is computed
/// ONCE before the loop, from the initial (pre-mutation) labels.
pub fn label_outliers(
    allocator: Allocator,
    indptr: []const i32,
    indices: []const i32,
    labels: []i64,
    rng_state: rng.State,
) !void {
    const n_rows = indptr.len - 1;

    var max_label: i64 = labels[0];
    for (labels) |l| {
        if (l > max_label) max_label = l;
    }

    var node_queue = std.ArrayList(i64).init(allocator);
    defer node_queue.deinit();

    for (0..n_rows) |i| {
        const local_rng_state = localState(rng_state, @intCast(i));
        var local = local_rng_state;
        if (labels[i] < 0) {
            node_queue.clearRetainingCapacity();
            try node_queue.append(@intCast(i));
            var unlabelled = true;
            var n_iter: u32 = 0;

            while (unlabelled and n_iter < 64 and node_queue.items.len > 0) {
                n_iter += 1;
                const current_node: usize = @intCast(node_queue.pop());
                const row_start: usize = @intCast(indptr[current_node]);
                const row_end: usize = @intCast(indptr[current_node + 1]);
                for (row_start..row_end) |k| {
                    const j: usize = @intCast(indices[k]);
                    if (labels[j] >= 0) {
                        labels[i] = labels[j];
                        unlabelled = false;
                        break;
                    } else {
                        try node_queue.append(@intCast(j));
                    }
                }
            }

            if (n_iter >= 64 or unlabelled) {
                const r = rng.tau_rand_int(&local);
                labels[i] = @mod(@as(i64, r), max_label + 1);
            }
        }
    }
}

/// Port of remap_labels. Mutates `labels` in place (also returned).
/// Non-negative labels are remapped densely 0..n_unique-1 in sorted order;
/// each -1 entry gets its own fresh successive label (does NOT collapse to
/// a single "noise" bucket), assigned in array-index order.
pub fn remap_labels(allocator: Allocator, labels: []i64) !void {
    // np.unique: sorted, deduped.
    const sorted = try allocator.dupe(i64, labels);
    defer allocator.free(sorted);
    std.mem.sort(i64, sorted, {}, std.sort.asc(i64));

    var unique = std.ArrayList(i64).init(allocator);
    defer unique.deinit();
    for (sorted, 0..) |v, idx| {
        if (idx == 0 or v != sorted[idx - 1]) try unique.append(v);
    }

    var start: usize = 0;
    if (unique.items.len > 0 and unique.items[0] == -1) start = 1;

    var mapping = std.AutoHashMap(i64, i64).init(allocator);
    defer mapping.deinit();
    var next_label: i64 = 0;
    var idx: i64 = 0;
    for (unique.items[start..], 0..) |l, ii| {
        try mapping.put(l, @intCast(ii));
        idx = @intCast(ii);
    }
    next_label = idx + 1;
    if (unique.items[start..].len == 0) next_label = 0;

    for (labels) |*l| {
        if (l.* < 0) {
            l.* = next_label;
            next_label += 1;
        } else {
            l.* = mapping.get(l.*).?;
        }
    }
}

/// Port of label_prop_loop. `labels` starts as all -1 (n_points entries),
/// seeded at `seed_position_draws[i] -> i` for i in 0..approx_n_parts (the
/// recorded numpy draws, replayed verbatim -- see PORT NOTE at file top),
/// then n_iter rounds of label_prop_iteration, one label_outliers call, and
/// remap_labels. Returns a freshly allocated partition array.
pub fn label_prop_loop(
    allocator: Allocator,
    indptr: []const i32,
    indices: []const i32,
    data: []const f32,
    n_points: usize,
    rng_state: rng.State,
    seed_position_draws: []const i64,
    n_iter: usize,
) ![]i64 {
    var labels = try allocator.alloc(i64, n_points);
    errdefer allocator.free(labels);
    @memset(labels, -1);

    for (seed_position_draws, 0..) |pos, i| {
        labels[@intCast(pos)] = @intCast(i);
    }

    for (0..n_iter) |_| {
        const new_labels = try label_prop_iteration(allocator, indptr, indices, data, labels, rng_state);
        allocator.free(labels);
        labels = new_labels;
    }

    try label_outliers(allocator, indptr, indices, labels, rng_state);
    try remap_labels(allocator, labels);
    return labels;
}

const SpmmResult = struct {
    indptr: []usize,
    indices: []usize,
    data: []f64,
};

/// Replicates scipy.sparse's csr_matmat (Gustavson/SMMP algorithm) exactly,
/// including its data-dependent nnz ORDER: each output row is built via a
/// "next" linked-list of first-discovered columns (a LIFO stack), and
/// emitted in REVERSE discovery order, dropping any column whose
/// accumulated sum is exactly 0.0. See the PORT NOTE at the call site in
/// label_propagation_init for why this order must be replicated bit-for-bit
/// (not just the resulting matrix's mathematical content).
fn spmmScipyOrder(
    allocator: Allocator,
    a_indptr: []const usize,
    a_indices: []const usize,
    a_data: []const f64,
    a_nrows: usize,
    b_indptr: []const usize,
    b_indices: []const usize,
    b_data: []const f64,
    b_ncols: usize,
) !SpmmResult {
    const next = try allocator.alloc(i64, b_ncols);
    defer allocator.free(next);
    @memset(next, -1);
    const sums = try allocator.alloc(f64, b_ncols);
    defer allocator.free(sums);
    @memset(sums, 0);

    const out_indptr = try allocator.alloc(usize, a_nrows + 1);
    errdefer allocator.free(out_indptr);
    out_indptr[0] = 0;
    var out_indices = std.ArrayList(usize).init(allocator);
    errdefer out_indices.deinit();
    var out_data = std.ArrayList(f64).init(allocator);
    errdefer out_data.deinit();

    for (0..a_nrows) |i| {
        var head: i64 = -2;
        var length: usize = 0;

        const jj_start = a_indptr[i];
        const jj_end = a_indptr[i + 1];
        for (jj_start..jj_end) |jj| {
            const j = a_indices[jj];
            const v = a_data[jj];
            const kk_start = b_indptr[j];
            const kk_end = b_indptr[j + 1];
            for (kk_start..kk_end) |kk| {
                const k = b_indices[kk];
                sums[k] += v * b_data[kk];
                if (next[k] == -1) {
                    next[k] = head;
                    head = @intCast(k);
                    length += 1;
                }
            }
        }

        for (0..length) |_| {
            const h: usize = @intCast(head);
            if (sums[h] != 0.0) {
                try out_indices.append(h);
                try out_data.append(sums[h]);
            }
            const temp = head;
            head = next[h];
            next[@intCast(temp)] = -1;
            sums[h] = 0;
        }
        out_indptr[i + 1] = out_indices.items.len;
    }

    return .{
        .indptr = out_indptr,
        .indices = try out_indices.toOwnedSlice(),
        .data = try out_data.toOwnedSlice(),
    };
}

/// Binary search a canonical (column-sorted) CSR row for `col`; 0.0 if absent.
fn csrGet(indptr: []const i32, indices: []const i32, data: []const f32, row: usize, col: i32) f32 {
    const start: usize = @intCast(indptr[row]);
    const end: usize = @intCast(indptr[row + 1]);
    var lo = start;
    var hi = end;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (indices[mid] == col) return data[mid];
        if (indices[mid] < col) lo = mid + 1 else hi = mid;
    }
    return 0.0;
}

pub const LabelPropagationInitParams = struct {
    n_label_prop_iter: usize,
    n_embedding_epochs: usize,
    approx_n_parts: usize,
    n_components: usize,
    scaling: f32,
    noise_level: f32,
    lp_rng_state: rng.State,
    lp_seed_position_draws: []const i64,
    /// Initial embedding for the reduced graph's node_embedding call
    /// (k_reduced x n_components row-major), recorded fixture draw
    /// (node_embedding's `initial_embedding=None` default path draws this
    /// from the advancing numpy RandomState -- not portably reproducible).
    node_embedding_init_normal: []const f32,
    node_embedding_rng_val: []const u32,
    /// node_embedding_shuffles[n] = node_order permutation applied after
    /// epoch n (length n_embedding_epochs, each k_reduced long).
    node_embedding_shuffles: []const []const u32,
};

/// Port of label_propagation_init, restricted to the non-recursive,
/// `upscaling="partition_expander"` path exercised by the fixture (see
/// PORT_NOTES.md -- the small-graph PCA/random/spectral/mds base case,
/// `recursive_init=True`, and the `jitter_expander`/plain-`upscaling`
/// branches are NOT ported: they are unused by `evoc_clusters` and/or
/// require sklearn PCA which has no portable equivalent here).
///
/// `graph` (indptr/indices/data) is the input CSR adjacency (i32 indices,
/// f32 weights, canonical/column-sorted per row -- the graph_construction
/// CSR convention). Returns a freshly allocated n_points x n_components
/// row-major f32 embedding.
pub fn label_propagation_init(
    allocator: Allocator,
    indptr: []const i32,
    indices: []const i32,
    data: []const f32,
    n_points: usize,
    params: LabelPropagationInitParams,
) ![]f32 {
    // partition = label_prop_loop(...)
    const partition = try label_prop_loop(
        allocator,
        indptr,
        indices,
        data,
        n_points,
        params.lp_rng_state,
        params.lp_seed_position_draws,
        params.n_label_prop_iter,
    );
    defer allocator.free(partition);

    var k: usize = 0;
    for (partition) |p| {
        const pu: usize = @intCast(p);
        if (pu + 1 > k) k = pu + 1;
    }

    // group counts -> inv_sqrt_count[b] = 1/sqrt(count_b) (normalize(...,
    // axis=0, norm="l2") on the one-hot base_reduction_map).
    const counts = try allocator.alloc(f64, k);
    defer allocator.free(counts);
    @memset(counts, 0);
    for (partition) |p| counts[@intCast(p)] += 1.0;
    const inv_sqrt_count = try allocator.alloc(f64, k);
    defer allocator.free(inv_sqrt_count);
    for (0..k) |b| inv_sqrt_count[b] = 1.0 / @sqrt(counts[b]);

    // reduced_graph = normalized_reduction_map.T @ graph @ base_reduction_map,
    // then clip(., 0, 1).
    //
    // PORT NOTE: scipy sparse @ (csr_matmat) does NOT produce column-sorted
    // rows -- it uses the classic Gustavson/SMMP linked-list algorithm,
    // which discovers columns in a data-dependent order and emits each row
    // in REVERSE first-discovery order (a LIFO stack), dropping any column
    // whose accumulated sum is exactly 0. This nnz ORDER is load-bearing:
    // reduced_graph's CSR position (`raw_index`) directly drives
    // node_embedding's negative-sampling arithmetic, so a differently
    // (e.g. sorted-ascending) ordered CSR would feed the SGD completely
    // different negative samples and diverge far outside tolerance, even
    // though the graph's *mathematical* content (which edges, what weight)
    // is identical. `spmmScipyOrder` below replicates scipy's csr_matmat
    // algorithm exactly to preserve this order.
    //
    // Build the two sparse operands directly (dense k x k accumulation was
    // tried and is WRONG for this reason -- do not simplify back to it):
    //   A = normalized_reduction_map.T: k rows, row a lists every point i
    //       with partition[i]==a (ascending i, matching scipy's csc->csr
    //       conversion order for the `.T` view), value = inv_sqrt_count[a].
    //   graph: the input CSR (n x n).
    //   base_reduction_map: n rows, row j has one entry (col partition[j],
    //       value 1.0).
    const a_indptr = try allocator.alloc(usize, k + 1);
    defer allocator.free(a_indptr);
    const a_indices = try allocator.alloc(usize, n_points);
    defer allocator.free(a_indices);
    const a_data = try allocator.alloc(f64, n_points);
    defer allocator.free(a_data);
    {
        @memset(a_indptr, 0);
        for (partition) |p| a_indptr[@as(usize, @intCast(p)) + 1] += 1;
        for (1..k + 1) |i| a_indptr[i] += a_indptr[i - 1];
        const cursor = try allocator.alloc(usize, k);
        defer allocator.free(cursor);
        @memcpy(cursor, a_indptr[0..k]);
        for (0..n_points) |i| {
            const a: usize = @intCast(partition[i]);
            a_indices[cursor[a]] = i;
            a_data[cursor[a]] = inv_sqrt_count[a];
            cursor[a] += 1;
        }
    }

    const graph_indptr = try allocator.alloc(usize, n_points + 1);
    defer allocator.free(graph_indptr);
    for (0..n_points + 1) |i| graph_indptr[i] = @intCast(indptr[i]);
    const graph_indices = try allocator.alloc(usize, indices.len);
    defer allocator.free(graph_indices);
    for (indices, 0..) |v, i| graph_indices[i] = @intCast(v);
    const graph_data = try allocator.alloc(f64, data.len);
    defer allocator.free(graph_data);
    for (data, 0..) |v, i| graph_data[i] = v;

    const m = try spmmScipyOrder(allocator, a_indptr, a_indices, a_data, k, graph_indptr, graph_indices, graph_data, n_points);
    defer allocator.free(m.indptr);
    defer allocator.free(m.indices);
    defer allocator.free(m.data);

    const c_indptr = try allocator.alloc(usize, n_points + 1);
    defer allocator.free(c_indptr);
    for (0..n_points + 1) |i| c_indptr[i] = i;
    const c_indices = try allocator.alloc(usize, n_points);
    defer allocator.free(c_indices);
    for (partition, 0..) |p, i| c_indices[i] = @intCast(p);
    const c_data = try allocator.alloc(f64, n_points);
    defer allocator.free(c_data);
    @memset(c_data, 1.0);

    const rg = try spmmScipyOrder(allocator, m.indptr, m.indices, m.data, k, c_indptr, c_indices, c_data, k);
    defer allocator.free(rg.indptr);
    defer allocator.free(rg.indices);
    defer allocator.free(rg.data);

    var rg_indptr = try allocator.alloc(u32, k + 1);
    defer allocator.free(rg_indptr);
    for (rg.indptr, 0..) |v, i| rg_indptr[i] = @intCast(v);
    var rg_indices = try allocator.alloc(u32, rg.indices.len);
    defer allocator.free(rg_indices);
    for (rg.indices, 0..) |v, i| rg_indices[i] = @intCast(v);
    var rg_data = try allocator.alloc(f32, rg.data.len);
    defer allocator.free(rg_data);
    for (rg.data, 0..) |v, i| {
        rg_data[i] = @floatCast(if (v > 1.0) 1.0 else if (v < 0.0) 0.0 else v);
    }

    // reduced_layout = node_embedding(reduced_graph, n_components,
    // n_embedding_epochs, initial_embedding=<recorded>, initial_alpha=0.001
    // * n_embedding_epochs, noise_level=noise_level, reproducible_flag=True).
    const reduced_layout = try allocator.dupe(f32, params.node_embedding_init_normal);
    defer allocator.free(reduced_layout);

    const shuffles_const = try allocator.alloc([]const u32, params.node_embedding_shuffles.len);
    defer allocator.free(shuffles_const);
    for (params.node_embedding_shuffles, 0..) |s, i| shuffles_const[i] = s;

    try node_embedding.node_embedding(
        allocator,
        reduced_layout,
        k,
        rg_indptr,
        rg_indices,
        rg_data,
        null,
        null,
        .{
            .n_components = params.n_components,
            .n_epochs = params.n_embedding_epochs,
            .initial_alpha = 0.001 * @as(f32, @floatFromInt(params.n_embedding_epochs)),
            .negative_sample_rate = 1.0,
            .noise_level = params.noise_level,
            .reproducible_flag = true,
            .rng_val = params.node_embedding_rng_val,
            .node_order_shuffles = shuffles_const,
        },
    );

    // upscaling="partition_expander":
    //   data_expander = normalize((graph .* graph^T) @ normalized_reduction_map, norm="l1")
    //   result = (data_expander @ reduced_layout
    //             + normalize(normalized_reduction_map, norm="l1") @ reduced_layout) / 2.0
    //
    // PORT NOTE: normalize(normalized_reduction_map, norm="l1") is a no-op
    // recovery of the plain 0/1 indicator: each row of
    // normalized_reduction_map has exactly one nonzero entry (value
    // 1/sqrt(count_b) > 0), and L1-row-normalizing a single-nonzero-entry
    // row always yields 1.0 in that slot. So that term is simply
    // `reduced_layout[partition[i], :]` (a gather), computed directly below
    // instead of building the sparse matrix.
    const dim = params.n_components;
    const result = try allocator.alloc(f32, n_points * dim);
    errdefer allocator.free(result);

    const dense_row = try allocator.alloc(f64, k);
    defer allocator.free(dense_row);

    for (0..n_points) |i| {
        @memset(dense_row, 0);
        const row_start: usize = @intCast(indptr[i]);
        const row_end: usize = @intCast(indptr[i + 1]);
        for (row_start..row_end) |idx| {
            const j: usize = @intCast(indices[idx]);
            const wij = data[idx];
            const wji = csrGet(indptr, indices, data, j, @intCast(i));
            if (wji != 0.0) {
                const had: f64 = @as(f64, wij) * @as(f64, wji);
                const b: usize = @intCast(partition[j]);
                dense_row[b] += had * inv_sqrt_count[b];
            }
        }
        var l1_norm: f64 = 0;
        for (dense_row) |v| l1_norm += @abs(v);
        if (l1_norm > 0.0) {
            for (dense_row) |*v| v.* /= l1_norm;
        }

        const gather_label: usize = @intCast(partition[i]);
        for (0..dim) |c| {
            var s: f64 = 0;
            for (0..k) |b| s += dense_row[b] * @as(f64, reduced_layout[b * dim + c]);
            const part2: f64 = reduced_layout[gather_label * dim + c];
            result[i * dim + c] = @floatCast((s + part2) / 2.0);
        }
    }

    // result = scaling * (result - result.mean(axis=0)), cast f32.
    const means = try allocator.alloc(f64, dim);
    defer allocator.free(means);
    @memset(means, 0);
    for (0..n_points) |i| {
        for (0..dim) |c| means[c] += @as(f64, result[i * dim + c]);
    }
    for (means) |*mv| mv.* /= @as(f64, @floatFromInt(n_points));

    for (0..n_points) |i| {
        for (0..dim) |c| {
            const v: f64 = @as(f64, result[i * dim + c]) - means[c];
            result[i * dim + c] = @floatCast(@as(f64, params.scaling) * v);
        }
    }

    return result;
}

/// Self-driven, recursive port of `label_propagation_init` for the
/// `evoc_clusters` pipeline (phase 9). Unlike `label_propagation_init` (the
/// fixture-parity, `recursive_init=False` port whose randomness is supplied
/// explicitly and verbatim), this drives every random draw from this
/// project's own `tau_rand` RNG, advancing `state` in place, and recurses
/// per the reference's `recursive_init=True` default: `label_prop_loop` ->
/// build a reduced graph -> recurse (or hit the PCA base case once the
/// reduced graph drops below `base_init_threshold`) -> `node_embedding` on
/// the reduced graph -> `partition_expander` upscaling back to the caller's
/// graph size.
///
/// `approx_n_parts` follows the reference's exact formula
/// (`clip(int(8*sqrt(n)), 256, 16384)` at the top level, shrinking `// 4`
/// each recursion level) -- callers at the top of the pipeline should pass
/// that value in; recursive calls compute it from `approx_n_parts / 4`
/// (matching the reference, which shrinks the CALLER's value, not
/// re-deriving from the reduced graph's own size).
///
/// PORT NOTE (scope): base case below `base_init_threshold` uses randomized
/// PCA (see `pca.zig`) instead of sklearn's `PCA`/`SpectralEmbedding`/`MDS`
/// -- bit-exact parity there is impossible (see PORT_NOTES.md /
/// PCA_VALIDATION.md). This whole function is validated by ARI against a
/// real Python `evoc.EVoC` run, not bit-exact fixtures.
pub const RecursiveInitParams = struct {
    n_label_prop_iter: usize,
    n_embedding_epochs: usize,
    approx_n_parts: usize,
    n_components: usize,
    scaling: f32,
    noise_level: f32,
    recursive_init: bool,
    base_init_threshold: usize,
    /// Original-space rows for this graph's points, row-major
    /// (n_points x n_features), used by the PCA base case. `null` falls back
    /// to standard-normal random init (should not happen along the
    /// `evoc_clusters` call path, which always threads `data` through).
    data_full: ?[]const f32,
    n_features: usize,
};

pub fn label_propagation_init_recursive(
    allocator: Allocator,
    indptr: []const i32,
    indices: []const i32,
    data: []const f32,
    n_points: usize,
    params: RecursiveInitParams,
    state: *rng.State,
) ![]f32 {
    const pca = @import("pca.zig");

    if (n_points < params.base_init_threshold) {
        // PCA base case (randomized SVD, see pca.zig).
        var source: []f32 = undefined;
        var owns_source = false;
        var n_feat: usize = undefined;
        if (params.data_full) |d| {
            source = try allocator.dupe(f32, d);
            owns_source = true;
            n_feat = params.n_features;
        } else {
            source = try allocator.alloc(f32, n_points * params.n_components);
            owns_source = true;
            for (source) |*v| v.* = pca.next_standard_normal(state);
            n_feat = params.n_components;
        }
        defer if (owns_source) allocator.free(source);

        const result = try pca.randomized_pca(allocator, source, n_points, n_feat, params.n_components, state);
        errdefer allocator.free(result);

        // result -= result.mean(); result /= (result.max() - result.min()) / 2.0
        var sum: f64 = 0.0;
        for (result) |v| sum += v;
        const mean = sum / @as(f64, @floatFromInt(result.len));
        var max_v: f32 = -std.math.inf(f32);
        var min_v: f32 = std.math.inf(f32);
        for (result) |v| {
            if (v > max_v) max_v = v;
            if (v < min_v) min_v = v;
        }
        const range = @max((@as(f64, max_v) - @as(f64, min_v)) / 2.0, 1e-12);
        for (result) |*v| v.* = @floatCast((@as(f64, v.*) - mean) / range);
        return result;
    }

    // 1. label-propagation partition, driven by our own RNG.
    const lp_rng_state: rng.State = .{
        @as(i64, rng.tau_rand_int(state)) *% 2,
        @as(i64, rng.tau_rand_int(state)) *% 2,
        @as(i64, rng.tau_rand_int(state)) *% 2,
    };
    const seed_position_draws = try allocator.alloc(i64, params.approx_n_parts);
    defer allocator.free(seed_position_draws);
    for (seed_position_draws) |*v| {
        const draw = rng.tau_rand_int(state);
        v.* = @mod(@as(i64, draw), @as(i64, @intCast(n_points)));
    }

    const partition = try label_prop_loop(
        allocator,
        indptr,
        indices,
        data,
        n_points,
        lp_rng_state,
        seed_position_draws,
        params.n_label_prop_iter,
    );
    defer allocator.free(partition);

    var n_parts: usize = 0;
    for (partition) |p| {
        const pu: usize = @intCast(p);
        if (pu + 1 > n_parts) n_parts = pu + 1;
    }

    const counts = try allocator.alloc(f64, n_parts);
    defer allocator.free(counts);
    @memset(counts, 0);
    for (partition) |p| counts[@intCast(p)] += 1.0;
    const inv_sqrt_count = try allocator.alloc(f64, n_parts);
    defer allocator.free(inv_sqrt_count);
    for (0..n_parts) |b| inv_sqrt_count[b] = 1.0 / @sqrt(counts[b]);
    const inv_count_l1 = try allocator.alloc(f64, n_parts);
    defer allocator.free(inv_count_l1);
    for (0..n_parts) |b| inv_count_l1[b] = 1.0 / counts[b];

    // reduced_graph = normalized_reduction_map.T @ graph @ base_reduction_map,
    // clipped [0,1] (see the non-recursive path above for the SMMP order
    // rationale -- identical construction here).
    const a_indptr = try allocator.alloc(usize, n_parts + 1);
    defer allocator.free(a_indptr);
    const a_indices = try allocator.alloc(usize, n_points);
    defer allocator.free(a_indices);
    const a_data = try allocator.alloc(f64, n_points);
    defer allocator.free(a_data);
    {
        @memset(a_indptr, 0);
        for (partition) |p| a_indptr[@as(usize, @intCast(p)) + 1] += 1;
        for (1..n_parts + 1) |i| a_indptr[i] += a_indptr[i - 1];
        const cursor = try allocator.alloc(usize, n_parts);
        defer allocator.free(cursor);
        @memcpy(cursor, a_indptr[0..n_parts]);
        for (0..n_points) |i| {
            const a: usize = @intCast(partition[i]);
            a_indices[cursor[a]] = i;
            a_data[cursor[a]] = inv_sqrt_count[a];
            cursor[a] += 1;
        }
    }

    const graph_indptr = try allocator.alloc(usize, n_points + 1);
    defer allocator.free(graph_indptr);
    for (0..n_points + 1) |i| graph_indptr[i] = @intCast(indptr[i]);
    const graph_indices = try allocator.alloc(usize, indices.len);
    defer allocator.free(graph_indices);
    for (indices, 0..) |v, i| graph_indices[i] = @intCast(v);
    const graph_data = try allocator.alloc(f64, data.len);
    defer allocator.free(graph_data);
    for (data, 0..) |v, i| graph_data[i] = v;

    const stage1 = try spmmScipyOrder(allocator, a_indptr, a_indices, a_data, n_parts, graph_indptr, graph_indices, graph_data, n_points);
    defer allocator.free(stage1.indptr);
    defer allocator.free(stage1.indices);
    defer allocator.free(stage1.data);

    const br_indptr = try allocator.alloc(usize, n_points + 1);
    defer allocator.free(br_indptr);
    for (0..n_points + 1) |i| br_indptr[i] = i;
    const br_indices = try allocator.alloc(usize, n_points);
    defer allocator.free(br_indices);
    for (partition, 0..) |p, i| br_indices[i] = @intCast(p);
    const br_data = try allocator.alloc(f64, n_points);
    defer allocator.free(br_data);
    @memset(br_data, 1.0);

    const reduced = try spmmScipyOrder(allocator, stage1.indptr, stage1.indices, stage1.data, n_parts, br_indptr, br_indices, br_data, n_parts);
    defer allocator.free(reduced.indptr);
    defer allocator.free(reduced.indices);
    defer allocator.free(reduced.data);

    const reduced_indptr = try allocator.alloc(i32, n_parts + 1);
    defer allocator.free(reduced_indptr);
    for (reduced.indptr, 0..) |v, i| reduced_indptr[i] = @intCast(v);
    const reduced_indices = try allocator.alloc(i32, reduced.indices.len);
    defer allocator.free(reduced_indices);
    for (reduced.indices, 0..) |v, i| reduced_indices[i] = @intCast(v);
    const reduced_data = try allocator.alloc(f32, reduced.data.len);
    defer allocator.free(reduced_data);
    for (reduced.data, 0..) |v, i| {
        reduced_data[i] = @floatCast(if (v > 1.0) 1.0 else if (v < 0.0) 0.0 else v);
    }
    const reduced_indptr_u32 = try allocator.alloc(u32, n_parts + 1);
    defer allocator.free(reduced_indptr_u32);
    for (reduced.indptr, 0..) |v, i| reduced_indptr_u32[i] = @intCast(v);
    const reduced_indices_u32 = try allocator.alloc(u32, reduced.indices.len);
    defer allocator.free(reduced_indices_u32);
    for (reduced.indices, 0..) |v, i| reduced_indices_u32[i] = @intCast(v);

    // reduced_data (original space) for the recursive PCA base case: mean of
    // each partition's points, matching `data_reducer @ data` where
    // `data_reducer = normalize(normalized_reduction_map.T, norm="l1")`
    // (row-normalized partition-indicator matrix, i.e. per-partition mean).
    var reduced_data_full: ?[]f32 = null;
    if (params.data_full) |full| {
        const acc = try allocator.alloc(f64, n_parts * params.n_features);
        defer allocator.free(acc);
        @memset(acc, 0);
        for (0..n_points) |i| {
            const p: usize = @intCast(partition[i]);
            for (0..params.n_features) |f| {
                acc[p * params.n_features + f] += @as(f64, full[i * params.n_features + f]) * inv_count_l1[p];
            }
        }
        const out = try allocator.alloc(f32, acc.len);
        for (acc, 0..) |v, i| out[i] = @floatCast(v);
        reduced_data_full = out;
    }
    defer if (reduced_data_full) |rdf| allocator.free(rdf);

    var reduced_init: ?[]f32 = null;
    if (params.recursive_init) {
        const next_approx_n_parts = @max(1, params.approx_n_parts / 4);
        reduced_init = try label_propagation_init_recursive(
            allocator,
            reduced_indptr,
            reduced_indices,
            reduced_data,
            n_parts,
            .{
                .n_label_prop_iter = params.n_label_prop_iter,
                .n_embedding_epochs = @min(params.n_embedding_epochs, 255),
                .approx_n_parts = next_approx_n_parts,
                .n_components = params.n_components,
                .scaling = params.scaling,
                .noise_level = params.noise_level,
                .recursive_init = true,
                .base_init_threshold = params.base_init_threshold,
                .data_full = reduced_data_full,
                .n_features = params.n_features,
            },
            state,
        );
    }
    defer if (reduced_init) |ri| allocator.free(ri);

    const initial_alpha = 0.001 * @as(f32, @floatFromInt(params.n_embedding_epochs));
    const reduced_layout = try node_embedding.node_embedding_repr_self_driven(
        allocator,
        reduced_indptr_u32,
        reduced_indices_u32,
        reduced_data,
        n_parts,
        params.n_components,
        params.n_embedding_epochs,
        reduced_init,
        initial_alpha,
        1.0,
        params.noise_level,
        state,
    );
    defer allocator.free(reduced_layout);

    // partition_expander upscaling (identical algebra to
    // label_propagation_init above).
    const dim = params.n_components;
    const result = try allocator.alloc(f32, n_points * dim);
    errdefer allocator.free(result);

    const dense_row = try allocator.alloc(f64, n_parts);
    defer allocator.free(dense_row);

    for (0..n_points) |i| {
        @memset(dense_row, 0);
        const row_start: usize = @intCast(indptr[i]);
        const row_end: usize = @intCast(indptr[i + 1]);
        for (row_start..row_end) |idx| {
            const j: usize = @intCast(indices[idx]);
            const wij = data[idx];
            const wji = csrGet(indptr, indices, data, j, @intCast(i));
            if (wji != 0.0) {
                const had: f64 = @as(f64, wij) * @as(f64, wji);
                const b: usize = @intCast(partition[j]);
                dense_row[b] += had * inv_sqrt_count[b];
            }
        }
        var row_sum: f64 = 0;
        for (dense_row) |v| row_sum += v;
        if (row_sum != 0.0) {
            for (dense_row) |*v| v.* /= row_sum;
        }

        const gather_label: usize = @intCast(partition[i]);
        for (0..dim) |c| {
            var s: f64 = 0;
            for (0..n_parts) |b| s += dense_row[b] * @as(f64, reduced_layout[b * dim + c]);
            const direct: f64 = reduced_layout[gather_label * dim + c];
            result[i * dim + c] = @floatCast((s + direct) / 2.0);
        }
    }

    // result = scaling * (result - result.mean(axis=0)), cast f32.
    const means = try allocator.alloc(f64, dim);
    defer allocator.free(means);
    @memset(means, 0);
    for (0..n_points) |i| {
        for (0..dim) |c| means[c] += @as(f64, result[i * dim + c]);
    }
    for (means) |*mv| mv.* /= @as(f64, @floatFromInt(n_points));

    for (0..n_points) |i| {
        for (0..dim) |c| {
            const v: f64 = @as(f64, result[i * dim + c]) - means[c];
            result[i * dim + c] = @floatCast(@as(f64, params.scaling) * v);
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const fixture = @import("fixture.zig");

fn readVecI32(alloc: Allocator, arr: std.json.Array) ![]i32 {
    const out = try alloc.alloc(i32, arr.items.len);
    for (arr.items, 0..) |v, i| out[i] = @intCast(v.integer);
    return out;
}

fn readVecU32(alloc: Allocator, arr: std.json.Array) ![]u32 {
    const out = try alloc.alloc(u32, arr.items.len);
    for (arr.items, 0..) |v, i| out[i] = @intCast(v.integer);
    return out;
}

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

fn readMatrixF32Flat(alloc: Allocator, arr: std.json.Array) ![]f32 {
    const rows = arr.items.len;
    const cols = arr.items[0].array.items.len;
    const out = try alloc.alloc(f32, rows * cols);
    for (arr.items, 0..) |row, i| {
        for (row.array.items, 0..) |v, j| out[i * cols + j] = @floatCast(fixture.numberAsF64(v));
    }
    return out;
}

fn readState(arr: std.json.Array) rng.State {
    return .{ arr.items[0].integer, arr.items[1].integer, arr.items[2].integer };
}

test "fixture parity: label_prop_iteration_n120" {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "label_propagation", "label_prop_iteration_n120");
    defer fx.deinit();
    const root = fx.value.object;
    const inputs = root.get("inputs").?.object;

    const indptr = try readVecI32(alloc, inputs.get("indptr").?.array);
    defer alloc.free(indptr);
    const indices = try readVecI32(alloc, inputs.get("indices").?.array);
    defer alloc.free(indices);
    const data = try readVecF32(alloc, inputs.get("data").?.array);
    defer alloc.free(data);
    const labels = try readVecI64(alloc, inputs.get("labels").?.array);
    defer alloc.free(labels);
    const rng_state = readState(inputs.get("rng_state").?.array);

    const result = try label_prop_iteration(alloc, indptr, indices, data, labels, rng_state);
    defer alloc.free(result);

    const expected = root.get("expected").?.object.get("labels").?.array;
    for (result, 0..) |v, i| {
        try std.testing.expectEqual(@as(i64, expected.items[i].integer), v);
    }
}

test "fixture parity: label_outliers_n120" {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "label_propagation", "label_outliers_n120");
    defer fx.deinit();
    const root = fx.value.object;
    const inputs = root.get("inputs").?.object;

    const indptr = try readVecI32(alloc, inputs.get("indptr").?.array);
    defer alloc.free(indptr);
    const indices = try readVecI32(alloc, inputs.get("indices").?.array);
    defer alloc.free(indices);
    const labels = try readVecI64(alloc, inputs.get("labels").?.array);
    defer alloc.free(labels);
    const rng_state = readState(inputs.get("rng_state").?.array);

    try label_outliers(alloc, indptr, indices, labels, rng_state);

    const expected = root.get("expected").?.object.get("labels").?.array;
    for (labels, 0..) |v, i| {
        try std.testing.expectEqual(@as(i64, expected.items[i].integer), v);
    }
}

test "fixture parity: label_outliers_disconnected" {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "label_propagation", "label_outliers_disconnected");
    defer fx.deinit();
    const root = fx.value.object;
    const inputs = root.get("inputs").?.object;

    const indptr = try readVecI32(alloc, inputs.get("indptr").?.array);
    defer alloc.free(indptr);
    const indices = try readVecI32(alloc, inputs.get("indices").?.array);
    defer alloc.free(indices);
    const labels = try readVecI64(alloc, inputs.get("labels").?.array);
    defer alloc.free(labels);
    const rng_state = readState(inputs.get("rng_state").?.array);

    try label_outliers(alloc, indptr, indices, labels, rng_state);

    const expected = root.get("expected").?.object.get("labels").?.array;
    for (labels, 0..) |v, i| {
        try std.testing.expectEqual(@as(i64, expected.items[i].integer), v);
    }
}

test "fixture parity: remap_labels" {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "label_propagation", "remap_labels");
    defer fx.deinit();
    const root = fx.value.object;
    const inputs = root.get("inputs").?.object;

    const labels = try readVecI64(alloc, inputs.get("labels").?.array);
    defer alloc.free(labels);

    try remap_labels(alloc, labels);

    const expected = root.get("expected").?.object.get("labels").?.array;
    for (labels, 0..) |v, i| {
        try std.testing.expectEqual(@as(i64, expected.items[i].integer), v);
    }
}

test "fixture parity: label_prop_loop_n120" {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "label_propagation", "label_prop_loop_n120");
    defer fx.deinit();
    const root = fx.value.object;
    const inputs = root.get("inputs").?.object;

    const indptr = try readVecI32(alloc, inputs.get("indptr").?.array);
    defer alloc.free(indptr);
    const indices = try readVecI32(alloc, inputs.get("indices").?.array);
    defer alloc.free(indices);
    const data = try readVecF32(alloc, inputs.get("data").?.array);
    defer alloc.free(data);
    const n_points: usize = @intCast(inputs.get("n_points").?.integer);
    const n_iter: usize = @intCast(inputs.get("n_iter").?.integer);
    const rng_state = readState(inputs.get("rng_state").?.array);
    const seed_position_draws = try readVecI64(alloc, inputs.get("seed_position_draws").?.array);
    defer alloc.free(seed_position_draws);

    const partition = try label_prop_loop(alloc, indptr, indices, data, n_points, rng_state, seed_position_draws, n_iter);
    defer alloc.free(partition);

    const expected = root.get("expected").?.object.get("partition").?.array;
    for (partition, 0..) |v, i| {
        try std.testing.expectEqual(@as(i64, expected.items[i].integer), v);
    }
}

test "fixture parity: label_propagation_init_n120" {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "label_propagation", "label_propagation_init_n120");
    defer fx.deinit();
    const root = fx.value.object;
    const tol = fixture.toleranceForField(root.get("tolerance").?, "embedding");
    const inputs = root.get("inputs").?.object;

    const indptr = try readVecI32(alloc, inputs.get("indptr").?.array);
    defer alloc.free(indptr);
    const indices = try readVecI32(alloc, inputs.get("indices").?.array);
    defer alloc.free(indices);
    const data = try readVecF32(alloc, inputs.get("data").?.array);
    defer alloc.free(data);
    const n_points: usize = @intCast(inputs.get("n_points").?.integer);
    const n_label_prop_iter: usize = @intCast(inputs.get("n_label_prop_iter").?.integer);
    const n_embedding_epochs: usize = @intCast(inputs.get("n_embedding_epochs").?.integer);
    const approx_n_parts: usize = @intCast(inputs.get("approx_n_parts").?.integer);
    const n_components: usize = @intCast(inputs.get("n_components").?.integer);
    const scaling: f32 = @floatCast(fixture.numberAsF64(inputs.get("scaling").?));
    const noise_level: f32 = @floatCast(fixture.numberAsF64(inputs.get("noise_level").?));
    const lp_rng_state = readState(inputs.get("lp_rng_state").?.array);
    const lp_seed_position_draws = try readVecI64(alloc, inputs.get("lp_seed_position_draws").?.array);
    defer alloc.free(lp_seed_position_draws);
    const node_embedding_init_normal = try readMatrixF32Flat(alloc, inputs.get("node_embedding_init_normal").?.array);
    defer alloc.free(node_embedding_init_normal);
    const node_embedding_rng_val = try readVecU32(alloc, inputs.get("node_embedding_rng_val").?.array);
    defer alloc.free(node_embedding_rng_val);

    const shuffles_arr = inputs.get("node_embedding_shuffles").?.array;
    const shuffles = try alloc.alloc([]u32, shuffles_arr.items.len);
    defer {
        for (shuffles) |s| alloc.free(s);
        alloc.free(shuffles);
    }
    for (shuffles_arr.items, 0..) |row, i| shuffles[i] = try readVecU32(alloc, row.array);
    const shuffles_const = try alloc.alloc([]const u32, shuffles.len);
    defer alloc.free(shuffles_const);
    for (shuffles, 0..) |s, i| shuffles_const[i] = s;

    _ = approx_n_parts; // encoded via lp_seed_position_draws.len already

    const result = try label_propagation_init(alloc, indptr, indices, data, n_points, .{
        .n_label_prop_iter = n_label_prop_iter,
        .n_embedding_epochs = n_embedding_epochs,
        .approx_n_parts = lp_seed_position_draws.len,
        .n_components = n_components,
        .scaling = scaling,
        .noise_level = noise_level,
        .lp_rng_state = lp_rng_state,
        .lp_seed_position_draws = lp_seed_position_draws,
        .node_embedding_init_normal = node_embedding_init_normal,
        .node_embedding_rng_val = node_embedding_rng_val,
        .node_embedding_shuffles = shuffles_const,
    });
    defer alloc.free(result);

    const expected = try readMatrixF32Flat(alloc, root.get("expected").?.object.get("embedding").?.array);
    defer alloc.free(expected);

    for (result, 0..) |v, i| {
        try fixture.expectF32(tol, v, expected[i]);
    }
}
