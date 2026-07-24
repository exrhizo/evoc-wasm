//! Port of evoc/node_embedding.py: UMAP-style SGD graph layout.
//! make_epochs_per_sample, rdist, clip, node_embedding_epoch,
//! node_embedding_epoch_repr, node_embedding. See PORTING.md / PORT_NOTES.md.
//!
//! PORT NOTE: negative sampling here does NOT use tau_rand. Targets are pure
//! integer arithmetic on a per-epoch scalar rng_state (numpy-MT drawn, fed in
//! as fixture input). The plain kernel uses ((n+p)*i*rng_state) % n_vertices;
//! the repr kernel node_order[(raw_index*(n+p+1)*rng_state) % n_vertices].
//! Numba promotes these products to 64-bit (verified against the fixtures) —
//! computed here in u64 (non-wrapping), NOT u32.

const std = @import("std");
const Allocator = std.mem.Allocator;
const rng = @import("rng.zig");

/// Port of make_epochs_per_sample.
///
/// PORT NOTE cast order: n_samples = max(n_epochs*(w/w.max()), 1.0) is a
/// float64 array cast to float32 as a whole, THEN divided into float(n_epochs)
/// (the division and its result are f32). Replicate the f32 cast on n_samples
/// before the divide.
pub fn make_epochs_per_sample(allocator: Allocator, weights: []const f32, n_epochs: usize) ![]f32 {
    const result = try allocator.alloc(f32, weights.len);
    errdefer allocator.free(result);

    // weights.max() as float64 (numpy reduction).
    var wmax: f64 = weights[0];
    for (weights) |w| {
        if (w > wmax) wmax = w;
    }

    const ne: f64 = @floatFromInt(n_epochs);
    for (weights, 0..) |w, i| {
        // n_samples computed in f64, then cast to f32 (the whole array), then
        // float(n_epochs) / that f32.
        const ns_f64 = @max(ne * (@as(f64, w) / wmax), 1.0);
        const ns_f32: f32 = @floatCast(ns_f64);
        result[i] = @as(f32, @floatCast(ne)) / ns_f32;
    }
    return result;
}

/// Port of rdist: squared euclidean, f32 accumulation (numba locals pin f32).
pub fn rdist(x: []const f32, y: []const f32) f32 {
    var result: f32 = 0.0;
    const dim = x.len;
    for (0..dim) |i| {
        const diff: f32 = x[i] - y[i];
        result += diff * diff;
    }
    return result;
}

/// Port of clip: `> hi` first, then `< lo`, else val (exact branch order).
pub fn clip(val: f32, lo: f32, hi: f32) f32 {
    if (val > hi) {
        return hi;
    } else if (val < lo) {
        return lo;
    } else {
        return val;
    }
}

/// Port of node_embedding_epoch (non-reproducible: COO edges, in-place
/// per-edge updates, no gamma, no *1.5 inflation on epochs_per_negative_sample).
/// `embedding` is n_vertices x dim row-major.
///
/// PORT NOTE: prange over edges -> sequential. Negative-sample index
/// k = ((n+p)*i*rng_state) % n_vertices computed in u64 (numba promotes to 64-bit).
pub fn node_embedding_epoch(
    embedding: []f32,
    head: []const u32,
    tail: []const u32,
    n_vertices: u32,
    epochs_per_sample: []const f32,
    rng_state: u32,
    dim: usize,
    alpha: f32,
    epochs_per_negative_sample: []const f32,
    epoch_of_next_negative_sample: []f32,
    epoch_of_next_sample: []f32,
    n: u32,
    noise_level: f32,
) void {
    for (0..epochs_per_sample.len) |i| { // prange, sequential here
        if (epoch_of_next_sample[i] <= @as(f32, @floatFromInt(n))) {
            const j = head[i];
            var k = tail[i];

            const current = embedding[@as(usize, j) * dim ..][0..dim];
            var other = embedding[@as(usize, k) * dim ..][0..dim];

            var dist_squared = rdist(current, other);

            if (dist_squared > 0.0) {
                const dist = @sqrt(dist_squared);
                const grad_coeff = (-2.0 * noise_level * dist - 2.0) /
                    (2.0 * dist_squared - 0.5 * dist + 1.0);

                for (0..dim) |di| {
                    const grad_d = grad_coeff * (current[di] - other[di]);
                    current[di] += grad_d * alpha;
                    other[di] += -grad_d * alpha;
                }
            }

            epoch_of_next_sample[i] += epochs_per_sample[i];
            const n_neg_samples: u32 = @intFromFloat(
                (@as(f32, @floatFromInt(n)) - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i],
            );

            var p: u32 = 0;
            while (p < n_neg_samples) : (p += 1) {
                const prod: u64 = (@as(u64, n) + @as(u64, p)) * @as(u64, i) * @as(u64, rng_state);
                k = @intCast(prod % @as(u64, n_vertices));
                other = embedding[@as(usize, k) * dim ..][0..dim];
                dist_squared = rdist(current, other);

                if (dist_squared > 1e-2) {
                    const grad_coeff = 4.0 / ((1.0 + 0.25 * dist_squared) * dist_squared);
                    for (0..dim) |di| {
                        const grad_d = clip(grad_coeff * (current[di] - other[di]), -4, 4);
                        current[di] += grad_d * alpha;
                    }
                }
            }

            epoch_of_next_negative_sample[i] += @as(f32, @floatFromInt(n_neg_samples)) * epochs_per_negative_sample[i];
        }
    }
}

/// Port of node_embedding_epoch_repr (reproducible: CSR, block-deferred
/// updates, gamma scaling, *1.5-inflated negative schedule). `embedding` and
/// `updates` are n_vertices x dim row-major; `node_order` is a permutation.
///
/// PORT NOTE: two passes per block — accumulate every node's positive+negative
/// gradient into `updates` first, then apply `embedding += updates` for the
/// same block. Negative-sample index uses node_order and u64 arithmetic.
pub fn node_embedding_epoch_repr(
    embedding: []f32,
    csr_indptr: []const u32,
    csr_indices: []const u32,
    n_vertices: u32,
    epochs_per_sample: []const f32,
    rng_state: u32,
    dim: usize,
    alpha: f32,
    epochs_per_negative_sample: []const f32,
    epoch_of_next_negative_sample: []f32,
    epoch_of_next_sample: []f32,
    n: u32,
    noise_level: f32,
    gamma: f32,
    updates: []f32,
    node_order: []const u32,
    block_size: u32,
) void {
    var block_start: u32 = 0;
    while (block_start < n_vertices) : (block_start += block_size) {
        const block_end = @min(block_start + block_size, n_vertices);

        // Pass 1: accumulate gradients into `updates` for every node in block.
        var node_idx: u32 = block_start;
        while (node_idx < block_end) : (node_idx += 1) { // prange, sequential
            const from_node = node_order[node_idx];
            const current = embedding[@as(usize, from_node) * dim ..][0..dim];

            for (csr_indptr[from_node]..csr_indptr[from_node + 1]) |raw_index| {
                if (epoch_of_next_sample[raw_index] <= @as(f32, @floatFromInt(n))) {
                    var to_node = csr_indices[raw_index];
                    var other = embedding[@as(usize, to_node) * dim ..][0..dim];

                    var dist_squared = rdist(current, other);

                    if (dist_squared > 0.0) {
                        const dist = @sqrt(dist_squared);
                        const grad_coeff = (-2.0 * noise_level * dist - 2.0) /
                            (2.0 * dist_squared - 0.5 * dist + 1.0);
                        for (0..dim) |d| {
                            const grad_d = grad_coeff * (current[d] - other[d]);
                            updates[@as(usize, from_node) * dim + d] += grad_d * alpha;
                        }
                    }

                    epoch_of_next_sample[raw_index] += epochs_per_sample[raw_index];

                    const n_neg_samples: u32 = @intFromFloat(
                        (@as(f32, @floatFromInt(n)) - epoch_of_next_negative_sample[raw_index]) / epochs_per_negative_sample[raw_index],
                    );

                    var p: u32 = 0;
                    while (p < n_neg_samples) : (p += 1) {
                        const prod: u64 = @as(u64, raw_index) * (@as(u64, n) + @as(u64, p) + 1) * @as(u64, rng_state);
                        to_node = node_order[@intCast(prod % @as(u64, n_vertices))];
                        other = embedding[@as(usize, to_node) * dim ..][0..dim];

                        dist_squared = rdist(current, other);

                        if (dist_squared > 1e-2) {
                            const grad_coeff = gamma * 4.0 / ((1.0 + 0.25 * dist_squared) * dist_squared);
                            if (grad_coeff > 0.0) {
                                for (0..dim) |d| {
                                    const grad_d = clip(grad_coeff * (current[d] - other[d]), -4, 4);
                                    updates[@as(usize, from_node) * dim + d] += grad_d * alpha;
                                }
                            }
                        }
                    }

                    epoch_of_next_negative_sample[raw_index] += @as(f32, @floatFromInt(n_neg_samples)) * epochs_per_negative_sample[raw_index];
                }
            }
        }

        // Pass 2: apply accumulated updates for the same block.
        node_idx = block_start;
        while (node_idx < block_end) : (node_idx += 1) {
            const from_node = node_order[node_idx];
            for (0..dim) |d| {
                embedding[@as(usize, from_node) * dim + d] += updates[@as(usize, from_node) * dim + d];
            }
        }
    }
}

/// Driver inputs for a full node_embedding run. Since numpy-MT is not portable,
/// the caller supplies the recorded per-epoch rng_val and (repr-path) per-epoch
/// node_order permutations verbatim from the fixture.
pub const NodeEmbeddingParams = struct {
    n_components: usize,
    n_epochs: usize,
    initial_alpha: f32,
    negative_sample_rate: f32,
    noise_level: f32,
    reproducible_flag: bool,
    /// rng_val[n] = per-epoch scalar (recorded numpy draw).
    rng_val: []const u32,
    /// node_order_shuffles[n] = the node_order permutation applied AFTER epoch
    /// n's kernel call (repr path only). Length n_epochs, each n_vertices.
    node_order_shuffles: ?[]const []const u32 = null,
};

/// Port of node_embedding (the loop over epochs). `embedding` is the initial
/// embedding (n_vertices x n_components row-major), updated in place and also
/// returned. `graph_*` are CSR arrays; head/tail (COO) are only needed for the
/// non-reproducible path.
///
/// PORT NOTE: `updates` is decayed (`*= (1-alpha)^2 * 0.5`) between epochs, not
/// re-zeroed. alpha for epoch n uses the value computed after epoch n-1 (epoch
/// 0 uses raw initial_alpha). gamma_schedule = linspace(0.5, 1.5, n_epochs).
pub fn node_embedding(
    allocator: Allocator,
    embedding: []f32,
    n_vertices: usize,
    graph_indptr: []const u32,
    graph_indices: []const u32,
    graph_data: []const f32,
    head: ?[]const u32,
    tail: ?[]const u32,
    params: NodeEmbeddingParams,
) !void {
    const n_epochs = params.n_epochs;
    const dim = params.n_components;

    const epochs_per_sample = try make_epochs_per_sample(allocator, graph_data, n_epochs);
    defer allocator.free(epochs_per_sample);

    const epochs_per_negative_sample = try allocator.alloc(f32, epochs_per_sample.len);
    defer allocator.free(epochs_per_negative_sample);
    for (epochs_per_sample, 0..) |e, i| {
        epochs_per_negative_sample[i] = e / params.negative_sample_rate;
        if (params.reproducible_flag) epochs_per_negative_sample[i] *= 1.5;
    }

    const epoch_of_next_negative_sample = try allocator.dupe(f32, epochs_per_negative_sample);
    defer allocator.free(epoch_of_next_negative_sample);
    const epoch_of_next_sample = try allocator.dupe(f32, epochs_per_sample);
    defer allocator.free(epoch_of_next_sample);

    const updates = try allocator.alloc(f32, embedding.len);
    defer allocator.free(updates);
    @memset(updates, 0);

    const node_order = try allocator.alloc(u32, n_vertices);
    defer allocator.free(node_order);
    for (0..n_vertices) |i| node_order[i] = @intCast(i);

    // gamma_schedule = linspace(0.5, 1.5, n_epochs)
    const nvu: u32 = @intCast(n_vertices);
    const block_size: u32 = @max(1024, nvu / 8);

    var alpha: f32 = params.initial_alpha;

    for (0..n_epochs) |n| {
        const gamma: f32 = if (n_epochs == 1)
            0.5
        else
            @floatCast(0.5 + (1.5 - 0.5) * @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(n_epochs - 1)));

        if (!params.reproducible_flag) {
            node_embedding_epoch(
                embedding,
                head.?,
                tail.?,
                nvu,
                epochs_per_sample,
                params.rng_val[n],
                dim,
                alpha,
                epochs_per_negative_sample,
                epoch_of_next_negative_sample,
                epoch_of_next_sample,
                @intCast(n),
                params.noise_level,
            );
        } else {
            node_embedding_epoch_repr(
                embedding,
                graph_indptr,
                graph_indices,
                nvu,
                epochs_per_sample,
                params.rng_val[n],
                dim,
                alpha,
                epochs_per_negative_sample,
                epoch_of_next_negative_sample,
                epoch_of_next_sample,
                @intCast(n),
                params.noise_level,
                gamma,
                updates,
                node_order,
                block_size,
            );
            // updates *= (1 - alpha)^2 * 0.5
            const decay: f32 = (1.0 - alpha) * (1.0 - alpha) * 0.5;
            for (updates) |*u| u.* *= decay;
            // random_state.shuffle(node_order): use the recorded permutation.
            if (params.node_order_shuffles) |shuffles| {
                @memcpy(node_order, shuffles[n]);
            }
        }
        alpha = @floatCast(@as(f64, params.initial_alpha) * (1.0 - @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(n_epochs))));
    }
}

/// Fisher-Yates shuffle driven by `tau_rand_int`, used by the self-driven
/// node-embedding path (see `node_embedding_repr_self_driven`).
fn fisher_yates_shuffle(v: []u32, state: *rng.State) void {
    var i: usize = v.len;
    while (i > 1) {
        i -= 1;
        const j: usize = @intCast(@mod(rng.tau_rand_int(state), @as(i32, @intCast(i + 1))));
        const tmp = v[i];
        v[i] = v[j];
        v[j] = tmp;
    }
}

/// Self-driven variant of `node_embedding` (the reproducible/`_repr` path)
/// for the `evoc_clusters` pipeline (phase 9): rather than consuming a
/// pre-recorded `rng_val` / `node_order_shuffles` (numpy-MT draws, not
/// portable -- see PORT_NOTES.md), this draws everything from this project's
/// own `tau_rand` RNG, advancing `state` in place. `initial_embedding = null`
/// draws `random_state.normal(scale=0.25, ...)` in the Python reference;
/// ported here as `tau_rand`-based standard-normal draws scaled by 0.25.
///
/// PORT NOTE (scope): this is not bit-exact with the Python reference (which
/// only a numpy-MT replay could achieve) -- it is validated via the
/// ARI-based end-to-end check described in PCA_VALIDATION.md, not the
/// fixture-parity mechanism used elsewhere in this port.
pub fn node_embedding_repr_self_driven(
    allocator: Allocator,
    graph_indptr: []const u32,
    graph_indices: []const u32,
    graph_data: []const f32,
    n_vertices: usize,
    n_components: usize,
    n_epochs: usize,
    initial_embedding: ?[]const f32,
    initial_alpha: f32,
    negative_sample_rate: f32,
    noise_level: f32,
    state: *rng.State,
) ![]f32 {
    const pca = @import("pca.zig");

    const embedding = try allocator.alloc(f32, n_vertices * n_components);
    errdefer allocator.free(embedding);
    if (initial_embedding) |e| {
        @memcpy(embedding, e);
    } else {
        for (embedding) |*v| v.* = pca.next_standard_normal(state) * 0.25;
    }

    const rng_val = try allocator.alloc(u32, n_epochs);
    defer allocator.free(rng_val);
    for (rng_val) |*v| {
        const draw = rng.tau_rand_int(state);
        v.* = @intCast(@mod(draw, std.math.maxInt(i32)));
    }

    const node_order = try allocator.alloc(u32, n_vertices);
    defer allocator.free(node_order);
    for (0..n_vertices) |i| node_order[i] = @intCast(i);

    const shuffles = try allocator.alloc([]u32, n_epochs);
    defer {
        for (shuffles) |s| allocator.free(s);
        allocator.free(shuffles);
    }
    for (0..n_epochs) |n| {
        fisher_yates_shuffle(node_order, state);
        shuffles[n] = try allocator.dupe(u32, node_order);
    }
    const shuffles_const = try allocator.alloc([]const u32, n_epochs);
    defer allocator.free(shuffles_const);
    for (shuffles, 0..) |s, i| shuffles_const[i] = s;

    try node_embedding(
        allocator,
        embedding,
        n_vertices,
        graph_indptr,
        graph_indices,
        graph_data,
        null,
        null,
        .{
            .n_components = n_components,
            .n_epochs = n_epochs,
            .initial_alpha = initial_alpha,
            .negative_sample_rate = negative_sample_rate,
            .noise_level = noise_level,
            .reproducible_flag = true,
            .rng_val = rng_val,
            .node_order_shuffles = shuffles_const,
        },
    );

    return embedding;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const fixture = @import("fixture.zig");

fn readVecF32(alloc: Allocator, arr: std.json.Array) ![]f32 {
    const out = try alloc.alloc(f32, arr.items.len);
    for (arr.items, 0..) |v, i| out[i] = @floatCast(fixture.numberAsF64(v));
    return out;
}

fn readVecU32(alloc: Allocator, arr: std.json.Array) ![]u32 {
    const out = try alloc.alloc(u32, arr.items.len);
    for (arr.items, 0..) |v, i| out[i] = @intCast(v.integer);
    return out;
}

fn readMatrixF32Flat(alloc: Allocator, arr: std.json.Array) ![]f32 {
    const rows = arr.items.len;
    const cols = arr.items[0].array.items.len;
    const out = try alloc.alloc(f32, rows * cols);
    errdefer alloc.free(out);
    for (arr.items, 0..) |row, i| {
        for (row.array.items, 0..) |v, j| out[i * cols + j] = @floatCast(fixture.numberAsF64(v));
    }
    return out;
}

test "clip: branch order (>hi, then <lo, else val)" {
    try std.testing.expectEqual(@as(f32, 4.0), clip(5.0, -4.0, 4.0));
    try std.testing.expectEqual(@as(f32, -4.0), clip(-5.0, -4.0, 4.0));
    try std.testing.expectEqual(@as(f32, 1.0), clip(1.0, -4.0, 4.0));
    try std.testing.expectEqual(@as(f32, 4.0), clip(4.0, -4.0, 4.0));
    try std.testing.expectEqual(@as(f32, -4.0), clip(-4.0, -4.0, 4.0));
}

test "rdist: identical vectors -> 0, simple case" {
    const x = [_]f32{ 1, 2, 3 };
    try std.testing.expectEqual(@as(f32, 0.0), rdist(&x, &x));
    const y = [_]f32{ 0, 0, 0 };
    try std.testing.expectEqual(@as(f32, 14.0), rdist(&x, &y));
}

test "make_epochs_per_sample: floor at 1.0 for sub-max weights" {
    const alloc = std.testing.allocator;
    const w = [_]f32{ 1.0, 2.0, 4.0 }; // max 4
    const eps = try make_epochs_per_sample(alloc, &w, 10);
    defer alloc.free(eps);
    // w=4 (max): n_samples = max(10, 1) = 10 -> 10/10 = 1
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), eps[2], 1e-5);
    // w=1: n_samples = max(2.5,1)=2.5 -> 10/2.5 = 4
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), eps[0], 1e-5);
}

test "fixture parity: clip" {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "node_embedding", "clip");
    defer fx.deinit();
    const cases = fx.value.object.get("inputs").?.object.get("cases").?.array;
    const values = fx.value.object.get("expected").?.object.get("values").?.array;
    for (cases.items, 0..) |c, i| {
        const co = c.object;
        const val: f32 = @floatCast(fixture.numberAsF64(co.get("val").?));
        const lo: f32 = @floatCast(fixture.numberAsF64(co.get("lo").?));
        const hi: f32 = @floatCast(fixture.numberAsF64(co.get("hi").?));
        try std.testing.expectEqual(@as(f32, @floatCast(fixture.numberAsF64(values.items[i]))), clip(val, lo, hi));
    }
}

test "fixture parity: rdist" {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "node_embedding", "rdist");
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

test "fixture parity: make_epochs_per_sample (both n_epochs)" {
    for ([_][]const u8{ "make_epochs_per_sample_n_epochs10", "make_epochs_per_sample_n_epochs50" }) |case| {
        const alloc = std.testing.allocator;
        const fx = try fixture.readFixture(alloc, "node_embedding", case);
        defer fx.deinit();
        const root = fx.value.object;
        const tol = fixture.toleranceForField(root.get("tolerance").?, "epochs_per_sample");
        const inputs = root.get("inputs").?.object;
        const weights = try readVecF32(alloc, inputs.get("weights").?.array);
        defer alloc.free(weights);
        const n_epochs: usize = @intCast(inputs.get("n_epochs").?.integer);
        const eps = try make_epochs_per_sample(alloc, weights, n_epochs);
        defer alloc.free(eps);
        const expected = root.get("expected").?.object.get("epochs_per_sample").?.array;
        for (0..eps.len) |i| {
            try fixture.expectF32(tol, eps[i], fixture.numberAsF64(expected.items[i]));
        }
    }
}

test "fixture parity: node_embedding_epoch_repr_single_epoch_n120" {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "node_embedding", "node_embedding_epoch_repr_single_epoch_n120");
    defer fx.deinit();
    const root = fx.value.object;
    const tol = fixture.toleranceForField(root.get("tolerance").?, "embedding_after");
    const inputs = root.get("inputs").?.object;

    const embedding = try readMatrixF32Flat(alloc, inputs.get("embedding_before").?.array);
    defer alloc.free(embedding);
    const csr_indptr = try readVecU32(alloc, inputs.get("csr_indptr").?.array);
    defer alloc.free(csr_indptr);
    const csr_indices = try readVecU32(alloc, inputs.get("csr_indices").?.array);
    defer alloc.free(csr_indices);
    const n_vertices: u32 = @intCast(inputs.get("n_vertices").?.integer);
    const epochs_per_sample = try readVecF32(alloc, inputs.get("epochs_per_sample").?.array);
    defer alloc.free(epochs_per_sample);
    const rng_state: u32 = @intCast(inputs.get("rng_state").?.integer);
    const dim: usize = @intCast(inputs.get("dim").?.integer);
    const alpha: f32 = @floatCast(fixture.numberAsF64(inputs.get("alpha").?));
    const epns = try readVecF32(alloc, inputs.get("epochs_per_negative_sample").?.array);
    defer alloc.free(epns);
    // NOTE: the fixture's "epoch_of_next_negative_sample"/"epoch_of_next_sample"
    // inputs are recorded AFTER the reference call mutated them in place
    // (gen_fixtures.py serializes the same numpy array post-call) -- the
    // documented pre-call state is all-zero (see the fixture's own `note`),
    // so we must construct fresh zero arrays here rather than reading them.
    const eons = try alloc.alloc(f32, epns.len);
    defer alloc.free(eons);
    @memset(eons, 0);
    const eos = try alloc.alloc(f32, epochs_per_sample.len);
    defer alloc.free(eos);
    @memset(eos, 0);
    const n: u32 = @intCast(inputs.get("n").?.integer);
    const noise_level: f32 = @floatCast(fixture.numberAsF64(inputs.get("noise_level").?));
    const gamma: f32 = @floatCast(fixture.numberAsF64(inputs.get("gamma").?));
    const node_order = try readVecU32(alloc, inputs.get("node_order").?.array);
    defer alloc.free(node_order);
    const block_size: u32 = @intCast(inputs.get("block_size").?.integer);

    const updates = try alloc.alloc(f32, embedding.len);
    defer alloc.free(updates);
    @memset(updates, 0);

    node_embedding_epoch_repr(
        embedding,
        csr_indptr,
        csr_indices,
        n_vertices,
        epochs_per_sample,
        rng_state,
        dim,
        alpha,
        epns,
        eons,
        eos,
        n,
        noise_level,
        gamma,
        updates,
        node_order,
        block_size,
    );

    const expected = root.get("expected").?.object;
    const exp_emb = try readMatrixF32Flat(alloc, expected.get("embedding_after").?.array);
    defer alloc.free(exp_emb);
    const exp_upd = try readMatrixF32Flat(alloc, expected.get("updates_after").?.array);
    defer alloc.free(exp_upd);
    const exp_eos = expected.get("epoch_of_next_sample_after").?.array;
    const exp_eons = expected.get("epoch_of_next_negative_sample_after").?.array;
    const eos_tol = fixture.toleranceForField(root.get("tolerance").?, "epoch_of_next_sample_after");

    for (0..embedding.len) |i| {
        try fixture.expectF32(tol, embedding[i], exp_emb[i]);
        try fixture.expectF32(tol, updates[i], exp_upd[i]);
    }
    for (0..eos.len) |i| {
        try fixture.expectF32(eos_tol, eos[i], fixture.numberAsF64(exp_eos.items[i]));
        try fixture.expectF32(eos_tol, eons[i], fixture.numberAsF64(exp_eons.items[i]));
    }
}

test "fixture parity: node_embedding_epoch_single_epoch_n120" {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "node_embedding", "node_embedding_epoch_single_epoch_n120");
    defer fx.deinit();
    const root = fx.value.object;
    const tol = fixture.toleranceForField(root.get("tolerance").?, "embedding_after");
    const inputs = root.get("inputs").?.object;

    const embedding = try readMatrixF32Flat(alloc, inputs.get("embedding_before").?.array);
    defer alloc.free(embedding);
    const head = try readVecU32(alloc, inputs.get("head").?.array);
    defer alloc.free(head);
    const tail = try readVecU32(alloc, inputs.get("tail").?.array);
    defer alloc.free(tail);
    const n_vertices: u32 = @intCast(inputs.get("n_vertices").?.integer);
    const epochs_per_sample = try readVecF32(alloc, inputs.get("epochs_per_sample").?.array);
    defer alloc.free(epochs_per_sample);
    const rng_state: u32 = @intCast(inputs.get("rng_state").?.integer);
    const dim: usize = @intCast(inputs.get("dim").?.integer);
    const alpha: f32 = @floatCast(fixture.numberAsF64(inputs.get("alpha").?));
    const epns = try readVecF32(alloc, inputs.get("epochs_per_negative_sample").?.array);
    defer alloc.free(epns);
    // NOTE: see the fixture parity comment above (repr test) -- these two
    // arrays are recorded post-call in the fixture; the true pre-call state
    // is all-zero (per the fixture's own `note`).
    const eons = try alloc.alloc(f32, epns.len);
    defer alloc.free(eons);
    @memset(eons, 0);
    const eos = try alloc.alloc(f32, epochs_per_sample.len);
    defer alloc.free(eos);
    @memset(eos, 0);
    const n: u32 = @intCast(inputs.get("n").?.integer);
    const noise_level: f32 = @floatCast(fixture.numberAsF64(inputs.get("noise_level").?));

    node_embedding_epoch(
        embedding,
        head,
        tail,
        n_vertices,
        epochs_per_sample,
        rng_state,
        dim,
        alpha,
        epns,
        eons,
        eos,
        n,
        noise_level,
    );

    const exp_emb = try readMatrixF32Flat(alloc, root.get("expected").?.object.get("embedding_after").?.array);
    defer alloc.free(exp_emb);
    for (0..embedding.len) |i| {
        try fixture.expectF32(tol, embedding[i], exp_emb[i]);
    }
}

test "fixture parity: node_embedding_e2e_n120" {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "node_embedding", "node_embedding_e2e_n120");
    defer fx.deinit();
    const root = fx.value.object;
    const tol = fixture.toleranceForField(root.get("tolerance").?, "final_embedding");
    const inputs = root.get("inputs").?.object;

    const indptr = try readVecU32(alloc, inputs.get("indptr").?.array);
    defer alloc.free(indptr);
    const indices = try readVecU32(alloc, inputs.get("indices").?.array);
    defer alloc.free(indices);
    const data = try readVecF32(alloc, inputs.get("data").?.array);
    defer alloc.free(data);
    const n_vertices: usize = @intCast(inputs.get("n_vertices").?.integer);
    const n_components: usize = @intCast(inputs.get("n_components").?.integer);
    const n_epochs: usize = @intCast(inputs.get("n_epochs").?.integer);
    const initial_alpha: f32 = @floatCast(fixture.numberAsF64(inputs.get("initial_alpha").?));
    const negative_sample_rate: f32 = @floatCast(fixture.numberAsF64(inputs.get("negative_sample_rate").?));
    const noise_level: f32 = @floatCast(fixture.numberAsF64(inputs.get("noise_level").?));

    const embedding = try readMatrixF32Flat(alloc, inputs.get("initial_embedding").?.array);
    defer alloc.free(embedding);
    const rng_val = try readVecU32(alloc, inputs.get("rng_val").?.array);
    defer alloc.free(rng_val);

    const shuffles_arr = inputs.get("node_order_shuffles").?.array;
    const shuffles = try alloc.alloc([]u32, shuffles_arr.items.len);
    defer {
        for (shuffles) |s| alloc.free(s);
        alloc.free(shuffles);
    }
    for (shuffles_arr.items, 0..) |row, i| shuffles[i] = try readVecU32(alloc, row.array);
    const shuffles_const = try alloc.alloc([]const u32, shuffles.len);
    defer alloc.free(shuffles_const);
    for (shuffles, 0..) |s, i| shuffles_const[i] = s;

    try node_embedding(
        alloc,
        embedding,
        n_vertices,
        indptr,
        indices,
        data,
        null,
        null,
        .{
            .n_components = n_components,
            .n_epochs = n_epochs,
            .initial_alpha = initial_alpha,
            .negative_sample_rate = negative_sample_rate,
            .noise_level = noise_level,
            .reproducible_flag = true,
            .rng_val = rng_val,
            .node_order_shuffles = shuffles_const,
        },
    );

    const exp = try readMatrixF32Flat(alloc, root.get("expected").?.object.get("final_embedding").?.array);
    defer alloc.free(exp);
    for (0..embedding.len) |i| {
        try fixture.expectF32(tol, embedding[i], exp[i]);
    }
}
