//! Port of evoc/float_nndescent.py (fast_cosine lives in distance.zig).
//! Keep the reference function names so the codebases diff side by side.
//! See PORTING.md.
//!
//! PORT NOTE: numba 2-D matrices are flat row-major slices with explicit
//! widths (n_features for data, max_candidates for candidate blocks,
//! leaf_width for leaf arrays). All numba prange loops become sequential
//! loops over the same ranges — the n_threads block/bucket structure is kept
//! intact (it determines iteration order and therefore heap contents).

const std = @import("std");
const Allocator = std.mem.Allocator;
const rng = @import("rng.zig");
const nnd = @import("nnd_heap.zig");
const distance = @import("distance.zig");
const fast_cosine = distance.fast_cosine;

// Used for a floating point "nearly zero" comparison
const EPS: f64 = 1e-8;
// PORT NOTE: the reference's INF is np.finfo(np.float32).max (largest finite
// f32), NOT infinity; used only in the delta_improv termination check.
const INF: f32 = std.math.floatMax(f32);

/// PORT NOTE: reference's ENABLE_NESTED_PARALLELISM comes from
/// nested_parallelism.py platform sniffing (true on Linux, where the
/// fixtures were generated). Both forest variants are behaviorally identical
/// single-threaded, so the flag only selects a name.
const ENABLE_NESTED_PARALLELISM = true;

/// PORT NOTE: dist_thresholds in the reference is the strided numpy view
/// current_graph[1][:, 0] — a LIVE view of each heap row's worst distance.
/// Represented as the base priorities slice plus row stride.
pub const DistThresholds = struct {
    priorities: []const f32,
    stride: usize,

    inline fn get(self: DistThresholds, p: i32) f32 {
        return self.priorities[@as(usize, @intCast(p)) * self.stride];
    }
};

inline fn dataRow(data: []const f32, n_features: usize, i: usize) []const f32 {
    return data[i * n_features ..][0..n_features];
}

pub const Split = struct {
    indices_left: []i32,
    indices_right: []i32,
};

/// Port of float_random_projection_split. Draws two random points via
/// tau_rand_int (mutating rng_state), splits on the hyperplane between
/// them; near-zero margins are resolved by coin flips drawn from a COPY of
/// rng_state with the point's position added to all three lanes.
///
/// PORT NOTE: numba's int32 % follows Python semantics (result takes the
/// divisor's sign) — @mod here, never @rem.
pub fn float_random_projection_split(
    allocator: Allocator,
    data: []const f32,
    n_features: usize,
    indices: []const i32,
    rng_state: *rng.State,
) !Split {
    const dim = n_features;

    // Select two random points, set the hyperplane between them
    const indices_size: i32 = @intCast(indices.len);
    const left_index: i32 = @mod(rng.tau_rand_int(rng_state), indices_size);
    var right_index: i32 = @mod(rng.tau_rand_int(rng_state), indices_size);
    right_index += @intFromBool(left_index == right_index);
    right_index = @mod(right_index, indices_size);
    const left = indices[@intCast(left_index)];
    const right = indices[@intCast(right_index)];
    const left_data = dataRow(data, n_features, @intCast(left));
    const right_data = dataRow(data, n_features, @intCast(right));

    // Compute the normal vector to the hyperplane (the vector between
    // the two points)
    const hyperplane_vector = try allocator.alloc(f32, dim);
    defer allocator.free(hyperplane_vector);
    var hyperplane_norm: f32 = 0.0;

    for (0..dim) |d| {
        hyperplane_vector[d] = left_data[d] - right_data[d];
        hyperplane_norm += hyperplane_vector[d] * hyperplane_vector[d];
    }

    hyperplane_norm = @sqrt(hyperplane_norm);
    if (@abs(@as(f64, hyperplane_norm)) < EPS) {
        hyperplane_norm = 1.0;
    }

    // Normalize in the same vector (avoiding second loop when possible)
    for (0..dim) |d| {
        hyperplane_vector[d] /= hyperplane_norm;
    }

    // Use temporary arrays sized for worst case, then trim
    const max_size = indices.len;
    const temp_left = try allocator.alloc(i32, max_size);
    defer allocator.free(temp_left);
    const temp_right = try allocator.alloc(i32, max_size);
    defer allocator.free(temp_right);
    var n_left: usize = 0;
    var n_right: usize = 0;

    // Single pass: classify points and directly populate result arrays
    for (0..indices.len) |idx| {
        var local_rng_state = rng.State{
            rng_state[0] +% @as(i64, @intCast(idx)),
            rng_state[1] +% @as(i64, @intCast(idx)),
            rng_state[2] +% @as(i64, @intCast(idx)),
        };
        const point_idx = indices[idx];
        const test_data = dataRow(data, n_features, @intCast(point_idx));
        var margin: f32 = 0.0;

        // Compute margin (dot product with hyperplane normal)
        for (0..dim) |d| {
            margin += hyperplane_vector[d] * test_data[d];
        }

        // Classify point and directly assign to appropriate array
        var classification: i32 = undefined;
        if (@abs(@as(f64, margin)) < EPS) {
            classification = @mod(rng.tau_rand_int(&local_rng_state), 2);
        } else {
            classification = if (margin > 0) 0 else 1;
        }

        if (classification == 0) {
            temp_left[n_left] = point_idx;
            n_left += 1;
        } else {
            temp_right[n_right] = point_idx;
            n_right += 1;
        }
    }

    // Handle degenerate case where all points end up on one side
    if (n_left == 0 or n_right == 0) {
        n_left = 0;
        n_right = 0;
        // Reassign randomly (this path consumes the SHARED rng_state)
        for (0..indices.len) |idx| {
            const point_idx = indices[idx];
            const classification = @mod(rng.tau_rand_int(rng_state), 2);
            if (classification == 0) {
                temp_left[n_left] = point_idx;
                n_left += 1;
            } else {
                temp_right[n_right] = point_idx;
                n_right += 1;
            }
        }
    }

    // Create final arrays with exact sizes (copy only what we need)
    const indices_left = try allocator.alloc(i32, n_left);
    errdefer allocator.free(indices_left);
    const indices_right = try allocator.alloc(i32, n_right);
    errdefer allocator.free(indices_right);

    for (0..n_left) |i| {
        indices_left[i] = temp_left[i];
    }
    for (0..n_right) |j| {
        indices_right[j] = temp_right[j];
    }

    return Split{ .indices_left = indices_left, .indices_right = indices_right };
}

/// Port of make_float_tree. Takes ownership of `indices` (Python's GC frees
/// the parent index array after a split; here it is freed explicitly).
/// Leaf index arrays are appended to `point_indices`, which owns them.
pub fn make_float_tree(
    allocator: Allocator,
    data: []const f32,
    n_features: usize,
    indices: []i32,
    point_indices: *std.ArrayList([]i32),
    rng_state: *rng.State,
    leaf_size: i64,
    max_depth: i64,
) !void {
    if (@as(i64, @intCast(indices.len)) > leaf_size and max_depth > 0) {
        const split = float_random_projection_split(allocator, data, n_features, indices, rng_state) catch |err| {
            allocator.free(indices);
            return err;
        };
        allocator.free(indices);

        make_float_tree(
            allocator,
            data,
            n_features,
            split.indices_left,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        ) catch |err| {
            allocator.free(split.indices_right);
            return err;
        };

        try make_float_tree(
            allocator,
            data,
            n_features,
            split.indices_right,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        );
    } else {
        point_indices.append(indices) catch |err| {
            allocator.free(indices);
            return err;
        };
    }
}

/// A leaf array: n_leaves x width row-major i32 matrix, rows padded with -1.
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

/// Port of make_float_leaf_array_serial.
pub fn make_float_leaf_array_serial(
    allocator: Allocator,
    data: []const f32,
    n_features: usize,
    rng_state: *rng.State,
    leaf_size: i64,
    max_depth: i64,
) !LeafArray {
    const n_samples = data.len / n_features;
    const indices = try allocator.alloc(i32, n_samples);
    for (indices, 0..) |*v, i| v.* = @intCast(i);

    var point_indices = std.ArrayList([]i32).init(allocator);
    defer {
        for (point_indices.items) |leaf| allocator.free(leaf);
        point_indices.deinit();
    }

    try make_float_tree(allocator, data, n_features, indices, &point_indices, rng_state, leaf_size, max_depth);

    const n_leaves = point_indices.items.len;

    var max_leaf_size: usize = @intCast(leaf_size);
    for (point_indices.items) |points| {
        max_leaf_size = @max(max_leaf_size, points.len);
    }

    const result = try allocator.alloc(i32, n_leaves * max_leaf_size);
    errdefer allocator.free(result);
    @memset(result, -1);
    for (point_indices.items, 0..) |points, i| {
        @memcpy(result[i * max_leaf_size ..][0..points.len], points);
    }

    return LeafArray{ .data = result, .n_leaves = n_leaves, .width = max_leaf_size, .allocator = allocator };
}

/// Port of make_float_leaf_array_parallel.
/// PORT NOTE: identical to the serial variant — the reference differs only
/// in prange vs range, which are the same on the single-threaded target.
pub fn make_float_leaf_array_parallel(
    allocator: Allocator,
    data: []const f32,
    n_features: usize,
    rng_state: *rng.State,
    leaf_size: i64,
    max_depth: i64,
) !LeafArray {
    return make_float_leaf_array_serial(allocator, data, n_features, rng_state, leaf_size, max_depth);
}

/// One leaf array per tree (per rng_states row).
pub const Forest = struct {
    arrays: []LeafArray,
    allocator: Allocator,

    pub fn deinit(self: *Forest) void {
        for (self.arrays) |*la| la.deinit();
        self.allocator.free(self.arrays);
        self.* = undefined;
    }
};

/// Port of make_float_forest_no_nested_parallelism.
pub fn make_float_forest_no_nested_parallelism(
    allocator: Allocator,
    data: []const f32,
    n_features: usize,
    rng_states: []rng.State,
    leaf_size: i64,
    max_depth: i64,
) !Forest {
    const arrays = try allocator.alloc(LeafArray, rng_states.len);
    var n_done: usize = 0;
    errdefer {
        for (arrays[0..n_done]) |*la| la.deinit();
        allocator.free(arrays);
    }
    for (0..rng_states.len) |i| { // prange, sequential here
        arrays[i] = try make_float_leaf_array_serial(allocator, data, n_features, &rng_states[i], leaf_size, max_depth);
        n_done += 1;
    }
    return Forest{ .arrays = arrays, .allocator = allocator };
}

/// Port of make_float_forest_with_nested_parallelism.
pub fn make_float_forest_with_nested_parallelism(
    allocator: Allocator,
    data: []const f32,
    n_features: usize,
    rng_states: []rng.State,
    leaf_size: i64,
    max_depth: i64,
) !Forest {
    const arrays = try allocator.alloc(LeafArray, rng_states.len);
    var n_done: usize = 0;
    errdefer {
        for (arrays[0..n_done]) |*la| la.deinit();
        allocator.free(arrays);
    }
    for (0..rng_states.len) |i| { // prange, sequential here
        arrays[i] = try make_float_leaf_array_parallel(allocator, data, n_features, &rng_states[i], leaf_size, max_depth);
        n_done += 1;
    }
    return Forest{ .arrays = arrays, .allocator = allocator };
}

/// Port of make_float_forest (dispatches on ENABLE_NESTED_PARALLELISM).
pub fn make_float_forest(
    allocator: Allocator,
    data: []const f32,
    n_features: usize,
    rng_states: []rng.State,
    leaf_size: i64,
    max_depth: i64,
) !Forest {
    if (ENABLE_NESTED_PARALLELISM) {
        return make_float_forest_with_nested_parallelism(allocator, data, n_features, rng_states, leaf_size, max_depth);
    } else {
        return make_float_forest_no_nested_parallelism(allocator, data, n_features, rng_states, leaf_size, max_depth);
    }
}

/// Port of generate_leaf_updates_float. `updates` is the flat
/// (n_threads, max_updates, 3) buffer; `leaf_block` is block_size x
/// leaf_width row-major. Note the reference emits a (p, p, -1.0)
/// self-update for every leaf point before its pair updates.
pub fn generate_leaf_updates_float(
    updates: []f32,
    max_updates: usize,
    n_updates_per_thread: []i32,
    leaf_block: []const i32,
    leaf_width: usize,
    dist_thresholds: DistThresholds,
    data: []const f32,
    n_features: usize,
    n_threads: usize,
) void {
    const block_size = leaf_block.len / leaf_width;
    const rows_per_thread = (block_size / n_threads) + 1;

    for (0..n_threads) |t| { // prange, sequential here
        var idx: usize = 0;
        for (0..rows_per_thread) |r| {
            const n = t * rows_per_thread + r;
            if (n >= block_size) break;

            for (0..leaf_width) |i| {
                const p = leaf_block[n * leaf_width + i];
                if (p < 0) break;
                const data_p = dataRow(data, n_features, @intCast(p));
                updates[(t * max_updates + idx) * 3 + 0] = @floatFromInt(p);
                updates[(t * max_updates + idx) * 3 + 1] = @floatFromInt(p);
                updates[(t * max_updates + idx) * 3 + 2] = -1.0;
                idx += 1;

                // Start from i+1 to skip self-comparison
                for (i + 1..leaf_width) |j| {
                    const q = leaf_block[n * leaf_width + j];
                    if (q < 0) break;

                    const d = fast_cosine(data_p, dataRow(data, n_features, @intCast(q)));
                    // Use max for better branch prediction than OR condition
                    const max_threshold = @max(dist_thresholds.get(p), dist_thresholds.get(q));
                    if (d < max_threshold) {
                        updates[(t * max_updates + idx) * 3 + 0] = @floatFromInt(p);
                        updates[(t * max_updates + idx) * 3 + 1] = @floatFromInt(q);
                        updates[(t * max_updates + idx) * 3 + 2] = d;
                        idx += 1;
                    }
                }
            }
        }

        n_updates_per_thread[t] = @intCast(idx);
    }
}

/// Port of init_rp_tree_float. `leaf_array` is n_leaves x leaf_width
/// row-major, rows padded with -1.
pub fn init_rp_tree_float(
    allocator: Allocator,
    data: []const f32,
    n_features: usize,
    current_graph: *nnd.Heap,
    leaf_array: []const i32,
    leaf_width: usize,
    n_threads: usize,
) !void {
    const n_leaves = leaf_array.len / leaf_width;
    const block_size = n_threads * 64;
    const n_blocks = n_leaves / block_size;

    const max_leaf_size = leaf_width;
    // int(block_size * M * (M-1) / (2 * n_threads)) + 1, computed in f64
    // (Python float division) then truncated.
    const updates_per_thread: usize = @as(usize, @intFromFloat(
        @as(f64, @floatFromInt(block_size * max_leaf_size * (max_leaf_size - 1))) /
            @as(f64, @floatFromInt(2 * n_threads)),
    )) + 1;
    const updates = try allocator.alloc(f32, n_threads * updates_per_thread * 3);
    defer allocator.free(updates);
    @memset(updates, 0); // np.zeros in the reference
    const n_updates_per_thread = try allocator.alloc(i32, n_threads);
    defer allocator.free(n_updates_per_thread);
    @memset(n_updates_per_thread, 0);
    const n_vertices = current_graph.n_points;
    const vertex_block_size = n_vertices / n_threads + 1;

    for (0..n_blocks + 1) |i| {
        const leaf_block_start = i * block_size;
        const leaf_block_end = @min(n_leaves, (i + 1) * block_size);

        const leaf_block = leaf_array[leaf_block_start * leaf_width .. leaf_block_end * leaf_width];
        const dist_thresholds = DistThresholds{
            .priorities = current_graph.distances,
            .stride = current_graph.size,
        };

        generate_leaf_updates_float(
            updates,
            updates_per_thread,
            n_updates_per_thread,
            leaf_block,
            leaf_width,
            dist_thresholds,
            data,
            n_features,
            n_threads,
        );

        for (0..n_threads) |t| { // prange, sequential here
            const block_start = t * vertex_block_size;
            const block_end = @min(block_start + vertex_block_size, n_vertices);

            for (0..n_threads) |j| {
                for (0..@as(usize, @intCast(n_updates_per_thread[j]))) |k| {
                    const base = (j * updates_per_thread + k) * 3;
                    const p: i32 = @intFromFloat(updates[base + 0]);

                    if (p == -1) continue;

                    const q: i32 = @intFromFloat(updates[base + 1]);
                    const d: f32 = updates[base + 2];

                    const up: usize = @intCast(p);
                    const uq: usize = @intCast(q);

                    if (up >= block_start and up < block_end) {
                        _ = nnd.flagged_heap_push(
                            current_graph.distancesRow(up),
                            current_graph.indicesRow(up),
                            current_graph.flagsRow(up),
                            d,
                            q,
                        );
                    }
                    if (uq >= block_start and uq < block_end) {
                        _ = nnd.flagged_heap_push(
                            current_graph.distancesRow(uq),
                            current_graph.indicesRow(uq),
                            current_graph.flagsRow(uq),
                            d,
                            p,
                        );
                    }
                }
            }
        }
    }
}

/// Port of init_random_float. Fills under-populated heap rows with random
/// neighbors. Each row uses a COPY of rng_state with the row index added to
/// all three lanes — the caller's rng_state is never mutated.
pub fn init_random_float(
    n_neighbors: usize,
    data: []const f32,
    n_features: usize,
    heap: *nnd.Heap,
    rng_state: *const rng.State,
) void {
    const n_samples = data.len / n_features;
    for (0..n_samples) |i| { // prange, sequential here
        var local_rng_state = rng.State{
            rng_state[0] +% @as(i64, @intCast(i)),
            rng_state[1] +% @as(i64, @intCast(i)),
            rng_state[2] +% @as(i64, @intCast(i)),
        };
        const idx_row = heap.indicesRow(i);
        if (idx_row[0] < 0) {
            var n_present: usize = 0;
            for (idx_row) |v| {
                if (v >= 0) n_present += 1;
            }
            for (0..n_neighbors - n_present) |_| {
                const r = rng.tau_rand_int(&local_rng_state);
                // PORT NOTE: np.abs on int32 wraps at INT32_MIN (stays
                // negative); the subsequent Python-style % still yields a
                // non-negative index. -% reproduces the wrap; @mod the sign.
                const abs32: i32 = if (r < 0) -%r else r;
                const idx: i64 = @mod(@as(i64, abs32), @as(i64, @intCast(n_samples)));

                var already_present = false;
                for (idx_row) |v| {
                    if (v == idx) {
                        already_present = true;
                        break;
                    }
                }
                if (already_present) continue;

                const d = fast_cosine(
                    dataRow(data, n_features, @intCast(idx)),
                    dataRow(data, n_features, i),
                );
                _ = nnd.flagged_heap_push(
                    heap.distancesRow(i),
                    heap.indicesRow(i),
                    heap.flagsRow(i),
                    d,
                    @intCast(idx),
                );
            }
        }
    }
}

/// Port of generate_graph_update_array_float_basic. Candidate blocks are
/// block_rows x max_candidates row-major i32 (padded with -1).
pub fn generate_graph_update_array_float_basic(
    update_array: []f32,
    max_updates: usize,
    n_updates_per_thread: []i32,
    new_candidate_block: []const i32,
    old_candidate_block: []const i32,
    max_candidates: usize,
    dist_thresholds: DistThresholds,
    data: []const f32,
    n_features: usize,
    n_threads: usize,
) void {
    const block_size = new_candidate_block.len / max_candidates;
    const max_new_candidates = max_candidates;
    const max_old_candidates = max_candidates;
    const rows_per_thread = (block_size / n_threads) + 1;

    for (0..n_threads) |t| { // prange, sequential here
        var idx: usize = 0;

        for (0..rows_per_thread) |r| {
            const i = t * rows_per_thread + r;
            if (i >= block_size or idx >= max_updates) break;

            for (0..max_new_candidates) |j| {
                if (idx >= max_updates) break;

                const p = new_candidate_block[i * max_candidates + j];
                if (p < 0) continue;
                const data_p = dataRow(data, n_features, @intCast(p));
                const dist_thresh_p = dist_thresholds.get(p);

                for (j + 1..max_new_candidates) |k| {
                    if (idx >= max_updates) break;

                    const q = new_candidate_block[i * max_candidates + k];
                    if (q < 0) continue;

                    // Compute distance once
                    const d = fast_cosine(data_p, dataRow(data, n_features, @intCast(q)));

                    // Use max for better branch prediction than OR condition
                    const dist_thresh_q = dist_thresholds.get(q);
                    const max_threshold = @max(dist_thresh_p, dist_thresh_q);
                    const threshold_check = d <= max_threshold;

                    if (threshold_check) {
                        update_array[(t * max_updates + idx) * 3 + 0] = @floatFromInt(p);
                        update_array[(t * max_updates + idx) * 3 + 1] = @floatFromInt(q);
                        update_array[(t * max_updates + idx) * 3 + 2] = d;
                        idx += 1;
                    }
                }

                for (0..max_old_candidates) |k| {
                    if (idx >= max_updates) break;

                    const q = old_candidate_block[i * max_candidates + k];
                    if (q < 0) continue;

                    const d = fast_cosine(data_p, dataRow(data, n_features, @intCast(q)));
                    const dist_thresh_q = dist_thresholds.get(q);
                    const max_threshold = @max(dist_thresh_p, dist_thresh_q);
                    const threshold_check = d <= max_threshold;

                    if (threshold_check) {
                        update_array[(t * max_updates + idx) * 3 + 0] = @floatFromInt(p);
                        update_array[(t * max_updates + idx) * 3 + 1] = @floatFromInt(q);
                        update_array[(t * max_updates + idx) * 3 + 2] = d;
                        idx += 1;
                    }
                }
            }
        }

        n_updates_per_thread[t] = @intCast(idx);
    }
}

/// Port of generate_graph_update_array_float (working-set version, used by
/// nn_descent_float). The working-set chunking changes only memory access
/// order, not the (i, p, q) enumeration order, so it emits updates in the
/// same sequence as the basic version.
pub fn generate_graph_update_array_float(
    update_array: []f32,
    max_updates: usize,
    n_updates_per_thread: []i32,
    new_candidate_block: []const i32,
    old_candidate_block: []const i32,
    max_candidates: usize,
    dist_thresholds: DistThresholds,
    data: []const f32,
    n_features: usize,
    n_threads: usize,
) void {
    const block_size = new_candidate_block.len / max_candidates;
    const max_new_candidates = max_candidates;
    const max_old_candidates = max_candidates;
    const rows_per_thread = (block_size / n_threads) + 1;

    // Working set size - process this many candidates at a time
    const working_set_size = 8;

    for (0..n_threads) |t| { // prange, sequential here
        var idx: usize = 0;

        for (0..rows_per_thread) |r| {
            const i = t * rows_per_thread + r;
            if (i >= block_size or idx >= max_updates) break;

            // Process new candidates in working set chunks
            var new_start: usize = 0;
            while (new_start < max_new_candidates and idx < max_updates) {
                const new_end = @min(new_start + working_set_size, max_new_candidates);

                // Process pairs within this working set
                for (new_start..new_end) |j| {
                    if (idx >= max_updates) break;

                    const p = new_candidate_block[i * max_candidates + j];
                    if (p < 0) continue;

                    const data_p = dataRow(data, n_features, @intCast(p));
                    const dist_thresh_p = dist_thresholds.get(p);

                    // Compare with other candidates in the same working set
                    for (j + 1..new_end) |k| {
                        if (idx >= max_updates) break;

                        const q = new_candidate_block[i * max_candidates + k];
                        if (q < 0) continue;

                        const d = fast_cosine(data_p, dataRow(data, n_features, @intCast(q)));
                        const dist_thresh_q = dist_thresholds.get(q);
                        const max_threshold = @max(dist_thresh_p, dist_thresh_q);
                        const threshold_check = d <= max_threshold;

                        if (threshold_check) {
                            update_array[(t * max_updates + idx) * 3 + 0] = @floatFromInt(p);
                            update_array[(t * max_updates + idx) * 3 + 1] = @floatFromInt(q);
                            update_array[(t * max_updates + idx) * 3 + 2] = d;
                            idx += 1;
                        }
                    }

                    // Compare with candidates in future working sets
                    for (new_end..max_new_candidates) |k| {
                        if (idx >= max_updates) break;

                        const q = new_candidate_block[i * max_candidates + k];
                        if (q < 0) continue;

                        const d = fast_cosine(data_p, dataRow(data, n_features, @intCast(q)));
                        const dist_thresh_q = dist_thresholds.get(q);
                        const max_threshold = @max(dist_thresh_p, dist_thresh_q);
                        const threshold_check = d <= max_threshold;

                        if (threshold_check) {
                            update_array[(t * max_updates + idx) * 3 + 0] = @floatFromInt(p);
                            update_array[(t * max_updates + idx) * 3 + 1] = @floatFromInt(q);
                            update_array[(t * max_updates + idx) * 3 + 2] = d;
                            idx += 1;
                        }
                    }

                    // Compare with old candidates in working set chunks
                    var old_start: usize = 0;
                    while (old_start < max_old_candidates and idx < max_updates) {
                        const old_end = @min(old_start + working_set_size, max_old_candidates);

                        for (old_start..old_end) |k| {
                            if (idx >= max_updates) break;

                            const q = old_candidate_block[i * max_candidates + k];
                            if (q < 0) continue;

                            const d = fast_cosine(data_p, dataRow(data, n_features, @intCast(q)));
                            const dist_thresh_q = dist_thresholds.get(q);
                            const max_threshold = @max(dist_thresh_p, dist_thresh_q);
                            const threshold_check = d <= max_threshold;

                            if (threshold_check) {
                                update_array[(t * max_updates + idx) * 3 + 0] = @floatFromInt(p);
                                update_array[(t * max_updates + idx) * 3 + 1] = @floatFromInt(q);
                                update_array[(t * max_updates + idx) * 3 + 2] = d;
                                idx += 1;
                            }
                        }

                        old_start = old_end;
                    }
                }

                new_start = new_end;
            }
        }

        n_updates_per_thread[t] = @intCast(idx);
    }
}

/// Port of generate_sorted_graph_update_array_float. Updates are bucketed by
/// target vertex block. `n_updates_per_block` is flat (n_threads,
/// n_threads + 1); `n_vertices` is data's row count.
pub fn generate_sorted_graph_update_array_float(
    allocator: Allocator,
    update_array: []f32,
    max_updates: usize,
    n_updates_per_block: []i32,
    new_candidate_block: []const i32,
    old_candidate_block: []const i32,
    max_candidates: usize,
    dist_thresholds: DistThresholds,
    data: []const f32,
    n_features: usize,
    n_threads: usize,
) !void {
    const block_size_candidates = new_candidate_block.len / max_candidates;
    const max_new_candidates = max_candidates;
    const max_old_candidates = max_candidates;
    const rows_per_thread = (block_size_candidates / n_threads) + 1;

    const n_vertices = data.len / n_features;
    const vertex_block_size = n_vertices / n_threads + 1;
    const max_updates_per_src_thread = max_updates / n_threads;

    // Reset update counts
    for (0..n_threads) |b| { // prange, sequential here
        for (0..n_threads + 1) |t| {
            n_updates_per_block[b * (n_threads + 1) + t] = 0;
        }
    }

    // Each thread generates updates and places them in appropriate buckets
    for (0..n_threads) |t| { // prange, sequential here
        // Thread-local counters for each bucket
        const local_counts = try allocator.alloc(i32, n_threads);
        defer allocator.free(local_counts);
        @memset(local_counts, 0);

        for (0..rows_per_thread) |r| {
            const i = t * rows_per_thread + r;
            if (i >= block_size_candidates) break;

            for (0..max_new_candidates) |j| {
                const p = new_candidate_block[i * max_candidates + j];
                if (p < 0) continue;

                const data_p = dataRow(data, n_features, @intCast(p));
                const dist_thresh_p = dist_thresholds.get(p);
                var p_block: usize = @as(usize, @intCast(p)) / vertex_block_size;
                if (p_block >= n_threads) {
                    p_block = n_threads - 1;
                }

                // Compare with other new candidates
                for (j + 1..max_new_candidates) |k| {
                    const q = new_candidate_block[i * max_candidates + k];
                    if (q < 0) continue;

                    const d = fast_cosine(data_p, dataRow(data, n_features, @intCast(q)));
                    const dist_thresh_q = dist_thresholds.get(q);
                    const max_threshold = @max(dist_thresh_p, dist_thresh_q);

                    if (d <= max_threshold) {
                        var q_block: usize = @as(usize, @intCast(q)) / vertex_block_size;
                        if (q_block >= n_threads) {
                            q_block = n_threads - 1;
                        }

                        // Place update in p's bucket
                        {
                            const bucket_idx: usize = @intCast(local_counts[p_block]);
                            const write_idx = t * max_updates_per_src_thread + bucket_idx;
                            if (write_idx < max_updates) {
                                update_array[(p_block * max_updates + write_idx) * 3 + 0] = @floatFromInt(p);
                                update_array[(p_block * max_updates + write_idx) * 3 + 1] = @floatFromInt(q);
                                update_array[(p_block * max_updates + write_idx) * 3 + 2] = d;
                                local_counts[p_block] += 1;
                            }
                        }

                        // If q is in a different block, also place in q's bucket
                        if (q_block != p_block) {
                            const bucket_idx: usize = @intCast(local_counts[q_block]);
                            const write_idx = t * max_updates_per_src_thread + bucket_idx;
                            if (write_idx < max_updates) {
                                update_array[(q_block * max_updates + write_idx) * 3 + 0] = @floatFromInt(p);
                                update_array[(q_block * max_updates + write_idx) * 3 + 1] = @floatFromInt(q);
                                update_array[(q_block * max_updates + write_idx) * 3 + 2] = d;
                                local_counts[q_block] += 1;
                            }
                        }
                    }
                }

                // Compare with old candidates
                for (0..max_old_candidates) |k| {
                    const q = old_candidate_block[i * max_candidates + k];
                    if (q < 0) continue;

                    const d = fast_cosine(data_p, dataRow(data, n_features, @intCast(q)));
                    const dist_thresh_q = dist_thresholds.get(q);
                    const max_threshold = @max(dist_thresh_p, dist_thresh_q);

                    if (d <= max_threshold) {
                        var q_block: usize = @as(usize, @intCast(q)) / vertex_block_size;
                        if (q_block >= n_threads) {
                            q_block = n_threads - 1;
                        }

                        // Place update in p's bucket
                        {
                            const bucket_idx: usize = @intCast(local_counts[p_block]);
                            const write_idx = t * max_updates_per_src_thread + bucket_idx;
                            if (write_idx < max_updates) {
                                update_array[(p_block * max_updates + write_idx) * 3 + 0] = @floatFromInt(p);
                                update_array[(p_block * max_updates + write_idx) * 3 + 1] = @floatFromInt(q);
                                update_array[(p_block * max_updates + write_idx) * 3 + 2] = d;
                                local_counts[p_block] += 1;
                            }
                        }

                        // If q is in a different block, also place in q's bucket
                        if (q_block != p_block) {
                            const bucket_idx: usize = @intCast(local_counts[q_block]);
                            const write_idx = t * max_updates_per_src_thread + bucket_idx;
                            if (write_idx < max_updates) {
                                update_array[(q_block * max_updates + write_idx) * 3 + 0] = @floatFromInt(p);
                                update_array[(q_block * max_updates + write_idx) * 3 + 1] = @floatFromInt(q);
                                update_array[(q_block * max_updates + write_idx) * 3 + 2] = d;
                                local_counts[q_block] += 1;
                            }
                        }
                    }
                }
            }
        }

        // Record total updates generated by this thread for each bucket
        for (0..n_threads) |b| {
            n_updates_per_block[b * (n_threads + 1) + t + 1] = local_counts[b];
        }
    }
}

/// int((mc**2 + mc*(mc-1)/2) * block_size) — Python float division then
/// truncation, computed in f64 exactly as written.
fn max_updates_per_thread_count(max_candidates: usize, block_size: usize) usize {
    const mc: f64 = @floatFromInt(max_candidates);
    return @intFromFloat((mc * mc + mc * (mc - 1.0) / 2.0) * @as(f64, @floatFromInt(block_size)));
}

fn nn_descent_termination_sum(distances: []const f32) f64 {
    // PORT NOTE: the reference driver uses np.sum on the masked f32 array
    // (numpy pairwise summation, f32 accumulator). Sequential f32
    // accumulation here; only used for the optional delta_improv early exit
    // (disabled in all fixtures), where the tiny rounding difference cannot
    // flip the comparison in practice.
    var sum: f32 = 0.0;
    for (distances) |d| {
        if (d < INF) sum += d;
    }
    return @floatCast(sum);
}

/// Port of nn_descent_float. Returns the deheap-sorted graph (read
/// .indices / .distances; distances are raw negative cosine values).
pub fn nn_descent_float(
    allocator: Allocator,
    data: []const f32,
    n_features: usize,
    n_neighbors: usize,
    rng_state: *rng.State,
    max_candidates: usize,
    n_iters: usize,
    delta: f64,
    delta_improv: ?f64,
    leaf_array: []const i32,
    leaf_width: usize,
    n_threads: usize,
) !nnd.Heap {
    const n_samples = data.len / n_features;
    var current_graph = try nnd.make_heap(allocator, n_samples, n_neighbors);
    errdefer current_graph.deinit();
    try init_rp_tree_float(allocator, data, n_features, &current_graph, leaf_array, leaf_width, n_threads);
    init_random_float(n_neighbors, data, n_features, &current_graph, rng_state);

    var n_vertices = n_samples;
    var block_size = 65536 / n_threads;
    var n_blocks = n_vertices / block_size;

    const max_updates_per_thread = max_updates_per_thread_count(max_candidates, block_size);
    const update_array = try allocator.alloc(f32, n_threads * max_updates_per_thread * 3);
    defer allocator.free(update_array); // np.empty: contents uninitialized
    const n_updates_per_thread = try allocator.alloc(i32, n_threads);
    defer allocator.free(n_updates_per_thread);
    @memset(n_updates_per_thread, 0);

    // For distance-based termination
    var prev_sum_dist: ?f64 = null;

    for (0..n_iters) |_| {
        var candidates = try nnd.build_candidates(allocator, &current_graph, max_candidates, rng_state, n_threads);
        defer candidates.deinit();

        var c: u64 = 0;
        n_vertices = n_samples;
        for (0..n_blocks + 1) |i| {
            const block_start = i * block_size;
            const block_end = @min(n_vertices, (i + 1) * block_size);

            const new_candidate_block = candidates.new_candidate_indices[block_start * max_candidates .. block_end * max_candidates];
            const old_candidate_block = candidates.old_candidate_indices[block_start * max_candidates .. block_end * max_candidates];

            const dist_thresholds = DistThresholds{
                .priorities = current_graph.distances,
                .stride = current_graph.size,
            };

            generate_graph_update_array_float(
                update_array,
                max_updates_per_thread,
                n_updates_per_thread,
                new_candidate_block,
                old_candidate_block,
                max_candidates,
                dist_thresholds,
                data,
                n_features,
                n_threads,
            );

            c += nnd.apply_graph_update_array(
                &current_graph,
                update_array,
                max_updates_per_thread,
                n_updates_per_thread,
                n_threads,
            );
        }

        // Check update count termination
        if (@as(f64, @floatFromInt(c)) <= delta * @as(f64, @floatFromInt(n_neighbors * n_samples))) {
            nnd.deheap_sort(current_graph.indices, current_graph.distances, n_neighbors);
            return current_graph;
        }

        // Check distance improvement termination (if enabled)
        if (delta_improv) |di| {
            const sum_dist = nn_descent_termination_sum(current_graph.distances);

            if (prev_sum_dist) |prev| {
                const rel_improv = @abs(sum_dist - prev) / @abs(prev);
                if (rel_improv < di) {
                    nnd.deheap_sort(current_graph.indices, current_graph.distances, n_neighbors);
                    return current_graph;
                }
            }

            prev_sum_dist = sum_dist;
        }

        block_size = @min(n_vertices, 2 * block_size);
        n_blocks = n_vertices / block_size;
    }

    nnd.deheap_sort(current_graph.indices, current_graph.distances, n_neighbors);
    return current_graph;
}

/// Port of nn_descent_float_sorted (bucketed update variant).
pub fn nn_descent_float_sorted(
    allocator: Allocator,
    data: []const f32,
    n_features: usize,
    n_neighbors: usize,
    rng_state: *rng.State,
    max_candidates: usize,
    n_iters: usize,
    delta: f64,
    delta_improv: ?f64,
    leaf_array: []const i32,
    leaf_width: usize,
    n_threads: usize,
) !nnd.Heap {
    const n_samples = data.len / n_features;
    var current_graph = try nnd.make_heap(allocator, n_samples, n_neighbors);
    errdefer current_graph.deinit();
    try init_rp_tree_float(allocator, data, n_features, &current_graph, leaf_array, leaf_width, n_threads);
    init_random_float(n_neighbors, data, n_features, &current_graph, rng_state);

    var n_vertices = n_samples;
    var block_size = 65536 / n_threads;
    var n_blocks = n_vertices / block_size;

    const max_updates_per_thread = max_updates_per_thread_count(max_candidates, block_size);
    // For sorted updates: buckets indexed by target block in the first dim.
    const sorted_update_array = try allocator.alloc(f32, n_threads * max_updates_per_thread * 3);
    defer allocator.free(sorted_update_array); // np.empty: uninitialized
    // (n_threads, n_threads + 1): column 0 unused, columns 1..n_threads
    // store the count from each generating thread.
    const n_updates_per_block = try allocator.alloc(i32, n_threads * (n_threads + 1));
    defer allocator.free(n_updates_per_block);
    @memset(n_updates_per_block, 0);

    // For distance-based termination
    var prev_sum_dist: ?f64 = null;

    for (0..n_iters) |_| {
        var candidates = try nnd.build_candidates(allocator, &current_graph, max_candidates, rng_state, n_threads);
        defer candidates.deinit();

        var c: u64 = 0;
        n_vertices = n_samples;
        for (0..n_blocks + 1) |i| {
            const block_start = i * block_size;
            const block_end = @min(n_vertices, (i + 1) * block_size);

            const new_candidate_block = candidates.new_candidate_indices[block_start * max_candidates .. block_end * max_candidates];
            const old_candidate_block = candidates.old_candidate_indices[block_start * max_candidates .. block_end * max_candidates];

            const dist_thresholds = DistThresholds{
                .priorities = current_graph.distances,
                .stride = current_graph.size,
            };

            // Reset update counts for this iteration
            @memset(n_updates_per_block, 0);

            try generate_sorted_graph_update_array_float(
                allocator,
                sorted_update_array,
                max_updates_per_thread,
                n_updates_per_block,
                new_candidate_block,
                old_candidate_block,
                max_candidates,
                dist_thresholds,
                data,
                n_features,
                n_threads,
            );

            c += nnd.apply_sorted_graph_updates(
                &current_graph,
                sorted_update_array,
                max_updates_per_thread,
                n_updates_per_block,
                n_threads,
            );
        }

        // Check update count termination
        if (@as(f64, @floatFromInt(c)) <= delta * @as(f64, @floatFromInt(n_neighbors * n_samples))) {
            nnd.deheap_sort(current_graph.indices, current_graph.distances, n_neighbors);
            return current_graph;
        }

        // Check distance improvement termination (if enabled)
        if (delta_improv) |di| {
            const sum_dist = nn_descent_termination_sum(current_graph.distances);

            if (prev_sum_dist) |prev| {
                const rel_improv = @abs(sum_dist - prev) / @abs(prev);
                if (rel_improv < di) {
                    nnd.deheap_sort(current_graph.indices, current_graph.distances, n_neighbors);
                    return current_graph;
                }
            }

            prev_sum_dist = sum_dist;
        }

        block_size = @min(n_vertices, 2 * block_size);
        n_blocks = n_vertices / block_size;
    }

    nnd.deheap_sort(current_graph.indices, current_graph.distances, n_neighbors);
    return current_graph;
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
        for (row.array.items, 0..) |v, j| {
            out[i * cols + j] = @floatCast(fixture.numberAsF64(v));
        }
    }
    return .{ .data = out, .rows = rows, .cols = cols };
}

fn readMatrixI32(alloc: Allocator, arr: std.json.Array) !struct { data: []i32, rows: usize, cols: usize } {
    const rows = arr.items.len;
    const cols = if (rows > 0) arr.items[0].array.items.len else 0;
    const out = try alloc.alloc(i32, rows * cols);
    errdefer alloc.free(out);
    for (arr.items, 0..) |row, i| {
        for (row.array.items, 0..) |v, j| {
            out[i * cols + j] = @intCast(v.integer);
        }
    }
    return .{ .data = out, .rows = rows, .cols = cols };
}

test "float_random_projection_split partitions indices without loss" {
    const alloc = std.testing.allocator;
    // 8 points in 2-D, distinct
    const data = [_]f32{ 0, 0, 1, 0, 0, 1, 1, 1, 2, 0, 0, 2, 2, 2, 3, 1 };
    var indices = [_]i32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    var state: rng.State = undefined;
    rng.seed(&state, 42);

    const split = try float_random_projection_split(alloc, &data, 2, &indices, &state);
    defer alloc.free(split.indices_left);
    defer alloc.free(split.indices_right);

    try std.testing.expectEqual(@as(usize, 8), split.indices_left.len + split.indices_right.len);
    var seen = [_]bool{false} ** 8;
    for (split.indices_left) |v| seen[@intCast(v)] = true;
    for (split.indices_right) |v| seen[@intCast(v)] = true;
    for (seen) |s| try std.testing.expect(s);
}

test "float_random_projection_split: identical points fall back to coin flips" {
    const alloc = std.testing.allocator;
    // all-identical points: hyperplane is the zero vector, margins all 0
    var data: [16]f32 = undefined;
    @memset(&data, 0.5);
    var indices = [_]i32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    var state: rng.State = undefined;
    rng.seed(&state, 7);

    const split = try float_random_projection_split(alloc, &data, 2, &indices, &state);
    defer alloc.free(split.indices_left);
    defer alloc.free(split.indices_right);
    try std.testing.expectEqual(@as(usize, 8), split.indices_left.len + split.indices_right.len);
}

test "make_float_leaf_array: every point appears exactly once, rows -1 padded" {
    const alloc = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(99);
    const n = 40;
    const d = 3;
    var data: [n * d]f32 = undefined;
    for (&data) |*v| v.* = prng.random().float(f32);

    var state: rng.State = undefined;
    rng.seed(&state, 3);
    var la = try make_float_leaf_array_serial(alloc, &data, d, &state, 8, 200);
    defer la.deinit();

    var counts = [_]u32{0} ** n;
    for (0..la.n_leaves) |i| {
        var seen_negative = false;
        for (0..la.width) |j| {
            const v = la.data[i * la.width + j];
            if (v < 0) {
                seen_negative = true;
            } else {
                // -1 padding is only at the tail of a row
                try std.testing.expect(!seen_negative);
                counts[@intCast(v)] += 1;
            }
        }
    }
    for (counts) |c| try std.testing.expectEqual(@as(u32, 1), c);
}

test "init_random_float fills empty heap rows" {
    const alloc = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(5);
    const n = 12;
    const d = 4;
    var data: [n * d]f32 = undefined;
    for (&data) |*v| v.* = prng.random().float(f32) - 0.5;

    var heap = try nnd.make_heap(alloc, n, 4);
    defer heap.deinit();
    var state: rng.State = undefined;
    rng.seed(&state, 11);
    init_random_float(4, &data, d, &heap, &state);

    for (0..n) |i| {
        var n_filled: usize = 0;
        for (heap.indicesRow(i)) |v| {
            if (v >= 0) n_filled += 1;
        }
        try std.testing.expect(n_filled > 0);
    }
}

test "fixture parity: make_float_forest_n64_d6_leaf10" {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "nndescent", "make_float_forest_n64_d6_leaf10");
    defer fx.deinit();

    const root = fx.value.object;
    const inputs = root.get("inputs").?.object;
    const m = try readMatrixF32(alloc, inputs.get("data").?.array);
    defer alloc.free(m.data);
    const leaf_size: i64 = inputs.get("leaf_size").?.integer;
    const max_depth: i64 = inputs.get("max_depth").?.integer;

    const states_in = inputs.get("rng_states").?.array;
    const rng_states = try alloc.alloc(rng.State, states_in.items.len);
    defer alloc.free(rng_states);
    for (states_in.items, 0..) |row, i| {
        for (0..3) |j| rng_states[i][j] = row.array.items[j].integer;
    }

    var forest = try make_float_forest(alloc, m.data, m.cols, rng_states, leaf_size, max_depth);
    defer forest.deinit();

    const exp_arrays = root.get("expected").?.object.get("leaf_arrays").?.array;
    try std.testing.expectEqual(exp_arrays.items.len, forest.arrays.len);
    for (exp_arrays.items, 0..) |exp_la_val, t| {
        const exp_la = exp_la_val.array;
        const got = forest.arrays[t];
        try std.testing.expectEqual(exp_la.items.len, got.n_leaves);
        try std.testing.expectEqual(exp_la.items[0].array.items.len, got.width);
        for (exp_la.items, 0..) |exp_row, i| {
            for (exp_row.array.items, 0..) |v, j| {
                try std.testing.expectEqual(
                    @as(i32, @intCast(v.integer)),
                    got.data[i * got.width + j],
                );
            }
        }
    }
}

fn checkNNDescentFixture(case: []const u8, comptime sorted: bool) !void {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "nndescent", case);
    defer fx.deinit();

    const root = fx.value.object;
    const tol_value = root.get("tolerance").?;
    const inputs = root.get("inputs").?.object;

    const m = try readMatrixF32(alloc, inputs.get("data").?.array);
    defer alloc.free(m.data);
    const la = try readMatrixI32(alloc, inputs.get("leaf_array").?.array);
    defer alloc.free(la.data);

    const n_neighbors: usize = @intCast(inputs.get("n_neighbors").?.integer);
    const max_candidates: usize = @intCast(inputs.get("max_candidates").?.integer);
    const n_iters: usize = @intCast(inputs.get("n_iters").?.integer);
    const delta = fixture.numberAsF64(inputs.get("delta").?);
    try std.testing.expect(inputs.get("delta_improv").? == .null);

    var state: rng.State = undefined;
    const state_in = inputs.get("rng_state").?.array;
    for (0..3) |i| state[i] = state_in.items[i].integer;

    const fn_impl = if (sorted) nn_descent_float_sorted else nn_descent_float;
    var graph = try fn_impl(
        alloc,
        m.data,
        m.cols,
        n_neighbors,
        &state,
        max_candidates,
        n_iters,
        delta,
        null,
        la.data,
        la.cols,
        1, // fixtures generated with NUMBA_NUM_THREADS=1
    );
    defer graph.deinit();

    const expected = root.get("expected").?.object;
    const exp_indices = expected.get("indices").?.array;
    const exp_distances = expected.get("distances").?.array;
    const dist_tol = fixture.toleranceForField(tol_value, "distances");
    const idx_tol = fixture.toleranceForField(tol_value, "indices");
    try std.testing.expect(idx_tol == .exact);

    for (0..m.rows) |i| {
        const idx_row = exp_indices.items[i].array;
        const dist_row = exp_distances.items[i].array;
        for (0..n_neighbors) |j| {
            try std.testing.expectEqual(
                @as(i32, @intCast(idx_row.items[j].integer)),
                graph.indices[i * n_neighbors + j],
            );
            try fixture.expectF32(
                dist_tol,
                graph.distances[i * n_neighbors + j],
                fixture.numberAsF64(dist_row.items[j]),
            );
        }
    }
}

test "fixture parity: nn_descent_float_n150_d8_k10" {
    try checkNNDescentFixture("nn_descent_float_n150_d8_k10", false);
}

test "fixture parity: nn_descent_float_sorted_n150_d8_k10" {
    try checkNNDescentFixture("nn_descent_float_sorted_n150_d8_k10", true);
}
