//! Port of the heap machinery of evoc/common_nndescent.py: make_heap,
//! siftdown, deheap_sort, build_candidates_heap_push, build_candidates,
//! flagged_heap_push, apply_graph_update_array, apply_sorted_graph_updates.
//! (seed / tau_rand_int / tau_rand live in rng.zig.) See PORTING.md.
//!
//! PORT NOTE: this module's deheap_sort takes (indices, distances) — the
//! ARGUMENT ORDER IS OPPOSITE of kdtree.zig's deheap_sort(distances,
//! indices). Each module keeps its own reference's convention; do not mix.
//!
//! PORT NOTE: numba 3-D update arrays (n_threads, max_updates, 3) are flat
//! []f32 here, indexed as update_array[(t * max_updates + j) * 3 + c].

const std = @import("std");
const Allocator = std.mem.Allocator;
const rng = @import("rng.zig");

/// Port of the make_heap tuple (indices, distances, flags): row-major
/// n_points x size matrices.
pub const Heap = struct {
    indices: []i32,
    distances: []f32,
    flags: []u8,
    n_points: usize,
    size: usize,
    allocator: Allocator,

    pub fn indicesRow(self: *const Heap, i: usize) []i32 {
        return self.indices[i * self.size ..][0..self.size];
    }

    pub fn distancesRow(self: *const Heap, i: usize) []f32 {
        return self.distances[i * self.size ..][0..self.size];
    }

    pub fn flagsRow(self: *const Heap, i: usize) []u8 {
        return self.flags[i * self.size ..][0..self.size];
    }

    pub fn deinit(self: *Heap) void {
        self.allocator.free(self.indices);
        self.allocator.free(self.distances);
        self.allocator.free(self.flags);
        self.* = undefined;
    }
};

/// Port of make_heap: indices -1, distances inf, flags 0.
pub fn make_heap(allocator: Allocator, n_points: usize, size: usize) !Heap {
    const indices = try allocator.alloc(i32, n_points * size);
    errdefer allocator.free(indices);
    @memset(indices, -1);

    const distances = try allocator.alloc(f32, n_points * size);
    errdefer allocator.free(distances);
    @memset(distances, std.math.inf(f32));

    const flags = try allocator.alloc(u8, n_points * size);
    errdefer allocator.free(flags);
    @memset(flags, 0);

    return Heap{
        .indices = indices,
        .distances = distances,
        .flags = flags,
        .n_points = n_points,
        .size = size,
        .allocator = allocator,
    };
}

/// Port of siftdown: restore the max-heap property for heap1 (weights) with
/// heap2 (elements) kept in sync.
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

/// Port of deheap_sort: reorder each heap row by increasing distance,
/// in place. Arguments are (indices, distances) — see module PORT NOTE.
/// `indices`/`distances` are n_samples x n_neighbors row-major.
///
/// PORT NOTE: reference pranges over rows; sequential here (rows are
/// independent — identical result on the single-threaded WASM target).
pub fn deheap_sort(indices: []i32, distances: []f32, n_neighbors: usize) void {
    const n_samples = indices.len / n_neighbors;
    for (0..n_samples) |i| {
        const idx_row = indices[i * n_neighbors ..][0..n_neighbors];
        const dist_row = distances[i * n_neighbors ..][0..n_neighbors];
        // starting from the end of the array and moving back
        var j: usize = n_neighbors - 1;
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

/// Port of build_candidates_heap_push. Like flagged_heap_push but without
/// flags. Returns 1 if the heap changed, 0 otherwise.
///
/// PORT NOTE: reference locals declare u16 heap indices; usize here
/// (behaviorally irrelevant for any heap size < 32768).
pub fn build_candidates_heap_push(priorities: []f32, indices: []i32, p: f32, n: i32) i32 {
    if (p >= priorities[0]) {
        return 0;
    }

    const size = priorities.len;

    // break if we already have this element.
    for (0..size) |i| {
        if (n == indices[i]) {
            return 0;
        }
    }

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

/// Port of flagged_heap_push: as build_candidates_heap_push, but with a
/// parallel flags array; the inserted element's flag is set to 1.
pub fn flagged_heap_push(priorities: []f32, indices: []i32, flags: []u8, p: f32, n: i32) i32 {
    if (p >= priorities[0]) {
        return 0;
    }

    const size = priorities.len;

    // break if we already have this element.
    for (0..size) |i| {
        if (n == indices[i]) {
            return 0;
        }
    }

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
        flags[i] = flags[i_swap];

        i = i_swap;
    }

    priorities[i] = p;
    indices[i] = n;
    flags[i] = 1;

    return 1;
}

/// Candidate arrays returned by build_candidates: n_vertices x
/// max_candidates row-major i32 matrices.
pub const Candidates = struct {
    new_candidate_indices: []i32,
    old_candidate_indices: []i32,
    max_candidates: usize,
    allocator: Allocator,

    pub fn deinit(self: *Candidates) void {
        self.allocator.free(self.new_candidate_indices);
        self.allocator.free(self.old_candidate_indices);
        self.* = undefined;
    }
};

/// Port of build_candidates.
///
/// PORT NOTE: in the reference, `local_rng_state = rng_state + n` is a numpy
/// array COPY — build_candidates never mutates the caller's rng_state (the
/// fixture's rng_state_after equals the input). The pointer here is const to
/// make that explicit. Each per-thread copy adds `n` to all three lanes.
pub fn build_candidates(
    allocator: Allocator,
    current_graph: *Heap,
    max_candidates: usize,
    rng_state: *const rng.State,
    n_threads: usize,
) !Candidates {
    const current_indices = current_graph.indices;
    const current_flags = current_graph.flags;

    const n_vertices = current_graph.n_points;
    const n_neighbors = current_graph.size;

    const new_candidate_indices = try allocator.alloc(i32, n_vertices * max_candidates);
    errdefer allocator.free(new_candidate_indices);
    @memset(new_candidate_indices, -1);
    const new_candidate_priority = try allocator.alloc(f32, n_vertices * max_candidates);
    defer allocator.free(new_candidate_priority);
    @memset(new_candidate_priority, std.math.inf(f32));

    const old_candidate_indices = try allocator.alloc(i32, n_vertices * max_candidates);
    errdefer allocator.free(old_candidate_indices);
    @memset(old_candidate_indices, -1);
    const old_candidate_priority = try allocator.alloc(f32, n_vertices * max_candidates);
    defer allocator.free(old_candidate_priority);
    @memset(old_candidate_priority, std.math.inf(f32));

    const block_size = n_vertices / n_threads + 1;

    for (0..n_threads) |n| { // prange, sequential here
        var local_rng_state = rng.State{
            rng_state[0] +% @as(i64, @intCast(n)),
            rng_state[1] +% @as(i64, @intCast(n)),
            rng_state[2] +% @as(i64, @intCast(n)),
        };
        const block_start = n * block_size;
        const block_end = @min(block_start + block_size, n_vertices);

        for (0..n_vertices) |i| {
            for (0..n_neighbors) |j| {
                const idx: i64 = current_indices[i * n_neighbors + j];
                if (idx < 0) continue;
                const uidx: usize = @intCast(idx);

                if ((i >= block_start and i < block_end) or
                    (uidx >= block_start and uidx < block_end))
                {
                    const isn = current_flags[i * n_neighbors + j];
                    const d = rng.tau_rand(&local_rng_state);

                    if (isn != 0) {
                        if (i >= block_start and i < block_end) {
                            _ = build_candidates_heap_push(
                                new_candidate_priority[i * max_candidates ..][0..max_candidates],
                                new_candidate_indices[i * max_candidates ..][0..max_candidates],
                                d,
                                @intCast(idx),
                            );
                        }
                        if (uidx >= block_start and uidx < block_end) {
                            _ = build_candidates_heap_push(
                                new_candidate_priority[uidx * max_candidates ..][0..max_candidates],
                                new_candidate_indices[uidx * max_candidates ..][0..max_candidates],
                                d,
                                @intCast(i),
                            );
                        }
                    } else {
                        if (i >= block_start and i < block_end) {
                            _ = build_candidates_heap_push(
                                old_candidate_priority[i * max_candidates ..][0..max_candidates],
                                old_candidate_indices[i * max_candidates ..][0..max_candidates],
                                d,
                                @intCast(idx),
                            );
                        }
                        if (uidx >= block_start and uidx < block_end) {
                            _ = build_candidates_heap_push(
                                old_candidate_priority[uidx * max_candidates ..][0..max_candidates],
                                old_candidate_indices[uidx * max_candidates ..][0..max_candidates],
                                d,
                                @intCast(i),
                            );
                        }
                    }
                }
            }
        }
    }

    const indices = current_graph.indices;
    const flags = current_graph.flags;

    for (0..n_vertices) |i| { // prange, sequential here
        for (0..n_neighbors) |j| {
            const idx: i64 = indices[i * n_neighbors + j];

            for (0..max_candidates) |k| {
                if (new_candidate_indices[i * max_candidates + k] == idx) {
                    flags[i * n_neighbors + j] = 0;
                    break;
                }
            }
        }
    }

    return Candidates{
        .new_candidate_indices = new_candidate_indices,
        .old_candidate_indices = old_candidate_indices,
        .max_candidates = max_candidates,
        .allocator = allocator,
    };
}

/// Port of apply_graph_update_array. `update_array` is the flat
/// (n_threads, max_updates, 3) buffer. Returns the number of heap changes.
pub fn apply_graph_update_array(
    current_graph: *Heap,
    update_array: []const f32,
    max_updates: usize,
    n_updates_per_thread: []const i32,
    n_threads: usize,
) u32 {
    var n_changes: u32 = 0;

    const n_vertices = current_graph.n_points;
    const block_size = n_vertices / n_threads + 1;
    const n_rows = update_array.len / (max_updates * 3);

    for (0..n_threads) |n| { // prange, sequential here
        const block_start = n * block_size;
        const block_end = @min(block_start + block_size, n_vertices);

        for (0..n_rows) |i| {
            for (0..@as(usize, @intCast(n_updates_per_thread[i]))) |j| {
                const base = (i * max_updates + j) * 3;
                const p: i32 = @intFromFloat(update_array[base + 0]);

                if (p == -1) break;

                const q: i32 = @intFromFloat(update_array[base + 1]);
                const d: f32 = update_array[base + 2];

                const up: usize = @intCast(p);
                const uq: usize = @intCast(q);

                if (up >= block_start and up < block_end) {
                    const added = flagged_heap_push(
                        current_graph.distancesRow(up),
                        current_graph.indicesRow(up),
                        current_graph.flagsRow(up),
                        d,
                        q,
                    );
                    n_changes += @intCast(added);
                }

                if (uq >= block_start and uq < block_end) {
                    const added = flagged_heap_push(
                        current_graph.distancesRow(uq),
                        current_graph.indicesRow(uq),
                        current_graph.flagsRow(uq),
                        d,
                        p,
                    );
                    n_changes += @intCast(added);
                }
            }
        }
    }

    return n_changes;
}

/// Port of apply_sorted_graph_updates. `n_updates_per_block` is the flat
/// (n_threads, n_threads + 1) i32 matrix (column 0 unused; columns
/// 1..n_threads hold counts from each generating thread).
pub fn apply_sorted_graph_updates(
    current_graph: *Heap,
    update_array: []const f32,
    max_updates: usize,
    n_updates_per_block: []const i32,
    n_threads: usize,
) u32 {
    var n_changes: u32 = 0;

    const n_vertices = current_graph.n_points;
    const vertex_block_size = n_vertices / n_threads + 1;
    const max_updates_per_thread = max_updates / n_threads;

    for (0..n_threads) |n| { // prange, sequential here
        const block_start = n * vertex_block_size;
        const block_end = @min(block_start + vertex_block_size, n_vertices);

        // Process all updates in this block's bucket
        // Updates were written by each thread at offset t * max_updates_per_thread
        for (0..n_threads) |t| {
            const thread_start = t * max_updates_per_thread;
            const thread_count: usize = @intCast(n_updates_per_block[n * (n_threads + 1) + t + 1]);

            for (0..thread_count) |j| {
                const idx = thread_start + j;
                const base = (n * max_updates + idx) * 3;
                const p: i32 = @intFromFloat(update_array[base + 0]);
                const q: i32 = @intFromFloat(update_array[base + 1]);
                const d: f32 = update_array[base + 2];

                const up: usize = @intCast(p);
                const uq: usize = @intCast(q);

                // Apply update to p if it's in this block
                if (up >= block_start and up < block_end) {
                    const added = flagged_heap_push(
                        current_graph.distancesRow(up),
                        current_graph.indicesRow(up),
                        current_graph.flagsRow(up),
                        d,
                        q,
                    );
                    n_changes += @intCast(added);
                }

                // Apply update to q if it's in this block
                if (uq >= block_start and uq < block_end) {
                    const added = flagged_heap_push(
                        current_graph.distancesRow(uq),
                        current_graph.indicesRow(uq),
                        current_graph.flagsRow(uq),
                        d,
                        p,
                    );
                    n_changes += @intCast(added);
                }
            }
        }
    }

    return n_changes;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const fixture = @import("fixture.zig");

test "make_heap initializes indices -1, distances inf, flags 0" {
    var heap = try make_heap(std.testing.allocator, 3, 4);
    defer heap.deinit();
    for (heap.indices) |v| try std.testing.expectEqual(@as(i32, -1), v);
    for (heap.distances) |v| try std.testing.expect(std.math.isInf(v));
    for (heap.flags) |v| try std.testing.expectEqual(@as(u8, 0), v);
}

test "flagged_heap_push: rejects duplicates and worse priorities, sets flag" {
    var heap = try make_heap(std.testing.allocator, 1, 3);
    defer heap.deinit();
    const pri = heap.distancesRow(0);
    const idx = heap.indicesRow(0);
    const flg = heap.flagsRow(0);

    try std.testing.expectEqual(@as(i32, 1), flagged_heap_push(pri, idx, flg, 2.0, 5));
    // duplicate index rejected
    try std.testing.expectEqual(@as(i32, 0), flagged_heap_push(pri, idx, flg, 1.0, 5));
    try std.testing.expectEqual(@as(i32, 1), flagged_heap_push(pri, idx, flg, 3.0, 6));
    try std.testing.expectEqual(@as(i32, 1), flagged_heap_push(pri, idx, flg, 1.0, 7));
    // heap now full with {2,3,1}; max 3 at root
    try std.testing.expectEqual(@as(f32, 3.0), pri[0]);
    try std.testing.expectEqual(@as(i32, 0), flagged_heap_push(pri, idx, flg, 3.5, 8));
    // pushed elements have flag 1
    for (flg) |f| try std.testing.expectEqual(@as(u8, 1), f);
}

test "build_candidates_heap_push: duplicate rejection" {
    var pri = [_]f32{ std.math.inf(f32), std.math.inf(f32) };
    var idx = [_]i32{ -1, -1 };
    try std.testing.expectEqual(@as(i32, 1), build_candidates_heap_push(&pri, &idx, 0.5, 3));
    try std.testing.expectEqual(@as(i32, 0), build_candidates_heap_push(&pri, &idx, 0.2, 3));
    try std.testing.expectEqual(@as(i32, 1), build_candidates_heap_push(&pri, &idx, 0.2, 4));
}

test "deheap_sort sorts rows ascending (indices, distances argument order)" {
    // one row heap: max-heap by construction via pushes
    var pri = [_]f32{ std.math.inf(f32), std.math.inf(f32), std.math.inf(f32), std.math.inf(f32) };
    var idx = [_]i32{ -1, -1, -1, -1 };
    var flg = [_]u8{ 0, 0, 0, 0 };
    _ = flagged_heap_push(&pri, &idx, &flg, 3.0, 30);
    _ = flagged_heap_push(&pri, &idx, &flg, 1.0, 10);
    _ = flagged_heap_push(&pri, &idx, &flg, 4.0, 40);
    _ = flagged_heap_push(&pri, &idx, &flg, 2.0, 20);
    deheap_sort(&idx, &pri, 4);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }, &pri);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 10, 20, 30, 40 }, &idx);
}

test "fixture parity: flagged_heap_push" {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "nndescent", "flagged_heap_push");
    defer fx.deinit();

    const root = fx.value.object;
    const inputs = root.get("inputs").?.object;
    const k: usize = @intCast(inputs.get("k").?.integer);
    const pushes = inputs.get("pushes").?.array;

    var heap = try make_heap(alloc, 1, k);
    defer heap.deinit();
    const pri = heap.distancesRow(0);
    const idx = heap.indicesRow(0);
    const flg = heap.flagsRow(0);

    const expected = root.get("expected").?.object;
    const push_returns = expected.get("push_returns").?.array;

    for (pushes.items, 0..) |push_val, i| {
        const push = push_val.object;
        const p: f32 = @floatCast(fixture.numberAsF64(push.get("p").?));
        const n: i32 = @intCast(push.get("n").?.integer);
        const ret = flagged_heap_push(pri, idx, flg, p, n);
        try std.testing.expectEqual(@as(i32, @intCast(push_returns.items[i].integer)), ret);
    }

    const exp_p = expected.get("heap_priorities").?.array;
    const exp_i = expected.get("heap_indices").?.array;
    const exp_f = expected.get("heap_flags").?.array;
    for (0..k) |i| {
        try fixture.expectF32(.exact, pri[i], fixture.numberAsF64(exp_p.items[i]));
        try std.testing.expectEqual(@as(i32, @intCast(exp_i.items[i].integer)), idx[i]);
        try std.testing.expectEqual(@as(u8, @intCast(exp_f.items[i].integer)), flg[i]);
    }
}

test "fixture parity: build_candidates" {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "nndescent", "build_candidates");
    defer fx.deinit();

    const root = fx.value.object;
    const inputs = root.get("inputs").?.object;
    const graph_in = inputs.get("current_graph").?.object;
    const in_indices = graph_in.get("indices").?.array;
    const in_priorities = graph_in.get("priorities").?.array;
    const in_flags = graph_in.get("flags").?.array;

    const n_points = in_indices.items.len;
    const k = in_indices.items[0].array.items.len;

    // Reconstruct the graph arrays directly (per inputs.note).
    var heap = try make_heap(alloc, n_points, k);
    defer heap.deinit();
    for (0..n_points) |i| {
        for (0..k) |j| {
            heap.indices[i * k + j] = @intCast(in_indices.items[i].array.items[j].integer);
            heap.distances[i * k + j] = @floatCast(fixture.numberAsF64(in_priorities.items[i].array.items[j]));
            heap.flags[i * k + j] = @intCast(in_flags.items[i].array.items[j].integer);
        }
    }

    const max_candidates: usize = @intCast(inputs.get("max_candidates").?.integer);
    const n_threads: usize = @intCast(inputs.get("n_threads").?.integer);
    const rng_state_in = inputs.get("rng_state").?.array;
    var state: rng.State = undefined;
    for (0..3) |i| state[i] = rng_state_in.items[i].integer;

    var candidates = try build_candidates(alloc, &heap, max_candidates, &state, n_threads);
    defer candidates.deinit();

    const expected = root.get("expected").?.object;
    const exp_new = expected.get("new_candidate_indices").?.array;
    const exp_old = expected.get("old_candidate_indices").?.array;
    const exp_flags = expected.get("graph_flags_after").?.array;
    const exp_state = expected.get("rng_state_after").?.array;

    for (0..n_points) |i| {
        for (0..max_candidates) |j| {
            try std.testing.expectEqual(
                @as(i32, @intCast(exp_new.items[i].array.items[j].integer)),
                candidates.new_candidate_indices[i * max_candidates + j],
            );
            try std.testing.expectEqual(
                @as(i32, @intCast(exp_old.items[i].array.items[j].integer)),
                candidates.old_candidate_indices[i * max_candidates + j],
            );
        }
        for (0..k) |j| {
            try std.testing.expectEqual(
                @as(u8, @intCast(exp_flags.items[i].array.items[j].integer)),
                heap.flags[i * k + j],
            );
        }
    }
    // build_candidates works on per-thread copies; the input state is unchanged.
    for (0..3) |i| {
        try std.testing.expectEqual(@as(i64, exp_state.items[i].integer), state[i]);
    }
}
