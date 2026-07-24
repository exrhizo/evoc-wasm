//! Port of evoc/disjoint_set.py. Keep the same structure and names so the
//! Python, Rust, and Zig sources diff side by side. See PORTING.md.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Port of RankDisjointSet: parent + rank arrays, union by rank.
pub const RankDisjointSet = struct {
    parent: []i32,
    rank: []i32,
    allocator: Allocator,

    pub fn deinit(self: *RankDisjointSet) void {
        self.allocator.free(self.parent);
        self.allocator.free(self.rank);
        self.* = undefined;
    }
};

/// Port of SizeDisjointSet: parent + size arrays, union by size.
pub const SizeDisjointSet = struct {
    parent: []i32,
    size: []i32,
    allocator: Allocator,

    pub fn deinit(self: *SizeDisjointSet) void {
        self.allocator.free(self.parent);
        self.allocator.free(self.size);
        self.* = undefined;
    }
};

/// Port of ds_rank_create: parent = arange(n), rank = zeros(n).
pub fn ds_rank_create(allocator: Allocator, n_elements: usize) !RankDisjointSet {
    const parent = try allocator.alloc(i32, n_elements);
    errdefer allocator.free(parent);
    const rank = try allocator.alloc(i32, n_elements);
    errdefer allocator.free(rank);

    for (parent, 0..) |*p, i| p.* = @intCast(i);
    @memset(rank, 0);

    return RankDisjointSet{ .parent = parent, .rank = rank, .allocator = allocator };
}

/// Port of ds_size_create: parent = arange(n), size = ones(n).
pub fn ds_size_create(allocator: Allocator, n_elements: usize) !SizeDisjointSet {
    const parent = try allocator.alloc(i32, n_elements);
    errdefer allocator.free(parent);
    const size = try allocator.alloc(i32, n_elements);
    errdefer allocator.free(size);

    for (parent, 0..) |*p, i| p.* = @intCast(i);
    @memset(size, 1);

    return SizeDisjointSet{ .parent = parent, .size = size, .allocator = allocator };
}

// PORT NOTE: the Python is
//     while disjoint_set.parent[x] != x:
//         x, disjoint_set.parent[x] = (
//             disjoint_set.parent[x],
//             disjoint_set.parent[disjoint_set.parent[x]],
//         )
// Python evaluates the RHS tuple *before* doing any assignment, then binds
// targets left to right. So per iteration:
//   1. rhs0 = parent[x]              (the new value of x)
//   2. rhs1 = parent[parent[x]]      (computed using the OLD x, i.e. parent[rhs0])
//   3. x = rhs0                      (x updated first)
//   4. parent[x] = rhs1              (store uses the NEW x from step 3, i.e. parent[rhs0] = rhs1)
// Net effect: parent[parent[x_old]] = parent[parent[parent[x_old]]] (path halving),
// then x becomes parent[x_old]. The store target coincides with the read of
// rhs0 (parent[x_old]) only if rhs0 == x_new, which it always is by
// construction, so the store is never a true no-op post-update; it writes
// path-halving compression onto the node x is moving to.
// `disjoint_set` is `*RankDisjointSet` or `*SizeDisjointSet` — both have a
// `.parent: []i32` field. Zig has no structural typing, so `anytype` stands
// in for Python's duck typing while keeping a single `ds_find` definition.
pub fn ds_find(disjoint_set: anytype, x_in: i32) i32 {
    const parent = disjoint_set.parent;
    var x = x_in;
    while (parent[@intCast(x)] != x) {
        const rhs0 = parent[@intCast(x)];
        const rhs1 = parent[@intCast(parent[@intCast(x)])];
        x = rhs0;
        parent[@intCast(x)] = rhs1;
    }
    return x;
}

/// Port of ds_union_by_rank.
pub fn ds_union_by_rank(disjoint_set: *RankDisjointSet, x_in: i32, y_in: i32) void {
    var x = ds_find(disjoint_set, x_in);
    var y = ds_find(disjoint_set, y_in);

    if (x == y) return;

    if (disjoint_set.rank[@intCast(x)] < disjoint_set.rank[@intCast(y)]) {
        const tmp = x;
        x = y;
        y = tmp;
    }

    disjoint_set.parent[@intCast(y)] = x;
    if (disjoint_set.rank[@intCast(x)] == disjoint_set.rank[@intCast(y)]) {
        disjoint_set.rank[@intCast(x)] += 1;
    }
}

/// Port of ds_union_by_size.
pub fn ds_union_by_size(disjoint_set: *SizeDisjointSet, x_in: i32, y_in: i32) void {
    var x = ds_find(disjoint_set, x_in);
    var y = ds_find(disjoint_set, y_in);

    if (x == y) return;

    if (disjoint_set.size[@intCast(x)] < disjoint_set.size[@intCast(y)]) {
        const tmp = x;
        x = y;
        y = tmp;
    }

    disjoint_set.parent[@intCast(y)] = x;
    disjoint_set.size[@intCast(x)] += disjoint_set.size[@intCast(y)];
}

const fixture = @import("fixture.zig");

test "ds_rank_create initializes parent to identity and rank to zero" {
    var ds = try ds_rank_create(std.testing.allocator, 5);
    defer ds.deinit();
    for (ds.parent, 0..) |p, i| try std.testing.expectEqual(@as(i32, @intCast(i)), p);
    for (ds.rank) |r| try std.testing.expectEqual(@as(i32, 0), r);
}

test "ds_size_create initializes parent to identity and size to one" {
    var ds = try ds_size_create(std.testing.allocator, 5);
    defer ds.deinit();
    for (ds.parent, 0..) |p, i| try std.testing.expectEqual(@as(i32, @intCast(i)), p);
    for (ds.size) |s| try std.testing.expectEqual(@as(i32, 1), s);
}

test "single element: find returns itself" {
    var ds = try ds_rank_create(std.testing.allocator, 1);
    defer ds.deinit();
    try std.testing.expectEqual(@as(i32, 0), ds_find(&ds, 0));
}

test "self-union is a no-op" {
    var ds = try ds_rank_create(std.testing.allocator, 3);
    defer ds.deinit();
    ds_union_by_rank(&ds, 1, 1);
    try std.testing.expectEqual(@as(i32, 1), ds.parent[1]);
    try std.testing.expectEqual(@as(i32, 0), ds.rank[1]);
}

test "union by rank: tie increments rank of new root" {
    var ds = try ds_rank_create(std.testing.allocator, 2);
    defer ds.deinit();
    ds_union_by_rank(&ds, 0, 1);
    // Tie (both rank 0): x==0 stays root (no swap since rank[0] < rank[1] is false).
    try std.testing.expectEqual(@as(i32, 0), ds_find(&ds, 1));
    try std.testing.expectEqual(@as(i32, 1), ds.rank[0]);
}

test "union by size: root chosen by larger size, size accumulates" {
    var ds = try ds_size_create(std.testing.allocator, 4);
    defer ds.deinit();
    ds_union_by_size(&ds, 0, 1); // size[0] becomes 2
    ds_union_by_size(&ds, 2, 3); // size[2] becomes 2
    ds_union_by_size(&ds, 0, 2); // tie in size -> x(=0) stays root
    const root = ds_find(&ds, 3);
    try std.testing.expectEqual(@as(i32, 0), root);
    try std.testing.expectEqual(@as(i32, 4), ds.size[0]);
}

test "fixture parity: union_rank_n24" {
    const alloc = std.testing.allocator;
    const data = try fixture.readFixture(alloc, "disjoint_set", "union_rank_n24");
    defer data.deinit();

    const root = data.value.object;
    const inputs = root.get("inputs").?.object;
    const n: usize = @intCast(inputs.get("n").?.integer);
    const union_ops = inputs.get("union_ops").?.array;

    var ds = try ds_rank_create(alloc, n);
    defer ds.deinit();

    const expected = root.get("expected").?.object;
    const expected_finds = expected.get("find_after_each_union").?.array;
    const expected_parent = expected.get("final_parent").?.array;
    const expected_rank = expected.get("final_rank").?.array;

    for (union_ops.items, 0..) |op, i| {
        const pair = op.array;
        const x: i32 = @intCast(pair.items[0].integer);
        const y: i32 = @intCast(pair.items[1].integer);
        ds_union_by_rank(&ds, x, y);
        const found = ds_find(&ds, x);
        try std.testing.expectEqual(@as(i32, @intCast(expected_finds.items[i].integer)), found);
    }

    for (ds.parent, 0..) |p, i| {
        try std.testing.expectEqual(@as(i32, @intCast(expected_parent.items[i].integer)), p);
    }
    for (ds.rank, 0..) |r, i| {
        try std.testing.expectEqual(@as(i32, @intCast(expected_rank.items[i].integer)), r);
    }
}

test "fixture parity: union_size_n24" {
    const alloc = std.testing.allocator;
    const data = try fixture.readFixture(alloc, "disjoint_set", "union_size_n24");
    defer data.deinit();

    const root = data.value.object;
    const inputs = root.get("inputs").?.object;
    const n: usize = @intCast(inputs.get("n").?.integer);
    const union_ops = inputs.get("union_ops").?.array;

    var ds = try ds_size_create(alloc, n);
    defer ds.deinit();

    const expected = root.get("expected").?.object;
    const expected_finds = expected.get("find_after_each_union").?.array;
    const expected_parent = expected.get("final_parent").?.array;
    const expected_size = expected.get("final_size").?.array;

    for (union_ops.items, 0..) |op, i| {
        const pair = op.array;
        const x: i32 = @intCast(pair.items[0].integer);
        const y: i32 = @intCast(pair.items[1].integer);
        ds_union_by_size(&ds, x, y);
        const found = ds_find(&ds, x);
        try std.testing.expectEqual(@as(i32, @intCast(expected_finds.items[i].integer)), found);
    }

    for (ds.parent, 0..) |p, i| {
        try std.testing.expectEqual(@as(i32, @intCast(expected_parent.items[i].integer)), p);
    }
    for (ds.size, 0..) |s, i| {
        try std.testing.expectEqual(@as(i32, @intCast(expected_size.items[i].integer)), s);
    }
}
