//! Port of the distance kernels:
//!   - fast_cosine                              (evoc/float_nndescent.py)
//!   - fast_int_inner_product_dissimilarity     (evoc/int8_nndescent.py)
//!   - fast_bit_jaccard, fast_bit_jaccard_u64   (evoc/uint8_nndescent.py)
//! See PORTING.md.

const std = @import("std");

/// np.finfo(np.float32).tiny — smallest positive *normal* f32 (NOT the
/// smallest positive subnormal / floatTrueMin).
const EXP_NEG_INF: f32 = std.math.floatMin(f32);

/// Port of fast_cosine. Sequential f32 accumulation, no manual vectorization
/// (numba locals force `result: float32`; fastmath is numba-only — Rust/Zig
/// stay plain sequential per PORTING.md so all three backends can be
/// cross-checked bit-for-bit against each other).
pub fn fast_cosine(x: []const f32, y: []const f32) f32 {
    var result: f32 = 0.0;
    const dim = x.len;
    for (0..dim) |i| {
        result += x[i] * y[i];
    }

    if (result > 0.0) {
        return -result;
    } else {
        return -EXP_NEG_INF;
    }
}

/// Port of fast_int_inner_product_dissimilarity. Accumulates in wrapping
/// i32 (matches numba locals `result: int32`), returns -(f32)result.
pub fn fast_int_inner_product_dissimilarity(x: []const i8, y: []const i8) f32 {
    var result: i32 = 0;
    const dim = x.len;
    for (0..dim) |i| {
        result +%= @as(i32, x[i]) *% @as(i32, y[i]);
    }
    return -@as(f32, @floatFromInt(result));
}

fn popcnt_u8(v: u8) u32 {
    return @popCount(v);
}

/// Port of fast_bit_jaccard: per-byte popcount of AND / OR, accumulated in
/// u32, returns -(and/or) or 0.0 if the union is empty.
pub fn fast_bit_jaccard(x: []const u8, y: []const u8) f32 {
    var result: u32 = 0;
    var denom: u32 = 0;
    const dim = x.len;

    for (0..dim) |i| {
        const and_val = x[i] & y[i];
        const or_val = x[i] | y[i];
        result += popcnt_u8(and_val);
        denom += popcnt_u8(or_val);
    }

    if (denom > 0) {
        return -(@as(f32, @floatFromInt(result)) / @as(f32, @floatFromInt(denom)));
    } else {
        return 0.0;
    }
}

/// Port of fast_bit_jaccard_u64: reinterprets the byte arrays 8 bytes at a
/// time as little-endian u64 words, popcounts in u64 accumulators.
/// REQUIRES: array length divisible by 8 (asserted, matching the reference's
/// documented precondition).
pub fn fast_bit_jaccard_u64(x: []const u8, y: []const u8) f32 {
    std.debug.assert(x.len % 8 == 0);
    std.debug.assert(y.len % 8 == 0);

    var result: u64 = 0;
    var denom: u64 = 0;

    const n_u64 = x.len / 8;

    for (0..n_u64) |i| {
        const offset = i * 8;
        const x_val = std.mem.readInt(u64, x[offset..][0..8], .little);
        const y_val = std.mem.readInt(u64, y[offset..][0..8], .little);

        const and_val = x_val & y_val;
        const or_val = x_val | y_val;

        result += @popCount(and_val);
        denom += @popCount(or_val);
    }

    if (denom > 0) {
        return -(@as(f32, @floatFromInt(result)) / @as(f32, @floatFromInt(denom)));
    } else {
        return 0.0;
    }
}

const fixture = @import("fixture.zig");

fn bitsEqualF32(a: f32, b: f32) bool {
    return @as(u32, @bitCast(a)) == @as(u32, @bitCast(b));
}

fn approxEqualF32(got: f32, want: f64, rel: f64, abs_tol: f64) bool {
    const diff = @abs(@as(f64, got) - want);
    if (diff <= abs_tol) return true;
    return diff <= rel * @abs(want);
}

test "fast_cosine: zero vectors -> dot is 0.0, not > 0.0, returns -tiny" {
    const x = [_]f32{ 0.0, 0.0, 0.0 };
    const y = [_]f32{ 0.0, 0.0, 0.0 };
    const got = fast_cosine(&x, &y);
    try std.testing.expect(bitsEqualF32(got, -EXP_NEG_INF));
}

test "fast_cosine: single element positive dot" {
    const x = [_]f32{2.0};
    const y = [_]f32{3.0};
    const got = fast_cosine(&x, &y);
    try std.testing.expectEqual(@as(f32, -6.0), got);
}

test "fast_cosine: identical unit-ish vectors give strongly negative result" {
    const x = [_]f32{ 0.6, 0.8 };
    const got = fast_cosine(&x, &x);
    try std.testing.expect(got < 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), got, 1e-6);
}

test "fast_cosine: negative dot returns -EXP_NEG_INF, not the negated dot" {
    const x = [_]f32{ 1.0, 0.0 };
    const y = [_]f32{ -1.0, 0.0 };
    const got = fast_cosine(&x, &y);
    try std.testing.expect(bitsEqualF32(got, -EXP_NEG_INF));
}

test "fast_int_inner_product_dissimilarity: single element" {
    const x = [_]i8{5};
    const y = [_]i8{7};
    try std.testing.expectEqual(@as(f32, -35.0), fast_int_inner_product_dissimilarity(&x, &y));
}

test "fast_int_inner_product_dissimilarity: zero vectors -> 0" {
    const x = [_]i8{ 0, 0, 0 };
    const y = [_]i8{ 0, 0, 0 };
    try std.testing.expectEqual(@as(f32, -0.0), fast_int_inner_product_dissimilarity(&x, &y));
}

test "fast_int_inner_product_dissimilarity: extreme values wrap in i32 the same as numba int32" {
    // 127 * 127 summed many times can exceed i32 range only for very large
    // dims; here just check a large-magnitude single-element case stays exact.
    const x = [_]i8{-128};
    const y = [_]i8{127};
    try std.testing.expectEqual(@as(f32, 16256.0), fast_int_inner_product_dissimilarity(&x, &y));
}

test "fast_bit_jaccard: zero vectors -> denom 0 -> returns 0.0" {
    const x = [_]u8{ 0, 0, 0, 0 };
    const y = [_]u8{ 0, 0, 0, 0 };
    try std.testing.expectEqual(@as(f32, 0.0), fast_bit_jaccard(&x, &y));
}

test "fast_bit_jaccard: identical vectors -> -1.0" {
    const x = [_]u8{ 255, 128, 1, 7 };
    try std.testing.expectEqual(@as(f32, -1.0), fast_bit_jaccard(&x, &x));
}

test "fast_bit_jaccard: single element" {
    const x = [_]u8{0b1100};
    const y = [_]u8{0b1010};
    // and = 0b1000 (1 bit), or = 0b1110 (3 bits) -> -(1/3)
    try std.testing.expectApproxEqAbs(@as(f32, -1.0 / 3.0), fast_bit_jaccard(&x, &y), 1e-7);
}

test "fast_bit_jaccard_u64: identical vectors (len divisible by 8) -> -1.0" {
    const x = [_]u8{ 255, 128, 1, 7, 9, 10, 11, 12 };
    try std.testing.expectEqual(@as(f32, -1.0), fast_bit_jaccard_u64(&x, &x));
}

test "fast_bit_jaccard_u64: zero vectors -> 0.0" {
    const x = [_]u8{0} ** 16;
    const y = [_]u8{0} ** 16;
    try std.testing.expectEqual(@as(f32, 0.0), fast_bit_jaccard_u64(&x, &y));
}

test "fast_bit_jaccard_u64 agrees with fast_bit_jaccard on aligned data" {
    const x = [_]u8{ 23, 53, 251, 54, 20, 100, 172, 150 };
    const y = [_]u8{ 62, 129, 63, 63, 189, 40, 216, 12 };
    try std.testing.expectEqual(fast_bit_jaccard(&x, &y), fast_bit_jaccard_u64(&x, &y));
}

test "fixture parity: fast_cosine" {
    const alloc = std.testing.allocator;
    const data = try fixture.readFixture(alloc, "distance", "fast_cosine");
    defer data.deinit();

    const root = data.value.object;
    const inputs = root.get("inputs").?.object;
    const pairs = inputs.get("pairs").?.array;
    const tolerance = root.get("tolerance").?.object;
    const rel = tolerance.get("rel").?.float;
    const abs_tol = tolerance.get("abs").?.float;

    const expected = root.get("expected").?.object;
    const values = expected.get("values").?.array;

    var xs = std.ArrayList(f32).init(alloc);
    defer xs.deinit();
    var ys = std.ArrayList(f32).init(alloc);
    defer ys.deinit();

    for (pairs.items, 0..) |pair_val, i| {
        const pair = pair_val.object;
        xs.clearRetainingCapacity();
        ys.clearRetainingCapacity();
        for (pair.get("x").?.array.items) |v| try xs.append(@floatCast(v.float));
        for (pair.get("y").?.array.items) |v| try ys.append(@floatCast(v.float));

        const got = fast_cosine(xs.items, ys.items);
        const want = values.items[i].float;
        try std.testing.expect(approxEqualF32(got, want, rel, abs_tol));
    }
}

test "fixture parity: fast_int_inner_product_dissimilarity" {
    const alloc = std.testing.allocator;
    const data = try fixture.readFixture(alloc, "distance", "fast_int_inner_product_dissimilarity");
    defer data.deinit();

    const root = data.value.object;
    const inputs = root.get("inputs").?.object;
    const pairs = inputs.get("pairs").?.array;

    const expected = root.get("expected").?.object;
    const values = expected.get("values").?.array;

    var xs = std.ArrayList(i8).init(alloc);
    defer xs.deinit();
    var ys = std.ArrayList(i8).init(alloc);
    defer ys.deinit();

    for (pairs.items, 0..) |pair_val, i| {
        const pair = pair_val.object;
        xs.clearRetainingCapacity();
        ys.clearRetainingCapacity();
        for (pair.get("x").?.array.items) |v| try xs.append(@intCast(v.integer));
        for (pair.get("y").?.array.items) |v| try ys.append(@intCast(v.integer));

        const got = fast_int_inner_product_dissimilarity(xs.items, ys.items);
        const want: f32 = @floatCast(values.items[i].float);
        try std.testing.expect(bitsEqualF32(got, want));
    }
}

test "fixture parity: fast_bit_jaccard and fast_bit_jaccard_u64" {
    const alloc = std.testing.allocator;
    const data = try fixture.readFixture(alloc, "distance", "fast_bit_jaccard");
    defer data.deinit();

    const root = data.value.object;
    const inputs = root.get("inputs").?.object;
    const pairs = inputs.get("pairs").?.array;

    const expected = root.get("expected").?.object;
    const values = expected.get("values").?.array;
    const values_u64 = expected.get("values_u64").?.array;

    var xs = std.ArrayList(u8).init(alloc);
    defer xs.deinit();
    var ys = std.ArrayList(u8).init(alloc);
    defer ys.deinit();

    for (pairs.items, 0..) |pair_val, i| {
        const pair = pair_val.object;
        xs.clearRetainingCapacity();
        ys.clearRetainingCapacity();
        for (pair.get("x").?.array.items) |v| try xs.append(@intCast(v.integer));
        for (pair.get("y").?.array.items) |v| try ys.append(@intCast(v.integer));

        const got = fast_bit_jaccard(xs.items, ys.items);
        const want: f32 = @floatCast(values.items[i].float);
        try std.testing.expect(bitsEqualF32(got, want));

        // All fixture arrays here happen to have length divisible by 8.
        if (xs.items.len % 8 == 0) {
            const got_u64 = fast_bit_jaccard_u64(xs.items, ys.items);
            const want_u64: f32 = @floatCast(values_u64.items[i].float);
            try std.testing.expect(bitsEqualF32(got_u64, want_u64));
        }
    }
}
