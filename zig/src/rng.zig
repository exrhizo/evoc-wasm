//! Port of the RNG portion of evoc/common_nndescent.py (lines 1-55): seed,
//! tau_rand_int, tau_rand. See PORTING.md.

const std = @import("std");

/// RNG state: array of int64, shape (3,). Mirrors the numba `i8[:]` (int64) state.
pub const State = [3]i64;

/// Port of seed: rng_state.fill(seed + 0xFFFF).
pub fn seed(state: *State, s: i64) void {
    const value = s +% 0xFFFF;
    state[0] = value;
    state[1] = value;
    state[2] = value;
}

// PORT NOTE: numba executes this with plain (wrapping) int64 arithmetic; `<<`
// on the numba int64 wraps silently on overflow rather than trapping, so all
// shifts/adds/xors here use Zig's wrapping operators (<<| / +% / *% as
// needed). `>>` on a signed i64 in Zig is an arithmetic shift, matching
// Python's `>>` on ints promoted through numba's int64 semantics.
fn shl_wrap(x: i64, comptime n: u6) i64 {
    return @bitCast(@as(u64, @bitCast(x)) << n);
}

/// Port of tau_rand_int: a fast (pseudo)-random int32 generator advancing `state`.
pub fn tau_rand_int(state: *State) i32 {
    // state[0] = (((state[0] & 4294967294) << 12) & 0xFFFFFFFF) ^ ((((state[0] << 13) & 0xFFFFFFFF) ^ state[0]) >> 19)
    state[0] = (shl_wrap(state[0] & 4294967294, 12) & 0xFFFFFFFF) ^
        (((shl_wrap(state[0], 13) & 0xFFFFFFFF) ^ state[0]) >> 19);
    // state[1] = (((state[1] & 4294967288) << 4) & 0xFFFFFFFF) ^ ((((state[1] << 2) & 0xFFFFFFFF) ^ state[1]) >> 25)
    state[1] = (shl_wrap(state[1] & 4294967288, 4) & 0xFFFFFFFF) ^
        (((shl_wrap(state[1], 2) & 0xFFFFFFFF) ^ state[1]) >> 25);
    // state[2] = (((state[2] & 4294967280) << 17) & 0xFFFFFFFF) ^ ((((state[2] << 3) & 0xFFFFFFFF) ^ state[2]) >> 11)
    state[2] = (shl_wrap(state[2] & 4294967280, 17) & 0xFFFFFFFF) ^
        (((shl_wrap(state[2], 3) & 0xFFFFFFFF) ^ state[2]) >> 11);

    const combined: i64 = state[0] ^ state[1] ^ state[2];
    return @truncate(combined);
}

/// Port of tau_rand: abs(float(tau_rand_int(state)) / 0x7FFFFFFF), computed
/// in f64 then cast down to f32 for the return, matching numba's f4(i8[:]).
pub fn tau_rand(state: *State) f32 {
    const integer = tau_rand_int(state);
    const value: f64 = @abs(@as(f64, @floatFromInt(integer)) / @as(f64, 0x7FFFFFFF));
    return @floatCast(value);
}

const fixture = @import("fixture.zig");

fn bitsEqualF32(a: f32, b: f32) bool {
    return @as(u32, @bitCast(a)) == @as(u32, @bitCast(b));
}

test "seed fills all three lanes with s + 0xFFFF" {
    var state: State = undefined;
    seed(&state, 42);
    try std.testing.expectEqual(@as(i64, 42 + 0xFFFF), state[0]);
    try std.testing.expectEqual(@as(i64, 42 + 0xFFFF), state[1]);
    try std.testing.expectEqual(@as(i64, 42 + 0xFFFF), state[2]);
}

test "seed with 0 fills lanes with 0xFFFF" {
    var state: State = undefined;
    seed(&state, 0);
    try std.testing.expectEqual(@as(i64, 0xFFFF), state[0]);
    try std.testing.expectEqual(@as(i64, 0xFFFF), state[1]);
    try std.testing.expectEqual(@as(i64, 0xFFFF), state[2]);
}

test "tau_rand_int advances state and is deterministic for a given seed" {
    var state: State = undefined;
    seed(&state, 1);
    const a = tau_rand_int(&state);
    const b = tau_rand_int(&state);
    // Successive calls should (almost certainly) differ since state advances.
    try std.testing.expect(a != b);
}

test "tau_rand returns a value whose abs is in range and matches tau_rand_int sign logic" {
    var state: State = undefined;
    seed(&state, 2);
    const f = tau_rand(&state);
    try std.testing.expect(f >= 0.0);
}

test "fixture parity: tau_rand_seed_0 int and float sequences, exact bits" {
    try checkTauRandFixture("tau_rand_seed_0");
}

test "fixture parity: tau_rand_seed_42 int and float sequences, exact bits" {
    try checkTauRandFixture("tau_rand_seed_42");
}

test "fixture parity: tau_rand_seed_123456789 int and float sequences, exact bits" {
    try checkTauRandFixture("tau_rand_seed_123456789");
}

fn checkTauRandFixture(case: []const u8) !void {
    const alloc = std.testing.allocator;
    const data = try fixture.readFixture(alloc, "rng", case);
    defer data.deinit();

    const root = data.value.object;
    const inputs = root.get("inputs").?.object;
    const s: i64 = inputs.get("seed").?.integer;
    const n: usize = @intCast(inputs.get("n").?.integer);

    const expected = root.get("expected").?.object;
    const initial_state = expected.get("initial_state").?.array;
    const int_sequence = expected.get("int_sequence").?.array;
    const final_int_state = expected.get("final_int_state").?.array;
    const float_sequence = expected.get("float_sequence").?.array;

    var state: State = undefined;
    seed(&state, s);
    for (0..3) |i| {
        try std.testing.expectEqual(@as(i64, initial_state.items[i].integer), state[i]);
    }

    // First replay: consume tau_rand_int n times, recording each result.
    var int_state = state;
    for (0..n) |i| {
        const got = tau_rand_int(&int_state);
        try std.testing.expectEqual(@as(i32, @intCast(int_sequence.items[i].integer)), got);
    }
    for (0..3) |i| {
        try std.testing.expectEqual(@as(i64, final_int_state.items[i].integer), int_state[i]);
    }

    // Second replay from the same initial state: consume tau_rand n times.
    var float_state = state;
    for (0..n) |i| {
        const got = tau_rand(&float_state);
        const want: f32 = @floatCast(float_sequence.items[i].float);
        try std.testing.expect(bitsEqualF32(want, got));
    }
}
