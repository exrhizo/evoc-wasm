//! Test-only helper for loading golden fixtures from $UVOC_FIXTURES. Never
//! shipped in the freestanding core; only referenced from `test` blocks.
//! See PORTING.md "Fixture format".

const std = @import("std");

/// Parsed form of a fixture's `tolerance` field. Three JSON shapes exist:
///   "exact"                                  -> .exact
///   {"rel": R} / {"rel": R, "abs": A}        -> .approx
///   {"default": ..., "fields": {name: ...}}  -> per-field (resolve with
///                                               toleranceForField)
pub const Tolerance = union(enum) {
    exact,
    approx: Approx,

    pub const Approx = struct { rel: f64, abs: f64 };
};

fn toleranceFromValue(v: std.json.Value) Tolerance {
    switch (v) {
        .string => |s| {
            std.debug.assert(std.mem.eql(u8, s, "exact"));
            return .exact;
        },
        .object => |o| {
            const rel = if (o.get("rel")) |r| numberAsF64(r) else 0.0;
            const abs = if (o.get("abs")) |a| numberAsF64(a) else 0.0;
            return .{ .approx = .{ .rel = rel, .abs = abs } };
        },
        else => unreachable,
    }
}

/// Resolves the tolerance that applies to `field` given a fixture's raw
/// `tolerance` JSON value, handling all three schema shapes.
pub fn toleranceForField(tolerance_value: std.json.Value, field: []const u8) Tolerance {
    if (tolerance_value == .object) {
        const o = tolerance_value.object;
        if (o.get("fields") != null or o.get("default") != null) {
            if (o.get("fields")) |fields| {
                if (fields.object.get(field)) |fv| return toleranceFromValue(fv);
            }
            return toleranceFromValue(o.get("default").?);
        }
    }
    return toleranceFromValue(tolerance_value);
}

/// JSON numbers may parse as .integer or .float depending on their lexical
/// form; normalize to f64. Out-of-range literals (e.g. the 1e999 written by
/// the Infinity rewrite in readFixture) surface as .number_string, which
/// parseFloat saturates to +/-inf.
pub fn numberAsF64(v: std.json.Value) f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        .number_string => |s| std.fmt.parseFloat(f64, s) catch unreachable,
        else => unreachable,
    };
}

/// Asserts `got` matches the fixture value `want` under `tol`. "Exact" f32
/// comparison means: cast the fixture f64 to f32 and compare bit patterns.
pub fn expectF32(tol: Tolerance, got: f32, want: f64) !void {
    switch (tol) {
        .exact => {
            const want_f32: f32 = @floatCast(want);
            try std.testing.expectEqual(@as(u32, @bitCast(want_f32)), @as(u32, @bitCast(got)));
        },
        .approx => |a| {
            const diff = @abs(@as(f64, got) - want);
            if (diff <= a.abs) return;
            if (diff <= a.rel * @abs(want)) return;
            std.debug.print("expectF32: got {e} want {e} (diff {e}, rel {e}, abs {e})\n", .{ got, want, diff, a.rel, a.abs });
            return error.TestExpectedApproxEq;
        },
    }
}

pub const Fixture = struct {
    parsed: std.json.Parsed(std.json.Value),
    value: std.json.Value,

    pub fn deinit(self: Fixture) void {
        self.parsed.deinit();
    }
};

/// Reads and parses fixtures/<component>/<case>.json using the UVOC_FIXTURES
/// env var set by build.zig (falls back to "../fixtures" relative to cwd,
/// which matches `zig build test`'s working directory when the env var is
/// unset for any reason).
pub fn readFixture(allocator: std.mem.Allocator, component: []const u8, case: []const u8) !Fixture {
    const fixtures_dir = std.process.getEnvVarOwned(allocator, "UVOC_FIXTURES") catch |err| switch (err) {
        error.EnvironmentVariableNotFound => try allocator.dupe(u8, "../fixtures"),
        else => return err,
    };
    defer allocator.free(fixtures_dir);

    const path = try std.fs.path.join(allocator, &.{ fixtures_dir, component, case });
    defer allocator.free(path);
    const full_path = try std.fmt.allocPrint(allocator, "{s}.json", .{path});
    defer allocator.free(full_path);

    const file = try std.fs.cwd().openFile(full_path, .{});
    defer file.close();

    const contents = try file.readToEndAlloc(allocator, 64 * 1024 * 1024);
    defer allocator.free(contents);

    // Python's json.dump emits bare Infinity tokens for float inf
    // (non-standard JSON; present in e.g. nndescent/build_candidates.json).
    // std.json rejects them, so rewrite to 1e999, which parseFloat saturates
    // to inf ("-Infinity" becomes "-1e999"). Fixture string fields never
    // contain the word "Infinity", so a plain textual replace is safe here.
    const fixed_len = std.mem.replacementSize(u8, contents, "Infinity", "1e999");
    const fixed = try allocator.alloc(u8, fixed_len);
    defer allocator.free(fixed);
    _ = std.mem.replace(u8, contents, "Infinity", "1e999", fixed);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, fixed, .{});
    return Fixture{ .parsed = parsed, .value = parsed.value };
}
