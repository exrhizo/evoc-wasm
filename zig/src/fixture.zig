//! Test-only helper for loading golden fixtures from $UVOC_FIXTURES. Never
//! shipped in the freestanding core; only referenced from `test` blocks.
//! See PORTING.md "Fixture format".

const std = @import("std");

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

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, contents, .{});
    return Fixture{ .parsed = parsed, .value = parsed.value };
}
