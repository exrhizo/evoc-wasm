//! THROWAWAY: not part of the regular build/test suite. Native (non-wasm)
//! timing of `EVoC.fit_predict` for comparison/results.md's "Execution
//! time" table (VALIDATION_PLAN.md section 3, items 3-4 -- wasm vs native
//! overhead). Not referenced from build.zig or root.zig; run directly with:
//!   zig run zig/src/native_bench.zig -O ReleaseFast --deps clustering=... (see scripts note)
//! Simplest invocation from zig/:
//!   zig run -O ReleaseFast src/native_bench.zig
//! Safe to delete once the numbers are recorded in comparison/results.md.

const std = @import("std");
const clustering = @import("clustering.zig");
const fixture = @import("fixture.zig");
const EVoC = clustering.EVoC;
const EvocClustersOptions = clustering.EvocClustersOptions;

fn readMatrixF32(alloc: std.mem.Allocator, arr: std.json.Array) !struct { data: []f32, rows: usize, cols: usize } {
    const rows = arr.items.len;
    const cols = arr.items[0].array.items.len;
    const data = try alloc.alloc(f32, rows * cols);
    for (arr.items, 0..) |row, i| {
        for (row.array.items, 0..) |v, j| {
            data[i * cols + j] = switch (v) {
                .float => |f| @floatCast(f),
                .integer => |n| @floatFromInt(n),
                else => unreachable,
            };
        }
    }
    return .{ .data = data, .rows = rows, .cols = cols };
}

fn benchOne(alloc: std.mem.Allocator, component: []const u8, case: []const u8, label: []const u8) !void {
    const fx = try fixture.readFixture(alloc, component, case);
    defer fx.deinit();
    const root = fx.value.object;
    const n_features: usize = @intCast(root.get("n_features").?.integer);
    const m = try readMatrixF32(alloc, root.get("data").?.array);
    defer alloc.free(m.data);

    const opts = EvocClustersOptions{
        .noise_level = 0.5,
        .base_min_cluster_size = 5,
        .n_neighbors = 15,
        .min_samples = 5,
        .n_epochs = 50,
        .neighbor_scale = 1.0,
        .min_similarity_threshold = 0.2,
        .max_layers = 10,
        .n_label_prop_iter = 20,
        .node_embedding_dim = null,
    };

    const warmup = 2;
    const iters = 10;
    var times: [iters]f64 = undefined;
    var i: usize = 0;
    while (i < warmup + iters) : (i += 1) {
        var evoc = EVoC.init(alloc, 0, opts);
        defer evoc.deinit();
        var timer = try std.time.Timer.start();
        _ = try evoc.fit_predict(m.data, n_features);
        const elapsed_ns = timer.read();
        if (i >= warmup) {
            times[i - warmup] = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        }
    }
    std.mem.sort(f64, &times, {}, std.sort.asc(f64));
    const median = times[iters / 2];
    const min = times[0];
    const max = times[iters - 1];
    std.debug.print("  {s} (n={d}): median={d:.3} ms  min={d:.3} ms  max={d:.3} ms\n", .{ label, m.rows, median, min, max });
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    std.debug.print("\n=== native zig EVoC.fit_predict (ReleaseFast build) ===\n", .{});
    try benchOne(alloc, "clustering", "panel_small_balanced", "panel_small_balanced");
    try benchOne(alloc, "clustering", "evoc_clusters_e2e_ari_check", "evoc_clusters_e2e_ari_check");
    try benchOne(alloc, "clustering", "panel_large_many_clusters", "panel_large_many_clusters");
}
