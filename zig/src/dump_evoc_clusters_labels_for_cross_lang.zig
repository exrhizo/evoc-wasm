//! NOT a validation test -- a one-shot harness run manually via
//! `zig test src/dump_evoc_clusters_labels_for_cross_lang.zig` (from the
//! `zig/` directory, with `EVOC_FIXTURES` and `EVOC_CROSS_LANG_OUT_DIR` set)
//! to dump Zig's `EVoC.fit_predict` labels for seeds 0..19 on the
//! `evoc_clusters_e2e_ari_check` fixture, as JSON, for the cross-language
//! (Rust vs Zig) agreement check described in
//! `rust/core/PCA_VALIDATION.md`. The companion Rust side is
//! `rust/core/tests/dump_evoc_clusters_labels_for_cross_lang.rs`, and
//! `scripts/compare_rust_zig_evoc_clusters.py` diffs the two dumps and
//! computes ARI. This file is intentionally NOT wired into `root.zig` /
//! `zig build test` -- it's a standalone harness, not part of the regular
//! suite.

const std = @import("std");
const fixture = @import("fixture.zig");
const clustering = @import("clustering.zig");

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

test "dump zig evoc_clusters labels for seed panel" {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "clustering", "evoc_clusters_e2e_ari_check");
    defer fx.deinit();
    const root = fx.value.object;
    const n_samples: usize = @intCast(root.get("n_samples").?.integer);
    const n_features: usize = @intCast(root.get("n_features").?.integer);

    const m = try readMatrixF32(alloc, root.get("data").?.array);
    defer alloc.free(m.data);
    try std.testing.expectEqual(n_samples * n_features, m.data.len);

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

    var out = std.ArrayList(u8).init(alloc);
    defer out.deinit();
    try out.appendSlice("{\n  \"labels\": {\n");

    var seed0_nn_inds: []i32 = &.{};
    var seed0_nn_dists: []f32 = &.{};
    defer if (seed0_nn_inds.len > 0) alloc.free(seed0_nn_inds);
    defer if (seed0_nn_dists.len > 0) alloc.free(seed0_nn_dists);

    var seed: i64 = 0;
    while (seed < 20) : (seed += 1) {
        var evoc = EVoC.init(alloc, seed, opts);
        defer evoc.deinit();
        const labels = try evoc.fit_predict(m.data, n_features);

        if (seed == 0) {
            seed0_nn_inds = try alloc.dupe(i32, evoc.result.?.nn_inds);
            seed0_nn_dists = try alloc.dupe(f32, evoc.result.?.nn_dists);
        }

        try out.writer().print("    \"{d}\": [", .{seed});
        for (labels, 0..) |l, i| {
            if (i > 0) try out.appendSlice(",");
            try out.writer().print("{d}", .{l});
        }
        try out.appendSlice(if (seed < 19) "],\n" else "]\n");
    }
    try out.appendSlice("  },\n");

    try out.appendSlice("  \"seed0_nn_inds\": [");
    for (seed0_nn_inds, 0..) |v, i| {
        if (i > 0) try out.appendSlice(",");
        try out.writer().print("{d}", .{v});
    }
    try out.appendSlice("],\n");

    try out.appendSlice("  \"seed0_nn_dists_bits\": [");
    for (seed0_nn_dists, 0..) |v, i| {
        if (i > 0) try out.appendSlice(",");
        try out.writer().print("{d}", .{@as(u32, @bitCast(v))});
    }
    try out.appendSlice("]\n");

    try out.appendSlice("}\n");

    const out_dir = std.process.getEnvVarOwned(alloc, "EVOC_CROSS_LANG_OUT_DIR") catch |err| switch (err) {
        error.EnvironmentVariableNotFound => try alloc.dupe(u8, "/tmp/evoc_cross_lang"),
        else => return err,
    };
    defer alloc.free(out_dir);

    try std.fs.cwd().makePath(out_dir);
    const out_path = try std.fmt.allocPrint(alloc, "{s}/zig_labels.json", .{out_dir});
    defer alloc.free(out_path);

    const f = try std.fs.cwd().createFile(out_path, .{});
    defer f.close();
    try f.writeAll(out.items);

    std.debug.print("wrote {s}\n", .{out_path});
}
