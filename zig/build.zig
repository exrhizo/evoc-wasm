const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);
    // Parity tests load golden fixtures relative to the repo, not the zig cache cwd.
    run_lib_unit_tests.setEnvironmentVariable("UVOC_FIXTURES", b.pathFromRoot("../fixtures"));

    const test_step = b.step("test", "Run unit + fixture parity tests");
    test_step.dependOn(&run_lib_unit_tests.step);

    // ---------------------------------------------------------------------
    // wasm32-freestanding build (see zig/WASM_API.md).
    //
    // Independent of the native target/optimize options above: always
    // wasm32-freestanding, always ReleaseSmall unless overridden, so
    // `zig build wasm` behaves the same regardless of `-Dtarget`/
    // `-Doptimize` flags a caller might also pass for the native steps.
    // ---------------------------------------------------------------------
    const wasm_simd = b.option(bool, "wasm-simd", "Enable wasm SIMD128 (baseline+simd128) for `zig build wasm`. MUST rerun `node scripts/wasm_fixture_runner.mjs` after enabling and confirm bit-exact fixture parity still holds -- SIMD can reorder floating-point ops.") orelse false;
    var wasm_target_query = std.Target.Query{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
    };
    if (wasm_simd) {
        wasm_target_query.cpu_features_add.addFeature(@intFromEnum(std.Target.wasm.Feature.simd128));
    }
    const wasm_target = b.resolveTargetQuery(wasm_target_query);
    const wasm_optimize = b.option(std.builtin.OptimizeMode, "wasm-optimize", "Optimize mode for `zig build wasm`") orelse .ReleaseSmall;

    const wasm_lib = b.addExecutable(.{
        .name = "uvoc_core",
        .root_source_file = b.path("src/wasm_api.zig"),
        .target = wasm_target,
        .optimize = wasm_optimize,
    });
    // Freestanding: no _start, export only the `export fn`s in wasm_api.zig
    // (plus the linear memory itself).
    wasm_lib.entry = .disabled;
    wasm_lib.rdynamic = true;

    const wasm_install = b.addInstallArtifact(wasm_lib, .{});
    const wasm_step = b.step("wasm", "Build wasm32-freestanding uvoc_core.wasm");
    wasm_step.dependOn(&wasm_install.step);

    // ---------------------------------------------------------------------
    // PROTOTYPE, additive/opt-in: separate wasm artifact for the threaded
    // RP-tree prototype (see wasm_api_rp_tree_proto.zig and
    // comparison/results.md's "Threaded RP-tree prototype" section). Built
    // by a dedicated `zig build wasm-rp-tree-proto` step only -- does not
    // touch the `wasm` step or `uvoc_core.wasm` above in any way.
    // ---------------------------------------------------------------------
    const rp_tree_proto_lib = b.addExecutable(.{
        .name = "uvoc_rp_tree_proto",
        .root_source_file = b.path("src/wasm_api_rp_tree_proto.zig"),
        .target = wasm_target,
        .optimize = wasm_optimize,
    });
    rp_tree_proto_lib.entry = .disabled;
    rp_tree_proto_lib.rdynamic = true;

    const rp_tree_proto_install = b.addInstallArtifact(rp_tree_proto_lib, .{});
    const rp_tree_proto_step = b.step("wasm-rp-tree-proto", "PROTOTYPE: build uvoc_rp_tree_proto.wasm (threaded RP-tree prototype, single-tree-build entry point)");
    rp_tree_proto_step.dependOn(&rp_tree_proto_install.step);
}
