//! PROTOTYPE, additive and opt-in: a separate wasm32-freestanding entry point
//! (its own `zig build` step, its own .wasm artifact) exposing a single
//! RP-tree build as one flat C-ABI call, so a JS host can build the trees of
//! an RP-forest in parallel across Node `worker_threads` -- one call per
//! (tree, rng_state) pair, one call per worker.
//!
//! This file does NOT modify or duplicate any phase 1-8 core algorithm file.
//! It calls the exact same `nndescent_float.make_float_leaf_array_serial`
//! that the existing single-threaded `wasm_api.zig` -> `knn_graph.zig` ->
//! `nndescent_float.make_float_forest` path already uses, so a JS caller
//! computing N independent trees' leaf arrays this way and re-assembling
//! them (pad-to-max-width + vstack, exactly as `knn_graph.make_forest`
//! does) is guaranteed to produce a byte-identical Forest to the existing
//! single-threaded path -- see comparison/results.md's "Threaded RP-tree
//! prototype" section for the verification.
//!
//! ## Why NOT shared WebAssembly.Memory + Atomics across worker instances
//!
//! Investigated first (per the task brief) and confirmed working at a toy
//! level: Zig 0.13 wasm32-freestanding CAN be built with
//! `-mcpu=baseline+atomics+bulk_memory --shared-memory --import-memory
//! --max-memory=N`, and multiple Node `worker_threads` instantiating the
//! SAME compiled module against the SAME `WebAssembly.Memory{shared:true}`
//! DO see each other's atomic writes (verified: 4 workers concurrently
//! incrementing a shared `std.atomic.Value(i32)` via an exported `bump()`
//! landed on the exact expected total, no lost updates).
//!
//! BUT: `std.Thread.spawn` itself does not work on this target at all
//! (`error: Cannot spawn thread when building in single-threaded mode` /
//! `Unsupported operating system freestanding` -- confirmed directly with a
//! minimal test file) so Zig-level threading primitives are a non-starter
//! regardless. That still leaves the "manual worker-per-instance,
//! shared-memory" pattern the task described as viable in principle -- BUT
//! it is unsafe for this specific use case: `std.heap.wasm_allocator`
//! (which `wasm_api.zig` and all of `make_float_leaf_array_serial`'s
//! allocations funnel through) is a bump/free-list allocator that assumes
//! it is the ONLY thing managing the module's linear memory. If N
//! independent module INSTANCES share one `WebAssembly.Memory`, each
//! instance's `wasm_allocator` independently believes it owns the entire
//! heap starting right after static data, and will hand out overlapping
//! addresses to concurrent callers in different workers -- silent heap
//! corruption, not a crash, so it would be an easy-to-miss correctness bug
//! rather than a loud failure. Making this actually safe would require
//! either (a) a custom, worker-partitioned allocator (each instance offset
//! to a private slice of the shared memory, negotiated up front -- real
//! engineering, not a prototype-scope change), or (b) not allocating at all
//! in the shared region. Given RP-tree construction is allocation-heavy
//! (one Vec-of-slices per tree, resized many times during the split
//! recursion), (b) isn't practical here either.
//!
//! ## The pattern actually used instead: private memory per worker
//!
//! Each RP-tree is fully independent (confirmed by reading
//! `nndescent_float.make_float_forest_no_nested_parallelism`: the only
//! inputs are `data` (read-only, shared logically but each worker gets its
//! own copy written into its own private wasm memory), `n_features`, and
//! that tree's own `rng_states[i]` row -- no cross-tree shared mutable
//! state at all, only a post-hoc pad+vstack merge). So instead of shared
//! memory, each Node worker_thread compiles+instantiates its OWN private
//! (non-shared, ordinary) instance of this same module, builds its
//! assigned tree(s) with its own private linear memory (no allocator
//! sharing, no races possible), and returns the resulting flat i32 buffer
//! to the main thread over `postMessage` as a transferable `ArrayBuffer`
//! (zero-copy). The main thread does the exact same pad-to-max-width +
//! vstack merge `make_forest` does, in the same tree order, and the result
//! is verified bit-identical to the single-threaded reference (built by
//! calling this same exported function sequentially, all trees, in one
//! instance) -- see scripts/wasm_threaded_prototype_zig.mjs.

const std = @import("std");
const rng = @import("rng.zig");
const nndf = @import("nndescent_float.zig");

const allocator = std.heap.wasm_allocator;

export fn alloc(len: usize) ?[*]u8 {
    const mem = allocator.alloc(u8, len) catch return null;
    return mem.ptr;
}

export fn dealloc(ptr: [*]u8, len: usize) void {
    allocator.free(ptr[0..len]);
}

/// Build ONE RP-tree's leaf array and pack it into a freshly-`alloc`ed
/// buffer the caller must `dealloc` (byte length = the return value):
///
///   [0]           : u32 n_leaves
///   [1]           : u32 width
///   [2 .. 2+n*w)  : i32, row-major n_leaves x width, rows padded with -1
///                   (identical padding convention to `LeafArray`/
///                   `make_forest`)
///
/// `data_ptr` points to `n_samples * n_features` f32 values (row-major),
/// already written into THIS instance's own linear memory by the caller
/// via `alloc`. Returns the packed buffer's pointer, or null on OOM.
/// `out_len_ptr` (a single u32 slot the caller `alloc`ed) receives the
/// buffer's total byte length.
export fn build_rp_tree_leaf_array(
    data_ptr: [*]const f32,
    n_samples: usize,
    n_features: usize,
    rng0: i64,
    rng1: i64,
    rng2: i64,
    leaf_size: i64,
    max_depth: i64,
    out_len_ptr: [*]u32,
) ?[*]u8 {
    const data = data_ptr[0 .. n_samples * n_features];
    var state: rng.State = .{ rng0, rng1, rng2 };

    var leaf = nndf.make_float_leaf_array_serial(
        allocator,
        data,
        n_features,
        &state,
        leaf_size,
        max_depth,
    ) catch return null;
    defer leaf.deinit();

    const header_bytes = 2 * @sizeOf(u32);
    const body_bytes = leaf.data.len * @sizeOf(i32);
    const total_bytes = header_bytes + body_bytes;

    const out = allocator.alloc(u8, total_bytes) catch return null;
    const out_u32: [*]u32 = @ptrCast(@alignCast(out.ptr));
    out_u32[0] = @intCast(leaf.n_leaves);
    out_u32[1] = @intCast(leaf.width);

    const out_i32: [*]i32 = @ptrCast(@alignCast(out.ptr + header_bytes));
    @memcpy(out_i32[0..leaf.data.len], leaf.data);

    out_len_ptr[0] = @intCast(total_bytes);
    return out.ptr;
}
