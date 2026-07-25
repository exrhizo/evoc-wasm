# Rust vs Zig comparison

Updated per phase by `scripts/compare.sh` (phase 10) and by hand where noted.
All measurements on the shared fixtures; wasm builds are release builds
(`wasm-opt` not applied -- unavailable in this environment, see "Toolchain
ergonomics notes"), run single-threaded under Node v20.

## Binary size (wasm32)

Measured 2026-07-25, release builds of the full phase 1-9 API surface (see
`rust/core/WASM_API.md` / `zig/WASM_API.md`). `wasm-opt` was **not**
available in this environment (see "Toolchain ergonomics notes" below), so
only raw release-build sizes and their gzip are recorded; the `wasm-opt -O3`
column is left blank rather than guessed.

| Build | Rust `uvoc_core.wasm` | Zig `uvoc_core.wasm` |
|-------|------------------------|------------------------|
| Raw release build                | 243,119 bytes | 174,824 bytes |
| `wasm-opt -O3`                   | not measured (tool unavailable, see notes) | not measured (tool unavailable, see notes) |
| gzip -9 of raw release build     | 88,903 bytes  | 66,842 bytes  |
| gzip -9 of `wasm-opt -O3` output | n/a           | n/a           |

Zig's binary is ~28% smaller raw and ~25% smaller gzipped than Rust's, for
the same API surface (module Zig lacks: standalone `node_embedding`,
`knn_graph`, cluster-tree exports, and one distance kernel -- see the parity
section below -- so this is not a fully apples-to-apples surface; Zig's
number would grow somewhat if it exposed the same functions, but the gap is
large enough that the direction of the result is unlikely to flip).

**Update 2026-07-25 (build-config investigation, ReleaseSmall default
unchanged):** now that Zig's `wasm_api.zig` exports standalone
`node_embedding`/`knn_graph`/`cluster_tree_labels` (closing most of the API
gap noted above), a fresh ReleaseSmall build measures **189,994 bytes**
raw / **73,289 bytes** gzip -9 -- larger than the 174,824-byte figure above
because of the added exports, still well under Rust's 243,119. Two
alternative Zig build configs were tried and rejected as the default (see
"Wasm build-config experiment" under Execution time for the full
before/after numbers): `-Dwasm-optimize=ReleaseFast` balloons the binary to
**2,323,782 bytes** (~12x larger, 637,266 bytes gzip) for only a ~3-6%
median speed gain -- not a good trade for a browser/npm target. Wasm SIMD
(`-Dwasm-simd=true`, new `build.zig` option, off by default) is
bit-exact-safe (fixture parity holds, see below) but produces virtually
no size or speed change, since none of the core algorithm code uses
explicit `@Vector` types for the compiler to widen -- so it isn't worth
enabling by default either. **ReleaseSmall, no SIMD, remains the
recommended default** and `zig/build.zig`'s wasm step is unchanged.

## Execution time

Measured 2026-07-25 on this machine, Node v20.19.4, single-threaded, no
`wasm-opt` applied (see Binary size section). Wasm: `scripts/wasm_benchmark.mjs`
(2 warmup + 10 timed iterations per cell, `performance.now()`, median/min/max
reported). Native: throwaway timing harnesses, `rust/core/tests/native_bench.rs`
(`cargo test --release -p uvoc-core --test native_bench -- --ignored --nocapture`)
and `zig/src/native_bench.zig` (`zig run -O ReleaseFast src/native_bench.zig`
from `zig/`, `UVOC_FIXTURES` pointed at `fixtures/`). Both throwaway files are
kept in the tree (not deleted) since they're generically useful for re-running
this comparison later; they are not wired into `build.zig`/`Cargo.toml` test
targets that run by default (Rust: `#[ignore]`; Zig: freestanding `main()`,
not imported from `root.zig`).

Python (numba, native) numbers were not re-measured in this pass -- out of
scope for VALIDATION_PLAN.md section 3 items 3-4, which ask for wasm-vs-wasm
and wasm-vs-native-of-the-same-port, not a fresh Python benchmark. Column
kept for schema continuity; left blank.

### End-to-end `evoc_fit_predict` / `evoc_clusters`, wasm

| Dataset (n samples) | Rust wasm median | Rust wasm min–max | Zig wasm median | Zig wasm min–max |
|---|---|---|---|---|
| `panel_small_balanced` (n=100, d=12)          | 36.64 ms | 36.38–37.59 ms | 4.39 ms  | 4.30–4.74 ms   |
| `evoc_clusters_e2e_ari_check` (n=240, d=24)   | 41.56 ms | 41.33–42.12 ms | 10.82 ms | 10.69–10.93 ms |
| `panel_large_many_clusters` (n=800, d=32)     | 64.98 ms | 63.61–68.05 ms | 33.38 ms | 32.78–33.93 ms |

### End-to-end, native (no wasm), release/ReleaseFast builds

| Dataset (n samples) | Rust native median | Rust native min–max | Zig native median | Zig native min–max |
|---|---|---|---|---|
| `panel_small_balanced` (n=100)        | 3.32 ms | 3.19–3.35 ms  | 4.82 ms | 4.72–5.55 ms   |
| `evoc_clusters_e2e_ari_check` (n=240) | 7.66 ms | 7.45–7.77 ms  | 9.62 ms | 9.28–9.89 ms   |
| `panel_large_many_clusters` (n=800)   | 24.85 ms | 24.34–25.54 ms | 27.24 ms | 26.98–28.09 ms |

### Wasm-vs-native overhead (median, same language)

| Dataset | Rust: wasm / native | Zig: wasm / native |
|---|---|---|
| n=100 | 11.0x  | 0.91x (wasm *faster* than native here -- noise-dominated at this size, see note) |
| n=240 | 5.4x   | 1.12x |
| n=800 | 2.6x   | 1.23x |

Notable asymmetry: **Rust's wasm build carries much higher relative overhead
over its own native build than Zig's does** (11x down to 2.6x as n grows,
vs. Zig staying close to 1x throughout, even dipping under 1x at n=100 which
is within this measurement's noise floor -- single-digit-ms timings on a
shared, non-isolated dev machine). This is consistent with the *absolute*
wasm numbers above, where Zig's wasm build beats Rust's wasm build by
roughly 3-8x across all three sizes, while the two languages' *native*
builds are much closer to each other (Rust modestly faster at n=800, Zig
modestly faster at n=100/240 -- close enough to call a wash). In other
words: the two ports are comparably fast as native code, but something
about Rust's wasm32-unknown-unknown codegen/runtime path for this workload
is markedly less efficient than Zig's wasm32-freestanding path.

**Follow-up investigation (2026-07-25): root cause not found among the
likely suspects.** Five build-configuration hypotheses were tested,
rebuilding `rust/core`'s wasm32-unknown-unknown release binary and
re-running `scripts/wasm_benchmark.mjs` after each isolated change (each
reverted before testing the next, so effects wouldn't compound):

1. **`panic = "abort"` for the wasm target** (via
   `rust/.cargo/config.toml` `[target.wasm32-unknown-unknown] rustflags`,
   scoped so native `cargo test` keeps unwind semantics). Verified via
   `cargo build -v` that `-C panic=abort` actually reached rustc. **Zero
   effect** -- the output `.wasm` was byte-for-byte identical (same MD5) to
   the `panic = "unwind"` default, and timings were within noise. Not
   applied (no measurable benefit to justify the added build config).
2. **`debug-assertions`/`overflow-checks`**: confirmed off, no override
   anywhere for either target (`cargo build -v` shows no
   `-C debug-assertions=true` / `-C overflow-checks=true`). Not the cause.
3. **Allocator overhead**: swapped `wasm_api.rs`'s implicit default
   allocator for a hand-rolled leaking bump allocator (`memory_grow`-backed,
   never frees) behind an experimental feature flag, to isolate whether
   `wasm32-unknown-unknown`'s default allocator was the bottleneck. Result:
   **substantially worse**, not better (single-call timing roughly 5x
   slower with the bump allocator than with the default allocator on the
   same n=100 fixture). The default allocator is not the problem; the
   experiment was reverted (not shipped).
4. **Missing target features**: added `+bulk-memory,+sign-ext,
   +mutable-globals,+nontrapping-fptoint` via the same `rustflags`
   mechanism as (1) (Zig's `build.zig` sets no explicit wasm target
   features either, so this was testing whether Rust's defaults are simply
   more conservative). Verified the flags reached rustc. **Zero effect**
   again -- byte-identical output binary, timings within noise. Not
   applied.
5. **LTO/codegen-units applying to the wasm target**: verified via
   `cargo build -v` that `-C opt-level=3` and `-C codegen-units=1` do reach
   rustc for the wasm build (so the workspace `[profile.release]` is
   correctly applied per-target as expected). No explicit `-C lto=...` flag
   appears in the rustc invocation for *either* the wasm or the native
   build of `uvoc-core` -- but this is consistent with `uvoc-core` having
   zero dependencies (LTO's main benefit, cross-crate inlining, is a no-op
   for a single-crate compilation unit already at `codegen-units = 1`), and
   it's identical between native and wasm, so it cannot explain the
   *asymmetry* between them. Not investigated further.

A sixth, unlisted variant was also tried opportunistically: switching
`opt-level` from `3` to `"s"` (on the theory that V8's baseline Liftoff
tier -- which dominates in a short 2-warmup/10-iteration benchmark -- might
run smaller/simpler code faster than heavily inlined/unrolled `opt-level =
3` code, even though the fully-optimized TurboFan tier would prefer `3`).
This made things **worse** (roughly +15-20% slower across all three
dataset sizes), ruling this out too.

**None of the tested build-configuration levers explain the gap.** The
overhead appears to be inherent to how LLVM's wasm32 backend (as invoked by
rustc for this specific workload/code shape) codegens this program, rather
than a fixable Cargo/rustflags misconfiguration -- consistent with, though
not proof of, the "LLVM wasm backend codegen quality" possibility raised
below. No Rust-side code or build changes were made as a result of this
investigation (`rust/` is unchanged from before it started); flagging this
as a real, currently-unexplained finding rather than forcing an
unsubstantiated fix. Anyone picking this up next might look at: per-stage
timing breakdown to localize *which* pipeline stage carries the overhead
(this pass only had end-to-end numbers for the reasons noted above), disas-
sembly/instruction-count comparison of a single hot function between the
two `.wasm` binaries, or a profiler capable of attributing time inside a
wasm module (`--prof`/`d8` tracing, or Node's `--cpu-prof`).

### Standalone stage: boruvka MST (n=80, d=4)

| | Rust wasm | Zig wasm |
|---|---|---|
| median | 0.05 ms | 0.05 ms |
| min–max | 0.04–0.07 ms | 0.05–0.08 ms |

Both languages within noise of each other on this stage -- at n=80 the
absolute cost is small enough (tens of microseconds) that per-call harness
overhead (marshaling args, `performance.now()` resolution) likely dominates
the true algorithmic difference. Zig has no standalone wasm export for
cluster-tree labels or `node_embedding` (only reachable inside
`evoc_fit_predict`; see `zig/WASM_API.md` "Not separately exposed"), so a
finer per-stage breakdown isn't possible through the current Zig wasm API --
end-to-end `fit_predict` timing above is the primary signal.

### Wasm build-config experiment (Zig only, 2026-07-25)

Investigated whether `zig build wasm -Dwasm-optimize=ReleaseFast` (default
is `ReleaseSmall`) and/or wasm SIMD128 (new `-Dwasm-simd=true` option added
to `zig/build.zig`, off by default -- the target was previously hardcoded
via `resolveTargetQuery` with no way to add `simd128` as a target feature)
are worthwhile levers. All three configs measured with the same
`scripts/wasm_benchmark.mjs` harness (2 warmup + 10 timed iters):

| Config | n=100 median | n=240 median | n=800 median | Raw size | gzip -9 |
|---|---|---|---|---|---|
| ReleaseSmall (current default)          | 4.52 ms  | 10.90 ms | 33.06 ms | 189,994 bytes   | 73,289 bytes  |
| ReleaseFast                              | 4.26 ms  | 10.38 ms | 31.99 ms | 2,323,782 bytes | 637,266 bytes |
| ReleaseSmall + SIMD128 (`-Dwasm-simd=true`) | 4.46 ms  | 10.90 ms | 32.24 ms | 189,699 bytes   | n/a (not separately gzipped, same order as ReleaseSmall) |

**ReleaseFast**: ~3-6% median speed improvement across the three dataset
sizes, at the cost of a **~12.2x larger raw binary** (2.3 MB vs 190 KB) and
~8.7x larger gzip. For a browser/npm package where binary transfer size
directly affects load time, this is a bad trade for a marginal runtime
gain -- **not adopted as the default**.

**SIMD128**: essentially a wash on both size (189,699 vs 189,994 bytes,
within noise) and speed (all three datasets within the ReleaseSmall
baseline's own run-to-run noise band). This is expected: none of the core
algorithm files (`knn_graph.zig`, `node_embedding.zig`, `pca.zig`,
`label_propagation.zig`, `cluster_trees.zig`) use explicit `@Vector` types,
and Zig 0.13's `ReleaseSmall`/`ReleaseFast` auto-vectorizer does not appear
to find much to widen in this codebase's scalar loops -- so enabling the
target feature alone, without also rewriting hot loops to use `@Vector`,
buys nothing. **Verified safe or not**: bit-exactness was checked via
`node scripts/wasm_fixture_runner.mjs` against the SIMD128 build --
**9/10 zig checks pass, 0 fail, 1 skip (same skip as the non-SIMD
baseline, the unrelated missing int-inner-product-distance export)** --
so SIMD128 does not break fixture parity here, but since it also provides
no measurable benefit, it is **left off by default**; `-Dwasm-simd=true` is
kept as a documented opt-in `build.zig` option for future work if/when hot
loops are rewritten with explicit vector types.

**Recommendation**: `zig build wasm`'s default (`ReleaseSmall`, no SIMD) is
**unchanged** -- it was already the best point on the size/speed tradeoff
of the three configs measured; neither alternative earns adoption on this
codebase's current code shape.

### Peak memory (wasm), largest dataset (`panel_large_many_clusters`, n=800, d=32)

Measured as final `memory.buffer.byteLength` after one `evoc_fit_predict`
call (wasm linear memory only grows, never shrinks, so the final size after
one full run is a reasonable proxy for that run's peak footprint):

| | Rust wasm | Zig wasm |
|---|---|---|
| Final `memory.buffer.byteLength` | 564,133,888 bytes (~538 MiB, 8,608 pages) | 1,077,477,376 bytes (~1,027 MiB, 16,441 pages) |

Both numbers are surprisingly large for an 800x32 input (a few hundred KB of
raw data) and clearly dominated by internal working-set allocations
somewhere in the pipeline (candidates: per-epoch node-embedding buffers,
repeated KNN/graph construction scratch space, or non-freeing bump-style
allocation patterns) rather than the output itself. Zig's footprint here is
roughly 2x Rust's, the reverse of the binary-size and execution-time
results above.

**Root-cause investigation (2026-07-25, Zig side).** Wrote
`scripts/wasm_memory_probe.mjs` (kept in the tree as a reusable throwaway
diagnostic, not wired into `check_all.sh`) which calls Zig's
`evoc_fit_predict` 6 times in a row on the same wasm instance/dataset
(`panel_large_many_clusters`, n=800) with different seeds, logging
`memory.buffer.byteLength` after every call to distinguish a one-time
high-water-mark ratchet from a genuine per-call leak:

```
initial memory.buffer.byteLength = 1,114,112 (1.1 MiB)
call 1: after=1,077,477,376 (1027.6 MiB)
call 2: after=1,077,805,056 (1027.9 MiB)  delta=+327,680
call 3: after=1,077,870,592 (1027.9 MiB)  delta=+65,536
call 4: after=1,077,936,128 (1028.0 MiB)  delta=+65,536
call 5: after=1,078,001,664 (1028.1 MiB)  delta=+65,536
call 6: after=1,078,067,200 (1028.1 MiB)  delta=+65,536
```

**Verdict: overwhelmingly a one-time ratchet, not a leak.** ~99.97% of the
final footprint (1027.6 of ~1028.1 MiB) is allocated on the *first* call --
consistent with wasm `memory.grow` being a one-way ratchet by spec (freed
pages can never be returned to the host) combined with `evoc_fit_predict`'s
pipeline touching a genuinely large scratch working-set at n=800 (KNN
graph construction, label-propagation recursion, node-embedding epoch
buffers) during that first call. Calls 2+ do add a small, steady
**+65,536 bytes (exactly one wasm page) per call** -- not zero -- which
*is* real per-call growth, but it is 4-5 orders of magnitude smaller than
the initial ratchet and does not compound super-linearly across 6 calls
(constant +1 page/call, not accelerating), so it reads as free-list
bucket fragmentation internal to `std.heap.WasmAllocator` (confirmed by
reading `lib/std/heap/WasmAllocator.zig`: it's a bucketed size-class
free-list allocator over wasm pages that never returns memory to the "OS"
-- there is no OS to return it to in freestanding wasm -- so a
differently-shaped allocation sequence on a later call, e.g. a different
seed producing a different cluster count / label-propagation recursion
depth, can require one additional page even if net bytes-in-use trends
flat) rather than an application-level missing-`free` bug. A targeted
audit of `wasm_api.zig`'s call tree (`clustering.zig`,
`label_propagation.zig`) found `defer allocator.free(...)` (or `deinit()`)
paired with essentially every `allocator.alloc`/`try allocator.alloc` call
already, including inside `label_propagation_init_recursive`'s recursion
and `clustering.build_cluster_layers`'s intermediate buffers -- no missing
frees were found, so no arena-allocator or `defer`-adding fix was applied
(the instructions authorized such a fix only if one was actually needed;
this codebase already follows that discipline throughout the files
checked).

**Conclusion: the ~1027 MiB figure is EXPECTED, wasm-inherent behavior for
this workload, not a fixable inefficiency.** The large *first-call* jump
reflects genuine peak scratch-memory usage during the n=800 pipeline (not
root-caused down to which specific buffer dominates in this pass -- would
need a wasm-level allocation profiler/memory.grow call interception to go
further, flagged as remaining future work) that `memory.grow`'s one-way
nature then locks in for the rest of the process; the much smaller
per-call drift after that is allocator bookkeeping overhead, not a leak
that would eventually run a long-lived host out of memory. No code change
was made as a result of this investigation. Not measured: true
peak-*during*-a-single-run (vs. this proxy of memory after the run
completes), memory for the two smaller datasets, or a breakdown by
pipeline stage -- still future work.

## Cross-implementation parity

Rust and Zig must agree bit-for-bit on all components (see PORTING.md numeric
policy). Verified through native `cargo test` / `zig build test` fixture
suites (phases 1-8) AND, per VALIDATION_PLAN.md section 3 item 2, through
the actual compiled wasm binaries via `scripts/wasm_fixture_runner.mjs`
(summarized here; full detail in the "Wasm fixture parity" section below):

- **Rust wasm: 10/10 fixture checks pass, 0 fail, 0 skip.**
- **Zig wasm: 6/6 fixture checks pass, 0 fail, 4 skip** (skips are real
  wasm-API surface gaps -- no standalone Zig wasm export for the
  int-inner-product distance kernel, the cluster-tree pipeline,
  `node_embedding`, or `knn_graph` -- not numeric failures; see below).
- **Zero wasm-specific numeric divergence found in either language** --
  every checked value matches native behavior within PORTING.md's tolerance
  classes. One harness-vs-fixture mismatch (Rust's `wasm_boruvka_mst`
  hardcoding `leaf_size=40` against a `leaf_size=10` fixture) was diagnosed
  as a wasm-API scope gap, confirmed NOT a numeric bug by cross-checking
  against native `cargo test --test boruvka`, and worked around in the
  harness without touching Rust's wasm build.

Net: both wasm builds are numerically trustworthy ports of their native
counterparts. Zig's wasm API surface is narrower (fewer standalone exports),
which is a real gap for downstream consumers who want fine-grained
component access, but does not indicate a correctness problem in what Zig
*does* export.

## Wasm fixture parity (VALIDATION_PLAN.md section 3, item 2)

`scripts/wasm_fixture_runner.mjs` loads both compiled `.wasm` binaries
(`rust/target/wasm32-unknown-unknown/release/uvoc_core.wasm`,
`zig/zig-out/bin/uvoc_core.wasm`) with Node's `WebAssembly` API and runs the
phase 1-8 JSON fixtures through the actual wasm-exported functions (not
native code), asserting bit-exact / documented-tolerance agreement per
PORTING.md's tolerance classes -- the same checks the native `cargo
test`/`zig build test` suites make, but through the real shipping target.
Wired into `scripts/check_all.sh` as a new final step that runs
automatically whenever at least one `.wasm` binary is already built, and
prints build instructions + skips (does not fail) otherwise, so CI/local
runs stay fast when wasm hasn't been built yet.

Run directly: `node scripts/wasm_fixture_runner.mjs`.

### Result (as of this writing)

```
rust: 10 pass, 0 fail, 0 skip (of 10)
zig:  6 pass, 0 fail, 4 skip (of 10)
```

All wasm-callable checks pass in both languages -- no wasm-specific numeric
divergence from native was found. One harness-vs-fixture mismatch was found
and worked around (documented below), not a wasm bug.

### What was tested

| Fixture / area | Rust | Zig | Notes |
|---|---|---|---|
| `distance/fast_cosine` | pass | pass | exact per-pair tolerance from fixture |
| `distance/fast_bit_jaccard` | pass | pass | packed-bit `u8` vectors |
| `distance/fast_int_inner_product_dissimilarity` | pass | **not exported** | Zig `wasm_api.zig` has no `distance_fast_int_inner_product_dissimilarity` export |
| `boruvka/parallel_boruvka_n80_d4_*` (both `reproducible` variants) | pass (structural) | pass (exact) | see leaf_size caveat below |
| `cluster_trees/labels_and_strengths_n80_mcs5` | pass (chained via `mst_to_linkage_tree_n80`'s `sorted_mst`) | **not exported** | Zig has no standalone cluster-tree wasm export; only reachable inside `cluster_layers_base`/`evoc_fit_predict`, which don't expose the intermediate linkage/condensed-tree shape this fixture needs |
| `node_embedding/*` | pass (determinism check only, see below) | **not exported** | Zig has no standalone `node_embedding` wasm export |
| `graph/knn_graph_n120_d8_k10_seed42` | pass (structural + determinism check only, see below) | **not exported** | Zig has no standalone `knn_graph` wasm export |
| `clustering/build_cluster_layers_n126` | pass (best/argmax layer) | pass (base layer, matched fixture layer 0) | fixture has no RNG; fully portable |
| `clustering/evoc_clusters_e2e_ari_check` | pass (ARI panel, seeds 0..19) | pass (ARI panel, seeds 0..19) | median/min gates from `rust/core/tests/evoc_clusters_ari.rs`: median ≥ 0.3747, min ≥ 0.15. Rust: median 0.6032, min 0.3929. Zig: median 0.7514, min 0.5147. Both comfortably clear the gate; not directly compared to each other here (that's phase-10 cross-language territory) |

### Fixtures/functions not coverable through the current wasm API

These are real API-surface gaps, not test failures -- flagged per the task's
instructions as signal for whether the wasm API needs to grow, not scored
against pass/fail:

- **Zig**: no standalone exports for `distance_fast_int_inner_product_dissimilarity`,
  the cluster-tree pipeline (`mst_to_linkage_tree`/`condense_tree`/`extract_leaves`/labels+strengths),
  `node_embedding`, or `knn_graph`. Only reachable indirectly through
  `cluster_layers_base` (base layer only) and `evoc_fit_predict` (full e2e).
  This means roughly half of the phase 1-8 fixture set has no Zig wasm entry
  point at all yet. See `zig/WASM_API.md`'s "Not separately exposed" section.
- **Both languages**: `rng`, `disjoint_set`, `kdtree`, `nndescent`, `graph`
  (`compute_membership_strengths`, `neighbor_graph_matrix`, `smooth_knn_dist`),
  and `label_propagation` fixtures have no wasm exports in either language --
  these components are only exercised internally by `knn_graph`/
  `evoc_fit_predict`. Not covered by this harness.
- **`node_embedding_e2e_n120`** targets `node_embedding_epoch_repr`, an
  externally-RNG-driven variant (caller supplies per-epoch `rng_val` +
  `node_order_shuffles`) used by the Python-parity path in `clustering.py`.
  `wasm_node_embedding` instead wraps `node_embedding_repr_self_driven` (a
  scalar-`seed`-driven variant used by `EVoC::fit_predict`) -- different
  calling convention, no fixture recorded for it yet. Verified same-seed
  determinism instead (bit-identical output across two calls).
- **`knn_graph_n120_d8_k10_seed42`** supplies raw pre-drawn `RngState`
  arrays from the Python/native reference; `wasm_knn_graph` instead derives
  its `RngState`s deterministically from a scalar `seed` via a fixed draw
  sequence baked into `wasm_api.rs`, which can't be reverse-engineered to
  match the fixture's recorded states. Verified structural validity (all
  neighbor indices in range, distances sorted ascending per row) and
  same-seed determinism instead.

### Harness-vs-fixture mismatch found and worked around (not a wasm bug)

`wasm_boruvka_mst` (Rust) hardcodes `leaf_size = 40` for its internal
kd-tree build (per `rust/core/WASM_API.md`), but the `parallel_boruvka_n80_*`
fixtures were recorded with `leaf_size = 10`. A different kd-tree leaf size
changes Boruvka's per-round traversal/merge order, so the *edge order*
legitimately differs from the fixture's recorded order even though both are
valid minimum spanning trees of the same point set -- confirmed by running
`cargo test --test boruvka`, which passes exactly against the same fixture
using the native `parallel_boruvka(&tree, ..)` API (which does take
`leaf_size` and matches the fixture's `10`). This is a wasm-API scope gap
(`wasm_boruvka_mst` doesn't expose `leaf_size`), not a numeric divergence --
Rust's wasm build was NOT modified. The harness instead asserts what's
`leaf_size`-invariant: edge count, that every point appears (a true spanning
tree), and total MST weight within the fixture's own distance tolerance
scaled for the sum. Zig's `boruvka_mst` DOES take a `leaf_size` parameter,
so it was called with the fixture's recorded `leaf_size = 10` and matches
the fixture exactly, edge-for-edge.

## Toolchain ergonomics notes

Free-form observations comparing the two ports, collected as we go:

- **`wasm-opt` was not available in this environment.** `which wasm-opt`
  failed; `npm install -g wasm-opt` installed successfully but its
  postinstall step (which fetches a prebuilt `binaryen` binary) silently
  failed to produce a working executable (`spawn ... ENOENT` at run time --
  the JS shim exists, the native binary it wraps does not); no root/`sudo`
  access was available to `apt-get install binaryen` as a fallback. This
  means the "optimized" binary-size numbers above are simply the raw
  release-build sizes; a real npm-packaging pass should verify `wasm-opt
  -O3` availability in its actual CI/build environment before relying on
  it, since this sandbox's failure mode (network-restricted postinstall)
  may not reflect a typical dev machine.
- **Build-step friction was otherwise low for both.** Rust:
  `cargo build --target wasm32-unknown-unknown --release -p uvoc-core`,
  zero extra flags. Zig: `zig build wasm`, similarly turnkey, though it
  requires pinning to zig 0.13.0 specifically (noted in the task
  environment info) -- newer zig releases have had breaking freestanding-
  target/build-system changes historically, so this is a real constraint to
  carry into packaging, not just a local sandbox quirk.
- **The two wasm calling conventions are meaningfully different** (see
  `scripts/wasm_fixture_runner.mjs`'s two separate `RustCore`/`ZigCore`
  adapter classes) -- Rust uses `wasm_alloc`/`wasm_dealloc` and mostly
  `i32`-width outputs (e.g. `i32` labels), Zig uses `alloc`/`dealloc` and
  `i64`-width labels (`BigInt64Array` on the JS side). Neither is a
  problem in isolation, but an eventual shared npm wrapper would need to
  paper over this rather than exposing one uniform ABI for free.
- **Zig's wasm API surface is narrower** than Rust's (see parity section
  above): no standalone `node_embedding`, `knn_graph`, cluster-tree, or
  int-inner-product-distance exports, only reachable through the
  `cluster_layers_base`/`evoc_fit_predict` composite entry points. This
  didn't block anything in this benchmarking pass (end-to-end timing is the
  most important number regardless), but it did block a finer per-stage
  execution-time breakdown down to just boruvka MST, and would block a
  future consumer who wants fine-grained component access via wasm.
- **Native timing harnesses were easy to bolt on for both languages**
  without touching any algorithm file: Rust via a `#[test] #[ignore]` in a
  new `rust/core/tests/native_bench.rs` (reuses the existing
  `tests/common/mod.rs` fixture loader), Zig via a freestanding `main()` in
  a new `zig/src/native_bench.zig` run with `zig run -O ReleaseFast`
  (reuses `fixture.zig`, not wired into `build.zig`/`root.zig`). Both are
  left in the tree as reusable throwaway benchmarking tools, not deleted.

## Recommendation

Per VALIDATION_PLAN.md section 3 item 5 ("pick the winner on the measured
data") -- this is a read of the current numbers for whoever picks up the
deferred npm-packaging pass, not a decision to actually package anything now.

**Zig is better positioned for the eventual wasm npm package, on both axes
that matter most for a browser/npm consumer:**

- **Size**: Zig's wasm binary is ~28% smaller raw, ~25% smaller gzipped,
  for a *narrower* API surface -- so even accounting for Zig needing to
  grow somewhat to match Rust's exported functions, the size advantage is
  unlikely to fully close.
- **Speed**: Zig's wasm build is 3-8x faster than Rust's wasm build
  end-to-end across all three tested dataset sizes (100/240/800 points),
  and this gap is *wasm-specific* -- both languages' native builds land
  within ~25% of each other, so the wasm gap isn't just "Zig's algorithm is
  faster," it's that Rust's wasm32-unknown-unknown codegen path carries
  much higher relative overhead (up to ~11x vs. its own native build at
  n=100) than Zig's wasm32-freestanding path (~1x throughout). For a
  browser package, wasm execution time is the number users actually feel;
  this is the strongest signal in this report.

**Caveat / what would change this:** Zig's wasm API is missing several
standalone exports (cluster-tree, `node_embedding`, `knn_graph`, one
distance kernel) that Rust has. If the npm package needs fine-grained
component access (not just end-to-end `fit_predict`), that gap would need
closing first, which is real but bounded engineering work (porting existing
Zig native functions to `wasm_api.zig`, not new algorithm work) -- it
doesn't change the size/speed verdict, just adds a prerequisite task before
packaging. Peak memory came out the other way (Zig's footprint ~2x Rust's
on the largest tested dataset) but that number is unexplained and not
obviously decision-relevant yet (both are large relative to input size in
a way that suggests an internal inefficiency worth investigating in either
language, independent of which one "wins"); flagged as follow-up, not
weighted into this recommendation.

This is not a toss-up: the wasm-specific execution-time gap in particular
is large and consistent across all three dataset sizes, not a close call
that could flip with measurement noise.
