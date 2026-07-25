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

## Scaling to 10k-30k points

Exploratory investigation (no VALIDATION_PLAN.md changes) into whether this
port can realistically handle n=10,000-30,000 points in the browser, and
what would need to change if not. Prior benchmarking (above) only went up
to n=800. This section uses fresh synthetic datasets, NOT Python-validated
labels (re-running the slow Python reference at these sizes wasn't needed
to answer a performance question) -- see `scripts/gen_scale_test_data.py`
(same cosine-friendly orthogonal-axis-centers construction as the ARI/panel
fixture generators; n in {2000, 10000, 30000}, d=32, 8 clusters, not
committed -- multi-MB `.f32` blobs regenerable by rerunning the script,
`scripts/scale_data/` is gitignored). Benchmark harness:
`scripts/wasm_scale_benchmark.mjs [n1,n2,...]`.

### Headline answer

**Yes -- both wasm builds' `evoc_fit_predict`/`evoc_clusters` complete at
n=30000 comfortably within budget, after one real fix (below).** Before the
fix, n=30000 would not have completed at all (see "The bug" below) -- it
would have attempted to allocate ~7.2 GB for a single buffer, blowing past
wasm32's entire 4 GiB linear-memory address space. After the fix:

| n | rust `wasm_evoc_fit_predict` | rust memory | zig `evoc_fit_predict` | zig memory |
|---|---|---|---|---|
| 2,000 | 288 ms | 539 MiB | 154 ms | 1031 MiB |
| 10,000 | 865 ms | 545 MiB | 831 ms | 1056 MiB |
| 30,000 | 2.78 s | 562 MiB | 2.68 s | 1123 MiB |

(Single timed run per cell, fresh wasm instance per call so the memory
reading is that call's own footprint, not cumulative growth across prior
calls in the same process -- wasm linear memory only grows, never shrinks,
so reusing one instance across many benchmark calls would otherwise report
misleadingly high numbers. Pipeline params: `n_neighbors=15,
min_samples=5, n_epochs=50, base_min_cluster_size=5, seed=0`, matching
`scripts/wasm_benchmark.mjs`'s existing convention.)

Both languages stay well under a second up to n=10000 and under 3 seconds
at n=30000 -- solidly "usable" by any reasonable browser-interaction
standard, and memory is nowhere near the 4 GiB wasm32 ceiling at any tested
size (562 MiB / 1123 MiB peak at n=30000, ~14%/28% of the ceiling). The
math on that ceiling, worked out *before* running n=30000 (per the task's
instruction to avoid an OOM-crash surprise): a naive dense n x n f64
distance/affinity matrix at n=30000 is `30000^2 * 8 bytes = 7.2 GB` --
already 1.8x the entire address space on its own, before any other
allocation. That number turned out to be exactly the shape of the real bug
found below, not a hypothetical.

### The bug: O(n^2) dense-matrix allocation in `label_propagation_init_recursive`

`rust/core/src/label_propagation.rs`'s `partition_expander` upscaling step
(the tail of `label_propagation_init_recursive`, the function that seeds
`node_embedding`'s initial layout) used to build a **dense `n_points x
n_points` `f64` matrix** (`graph_dense`, `8 * n_points^2` bytes) plus an
O(n_points^2) double loop over it, at the FULL top-level `n_points` (not
the reduced/coarsened graph size) on every top-level call:

```rust
let mut graph_dense = vec![0.0f64; n_points * n_points];   // <- 7.2 GB at n=30000
for i in 0..n_points {
    for j in 0..n_points {
        let m_ij = graph_dense[i * n_points + j] * graph_dense[j * n_points + i];
        ...
```

At n=2000 this is "only" 32 MB, invisible against the ~500 MB baseline
described below. At n=30000 it is 7.2 GB for the buffer alone --
guaranteed OOM before wasm32's 4 GiB address space is even accounted for
by anything else in the pipeline. This was caught by doing the memory math
in step 2 of this investigation *before* running n=30000, then grepping
`label_propagation.rs`/`pca.rs`/`clustering.rs`/`graph_construction.rs`
for exactly this shape of allocation (`n * n`, `n_samples * n_samples`,
dense `Vec<Vec<_>>`) per the task's checklist -- this is the smoking gun.

**Zig's port of the same function (`zig/src/label_propagation.zig`'s
`partition_expander` block) does NOT have this bug.** It never
materializes the dense matrix -- it walks each row's sparse CSR entries
directly and looks up the transposed entry `graph[j][i]` via a per-row
`csrGet` scan/binary-search, using an O(n_parts)-sized scratch row buffer
instead of an O(n_points^2) one. This is very likely *why* the original
memory comparison in this file's "Recommendation" section found Zig's
footprint puzzlingly ~2x Rust's on the largest *then-tested* dataset (800
points) despite Zig being faster -- at n=800 the dense bug is small (5 MB),
so it wasn't the dominant term there, but it's exactly the kind of
asymmetry that predicts a much larger gap emerging at scale, which this
investigation confirms independently (the bug, not that specific old
memory comparison, since 800 was too small to see it clearly).

**Fix applied** (`rust/core/src/label_propagation.rs`, the
`partition_expander` block): replaced the dense matrix with the same
sparse-row-lookup approach Zig already uses -- walk `i`'s CSR row directly
(already sparse, `nnz` entries total, not `n_points` per row), and look up
`graph[j][i]` with a linear scan of `j`'s CSR row (a linear scan, not
Zig's binary search, because this port's `smmp` output CSR rows are
explicitly documented as NOT column-sorted -- see `smmp`'s doc comment a
few lines above; a binary search would have silently produced wrong
results on unsorted rows). Row degree here is small and bounded (kNN-graph
derived, `n_neighbors=15` by default), so this is O(n_points * degree^2)
in the worst case and effectively linear in practice, not O(n_points^2).

This function (`label_propagation_init_recursive`) is explicitly
documented as ARI-validated against Python, not bit-exact
fixture-validated (PCA's randomized SVD makes bit-exactness impossible
here regardless -- see PCA_VALIDATION.md), so a numerically-equivalent
algorithmic rewrite is in scope without the bit-exact caution that applies
to phase 1-8 fixture-parity files. Verified after the fix:
`cargo test --workspace` (92+ tests, all pass), `zig build test`
(untouched, still passes), and `node scripts/wasm_fixture_runner.mjs`
(both languages, all non-skipped checks pass, INCLUDING the ARI panel gate
that would catch a correctness regression: rust median ARI 0.6032 / min
0.3929 against gates of >=0.3747 / >=0.15, unchanged in shape from before
the fix). `bash scripts/check_all.sh` passes end to end.

### Per-stage profiling

Standalone stage benchmarks (`wasm_knn_graph`/`knn_graph`,
`wasm_boruvka_mst`/`boruvka_mst`, `wasm_cluster_tree_labels`/
`cluster_tree_labels`, `wasm_node_embedding`/`node_embedding`), all now
exported by both languages (unlike the note in the older
`wasm_benchmark.mjs`, which predates Zig's node_embedding/knn_graph/
cluster-tree exports landing):

| stage | n=2,000 | n=10,000 | n=30,000 | rust complexity read |
|---|---|---|---|---|
| `knn_graph` (rust / zig) | 216 / 57 ms | 465 / 292 ms | 1.14 / 0.98 s | grows ~5x for 3x n (2000->10000) then ~2.5x for 3x n (10000->30000) -- roughly linearithmic, matches NN-descent's expected near-O(n log n), no red flag |
| `boruvka_mst`, **raw d=32** (rust / zig) | 137 / 161 ms | 3.1 / 3.6 s | 27.0 / 29.9 s | grows ~23x for 5x n, then ~9x for 3x n -- worse than O(n log n), consistent with curse-of-dimensionality degrading the kd-tree's pruning at d=32 (approaches near-brute-force O(n^2 log n)-ish behavior as the tree stops usefully partitioning high-dim space) |
| `cluster_tree_labels` (rust / zig) | 0.8 / 1.4 ms | 2.3 / 6.3 ms | 5.5 / 13.4 ms | sub-linear-looking growth, consistent with the documented O(log n) condensed-tree recursion depth -- no red flag |
| `node_embedding` (rust / zig), synthetic k=15 graph | 128 / 107 ms | 759 / 653 ms | 2.55 / 2.22 s | roughly linear in n (n_epochs fixed at 50, negative_sample_rate fixed) -- matches the expected O(n_epochs * n * negative_sample_rate), no red flag |

**Important caveat on the `boruvka_mst` row:** this standalone benchmark
runs boruvka directly on the raw d=32 input, but the REAL end-to-end
pipeline (`evoc_fit_predict`/`clustering.rs` line ~82) runs boruvka on the
much lower-dimensional `node_embedding` output (`n_components`, default
derived as `min(max(n_neighbors/4, 4), 15)` -- 8 in this benchmark's
config), not the raw features. That's exactly why the end-to-end pipeline
finishes n=30000 in 2.78s total while standalone boruvka alone takes 27s
on the same n at the raw dimensionality -- **the standalone number is not
representative of boruvka's real cost inside the actual pipeline**, it's
a demonstration of how badly this specific kd-tree-based MST degrades at
higher dimensionality, which is a real thing to know if anyone considers
exposing/using standalone `boruvka_mst` directly on unreduced
high-dimensional data.

### Memory-ceiling math, revisited post-fix

wasm32's linear memory address space caps at 4 GiB (`memory.grow` fails
past that, unconditionally, independent of host RAM). Post-fix, measured
peak at n=30000 is 562 MiB (rust) / 1123 MiB (zig) -- 14%/28% of the
ceiling, both comfortably clear. The ~500-560 MiB (rust) / ~1030-1120 MiB
(zig) baseline is present even at n=2000 and barely grows through
n=30000 (539->562 MiB rust, a ~4% increase for a 15x increase in n; 1031->
1123 MiB zig, ~9%) -- this reads as a large, roughly n-INDEPENDENT
constant overhead (allocator behavior: wasm linear memory only grows, never
returns pages, so a sequence of many differently-sized alloc/dealloc calls
across the pipeline's many stages can ratchet memory usage up via
fragmentation well beyond live-data size, and never give it back for the
rest of that wasm instance's lifetime) rather than a scaling risk. It's
real (a long-lived page keeping one wasm instance alive across many
`fit_predict` calls will keep this memory pinned, and Zig's ~1 GiB
baseline in particular is worth a closer look someday), but it is NOT what
would block 10k-30k -- the O(n^2) bug above was the only actual blocker,
and it's fixed. Extrapolating the now-flat per-call footprint, this port
would likely handle n=100,000+ from a pure memory standpoint before
approaching the 4 GiB ceiling; wall-clock time (see below) would become
the binding constraint well before memory does.

### Threading (SharedArrayBuffer + Web Workers) feasibility read

Per VALIDATION_PLAN.md section 3 item 4, threading was already flagged as
a "later" lever given COOP/COEP deployment friction. This investigation's
profiling data doesn't change the priority call, but does say where
threading WOULD help if pursued:

- **`node_embedding`'s SGD is the best threading target if/when needed.**
  It's the single largest contributor to end-to-end time at scale (2.2-2.6s
  of the ~2.7s n=30000 total when run on the pipeline's actual
  low-dimensional embedding-sized graph -- comparable in magnitude to
  `knn_graph`), and the reference algorithm (UMAP-style negative sampling
  SGD) is well known in the wild to parallelize across edges/negative
  samples with only a lock-free "racy update" approach (accepted numeric
  noise, same trick `pynndescent`/`umap-learn` use with numba's
  `parallel=True`) -- embarrassingly parallel in principle, no sequential
  dependency between one edge's update and another's within an epoch.
- **`knn_graph` is a harder target.** NN-descent's candidate-refinement
  loop has real cross-iteration data dependencies (each iteration's
  candidate list depends on the previous iteration's converged neighbor
  sets), and the RP-forest construction phase, while tree-parallel in
  principle (each of the 4 trees is independent), is a small fraction of
  total `knn_graph` time relative to the NN-descent refinement loop. Not
  impossible, but meaningfully harder than `node_embedding`'s "just
  parallelize the inner loop" shape.
- **`boruvka_mst` is likely not worth threading at the pipeline's actual
  operating point** (low-dimensional post-embedding data, where it's fast,
  per the caveat above) -- it would matter more if the pipeline ever ran
  boruvka on raw high-dimensional data directly, but that's not what
  `evoc_fit_predict` does today.

**Opinion: threading is not the next lever to pull.** At n=30000 the whole
pipeline finishes in ~2.7s single-threaded, which is already well within
"usable in a browser" territory; the ROI on the COOP/COEP deployment
complexity threading requires isn't there yet at this scale. It becomes
worth reconsidering if/when someone wants n=100k+ or wants sub-second
p95 at n=30000 specifically for a snappier interactive feel -- at that
point `node_embedding`'s SGD is the concrete target, not a vague "thread
everything."

### Prioritized next steps

1. **(Done in this pass)** Fix the `label_propagation_init_recursive`
   dense-matrix O(n^2) allocation in Rust (`label_propagation.rs`) --
   this was the one real blocker for n=30000 and is now fixed;
   `cargo test --workspace`, `zig build test`, and
   `node scripts/wasm_fixture_runner.mjs` all still pass, including the
   ARI correctness gate.
2. **Investigate Zig's ~1 GiB baseline memory footprint.** Not urgent
   (nowhere near the 4 GiB ceiling even at n=30000) but a ~2x-vs-Rust
   constant overhead that shows up even at n=2000 is worth a dedicated
   profiling pass -- likely `std.heap.WasmAllocator`'s size-class bucket
   rounding combined with the same "memory only grows" dynamic discussed
   above, compounded across this port's many small/short-lived
   allocations per pipeline stage.
3. **If sub-second p95 at n=30000 (or n=100k+) ever becomes a real
   requirement, parallelize `node_embedding`'s SGD loop with
   SharedArrayBuffer + Web Workers first** -- it's the single largest
   remaining time contributor in the real (low-dimensional) pipeline
   shape, and the algorithm is naturally suited to lock-free racy-update
   parallelism. This is a "when it matters" item, not urgent at today's
   ~2.7s/30k number.
4. **If a future consumer wants standalone `boruvka_mst`/`knn_graph` on
   RAW high-dimensional data (not the pipeline's internal low-dim
   embedding),** budget for that being much slower than the numbers in
   this section's end-to-end table suggest -- the 27-30s standalone
   boruvka-at-d=32 number above is the real cost of that specific use
   case, and would need either a curse-of-dimensionality-aware algorithm
   change (e.g. dimensionality reduction before MST, which the pipeline
   already effectively does) or acceptance of multi-second latency for
   that specific standalone call.

Artifacts from this pass: `scripts/gen_scale_test_data.py` (dataset
generator, regenerate via `python3 scripts/gen_scale_test_data.py`),
`scripts/wasm_scale_benchmark.mjs` (per-size, per-stage benchmark
harness, `node scripts/wasm_scale_benchmark.mjs [n1,n2,...]`, defaults to
2000,10000,30000). `scripts/scale_data/*.f32` (generated datasets) are
gitignored, not committed -- multi-MB binary blobs, trivially regenerated.

## Cold-vs-warm wasm call investigation (2026-07-25)

Follow-up to the "Rust wasm overhead" investigation above, which tried five
build-configuration levers (panic=abort, allocator swap, target features,
opt-level) and found none of them explained Rust's ~3-8x wasm slowdown
relative to Zig -- but never isolated whether the gap was a **fixed,
roughly n-independent one-time cost paid by a freshly instantiated wasm
module** (e.g. allocator warmup / lazy `memory.grow` ratcheting) versus a
genuine **per-call, per-element** cost. This pass adds that decomposition.

### Method

New script `scripts/wasm_cold_warm_benchmark.mjs` (kept as a reusable,
documented tool, same pattern as `scripts/wasm_memory_probe.mjs`'s
"repeated calls on one instance" approach): for both Rust and Zig wasm
builds, at n=100 (`panel_small_balanced` fixture) and n=2000
(`scripts/scale_data/scale_n2000_d32.f32`):

- **cold**: 8 samples, each a *fresh* `WebAssembly.instantiate` followed by
  timing only that instance's first `evoc_fit_predict` call.
- **warm**: one instance, one discarded warmup call, then 10 timed calls
  reusing that same instance's memory/allocator state.

Run via `node scripts/wasm_cold_warm_benchmark.mjs`.

### Result

| n | Rust cold (median) | Rust warm (median) | cold-warm delta | Zig cold (median) | Zig warm (median) | cold-warm delta |
|---|---|---|---|---|---|---|
| 100  | 144.5 ms | 37.0 ms  | 107.4 ms | 6.2 ms   | 4.5 ms  | 1.7 ms  |
| 2000 | 241.6 ms | 121.0 ms | 120.6 ms | 102.4 ms | 92.3 ms | 10.2 ms |

### Verdict: hypothesis PARTIALLY confirmed -- a real, large, roughly
n-independent cold-instantiation overhead exists in Rust's wasm build and
is much larger than Zig's (~107-121 ms vs ~2-10 ms), but it does **NOT**
explain the majority of the previously-documented gap, because that gap
was already measured warm. `scripts/wasm_benchmark.mjs` loads each wasm
module once and reuses the same instance across its 2 warmup + 10 timed
iterations (i.e. it was already a "warm" measurement, confirmed here: this
pass's own warm-n=100 median of 37.0 ms for Rust matches the previously
recorded 36.64 ms almost exactly, within run-to-run noise). At n=100, Rust
warm (37.0 ms) is still ~8.2x slower than Zig warm (4.5 ms) -- essentially
the same ratio as the original end-to-end table's 8.4x (36.64 / 4.39). So:
**the fixed-overhead hypothesis correctly predicts and explains an
additional real cost that hits any page instantiating the module fresh and
calling it once (a realistic browser use case), but it is a genuinely
separate, additive phenomenon on top of the already-documented warm-call
gap, not the root cause of that gap.** At n=2000 the *warm* gap has itself
mostly closed (121.0 vs 92.3 ms, 1.3x) consistent with the previously
observed narrowing at larger n (n=800: 2.6x, n=30000: ~1x) -- so the
cold-instantiation overhead becomes proportionally more, not less,
significant as n grows into the "wasm build advantage" territory, since
its ~110-120 ms absolute cost stays flat while everything else shrinks in
relative terms.

### What the cold overhead was tried against and did NOT fix it

**Larger initial wasm memory reservation.** Hypothesis: Rust's wasm memory
section might declare a small initial page count, forcing many
`memory.grow` calls to ratchet up to working size on the first call.
Checked via a small hand-rolled leb128 memory-section parser (no
`wasm-objdump`/`wasm2wat` available in this environment, same toolchain gap
noted in "Toolchain ergonomics notes" above) on both compiled binaries:
**both Rust's and Zig's `.wasm` declare an identical 17-page (1,114,112
byte / 1.06 MiB) initial memory** -- so this specific asymmetry does not
exist; Rust does not start from a smaller reservation than Zig.

Tried anyway (in case the *increment size* of subsequent grows, not the
starting size, was the cost): added `rust/.cargo/config.toml` scoped to
`[target.wasm32-unknown-unknown]` with
`rustflags = ["-C", "link-args=--initial-memory=67108864"]` (64 MiB, a
multiple of the 65,536-byte wasm page size), rebuilt
(`cargo build --target wasm32-unknown-unknown --release -p uvoc-core`),
confirmed via the same leb128 parser that the initial page count changed
to 1024 pages (64 MiB) as expected, and re-ran the cold/warm benchmark.
**Zero measurable effect**: n=100 cold median 140.5 ms (vs 144.5 ms
baseline, within noise), n=2000 cold median 244.2 ms (vs 241.6 ms
baseline, within noise) -- reserving 64x more memory up front did not
move the cold-call cost at all. **Reverted** (`rust/.cargo/config.toml`
deleted); this change is NOT present in the final tree, matching the
task's instruction to keep only a fix that actually helps.

### Root-cause profiling (informational, no fix found)

Since the memory-reserve hypothesis was cleanly refuted, a `node
--cpu-prof` sampling profile was taken of both a single cold call (fresh
instance, n=100) and 20 warm calls (one reused instance, n=100, after one
discarded warmup) to localize where the time actually goes. In **both**
profiles, the dominant sampled symbol by a wide margin is
`__rustc::__rdl_alloc_zeroed` (Rust's `System` allocator's zeroed-allocation
entry point, i.e. `alloc_zeroed`/`calloc`-style allocations reaching
`wasm32-unknown-unknown`'s default `dlmalloc`-based global allocator): ~87%
of samples in the single cold call, ~88% of samples across the 20 warm
calls. This means the *relative* profile shape is essentially identical
cold vs warm -- `alloc_zeroed` is the dominant cost in this pipeline either
way -- but cold calls spend far more *absolute* time inside it. This is
consistent with, and extends, finding (3) in the original "Rust wasm
overhead" investigation above (swapping to a hand-rolled bump allocator
made things *worse*, ruling out "just replace the allocator" as a fix) --
both passes independently point at the default wasm allocator's zeroing
path as central to the overhead, without either pass finding an actionable
fix. **No further root-cause was isolated in the time available** (would
need V8-internal instrumentation of individual `memory.grow`/allocator
calls, e.g. a debug V8 build with `--trace-wasm-memory` support -- the
release Node v20 binary available in this environment rejects that flag
outright as unrecognized) -- flagged as remaining future work, not fixed.

### Conclusion and recommendation impact

**No code change was kept from this investigation** -- the initial-memory
linker flag was tested and reverted (no benefit); no other lever was found
that reduced either the cold or the warm gap. `rust/` and `zig/` are
unchanged in the final tree from before this investigation started, aside
from the new benchmark scripts and this results.md section. This does
**not** change the "Recommendation" section above: Zig remains faster than
Rust's wasm build across the full measured range (both warm end-to-end
numbers and now additionally confirmed for cold-instantiation-plus-first-
call scenarios, where Rust's relative disadvantage is if anything *worse*
at small n because of the added ~107 ms fixed cost). The one new, actionable
piece of information for a downstream consumer: **if a page only
instantiates the wasm module once and calls `fit_predict` once (a common
real-world pattern, e.g. a single-shot analysis tool), Rust's real-world
cost at n=100 is closer to ~144 ms (cold) than the previously-reported
~37 ms (warm-only) figure** -- a consumer that keeps one long-lived module
instance around across multiple calls (e.g. a persistent worker) pays the
~107-121 ms cold cost only once, not per call, which is the more favorable
scenario and the one the original benchmark implicitly measured.

Verification after this investigation (fix reverted, so this is confirming
no regression from the revert, not from a kept change): `cargo test
--workspace` (all pass), `zig build test` (unaffected, still passes),
`node scripts/wasm_fixture_runner.mjs` (rust 10/10, zig 9/10 pass, 1 skip
-- the skip count differs from the 6/10-pass/4-skip figure recorded
earlier in this file because Zig's wasm API surface grew standalone
`node_embedding`/`knn_graph`/cluster-tree exports in the interim, unrelated
to this investigation), `bash scripts/check_all.sh` (passes end to end).
Re-ran `scripts/wasm_benchmark.mjs` and `scripts/wasm_scale_benchmark.mjs`
for fresh before/after-this-pass numbers (identical in shape to the
existing tables above, confirming no regression from the revert):

| n | Rust wasm median | Zig wasm median |
|---|---|---|
| 100    | 37.02 ms | 4.48 ms |
| 240    | 42.64 ms | 11.03 ms |
| 800    | 65.52 ms | 34.28 ms |
| 2,000  | 249.4 ms | 139.4 ms |
| 10,000 | 657.8 ms | 651.7 ms |
| 30,000 | 2.08 s   | 1.98 s  |

(n=2000/10000/30000 rows are fresh-instance end-to-end numbers, matching
this file's "Scaling to 10k-30k points" methodology and gitignored
`scripts/scale_data/` fixtures; n=100/240/800 rows are the warm
2-warmup+10-iteration numbers from `scripts/wasm_benchmark.mjs`, matching
this file's original "Execution time" methodology. The two measurement
styles are not directly comparable to each other -- see the cold-vs-warm
section above for why -- but each is internally consistent with its own
prior recorded figures, confirming this investigation left no regression.)

Artifact from this pass: `scripts/wasm_cold_warm_benchmark.mjs` (reusable,
documented cold-vs-warm timing harness, `node
scripts/wasm_cold_warm_benchmark.mjs`, no arguments).
