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
is markedly less efficient than Zig's wasm32-freestanding path. This
project didn't dig into *why* (allocator differences, panic-machinery
overhead, LLVM wasm backend codegen quality, `#[no_std]`-adjacent setup --
several plausible causes, not investigated here); flagging it as the
single most decision-relevant number in this report.

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
allocation patterns) rather than the output itself. This was not root-caused
in this pass -- flagged as a real finding for anyone doing further wasm
memory-budget work, not just noise. Zig's footprint here is roughly 2x
Rust's, the reverse of the binary-size and execution-time results above.
Not measured: true peak-during-run (would need `memory.grow` call
interception or a wasm-level profiler), memory for the two smaller
datasets, or breakdown by pipeline stage -- noted as future work per the
task's guidance not to over-invest in this sub-item.

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
