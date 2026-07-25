# Rust vs Zig comparison

Updated per phase by `scripts/compare.sh` (phase 10) and by hand where noted.
All measurements on the shared fixtures; wasm builds are release/optimized,
run single-threaded under Node v20.

## Binary size (wasm32, optimized)

| Phase | Components included | Rust .wasm | Zig .wasm |
|-------|---------------------|------------|-----------|
| —     | (none yet)          | —          | —         |

## Execution time

| Benchmark | Rust | Zig | Python (numba, native) |
|-----------|------|-----|------------------------|
| —         | —    | —   | —                      |

## Cross-implementation parity

Rust and Zig must agree bit-for-bit on all components (see PORTING.md numeric
policy). Verified: (pending phase 10 harness).

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

- (none yet)
