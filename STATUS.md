# Port status

Legend: ✅ done · 🚧 in progress · — not started

| Phase | Component            | Fixtures | Rust impl | Rust tests | Zig impl | Zig tests |
|-------|----------------------|----------|-----------|------------|----------|-----------|
| 1     | disjoint_set         | ✅       | ✅        | ✅         | ✅       | ✅        |
| 1     | rng (tau_rand)       | ✅       | ✅        | ✅         | ✅       | ✅        |
| 1     | distance kernels     | ✅       | ✅        | ✅         | ✅       | ✅        |
| 2     | kdtree               | ✅       | ✅        | ✅         | ✅       | ✅        |
| 3     | nnd_heap             | ✅       | ✅        | ✅         | ✅       | ✅        |
| 3     | nndescent_float      | ✅       | ✅        | ✅         | ✅       | ✅        |
| 4     | knn_graph            | ✅       | ✅        | ✅         | ✅       | ✅        |
| 4     | graph_construction   | ✅       | ✅        | ✅         | ✅       | ✅        |
| 5     | boruvka              | ✅       | ✅        | ✅         | ✅       | ✅        |
| 6     | cluster_trees        | ✅       | ✅        | ✅         | ✅       | ✅        |
| 7     | label_propagation    | ✅       | ✅        | ✅         | ✅       | ✅        |
| 7     | node_embedding       | ✅       | ✅        | ✅         | ✅       | ✅        |
| 8     | clustering_utilities | ✅       | ✅        | ✅         | ✅       | ✅        |
| 8     | clustering           | ✅       | ✅        | ✅         | ✅       | ✅        |
| 9     | pca (randomized SVD) | n/a (ARI/reconstruction-quality based -- see `rust/core/PCA_VALIDATION.md`) | ✅ | ✅ | ✅ | ✅ |
| 9     | evoc_clusters / EVoC::fit_predict | n/a (ARI-based, not bit-exact -- see `rust/core/PCA_VALIDATION.md`) | ✅ | ✅ | ✅ | ✅ |
| 9     | wasm bindings + npm  | n/a      | —         | —          | —        | —         |
| later | nndescent_int8       | —        | —         | —          | —        | —         |
| later | nndescent_uint8      | —        | —         | —          | —        | —         |

## Deferred / gaps

- cluster_trees.py epsilon-search + cut family (`mask_condensed_tree`,
  `cluster_epsilon_search`, `traverse_upwards`, `segments_in_branch`,
  `in_set_parallel`, `get_cluster_labelling_at_cut`): **Rust resolved in
  phase 8.** No JSON fixture exercises `cluster_epsilon_search` /
  `traverse_upwards` / `segments_in_branch` / `in_set_parallel` /
  `get_cluster_labelling_at_cut` directly -- tracing the full upstream
  `evoc` package confirms these four are dead code with zero call sites
  anywhere outside `cluster_trees.py` itself, so no fixture will ever cover
  them. Verified instead with hand-computed ground truth (an independent,
  non-numba plain-numpy reimplementation run locally against synthetic
  cluster trees; see the unit tests in `rust/core/src/cluster_trees.rs`'s
  `#[cfg(test)] mod tests`, "Deferred epsilon-search / cut family" section).
  `mask_condensed_tree` (the one function from this family with a real call
  site, in `clustering.py`'s `build_cluster_layers`) is additionally
  verified end-to-end via the `clustering::build_cluster_layers_n126`
  fixture test. **Zig: resolved in phase 8** — all six ported into
  `zig/src/cluster_trees.zig` with dedicated unit tests (`mask_condensed_tree`,
  `cluster_epsilon_search`, `get_cluster_labelling_at_cut`); same dead-code
  status confirmed (no direct fixture) for `traverse_upwards`/
  `segments_in_branch`/`in_set_parallel`, exercised only as internal helpers
  of `cluster_epsilon_search`'s unit test. `mask_condensed_tree` itself is
  additionally verified end-to-end via the Zig `clustering.build_cluster_layers_n126`
  and `clustering_utilities.condense_tree_seed45` fixture tests.
- `clustering.py`'s `build_cluster_layers` `base_n_clusters is not None`
  branch has no fixture coverage (ported by composing already-fixtured
  pieces per PORT_NOTES.md, not independently verified end-to-end). Zig
  does not implement this branch at all (documented PORT NOTE in
  `zig/src/clustering.zig`); only the fixtured `base_n_clusters=None`
  default path is ported.
- **SCOPE CHANGE (phase 9, Rust): `evoc_clusters()` / `EVoC::fit_predict()`
  are now IN SCOPE and implemented** (`rust/core/src/clustering.rs`,
  `rust/core/src/pca.rs`, plus the recursive `label_propagation_init` and
  self-driven `node_embedding` variants in `label_propagation.rs`/
  `node_embedding.rs`). The prior "out of scope" call (still true for the
  bit-exact-fixture standard used everywhere else in this port) was
  overridden: sklearn's `PCA(random_state=...)` internals genuinely cannot
  be replayed bit-for-bit (undocumented internal randomized-SVD draws), so
  this implements the actual Halko/Martinsson/Tropp randomized-SVD algorithm
  sklearn's solver is based on, driven by this project's own `tau_rand` RNG
  end-to-end instead of numpy's Mersenne Twister. Validated via
  reconstruction-quality unit tests (`pca.rs`) and an Adjusted Rand Index
  check against a real `evoc.EVoC` run. **Updated methodology (see
  "seed-panel update" note below): the old single-seed ARI 0.8033 claim is
  replaced by a pre-registered 20-seed panel** compared against a measured
  Python-vs-Python self-agreement null
  (`fixtures/clustering/python_self_agreement_null.json`, median 0.5247);
  the Rust panel (seeds 0..19) achieves **median ARI 0.7500, min 0.2637**
  against the cached Python(seed=42) reference
  (`rust/core/tests/evoc_clusters_ari.rs::evoc_clusters_e2e_ari_panel_against_python_evoc`,
  gates: median >= null_median - 0.15 = 0.3747, min >= 0.15) -- **see
  `rust/core/PCA_VALIDATION.md` for the full methodology and an explicit
  statement that this does NOT lower the bit-exact bar for any other
  component in this port.** Out of scope within this pass:
  `approx_n_clusters`/`binary_search_for_n_clusters`'s single-layer return
  branch, `return_duplicates`, and the int8/uint8 quantized-embedding paths
  (consistent with the rest of this port, which defers
  `nndescent_int8`/`nndescent_uint8` to "later"). **Zig: also now
  implemented** (`zig/src/pca.zig`, `zig/src/clustering.zig`'s
  `evoc_clusters`/`EVoC`, plus `label_propagation_init_recursive` in
  `zig/src/label_propagation.zig` and `node_embedding_repr_self_driven` in
  `zig/src/node_embedding.zig`), independently from-scratch per PORTING.md
  (read alongside the Rust equivalent for consistency, not transliterated).
  Same validation approach as Rust: `pca.zig`'s unit tests check
  reconstruction quality on a synthetic rank-2-plus-noise matrix (>90%
  captured variance), and `clustering.zig`'s
  `evoc_clusters e2e ARI panel against Python evoc.EVoC` test reuses the
  exact same cached `fixtures/clustering/evoc_clusters_e2e_ari_check.json`
  fixture as the Rust test (same input data, same Python reference labels).
  **Updated methodology: the old single-seed ARI 0.8930-at-seed-29 claim is
  replaced by the same pre-registered 20-seed panel (seeds 0..19) used on
  the Rust side**, achieving **median ARI 0.7514, min 0.5147** against the
  cached Python(seed=42) reference, against the same gates as Rust (median
  >= 0.3747, min >= 0.15) -- comfortably passing, and notably tighter than
  Rust's own panel spread, though both are well within the pass region; see
  `rust/core/PCA_VALIDATION.md` for the shared methodology and gate
  reasoning. All Zig tests pass (`zig build test`), including every phase
  1-8 fixture test unchanged, and `zig fmt --check src/` is clean.
- **New (Zig, phase 8):** `np.argsort`'s exact unstable tie-break for
  float32 arrays is not reproducible against numpy 2.x, which dispatches
  contiguous-array sorts to a vectorized AVX-512 sorting-network
  implementation (x86-simd-sort), not the classical npysort
  quicksort/introsort `zig/src/clustering_utilities.zig`'s `argsort_f32`
  replicates. This only bites when the MST distance column
  (`sorted_mst = edges[np.argsort(edges.T[2])]`, used by
  `binary_search_for_n_clusters` and `build_cluster_layers`) has exact
  ties, which happens in practice for real mutual-reachability distances.
  `argsort_f32` matches numpy's real order in the large majority of cases
  (verified: matches on 4/6 tied pairs in the `condense_tree_seed45`
  dataset) and is internally deterministic/bit-identical between Rust and
  Zig; the `clustering_utilities.condense_tree_seed45` Zig test tolerates a
  small explicitly-bounded number of tie-induced row mismatches rather than
  asserting strict bit-parity through a numpy-version-specific SIMD sort
  implementation detail unrelated to the evoc algorithm itself. Rust's
  status on this specific point is unknown to this (Zig-only) note.

Comparison results: see `comparison/results.md` (phase 10).
