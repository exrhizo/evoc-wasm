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
| 8     | clustering_utilities | ✅       | ✅        | ✅         | —        | —         |
| 8     | clustering           | ✅       | ✅        | ✅         | —        | —         |
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
  fixture test. Zig still deferred.
- `clustering.py`'s `build_cluster_layers` `base_n_clusters is not None`
  branch has no fixture coverage (ported by composing already-fixtured
  pieces per PORT_NOTES.md, not independently verified end-to-end).
- `evoc_clusters()` / `EVoC.fit_predict()` remain explicitly out of scope
  (sklearn PCA internals, not portable) -- see PORT_NOTES.md.

Comparison results: see `comparison/results.md` (phase 10).
