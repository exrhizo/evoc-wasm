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
| 7     | label_propagation    | —        | —         | —          | —        | —         |
| 7     | node_embedding       | —        | —         | —          | —        | —         |
| 8     | clustering_utilities | —        | —         | —          | —        | —         |
| 8     | clustering           | —        | —         | —          | —        | —         |
| 9     | wasm bindings + npm  | n/a      | —         | —          | —        | —         |
| later | nndescent_int8       | —        | —         | —          | —        | —         |
| later | nndescent_uint8      | —        | —         | —          | —        | —         |

## Deferred / gaps

- cluster_trees.py epsilon-search + cut family (`mask_condensed_tree`,
  `cluster_epsilon_search`, `traverse_upwards`, `segments_in_branch`,
  `in_set_parallel`, `get_cluster_labelling_at_cut`): no phase-6 fixture
  exercises these. Rust ported them (unverified); Zig deferred. Phase 8 must
  fixture + verify them in both languages (used by `build_cluster_layers` and
  `binary_search_for_n_clusters`).

Comparison results: see `comparison/results.md` (phase 10).
