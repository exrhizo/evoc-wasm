# Port traps by module

Parity hazards discovered while generating fixtures from the reference. Porters:
read the section for your phase before implementing. This complements (does not
replace) reading the reference source and the fixtures' own `inputs.note`.

## graph_construction.py (phase 4)

- `smooth_knn_dist`: numba `locals` pin `psum/lo/mid/hi` to **f32** despite
  `fastmath=True` — accumulate in f32. Binary search: `mid` starts 1.0, `hi`
  starts `inf`; while `hi==inf` the miss branch **doubles** `mid` instead of
  bisecting. `rho[i]` set only if there is ≥1 nonzero distance (else 0.0). Inner
  psum loop starts at `j=1` (index 0 is self/rho, excluded). `mean_distances` is
  a single `np.mean` over the whole matrix, used only in the `rho==0` fallback;
  `mean_ith_distances` is per-row.
- `compute_membership_strengths`: branch order is load-bearing — `idx==-1` skips
  the slot entirely (stays pre-zeroed, no row written), then `idx==i` ⇒ `val=0`,
  then `(dist-rho)<=0 or sigma==0` ⇒ `val=1`, else the exp formula.
- `neighbor_graph_matrix`: symmetrization is the fuzzy-set union
  `A + Aᵀ − A·Aᵀ` (elementwise product), **not** boolean OR. `eliminate_zeros`
  is called both before and after. The `n_neighbors` arg here is a **float**.

## knn_graph.py (phase 4)

- Ports call the **internals** (`make_forest` + `nn_descent`), never numpy's
  Mersenne Twister. The fixture records the exact `rng_state` (drawn first, for
  nn_descent) and `rng_states` (drawn second, one row per RP-tree, inside
  `make_forest`) — order is load-bearing; both come from the same advancing
  RandomState. `INT32_MIN/MAX` are `iinfo(int32).min+1` / `.max-1`.
- Fixture bakes single-thread defaults: n_trees=4, leaf_size=10, n_iters=7
  (`max(5, round(log2(n)))`), effective_max_candidates=15 (`min(60,int(k*1.5))`).
  All recorded in `inputs`.
- Float distance transform: `max(-log2(-d), 0.0)` (clamped at 0). (int8 uses
  `1/(-d)`, uint8 uses unclamped `-log2(-d)` — for later.)

## boruvka.py (phase 5)

- Everything runs in **squared** rdist space until the very end;
  `parallel_boruvka` applies one final `sqrt` to the whole edge array's distance
  column. `initialize_boruvka_from_knn` is fixtured **pre-sqrt** (its natural
  output) — don't expect final euclidean numbers there.
- `initialize_boruvka_from_knn` edge weight = `max(core_dist[i], knn_dist[i,j])`
  at the **first** neighbor `j` (from j=1, skipping self) with
  `core_distances[i] >= core_distances[k]`; breaks on first hit, not best.
- `merge_components` uses a numba typed dict keyed by component id, value tuple
  `(i64, i64, f32)`; port as array/hashmap keyed by component. First-improving
  edge per component.
- `reproducible=True` only bounds multithreaded races (adaptive block_size);
  with n_threads=1 it is bit-identical to `reproducible=False`. Real divergence
  is multithreaded — out of scope (WASM single-threaded).

## cluster_trees.py (phase 6)

- `mst_to_linkage_tree`: linkage `result` is **f64** (numpy default), even though
  MST distances are f32. Row convention: larger component id goes in the left
  slot (`left_component > right_component`).
- `condense_tree`: `CondensedTree` fields `parent:i64, child:i64,
  lambda_val:f32, child_size:i64`. `lambda = 1/d if d>0 else inf`. The five-way
  branch is deliberately ordered for the common (`left` singleton) case first —
  replicate the exact branch order, not just the logical union.
  `eliminate_branch` marks flattened internal descendants `ignore=True` via BFS.
- `extract_leaves`: leaf = node id never appearing as a parent with
  `child_size>1`. `n_points = parent.min()`, `n_nodes = parent.max()+1`.
- `get_cluster_label_vector` has a single-cluster fast path
  (`get_single_cluster_label_vector`, different epsilon semantics) when
  `len(clusters)==1`. The n80 fixture yields 4 clusters, so **only the
  multi-cluster path is covered** — add a single-cluster case if you touch that
  path.
- `extract_eom_clusters` uses numba typed dicts whose iteration order is
  incidental; the fixture stores `eom_clusters` **sorted** — compare as
  sets/sorted, don't rely on reference order.
