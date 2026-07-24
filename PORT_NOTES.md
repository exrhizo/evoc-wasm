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

## label_propagation.py (phase 7)

- **Two independent RNG mechanisms coexist in this module — do not conflate
  them.** (1) `label_prop_iteration` / `original_label_prop_iteration` /
  `label_outliers` use `tau_rand` / `tau_rand_int` on a **per-row derived
  state**: `local_rng_state = rng_state + i` — this is numpy elementwise
  addition producing a **fresh 3-int64 array per row/node `i`**, NOT the
  shared `rng_state` array advancing across rows. `tau_rand`/`tau_rand_int`
  then mutate that local copy in place for however many draws that row makes.
  Because `numba.prange` parallelizes over `i`, this per-row-offset design is
  what makes the kernel race-free across threads — replicate the `rng_state +
  i` derivation exactly (integer array addition, no wraparound tricks) rather
  than trying to advance a single shared state serially. (2) `label_prop_loop`
  / `original_label_prop_loop` / `label_propagation_init` draw from the
  **passed-in `numpy.random.RandomState`** (Mersenne Twister) for: one
  `randint(INT32_MIN, INT32_MAX, 3)` to produce the fixed `rng_state` array
  used by mechanism (1) for the rest of the loop, then `approx_n_parts` (or
  `int(1.25*approx_n_parts)` in the `original_*` variant) individual
  `randint(labels.shape[0])` calls, each immediately setting
  `labels[draw] = i` (0-based loop index) as a partition seed — collisions
  (same position drawn twice) simply overwrite, they are not resampled.
- `label_prop_iteration` vs `original_label_prop_iteration`: **byte-identical
  logic**, both walk the exact same vote/tie-break structure. The only
  external difference is which loop (`label_prop_loop` vs
  `original_label_prop_loop`) calls them and how many partition seeds get
  drawn beforehand (`approx_n_parts` vs `int(1.25*approx_n_parts)`). Fixtures
  cover `label_prop_iteration` only; port `original_label_prop_iteration` as a
  structurally identical function if the reference's public API needs it, but
  no separate fixture exists for it (would be redundant).
- `label_prop_iteration`'s tie-break: `max_vote` starts at `1` and `tie_count`
  starts at `1` — **not** at the first vote's value. A vote exactly equal to
  `1` ties the initial sentinel; votes strictly less than `1` never register
  at all (the `else: continue` branch), which matters since `data[k]` (edge
  weight) values can be `< 1.0`. Branch order inside the per-node vote loop:
  skip `l == -1` entirely first (noise votes never count), then
  `votes[l] > max_vote` (new leader, resets `tie_count = 1`), then `votes[l]
  == max_vote` (tie: increment `tie_count`, and either take the vote
  unconditionally if `current_l == -1` i.e. this node was previously
  unlabeled, or flip a `tau_rand(local_rng_state) < 1.0 / tie_count` coin
  using the SAME per-row local state — replicate this exact
  reservoir-sampling-style tie-break, not a "pick first" or "pick last"
  simplification).
- `label_outliers`: `max_label = labels.max()` is computed **once, before**
  the `numba.prange` loop, from the labels array as handed in (not updated as
  other threads resolve nodes). Per unresolved node `i`: BFS via a Python-list
  used as a stack (`node_queue.pop()` pops from the **end** — LIFO/DFS order
  despite the `node_queue` name), capped at `n_iter < 64` **and**
  `len(node_queue) > 0`. First neighbor found with `labels[j] >= 0` wins
  (`break`), copying `labels[j]`; every other unresolved neighbor visited so
  far gets pushed onto the queue for the next pop. If the cap is hit or the
  queue drains before finding a labeled neighbor (`n_iter >= 64 or
  unlabelled`), falls back to `labels[i] = tau_rand_int(local_rng_state) %
  (max_label + 1)` — the disconnected/isolated-node case is covered by the
  `label_outliers_disconnected` fixture (node with zero edges: the BFS
  never finds a neighbor and falls straight to the random-assignment branch
  on the very first pop). `label_outliers` is `@numba.njit(cache=True)`
  **without** `fastmath=True` and is invoked via `numba.prange` but is
  logically single-pass-per-node (no cross-node interaction beyond reading
  `labels[j]`, which may itself still be `< 0` if `j` hasn't been visited by
  another "thread" yet in the real parallel scheduler — the reference accepts
  this race as intentional/benign since a not-yet-resolved neighbor is simply
  treated the same as any other unresolved node and pushed onto the queue).
- `remap_labels`: `np.unique(labels)` sorted-and-deduped; if the smallest
  unique value is `-1` it's dropped from the mapping (all remaining entries
  are non-negative by construction at that point in practice, but the guard
  is unconditional). Existing non-negative labels remap through `mapping` to
  their sorted-rank index. Crucially, remaining `-1` entries do **not**
  collapse to one label — each becomes a **new singleton cluster**, assigned
  `next_label = len(unique_labels), len(unique_labels)+1, ...` **in array
  index order** (first `-1` seen gets the smallest new label). This means
  `remap_labels`'s output label count can exceed the count of resolved
  clusters if any `-1`s survive (should not happen after `label_outliers` in
  the normal pipeline, but the function itself has no such guarantee).
- `label_propagation_init`: **branches on `graph.shape[0] <
  base_init_threshold` (default 64)**. Below threshold: delegates entirely to
  `PCA`/`SpectralEmbedding`/`MDS`/random-normal via sklearn, each seeded with
  the **raw `random_state` object itself** (sklearn/scipy internals draw from
  it in ways that are not meaningfully portable to Rust/Zig — **not
  fixtured**; if a port needs to hit a graph below this threshold, either
  raise `base_init_threshold` so the caller never takes this branch, or treat
  it as an intentionally-unported edge case, matching `PORTING.md`'s
  ground-truth-only policy). At/above threshold: calls `label_prop_loop` (see
  above) to get a `partition` vector, builds a reduced graph via sparse
  matrix algebra (`base_reduction_map`, `normalized_reduction_map =
  normalize(..., axis=0, norm="l2")`, `reduced_graph =
  normalized_reduction_map.T @ graph @ base_reduction_map`, clipped to `[0,
  1]` — no randomness in this algebra), then **recurses** into
  `label_propagation_init` again if `recursive_init=True` (the library
  default) — each recursion level draws more from the same advancing
  `random_state` and, once the reduced graph itself drops below
  `base_init_threshold`, hits the unportable PCA path. **Fixtures use
  `recursive_init=False`** to keep the whole call to exactly one
  `label_prop_loop` + one `node_embedding` (which draws its own init `normal`
  + per-epoch `rng_val`/`shuffle` from the SAME advancing `random_state`,
  since `label_propagation_init` passes `initial_embedding=None` down to it
  on the non-recursive path — see node_embedding.py notes below for that
  draw's exact shape/order). Every RNG value `label_propagation_init` and its
  callees draw was traced and is recorded explicitly in the
  `label_propagation_init_n120` fixture (`lp_rng_state`,
  `lp_seed_position_draws`, `node_embedding_init_normal`,
  `node_embedding_rng_val`, `node_embedding_shuffles`, in that exact call
  order) — replay by feeding these arrays in verbatim rather than trying to
  reproduce numpy's Mersenne Twister. Verified: an independent
  `RandomState(321)` replay of the whole function reproduces the fixture's
  `embedding` bit-for-bit, confirming the recorded draws are sufficient and in
  the right order.
- `label_propagation_init`'s `upscaling="partition_expander"` (the default,
  and what `clustering.py`'s `evoc_clusters` uses) has no further randomness
  after `node_embedding` returns — pure sparse-matrix algebra
  (`data_expander @ reduced_layout + normalize(normalized_reduction_map,
  norm="l1") @ reduced_layout) / 2.0`, then `scaling * (result -
  result.mean(axis=0))` cast to f32. The `"jitter_expander"` and default-`else`
  (plain `upscaling`) branches draw additional `random_state.normal(...)`
  noise and are **not exercised or fixtured** — `evoc_clusters` never
  requests them.

## node_embedding.py (phase 7)

- **Negative sampling in this module does NOT use `tau_rand`.** This is the
  single most important trap: unlike every other stochastic kernel in the
  reference (RP-tree splits, NN-descent candidate sampling, label-propagation
  tie-breaks), `node_embedding_epoch` / `node_embedding_epoch_repr` compute
  negative-sample targets via **pure integer arithmetic**, no RNG state
  mutation at all: `k = ((n + p) * i * rng_state) % n_vertices` (plain
  kernel, `i` = sample index) or `to_node = node_order[(raw_index * (n + p +
  1) * rng_state) % n_vertices]` (repr kernel, `raw_index` = CSR edge index).
  `rng_state` here is a **single scalar** (`u4`/uint32 in the repr kernel's
  numba signature), not a 3-int64 tau_rand state array. It is passed in fresh
  per epoch — see below.
- The scalar `rng_state` fed into each epoch comes from `node_embedding`'s
  **one-time upfront draw** `rng_val = random_state.randint(INT32_MAX,
  size=n_epochs)` — a numpy-MT draw of `n_epochs` values, done **once before
  the epoch loop starts**, not per-epoch. Epoch `n`'s kernel call receives
  `np.uint32(rng_val[n])` (repr path) or the raw `rng_val[n]` (non-repr
  path). Since this one draw is a numpy MT draw, it is **not** portably
  reproducible — record `rng_val` explicitly as a fixture input (all node_embedding
  fixtures do this).
- `reproducible_flag=True` (the **only** value `clustering.py`'s
  `evoc_clusters`/`node_embedding` call site ever passes) selects
  `node_embedding_epoch_repr`, not `node_embedding_epoch`. The two kernels
  differ in more than just parallelization strategy:
  - `node_embedding_epoch` (non-repr): takes COO edge lists (`head`, `tail`
    arrays from `graph.tocoo()`), updates `embedding` **directly in place**
    per-edge as it goes (no deferred accumulator), and its negative-sample
    `grad_coeff` has no `gamma` multiplier. `epochs_per_negative_sample` has
    **no `*1.5` inflation** on this path.
  - `node_embedding_epoch_repr` (repr): takes CSR (`csr_indptr`,
    `csr_indices`), processes nodes in **blocks** of `block_size` (default
    `max(1024, n_vertices // 8)`, recorded per-fixture) via `node_order`
    (initially `arange(n_vertices)`, **shuffled once per epoch** — see
    below), accumulates all of a block's positive+negative gradient
    contributions into a separate `updates` array first, and only applies
    `embedding[from_node, d] += updates[from_node, d]` in a **second pass
    over the same block** after all nodes in that block have computed their
    updates (this is what makes the ordering deterministic/reproducible
    under parallelism — updates from node A processed in the same block
    never see node B's already-applied update). Negative-sample `grad_coeff`
    is scaled by `gamma = gamma_schedule[n]` where `gamma_schedule =
    np.linspace(0.5, 1.5, n_epochs)` (deterministic, recompute — not drawn
    from RNG), and only applied if `grad_coeff > 0.0` (extra guard not
    present in the non-repr kernel). `epochs_per_negative_sample` **is**
    inflated by `*1.5` when `reproducible_flag=True`, before being passed in.
  - **Port whichever kernel matches your use case, but `evoc_clusters`'s
    default path is `_repr` — that's the one fixtured for the end-to-end
    case.** Both single-epoch kernels are fixtured in isolation too
    (`node_embedding_epoch_repr_single_epoch_n120`,
    `node_embedding_epoch_single_epoch_n120`) for ports that need both.
  - **`updates` is NOT re-zeroed between epochs.** After each
    `node_embedding_epoch_repr` call, `node_embedding` does `updates *= (1.0
    - alpha) ** 2 * 0.5` — a decay, not a reset — so gradient momentum
    carries across epochs. Any single-epoch fixture/test that only calls the
    kernel once starts `updates` at zero (matching epoch 0's real starting
    condition), but a multi-epoch harness must replicate this decay between
    calls, not clear the array.
  - **`random_state.shuffle(node_order)` is called once per epoch**, **after**
    that epoch's kernel call, only on the `reproducible_flag=True` path. This
    is another numpy-MT draw — recorded explicitly per-epoch in the
    `node_embedding_shuffles/node_order_shuffles` fixture fields. `node_order`
    persists (in its shuffled state) into the next epoch's kernel call — it is
    not reset to `arange` each epoch.
- `epoch_of_next_sample[i] <= n` (and the equivalent negative-sample gate) is
  the schedule-firing condition. Under the natural epoch-0 initial condition
  (`epoch_of_next_sample = epochs_per_sample.copy()`, and
  `epochs_per_sample[i] >= 1.0` always, from `make_epochs_per_sample`'s
  `max(..., 1.0)` floor), **almost nothing fires on epoch 0** — most of the
  graph's structure only starts updating from epoch 1 onward as schedules
  catch up. The single-epoch isolation fixtures
  (`node_embedding_epoch_repr_single_epoch_n120`,
  `node_embedding_epoch_single_epoch_n120`) deliberately record
  `epoch_of_next_sample` / `epoch_of_next_negative_sample` as **all-zero**
  (not the natural epoch-0 value) specifically so every sample fires on that
  one recorded call — this isolates the per-edge SGD/negative-sampling math
  from the outer scheduling, but is NOT what epoch 0 of a real multi-epoch
  run looks like. Use the end-to-end fixture
  (`node_embedding_e2e_n120`) to test the full schedule-driven behavior
  across epochs.
- `make_epochs_per_sample`: `result = float(n_epochs) / np.float32(n_samples)`
  where `n_samples = np.maximum(n_epochs * (weights / weights.max()), 1.0)` —
  note the **cast order**: `n_samples` is computed as a float64 numpy
  expression, cast to `np.float32` as a whole array, and only then divided
  into the Python `float(n_epochs)` (which numpy will broadcast/cast against
  the f32 array) — replicate the f32 cast happening on `n_samples` before the
  division, not on the final result.
- `rdist(x, y)`: numba `locals` pin `result`/`diff` to **f32** with
  `fastmath=True` — accumulate in f32, plain squared-euclidean, no sqrt.
  `clip(val, lo, hi)`: `> hi` first, then `< lo`, else `val` — replicate this
  exact branch order (matters at NaN, though NaN is not expected in practice
  here).
- `node_embedding`'s `initial_embedding=None` default draws `random_state.normal(
  scale=0.25, size=(graph.shape[0], n_components))` — another numpy-MT draw,
  done **before** `rng_val` is drawn. All node_embedding fixtures here pass
  `initial_embedding` explicitly (recorded) to sidestep this, **except** the
  `label_propagation_init_n120` fixture in label_propagation.py's notes above,
  which documents this exact draw happening internally (since
  `label_propagation_init` calls `node_embedding` with `initial_embedding=None`
  on its non-recursive path).
- Learning-rate schedule: `alpha` for epoch `n` (`n >= 1`) is computed
  **after** epoch `n-1`'s kernel call and shuffle, as `initial_alpha * (1.0 -
  float(n-1) / float(n_epochs))` in effect — epoch 0 uses the raw
  `initial_alpha` unchanged (the update happens at the bottom of the loop
  body, so it takes effect starting the *next* iteration). Off-by-one errors
  here silently shift the whole decay schedule by one epoch.

## clustering_utilities.py (phase 8)

- All functions in this module are **pure/deterministic given their inputs**
  — no `random_state`, no numba `prange`-induced nondeterminism (several are
  plain sequential `@numba.njit(cache=True)`, not `parallel=True`), no numpy
  MT usage anywhere **except** inside `binary_search_for_n_clusters`'s call to
  `parallel_boruvka(..., reproducible=False)` (which per the phase-5 boruvka
  notes is bit-identical to `reproducible=True` at `n_threads=1`, so this is
  moot for the WASM single-threaded target). This makes the whole module
  straightforward to fixture directly — the only real hazard is replicating
  exact (sometimes non-obvious) branch/iteration order, not RNG plumbing.
- `find_peaks`: ported near-verbatim from scipy's peak finder. First and last
  samples can **never** be peaks (`i` starts at 1, loop bound is
  `x.shape[0]-1`, both exclusive of the endpoints). A **plateau** (several
  equal consecutive values forming a local max) counts as **one** peak, with
  `midpoint = (left_edge + right_edge) // 2` — integer floor division, so an
  even-width plateau's midpoint rounds toward the **left** half. The
  `i_ahead` scan-forward-past-ties loop is what detects plateaus; replicate it
  exactly (don't shortcut to `x[i-1] < x[i] and x[i+1] < x[i]`, which misses
  plateaus entirely). Two type signatures exist (`float32`/`float64` input) —
  keep both if your language needs monomorphized variants.
- `_binary_search_for_n_clusters`: bisects `min_cluster_size` in `[2,
  n_samples // 2]` using Python's `round()` semantics
  (`round((lo+hi)/2.0)` — banker's/round-half-to-even on exact `.5`
  boundaries; do **not** substitute `floor(x + 0.5)`, which rounds `.5`
  away from zero and can diverge on ties). Terminates the `while` loop when
  the bracket narrows to width 1, **or early** if the computed `mid` lands
  exactly on either current bound (`break`, without recomputing `mid_tree`
  for that iteration — the last-computed `lower_tree`/`upper_tree` from
  before the loop, or from the previous iteration, are what's used in the
  final selection, not a naive recompute at the break point). Final
  selection is a **three-way branch**, all three must be replicated exactly:
  (1) `lower` strictly closer to `approx_n_clusters` → use `lower_bound`; (2)
  `upper` strictly closer → use `upper_bound`; (3) **exact tie** → recompute
  **both** `lower`/`upper` clusters+strengths and pick whichever has **more
  non-noise points** (`np.sum(clusters >= 0)`), not an arbitrary
  first-found. `binary_search_for_n_clusters` (the public wrapper) always
  calls `parallel_boruvka(..., reproducible=False)` regardless of caller
  intent — hardcoded, not a passthrough parameter.
- `min_cluster_size_barcode`: `n_nodes = cluster_tree.child[-1] - n_points +
  1` — **assumes `cluster_tree` is non-empty and its rows are sorted with the
  root-adjacent pair last** (this is condense_tree's/mask_condensed_tree's
  natural output order; do not re-sort before calling). Iterates **row pairs**
  in **reverse**, stepping by 2 (`for idx in range(n_rows-1, 0, -2)`) — the
  cluster tree's rows come in adjacent left/right-sibling pairs from
  `condense_tree`'s branch-split logic, and this function processes one
  sibling pair per iteration, writing to **both** `out_idx-1` and `out_idx`
  slots in one step (`out_idx = cluster_tree.child[idx] - n_points`,
  indexing the RIGHT sibling; `out_idx-1` is the LEFT). `size_deaths` at a
  pair = `min(child_size[idx-1], child_size[idx])` (the smaller of the two
  siblings' sizes — the larger sibling doesn't "die" at this lambda, only
  shrinks). `size_births[parent]` = `max` of the two children's own births
  and their shared death size — a parent's birth size is never smaller than
  what its children needed to exist. Index 0 (the tree root) is
  special-cased **before** the loop: `lambda_deaths[0] = 0`, `size_deaths[0]
  = n_points`, `parents[0] = n_points` (i.e. the root's own "parent" sentinel
  is `n_points`, not `-1` or itself).
- `compute_total_persistence`: **loop starts at `i = 1`, skipping index 0**
  (the barcode's root entry) intentionally — comment says "maintain
  left-open (birth, death] interval". `sizes = np.unique(births)` (sorted,
  deduped) is the persistence curve's x-axis. Both `birth_idx`/`death_idx`
  lookups are **linear scans** for the first `sizes[j] >= birth`/`>= death`
  despite the "manual binary search" comment in the source — replicate as a
  first-match linear scan (identical result to a real binary search since
  `sizes` is sorted, but do not assume the reference actually bisects — for
  parity purposes the *algorithm* described (linear) is ground truth, and any
  equivalent-output binary search is an acceptable substitution *only if* you
  independently verify equivalence; when in doubt, port literally). Entries
  with `death <= birth` are skipped (`continue`) — persistence intervals with
  zero or negative width contribute nothing. Where an interval **is** valid,
  `total_persistence[k] += (death - birth) * lambda_death` is added to
  **every** bucket `k` in `[birth_idx, death_idx)` (a range-add, not a single
  bucket) — this is what makes the persistence curve piecewise and gives
  `find_peaks` something non-trivial to find.
- `extract_clusters_by_id`: thin wrapper — `n_samples =
  condensed_tree.parent[0]` (reads it off the tree itself, does **not** take
  `n_samples` as a parameter) and `cluster_selection_epsilon=0.0` is
  hardcoded. Delegates to `get_cluster_label_vector`/
  `get_point_membership_strength_vector` from `cluster_trees.py` (phase 6) —
  no new logic of its own beyond the parameter derivation.
- `jaccard_similarity`: builds `union_set` starting from **all of** `set_a_array`
  (via `set(...)`), then folds in `set_b_array` element by element, counting
  `intersection_count` for every `b`-item already in the (growing) union set
  — this is algorithmically asymmetric in *how* it computes the answer (it
  matters only if `set_a_array`/`set_b_array` have internal duplicates, which
  would double-count intersections against a naive symmetric formula; the
  reference's arrays in practice are unique index sets so this doesn't bite,
  but do not "simplify" to a symmetric set-intersection/set-union formula
  without checking duplicate-handling matches). Empty/empty → `union_count ==
  0` → returns `0.0` exactly (guarded, not a NaN/divide-by-zero).
- `estimate_cluster_similarity`: "active at birth `X`" means `births[i] <= X
  and deaths[i] > X` — a half-open `[birth, death)` window read at a point,
  which is a **different** convention from `compute_total_persistence`'s
  documented `(birth, death]` interval semantics for the same underlying
  barcode arrays — both are correct for their own purpose, just don't
  assume they're interchangeable or that one implies the other's boundary
  handling.
- `select_diverse_peaks`: sorts candidate peaks by `total_persistence`
  **descending** via `np.argsort(...)[::-1]` — note this reverses a
  stable-ascending argsort, so **ties in persistence end up in descending
  original-index order** among themselves (not simply "insertion order";
  verify against `np.argsort`'s documented stability + the `[::-1]` reversal
  if your language's sort isn't stable in the same way). Greedy selection:
  walk sorted peaks, accept a peak only if its `estimate_cluster_similarity`
  against **every already-selected** peak is `<= min_similarity_threshold`
  (reject — `is_diverse = False`, `break` early — on the **first**
  disqualifying comparison, don't scan the rest). Stops once `max_layers`
  peaks are accepted, **not** once all peaks are considered — later
  higher-persistence... no, peaks are pre-sorted descending, so `max_layers`
  simply caps the greedy output at the top-K diverse peaks. `len(peaks) == 0`
  short-circuits to an empty result before any sorting/allocation.
- `_build_cluster_tree` / `build_cluster_tree`: operates on a **list of label
  vectors**, index 0 = finest/base layer, increasing index = coarser layers
  (this is the opposite convention from `build_cluster_layers`'s *output*
  ordering, which sorts **descending** by cluster count — i.e. layer 0 in
  `_build_cluster_tree`'s input is layer 0 in the sorted `cluster_layers`
  output only by coincidence of the fixture's specific case; do not assume
  index correspondence between the two functions' layer numbering without
  checking). For each `upper_layer` from 1 upward, scans every
  `lower_layer` from `upper_layer - 1` down to 0 (**not** just
  `upper_layer - 1`) via `np.argsort` + `np.bincount`/`np.cumsum` grouping,
  recording a `(parent_layer, parent_label, child_layer, child_label)` edge
  **the first time** that `(child_layer, child_label)` pair is seen across
  the whole scan (a numba `set` per layer, seeded with `{-1}` so noise is
  pre-excluded from ever being "found") — this first-seen-wins dedup means a
  child only ever gets ONE parent recorded, from whichever `upper_layer`
  happens to claim it first in the `upper_layer`-ascending / `lower_layer`-
  descending double loop, not necessarily its immediate/nearest parent
  layer. After the double loop, any `(layer, label)` pair still unclaimed
  gets attached under a synthetic virtual root `(len(labels), 0)` — labels
  that never got claimed by any coarser layer (e.g. the globally coarsest
  layer's own clusters) land here. `build_cluster_tree` (the Python wrapper)
  turns the flat `(parent_layer, parent_cluster, child_layer, child_cluster)`
  tuple list into a `dict[(layer,cluster)] -> list[(layer,cluster)]`, with
  child lists in **whatever order they were appended** (insertion order via
  the double loop above) — the phase-8 fixture canonicalizes this dict as a
  sorted list of `{parent_layer, parent_cluster, children: [sorted pairs]}`
  entries for comparison; **compare as sets, not as ordered lists**, same
  caveat as `extract_eom_clusters` in phase 6.
- `find_duplicates`: `duplicate_distance = np.max(knn_dists.T[0])` — the
  **single largest first-nearest-neighbor distance across the WHOLE
  dataset** sets a **global** threshold for what counts as "duplicate"
  everywhere; one noisy/outlier point with a large 1st-neighbor distance
  inflates the threshold for every other point's comparisons too (this is
  almost certainly intentional-if-surprising — replicate literally, do not
  "fix" it to a per-point or percentile threshold). For every `(i,
  knn_inds[i,j])` pair with `knn_dists[i,j] <= duplicate_distance`:
  canonicalize as `(min(i,k), max(i,k))` and insert into a `set` (automatic
  dedup); the `i == k` self-neighbor case is explicitly skipped via the
  `elif/else` chain (not merely redundant with the set-dedup — it actively
  prevents `(i,i)` pairs from ever being considered). Real cosine-similarity
  nn-descent output essentially never produces exact duplicates in practice
  (see the `find_duplicates_nndescent_n150` fixture's empty result) — use
  the synthetic fixture for actually exercising the pair-detection logic.

## clustering.py (phase 8)

- `build_cluster_layers` is **fully deterministic given its `data` input** —
  it takes **no `random_state` parameter** and has no RNG usage of its own.
  Its only source of nondeterminism risk is `parallel_boruvka`'s
  `reproducible_flag` passthrough (multithreaded races only — verified
  bit-identical to `reproducible_flag=False` at `n_threads=1`, per the
  phase-5 boruvka notes, and re-verified directly for this module's fixture).
  **This is the reason `build_cluster_layers` — not `evoc_clusters` — is the
  phase-8 end-to-end parity target**; see below.
- `mask = condensed_tree.child >= n_samples; cluster_tree =
  mask_condensed_tree(condensed_tree, mask)` — this is a **different mask**
  from `cluster_tree_from_condensed_tree`'s `child_size > 1` mask used
  elsewhere (phase 6 / `extract_eom_clusters`'s caller). `build_cluster_layers`
  uses the `child >= n_samples` (internal-node-only, by id range) variant
  specifically — do not substitute the other mask function; they select
  different (usually overlapping but not identical) row subsets.
- `base_n_clusters is not None` branch (searching for an exact cluster
  count via `_binary_search_for_n_clusters`) is a **separate code path**
  from the default (`base_n_clusters=None`, plain `condense_tree` at
  `base_min_cluster_size`) — **the phase-8 fixture only exercises the
  `base_n_clusters=None` default path.** If you need the binary-search
  branch, `clustering_utilities.py`'s `binary_search_for_n_clusters_n60` /
  `_binary_search_for_n_clusters`-related fixtures cover that function in
  isolation, but there is no fixture for `build_cluster_layers` itself
  invoked with `base_n_clusters` set — port this branch carefully by
  composing the already-fixtured pieces (this is a known coverage gap, not
  an oversight to silently work around).
- Guard before computing the persistence barcode: `if len(cluster_tree.child)
  > 0 and cluster_tree.child[-1] >= n_samples:` — an **empty or degenerate**
  cluster tree (e.g. a single dominant cluster with no internal branching)
  skips the whole barcode/persistence/peaks computation and falls back to
  empty arrays for `births/deaths/parents/lambda_deaths/sizes/
  total_persistence/peaks`; `select_diverse_peaks` then short-circuits
  (`len(peaks)==0`) to zero additional layers, so the function still returns
  correctly with just the base layer. This all-empty fallback path is a
  legitimate output shape (arrays of length 0 / a single-layer result), not
  an error — a port must handle it without crashing on empty-array indexing.
- Base layer is **always** included first (`persistence_scores.append(0.0)`
  — the base layer's persistence score is hardcoded to `0.0`, it is not
  computed), before any peak-derived layers are appended. **Final step
  re-sorts all layers** by `n_clusters_per_layer = [layer.max() + 1 for
  layer in cluster_layers]` **descending** (`np.argsort(...)[::-1]`) — the
  base layer is NOT guaranteed to end up first or last in the final output;
  its position depends on how many clusters it has relative to the
  peak-derived layers. The `build_cluster_layers_n126` fixture's `expected`
  arrays are recorded **already in this final sorted order** — a port's
  output should match that order directly, no additional reordering needed
  on the port side beyond replicating the same sort.
- **`evoc_clusters()` / `EVoC.fit_predict()` are NOT fixtured end-to-end, by
  design — this is a deliberate scope decision, not an oversight.** Tracing
  a real `evoc_clusters(data, n_neighbors=15, n_epochs=10, ...,
  reproducible_flag=True)` call with an instrumented `RandomState` subclass
  (same technique used for the label_propagation/node_embedding fixtures)
  showed **hundreds of numpy-MT draws** even for a 150-point input: `knn_graph`
  draws its `rng_state`/`rng_states` (phase 4, already portable in
  isolation), then `label_propagation_init` recurses (default
  `recursive_init=True`, unlike the phase-7 fixture's `recursive_init=False`)
  — each recursion level draws its own `rng_state` + `approx_n_parts`
  seed-position draws (`approx_n_parts` itself shrinks by `// 4` each level,
  from `np.clip(int(8*sqrt(n)), 256, 16384)`, so the exact recursion depth
  and per-level draw count is **data-size-dependent**, not a fixed constant),
  and once a reduced graph drops below `base_init_threshold` (64) it hits
  sklearn's `PCA(random_state=...)`, whose *own* internal randomized-SVD
  draws from that `RandomState` are effectively unrecordable/unportable
  (sklearn does not expose a stable, versioned draw sequence the way this
  project's own `tau_rand`/numpy-MT-tracing approach relies on). Then
  `node_embedding` draws its own `rng_val` + per-epoch shuffles on top of
  all that. Recording every one of these draws for an arbitrary input size
  would require re-deriving (and keeping in lockstep with) the recursion
  depth and PCA usage as a function of `data.shape[0]` and `approx_n_parts`
  scaling — fragile, and the PCA leg is a hard blocker regardless.
  **`build_cluster_layers` sidesteps all of this** by taking a
  pre-computed embedding as input, which is why it — not `evoc_clusters` —
  is the portable phase-8 capstone. If a full `evoc_clusters` port needs
  validating, compose it from the already-fixtured pieces (knn_graph,
  label_propagation_init with `recursive_init=False` at the top level only
  if your port controls that parameter, node_embedding, build_cluster_layers)
  and validate each stage's boundary against its own fixture rather than
  expecting one end-to-end fixture to cover the whole thing.
