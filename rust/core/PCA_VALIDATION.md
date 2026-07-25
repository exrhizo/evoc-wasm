# PCA / `evoc_clusters` validation methodology (phase 9)

This document exists because phase 9 (`evoc_clusters`/`EVoC::fit_predict`,
`rust/core/src/pca.rs` + `rust/core/src/clustering.rs`) is validated
**differently** from every other component in this port. Read this before
assuming a lowered bar applies elsewhere -- it does not: everything else in
this repo (phases 1-8) is still held to bit-exact or documented-tolerance
fixture parity per `PORTING.md`. Only the PCA-touching parts of phase 9 use
the methodology below, because bit-exact parity there is not achievable even
in principle.

## Why this component can't be fixtured like the rest

`evoc.EVoC.fit_predict`'s default path (`node_embedding_init="label_prop"`)
calls `label_propagation_init` with `recursive_init=True` (the library
default). Each recursion level partitions the graph and recurses into a
smaller reduced graph; once a reduced graph drops below
`base_init_threshold` (64), the reference calls sklearn's
`PCA(n_components=..., random_state=...).fit_transform(data)`. Sklearn's
randomized-SVD solver draws from that `RandomState` via internal, undocumented,
version-dependent operations (LAPACK calls, internal reshaping, etc.) that
are not a stable "record the draws and replay them" surface the way this
project's own `tau_rand` calls are. There is no way to port this bit-for-bit
without literally reimplementing sklearn's internal solver revision-for-
revision, which upstream itself does not guarantee stable across releases.

Phase 8's `build_cluster_layers` sidestepped this entirely by taking a
pre-computed embedding as input. Phase 9 is explicitly in scope (per the
task directive overriding the earlier "out of scope" call), so this port
takes the approach below instead of skipping it.

## The approach

1. **Implement the real algorithm, not a replay.** `rust/core/src/pca.rs`
   implements the Halko/Martinsson/Tropp randomized SVD algorithm --
   random Gaussian projection, `n_power_iterations` power-iteration steps
   with QR re-orthonormalization between each, then an exact (Jacobi) SVD of
   the resulting small projected matrix -- which is the actual algorithm
   sklearn's `PCA(svd_solver='randomized')` is based on. This is not an
   approximation of the reference in the "close enough" sense; it is the same
   published algorithm, just with its random projection matrix drawn from
   this project's own `tau_rand` RNG (seeded, deterministic) instead of
   numpy's Mersenne Twister.
2. **The whole `evoc_clusters` pipeline is self-driven by one `tau_rand`
   state.** Rather than threading pre-recorded numpy-MT draws through (as the
   phase 7 fixture-parity functions do for their *own* isolated unit tests),
   `evoc_clusters` seeds a single `RngState` and advances it through every
   stochastic step in the pipeline: RP-tree construction and NN-descent
   candidate sampling (`knn_graph`), label-propagation partition seeding and
   tie-breaks (`label_propagation_init_recursive`, phase 9's new recursive
   port in `label_propagation.rs`), the randomized-PCA base case, and node
   embedding SGD (`node_embedding_repr_self_driven` in `node_embedding.rs`).
   This keeps the whole pipeline deterministic (same seed -> same output) and
   cross-language reproducible against the Zig port (once ported there too),
   without pretending to replay numpy's MT draw-for-draw.
3. **Validate with two independent checks, neither of which is bit-exact
   fixture parity:**

   - **Reconstruction-quality unit tests** (`rust/core/src/pca.rs`'s
     `#[cfg(test)]` module): a synthetic rank-2 signal embedded in 20-D with
     small Gaussian noise is projected to 2 components and the captured
     variance ratio is checked against a >90% threshold
     (`low_rank_reconstruction_quality`), plus a basic output-shape check and
     a standard-normal-draw sanity check (mean/variance) for the Box-Muller
     sampler `pca.rs` uses to build the random projection matrix.
   - **End-to-end Adjusted Rand Index (ARI) panel against real Python
     `evoc.EVoC`** (`rust/core/tests/evoc_clusters_ari.rs`,
     `zig/src/clustering.zig`): a synthetic dataset is clustered by both the
     real Python reference (`evoc.EVoC(random_state=42).fit_predict(X)`,
     cached as a fixture -- see below) and this port's `EVoC::fit_predict`
     run across a **pre-registered panel of 20 seeds (0..19)**, and each of
     the 20 label assignments is compared against the cached Python labels
     via ARI. The panel distribution (min/p25/median/p75/max) is asserted
     against gates derived from a measured null rather than a single
     cherry-picked seed against a fixed threshold -- see the "Python-vs-Python
     null" section below for the full history of why this replaced the
     original single-seed >0.8 bar, and "Panel results" for current numbers
     and gate definitions.

## Fixture dataset and why it isn't `make_blobs` verbatim

`scripts/gen_evoc_clusters_ari_fixture.py` generates the fixture consumed by
`evoc_clusters_ari.rs`. The dataset is **not** plain
`sklearn.datasets.make_blobs` output -- an early attempt at this fixture used
exactly that (200 points, 16 features, 5 centers, `cluster_std=1.0`) and both
the Python reference and this Rust port produced heavily-fragmented,
noise-heavy clusterings (ARI ~0.13 against each other, and neither recovered
the 5 true blobs well: Python found 88/200 points as noise across 10 small
clusters). The reason is `evoc_clusters`' documented assumption that float
input is a **cosine-distance** embedding (`knn_graph` unit-normalizes every
row before building the graph) -- generic Euclidean blobs placed at
arbitrary locations lose their separation once projected onto the unit
sphere unless the *directions* from the origin, not just the raw Euclidean
positions, are distinct per cluster.

The fixture generator instead places cluster centers along scaled orthogonal
axes (`np.eye(n_clusters, n_features) * center_scale`) with Gaussian noise,
which keeps each cluster's direction distinct after row-normalization --
matching the kind of CLIP-vector-like embedding data `evoc` targets in
practice. With this dataset (6 clusters x 40 points, 24 features,
`noise_scale=0.5`, `center_scale=6.0`, seed 7) both the Python reference and
this port recover clean, well-separated clusterings. (The original
single-seed number reported here, "ARI 0.8033 at seed 42," has been
superseded by the seed-panel methodology below -- see "Panel results.") The
fixture JSON (`fixtures/clustering/evoc_clusters_e2e_ari_check.json`) caches
both the input data and the Python reference labels (`random_state=42`) so
the test suite never needs a live Python/sklearn/numba environment to run.

## Regenerating the fixture

```
uv venv /path/to/venv --python 3.12
uv pip install --python /path/to/venv/bin/python numpy scipy scikit-learn numba tqdm
PYTHONPATH=/path/to/evoc-reference /path/to/venv/bin/python \
    scripts/gen_evoc_clusters_ari_fixture.py
```

(`/scratch/pyenv` and `/scratch/evoc-reference` per PORTING.md's documented
locations, if available in your environment; substitute any Python 3.12
environment with those packages plus the `evoc` reference source on
`PYTHONPATH` otherwise.)

## What this does and does not claim

- Does NOT claim bit-exact parity with sklearn's PCA, numpy's MT draws, or
  the Python reference's exact cluster label assignment. That is impossible
  by construction once PCA is reached, and is explicitly acknowledged as
  such rather than silently approximated.
- Does NOT lower the bar for phases 1-8 or any non-PCA part of phase 9
  (`knn_graph`, `graph_construction`, `boruvka`, `cluster_trees`,
  `label_propagation`'s non-recursive path, `node_embedding`'s
  fixture-covered kernels, `clustering_utilities`, `build_cluster_layers`) --
  those remain bit-exact/tolerance-fixture-verified as before.
- DOES claim: the randomized PCA is a faithful implementation of the
  published Halko/Martinsson/Tropp algorithm, its reconstruction quality
  meets a reasonable bar on synthetic low-rank data, and the full
  `evoc_clusters` pipeline built on top of it produces clusterings whose
  agreement with the real Python reference, measured across a pre-registered
  20-seed panel, is not dramatically worse than Python's own measured
  seed-to-seed self-agreement on the same representative (if hand-tuned for
  cosine-friendliness) synthetic dataset -- see "Panel results" below.

## Python-vs-Python null: how variable is Python's own output?

The original single-seed ARI numbers (Rust 0.80 @ seed 42, Zig 0.89 @ seed
29, observed range 0.51-0.89 across seeds tried informally) were each a
single sample and not trustworthy validation on their own -- a single seed's
ARI-vs-Python could land almost anywhere in a wide range by chance alone,
independent of port correctness. `scripts/measure_python_null.py` measures
whether Python's *own* `evoc.EVoC.fit_predict` is this variable run-to-run,
on the exact same X as the ARI fixture: pairwise ARI across
`EVoC(random_state=s)` for s in 0..19 (190 pairs), results cached in
`fixtures/clustering/python_self_agreement_null.json`. Result: Python's own
self-agreement is broad:

| stat | value |
|------|-------|
| min | 0.0746 |
| p25 | 0.4226 |
| median | 0.5247 |
| p75 | 0.6329 |
| max | 0.9459 |
| mean | 0.5034 |
| std | 0.1972 |

This indicates the port's seed-to-seed variability observed earlier was
largely inherent algorithm stochasticity (small dataset, randomized PCA,
HDBSCAN-style cluster extraction all being seed-sensitive), not primarily a
port quality gap -- confirmed below by the full seed-panel comparison.

## Panel results: pre-registered 20-seed comparison against the null

Per VALIDATION_PLAN.md section 2B, the single-seed ARI tests were replaced
with a pre-registered panel: `EVoC::fit_predict`/`evoc_clusters` is run for
seeds 0..19 (the SAME 20 seeds used to measure the Python self-agreement
null above, fixed in advance, not chosen post-hoc) on the same fixture
dataset, and each run's labels are compared via ARI against the cached
Python reference labels (`random_state=42`,
`fixtures/clustering/evoc_clusters_e2e_ari_check.json`). Implemented in
`rust/core/tests/evoc_clusters_ari.rs::evoc_clusters_e2e_ari_panel_against_python_evoc`
and `zig/src/clustering.zig`'s `"evoc_clusters e2e ARI panel against Python
evoc.EVoC"` test.

**Gates** (pre-registered, applied identically to both languages):

- `median(ARI_port↔py) >= median(ARI_py↔py) - 0.15` = `0.5247 - 0.15` =
  **0.3747**. The 0.15 margin accounts for port-vs-Python(seed 42) not being
  quite the same comparison as the null's Python-vs-Python pairwise measure
  (a different implementation of the same algorithm, compared against one
  fixed reference run rather than averaged pairwise over many Python runs).
  This gate does NOT require the port to beat Python's own internal
  consistency -- only to not be dramatically worse than Python is with
  itself.
- `min(ARI_port↔py) >= 0.15` over the panel, a floor calibrated loosely
  below Python's own observed floor (0.0746) as a smoke alarm for
  pathological/broken seeds (e.g. a crash-adjacent degenerate clustering)
  rather than ordinary seed-sensitivity.

**Results:**

| language | seeds | min | p25 | median | p75 | max |
|----------|-------|------|------|--------|------|------|
| Python self-null (py vs py) | 0..19 (190 pairs) | 0.0746 | 0.4226 | 0.5247 | 0.6329 | 0.9459 |
| Rust (vs Python seed=42) | 0..19 (20 runs) | 0.2637 | 0.4851 | 0.7500 | 0.8737 | 0.9459 |
| Zig (vs Python seed=42) | 0..19 (20 runs) | 0.5147 | 0.6915 | 0.7514 | 0.7841 | 0.8678 |

Both ports comfortably clear both gates (median gate 0.3747, min gate
0.15). Notably, both ports' median ARI-vs-Python (0.75) *exceeds* Python's
own median self-agreement (0.52) -- plausible since each port is compared
against one fixed, relatively "typical" Python reference run rather than
averaged pairwise against 19 other equally-random Python runs, so this is
not evidence the port is "better than Python," just a reflection of the
different comparison structure (fixed reference vs. all-pairs). Zig's panel
is additionally tighter (narrower IQR, higher min) than Rust's on this
dataset; both are within the pass region and this difference is not
currently investigated further (see VALIDATION_PLAN.md section 2D for
planned Rust-vs-Zig direct comparison, which is a stronger/cheaper signal
than either port's ARI-vs-Python and remains a follow-up).

This panel-vs-null comparison supersedes the single-seed claims (Rust 0.8033
@ seed 42, Zig 0.8930 @ seed 29) throughout this document and `STATUS.md`.
