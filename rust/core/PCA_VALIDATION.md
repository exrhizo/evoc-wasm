# PCA / `evoc_clusters` validation methodology (phase 9)

## Executive summary

- `evoc_clusters`/`EVoC::fit_predict` cannot be bit-exact vs. Python (sklearn's
  PCA internals aren't a replayable surface), so it's validated statistically
  instead. See sections below for full detail; this is the short version.
- **Measured a null first:** Python's own seed-to-seed self-agreement (ARI
  across 20 of its own runs) has median 0.52, range 0.07-0.95 on the primary
  fixture dataset. This means single-seed ARI numbers (the old "Rust 0.80,
  Zig 0.89" claims) were nearly meaningless on their own -- replaced with
  pre-registered 20-seed panels gated against this null.
- **Both ports pass** the panel gates on the original dataset and on 4
  additional dataset shapes (varying n/dims/cluster count/balance/noise) --
  see "Panel results" and "Dataset panel".
- **Rust vs. Zig are NOT bit-identical** to each other, even though both are
  driven by the same (already bit-verified) `tau_rand` RNG. Traced to
  float32-ULP-scale summation-order noise in the very first pipeline stage
  (`knn_graph`), which compounds through the seed-sensitive pipeline to a
  median cross-language ARI of 0.67. See "Cross-language agreement".
- **Two real findings surfaced, deliberately not silently patched:**
  1. A seed-collision bug: `evoc_clusters`/`EVoC::new` seed the tausworthe
     state directly from the caller's integer instead of routing through a
     `RandomState`-style intermediate step like Python does, so ~half of all
     adjacent integer seeds (e.g. 1 and 2) produce byte-identical RNG streams
     in both languages. Low priority unless independent seed control matters.
  2. Scale invariance fails: multiplying input rows by a positive constant
     should be a no-op (cosine-distance row-normalization) but isn't exactly
     one in float32, and that ULP noise compounds the same way the
     cross-language divergence does. Both findings point at the same root
     cause -- this pipeline is highly sensitive to any ULP-scale perturbation
     of its inputs, from whatever source. Neither was fixed here (validation
     work only); both are documented with disabled regression tests ready to
     re-enable once addressed.
- **Practical takeaway:** the port's quality is not distinguishable from
  Python's own inherent run-to-run variability on this class of algorithm at
  small-to-medium dataset sizes. That is a real, evidence-based conclusion,
  not an excuse -- but it also means "ARI > 0.8" was never the right bar to
  begin with, and the seed-collision/ULP-sensitivity findings above are the
  more actionable next steps if tighter agreement is wanted later.

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

## Cross-language (Rust vs Zig) agreement

Per VALIDATION_PLAN.md section 2D, this checks Rust's and Zig's
`EVoC::fit_predict` directly against each other (same fixture, same seed
panel 0..19), rather than each against Python. This is a stronger/cheaper
signal than either port's ARI-vs-Python: `tau_rand` itself was already
verified bit-identical between the two languages in phases 1-8, and both
languages implement the same "self-driven" pipeline structure (one
`tau_rand` state threaded through RP-tree/NN-descent sampling, label-prop
partition seeding, randomized PCA, and node-embedding SGD), so it was a
live, previously-untested question whether the two ports would produce
bit-identical output given the same seed.

**Method:** `rust/core/tests/dump_evoc_clusters_labels_for_cross_lang.rs`
(a `#[ignore]`d, manually-run test, not part of `cargo test --workspace`)
and `zig/src/dump_evoc_clusters_labels_for_cross_lang.zig` (a standalone
harness file, not wired into `root.zig`/`zig build test`) each run
`EVoC::fit_predict`/`EVoC.fit_predict` for seeds 0..19 on
`fixtures/clustering/evoc_clusters_e2e_ari_check.json` and dump labels (plus,
for seed 0 only, the raw `nn_inds`/`nn_dists` from the first pipeline stage,
`knn_graph`) to JSON. `scripts/compare_rust_zig_evoc_clusters.py` diffs the
two dumps: partition equality (labels compared up to an arbitrary
permutation of cluster ids, not raw integer equality) and, where not
identical, Adjusted Rand Index. `scripts/compare_rust_zig_evoc_clusters.sh`
runs the whole pipeline end to end (both dumps + the comparison) and is the
regression check (exit non-zero on failure); it is not currently wired into
`scripts/check_all.sh` by default (skips gracefully if `zig` isn't on
`PATH`) but can be, or run standalone.

**Result: NOT bit-identical on any of the 20 seeds.**

| | min | p25 | median | p75 | max |
|---|------|------|--------|------|------|
| ARI(Rust, Zig), seeds 0..19 | 0.1177 | 0.3781 | 0.6704 | 0.7073 | 0.8512 |

**Where the divergence starts:** the very first stochastic pipeline stage.
For seed 0, `knn_graph`'s output neighbor *indices* (`nn_inds`) are
bit-identical between Rust and Zig (all 3600 values), but the neighbor
*distances* (`nn_dists`) are not: 30/3600 values differ, by at most
`1.49e-8` in absolute terms -- i.e. float32-ULP-scale noise, not a
structural/algorithmic difference. This is consistent with ordinary
floating-point non-associativity between two independently-written,
same-algorithm float32 implementations (e.g. summation order in row
normalization or the cosine-distance dot product) rather than a bug: the
two ports were never claimed or required to be bit-exact here (fixture-based
"exact" parity in phases 1-8 only applies when the RNG state is supplied
externally as fixture input, keeping every downstream computation's operand
order pinned; the self-driven pipeline recomputes those operands from
scratch in each language, so tiny ordering differences are expected). The
same ULP-scale noise, propagated through NN-descent/RP-tree candidate
selection (which can flip on ties), randomized PCA, label propagation, and
node-embedding SGD, compounds into the substantially different final
partitions above -- consistent with, not contradicted by, the existing
finding that this pipeline is itself fairly seed-sensitive even
*within* one language (see the Python self-agreement null above, and
Rust's/Zig's own ARI-vs-Python spread).

**Threshold chosen:** `compare_rust_zig_evoc_clusters.sh` gates on
`median(ARI_rust↔zig) >= 0.5`. This is calibrated from what was actually
observed (median 0.67, min 0.12), not picked to make a pre-decided number
pass: it sits below the observed median with real margin, so it functions
as a drift detector for a future change that meaningfully breaks
cross-language agreement (e.g. an accidental algorithmic divergence, not
just ULP noise), while tolerating the inherent seed-to-seed variance already
established by the Python self-agreement null (whose own min was 0.0746
across a wider 190-pair sample). It intentionally does NOT require
bit-identical output, which this investigation found is not the current
(or an easily reachable) state.

**Separate finding, NOT fixed here (out of scope for a validation pass --
flagged per this task's instructions instead of silently patched):** while
generating the seed panel, both Rust's and Zig's dumps showed IDENTICAL
final labels for seed pairs `(1,2)`, `(3,4)`, `(5,6)`, ..., `(17,18)` --
i.e. consecutive odd/even seed pairs collapse to the exact same output,
independently in both languages. This traces to `rng::seed`/`rng.zig`'s
`seed` function (`rng_state.fill(s + 0xFFFF)`), which is a faithful, already
bit-verified port of `evoc/common_nndescent.py`'s `seed()`. The bug is not
in the port of that function -- it's in how `evoc_clusters`/`EVoC::new` use
it: they call this low-level `seed()` directly with the caller's top-level
seed to initialize the tausworthe state. But `tau_rand_int`'s masks
(`0xFFFFFFFE`, `0xFFFFFFF8`, `0xFFFFFFF0` for lanes 0/1/2 respectively) clear
each lane's lowest 1/3/4 bits on first use and every use thereafter, so two
seeds differing only in those masked-off low bits (e.g. `1` and `2`, whose
`+0xFFFF`-shifted values `0x10001`/`0x10000` differ only in bit 0) produce
byte-for-byte identical RNG streams forever after. The real Python
`evoc.EVoC` does NOT do this: it uses
`sklearn.utils.check_random_state(random_state)` to get a NumPy
`RandomState`, then builds the internal `rng_state` array via
`current_random_state.randint(INT32_MIN, INT32_MAX, 3)` (see
`evoc/knn_graph.py`) -- i.e. the user-facing seed only ever seeds NumPy's
Mersenne Twister, which is then sampled to produce a well-distributed
tausworthe state, never handed directly to the low-level `seed()` function.
Both ports' "self-driven" substitution (documented elsewhere in this file as
an intentional, already-approximate stand-in for NumPy's RNG, since matching
NumPy's Mersenne Twister bit-for-bit is out of scope) took a shortcut that
happens to make ~half of all adjacent integer seeds statistically
non-independent of each other. This does not affect the cross-language
comparison's validity (Rust and Zig exhibit the exact same collapsing
behavior, symmetrically, so the 20-seed panel above is genuinely 11
distinct effective seeds tested twice each in both languages, not a bug
that differs by language) but is worth a future fix (e.g. drawing 3
well-separated `tau_rand_int` values from a `seed()`-initialized *scratch*
state to build the real starting `rng_state`, mirroring what
`current_random_state.randint(...)` achieves) if finer-grained seed control
ever matters (e.g. for the seed panels used throughout this validation
effort, or for downstream users relying on `random_state` for genuinely
independent runs).

## Dataset panel (VALIDATION_PLAN.md section 2C)

The ARI panel above (`evoc_clusters_ari.rs` / the "evoc_clusters e2e ARI
panel" Zig test) only ever exercised ONE synthetic dataset shape (240x24,
6 balanced clusters). `scripts/gen_evoc_clusters_panel_fixtures.py` extends
this to 4 additional datasets, all using the same cosine-friendly
orthogonal-axis-centers construction established above (NOT plain
`make_blobs` -- see "why isn't it `make_blobs` verbatim" -- centers are
placed along scaled orthogonal axes so cluster identity survives
`knn_graph`'s row-normalization to unit vectors), varying n, dimensionality,
cluster count, class balance, and noise scale:

| dataset | n | dims | clusters | sizes | noise_scale |
|---|---|---|---|---|---|
| `small_balanced` | 100 | 12 | 4 | 25 each | 0.35 |
| `large_many_clusters` | 800 | 32 | 8 | 100 each | 0.6 |
| `imbalanced` | 325 | 16 | 5 | 20/35/60/90/120 | 0.5 |
| `high_dim_noisy` | 300 | 64 | 6 | 50 each | 0.9 |

For each dataset the same script also measures that dataset's OWN Python
self-agreement null (pairwise ARI across `evoc.EVoC(random_state=s)` for `s`
in 0..9 -- a SMALLER 10-seed panel than the original single-dataset null's
0..19, since this covers 4x the datasets, 45 pairs each instead of 190) and
records a fixed `evoc.EVoC(random_state=42)` reference labeling, all cached
in `fixtures/clustering/panel_<name>.json` / `panel_<name>_null.json`.
Regenerate via:

```
PYTHONPATH=/path/to/evoc-reference /path/to/venv/bin/python \
    scripts/gen_evoc_clusters_panel_fixtures.py
```

(this run used a freshly-built venv at `/wrk/tmp/uvoc-pyenv` with
numpy/scipy/scikit-learn/numba/tqdm installed via `uv`, since the
previously-documented `/scratch/pyenv` and `/wrk/claw-scratch/uvoc-wasm/pyenv`
copies were both stale/broken -- the latter's `python` symlink pointed at a
`uv`-managed interpreter owned by a different user account that no longer
exists on this machine; `/wrk/claw-scratch/uvoc-wasm/evoc-reference` (the
`evoc` source checkout) was still valid and reused as-is).

**Gates**, per dataset, mirroring the original panel's derivation but using
each dataset's OWN null instead of the single shared one (dataset shapes
differ enough in inherent stochasticity -- null medians range 0.27 to 0.88
across the 4 new datasets -- that a single shared gate would be either too
loose for the easy datasets or impossible for the hard ones):
`median(ARI_port↔py) >= null_median - 0.15`, `min(ARI_port↔py) >= 0.01`
(a much looser absolute floor than the single-dataset panel's 0.15, since
some of these datasets' own null_min is as low as 0.0296 -- `imbalanced` in
particular is a genuinely hard clustering problem even for Python with
itself). Implemented in
`rust/core/tests/evoc_clusters_panel_ari.rs::evoc_clusters_dataset_panel_ari_against_python`
and `zig/src/clustering.zig`'s `"evoc_clusters dataset panel ARI against
Python"` test, each running a 10-seed panel (0..9) per dataset.

**Results** (both ports cleared every gate on every dataset):

| dataset | null median (py↔py) | Rust median (vs py) | Zig median (vs py) |
|---|---|---|---|
| `small_balanced` | 0.8822 | 0.7604 | 0.8859 |
| `large_many_clusters` | 0.4050 | 0.6001 | 0.5103 |
| `imbalanced` | 0.2739 | 0.2439 | 0.2635 |
| `high_dim_noisy` | 0.6350 | 0.7246 | 0.7229 |

(Full min/p25/median/p75/max plus per-seed values are printed by both test
suites; see `cargo test --test evoc_clusters_panel_ari -- --nocapture` /
`zig build test`.) Every dataset's median comfortably cleared its gate
(margins ranged from ~0.03 for `imbalanced` -- the hardest case, as expected
given its severe class imbalance and highest noise-to-signal difficulty
among the four -- up to several tenths for the others), and no seed on any
dataset in either language dropped below the 0.01 floor. This is consistent
with the original single-dataset finding: the port's agreement with Python
tracks Python's own self-agreement reasonably closely across a genuinely
varied set of dataset shapes, not just the one hand-tuned dataset used
originally.

**Python-free internal metrics sanity check** (task PART 1 item 3): both
test suites also record, per seed, the number of non-noise clusters found
and the noise-point fraction, and print a comparison against Python's
recorded metrics on the same dataset (fixed reference run). Eyeballing the
printed output (`cargo test ... -- --nocapture` / `zig build test`'s
stdout): both ports' cluster counts and noise fractions land in the same
rough neighborhood as Python's for every dataset (e.g. `imbalanced`: Python
finds 13 clusters/40% noise; Rust's 10-seed panel finds 7-12
clusters/18-36% noise; Zig's finds 9-13 clusters/19-44% noise -- broadly
overlapping, not wildly divergent in either direction). No dataset showed
the "similar ARI but systematically much noisier/emptier" failure mode this
check was meant to catch. This is an eyeball/range check, not a formal
statistical test, per the task's scope.

## Metamorphic property tests (VALIDATION_PLAN.md section 2E)

Two Python-free property tests were added to both languages
(`rust/core/tests/evoc_clusters_metamorphic.rs`;
`zig/src/clustering.zig`'s `"determinism..."` and
`disabled_scale_invariance_..."` tests).

**1. Determinism: PASSES in both languages.** Running
`EVoC::fit_predict`/`evoc_clusters` twice with the same seed on the same
data produces byte-identical label output (checked across seeds 0, 7, 42,
123, on the original single-dataset fixture). This confirms explicitly, as
a regression guard, what the self-driven `tau_rand` design implies but had
never been directly asserted: `EVoC::fit_predict`/`evoc_clusters` has no
hidden non-determinism (e.g. no reliance on hash-map iteration order,
thread scheduling, or uninitialized memory) in either port.

**2. Scale invariance: FAILS in both languages -- documented finding, NOT
fixed (out of scope per task instructions).** The expectation was that
multiplying every input row by a positive constant should not change the
output labels at all, since `knn_graph` row-normalizes every point to a
unit vector (cosine distance) before any stochastic step -- in exact
arithmetic, `x/‖x‖ == (s·x)/‖s·x‖` for any `s > 0`. In practice this is
**not exactly true in float32**: at seed 0, scaling by `0.001` changes
171 of 240 labels in the Rust port (confirmed by direct array comparison,
not just an ARI/permutation check -- this is not a relabeling, the
partitions themselves differ substantially). The two float32 computations
`x / ||x||` and `(x * s) / ||x * s||` do not round to bit-identical results
(different intermediate magnitudes round differently at each step), so the
very first stochastic pipeline stage (`knn_graph`'s NN-descent/RP-tree
candidate selection, which can flip on near-ties) already observes slightly
different floats between the scaled and unscaled runs. That ULP-scale
difference then compounds through the rest of the self-driven,
seed-sensitive pipeline (randomized PCA, label propagation,
node-embedding SGD) -- the exact same mechanism already documented above in
"Cross-language (Rust vs Zig) agreement", where 30/3600 ULP-scale
`nn_dists` differences between two independently-written same-algorithm
implementations were found to compound into a median cross-language ARI of
only 0.67. Scale invariance failing for the same structural reason, within
a single language/implementation, is consistent with (not contradicted by)
that finding -- it is more evidence that this pipeline, taken as a whole,
is extremely sensitive to any ULP-scale perturbation of its input, whatever
the perturbation's source (cross-language float summation order,
scale-then-normalize rounding, or ordinary seed variation).

This is flagged here rather than silently patched, per the task's explicit
instruction: fixing it would require either (a) normalizing before the
scale is ever applied in a way that's insensitive to the scale's magnitude
(not obviously safe against making some other property worse) or (b)
switching `knn_graph`'s row-normalization arithmetic to something less
rounding-sensitive (e.g. compute in f64 internally) -- either is a change
to core algorithm files (`knn_graph`/`clustering.rs`/`clustering.zig`),
explicitly out of scope for this validation-only task. The property tests
are kept in the suite in a disabled state (`#[ignore]` in Rust, a
`disabled_`-prefixed non-`test` function in Zig, matching that language's
idiom for "written but not run by default") with an exact-equality
assertion preserved as-written (not loosened to an ARI/permutation check)
so the test continues to state the property that was actually asked for,
and will fail loudly again if anyone re-enables it after a future change to
the normalization/RNG pipeline.
