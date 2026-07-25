# Plan: trustworthy evoc_clusters + performant WASM

Status: plan only (nothing here is implemented yet). Author: claude, 2026-07-25.

## 1. Where we stand (quality assessment)

**Strong:**
- Phases 1–8 are held to bit-exact / documented-tolerance fixture parity in two
  independently written implementations (~20k LOC total, zero runtime deps in
  either core). Two independent ports agreeing bit-for-bit is itself a powerful
  correctness argument — a shared bug would have to be made twice.
- Quality gates are real and enforced: `cargo fmt`/`clippy -D warnings`/tests,
  `zig fmt --check`/`zig build test`, unified in `scripts/check_all.sh` + CI.
- Known non-portabilities (sklearn PCA internals, numpy 2.x AVX-512 argsort
  tie-breaks) are documented explicitly rather than papered over with loosened
  tolerances.

**Weak (the honest list):**
1. **Phase 9 ARI validation is a single seed on a single dataset.** Zig's
   reported 0.89 came after trying a handful of seeds (observed range
   0.51–0.89) — a post-hoc selected maximum is a biased estimate, not a
   measurement. Rust's 0.8033 at seed 42 has no disclosed seed history either.
2. **We don't know the null distribution.** We have never measured how much
   Python `evoc` disagrees *with itself* across `random_state`s on this data.
   Without that, we can't tell whether 0.51–0.89 is "our port has a quality
   gap" or "that's just how stochastic this algorithm is."
3. **Rust and Zig phase 9 pipelines have never been compared to each other**
   (different seeds, cross-parity never asserted), even though both are
   self-driven by the same `tau_rand` design — the one place bit-parity might
   actually be recoverable.
4. Scope gaps, documented but real: Zig lacks `build_cluster_layers`'s
   `base_n_clusters` branch; both lack `approx_n_clusters` single-layer
   branch, `return_duplicates`, int8/uint8 paths.
5. Nothing has been built or tested on wasm32 yet; `comparison/results.md`
   is an empty skeleton.

## 2. Statistical validation plan (the "solid" part)

Principle: replace one cherry-picked number with **pre-registered
distributions compared against a measured null**. Order matters — step A is
the highest-information experiment and gates interpretation of everything
else.

### A. Measure the null: Python self-agreement
Run Python `evoc.EVoC(random_state=s).fit_predict(X)` for a fixed panel of
seeds (e.g. s ∈ 0..19) on the existing fixture dataset. Compute pairwise ARI
among Python's own runs → distribution `ARI_py↔py`.
- If Python's self-agreement is itself broad (median ~0.7–0.85), then our
  observed 0.51–0.89 spread is inherent algorithm stochasticity and the port
  is fine — and we should say so with data.
- If Python self-agreement is tight (~0.95+), our spread indicates a real
  quality gap in the self-driven pipeline (likely in the recursive
  label-prop/PCA leg) and we debug before claiming done.

### B. Pre-registered seed panels for both ports
Fix the panel **before** running: seeds 0..19 for Rust and Zig each, no
substitutions. For each seed compute ARI vs the Python reference labels.
Report min / quartiles / median, not max. Acceptance gates (pre-registered):
- median(ARI_port↔py) ≥ median(ARI_py↔py) − small margin (0.05), i.e. the
  port agrees with Python about as well as Python agrees with itself;
- distributions not significantly worse by a rank test (Mann-Whitney U,
  one-sided, α = 0.05);
- a hard floor on min (e.g. ≥ 0.5) as a smoke alarm for pathological seeds.
The shipped test then asserts the *panel median*, not one seed.

### C. Dataset panel, not one dataset
3–5 synthetic datasets varying n (200–2000), dims, cluster count, imbalance,
noise scale — all cosine-friendly per PCA_VALIDATION.md's finding — plus one
realistic embedding sample (small real sentence/CLIP-style vectors) so we're
not only validating on data shaped like our generator. Same A/B protocol per
dataset. Also record Python-free internal metrics per run (cluster count,
noise fraction, cosine silhouette) and check the port's distribution matches
Python's — catches "similar ARI but systematically noisier" failure modes.

### D. Cross-language agreement (the strongest cheap signal)
Run Rust and Zig with the **same** seed on the same data. Two outcomes:
- Best case: labels bit-identical → assert it across the whole seed panel.
  Both pipelines are tau_rand-self-driven with the same structure; the only
  known obstacle is the independently written PCA (QR/Jacobi ordering,
  float-summation order). Worth one focused investigation; if PCA is the
  only divergence point, consider pinning one summation order to make the
  self-driven path bit-identical cross-language like everything else.
- Fallback: assert ARI(rust, zig) is very high (≥ 0.95 median) across the
  panel and document why bit-parity is unattainable.

### E. Metamorphic / property tests (no Python needed)
- Determinism: same seed twice → byte-identical labels (both langs).
- Permutation invariance: shuffle rows → labels shuffle correspondingly
  (ARI = 1.0 up to relabeling) — note evoc is *not* strictly permutation
  invariant through RNG-driven paths, so run this as ARI-near-1 with the
  measured null, and document.
- Scale invariance: multiplying all rows by a constant must not change
  labels (cosine normalization guarantee — this one should be exact).

### F. Localize the seed-sensitivity (diagnostic, if A says we have a gap)
Stage-swap experiment using fixtures: feed Python's recorded intermediate
(e.g. its label-prop init or PCA output) into our downstream, and our
upstream into comparison with Python's downstream. Whichever swap collapses
the ARI variance identifies the stage responsible.

Deliverable for this section: `scripts/validation_panel.py` + native test
updates + a short results write-up appended to `PCA_VALIDATION.md` replacing
the single-number claims with distributions.

## 3. WASM: performant and trustworthy

1. **Builds.** Rust: `wasm32-unknown-unknown`, plain `extern "C"` exports
   (no wasm-bindgen needed — core is dependency-free), flat
   `Float32Array`/pointer+len API. Zig: `wasm32-freestanding` export the same
   C-shaped API. Identical API surface so the JS harness and eventual npm
   package are implementation-agnostic (per the "whichever performs best is
   the package" decision).
2. **Trust: run the fixture suite *inside* wasm.** A Node runner loads each
   .wasm, feeds every phase 1–8 fixture through it, and asserts the same
   bit-exact results as native. Wasm's deterministic IEEE-754 semantics (no
   FMA contraction, no x87) actually make this *more* reproducible than
   native — this proves the parity story survives the target platform, which
   is the whole point.
3. **Benchmarks.** Per-stage timings (knn_graph, boruvka MST, node embedding,
   cluster layers, end-to-end fit_predict), binary size (opt + stripped +
   gzip), peak linear-memory, across 2–3 dataset sizes; Node 20 and one
   browser. Fill `comparison/results.md` for real. Include native Python
   (numba) as the reference row.
4. **Perf levers, in order of expected payoff:** (a) zero-copy I/O into
   linear memory; (b) `+simd128` builds (distance kernels and SGD inner loops
   are the hot spots — but verify bit-parity is preserved or gate SIMD builds
   behind their own fixture run); (c) arena allocation for the per-fit
   scratch; (d) threads/SharedArrayBuffer only later if a real workload
   demands it (COOP/COEP headers make it deployment-hostile).
5. **Pick the winner** on the measured data → that one gets the npm packaging
   pass (deferred until then, per prior decision).

## 4. Streaming rows & refit (design sketch, future)

evoc is batch by nature, but the pieces admit a practical incremental story:

- **API surface** (wasm-friendly, state lives in linear memory):
  `init(opts, seed)` → `push_rows(ptr, n)` → `fit()` → `labels()/layers()`
  → `push_rows(...)` → `refit()` → … plus `serialize()/deserialize()` for
  save/load of model state.
- **Levels of "refit", cheapest first:**
  1. *Assign-only:* new rows get labels from nearest existing embedded
     neighbors (kNN vote in embedding space). No model change; O(new·log n).
  2. *Warm refit:* append rows, incrementally update the kNN graph
     (NN-descent naturally supports inserting points into existing heaps and
     re-descending locally), warm-start the node embedding from previous
     coordinates (new points initialized at their neighbors' centroid), then
     rebuild MST + cluster layers (those stages are cheap relative to
     knn/embedding). Deterministic given seed + insertion order.
  3. *Full refit:* rerun everything; the fallback and the ground truth.
- **Validation of refit:** ARI(warm refit, full refit) across the seed panel
  must stay high (bar set from section 2's measured null); drift triggers a
  scheduled full refit. This reuses the exact statistical machinery from
  section 2 — build that first.

## 5. Sequencing

1. Section 2A+2B (Python null + seed panels) — highest information per hour,
   directly answers "are we solid."
2. 2D cross-language same-seed check (may upgrade phase 9 to bit-parity).
3. 2C/2E dataset panel + property tests; update PCA_VALIDATION.md with
   distributions; fix anything 2F uncovers.
4. Wasm builds + fixture-suite-in-wasm + benchmark harness → fill
   `comparison/results.md` → pick the winner.
5. npm packaging for the winner.
6. Streaming/refit API prototype (design above), validated with the
   section-2 machinery.
