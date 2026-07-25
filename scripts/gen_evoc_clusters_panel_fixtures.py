#!/usr/bin/env python
"""Generate the phase-9 evoc_clusters dataset PANEL fixtures (section 2C of
VALIDATION_PLAN.md).

`evoc_clusters_e2e_ari_check.json` (+ `python_self_agreement_null.json`)
validate the port against Python on exactly ONE synthetic dataset shape (240
points, 24 dims, 6 balanced clusters). This script produces several
additional datasets varying n, dimensionality, cluster count, class balance,
and noise scale -- all using the SAME cosine-friendly orthogonal-axis-centers
construction as `gen_evoc_clusters_ari_fixture.py` (see that script's
docstring, and PCA_VALIDATION.md's "why isn't it make_blobs verbatim"
section: plain Euclidean `make_blobs` loses separation once `knn_graph`
row-normalizes to unit vectors, so centers must be placed along distinct
*directions*, not just distinct locations).

For each dataset this writes two fixtures:
  - `fixtures/clustering/panel_<name>.json`: the data, a fixed Python
    reference labeling (`evoc.EVoC(random_state=PANEL_EVOC_SEED)`), and
    dataset metadata.
  - `fixtures/clustering/panel_<name>_null.json`: Python's own pairwise
    self-agreement ARI across a SMALLER seed panel (0..9, not 0..19 -- this
    is 4 datasets x self-agreement, so a smaller panel keeps this cheap)
    plus simple Python-free-comparable internal metrics (n_clusters found,
    noise fraction) per seed, for the port's own eyeball sanity check
    against.

Usage (from repo root):
    PYTHONPATH=/path/to/evoc-reference /path/to/venv/bin/python \\
        scripts/gen_evoc_clusters_panel_fixtures.py
"""
import itertools
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

import evoc

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "fixtures" / "clustering"

NULL_SEEDS = list(range(10))  # smaller panel than the original's 0..19 (5x the datasets)
PANEL_EVOC_SEED = 42  # fixed Python reference seed, matches the original fixture's convention


def make_cosine_friendly_dataset(n_clusters, n_features, cluster_sizes, noise_scale, center_scale, seed):
    """Same construction as gen_evoc_clusters_ari_fixture.py's
    make_cosine_friendly_blobs, generalized to per-cluster sizes (for class
    imbalance) instead of one uniform n_per.
    """
    rng = np.random.RandomState(seed)
    centers = np.eye(n_clusters, n_features) * center_scale
    chunks = []
    labels = []
    for c, n_per in enumerate(cluster_sizes):
        pts = centers[c] + rng.normal(scale=noise_scale, size=(n_per, n_features))
        chunks.append(pts)
        labels.extend([c] * n_per)
    X = np.vstack(chunks).astype(np.float32)
    order = rng.permutation(len(labels))
    return X[order], np.array(labels)[order]


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def internal_metrics(labels):
    labels = np.asarray(labels)
    n = len(labels)
    n_noise = int(np.sum(labels == -1))
    n_clusters = len(set(labels.tolist()) - {-1})
    return {"n_clusters": n_clusters, "noise_fraction": n_noise / n if n else 0.0}


# Dataset panel: name -> (n_clusters, n_features, cluster_sizes, noise_scale, center_scale, blobs_seed)
DATASETS = {
    # small n, low dims, few clusters, balanced, low noise
    "small_balanced": dict(
        n_clusters=4, n_features=12, cluster_sizes=[25] * 4,
        noise_scale=0.35, center_scale=6.0, blobs_seed=1,
    ),
    # large n, higher dims, more clusters, balanced, moderate noise
    "large_many_clusters": dict(
        n_clusters=8, n_features=32, cluster_sizes=[100] * 8,
        noise_scale=0.6, center_scale=6.0, blobs_seed=2,
    ),
    # class-imbalanced clusters (ratio spread ~1:6)
    "imbalanced": dict(
        n_clusters=5, n_features=16,
        cluster_sizes=[20, 35, 60, 90, 120],
        noise_scale=0.5, center_scale=6.0, blobs_seed=3,
    ),
    # high dimensionality relative to n, high noise
    "high_dim_noisy": dict(
        n_clusters=6, n_features=64, cluster_sizes=[50] * 6,
        noise_scale=0.9, center_scale=6.0, blobs_seed=4,
    ),
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overview = {}

    for name, cfg in DATASETS.items():
        n_clusters = cfg["n_clusters"]
        n_features = cfg["n_features"]
        cluster_sizes = cfg["cluster_sizes"]
        noise_scale = cfg["noise_scale"]
        center_scale = cfg["center_scale"]
        blobs_seed = cfg["blobs_seed"]
        n_samples = sum(cluster_sizes)

        print(f"=== dataset {name}: n={n_samples} dims={n_features} clusters={n_clusters} "
              f"sizes={cluster_sizes} noise={noise_scale} ===", file=sys.stderr)

        X, _y_true = make_cosine_friendly_dataset(
            n_clusters, n_features, cluster_sizes, noise_scale, center_scale, blobs_seed
        )

        reference_labels = evoc.EVoC(random_state=PANEL_EVOC_SEED).fit_predict(X)
        ref_metrics = internal_metrics(reference_labels)
        print(f"  reference (seed={PANEL_EVOC_SEED}): {ref_metrics}", file=sys.stderr)

        data_fixture = {
            "component": "clustering",
            "case": f"panel_{name}",
            "note": (
                f"Dataset panel fixture (VALIDATION_PLAN.md section 2C), variant "
                f"'{name}'. Same cosine-friendly orthogonal-axis-centers "
                f"construction as evoc_clusters_e2e_ari_check.json (see "
                f"gen_evoc_clusters_ari_fixture.py / PCA_VALIDATION.md), varying "
                f"n/dims/cluster count/balance/noise. {n_clusters} clusters, sizes "
                f"{cluster_sizes}, {n_features} features, noise_scale={noise_scale}, "
                f"center_scale={center_scale}, blobs_seed={blobs_seed}. "
                f"python_labels are evoc.EVoC(random_state={PANEL_EVOC_SEED}).fit_predict(X) "
                f"with library defaults."
            ),
            "n_samples": int(X.shape[0]),
            "n_features": n_features,
            "n_true_clusters": n_clusters,
            "cluster_sizes": cluster_sizes,
            "noise_scale": noise_scale,
            "center_scale": center_scale,
            "blobs_seed": blobs_seed,
            "data": X.tolist(),
            "python_labels": reference_labels.tolist(),
            "python_reference_seed": PANEL_EVOC_SEED,
            "python_internal_metrics": ref_metrics,
        }
        with open(OUT_DIR / f"panel_{name}.json", "w") as f:
            json.dump(data_fixture, f)
        print(f"  wrote panel_{name}.json", file=sys.stderr)

        # Smaller self-agreement null: seeds 0..9.
        all_labels = {}
        all_metrics = {}
        for s in NULL_SEEDS:
            labels = evoc.EVoC(random_state=s).fit_predict(X)
            all_labels[s] = np.asarray(labels)
            all_metrics[s] = internal_metrics(labels)

        pairwise = []
        for s1, s2 in itertools.combinations(NULL_SEEDS, 2):
            ari = adjusted_rand_score(all_labels[s1], all_labels[s2])
            pairwise.append({"seed_a": s1, "seed_b": s2, "ari": float(ari)})

        ari_values = [p["ari"] for p in pairwise]
        ari_sorted = sorted(ari_values)
        summary = {
            "n_pairs": len(ari_values),
            "min": min(ari_values),
            "p25": percentile(ari_sorted, 25),
            "median": percentile(ari_sorted, 50),
            "p75": percentile(ari_sorted, 75),
            "max": max(ari_values),
            "mean": statistics.fmean(ari_values),
            "std": statistics.pstdev(ari_values),
        }
        n_clusters_found = [all_metrics[s]["n_clusters"] for s in NULL_SEEDS]
        noise_fracs = [all_metrics[s]["noise_fraction"] for s in NULL_SEEDS]
        null_fixture = {
            "component": "clustering",
            "case": f"panel_{name}_null",
            "note": (
                f"Python evoc.EVoC self-agreement null for dataset panel variant "
                f"'{name}', smaller seed panel (0..9, 45 pairs) than the original "
                f"single-dataset null's 0..19 (190 pairs) since this covers 4 "
                f"additional datasets. Same X as panel_{name}.json."
            ),
            "source_fixture": f"panel_{name}.json",
            "n_samples": int(X.shape[0]),
            "n_features": n_features,
            "seeds": NULL_SEEDS,
            "evoc_version": getattr(evoc, "__version__", "unknown"),
            "pairwise_ari": pairwise,
            "summary": summary,
            "python_internal_metrics_by_seed": all_metrics,
            "python_internal_metrics_summary": {
                "n_clusters_found_min": min(n_clusters_found),
                "n_clusters_found_max": max(n_clusters_found),
                "n_clusters_found_median": statistics.median(n_clusters_found),
                "noise_fraction_min": min(noise_fracs),
                "noise_fraction_max": max(noise_fracs),
                "noise_fraction_median": statistics.median(noise_fracs),
            },
        }
        with open(OUT_DIR / f"panel_{name}_null.json", "w") as f:
            json.dump(null_fixture, f, indent=2)
        print(f"  wrote panel_{name}_null.json: median={summary['median']:.4f} "
              f"min={summary['min']:.4f} max={summary['max']:.4f}", file=sys.stderr)

        overview[name] = {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_clusters": n_clusters,
            "null_median": summary["median"],
            "null_min": summary["min"],
        }

    print("\n=== overview ===", file=sys.stderr)
    for name, o in overview.items():
        print(f"  {name}: {o}", file=sys.stderr)


if __name__ == "__main__":
    main()
