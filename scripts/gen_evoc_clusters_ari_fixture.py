#!/usr/bin/env python
"""Generate the phase-9 evoc_clusters end-to-end ARI-check fixture.

Runs Python's real `evoc.EVoC().fit_predict(X)` on a small synthetic
make_blobs dataset with a fixed seed, and writes the input data + resulting
labels to a checked-in JSON fixture. The Rust port's `evoc_clusters` (which
uses randomized PCA + this project's own `tau_rand` RNG instead of sklearn's
internal randomized-SVD draws -- see PORT_NOTES.md and PCA_VALIDATION.md) is
NOT expected to reproduce these labels bit-for-bit or even up to a fixed
relabeling; it is compared via Adjusted Rand Index (ARI) against this
fixture, with a pragmatic bar of ARI > 0.8 (see
`rust/core/tests/evoc_clusters_ari.rs`).

Usage (from repo root):
    /scratch/pyenv/bin/python scripts/gen_evoc_clusters_ari_fixture.py

Requires numpy, scipy, scikit-learn, numba, tqdm and a working `evoc`
reference on PYTHONPATH (see PORTING.md for the reference clone location).
"""
import json
import sys
from pathlib import Path

import numpy as np

import evoc

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "fixtures" / "clustering" / "evoc_clusters_e2e_ari_check.json"

N_CLUSTERS = 6
N_FEATURES = 24
N_PER_CLUSTER = 40
NOISE_SCALE = 0.5
CENTER_SCALE = 6.0
BLOBS_SEED = 7
EVOC_SEED = 7


def make_cosine_friendly_blobs(n_clusters, n_features, n_per, noise_scale, center_scale, seed):
    """Synthetic data suited to evoc's cosine-distance assumption (see
    `evoc_clusters`'s docstring: float data is treated as cosine-distance
    embeddings). Plain `sklearn.datasets.make_blobs` output clusters poorly
    under evoc's defaults because knn_graph row-normalizes every point to
    unit norm before building the graph -- if blob centers aren't spread
    across distinct *directions* from the origin, normalization collapses
    the very separation make_blobs created. Centers here are placed along
    orthogonal axes (scaled identity rows) so each cluster keeps a distinct
    direction after normalization.
    """
    rng = np.random.RandomState(seed)
    centers = np.eye(n_clusters, n_features) * center_scale
    chunks = []
    labels = []
    for c in range(n_clusters):
        pts = centers[c] + rng.normal(scale=noise_scale, size=(n_per, n_features))
        chunks.append(pts)
        labels.extend([c] * n_per)
    X = np.vstack(chunks).astype(np.float32)
    # shuffle so cluster order isn't trivially contiguous
    order = rng.permutation(len(labels))
    return X[order], np.array(labels)[order]


def main():
    X, _y_true = make_cosine_friendly_blobs(
        N_CLUSTERS, N_FEATURES, N_PER_CLUSTER, NOISE_SCALE, CENTER_SCALE, BLOBS_SEED
    )

    labels = evoc.EVoC(random_state=EVOC_SEED).fit_predict(X)

    fixture = {
        "component": "clustering",
        "case": "evoc_clusters_e2e_ari_check",
        "note": (
            "NOT a bit-exact/tolerance fixture like the rest of this port -- "
            "compared via Adjusted Rand Index, see PCA_VALIDATION.md. Data is "
            "synthetic cosine-friendly blobs (see make_cosine_friendly_blobs in "
            "the generator script): 6 clusters x 30 points, 24 features, cluster "
            "centers on scaled orthogonal axes, noise_scale=0.8, seed=7. Labels "
            "are evoc.EVoC(random_state=7).fit_predict(X) with the library's own "
            "defaults (n_neighbors=15, n_epochs=50, base_min_cluster_size=5, "
            "min_samples=5, noise_level=0.5)."
        ),
        "n_samples": X.shape[0],
        "n_features": N_FEATURES,
        "data": X.tolist(),
        "python_labels": labels.tolist(),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(fixture, f)

    print(f"Wrote {OUT_PATH}", file=sys.stderr)
    print(f"n_clusters (incl. noise as -1): {len(set(labels.tolist()))}", file=sys.stderr)


if __name__ == "__main__":
    main()
