#!/usr/bin/env python
"""Generate large synthetic X arrays for the wasm scaling investigation
(comparison/results.md "Scaling to 10k-30k points"). NOT a correctness
fixture -- no Python reference labels are computed here (re-running the
Python `evoc` reference at n=30000 would itself be slow and isn't needed to
answer a performance question). Uses the SAME cosine-friendly
orthogonal-axis-centers construction as gen_evoc_clusters_ari_fixture.py /
gen_evoc_clusters_panel_fixtures.py (see PCA_VALIDATION.md for why plain
make_blobs doesn't work for this algorithm: knn_graph row-normalizes to
unit vectors, so cluster separation must come from distinct *directions*,
not just distinct locations).

Output: raw flat float32 binary files (row-major n_samples x n_features),
NOT committed (see .gitignore addition) -- multi-MB+ blobs, trivially
regenerable by rerunning this script. Written under scripts/scale_data/.

Usage:
    python scripts/gen_scale_test_data.py
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "scripts" / "scale_data"

N_FEATURES = 32
N_CLUSTERS = 8
NOISE_SCALE = 0.35
CENTER_SCALE = 3.0
SEED = 1234

SIZES = [2000, 10000, 30000]


def make_cosine_friendly_dataset(n_samples, n_clusters, n_features, noise_scale, center_scale, seed):
    rng = np.random.RandomState(seed)
    centers = np.eye(n_clusters, n_features) * center_scale
    base = n_samples // n_clusters
    sizes = [base] * n_clusters
    for i in range(n_samples - base * n_clusters):
        sizes[i] += 1
    chunks = []
    for c, n_per in enumerate(sizes):
        pts = centers[c] + rng.normal(scale=noise_scale, size=(n_per, n_features))
        chunks.append(pts)
    X = np.vstack(chunks).astype(np.float32)
    order = rng.permutation(len(X))
    return X[order]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for n in SIZES:
        X = make_cosine_friendly_dataset(n, N_CLUSTERS, N_FEATURES, NOISE_SCALE, CENTER_SCALE, SEED)
        out_path = OUT_DIR / f"scale_n{n}_d{N_FEATURES}.f32"
        X.tofile(out_path)
        print(f"wrote {out_path} : shape={X.shape} bytes={out_path.stat().st_size}")


if __name__ == "__main__":
    sys.exit(main())
