#!/usr/bin/env python3
"""Cross-language (Rust vs Zig) agreement check for `evoc_clusters`/`EVoC`.

Compares Rust's and Zig's `EVoC::fit_predict` (Rust) / `EVoC.fit_predict`
(Zig) output on the same seed panel (0..19) and the same fixture
(`fixtures/clustering/evoc_clusters_e2e_ari_check.json`), for:

  (a) exact partition equality (labels identical up to a permutation of
      cluster ids -- cluster ids are arbitrary, so raw integer equality is
      the wrong bar);
  (b) Adjusted Rand Index between Rust and Zig labels, as a fallback trust
      signal when (a) fails.

Both languages' pipelines are "self-driven": each seeds one tau_rand RNG
state and threads it through every stochastic step (RP-tree/NN-descent
sampling, label-prop partition seeding, randomized PCA, node-embedding
SGD). Since tau_rand itself is bit-identical between Rust and Zig (verified
in phases 1-8), and pipeline structure matches, it was previously unknown
whether the two full pipelines would produce bit-identical output. See
`rust/core/PCA_VALIDATION.md`, "Cross-language (Rust vs Zig) agreement".

Usage:
    # 1. Generate the Rust dump:
    cd rust/core && EVOC_CROSS_LANG_OUT_DIR=/tmp/evoc_cross_lang \
        cargo test --test dump_evoc_clusters_labels_for_cross_lang -- --ignored

    # 2. Generate the Zig dump:
    cd zig && EVOC_FIXTURES=$(pwd)/../fixtures \
        EVOC_CROSS_LANG_OUT_DIR=/tmp/evoc_cross_lang \
        zig test src/dump_evoc_clusters_labels_for_cross_lang.zig

    # 3. Compare:
    python3 scripts/compare_rust_zig_evoc_clusters.py \
        --rust /tmp/evoc_cross_lang/rust_labels.json \
        --zig /tmp/evoc_cross_lang/zig_labels.json \
        --min-median-ari 0.85
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from math import comb


def adjusted_rand_index(a: list[int], b: list[int]) -> float:
    """Matches sklearn.metrics.adjusted_rand_score's definition."""
    assert len(a) == len(b)
    n = len(a)
    a_labels = sorted(set(a))
    b_labels = sorted(set(b))
    a_index = {l: i for i, l in enumerate(a_labels)}
    b_index = {l: i for i, l in enumerate(b_labels)}

    contingency = [[0] * len(b_labels) for _ in range(len(a_labels))]
    for x, y in zip(a, b):
        contingency[a_index[x]][b_index[y]] += 1

    sum_comb_c = sum(comb(c, 2) for row in contingency for c in row)
    row_sums = [sum(row) for row in contingency]
    col_sums = [sum(row[j] for row in contingency) for j in range(len(b_labels))]
    sum_comb_row = sum(comb(r, 2) for r in row_sums)
    sum_comb_col = sum(comb(c, 2) for c in col_sums)
    total_comb = comb(n, 2)

    expected_index = sum_comb_row * sum_comb_col / total_comb
    max_index = 0.5 * (sum_comb_row + sum_comb_col)
    if abs(max_index - expected_index) < 1e-12:
        return 1.0
    return (sum_comb_c - expected_index) / (max_index - expected_index)


def partitions_equal(a: list[int], b: list[int]) -> bool:
    """True iff `a` and `b` induce the exact same partition of indices, up
    to an arbitrary relabeling of cluster ids (a bijection between a's
    labels and b's labels, checked both directions)."""
    if len(a) != len(b):
        return False
    a_to_b: dict[int, int] = {}
    b_to_a: dict[int, int] = {}
    for x, y in zip(a, b):
        if x in a_to_b:
            if a_to_b[x] != y:
                return False
        else:
            a_to_b[x] = y
        if y in b_to_a:
            if b_to_a[y] != x:
                return False
        else:
            b_to_a[y] = x
    return True


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = p * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rust", required=True)
    ap.add_argument("--zig", required=True)
    ap.add_argument("--min-median-ari", type=float, default=0.85)
    ap.add_argument("--require-bit-identical", action="store_true")
    args = ap.parse_args()

    rust_raw = json.load(open(args.rust))
    zig_raw = json.load(open(args.zig))
    rust = rust_raw["labels"] if "labels" in rust_raw else rust_raw
    zig = zig_raw["labels"] if "labels" in zig_raw else zig_raw

    seeds = sorted(set(int(s) for s in rust) & set(int(s) for s in zig), key=int)
    if not seeds:
        print("no shared seeds between dumps", file=sys.stderr)
        return 2

    aris = []
    n_identical = 0
    for seed in seeds:
        a = rust[str(seed)]
        b = zig[str(seed)]
        identical = partitions_equal(a, b)
        ari = 1.0 if identical else adjusted_rand_index(a, b)
        aris.append(ari)
        if identical:
            n_identical += 1
        print(f"seed={seed:2d}  bit-identical-partition={identical!s:5}  ari={ari:.4f}")

    aris_sorted = sorted(aris)
    summary = {
        "n_seeds": len(seeds),
        "n_bit_identical": n_identical,
        "min": aris_sorted[0],
        "p25": percentile(aris_sorted, 0.25),
        "median": percentile(aris_sorted, 0.5),
        "p75": percentile(aris_sorted, 0.75),
        "max": aris_sorted[-1],
    }
    print("\nsummary:", json.dumps(summary, indent=2))

    if "seed0_nn_inds" in rust_raw and "seed0_nn_inds" in zig_raw:
        inds_match = rust_raw["seed0_nn_inds"] == zig_raw["seed0_nn_inds"]
        dists_match = rust_raw["seed0_nn_dists_bits"] == zig_raw["seed0_nn_dists_bits"]
        print(
            f"\nfirst-stage (seed=0 knn_graph) check: nn_inds bit-identical={inds_match}  "
            f"nn_dists bit-identical={dists_match}"
        )

    if args.require_bit_identical and n_identical != len(seeds):
        print(
            f"FAIL: expected bit-identical partitions on all {len(seeds)} seeds, "
            f"got {n_identical}",
            file=sys.stderr,
        )
        return 1

    if summary["median"] < args.min_median_ari:
        print(
            f"FAIL: median ARI {summary['median']:.4f} < threshold {args.min_median_ari}",
            file=sys.stderr,
        )
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
