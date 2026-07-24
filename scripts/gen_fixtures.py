#!/usr/bin/env python
"""Generate golden parity fixtures from the reference evoc implementation.

Usage:
    /scratch/pyenv/bin/python scripts/gen_fixtures.py [component ...]

With no arguments, generates all fixtures whose generator is implemented.
Requires numpy + numba and the evoc reference checkout (env EVOC_REF,
default /scratch/evoc-reference). Fixtures land in fixtures/<component>/.

Fixture files are the only ground truth for parity tests; never hand-edit.
See PORTING.md for the schema and tolerance semantics.
"""

import json
import os
import sys

import numpy as np

# The WASM target is single-threaded; parallel numba kernels can partition
# work (and consume per-thread RNG streams) differently by thread count, so
# ground truth must be generated with exactly one thread.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

EVOC_REF = os.environ.get("EVOC_REF", "/scratch/evoc-reference")
sys.path.insert(0, EVOC_REF)

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixtures")


def write(component, case, tolerance, inputs, expected):
    d = os.path.join(FIXTURES_DIR, component)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, case + ".json")
    with open(path, "w") as f:
        json.dump(
            {
                "component": component,
                "case": case,
                "tolerance": tolerance,
                "inputs": inputs,
                "expected": expected,
            },
            f,
            indent=1,
        )
        f.write("\n")
    print("wrote", os.path.relpath(path, os.path.join(FIXTURES_DIR, "..")))


def gen_rng():
    from evoc.common_nndescent import seed, tau_rand_int, tau_rand

    for s in [0, 42, 123456789]:
        state = np.empty(3, dtype=np.int64)
        seed(state, s)
        initial_state = [int(x) for x in state]

        st = state.copy()
        ints = [int(tau_rand_int(st)) for _ in range(64)]
        final_int_state = [int(x) for x in st]

        st = state.copy()
        floats = [float(tau_rand(st)) for _ in range(64)]

        write(
            "rng",
            f"tau_rand_seed_{s}",
            "exact",
            {"seed": s, "n": 64},
            {
                "initial_state": initial_state,
                "int_sequence": ints,
                "final_int_state": final_int_state,
                "float_sequence": floats,
            },
        )


def gen_disjoint_set():
    from evoc.disjoint_set import (
        ds_rank_create,
        ds_size_create,
        ds_find,
        ds_union_by_rank,
        ds_union_by_size,
    )

    rng = np.random.RandomState(7)
    n = 24
    ops = [[int(rng.randint(n)), int(rng.randint(n))] for _ in range(40)]

    for kind in ["rank", "size"]:
        if kind == "rank":
            ds = ds_rank_create(n)
            union = ds_union_by_rank
            aux_name = "rank"
        else:
            ds = ds_size_create(n)
            union = ds_union_by_size
            aux_name = "size"
        finds = []
        for x, y in ops:
            union(ds, np.int32(x), np.int32(y))
            finds.append(int(ds_find(ds, np.int32(x))))
        write(
            "disjoint_set",
            f"union_{kind}_n{n}",
            "exact",
            {"n": n, "union_ops": ops,
             "note": "after each union(x,y), call find(x) and record it"},
            {
                "find_after_each_union": finds,
                "final_parent": [int(p) for p in ds.parent],
                f"final_{aux_name}": [int(v) for v in getattr(ds, aux_name)],
            },
        )


def gen_distance():
    from evoc.float_nndescent import fast_cosine
    from evoc.int8_nndescent import fast_int_inner_product_dissimilarity
    from evoc.uint8_nndescent import fast_bit_jaccard, fast_bit_jaccard_u64

    rng = np.random.RandomState(99)

    cosine_cases = []
    for dim in [8, 64, 384]:
        x = rng.normal(size=dim).astype(np.float32)
        x /= np.float32(np.linalg.norm(x))
        y = rng.normal(size=dim).astype(np.float32)
        y /= np.float32(np.linalg.norm(y))
        cosine_cases.append((x, y))                      # generic pair
        cosine_cases.append((x, (-x).astype(np.float32)))  # dot<0 -> -tiny branch
        cosine_cases.append((x, x))                       # identical
    write(
        "distance",
        "fast_cosine",
        {"rel": 1e-5, "abs": 1e-7},
        {"pairs": [{"x": x.tolist(), "y": y.tolist()} for x, y in cosine_cases]},
        {"values": [float(fast_cosine(x, y)) for x, y in cosine_cases]},
    )

    int8_cases = []
    for dim in [8, 64, 384]:
        x = rng.randint(-127, 128, size=dim).astype(np.int8)
        y = rng.randint(-127, 128, size=dim).astype(np.int8)
        int8_cases.append((x, y))
    int8_cases.append(
        (np.full(384, 127, dtype=np.int8), np.full(384, 127, dtype=np.int8))
    )
    int8_cases.append(
        (np.full(384, -127, dtype=np.int8), np.full(384, 127, dtype=np.int8))
    )
    write(
        "distance",
        "fast_int_inner_product_dissimilarity",
        "exact",
        {"pairs": [{"x": x.tolist(), "y": y.tolist()} for x, y in int8_cases]},
        {"values": [float(fast_int_inner_product_dissimilarity(x, y)) for x, y in int8_cases]},
    )

    u8_cases = []
    for dim in [8, 32, 128]:  # divisible by 8 so the u64 variant also applies
        x = rng.randint(0, 256, size=dim).astype(np.uint8)
        y = rng.randint(0, 256, size=dim).astype(np.uint8)
        u8_cases.append((x, y))
    u8_cases.append((np.zeros(32, dtype=np.uint8), np.zeros(32, dtype=np.uint8)))  # denom=0
    u8_cases.append((np.full(32, 255, dtype=np.uint8), np.full(32, 255, dtype=np.uint8)))
    write(
        "distance",
        "fast_bit_jaccard",
        "exact",
        {"pairs": [{"x": x.tolist(), "y": y.tolist()} for x, y in u8_cases]},
        {
            "values": [float(fast_bit_jaccard(x, y)) for x, y in u8_cases],
            "values_u64": [float(fast_bit_jaccard_u64(x, y)) for x, y in u8_cases],
        },
    )


def gen_kdtree():
    from evoc.numba_kdtree import (
        build_kdtree,
        rdist,
        point_to_node_lower_bound_rdist,
        simple_heap_push,
        deheap_sort,
        parallel_tree_query,
    )

    rng = np.random.RandomState(1234)

    # --- build ---
    build_cases = [
        ("n100_d4_leaf10", 100, 4, 10),
        ("n37_d3_leaf5", 37, 3, 5),
        ("n20_d2_leaf40_single_node", 20, 2, 40),
    ]
    datasets = {}
    for case, n, d, leaf_size in build_cases:
        data = rng.normal(size=(n, d)).astype(np.float32)
        datasets[case] = (data, leaf_size)
        tree = build_kdtree(data, leaf_size=leaf_size)
        write(
            "kdtree",
            f"build_{case}",
            {"default": "exact", "fields": {"radius": {"rel": 1e-5, "abs": 1e-7}}},
            {"data": data.tolist(), "leaf_size": leaf_size},
            {
                "idx_array": [int(i) for i in tree.idx_array],
                "idx_start": [int(i) for i in tree.idx_start],
                "idx_end": [int(i) for i in tree.idx_end],
                "is_leaf": [bool(b) for b in tree.is_leaf],
                "radius": [float(r) for r in tree.radius],
                "node_bounds_lower": tree.node_bounds[0].tolist(),
                "node_bounds_upper": tree.node_bounds[1].tolist(),
            },
        )

    # --- rdist / point_to_node_lower_bound_rdist ---
    pairs = []
    for dim in [2, 4, 16]:
        x = rng.normal(size=dim).astype(np.float32)
        y = rng.normal(size=dim).astype(np.float32)
        pairs.append((x, y))
    pairs.append((pairs[0][0], pairs[0][0]))  # identical -> 0
    write(
        "kdtree",
        "rdist",
        {"rel": 1e-5, "abs": 1e-7},
        {"pairs": [{"x": x.tolist(), "y": y.tolist()} for x, y in pairs]},
        {"values": [float(rdist(x, y)) for x, y in pairs]},
    )

    lb_cases = []
    for dim in [2, 4, 16]:
        lower = rng.normal(size=dim).astype(np.float32)
        upper = (lower + np.abs(rng.normal(size=dim))).astype(np.float32)
        inside = ((lower + upper) / 2).astype(np.float32)
        outside = (upper + np.abs(rng.normal(size=dim))).astype(np.float32)
        below = (lower - np.abs(rng.normal(size=dim))).astype(np.float32)
        for pt in (inside, outside, below):
            lb_cases.append((lower, upper, pt))
    write(
        "kdtree",
        "point_to_node_lower_bound_rdist",
        {"rel": 1e-5, "abs": 1e-7},
        {
            "note": "call as in the reference: fn(bounds_lower, bounds_upper, pt); "
                    "the reference's parameter names are swapped vs the caller",
            "cases": [
                {"bounds_lower": lo.tolist(), "bounds_upper": up.tolist(), "pt": pt.tolist()}
                for lo, up, pt in lb_cases
            ],
        },
        {
            "values": [
                float(point_to_node_lower_bound_rdist(lo, up, pt))
                for lo, up, pt in lb_cases
            ]
        },
    )

    # --- simple_heap_push + deheap_sort ---
    k = 7
    heap_p = np.full(k, np.inf, dtype=np.float32)
    heap_i = np.full(k, -1, dtype=np.int32)
    pushes = []
    push_returns = []
    for m in range(25):
        p = float(np.float32(rng.uniform(0.0, 10.0)))
        pushes.append({"p": p, "n": m})
        push_returns.append(
            int(simple_heap_push(heap_p, heap_i, np.float32(p), np.int32(m)))
        )
    heap_after_p = heap_p.copy()
    heap_after_i = heap_i.copy()
    sorted_p, sorted_i = deheap_sort(
        heap_p.reshape(1, -1).copy(), heap_i.reshape(1, -1).copy()
    )
    write(
        "kdtree",
        "heap_push_deheap_sort",
        "exact",
        {"k": k, "pushes": pushes},
        {
            "push_returns": push_returns,
            "heap_priorities": [float(v) for v in heap_after_p],
            "heap_indices": [int(v) for v in heap_after_i],
            "sorted_priorities": [float(v) for v in sorted_p[0]],
            "sorted_indices": [int(v) for v in sorted_i[0]],
        },
    )

    # --- parallel_tree_query ---
    data, leaf_size = datasets["n100_d4_leaf10"]
    tree = build_kdtree(data, leaf_size=leaf_size)
    queries = np.vstack([data[:10], rng.normal(size=(5, 4)).astype(np.float32)]).astype(
        np.float32
    )
    for output_rdist in (True, False):
        dists, inds = parallel_tree_query(tree, queries, np.int64(5), output_rdist)
        write(
            "kdtree",
            f"parallel_tree_query_k5_rdist_{str(output_rdist).lower()}",
            {"default": "exact", "fields": {"distances": {"rel": 1e-4, "abs": 1e-6}}},
            {
                "dataset": "build_n100_d4_leaf10 (same data/leaf_size fixture)",
                "data": data.tolist(),
                "leaf_size": leaf_size,
                "queries": queries.tolist(),
                "k": 5,
                "output_rdist": output_rdist,
            },
            {"distances": dists.tolist(), "indices": inds.tolist()},
        )


GENERATORS = {
    "rng": gen_rng,
    "disjoint_set": gen_disjoint_set,
    "distance": gen_distance,
    "kdtree": gen_kdtree,
}


def main():
    targets = sys.argv[1:] or list(GENERATORS)
    for name in targets:
        if name not in GENERATORS:
            sys.exit(f"no generator for {name!r}; known: {', '.join(GENERATORS)}")
        print(f"--- generating {name} fixtures ---")
        GENERATORS[name]()


if __name__ == "__main__":
    main()
