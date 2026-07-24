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


GENERATORS = {
    "rng": gen_rng,
    "disjoint_set": gen_disjoint_set,
    "distance": gen_distance,
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
