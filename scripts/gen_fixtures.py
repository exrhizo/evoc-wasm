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


def _unit_rows(rng, n, d):
    data = rng.normal(size=(n, d)).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True).astype(np.float32)
    return data.astype(np.float32)


def gen_nndescent():
    from evoc.common_nndescent import make_heap, flagged_heap_push, build_candidates
    from evoc.float_nndescent import (
        make_float_forest,
        nn_descent_float,
        nn_descent_float_sorted,
    )

    rng = np.random.RandomState(4242)

    # --- flagged_heap_push on a single heap row ---
    k = 7
    heap_i, heap_p, heap_f = make_heap(1, k)
    pi, pp, pf = heap_i[0], heap_p[0], heap_f[0]
    pushes = []
    returns = []
    for m in range(20):
        p = float(np.float32(rng.uniform(0.0, 5.0)))
        n_val = int(rng.randint(0, 15))
        pushes.append({"p": p, "n": n_val})
        returns.append(int(flagged_heap_push(pp, pi, pf, np.float32(p), np.int32(n_val))))
    write(
        "nndescent",
        "flagged_heap_push",
        "exact",
        {"k": k, "pushes": pushes,
         "note": "fresh make_heap row (indices -1, priorities inf, flags 0); "
                 "apply pushes in order, record each return"},
        {
            "push_returns": returns,
            "heap_priorities": [float(v) for v in pp],
            "heap_indices": [int(v) for v in pi],
            "heap_flags": [int(v) for v in pf],
        },
    )

    # --- build_candidates ---
    n_points, k = 30, 8
    graph_i, graph_p, graph_f = make_heap(n_points, k)
    for i in range(n_points):
        for _ in range(k + 4):
            j = int(rng.randint(0, n_points))
            if j == i:
                continue
            d = float(np.float32(rng.uniform(0.1, 2.0)))
            flagged_heap_push(
                graph_p[i], graph_i[i], graph_f[i], np.float32(d), np.int32(j)
            )
    in_graph = {
        "indices": graph_i.tolist(),
        "priorities": [[float(v) for v in row] for row in graph_p],
        "flags": graph_f.tolist(),
    }
    rng_state = np.array([12345, 67890, 13579], dtype=np.int64)
    max_candidates = 6
    result = build_candidates(
        (graph_i, graph_p, graph_f), np.int64(max_candidates), rng_state, 1
    )
    new_cand, old_cand = result[0], result[1]
    write(
        "nndescent",
        "build_candidates",
        "exact",
        {
            "current_graph": in_graph,
            "max_candidates": max_candidates,
            "rng_state": [12345, 67890, 13579],
            "n_threads": 1,
            "note": "current_graph arrays are the direct inputs (reconstruct, don't re-push); "
                    "build_candidates may mutate graph flags — graph_flags_after is expected too",
        },
        {
            "new_candidate_indices": new_cand.tolist(),
            "old_candidate_indices": old_cand.tolist(),
            "graph_flags_after": graph_f.tolist(),
            "rng_state_after": [int(v) for v in rng_state],
        },
    )

    # --- make_float_forest (2 trees, serial path) ---
    data = _unit_rows(rng, 64, 6)
    rng_states = np.array([[111, 222, 333], [444, 555, 666]], dtype=np.int64)
    forest = make_float_forest(data, rng_states.copy(), 10, 200)
    write(
        "nndescent",
        "make_float_forest_n64_d6_leaf10",
        "exact",
        {
            "data": data.tolist(),
            "rng_states": rng_states.tolist(),
            "leaf_size": 10,
            "max_depth": 200,
        },
        {"leaf_arrays": [la.tolist() for la in forest]},
    )

    # --- nn_descent_float / nn_descent_float_sorted ---
    # leaf_array is a required input in practice (the None default is unusable:
    # it is passed straight into init_rp_tree_float). Build one via the forest
    # and record it verbatim — the ports take leaf_array as an input.
    data = _unit_rows(rng, 150, 8)
    nnd_rng_states = np.array([[111, 222, 333], [444, 555, 666]], dtype=np.int64)
    nnd_forest = make_float_forest(data, nnd_rng_states.copy(), 10, 200)
    max_leaf = max(la.shape[1] for la in nnd_forest)
    leaf_array = np.vstack(
        [
            np.pad(la, ((0, 0), (0, max_leaf - la.shape[1])), constant_values=-1)
            for la in nnd_forest
        ]
    )
    for name, fn in [
        ("nn_descent_float", nn_descent_float),
        ("nn_descent_float_sorted", nn_descent_float_sorted),
    ]:
        st = np.array([987654321, 192837465, 1029384756], dtype=np.int64)
        inds, dists = fn(
            data,
            10,
            st,
            max_candidates=12,
            n_iters=6,
            delta=0.001,
            delta_improv=None,
            leaf_array=leaf_array,
            verbose=False,
        )
        write(
            "nndescent",
            f"{name}_n150_d8_k10",
            {"default": "exact", "fields": {"distances": {"rel": 1e-4, "abs": 1e-6}}},
            {
                "data": data.tolist(),
                "n_neighbors": 10,
                "rng_state": [987654321, 192837465, 1029384756],
                "max_candidates": 12,
                "n_iters": 6,
                "delta": 0.001,
                "delta_improv": None,
                "leaf_array": leaf_array.tolist(),
                "note": "raw nn_descent output: distances are negative cosine similarity, "
                        "BEFORE the knn_graph wrapper's -log2 transform",
            },
            {"indices": inds.tolist(), "distances": dists.tolist()},
        )


def gen_graph():
    from evoc.graph_construction import (
        smooth_knn_dist,
        compute_membership_strengths,
        neighbor_graph_matrix,
    )
    from evoc.knn_graph import make_forest, nn_descent, INT32_MIN, INT32_MAX
    from sklearn.utils import check_random_state
    import numba

    rng = np.random.RandomState(5)

    # --- smooth_knn_dist ---
    # Rows sorted ascending like real knn output; row 0 has a single leading
    # zero (self-distance), row 1 has two leading zeros (duplicate points) to
    # hit the rho branch and the "zero distances beyond index 0" case.
    n, k = 40, 15
    dists = np.abs(rng.normal(size=(n, k))).astype(np.float32)
    dists.sort(axis=1)
    dists[0, 0] = 0.0
    dists[1, :2] = 0.0
    sigmas, rhos = smooth_knn_dist(dists, float(k))
    write(
        "graph",
        "smooth_knn_dist_n40_k15",
        {"rel": 1e-4, "abs": 1e-6},
        {
            "distances": dists.tolist(),
            "k": float(k),
            "n_iter": 64,
            "bandwidth": 1.0,
            "note": "distances rows are sorted ascending (as real knn output would "
                    "be); row 0 has one leading zero, row 1 has two leading zeros "
                    "to exercise the rho / zero-distance branches",
        },
        {"sigmas": sigmas.tolist(), "rhos": rhos.tolist()},
    )

    # --- compute_membership_strengths ---
    # Reuse the smooth_knn_dist distances/sigmas/rhos above; build indices with
    # a -1 padded slot (missing neighbor) and a self-index (idx == i) to hit
    # both special-case branches.
    inds = rng.randint(0, n, size=(n, k)).astype(np.int32)
    inds[0, -1] = -1
    inds[0, -2] = -1
    inds[5, 0] = 5  # self-index branch (val = 0.0)
    rows, cols, vals = compute_membership_strengths(inds, dists, sigmas, rhos)
    write(
        "graph",
        "compute_membership_strengths_n40_k15",
        {"default": "exact", "fields": {"vals": {"rel": 1e-4, "abs": 1e-6}}},
        {
            "knn_indices": inds.tolist(),
            "knn_dists": dists.tolist(),
            "sigmas": sigmas.tolist(),
            "rhos": rhos.tolist(),
            "note": "knn_indices row 0 has a -1 padded tail (missing neighbors); "
                    "row 5's first neighbor is a self-index (idx == i)",
        },
        {"rows": rows.tolist(), "cols": cols.tolist(), "vals": vals.tolist()},
    )

    # --- neighbor_graph_matrix ---
    # Real small pipeline: reuse the existing nn_descent_float fixture's knn
    # output, apply knn_graph.py's float distance transform
    # (np.maximum(-np.log2(-d), 0.0)) to get the same transformed distances
    # knn_graph() would hand to neighbor_graph_matrix, then build the graph.
    nnd_path = os.path.join(FIXTURES_DIR, "nndescent", "nn_descent_float_n150_d8_k10.json")
    with open(nnd_path) as f:
        nnd_fixture = json.load(f)
    knn_inds = np.array(nnd_fixture["expected"]["indices"], dtype=np.int32)
    raw_dists = np.array(nnd_fixture["expected"]["distances"], dtype=np.float32)
    transformed_dists = np.maximum(-np.log2(-raw_dists), 0.0).astype(np.float32)

    for symmetrize in (True, False):
        graph = neighbor_graph_matrix(
            1.0 * 15, knn_inds, transformed_dists, symmetrize=symmetrize
        )
        csr = graph.tocsr()
        csr.sort_indices()
        write(
            "graph",
            f"neighbor_graph_matrix_symmetrize_{str(symmetrize).lower()}",
            {"default": "exact", "fields": {"data": {"rel": 1e-4, "abs": 1e-6}}},
            {
                "source": "fixtures/nndescent/nn_descent_float_n150_d8_k10.json "
                           "(expected.indices / expected.distances)",
                "knn_indices": knn_inds.tolist(),
                "knn_dists_transformed": transformed_dists.tolist(),
                "n_neighbors": 15.0,
                "symmetrize": symmetrize,
                "note": "knn_dists_transformed = np.maximum(-np.log2(-raw_dists), 0.0) "
                        "applied to the nn_descent_float fixture's raw (negative "
                        "cosine similarity) distances, exactly as knn_graph.nn_descent "
                        "does for float input. Expected matrix is .tocsr().sort_indices() "
                        "canonicalized (indptr/indices exact, data rel 1e-4).",
            },
            {
                "indptr": [int(x) for x in csr.indptr],
                "indices": [int(x) for x in csr.indices],
                "data": [float(x) for x in csr.data],
            },
        )

    # --- knn_graph end-to-end ---
    # numpy's Mersenne Twister isn't available in Rust/Zig, so instead of
    # replaying RandomState draws we call the internals (make_forest,
    # nn_descent) directly with recorded rng_state / rng_states, exactly as
    # knn_graph() does internally. This was verified against a real
    # knn_graph(data, n_neighbors=10, random_state=42) call: the manual
    # replay below reproduces knn_graph's own inds/dists bit-for-bit.
    data = rng.normal(size=(120, 8)).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True).astype(np.float32)
    assert np.allclose(np.linalg.norm(data, axis=1), 1.0)

    n_neighbors = 10
    crs = check_random_state(42)
    rng_state = crs.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

    # Single-thread default (NUMBA_NUM_THREADS=1 is forced for fixture gen):
    # n_trees = max(4, min(8, numba.get_num_threads())) == 4
    n_trees = max(4, min(8, numba.get_num_threads()))
    n_iters = max(5, int(round(np.log2(data.shape[0]))))
    leaf_size = max(10, n_neighbors)
    effective_max_candidates = min(60, int(n_neighbors * 1.5))

    rng_states = crs.randint(INT32_MIN, INT32_MAX, size=(n_trees, 3)).astype(np.int64)
    leaf_array = make_forest(
        data, n_neighbors, n_trees, leaf_size, check_random_state(42), np.float32, max_depth=200
    )
    # make_forest draws its own rng_states internally from the RandomState
    # (same draw as rng_states above, since it's a fresh check_random_state(42)
    # advanced past the same first draw) -- reconstruct via a fresh RandomState
    # replay to keep the two draws (rng_state, then rng_states) in lockstep.
    crs2 = check_random_state(42)
    _ = crs2.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)  # consume rng_state draw
    leaf_array = make_forest(
        data, n_neighbors, n_trees, leaf_size, crs2, np.float32, max_depth=200
    )

    nn_inds, nn_dists = nn_descent(
        data,
        n_neighbors,
        rng_state.copy(),
        effective_max_candidates,
        n_iters,
        0.001,
        np.float32,
        leaf_array=leaf_array,
        verbose=False,
        use_sorted_updates=True,
        delta_improv=0.001,
    )

    # Verify the replay is watertight against the real public entry point.
    ref_inds, ref_dists = __import__("evoc.knn_graph", fromlist=["knn_graph"]).knn_graph(
        data.copy(), n_neighbors=n_neighbors, random_state=42
    )
    assert np.array_equal(nn_inds, ref_inds), "knn_graph replay indices mismatch"
    assert np.allclose(nn_dists, ref_dists), "knn_graph replay distances mismatch"

    write(
        "graph",
        "knn_graph_n120_d8_k10_seed42",
        {"default": "exact", "fields": {"nn_dists": {"rel": 1e-4, "abs": 1e-6}}},
        {
            "data": data.tolist(),
            "n_neighbors": n_neighbors,
            "random_state": 42,
            "n_trees": n_trees,
            "leaf_size": leaf_size,
            "n_iters": n_iters,
            "effective_max_candidates": effective_max_candidates,
            "delta": 0.001,
            "delta_improv": 0.001,
            "use_sorted_updates": True,
            "rng_state": [int(v) for v in rng_state],
            "rng_states": rng_states.tolist(),
            "note": "data is pre-unit-normalized (knn_graph's normalization path is "
                    "a no-op here, verified by np.allclose(norms, 1.0)). This fixture "
                    "was NOT produced by replaying numpy's Mersenne Twister draw-by-draw "
                    "in the ports; it records the ACTUAL rng_state (drawn first) and "
                    "rng_states (drawn second, one triple per tree, inside make_forest) "
                    "that a real check_random_state(42) produces for this call shape. "
                    "Ports should feed rng_state into nn_descent's RNG and rng_states "
                    "(row i) into tree i's RP-tree construction RNG, exactly matching "
                    "make_forest()'s per-tree draw order. This was verified: calling "
                    "make_forest(data, n_neighbors, n_trees=4, leaf_size=10, "
                    "RandomState(42) after consuming the first rng_state draw, float32, "
                    "max_depth=200) then nn_descent(...) with these exact recorded "
                    "values reproduces knn_graph(data, n_neighbors=10, random_state=42)'s "
                    "output bit-for-bit (assert checked at fixture-generation time).",
        },
        {"nn_inds": nn_inds.tolist(), "nn_dists": nn_dists.tolist()},
    )


def gen_boruvka():
    from evoc.numba_kdtree import build_kdtree, parallel_tree_query
    from evoc.boruvka import parallel_boruvka, initialize_boruvka_from_knn
    from evoc.disjoint_set import ds_rank_create

    rng = np.random.RandomState(31)
    n, d = 80, 4
    data = rng.normal(size=(n, d)).astype(np.float32)
    leaf_size = 10
    tree = build_kdtree(data, leaf_size=leaf_size)
    min_samples = 5
    n_threads = 1

    for reproducible in (False, True):
        edges = parallel_boruvka(
            tree, np.int64(n_threads), np.int64(min_samples), reproducible
        )
        write(
            "boruvka",
            f"parallel_boruvka_n80_d4_reproducible_{str(reproducible).lower()}",
            {"default": "exact", "fields": {"distance": {"rel": 1e-4, "abs": 1e-6}}},
            {
                "data": data.tolist(),
                "leaf_size": leaf_size,
                "n_threads": n_threads,
                "min_samples": min_samples,
                "reproducible": reproducible,
                "note": "build_kdtree(data, leaf_size=leaf_size) then "
                        "parallel_boruvka(tree, n_threads, min_samples, reproducible). "
                        "Edges are recorded exactly as returned (no re-sorting); "
                        "each row is [point_a, point_b, distance] with point_a/point_b "
                        "stored as floats by the reference (cast to int32 for parity) "
                        "and distance already sqrt-transformed (all_edges[:,2] = "
                        "sqrt(all_edges[:,2]) is applied before return).",
            },
            {
                "point_a": [int(v) for v in edges[:, 0]],
                "point_b": [int(v) for v in edges[:, 1]],
                "distance": [float(v) for v in edges[:, 2]],
            },
        )

    # --- initialize_boruvka_from_knn, isolated ---
    # Cleanly callable on its own: needs knn indices/distances (from
    # parallel_tree_query with k=min_samples+1, output_rdist=True), per-point
    # core distances (== last column of those rdist distances), and a fresh
    # disjoint set. Record the query outputs as inputs so this stage is
    # independently testable from recorded data alone.
    distances, neighbors = parallel_tree_query(
        tree, tree.data, k=np.int64(min_samples + 1), output_rdist=True
    )
    core_distances = distances.T[-1].copy()
    ds = ds_rank_create(tree.data.shape[0])
    initial_edges = initialize_boruvka_from_knn(neighbors, distances, core_distances, ds)
    write(
        "boruvka",
        "initialize_boruvka_from_knn_n80_d4",
        {"default": "exact", "fields": {"distance": {"rel": 1e-4, "abs": 1e-6}}},
        {
            "knn_indices": neighbors.tolist(),
            "knn_distances_rdist": distances.tolist(),
            "core_distances": core_distances.tolist(),
            "n_points": n,
            "note": "knn_indices/knn_distances_rdist are parallel_tree_query(tree, "
                    "tree.data, k=min_samples+1, output_rdist=True) output (rdist = "
                    "squared euclidean, NOT sqrt'd). core_distances is the last "
                    "column of knn_distances_rdist. disjoint_set is freshly created "
                    "via ds_rank_create(n_points) (rank-based, all singletons) before "
                    "calling initialize_boruvka_from_knn. Output edge distances are "
                    "raw max(core_distance, knn_distance) in rdist space, NOT sqrt'd "
                    "(that sqrt only happens once, at the very end of "
                    "parallel_boruvka, applied to the whole accumulated edge set).",
        },
        {
            "point_a": [int(v) for v in initial_edges[:, 0]],
            "point_b": [int(v) for v in initial_edges[:, 1]],
            "distance": [float(v) for v in initial_edges[:, 2]],
        },
    )


def gen_cluster_trees():
    from evoc.numba_kdtree import build_kdtree
    from evoc.boruvka import parallel_boruvka
    from evoc.cluster_trees import (
        mst_to_linkage_tree,
        condense_tree,
        extract_leaves,
        get_cluster_label_vector,
        get_point_membership_strength_vector,
        extract_eom_clusters,
        cluster_tree_from_condensed_tree,
    )

    # A blob-structured dataset (4 well-separated gaussian clusters) gives a
    # condensed tree with real cluster structure -- a single-gaussian dataset
    # collapses to one dominant cluster and an empty cluster_tree, which
    # would make extract_eom_clusters trivially empty and under-exercise the
    # recursion.
    rng = np.random.RandomState(31)
    n, d = 80, 4
    centers = rng.uniform(-8, 8, size=(4, d))
    data = np.vstack(
        [centers[i] + rng.normal(scale=0.6, size=(20, d)) for i in range(4)]
    ).astype(np.float32)

    leaf_size = 10
    tree = build_kdtree(data, leaf_size=leaf_size)
    min_samples = 5
    edges = parallel_boruvka(tree, np.int64(1), np.int64(min_samples), False)
    sorted_mst = edges[np.argsort(edges.T[2])].astype(np.float32)

    write(
        "cluster_trees",
        "mst_to_linkage_tree_n80",
        {"default": "exact", "fields": {"delta": {"rel": 1e-4, "abs": 1e-6}}},
        {
            "sorted_mst": {
                "point_a": [int(v) for v in sorted_mst[:, 0]],
                "point_b": [int(v) for v in sorted_mst[:, 1]],
                "distance": [float(v) for v in sorted_mst[:, 2]],
            },
            "note": "sorted_mst is boruvka's parallel_boruvka(data 80x4 gaussian-blobs "
                    "seed 31, n_threads=1, min_samples=5, reproducible=False) edge set, "
                    "sorted by np.argsort(edges.T[2]) exactly as clustering.py does "
                    "before calling mst_to_linkage_tree. Reconstruct the (n-1, 3) "
                    "float32 array from point_a/point_b/distance columns in order.",
        },
        {
            "left": [int(v) for v in mst_to_linkage_tree(sorted_mst)[:, 0]],
            "right": [int(v) for v in mst_to_linkage_tree(sorted_mst)[:, 1]],
            "delta": [float(v) for v in mst_to_linkage_tree(sorted_mst)[:, 2]],
            "size": [int(v) for v in mst_to_linkage_tree(sorted_mst)[:, 3]],
        },
    )

    hierarchy = mst_to_linkage_tree(sorted_mst)
    min_cluster_size = 5
    ctree = condense_tree(hierarchy, min_cluster_size)
    write(
        "cluster_trees",
        "condense_tree_n80_mcs5",
        {"default": "exact", "fields": {"lambda_val": {"rel": 1e-4, "abs": 1e-6}}},
        {
            "hierarchy": {
                "left": [int(v) for v in hierarchy[:, 0]],
                "right": [int(v) for v in hierarchy[:, 1]],
                "delta": [float(v) for v in hierarchy[:, 2]],
                "size": [int(v) for v in hierarchy[:, 3]],
            },
            "min_cluster_size": min_cluster_size,
            "note": "hierarchy columns reconstruct mst_to_linkage_tree's (n-1, 4) "
                    "float64 output (left, right, delta, size). CondensedTree fields "
                    "in the reference are parent:int64, child:int64, "
                    "lambda_val:float32, child_size:int64.",
        },
        {
            "parent": [int(v) for v in ctree.parent],
            "child": [int(v) for v in ctree.child],
            "lambda_val": [float(v) for v in ctree.lambda_val],
            "child_size": [int(v) for v in ctree.child_size],
        },
    )

    leaves = extract_leaves(ctree)
    n_samples = n
    cluster_selection_epsilon = 0.0
    clusters = get_cluster_label_vector(ctree, leaves, cluster_selection_epsilon, n_samples)
    strengths = get_point_membership_strength_vector(ctree, leaves, clusters)
    write(
        "cluster_trees",
        "labels_and_strengths_n80_mcs5",
        {"default": "exact", "fields": {"strengths": {"rel": 1e-4, "abs": 1e-6}}},
        {
            "condensed_tree": {
                "parent": [int(v) for v in ctree.parent],
                "child": [int(v) for v in ctree.child],
                "lambda_val": [float(v) for v in ctree.lambda_val],
                "child_size": [int(v) for v in ctree.child_size],
            },
            "cluster_selection_epsilon": cluster_selection_epsilon,
            "n_samples": n_samples,
            "note": "leaves = extract_leaves(condensed_tree) (allow_single_cluster "
                    "defaults True but is irrelevant to extract_leaves itself -- it "
                    "takes no cluster_selection input); "
                    "clusters = get_cluster_label_vector(condensed_tree, leaves, "
                    "cluster_selection_epsilon, n_samples); "
                    "strengths = get_point_membership_strength_vector(condensed_tree, "
                    "leaves, clusters). 'leaves' (extract_leaves output) is included "
                    "below as expected.leaves for the extract_leaves parity check.",
        },
        {
            "leaves": [int(v) for v in leaves],
            "labels": [int(v) for v in clusters],
            "strengths": [float(v) for v in strengths],
        },
    )

    cluster_tree = cluster_tree_from_condensed_tree(ctree)
    eom_clusters = extract_eom_clusters(ctree, cluster_tree, allow_single_cluster=False)
    write(
        "cluster_trees",
        "extract_eom_clusters_n80_mcs5",
        "exact",
        {
            "condensed_tree": {
                "parent": [int(v) for v in ctree.parent],
                "child": [int(v) for v in ctree.child],
                "lambda_val": [float(v) for v in ctree.lambda_val],
                "child_size": [int(v) for v in ctree.child_size],
            },
            "allow_single_cluster": False,
            "note": "cluster_tree = cluster_tree_from_condensed_tree(condensed_tree) "
                    "(mask = condensed_tree.child_size > 1) is recomputed by the port "
                    "from condensed_tree -- it is deterministic and not separately "
                    "fixtured. eom_clusters is the set of selected cluster node ids; "
                    "order is NOT significant (reference builds it from an unordered "
                    "numba typed-dict iteration) -- compare as sets/sorted.",
        },
        {
            "cluster_tree_parent": [int(v) for v in cluster_tree.parent],
            "cluster_tree_child": [int(v) for v in cluster_tree.child],
            "cluster_tree_lambda_val": [float(v) for v in cluster_tree.lambda_val],
            "cluster_tree_child_size": [int(v) for v in cluster_tree.child_size],
            "eom_clusters": sorted(int(v) for v in eom_clusters),
        },
    )


GENERATORS = {
    "rng": gen_rng,
    "disjoint_set": gen_disjoint_set,
    "distance": gen_distance,
    "kdtree": gen_kdtree,
    "nndescent": gen_nndescent,
    "graph": gen_graph,
    "boruvka": gen_boruvka,
    "cluster_trees": gen_cluster_trees,
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
