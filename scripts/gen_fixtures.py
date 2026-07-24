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


def _load_n120_graph():
    """Shared 120-node symmetrized graph, reused by label_propagation and
    node_embedding fixtures. Built exactly as gen_graph's neighbor_graph_matrix
    case does, from the committed knn_graph_n120_d8_k10_seed42 fixture."""
    from evoc.graph_construction import neighbor_graph_matrix

    nnd_path = os.path.join(FIXTURES_DIR, "graph", "knn_graph_n120_d8_k10_seed42.json")
    with open(nnd_path) as f:
        fx = json.load(f)
    knn_inds = np.array(fx["expected"]["nn_inds"], dtype=np.int32)
    knn_dists = np.array(fx["expected"]["nn_dists"], dtype=np.float32)
    n_neighbors = fx["inputs"]["n_neighbors"]
    graph = neighbor_graph_matrix(1.0 * n_neighbors, knn_inds, knn_dists, symmetrize=True)
    csr = graph.tocsr()
    csr.sort_indices()
    return csr


class _TracedRS(np.random.RandomState):
    """np.random.RandomState subclass that records every randint/normal/shuffle
    call and its result, in order. Used to make numpy-Mersenne-Twister-driven
    reference code portable: ports cannot reproduce the MT stream itself, so we
    record every value it produced and feed those values in as explicit fixture
    inputs, in the same call order the reference consumes them."""

    def __init__(self, seed):
        super().__init__(seed)
        self.trace = []

    def randint(self, *a, **k):
        r = super().randint(*a, **k)
        self.trace.append(("randint", np.array(r).tolist()))
        return r

    def normal(self, *a, **k):
        r = super().normal(*a, **k)
        self.trace.append(("normal", np.array(r).tolist()))
        return r

    def shuffle(self, x):
        super().shuffle(x)
        self.trace.append(("shuffle", np.array(x).tolist()))
        return None


def gen_label_propagation():
    from evoc.label_propagation import (
        label_prop_iteration,
        label_outliers,
        remap_labels,
        label_prop_loop,
        label_propagation_init,
    )
    from sklearn.utils import check_random_state

    csr = _load_n120_graph()
    n = csr.shape[0]

    # --- label_prop_iteration standalone ---
    # Seed ~1/4 of nodes with labels drawn from a small pool (0..4) so multiple
    # neighbors of an unlabeled node can cast votes for the SAME label
    # (exercises the votes[l] += branch) and so ties are likely (exercises the
    # tie_count / tau_rand tie-break branch and the current_l == -1 branch).
    rng = np.random.RandomState(11)
    labels0 = np.full(n, -1, dtype=np.int64)
    seed_positions = rng.choice(n, size=n // 4, replace=False)
    for i, pos in enumerate(seed_positions):
        labels0[pos] = int(i % 5)
    rng_state = np.array([555, 777, 999], dtype=np.int64)
    result1 = label_prop_iteration(csr.indptr, csr.indices, csr.data, labels0, rng_state.copy())
    write(
        "label_propagation",
        "label_prop_iteration_n120",
        "exact",
        {
            "indptr": [int(x) for x in csr.indptr],
            "indices": [int(x) for x in csr.indices],
            "data": [float(x) for x in csr.data],
            "labels": [int(x) for x in labels0],
            "rng_state": [int(x) for x in rng_state],
            "note": "labels0: ~n/4 of nodes seeded with labels drawn from a pool of "
                    "5 distinct label values (i % 5), rest -1. This creates real vote "
                    "competition and label collisions among a labeled node's "
                    "neighbors, exercising votes[l] accumulation and the tie_count "
                    "/ tau_rand(local_rng_state) < 1/tie_count tie-break branch. "
                    "local_rng_state = rng_state + i (elementwise, a FRESH array per "
                    "row i, not an advancing shared state) -- tau_rand mutates that "
                    "local copy in place per row; rows are independent RNG streams.",
        },
        {"labels": [int(x) for x in result1]},
    )

    # --- label_outliers standalone ---
    # Build a labels vector with a chain of unresolved nodes so the BFS queue in
    # label_outliers actually walks more than one hop, plus at least one node
    # guaranteed to fall through to the tau_rand_int random-assignment branch
    # (n_iter >= 64 or unlabelled) by disconnecting it entirely (self-loop-only
    # row via indptr slicing is invasive; instead we accept whatever the real
    # graph's connectivity gives us and record the branch outcome as ground
    # truth -- the graph is connected so this exercises the BFS-success branch;
    # the random-assignment branch is covered by construction in the synthetic
    # case below).
    labels_partial = result1.copy()
    rng_state2 = np.array([222, 333, 444], dtype=np.int64)
    result2 = label_outliers(csr.indptr, csr.indices, labels_partial.copy(), rng_state2.copy())
    write(
        "label_propagation",
        "label_outliers_n120",
        "exact",
        {
            "indptr": [int(x) for x in csr.indptr],
            "indices": [int(x) for x in csr.indices],
            "labels": [int(x) for x in labels_partial],
            "rng_state": [int(x) for x in rng_state2],
            "note": "labels_partial is label_prop_iteration_n120's output (still has "
                    "unresolved -1 entries). local_rng_state = rng_state + i per row, "
                    "same convention as label_prop_iteration. Graph is connected so "
                    "every -1 node resolves via BFS within 64 hops in this fixture "
                    "(the tau_rand_int fallback branch is NOT exercised here -- see "
                    "label_outliers_disconnected for that).",
        },
        {"labels": [int(x) for x in result2]},
    )

    # --- label_outliers: disconnected-node fallback branch ---
    # A tiny graph where node 0 has no edges at all: node_queue starts as [0],
    # the for-loop over indptr[0]:indptr[1] (empty range) never executes, so
    # `unlabelled` stays True and n_iter never increments past 0 -- falls
    # through to the tau_rand_int(local_rng_state) % (max_label + 1) branch.
    iso_indptr = np.array([0, 0, 2, 4], dtype=np.int64)
    iso_indices = np.array([2, 1, 1, 2], dtype=np.int32)
    iso_data = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    iso_labels = np.array([-1, 0, 1], dtype=np.int64)
    iso_rng_state = np.array([42, 84, 126], dtype=np.int64)
    iso_result = label_outliers(iso_indptr, iso_indices, iso_labels.copy(), iso_rng_state.copy())
    write(
        "label_propagation",
        "label_outliers_disconnected",
        "exact",
        {
            "indptr": [int(x) for x in iso_indptr],
            "indices": [int(x) for x in iso_indices],
            "labels": [int(x) for x in iso_labels],
            "rng_state": [int(x) for x in iso_rng_state],
            "note": "node 0 has zero edges (indptr[0]==indptr[1]==0), so its BFS "
                    "node_queue=[0] pops with no neighbors, unlabelled stays True, "
                    "and it falls through to labels[0] = tau_rand_int(rng_state+0) "
                    "% (max_label+1). max_label = labels.max() computed BEFORE the "
                    "loop (== 1 here, from the initial labels array, not updated "
                    "labels), so modulus is 2 regardless of what other nodes resolve to.",
        },
        {"labels": [int(x) for x in iso_result]},
    )

    # --- remap_labels standalone ---
    # Mix of already-nonnegative (possibly non-contiguous) labels and -1 entries
    # to exercise both the mapping[] compaction and the next_label assignment
    # for unresolved nodes.
    remap_input = np.array([5, -1, 5, 2, -1, 2, 9, -1], dtype=np.int64)
    remap_result = remap_labels(remap_input.copy())
    write(
        "label_propagation",
        "remap_labels",
        "exact",
        {
            "labels": [int(x) for x in remap_input],
            "note": "unique non-negative labels {2,5,9} remap to {0,1,2} in sorted "
                    "order (np.unique); each remaining -1 gets a FRESH successive "
                    "label starting at next_label = len(unique_labels) -- i.e. -1 "
                    "entries do NOT collapse to a single label, each becomes its "
                    "own new singleton cluster, assigned in array-index order.",
        },
        {"labels": [int(x) for x in remap_result]},
    )

    # --- label_prop_loop end to end (numpy MT seeding draws recorded) ---
    rs = _TracedRS(321)
    labels_init = np.full(n, -1, dtype=np.int64)
    approx_n_parts = 16
    n_iter = 5
    partition = label_prop_loop(
        csr.indptr, csr.indices, csr.data, labels_init, rs, n_iter=n_iter, approx_n_parts=approx_n_parts
    )
    assert rs.trace[0][0] == "randint" and len(rs.trace[0][1]) == 3
    rng_state_drawn = rs.trace[0][1]
    seed_draws = [t[1] for t in rs.trace[1:] if t[0] == "randint"][:approx_n_parts]
    write(
        "label_propagation",
        "label_prop_loop_n120",
        "exact",
        {
            "indptr": [int(x) for x in csr.indptr],
            "indices": [int(x) for x in csr.indices],
            "data": [float(x) for x in csr.data],
            "n_points": n,
            "n_iter": n_iter,
            "approx_n_parts": approx_n_parts,
            "rng_state": rng_state_drawn,
            "seed_position_draws": seed_draws,
            "note": "label_prop_loop draws from a numpy RandomState (NOT portable "
                    "bit-for-bit); this fixture records every draw it makes in order "
                    "so ports can replay deterministically without numpy's Mersenne "
                    "Twister. Draw order: (1) rng_state = "
                    "random_state.randint(INT32_MIN, INT32_MAX, 3) once -- this "
                    "becomes the FIXED array fed (by reference/copy, per-row offset) "
                    "into every label_prop_iteration/label_outliers call for the rest "
                    "of the loop; (2) approx_n_parts calls to "
                    "random_state.randint(labels.shape[0]) -- each draws ONE seed "
                    "position, and labels[pos] = i is set immediately (i = draw "
                    "index, 0-based) -- collisions (same position drawn twice) are "
                    "possible and simply overwrite. After seeding: n_iter calls to "
                    "label_prop_iteration (feeding forward each iteration's output "
                    "as the next iteration's input labels), then ONE label_outliers "
                    "call, then remap_labels. rng_state itself is never redrawn "
                    "during the iterations -- it is reused as-is (only the +i row "
                    "offset varies per node inside each numba kernel call).",
        },
        {"partition": [int(x) for x in partition]},
    )

    # --- label_propagation_init end to end ---
    # graph.shape[0]=120 >= base_init_threshold=64, so this takes the
    # label_prop_loop + node_embedding upscaling path (not the small-graph PCA
    # base case). recursive_init=False so there is exactly ONE level of
    # label_prop_loop + ONE node_embedding call on the reduced graph -- this
    # keeps the RNG trace flat and fully recordable (recursive_init=True would
    # recurse into ANOTHER label_propagation_init call, multiplying draws and,
    # below base_init_threshold, invoking sklearn PCA(random_state=...) which
    # draws from the RandomState in an sklearn/scipy-internal, effectively
    # unportable way -- avoid that path entirely for fixtures).
    n_components = 4
    n_label_prop_iter = 5
    n_embedding_epochs = 5
    approx_n_parts2 = 16
    rs2 = _TracedRS(321)
    emb = label_propagation_init(
        csr,
        n_label_prop_iter=n_label_prop_iter,
        n_embedding_epochs=n_embedding_epochs,
        approx_n_parts=approx_n_parts2,
        n_components=n_components,
        scaling=0.1,
        random_scale=1.0,
        noise_level=0.5,
        random_state=rs2,
        data=None,
        recursive_init=False,
        base_init="pca",
        base_init_threshold=64,
        upscaling="partition_expander",
    )
    # Determinism check: an independent replay from a fresh seed-321 RandomState
    # must reproduce the same embedding bit-for-bit (numba kernels + single
    # thread + the same MT draw sequence => deterministic).
    rs2b = check_random_state(321)
    emb_b = label_propagation_init(
        csr,
        n_label_prop_iter=n_label_prop_iter,
        n_embedding_epochs=n_embedding_epochs,
        approx_n_parts=approx_n_parts2,
        n_components=n_components,
        scaling=0.1,
        random_scale=1.0,
        noise_level=0.5,
        random_state=rs2b,
        data=None,
        recursive_init=False,
        base_init="pca",
        base_init_threshold=64,
        upscaling="partition_expander",
    )
    assert np.array_equal(emb, emb_b), "label_propagation_init is not deterministic for a fixed seed"

    # Unpack the trace into the named draws ports need, in call order:
    # 0: rng_state (3 int64, label_prop_loop's internal RNG state)
    # 1..approx_n_parts2: seed position draws (label_prop_loop's partition seeding)
    # next: normal draw = node_embedding's own initial_embedding (since
    #   label_propagation_init passes initial_embedding=None on the recursive_init=False
    #   path, so node_embedding draws its own init from the SAME advancing random_state)
    # next: randint draw = node_embedding's rng_val (one uint32-range int per epoch)
    # next n_embedding_epochs: shuffle draws = node_embedding's per-epoch node_order shuffles
    idx = 0
    assert rs2.trace[idx][0] == "randint"
    lp_rng_state = rs2.trace[idx][1]
    idx += 1
    seed_draws2 = []
    for _ in range(approx_n_parts2):
        assert rs2.trace[idx][0] == "randint"
        seed_draws2.append(rs2.trace[idx][1])
        idx += 1
    assert rs2.trace[idx][0] == "normal"
    ne_init_normal = rs2.trace[idx][1]
    idx += 1
    assert rs2.trace[idx][0] == "randint"
    ne_rng_val = rs2.trace[idx][1]
    idx += 1
    ne_shuffles = []
    for _ in range(n_embedding_epochs):
        assert rs2.trace[idx][0] == "shuffle"
        ne_shuffles.append(rs2.trace[idx][1])
        idx += 1
    assert idx == len(rs2.trace), "unexpected trailing RNG draws in label_propagation_init trace"

    write(
        "label_propagation",
        "label_propagation_init_n120",
        {"rel": 1e-4, "abs": 1e-6},
        {
            "indptr": [int(x) for x in csr.indptr],
            "indices": [int(x) for x in csr.indices],
            "data": [float(x) for x in csr.data],
            "n_points": n,
            "n_label_prop_iter": n_label_prop_iter,
            "n_embedding_epochs": n_embedding_epochs,
            "approx_n_parts": approx_n_parts2,
            "n_components": n_components,
            "scaling": 0.1,
            "random_scale": 1.0,
            "noise_level": 0.5,
            "recursive_init": False,
            "base_init_threshold": 64,
            "upscaling": "partition_expander",
            "lp_rng_state": lp_rng_state,
            "lp_seed_position_draws": seed_draws2,
            "node_embedding_init_normal": ne_init_normal,
            "node_embedding_rng_val": ne_rng_val,
            "node_embedding_shuffles": ne_shuffles,
            "note": "graph is the shared 120-node fixture (>= base_init_threshold=64) "
                    "so this takes the label_prop_loop + node_embedding-upscaling "
                    "path, NOT the small-graph PCA/random/spectral/mds base case. "
                    "recursive_init=False keeps this to a single non-recursive level "
                    "(no nested label_propagation_init / no sklearn PCA call). "
                    "label_propagation_init does NOT draw from tau_rand/rng_state "
                    "itself for its own logic -- ALL its randomness is either (a) "
                    "delegated to label_prop_loop (which draws lp_rng_state + "
                    "lp_seed_position_draws from the passed numpy RandomState, then "
                    "uses tau_rand internally with that FIXED rng_state array) or "
                    "(b) delegated to node_embedding (called with "
                    "initial_embedding=None on this path, so node_embedding draws "
                    "its OWN init from the same advancing RandomState -- see "
                    "node_embedding_init_normal/rng_val/shuffles below). Draw order "
                    "recorded here is exactly the call order: lp_rng_state (1x3 "
                    "int64), lp_seed_position_draws (approx_n_parts scalars, one per "
                    "label_prop_loop seeding iteration), node_embedding_init_normal "
                    "(reduced_graph.shape[0] x n_components floats -- reduced_graph "
                    "is the label-propagation-partitioned graph, NOT the input "
                    "graph; its size is data-dependent, read its shape off this "
                    "array), node_embedding_rng_val (n_embedding_epochs ints, one "
                    "per SGD epoch), node_embedding_shuffles (n_embedding_epochs "
                    "permutation arrays of node_order, applied one per epoch AFTER "
                    "that epoch's SGD update, each of length reduced_graph.shape[0]). "
                    "Feed lp_rng_state/lp_seed_position_draws into a replay of "
                    "label_prop_loop's body (label_prop_iteration x n_iter, then "
                    "ONE label_outliers, then remap_labels -- see "
                    "label_prop_loop_n120 fixture for that in isolation), then build "
                    "the reduced graph exactly as label_propagation_init does "
                    "(base_reduction_map / normalized_reduction_map / reduced_graph "
                    "= normalized_reduction_map.T @ graph @ base_reduction_map, "
                    "clipped to [0,1]), then feed node_embedding_init_normal as "
                    "node_embedding's initial_embedding and "
                    "node_embedding_rng_val[n]/node_embedding_shuffles[n] as its "
                    "per-epoch RNG source (see node_embedding fixtures for exact "
                    "epoch semantics), then finish with the partition_expander "
                    "upscaling algebra (no further randomness) to get the final "
                    "n_points x n_components embedding. Determinism was verified: "
                    "an independent RandomState(321) replay of the WHOLE function "
                    "reproduces this embedding bit-for-bit.",
        },
        {"embedding": emb.tolist()},
    )


def gen_node_embedding():
    from evoc.node_embedding import (
        make_epochs_per_sample,
        rdist,
        clip,
        node_embedding_epoch,
        node_embedding_epoch_repr,
        node_embedding,
    )
    from sklearn.utils import check_random_state

    csr = _load_n120_graph()
    n = csr.shape[0]

    # --- make_epochs_per_sample standalone ---
    for n_epochs in (10, 50):
        eps = make_epochs_per_sample(csr.data, n_epochs)
        write(
            "node_embedding",
            f"make_epochs_per_sample_n_epochs{n_epochs}",
            {"rel": 1e-5},
            {
                "weights": [float(x) for x in csr.data],
                "n_epochs": n_epochs,
                "note": "weights is the shared 120-node graph's csr.data (edge "
                        "weights). result = n_epochs / max(n_epochs * (weights / "
                        "weights.max()), 1.0), computed in f32; the max(..., 1.0) "
                        "floor means edges far below the max weight get exactly "
                        "n_epochs as their epochs_per_sample (never sampled more "
                        "than once every n_epochs epochs).",
            },
            {"epochs_per_sample": [float(x) for x in eps]},
        )

    # --- rdist / clip standalone ---
    rng = np.random.RandomState(3)
    rdist_pairs = []
    for dim in (4, 6, 16):
        x = rng.normal(size=dim).astype(np.float32)
        y = rng.normal(size=dim).astype(np.float32)
        rdist_pairs.append((x, y))
    rdist_pairs.append((rdist_pairs[0][0], rdist_pairs[0][0]))  # identical -> 0
    write(
        "node_embedding",
        "rdist",
        {"rel": 1e-5, "abs": 1e-7},
        {"pairs": [{"x": x.tolist(), "y": y.tolist()} for x, y in rdist_pairs]},
        {"values": [float(rdist(x, y)) for x, y in rdist_pairs]},
    )

    clip_cases = [
        {"val": 5.0, "lo": -4.0, "hi": 4.0},
        {"val": -5.0, "lo": -4.0, "hi": 4.0},
        {"val": 1.0, "lo": -4.0, "hi": 4.0},
        {"val": 4.0, "lo": -4.0, "hi": 4.0},   # boundary: val == hi
        {"val": -4.0, "lo": -4.0, "hi": 4.0},  # boundary: val == lo
    ]
    write(
        "node_embedding",
        "clip",
        "exact",
        {"cases": clip_cases},
        {
            "values": [
                float(clip(np.float32(c["val"]), np.float32(c["lo"]), np.float32(c["hi"])))
                for c in clip_cases
            ]
        },
    )

    # --- node_embedding_epoch_repr: ONE epoch, isolated ---
    # reproducible_flag=True is the path clustering.py actually uses (evoc_clusters
    # calls node_embedding with its default reproducible_flag=True); this fixture
    # targets THAT kernel, not the plain node_embedding_epoch (see the note below
    # for the distinction). epoch_of_next_sample / epoch_of_next_negative_sample
    # are recorded as all-ZERO (rather than initialized to epochs_per_sample /
    # epochs_per_negative_sample as node_embedding would on epoch 0) so that
    # EVERY sample's `epoch_of_next_sample[i] <= n` gate fires on this single
    # call -- with the natural epoch-0 schedule almost nothing fires (schedules
    # start at >=1), which would under-exercise the inner SGD loop. This isolates
    # exactly the update math + negative-sampling index arithmetic for one pass.
    n_components = 5
    rng2 = np.random.RandomState(9)
    embedding = rng2.normal(scale=0.25, size=(n, n_components)).astype(np.float32, order="C")
    epochs_per_sample = make_epochs_per_sample(csr.data, 10).astype(np.float32, order="C")
    negative_sample_rate = 1.0
    epochs_per_negative_sample = (epochs_per_sample / negative_sample_rate) * 1.5  # reproducible_flag=True
    epoch_of_next_negative_sample = np.zeros_like(epochs_per_negative_sample)
    epoch_of_next_sample = np.zeros_like(epochs_per_sample)
    csr_indptr = csr.indptr.astype(np.uint32)
    csr_indices = csr.indices.astype(np.uint32)
    updates = np.zeros_like(embedding)
    node_order = np.arange(n, dtype=np.uint32)
    embedding_before = embedding.copy()
    rng_state_scalar = 123456
    gamma = 1.0
    alpha = 0.1
    block_size = 4096
    node_embedding_epoch_repr(
        embedding,
        csr_indptr,
        csr_indices,
        np.uint32(n),
        epochs_per_sample,
        np.uint32(rng_state_scalar),
        n_components,
        np.float32(alpha),
        epochs_per_negative_sample,
        epoch_of_next_negative_sample,
        epoch_of_next_sample,
        np.uint8(0),
        np.float32(0.5),
        gamma,
        updates,
        node_order,
        np.uint32(block_size),
    )
    write(
        "node_embedding",
        "node_embedding_epoch_repr_single_epoch_n120",
        {"rel": 1e-4, "abs": 1e-6},
        {
            "embedding_before": embedding_before.tolist(),
            "csr_indptr": [int(x) for x in csr_indptr],
            "csr_indices": [int(x) for x in csr_indices],
            "n_vertices": n,
            "epochs_per_sample": [float(x) for x in epochs_per_sample],
            "rng_state": rng_state_scalar,
            "dim": n_components,
            "alpha": alpha,
            "epochs_per_negative_sample": [float(x) for x in epochs_per_negative_sample],
            "epoch_of_next_negative_sample": [float(x) for x in epoch_of_next_negative_sample],
            "epoch_of_next_sample": [float(x) for x in epoch_of_next_sample],
            "n": 0,
            "noise_level": 0.5,
            "gamma": gamma,
            "node_order": [int(x) for x in node_order],
            "block_size": block_size,
            "note": "This is node_embedding_epoch_repr (the reproducible_flag=True "
                    "kernel that clustering.py's node_embedding(..., "
                    "reproducible_flag=True) actually calls -- NOT the plain "
                    "node_embedding_epoch, which is only used when "
                    "reproducible_flag=False). epoch_of_next_sample and "
                    "epoch_of_next_negative_sample are deliberately recorded as "
                    "all-zero (NOT the natural epoch-0 values, which are >=1 and "
                    "would mean almost nothing fires) so every edge's positive "
                    "update AND negative-sampling loop fire on this single call -- "
                    "this isolates the per-edge SGD math from the outer epoch-loop "
                    "scheduling. rng_state here is a single scalar (uint32), NOT a "
                    "tau_rand advancing 3-int64 state -- negative sample target "
                    "index is `node_order[(raw_index * (n + p + 1) * rng_state) % "
                    "n_vertices]`, pure integer arithmetic, no RNG state mutation. "
                    "updates[] accumulates positive+negative gradient contributions "
                    "per from_node across the WHOLE block before being added into "
                    "embedding at the end of the block (node_embedding itself then "
                    "scales updates by (1-alpha)^2*0.5 AFTER this call and carries "
                    "it to the next epoch -- that carry-over is NOT part of this "
                    "kernel and is out of scope for this fixture). Expected "
                    "embedding_after already has updates applied (this kernel "
                    "applies them internally, in its last per-block loop) -- do NOT "
                    "separately add updates_after to embedding_before.",
        },
        {
            "embedding_after": embedding.tolist(),
            "updates_after": updates.tolist(),
            "epoch_of_next_sample_after": [float(x) for x in epoch_of_next_sample],
            "epoch_of_next_negative_sample_after": [float(x) for x in epoch_of_next_negative_sample],
        },
    )

    # --- node_embedding_epoch (non-reproducible kernel), ONE epoch, isolated ---
    # Included for completeness / in case a port wants the simpler non-blocked
    # kernel, but note clustering.py never calls this path with the default
    # reproducible_flag=True.
    embedding3 = rng2.normal(scale=0.25, size=(n, n_components)).astype(np.float32, order="C")
    coo = csr.tocoo()
    head = coo.row.astype(np.uint32)
    tail = coo.col.astype(np.uint32)
    eps3 = make_epochs_per_sample(csr.data, 10).astype(np.float32, order="C")
    epns3 = eps3 / negative_sample_rate  # reproducible_flag=False path: no *1.5
    eons3 = np.zeros_like(epns3)
    eos3 = np.zeros_like(eps3)
    embedding3_before = embedding3.copy()
    rng_state_scalar3 = 654321
    node_embedding_epoch(
        embedding3, head, tail, np.uint32(n), eps3, rng_state_scalar3,
        n_components, np.float32(0.1), epns3, eons3, eos3, 0, np.float32(0.5),
    )
    write(
        "node_embedding",
        "node_embedding_epoch_single_epoch_n120",
        {"rel": 1e-4, "abs": 1e-6},
        {
            "embedding_before": embedding3_before.tolist(),
            "head": [int(x) for x in head],
            "tail": [int(x) for x in tail],
            "n_vertices": n,
            "epochs_per_sample": [float(x) for x in eps3],
            "rng_state": rng_state_scalar3,
            "dim": n_components,
            "alpha": 0.1,
            "epochs_per_negative_sample": [float(x) for x in epns3],
            "epoch_of_next_negative_sample": [float(x) for x in eons3],
            "epoch_of_next_sample": [float(x) for x in eos3],
            "n": 0,
            "noise_level": 0.5,
            "note": "node_embedding_epoch is the reproducible_flag=False kernel: "
                    "it takes COO (head, tail) edge lists rather than CSR + "
                    "node_order/updates/blocking, and updates `embedding` DIRECTLY "
                    "in place per-edge (no deferred updates[] accumulator, no "
                    "gamma_schedule term in the negative-sample grad_coeff, and "
                    "epochs_per_negative_sample has no *1.5 reproducibility "
                    "inflation). clustering.py's default reproducible_flag=True "
                    "never exercises this kernel -- fixture it anyway for "
                    "completeness/parity coverage. epoch_of_next_sample / "
                    "epoch_of_next_negative_sample zeroed for the same reason as "
                    "the _repr fixture (force every sample to fire on this call).",
        },
        {"embedding_after": embedding3.tolist()},
    )

    # --- node_embedding end-to-end (reproducible_flag=True, the path clustering.py uses) ---
    n_components_e2e = 5
    n_epochs_e2e = 10
    rng3 = np.random.RandomState(202)
    init_embedding = rng3.normal(scale=0.25, size=(n, n_components_e2e)).astype(np.float32, order="C")

    rs = _TracedRS(303)
    final_embedding = node_embedding(
        csr,
        n_components_e2e,
        n_epochs_e2e,
        initial_embedding=init_embedding.copy(),
        initial_alpha=0.1,
        negative_sample_rate=1.0,
        noise_level=0.5,
        random_state=rs,
        reproducible_flag=True,
        verbose=False,
    )
    # Determinism check against an independent fresh-seed replay.
    rs_b = check_random_state(303)
    final_embedding_b = node_embedding(
        csr,
        n_components_e2e,
        n_epochs_e2e,
        initial_embedding=init_embedding.copy(),
        initial_alpha=0.1,
        negative_sample_rate=1.0,
        noise_level=0.5,
        random_state=rs_b,
        reproducible_flag=True,
        verbose=False,
    )
    assert np.array_equal(final_embedding, final_embedding_b), "node_embedding is not deterministic for a fixed seed"

    # Trace: since initial_embedding is provided, node_embedding does NOT draw its
    # own init `normal` -- draw order is: rng_val (1x randint, size=n_epochs),
    # then n_epochs shuffles (one per epoch, applied AFTER that epoch's SGD pass).
    assert rs.trace[0][0] == "randint"
    rng_val = rs.trace[0][1]
    shuffles = [t[1] for t in rs.trace[1:] if t[0] == "shuffle"]
    assert len(shuffles) == n_epochs_e2e
    assert len(rs.trace) == 1 + n_epochs_e2e

    write(
        "node_embedding",
        "node_embedding_e2e_n120",
        {"rel": 1e-3, "abs": 1e-5},
        {
            "indptr": [int(x) for x in csr.indptr],
            "indices": [int(x) for x in csr.indices],
            "data": [float(x) for x in csr.data],
            "n_vertices": n,
            "n_components": n_components_e2e,
            "n_epochs": n_epochs_e2e,
            "initial_embedding": init_embedding.tolist(),
            "initial_alpha": 0.1,
            "negative_sample_rate": 1.0,
            "noise_level": 0.5,
            "reproducible_flag": True,
            "rng_val": rng_val,
            "node_order_shuffles": shuffles,
            "note": "reproducible_flag=True is the ONLY path clustering.py's "
                    "node_embedding() call uses (it never passes "
                    "reproducible_flag=False); this end-to-end fixture targets "
                    "node_embedding_epoch_repr exclusively, called once per epoch "
                    "in a python for-loop (see node_embedding()'s source). Because "
                    "initial_embedding is provided, node_embedding does NOT draw "
                    "its own normal() init -- the FIRST RandomState draw is "
                    "rng_val = random_state.randint(INT32_MAX, size=n_epochs), "
                    "done ONCE up front, before the epoch loop. Inside the loop, "
                    "epoch n's rng_state argument to node_embedding_epoch_repr is "
                    "np.uint32(rng_val[n]) -- a single scalar per epoch, reused for "
                    "every negative-sample draw within that epoch (see the "
                    "single-epoch fixture note for the index arithmetic; there is "
                    "no tau_rand advancing state anywhere in node_embedding). "
                    "AFTER each epoch's node_embedding_epoch_repr call: `updates *= "
                    "(1.0 - alpha) ** 2 * 0.5` (updates array persists/decays "
                    "across epochs, it is NOT re-zeroed), then "
                    "`random_state.shuffle(node_order)` (one shuffle per epoch, "
                    "recorded in order in node_order_shuffles -- node_order starts "
                    "as arange(n_vertices) before epoch 0 and is permuted in place "
                    "by each shuffle), then `alpha = initial_alpha * (1 - n / "
                    "n_epochs)` for the NEXT epoch (note: alpha for epoch n uses "
                    "the value computed after epoch n-1; epoch 0 uses the raw "
                    "initial_alpha unchanged). gamma_schedule = "
                    "np.linspace(0.5, 1.5, n_epochs) (deterministic, no RNG; "
                    "gamma_schedule[n] is epoch n's gamma argument -- recompute it, "
                    "not recorded separately here). csr_indptr/csr_indices passed "
                    "to the kernel are graph.indptr/graph.indices cast to uint32 "
                    "(reconstruct from indptr/indices above). Determinism was "
                    "verified: an independent RandomState(303) replay of the WHOLE "
                    "function reproduces final_embedding bit-for-bit. Python "
                    "tolerance is loose (fastmath accumulation over 10 epochs) but "
                    "Rust and Zig MUST match each other bit-for-bit per PORTING.md.",
        },
        {"final_embedding": final_embedding.tolist()},
    )


def _hierarchical_blobs_seed45():
    """A dataset with genuine two-level cluster structure (2-3 super-clusters,
    each with 2-4 sub-clusters), chosen (by trying seeds) because it produces a
    non-trivial condensed/cluster tree: min_cluster_size_barcode's persistence
    curve has an actual interior local maximum, so find_peaks/select_diverse_peaks
    are exercised non-trivially -- most simple blob configurations collapse to a
    monotone persistence curve with zero peaks. Returns (data, n_samples,
    condensed_tree, cluster_tree, sorted_mst) built via the real reference
    pipeline (kdtree -> boruvka -> mst_to_linkage_tree -> condense_tree)."""
    from evoc.numba_kdtree import build_kdtree
    from evoc.boruvka import parallel_boruvka
    from evoc.cluster_trees import mst_to_linkage_tree, condense_tree, mask_condensed_tree

    rng = np.random.RandomState(45)
    n_super = rng.randint(2, 4)
    super_centers = rng.uniform(-50, 50, size=(n_super, 4))
    data_list = []
    for sc in super_centers:
        n_sub = rng.randint(2, 5)
        sub_centers = sc + rng.normal(scale=rng.uniform(3, 12), size=(n_sub, 4))
        for subc in sub_centers:
            npts = rng.randint(8, 20)
            data_list.append(subc + rng.normal(scale=rng.uniform(0.3, 1.2), size=(npts, 4)))
    data = np.vstack(data_list).astype(np.float32)
    n_samples = data.shape[0]

    numba_tree = build_kdtree(data.astype(np.float32))
    edges = parallel_boruvka(numba_tree, np.int64(1), np.int64(5), True)
    sorted_mst = edges[np.argsort(edges.T[2])].astype(np.float32)
    uncondensed_tree = mst_to_linkage_tree(sorted_mst)
    condensed_tree = condense_tree(uncondensed_tree, 5)
    mask = condensed_tree.child >= n_samples
    cluster_tree = mask_condensed_tree(condensed_tree, mask)
    return data, n_samples, condensed_tree, cluster_tree, sorted_mst


def _condensed_tree_to_json(ct):
    return {
        "parent": [int(v) for v in ct.parent],
        "child": [int(v) for v in ct.child],
        "lambda_val": [float(v) for v in ct.lambda_val],
        "child_size": [int(v) for v in ct.child_size],
    }


def gen_clustering_utilities():
    from evoc.numba_kdtree import build_kdtree
    from evoc.boruvka import parallel_boruvka
    from evoc.cluster_trees import CondensedTree, mst_to_linkage_tree, mask_condensed_tree
    from evoc.clustering_utilities import (
        find_peaks,
        _binary_search_for_n_clusters,
        binary_search_for_n_clusters,
        min_cluster_size_barcode,
        compute_total_persistence,
        extract_clusters_by_id,
        jaccard_similarity,
        estimate_cluster_similarity,
        select_diverse_peaks,
        build_cluster_tree,
        find_duplicates,
    )

    data, n_samples, condensed_tree, cluster_tree, sorted_mst = _hierarchical_blobs_seed45()

    # Record the shared seed-45 dataset + its condensed_tree/cluster_tree once,
    # as a reference fixture other cases in this module point back to by name
    # (avoids repeating the full data/tree arrays in every fixture's inputs).
    write(
        "clustering_utilities",
        "condense_tree_seed45",
        {"default": "exact", "fields": {"lambda_val": {"rel": 1e-4, "abs": 1e-6}}},
        {
            "data": data.tolist(),
            "n_samples": n_samples,
            "note": "data is a 2-3-super-cluster / 2-4-sub-cluster-each hierarchical "
                    "blobs dataset (RandomState(45); see _hierarchical_blobs_seed45 "
                    "in gen_fixtures.py for the exact generation recipe -- pure "
                    "numpy RandomState draws for cluster centers/sizes, no "
                    "reference-library randomness involved, so it is exactly "
                    "reproducible by re-running that recipe if needed, but the "
                    "'data' array here is the portable ground truth). "
                    "condensed_tree = condense_tree(mst_to_linkage_tree("
                    "sorted_mst), min_cluster_size=5) where sorted_mst = "
                    "parallel_boruvka(build_kdtree(data), n_threads=1, "
                    "min_samples=5, reproducible=True) sorted by distance. "
                    "cluster_tree = mask_condensed_tree(condensed_tree, "
                    "condensed_tree.child >= n_samples) (internal-nodes-only "
                    "view). Other fixtures in this module (min_cluster_size_barcode_seed45, "
                    "compute_total_persistence_seed45, select_diverse_peaks_seed45, "
                    "extract_clusters_by_id_seed45_*) reference this dataset by "
                    "name instead of repeating 'data' -- this fixture is the "
                    "portable source of truth for it.",
        },
        {
            "condensed_tree": _condensed_tree_to_json(condensed_tree),
            "cluster_tree": _condensed_tree_to_json(cluster_tree),
        },
    )

    # --- find_peaks: synthetic 1-D signal with 3 clean interior peaks (incl. a
    # plateau to exercise the i_ahead scan-past-ties logic) ---
    x = np.array([0, 1, 3, 2, 1, 0, 2, 5, 3, 1, 0, 4, 4, 1], dtype=np.float64)
    peaks_synth = find_peaks(x)
    write(
        "clustering_utilities",
        "find_peaks_synthetic",
        "exact",
        {
            "x": x.tolist(),
            "note": "1-D signal with 3 local maxima at indices 2 (value 3), 7 "
                    "(value 5), and 11 (value 4, a width-2 plateau at indices "
                    "11-12 -- find_peaks's midpoint = (left_edge+right_edge)//2 "
                    "picks the LEFT of the plateau pair). First/last samples can "
                    "never be peaks by construction (i starts at 1, loop bound is "
                    "x.shape[0]-1).",
        },
        {"peak_indices": [int(v) for v in peaks_synth]},
    )
    # No-peak / monotone case (also realistic: this is what the n80 phase-6 tree gives).
    x_flat = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], dtype=np.float64)
    write(
        "clustering_utilities",
        "find_peaks_monotone_empty",
        "exact",
        {"x": x_flat.tolist(), "note": "strictly decreasing -- no interior local maximum, expect empty result."},
        {"peak_indices": [int(v) for v in find_peaks(x_flat)]},
    )

    # --- min_cluster_size_barcode / compute_total_persistence / find_peaks (real tree) ---
    min_cluster_size = 5
    births, deaths, parents, lambda_deaths = min_cluster_size_barcode(cluster_tree, n_samples, min_cluster_size)
    write(
        "clustering_utilities",
        "min_cluster_size_barcode_seed45",
        {"rel": 1e-4, "abs": 1e-6},
        {
            "cluster_tree": _condensed_tree_to_json(cluster_tree),
            "n_points": n_samples,
            "min_size": min_cluster_size,
            "note": "cluster_tree is mask_condensed_tree(condensed_tree, "
                    "condensed_tree.child >= n_points) applied to condense_tree("
                    "mst_to_linkage_tree(sorted_mst), min_cluster_size=5) built "
                    "from a real hierarchical-blobs dataset (2-3 super-clusters "
                    "each with 2-4 sub-clusters, RandomState(45) recipe -- see "
                    "condense_tree_seed45 fixture for the full condensed_tree this "
                    "was masked from, and the 'data' field there for the raw "
                    "points). n_nodes = cluster_tree.child[-1] - n_points + 1. "
                    "Iterates row-PAIRS in reverse (idx = n_rows-1, n_rows-3, ...) "
                    "-- cluster_tree rows come in adjacent left/right sibling "
                    "pairs from condense_tree's branch-split output; out_idx = "
                    "cluster_tree.child[idx] - n_points indexes the RIGHT sibling "
                    "of the pair, out_idx-1 the LEFT. lambda_deaths uses exp(-1/"
                    "lambda_val) -- lambda_val is never 0 here (guaranteed by "
                    "condense_tree's lambda=1/d convention), but guard for it if "
                    "porting to a context where it could be.",
        },
        {
            "size_births": [float(v) for v in births],
            "size_deaths": [float(v) for v in deaths],
            "parents": [int(v) for v in parents],
            "lambda_deaths": [float(v) for v in lambda_deaths],
        },
    )

    sizes, total_persistence = compute_total_persistence(births, deaths, lambda_deaths)
    write(
        "clustering_utilities",
        "compute_total_persistence_seed45",
        {"rel": 1e-4, "abs": 1e-6},
        {
            "births": [float(v) for v in births],
            "deaths": [float(v) for v in deaths],
            "lambda_deaths": [float(v) for v in lambda_deaths],
            "note": "births/deaths/lambda_deaths are min_cluster_size_barcode_seed45's "
                    "size_births/size_deaths/lambda_deaths outputs verbatim. sizes = "
                    "np.unique(births) (sorted, deduped). Loop starts at i=1 "
                    "(index 0, the root's (0-birth, n_points-death) entry, is "
                    "SKIPPED intentionally -- 'maintain left-open (birth, death] "
                    "interval'). Both birth_idx/death_idx searches are LINEAR "
                    "scans for the first sizes[j] >= birth/death (not true binary "
                    "search despite the comment -- replicate as a linear scan, "
                    "first-match-wins, not an actual bisection); death_idx "
                    "defaults to len(sizes) if no match (death exceeds the largest "
                    "known size). Entries with death <= birth are skipped "
                    "entirely (continue).",
        },
        {"sizes": [float(v) for v in sizes], "total_persistence": [float(v) for v in total_persistence]},
    )

    peaks = find_peaks(total_persistence)
    selected_peaks = select_diverse_peaks(
        peaks, total_persistence, sizes, births, deaths,
        min_similarity_threshold=0.2, max_layers=9,
    )
    write(
        "clustering_utilities",
        "select_diverse_peaks_seed45",
        "exact",
        {
            "peaks": [int(v) for v in peaks],
            "total_persistence": [float(v) for v in total_persistence],
            "sizes": [float(v) for v in sizes],
            "births": [float(v) for v in births],
            "deaths": [float(v) for v in deaths],
            "min_similarity_threshold": 0.2,
            "max_layers": 9,
            "note": "peaks = find_peaks(total_persistence) from the "
                    "compute_total_persistence_seed45 fixture (this dataset "
                    "yields exactly one real peak, at index 1 into "
                    "sizes/total_persistence -- a small but genuine multi-layer "
                    "case; the phase-6 n80 tree's persistence curve is monotone "
                    "and yields ZERO peaks, so it can't exercise this function "
                    "meaningfully). Peaks sorted by persistence descending "
                    "(np.argsort(...)[::-1] -- ties broken by argsort's stable "
                    "order, i.e. original ascending-index order among equal "
                    "values). For each candidate peak in that order: compute "
                    "jaccard-based estimate_cluster_similarity against every "
                    "ALREADY-selected peak's birth size; reject (skip) if "
                    "similarity > threshold for ANY already-selected peak, else "
                    "accept. Stops early once max_layers peaks are selected.",
        },
        {"selected_peak_indices": [int(v) for v in selected_peaks]},
    )

    # --- extract_clusters_by_id, using the selected peak(s) above ---
    for peak in selected_peaks:
        best_birth = sizes[peak]
        selected_clusters = np.where((births <= best_birth) & (deaths > best_birth))[0] + n_samples
        labels, strengths = extract_clusters_by_id(condensed_tree, selected_clusters)
        write(
            "clustering_utilities",
            f"extract_clusters_by_id_seed45_peak{int(peak)}",
            {"default": "exact", "fields": {"strengths": {"rel": 1e-4, "abs": 1e-6}}},
            {
                "condensed_tree": _condensed_tree_to_json(condensed_tree),
                "selected_ids": [int(v) for v in selected_clusters],
                "note": "condensed_tree is condense_tree(mst_to_linkage_tree("
                        "sorted_mst), min_cluster_size=5) from the same seed-45 "
                        "hierarchical-blobs dataset (full tree, NOT the "
                        "child>=n_points-masked cluster_tree). selected_clusters = "
                        "where((births<=best_birth) & (deaths>best_birth))[0] + "
                        "n_samples, i.e. cluster NODE IDS (offset by n_samples "
                        "since node ids >= n_samples are internal cluster nodes). "
                        "extract_clusters_by_id = get_cluster_label_vector(tree, "
                        "selected_ids, cluster_selection_epsilon=0.0, n_samples="
                        "tree.parent[0]) + get_point_membership_strength_vector "
                        "(already-ported in cluster_trees phase 6 -- this fixture "
                        "just pins the selected_ids this call site actually "
                        "produces).",
            },
            {"labels": [int(v) for v in labels], "strengths": [float(v) for v in strengths]},
        )

    # --- jaccard_similarity / estimate_cluster_similarity, standalone ---
    jaccard_cases = [
        {"a": [1, 2, 3], "b": [2, 3, 4]},
        {"a": [1, 2, 3], "b": [1, 2, 3]},
        {"a": [1, 2, 3], "b": [4, 5, 6]},
        {"a": [], "b": []},
        {"a": [7], "b": []},
    ]
    write(
        "clustering_utilities",
        "jaccard_similarity",
        "exact",
        {
            "cases": jaccard_cases,
            "note": "set_a_array/set_b_array are int64 arrays in the reference. "
                    "Empty/empty -> union_count==0 -> returns 0.0 (not NaN, not "
                    "an error). Order matters for the ALGORITHM (union_set starts "
                    "as set(a), then b's items are folded in) but NOT for the "
                    "final numeric result (jaccard is symmetric) -- still replicate "
                    "the exact iteration for parity of any future extension.",
        },
        {
            "values": [
                float(jaccard_similarity(np.array(c["a"], dtype=np.int64), np.array(c["b"], dtype=np.int64)))
                for c in jaccard_cases
            ]
        },
    )

    est_cases = [
        {"birth_a": float(sizes[0]), "birth_b": float(sizes[-1])},
        {"birth_a": float(sizes[0]), "birth_b": float(sizes[0])},
    ]
    write(
        "clustering_utilities",
        "estimate_cluster_similarity_seed45",
        "exact",
        {
            "births": [float(v) for v in births],
            "deaths": [float(v) for v in deaths],
            "cases": est_cases,
            "note": "births/deaths from min_cluster_size_barcode_seed45. For each "
                    "case: clusters_a = indices i where births[i]<=birth_a AND "
                    "deaths[i]>birth_a (half-open interval, matching "
                    "compute_total_persistence's left-open/right-closed (birth, "
                    "death] convention read backwards); same for clusters_b with "
                    "birth_b. Result = jaccard_similarity(clusters_a, clusters_b) "
                    "-- same birth_a/birth_b (2nd case) is NOT guaranteed to give "
                    "similarity 1.0 in general (only if clusters_a's construction "
                    "is deterministic given identical inputs, which it is here "
                    "since both loops use the same arrays) so this case should "
                    "come out as exactly 1.0.",
        },
        {
            "values": [
                float(estimate_cluster_similarity(births, deaths, np.float64(c["birth_a"]), np.float64(c["birth_b"])))
                for c in est_cases
            ]
        },
    )

    # --- build_cluster_tree: small synthetic multi-layer label vectors ---
    labels_layer0 = np.array([0, 0, 1, 1, 2, 2, -1, 3, 3], dtype=np.int64)
    labels_layer1 = np.array([0, 0, 0, 0, 1, 1, -1, 1, 1], dtype=np.int64)
    labels_layer2 = np.array([0, 0, 0, 0, 0, 0, -1, 0, 0], dtype=np.int64)
    tree_dict = build_cluster_tree([labels_layer0, labels_layer1, labels_layer2])
    # Serialize the dict deterministically: sorted (layer, cluster) keys, each
    # value list also sorted (values are appended in a data-order-dependent
    # sequence in the reference; sort for a canonical, order-independent fixture).
    sorted_items = []
    for key in sorted(tree_dict.keys()):
        children = sorted(tree_dict[key])
        sorted_items.append(
            {
                "parent_layer": int(key[0]),
                "parent_cluster": int(key[1]),
                "children": [[int(c[0]), int(c[1])] for c in children],
            }
        )
    write(
        "clustering_utilities",
        "build_cluster_tree_3layers",
        "exact",
        {
            "labels": [
                [int(v) for v in labels_layer0],
                [int(v) for v in labels_layer1],
                [int(v) for v in labels_layer2],
            ],
            "note": "labels[0] is the FINEST (base) layer, labels[-1] the "
                    "COARSEST; -1 entries are noise at that layer, excluded from "
                    "the tree. _build_cluster_tree walks upper_layer from 1 "
                    "upward, and for each upper_layer scans lower_layer from "
                    "upper_layer-1 down to 0, grouping lower-layer labels by "
                    "which upper-layer cluster they fall under (via argsort + "
                    "np.split on bincount cumsum) and recording a (parent_layer, "
                    "parent_label, child_layer, child_label) edge the FIRST time "
                    "a given (child_layer, child_label) is seen (found[] set "
                    "dedup, seeded with {-1} per layer so noise is never "
                    "recorded). After that pass, any (layer, label) not yet found "
                    "gets attached as a synthetic child of (len(labels), 0) -- "
                    "a virtual root above the top layer. The reference's "
                    "iteration order over dict/set is NOT guaranteed stable; "
                    "this fixture's 'expected' is a CANONICALIZED form: a list "
                    "sorted by (parent_layer, parent_cluster), each entry's "
                    "children list sorted too. Compare as sets, not sequences.",
        },
        {"tree": sorted_items},
    )

    # --- find_duplicates: synthetic knn with genuine duplicate/near-duplicate structure ---
    knn_inds = np.array(
        [
            [1, 2, 3],
            [0, 2, 3],
            [3, 0, 1],
            [2, 0, 1],
            [5, 6, 0],
            [4, 6, 0],
            [4, 5, 0],
        ],
        dtype=np.int32,
    )
    knn_dists = np.array(
        [
            [0.0, 0.5, 0.6],
            [0.0, 0.4, 0.5],
            [0.1, 0.5, 0.6],
            [0.1, 0.5, 0.6],
            [0.9, 0.95, 1.0],
            [0.9, 0.92, 1.0],
            [0.9, 0.92, 1.0],
        ],
        dtype=np.float32,
    )
    dup = find_duplicates(knn_inds, knn_dists)
    write(
        "clustering_utilities",
        "find_duplicates_synthetic",
        "exact",
        {
            "knn_indices": knn_inds.tolist(),
            "knn_dists": knn_dists.tolist(),
            "note": "duplicate_distance = max(knn_dists[:, 0]) (the largest "
                    "NEAREST-neighbor distance across all points, i.e. the "
                    "reference's own worst-case 1st-neighbor distance sets the "
                    "threshold for what counts as 'duplicate' EVERYWHERE -- a "
                    "single noisy row inflates the threshold globally). For "
                    "every (i, knn_indices[i,j]) pair with knn_dists[i,j] <= "
                    "that threshold: canonicalize as (min(i,k), max(i,k)) and add "
                    "to a set (dedups automatically); i==k (self-neighbor) pairs "
                    "are explicitly skipped via the elif/else chain. Row 0/1 rows "
                    "2/3 are mutually 0.0-or-0.1 distance (all pairwise "
                    "duplicates); row 4/5/6 sit right at the 0.9 threshold "
                    "(inclusive <=) so (4,5) qualifies but the 0.92/0.95 second "
                    "neighbors do not.",
        },
        {"duplicate_pairs": sorted([[int(a), int(b)] for a, b in dup])},
    )

    # find_duplicates reusing a real nndescent fixture's knn output (0 duplicates
    # expected -- real cosine-similarity distances rarely land at the exact max,
    # this documents that as ground truth rather than forcing a positive case).
    nnd_path = os.path.join(FIXTURES_DIR, "nndescent", "nn_descent_float_n150_d8_k10.json")
    with open(nnd_path) as f:
        nnd_fixture = json.load(f)
    real_inds = np.array(nnd_fixture["expected"]["indices"], dtype=np.int32)
    real_dists = np.array(nnd_fixture["expected"]["distances"], dtype=np.float32)
    real_dup = find_duplicates(real_inds, real_dists)
    write(
        "clustering_utilities",
        "find_duplicates_nndescent_n150",
        "exact",
        {
            "source": "fixtures/nndescent/nn_descent_float_n150_d8_k10.json "
                       "(expected.indices / expected.distances, raw negative-"
                       "cosine-similarity distances, unmodified)",
            "knn_indices": real_inds.tolist(),
            "knn_dists": real_dists.tolist(),
            "note": "Real nn_descent output on random unit vectors essentially "
                    "never has exact near-duplicate points, so duplicate_pairs is "
                    "empty here -- this is a legitimate ground-truth empty-set "
                    "case, not a placeholder; see find_duplicates_synthetic for "
                    "the non-trivial case.",
        },
        {"duplicate_pairs": sorted([[int(a), int(b)] for a, b in real_dup])},
    )

    # --- binary_search_for_n_clusters end-to-end + _binary_search_for_n_clusters isolated ---
    rng_bs = np.random.RandomState(88)
    bdata = np.vstack(
        [
            rng_bs.normal(loc=[0, 0], scale=0.5, size=(20, 2)),
            rng_bs.normal(loc=[10, 10], scale=0.5, size=(20, 2)),
            rng_bs.normal(loc=[-10, 10], scale=0.5, size=(20, 2)),
        ]
    ).astype(np.float32)
    btree = build_kdtree(bdata)
    bedges = parallel_boruvka(btree, np.int64(1), np.int64(5), True)
    bsorted = bedges[np.argsort(bedges.T[2])].astype(np.float32)
    buncond = mst_to_linkage_tree(bsorted)
    approx_n_clusters = 3
    leaves, clusters, strengths = _binary_search_for_n_clusters(buncond, approx_n_clusters, bdata.shape[0])
    write(
        "clustering_utilities",
        "binary_search_for_n_clusters_n60",
        {"default": "exact", "fields": {"strengths": {"rel": 1e-4, "abs": 1e-6}}},
        {
            "data": bdata.tolist(),
            "approx_n_clusters": approx_n_clusters,
            "min_samples": 5,
            "n_threads": 1,
            "uncondensed_tree": {
                "left": [int(v) for v in buncond[:, 0]],
                "right": [int(v) for v in buncond[:, 1]],
                "delta": [float(v) for v in buncond[:, 2]],
                "size": [int(v) for v in buncond[:, 3]],
            },
            "note": "3 well-separated gaussian blobs (20 points each, seed 88) so "
                    "approx_n_clusters=3 is exactly achievable; this exercises "
                    "the FULL binary_search_for_n_clusters pipeline (build_kdtree "
                    "-> parallel_boruvka(reproducible=False as the public "
                    "wrapper hardcodes) -> mst_to_linkage_tree -> "
                    "_binary_search_for_n_clusters). uncondensed_tree is provided "
                    "as an input so _binary_search_for_n_clusters is independently "
                    "testable without re-deriving the MST. Binary search bisects "
                    "min_cluster_size in [2, n_samples//2] on "
                    "len(extract_leaves(condense_tree(tree, mid))) vs "
                    "approx_n_clusters, using round-half-to-even-unspecified "
                    "Python round() (banker's rounding on the .5 boundary -- "
                    "replicate round(x) semantics, not int(x+0.5)). Terminates "
                    "when the bracket narrows to width 1 OR mid collapses onto a "
                    "bound; final choice is whichever of lower/upper bound's "
                    "n_clusters is closer to approx_n_clusters, with a 3-way tie-"
                    "break (prefer more non-noise points assigned) when both "
                    "bounds are equidistant -- replicate ALL THREE branches "
                    "(strictly-lower-closer, strictly-upper-closer, tie), not "
                    "just the common case.",
        },
        {
            "leaves": [int(v) for v in leaves],
            "labels": [int(v) for v in clusters],
            "strengths": [float(v) for v in strengths],
        },
    )


def gen_clustering():
    from evoc.knn_graph import knn_graph
    from evoc.graph_construction import neighbor_graph_matrix
    from evoc.label_propagation import label_propagation_init
    from evoc.node_embedding import node_embedding
    from evoc.clustering import build_cluster_layers
    from sklearn.utils import check_random_state

    # --- build_cluster_layers: the portable capstone fixture ---
    # build_cluster_layers(data, ...) takes an ALREADY-COMPUTED embedding and has
    # NO random_state parameter and NO internal RNG usage of its own (kdtree
    # build, boruvka MST, condense_tree, min_cluster_size_barcode,
    # compute_total_persistence, find_peaks, select_diverse_peaks,
    # extract_clusters_by_id are all deterministic given their inputs -- see the
    # cluster_trees (phase 6) and clustering_utilities (phase 8a) notes). That
    # makes it fully and exactly recordable/replayable: record a realistic
    # node-embedding-shaped input, record the output, done. This is the intended
    # top-level parity target for phase 8 (see the evoc_clusters non-portability
    # note below for why the fuller public entry point is NOT used instead).
    #
    # The input embedding below is produced by actually running the reference
    # pipeline (knn_graph -> neighbor_graph_matrix -> label_propagation_init ->
    # node_embedding) on a small hierarchical-blobs dataset -- i.e. it looks
    # like real EVoC output, not an arbitrary point cloud -- chosen (by trying
    # seeds) because it yields a genuine 2-layer result (base layer + one
    # selected persistence peak), exercising find_peaks/select_diverse_peaks/
    # extract_clusters_by_id/the layer-sorting step non-trivially. The
    # embedding itself is recorded as a plain float32 array input; ports do NOT
    # need to reproduce how it was produced (that is phases 7a/7b's job) --
    # only build_cluster_layers's own logic is under test here.
    seed = 1
    rng = np.random.RandomState(seed)
    n_super = rng.randint(3, 5)
    super_centers = rng.uniform(-40, 40, size=(n_super, 12))
    data_list = []
    for sc in super_centers:
        n_sub = rng.randint(2, 4)
        sub_centers = sc + rng.normal(scale=rng.uniform(5, 10), size=(n_sub, 12))
        for subc in sub_centers:
            npts = rng.randint(10, 18)
            data_list.append(subc + rng.normal(scale=rng.uniform(0.4, 1.0), size=(npts, 12)))
    data = np.vstack(data_list).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True).astype(np.float32)
    n_samples = data.shape[0]

    rs = check_random_state(seed * 7 + 1)
    nn_inds, nn_dists = knn_graph(data, n_neighbors=15, random_state=rs)
    graph = neighbor_graph_matrix(1.0 * 15, nn_inds, nn_dists, True)
    n_embedding_components = min(max(15 // 4, 4), 15)
    init_embedding = label_propagation_init(
        graph,
        n_components=n_embedding_components,
        approx_n_parts=int(np.clip(int(8 * np.sqrt(n_samples)), 256, 16384)),
        random_scale=0.1,
        scaling=0.5,
        noise_level=0.5,
        random_state=rs,
        data=data,
        n_label_prop_iter=20,
    )
    embedding = node_embedding(
        graph,
        n_components=n_embedding_components,
        n_epochs=15,
        initial_embedding=init_embedding,
        negative_sample_rate=1.0,
        noise_level=0.5,
        random_state=rs,
        verbose=False,
        reproducible_flag=True,
        initial_alpha=0.1,
    )

    min_samples = 5
    base_min_cluster_size = 5
    max_layers = 10
    cluster_layers, membership_strength_layers, persistence_scores = build_cluster_layers(
        embedding,
        min_samples=min_samples,
        base_min_cluster_size=base_min_cluster_size,
        max_layers=max_layers,
        reproducible_flag=True,
    )
    # Determinism check.
    cl2, ms2, ps2 = build_cluster_layers(
        embedding,
        min_samples=min_samples,
        base_min_cluster_size=base_min_cluster_size,
        max_layers=max_layers,
        reproducible_flag=True,
    )
    assert all(np.array_equal(a, b) for a, b in zip(cluster_layers, cl2))
    assert all(np.allclose(a, b) for a, b in zip(membership_strength_layers, ms2))
    assert persistence_scores == ps2
    # reproducible_flag only bounds a multithreaded boruvka race (see boruvka
    # phase-5 notes); at n_threads=1 it must be bit-identical to False.
    cl3, ms3, ps3 = build_cluster_layers(
        embedding,
        min_samples=min_samples,
        base_min_cluster_size=base_min_cluster_size,
        max_layers=max_layers,
        reproducible_flag=False,
    )
    assert all(np.array_equal(a, b) for a, b in zip(cluster_layers, cl3)), (
        "reproducible_flag changed build_cluster_layers output at n_threads=1"
    )

    write(
        "clustering",
        "build_cluster_layers_n126",
        {"default": "exact", "fields": {"membership_strengths": {"rel": 1e-4, "abs": 1e-6}, "persistence_scores": {"rel": 1e-4, "abs": 1e-6}}},
        {
            "embedding": embedding.tolist(),
            "min_samples": min_samples,
            "base_min_cluster_size": base_min_cluster_size,
            "base_n_clusters": None,
            "reproducible_flag": True,
            "min_similarity_threshold": 0.2,
            "max_layers": max_layers,
            "note": "embedding (126x4 float32) is a REAL node_embedding output "
                    "from running the reference pipeline (knn_graph -> "
                    "neighbor_graph_matrix -> label_propagation_init -> "
                    "node_embedding) on a hierarchical-blobs dataset (12-D unit-"
                    "normalized, RandomState(1) recipe -- see gen_clustering in "
                    "gen_fixtures.py). Ports do NOT need to reproduce how this "
                    "embedding was derived (that is the label_propagation / "
                    "node_embedding fixtures' job) -- treat 'embedding' as an "
                    "opaque recorded input; only build_cluster_layers's own logic "
                    "is under test here. build_cluster_layers has NO "
                    "random_state parameter and no internal RNG of its own (numba "
                    "kdtree build, boruvka MST -- reproducible_flag only affects "
                    "multithreaded races, verified bit-identical to False here "
                    "at n_threads=1 -- condense_tree, and all of "
                    "clustering_utilities's persistence/peak-selection functions "
                    "are deterministic given their inputs), so this fixture is "
                    "the exact, fully portable end-to-end target for phase 8 "
                    "(see the evoc_clusters_not_portable note in PORT_NOTES.md "
                    "phase 8 for why the fuller evoc_clusters()/EVoC.fit_predict() "
                    "entry points are NOT fixtured end-to-end instead). Base "
                    "layer (index into the pre-sort list, always index 0 before "
                    "sorting) uses condense_tree(uncondensed_tree, "
                    "base_min_cluster_size) directly (base_n_clusters is None "
                    "here, so the base_n_clusters branch in build_cluster_layers "
                    "is NOT exercised by this fixture -- see PORT_NOTES.md for "
                    "that gap). This dataset/seed yields exactly 2 cluster "
                    "layers (a base layer with persistence_score 0.0, plus ONE "
                    "selected persistence peak) -- output arrays below are "
                    "ALREADY in the function's final sorted order (descending "
                    "by n_clusters = layer.max()+1), matching what a port's "
                    "output should be directly, no further reordering needed.",
        },
        {
            "cluster_layers": [[int(v) for v in layer] for layer in cluster_layers],
            "membership_strengths": [[float(v) for v in layer] for layer in membership_strength_layers],
            "persistence_scores": [float(v) for v in persistence_scores],
        },
    )

    # --- evoc_clusters / EVoC.fit_predict: documented as NOT bit-for-bit
    # portable end-to-end, by design -- see PORT_NOTES.md phase 8 for the full
    # explanation. Left unfixtured deliberately (no partial/best-effort fixture
    # written here to avoid implying it is a parity target).


GENERATORS = {
    "rng": gen_rng,
    "disjoint_set": gen_disjoint_set,
    "distance": gen_distance,
    "kdtree": gen_kdtree,
    "nndescent": gen_nndescent,
    "graph": gen_graph,
    "boruvka": gen_boruvka,
    "cluster_trees": gen_cluster_trees,
    "label_propagation": gen_label_propagation,
    "node_embedding": gen_node_embedding,
    "clustering_utilities": gen_clustering_utilities,
    "clustering": gen_clustering,
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
