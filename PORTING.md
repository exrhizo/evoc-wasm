# Porting contract

This document is the shared contract between the Rust and Zig implementations.
Read it before implementing any component. The reference source is a clone of
TutteInstitute/evoc at `/scratch/evoc-reference/evoc/` (not part of this repo).

## Module map

| Python reference          | Rust (`rust/core/src/`)    | Zig (`zig/src/`)          |
|---------------------------|----------------------------|---------------------------|
| disjoint_set.py           | disjoint_set.rs            | disjoint_set.zig          |
| common_nndescent.py (RNG) | rng.rs                     | rng.zig                   |
| float/int8/uint8 distance kernels | distance.rs        | distance.zig              |
| common_nndescent.py (heaps)| nnd_heap.rs               | nnd_heap.zig              |
| numba_kdtree.py           | kdtree.rs                  | kdtree.zig                |
| float_nndescent.py        | nndescent_float.rs         | nndescent_float.zig       |
| int8_nndescent.py         | nndescent_int8.rs          | nndescent_int8.zig        |
| uint8_nndescent.py        | nndescent_uint8.rs         | nndescent_uint8.zig       |
| knn_graph.py              | knn_graph.rs               | knn_graph.zig             |
| graph_construction.py     | graph_construction.rs      | graph_construction.zig    |
| boruvka.py                | boruvka.rs                 | boruvka.zig               |
| cluster_trees.py          | cluster_trees.rs           | cluster_trees.zig         |
| label_propagation.py      | label_propagation.rs       | label_propagation.zig     |
| node_embedding.py         | node_embedding.rs          | node_embedding.zig        |
| clustering_utilities.py   | clustering_utilities.rs    | clustering_utilities.zig  |
| clustering.py             | clustering.rs              | clustering.zig            |

Keep the Python function names (snake_case) in both languages so the three
codebases can be diffed side by side. Port faithfully: same algorithms, same
iteration order, same integer widths, same branch structure. Do not
"improve" the algorithm; note any intentional deviation in a comment
starting with `PORT NOTE:`.

## Numeric parity policy

Three classes, declared per fixture in its `tolerance` field:

1. `"exact"` — integer, RNG, popcount, and tree-structure outputs. All three
   implementations must agree bit-for-bit. Also applies to f32 results whose
   computation is fully order-specified and uses only individually-rounded
   IEEE ops (e.g. `tau_rand`, jaccard's single division).
2. `{"rel": R}` / `{"rel": R, "abs": A}` — float accumulations. The Python
   reference is compiled by numba with `fastmath=True`, which may reorder
   reductions, so Python values are matched within tolerance. Rust and Zig
   MUST still use the plainly-written sequential order (no fastmath, no
   manual vectorization) — as a result **Rust and Zig must be bit-identical
   to each other everywhere**, even where they only epsilon-match Python.
   That cross-check is asserted in the phase-10 comparison harness.
3. Stochastic code (RP-tree splits, NN-descent sampling, SGD negative
   sampling) is made deterministic by the shared `tau_rand` RNG: same seed →
   same call sequence → same output. Never substitute a different RNG.

Accumulator widths matter: e.g. `fast_cosine` accumulates in the loop's
declared f32 (numba locals say float; check each kernel's `locals=` and
signature), int8 inner product accumulates in i32. Match them exactly.

- f32 dot/sum loops: accumulate in f32 unless the reference uses f64.
- `fast_cosine(x, y)` returns `-dot` if `dot > 0` else `-f32_tiny` where
  `f32_tiny = smallest positive normal f32 = 1.1754944e-38`
  (`np.finfo(np.float32).tiny`).

## Fixture format

`fixtures/<component>/<case>.json`, generated only by `scripts/gen_fixtures.py`
(run with `/scratch/pyenv/bin/python`). Never hand-edit fixtures.

```json
{
  "component": "rng",
  "case": "tau_rand_seed_42",
  "tolerance": "exact",            // or {"rel": 1e-5} / {"rel": 1e-5, "abs": 1e-8}
  "inputs":   { ... },             // arrays as nested JSON lists
  "expected": { ... }
}
```

JSON numbers are doubles; f32 values round-trip losslessly through double, so
"exact" f32 comparison means: parse as f64, cast to f32, compare bits.
i64 values in fixtures are within 2^53 or stored as strings (none needed yet).

Tests locate the fixtures directory via the `UVOC_FIXTURES` env var, with a
compile-time fallback (`CARGO_MANIFEST_DIR/../../fixtures` in Rust; the build
sets it for `zig build test` in Zig).

## Language conventions

Both cores must be freestanding-friendly (target: wasm32 in a Web Worker):
no threads, no filesystem, no clock in core code; allocation is fine (Rust
std collections; Zig takes an `std.mem.Allocator` parameter). Test-only code
(JSON fixture loading) may use OS facilities freely — it never ships.

- **Rust**: `rust/core` has zero runtime dependencies; `serde_json` is a
  dev-dependency only. One integration-test file per component in
  `rust/core/tests/`. Unit tests colocated in `#[cfg(test)] mod tests`.
- **Zig**: Zig 0.13.0. Tests colocated in `test` blocks; every new module
  must be reachable from `src/root.zig` (add both the `pub const` import and
  keep the `refAllDecls` test). Fixture parity tests live in the module file
  too, loading JSON via `std.json`.
- Indexing: the reference uses i32 for point indices (i64 for a few
  aggregates). Keep the same widths; do not widen to usize in stored data
  structures (this affects WASM memory footprint and parity).

## Definition of done (per component, per language)

1. Implementation matching the reference structure.
2. Colocated unit tests for edge cases (empty, single element, ties).
3. Fixture parity tests pass: `cargo test` in `rust/`, `zig build test` in `zig/`.
4. `STATUS.md` row updated.
