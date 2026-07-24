# uvoc-wasm

A from-scratch **Rust and Zig** port (both, for comparison) of
[TutteInstitute/evoc](https://github.com/TutteInstitute/evoc)'s clustering
algorithm, compiled to WASM, packaged for `npm install` (working name
`evoc-wasm`), for in-browser clustering of chunked-document embeddings.

## Why

A planned browser extension chunks a text stream into documents, each with
metadata columns (float / enum), and needs to cluster them **live in the
browser** — ideally inside a Web Worker, but architected so a standalone /
main-thread build is easy from the same core.

## Goal: a faithful PLSCAN port

Not a simplified/MVP clustering algorithm — the real EVoC pipeline.

## What EVoC actually does

BSD-2-Clause. Core class `evoc.clustering.EVoC`, sklearn-style
(`fit`/`fit_predict`). Pipeline:

1. Approximate k-NN graph (RP-tree + NN-Descent)
2. Graph symmetrization
3. Node embedding (UMAP-style low-dim layout)
4. Optional near-duplicate detection
5. Parallel Boruvka MST over mutual-reachability/core distances (HDBSCAN
   convention)
6. MST → linkage → condensed tree
7. Excess-of-Mass cluster extraction
8. Multi-granularity `cluster_layers_` via repeated condensation

Heavy use of `numba.njit(parallel=True)`/`prange` and `jitclass`-backed
KD-tree and union-find structures in the Python original — none of which
have a drop-in Rust/Zig equivalent, so this is a genuine from-scratch port,
not a binding.

Distance metric: cosine (float32/float64, unit-normalized), with int8/uint8
quantized variants also supported in the original.

Key constructor params (from the Python original): `n_neighbors=15`,
`min_samples=5`, `base_min_cluster_size=5`, `noise_level=0.5`,
`n_epochs=50`, `max_layers=10`.

## Architecture goal

- Primary target: Web Worker (clustering runs off the main thread).
- The core algorithm should live in a library crate/module clean of any
  worker-specific glue, so a standalone/main-thread WASM build is easy to
  produce from the same core with a thin different entry point.
- Both `rust/` and `zig/` implementations are being built for direct
  comparison (performance, binary size, WASM toolchain ergonomics) — not to
  pick a winner up front.

## Repository layout

- `PORTING.md` — the porting contract both implementations follow (module
  map, numeric parity policy, fixture schema). Read this first.
- `STATUS.md` — component-by-component progress matrix.
- `fixtures/` — golden test vectors generated from the reference Python
  implementation by `scripts/gen_fixtures.py`; ground truth for parity tests.
- `rust/` — Cargo workspace (`core` = dependency-free algorithm crate;
  `wasm` wrapper added in phase 9). `cd rust && cargo test`.
- `zig/` — Zig 0.13 package, tests colocated. `cd zig && zig build test`.
- `comparison/results.md` — binary size / speed / parity comparison.

The reference implementation is cloned (read-only) at
`/scratch/evoc-reference` in the sandbox; it is not part of this repo.

## Status

In progress — see `STATUS.md`. Work happens in the sandboxed claw agent
(`lungta-claw uvoc-wasm exec`).
