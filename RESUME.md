# Resume notes: threading experiments worth trying next

Status: notes only, nothing below is implemented yet unless marked done.
Written 2026-07-25 to capture the design discussion so this can be picked
back up later without re-deriving it. For the full evidentiary trail behind
every claim here, see `comparison/results.md` (all benchmark numbers/writeups)
and `VALIDATION_PLAN.md` (the original statistics/wasm roadmap this grew out
of).

## Where we are

Both languages have working wasm32 builds of the full `evoc_clusters`/
`EVoC::fit_predict` pipeline, benchmarked native + wasm from n=100 to
n=30,000, with a fixture-suite-in-wasm harness proving bit-parity survives
compilation. An O(n^2) memory bug was found and fixed at scale (Rust's
`label_propagation.rs`). Per-stage profiling of the *real* pipeline (not a
synthetic standalone benchmark) found:

| stage | share of total time |
|---|---|
| `knn_graph` (RP-tree + NN-descent) | ~45% |
| `node_embedding` (SGD layout) | ~22% |
| `label_propagation_init_recursive` | ~21% |

Single-threaded Python (numba, correctly thread-constrained for a fair
comparison) is now slower than single-threaded Rust/Zig at every size tested
(1.2x–2.3x). Multi-threaded Python (24 cores) is faster than our
single-threaded ports at large n — that gap is what these experiments are
about closing.

Two threaded-RP-tree prototypes were built (Rust: `53655ea`, Zig: `9c7749a`),
both bit-exact vs single-threaded, both using a **separate-private-memory-
per-worker** design (not real shared memory) because each language's default
wasm allocator (Rust's dlmalloc, Zig's `wasm_allocator`) isn't safe to share
across module instances. Result: real 2–2.4x speedup on RP-forest
construction alone, but RP-forest is only ~4-5% of `knn_graph`'s time, so
this saved only ~1-2% of total end-to-end wall-clock. **Confirmed via
grepping the Python reference**: `evoc`'s numba code parallelizes RP-forest
construction (as guessed) AND NN-descent's refinement rounds (not originally
guessed) via per-round work-chunking with private per-thread buffers merged
before the next (still-sequential) round.

**The real conclusion so far**: RP-tree threading was the safe, easy,
bit-exact-preserving experiment, and it wasn't the one that matters much.
NN-descent refinement is ~40% of the pipeline (the actual majority of
`knn_graph`'s time) and is where Python's real multi-threaded advantage
lives. That's the next experiment, and it's harder: NN-descent has genuine
cross-point read/write hazards (updating point A's neighbor heap can require
reading point B's current neighbors), unlike RP-tree's fully independent
per-tree construction.

## Experiment 1 (next, highest priority): pre-partitioned shared-memory NN-descent threading

**The core idea** (this is the one worth trying next): the two prior
prototypes' shared-memory problem was never a language limitation — Zig's
`std.mem.Allocator` is an interface, nothing forces the default
`wasm_allocator`; the actual problem was two *independent* allocators (one
per worker's private instance) racing to hand out addresses in the *same*
linear memory with no coordination between them.

**The fix**: don't give each worker its own allocator at all. Pre-partition
the shared `WebAssembly.Memory` on the JS side, before any worker starts —
hand each worker a fixed base offset + size into the one shared buffer, and
have the Zig/Rust code write only within its own statically-assigned region
(raw pointer arithmetic, no runtime allocation decisions, no race to
resolve). This is actually the SAME calling convention the existing
`wasm_api.rs`/`wasm_api.zig` already use (caller-provided pointer + length,
not internal malloc for I/O) — just extended so multiple workers each get a
disjoint slice of one real shared buffer instead of one worker owning the
whole thing.

**Why this unblocks NN-descent specifically** (and not just RP-tree, which
didn't need it): NN-descent's cross-point hazards mean workers may need to
*read* each other's regions (e.g. point A's thread reading point B's current
neighbor list, where B is owned by another thread) — private-per-instance
memory (what the RP-tree prototypes used) can't do this at all, since each
worker's "point B" would be a different copy. Real shared memory is a
prerequisite for attempting NN-descent threading at all, not just an
optimization of the RP-tree approach.

**Concrete design to prototype**:
1. JS side: allocate one `WebAssembly.Memory{shared: true}`, decide region
   boundaries up front (e.g. point data, per-point neighbor-heap arrays,
   per-point candidate scratch — sized from n/k known ahead of time).
2. Each worker's wasm instance gets the same shared memory; JS passes each
   worker its assigned point-index range (not a memory offset necessarily —
   depends on whether data is laid out per-point-contiguous or otherwise).
3. Within a round: each worker computes candidate updates only for its own
   assigned points, reading (not writing) other points' current neighbor
   lists freely (races on reads of "current" data are the standard
   NN-descent parallelization pattern — Python's own numba implementation
   does exactly this: private per-thread output buffers, sequential merge
   between rounds). Writes should go to each worker's own private output
   buffer for the round; merge into the shared structure single-threaded
   between rounds (matching Python's `generate_leaf_updates_float` pattern
   found in the reference source, per the RP-tree prototype's Step 1
   research — re-read that function in
   `evoc-reference/evoc/float_nndescent.py` before implementing, to mirror
   its actual synchronization structure rather than reinventing it).
4. Verify bit-exactness is achievable or isn't, same rigor as always: if
   read-time races on "in-progress-this-round" data make results
   non-deterministic vs the single-threaded reference ordering, measure ARI/
   structural agreement instead and document why, don't force a false
   bit-parity claim.
5. Determine crossover threshold empirically (RP-tree found Rust ~8-10k,
   Zig ~2k — expect NN-descent's threshold to differ, measure don't guess).
6. Benchmark full pipeline (not just the stage) before/after, and compare
   against multi-threaded Python's full-pipeline numbers already in
   `comparison/results.md` — this is the first experiment that could
   plausibly move that specific comparison meaningfully.

**Suggested order**: prototype in ONE language first (Zig, since its
allocator-interface flexibility is more directly suited to a custom
region-based approach, and its RP-tree prototype found a better crossover
point) to validate the pre-partitioned-shared-memory approach and the
bit-exactness question before duplicating the effort into Rust.

## Experiment 2: extend threading to node_embedding (SGD) and label_propagation

Only worth doing AFTER experiment 1, and only if it's still not competitive
with Python. `node_embedding`'s SGD (~22% of total) is the more
"embarrassingly parallel"-looking of the two remaining stages (per-sample
gradient updates with negative sampling), but SGD parallelization has a
well-known correctness subtlety (concurrent updates to shared embedding
coordinates = a Hogwild!-style race, which changes results in a way that's
inherently approximate, not bit-exact-preservable even in principle) — this
would need to be validated via the same ARI/statistical methodology already
built out in `VALIDATION_PLAN.md` section 2, not via fixture bit-exactness.
`label_propagation_init_recursive`'s recursion structure (~21%) is less
obviously parallel (each level's recursion depends on the previous level's
partition) — lower priority, look at it only if the first two don't close
the gap.

## Experiment 3 (smaller, opportunistic): investigate whether Python's
memoized/threaded gains are reachable without full threading

Worth a quick look before investing in Experiment 1: is any of Python's
speed advantage coming from something OTHER than thread count that we could
capture more cheaply (e.g. algorithmic differences, different default
hyperparameters like leaf_size/n_trees, numba's loop-vectorization on top of
threading)? The single-threaded Python numbers already in
`comparison/results.md` suggest most of the historical gap really was thread
count (single-threaded Python is now slower than single-threaded Rust/Zig
everywhere), so this is probably a small effect, but worth a sanity check
before assuming threading is the only lever.

## Non-goals / already-decided

- Browser deployment of any of this needs COOP/COEP headers for
  `SharedArrayBuffer` (Spectre-mitigation cross-origin isolation
  requirement) — real deployment friction, already flagged, not a reason to
  avoid prototyping in Node first (Node's `worker_threads` +
  `SharedArrayBuffer` don't need those headers).
- npm packaging is still deliberately deferred until a language is chosen as
  the winner (per the original project decision) — these threading
  experiments are inputs to that decision, not blocked on it.
- Don't reopen the RP-tree-only threading path further — it's provably
  capped at ~1-2% total speedup given the profiling data; not worth more
  tuning.
