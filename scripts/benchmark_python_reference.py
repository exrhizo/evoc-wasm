#!/usr/bin/env python
"""Python reference (`evoc.EVoC`) wall-clock time + peak memory benchmark,
for comparison/results.md's "Python reference: time and memory" section.

There was previously NO Python performance baseline in results.md -- only
Rust/Zig native and wasm numbers, plus separate Python ARI/correctness
validation (see PCA_VALIDATION.md). This script adds the missing dimension:
how does `evoc.EVoC(...).fit_predict(X)` (the numba-JIT reference
implementation this whole project ports) compare on time and memory to the
Rust/Zig native and wasm numbers already in results.md?

Reuses the SAME datasets already used elsewhere in this project for an
apples-to-apples comparison:
  - n=100, 240, 800: the exact fixture data (`fixtures/clustering/
    panel_small_balanced.json`, `evoc_clusters_e2e_ari_check.json`,
    `panel_large_many_clusters.json`) used for the existing Rust/Zig
    native + wasm end-to-end numbers in results.md's "Execution time"
    section (same n_features per fixture: 12/24/32 respectively).
  - n=2000, 10000, 30000: `scripts/gen_scale_test_data.py`'s cosine-friendly
    orthogonal-axis-centers generator (d=32, 8 clusters, seed=1234), the
    exact same construction used for results.md's "Scaling to 10k-30k
    points" section (regenerated in-process here rather than reading the
    gitignored .f32 blobs, so this script has no file-existence
    dependency on scripts/scale_data/ having been populated first).

For each size: one COLD run (first call in a fresh process -- numba's JIT
must compile every `@njit`-decorated kernel it hits, which is the realistic
"first time this process imports evoc and calls fit_predict" cost) and a
few WARM runs (same process, JIT already compiled, only algorithmic time
remains) are reported separately, since the JIT/no-JIT distinction matters
a lot for a reference implementation nobody would normally re-import per
call. Each dataset size runs in its own subprocess (via `-c` so numba's JIT
cache from one size can't warm a later size, keeping "cold" honest) with a
generous timeout (default 10 minutes) so a pathologically slow large-n run
is reported as "did not complete" rather than hanging the whole sweep.

Peak memory: `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` (Linux:
kibibytes) read at the end of the subprocess, i.e. the process's peak RSS
over its whole lifetime (import + JIT compile + all runs) -- simplest
reliable option available in every Python without extra dependencies.

Usage:
    /wrk/claw-scratch/uvoc-wasm/pyenv/bin/python \\
        scripts/benchmark_python_reference.py [n1,n2,...]

Default sizes: 100,240,800,2000,10000,30000. Requires `evoc` installed in
the running interpreter's environment (see comparison/results.md for the
exact venv path used to produce the numbers currently recorded there).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "fixtures" / "clustering"
DEFAULT_TIMEOUT_S = 10 * 60

FIXTURE_FOR_SIZE = {
    100: "panel_small_balanced.json",
    240: "evoc_clusters_e2e_ari_check.json",
    800: "panel_large_many_clusters.json",
}

# Same construction as scripts/gen_scale_test_data.py, inlined so this
# script can regenerate 2000/10000/30000 without depending on the
# gitignored scripts/scale_data/*.f32 blobs having been produced first.
GEN_SCALE_SNIPPET = '''
import numpy as np
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
X = make_cosine_friendly_dataset({n}, 8, 32, 0.35, 3.0, 1234)
'''

FIXTURE_SNIPPET = '''
import json
import numpy as np
d = json.loads(open({path!r}).read())
X = np.array(d["data"], dtype=np.float32)
'''

WORKER_TEMPLATE = '''
import time, resource, json, sys
{load}
import evoc

n_warm = {n_warm}
seed = 0

t0 = time.perf_counter()
model = evoc.EVoC(random_state=seed)
labels_cold = model.fit_predict(X)
t_cold = time.perf_counter() - t0

warm_times = []
for _ in range(n_warm):
    t0 = time.perf_counter()
    model = evoc.EVoC(random_state=seed)
    model.fit_predict(X)
    warm_times.append(time.perf_counter() - t0)

peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
result = {{
    "n_samples": int(X.shape[0]),
    "n_features": int(X.shape[1]),
    "cold_s": t_cold,
    "warm_s": sorted(warm_times),
    "peak_rss_kb": peak_kb,
}}
print("RESULT_JSON:" + json.dumps(result))
'''


def run_one(n, timeout_s, n_warm):
    if n in FIXTURE_FOR_SIZE:
        path = FIXTURES_DIR / FIXTURE_FOR_SIZE[n]
        load = FIXTURE_SNIPPET.format(path=str(path))
    else:
        load = GEN_SCALE_SNIPPET.format(n=n)

    script = WORKER_TEMPLATE.format(load=load, n_warm=n_warm)
    t_wall0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"n": n, "error": f"did not complete in {timeout_s / 60:.0f} minutes"}
    wall = time.time() - t_wall0

    if proc.returncode != 0:
        return {
            "n": n,
            "error": f"subprocess failed (exit {proc.returncode}): {proc.stderr[-2000:]}",
        }

    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            result = json.loads(line[len("RESULT_JSON:"):])
            result["wall_s"] = wall
            return result

    return {"n": n, "error": f"no RESULT_JSON in stdout: {proc.stdout[-2000:]} {proc.stderr[-2000:]}"}


def main():
    sizes = [100, 240, 800, 2000, 10000, 30000]
    if len(sys.argv) > 1:
        sizes = [int(x) for x in sys.argv[1].split(",")]

    timeout_s = DEFAULT_TIMEOUT_S
    results = []
    for n in sizes:
        n_warm = 3 if n <= 2000 else 1
        print(f"=== n={n} ===", flush=True)
        r = run_one(n, timeout_s, n_warm)
        results.append(r)
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue
        warm = r["warm_s"]
        warm_median = warm[len(warm) // 2] if warm else float("nan")
        print(f"  cold (incl. JIT compile): {r['cold_s'] * 1000:.1f} ms")
        print(f"  warm (median of {len(warm)}):    {warm_median * 1000:.1f} ms  (min={min(warm) * 1000:.1f} ms, max={max(warm) * 1000:.1f} ms)" if warm else "  warm: n/a")
        print(f"  peak RSS: {r['peak_rss_kb'] / 1024:.1f} MiB")

    print("\n=== summary (JSON) ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
