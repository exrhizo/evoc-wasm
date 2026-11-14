//! THROWAWAY: not part of the regular test suite. Native (non-wasm) timing
//! of `EVoC::fit_predict` for comparison/results.md's "Execution time"
//! table (VALIDATION_PLAN.md section 3, items 3-4 -- wasm vs native
//! overhead). Ignored by default; run with:
//!   cargo test --release -p evoc-core --test native_bench -- --ignored --nocapture
//! Safe to delete once the numbers are recorded in comparison/results.md.

mod common;

use common::{as_f32, load};
use evoc_core::clustering::{EVoC, EvocClustersOptions};
use std::time::Instant;

fn json_f32_mat(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter().map(as_f32))
        .collect()
}

fn bench_one(component: &str, case: &str, label: &str) {
    let fx = load(component, case);
    let n_features = fx["n_features"].as_u64().unwrap() as usize;
    let data = json_f32_mat(&fx["data"]);
    let n_samples = data.len() / n_features;

    let warmup = 2;
    let iters = 10;
    let mut times = Vec::with_capacity(iters);
    for i in 0..(warmup + iters) {
        let opts = EvocClustersOptions::default();
        let mut evoc = EVoC::new(0, opts);
        let t0 = Instant::now();
        evoc.fit_predict(&data, n_features);
        let dt = t0.elapsed();
        if i >= warmup {
            times.push(dt.as_secs_f64() * 1000.0);
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    let min = times[0];
    let max = times[times.len() - 1];
    println!("  {label} (n={n_samples}): median={median:.3} ms  min={min:.3} ms  max={max:.3} ms");
}

#[test]
#[ignore]
fn native_fit_predict_bench() {
    println!("\n=== native rust EVoC::fit_predict (release build) ===");
    bench_one("clustering", "panel_small_balanced", "panel_small_balanced");
    bench_one(
        "clustering",
        "evoc_clusters_e2e_ari_check",
        "evoc_clusters_e2e_ari_check",
    );
    bench_one(
        "clustering",
        "panel_large_many_clusters",
        "panel_large_many_clusters",
    );
}
