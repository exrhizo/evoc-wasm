//! End-to-end ARI-based validation for `evoc_clusters`/`EVoC::fit_predict`
//! (phase 9). NOT a bit-exact fixture test like the rest of this port --
//! `label_propagation_init`'s recursive path reaches randomized PCA, which is
//! intentionally NOT bit-parity with sklearn's internal randomized-SVD draws
//! (see PORTING.md / PORT_NOTES.md / `rust/core/PCA_VALIDATION.md`). Instead
//! this compares the Rust port's cluster labels, across a **pre-registered
//! panel of seeds 0..19**, against a real
//! `evoc.EVoC(random_state=42).fit_predict(X)` run (recorded in
//! `fixtures/clustering/evoc_clusters_e2e_ari_check.json` by
//! `scripts/gen_evoc_clusters_ari_fixture.py`) via Adjusted Rand Index.
//!
//! A single-seed ARI number is NOT trustworthy validation on its own: a
//! background measurement of Python's *own* seed-to-seed self-agreement
//! (`scripts/measure_python_null.py`, cached in
//! `fixtures/clustering/python_self_agreement_null.json`) found that
//! `evoc.EVoC`'s output is itself broadly variable across `random_state`
//! (median pairwise ARI ~0.52, p25/p75 0.42/0.63, min 0.07, max 0.95, across
//! 190 pairs from 20 seeds on this exact dataset). Any single-seed
//! port-vs-Python ARI (e.g. the old cherry-picked 0.8033 at seed 42) is thus
//! nearly meaningless in isolation -- it could land anywhere in that range
//! by chance alone, independent of port correctness. See
//! `rust/core/PCA_VALIDATION.md` for the full methodology and gate
//! reasoning.

mod common;

use common::{as_f32, load};
use evoc_core::clustering::{EVoC, EvocClustersOptions};

fn json_f32_mat(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter().map(as_f32))
        .collect()
}

fn json_i64_vec(v: &serde_json::Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

/// `n choose 2`, for non-negative `n`.
fn n_choose_2(n: i64) -> f64 {
    (n as f64) * ((n - 1) as f64) / 2.0
}

/// Adjusted Rand Index between two integer label assignments of the same
/// length (matches `sklearn.metrics.adjusted_rand_score`'s definition,
/// including its treatment of any label value -- e.g. `-1` for noise -- as
/// just another ordinary cluster id, not special-cased).
fn adjusted_rand_index(a: &[i64], b: &[i64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len();

    let mut a_labels: Vec<i64> = a.to_vec();
    a_labels.sort_unstable();
    a_labels.dedup();
    let mut b_labels: Vec<i64> = b.to_vec();
    b_labels.sort_unstable();
    b_labels.dedup();

    let a_index: std::collections::HashMap<i64, usize> =
        a_labels.iter().enumerate().map(|(i, &l)| (l, i)).collect();
    let b_index: std::collections::HashMap<i64, usize> =
        b_labels.iter().enumerate().map(|(i, &l)| (l, i)).collect();

    let mut contingency = vec![0i64; a_labels.len() * b_labels.len()];
    for i in 0..n {
        let ai = a_index[&a[i]];
        let bi = b_index[&b[i]];
        contingency[ai * b_labels.len() + bi] += 1;
    }

    let mut sum_comb_c = 0.0f64;
    for &c in contingency.iter() {
        sum_comb_c += n_choose_2(c);
    }

    let mut row_sums = vec![0i64; a_labels.len()];
    let mut col_sums = vec![0i64; b_labels.len()];
    for i in 0..a_labels.len() {
        for j in 0..b_labels.len() {
            row_sums[i] += contingency[i * b_labels.len() + j];
            col_sums[j] += contingency[i * b_labels.len() + j];
        }
    }

    let sum_comb_row: f64 = row_sums.iter().map(|&r| n_choose_2(r)).sum();
    let sum_comb_col: f64 = col_sums.iter().map(|&c| n_choose_2(c)).sum();

    let total_comb = n_choose_2(n as i64);
    let expected_index = sum_comb_row * sum_comb_col / total_comb;
    let max_index = 0.5 * (sum_comb_row + sum_comb_col);

    if (max_index - expected_index).abs() < 1e-12 {
        // Degenerate case (e.g. every point its own cluster in both, or a
        // single shared cluster): sklearn returns 1.0 here.
        return 1.0;
    }

    (sum_comb_c - expected_index) / (max_index - expected_index)
}

#[test]
fn ari_sanity_checks() {
    let a = [0i64, 0, 0, 1, 1, 1];
    assert!((adjusted_rand_index(&a, &a) - 1.0).abs() < 1e-9);

    let b = [0i64, 1, 0, 1, 0, 1]; // effectively random wrt a
    let ari_random = adjusted_rand_index(&a, &b);
    assert!(ari_random < 1.0);

    // permuting labels doesn't change ARI
    let c = [5i64, 5, 5, 9, 9, 9];
    assert!((adjusted_rand_index(&a, &c) - 1.0).abs() < 1e-9);
}

/// Simple linear-interpolation percentile, matching
/// `scripts/measure_python_null.py`'s `percentile()` (and thus how the
/// cached null distribution's p25/median/p75 were computed) so the two
/// distributions are directly comparable.
fn percentile(sorted_vals: &[f64], p: f64) -> f64 {
    if sorted_vals.is_empty() {
        return f64::NAN;
    }
    let k = (sorted_vals.len() - 1) as f64 * (p / 100.0);
    let f = k.floor() as usize;
    let c = (f + 1).min(sorted_vals.len() - 1);
    if f == c {
        sorted_vals[f]
    } else {
        sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f as f64)
    }
}

struct Distribution {
    min: f64,
    p25: f64,
    median: f64,
    p75: f64,
    max: f64,
}

fn summarize(values: &[f64]) -> Distribution {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Distribution {
        min: sorted[0],
        p25: percentile(&sorted, 25.0),
        median: percentile(&sorted, 50.0),
        p75: percentile(&sorted, 75.0),
        max: *sorted.last().unwrap(),
    }
}

/// Pre-registered seed panel: seeds 0..19, the SAME 20 seeds used by
/// `scripts/measure_python_null.py` to measure Python's own self-agreement
/// null distribution. Fixed here (not chosen post-hoc) so the reported
/// panel is a measurement, not a cherry-picked maximum.
const SEED_PANEL: [i64; 20] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
];

/// Measured null (`fixtures/clustering/python_self_agreement_null.json`):
/// Python `evoc.EVoC`'s own pairwise ARI across random_state 0..19 on this
/// exact dataset has median 0.5247, p25 0.4226, p75 0.6329, min 0.0746,
/// max 0.9459 (190 pairs). This is Python disagreeing with ITSELF -- so a
/// port that merely lands in a comparably broad range is not thereby
/// buggy; the algorithm (small dataset, randomized PCA feeding a recursive
/// label-prop/HDBSCAN-style extraction) is inherently seed-sensitive.
const PYTHON_NULL_MEDIAN: f64 = 0.5246615432510635;

/// Gate 1: the port's median ARI-vs-Python(seed 42) should be within a
/// margin of Python's own median self-agreement. The margin (0.15) accounts
/// for port-vs-Python not being quite the same comparison as
/// Python-vs-Python (different implementation of the *same* algorithm,
/// compared against one fixed Python reference run rather than pairwise
/// among many Python runs) -- so some looseness relative to the null's own
/// median is expected without indicating a port defect. This is NOT a bar
/// requiring the port to beat Python's internal consistency; it only
/// requires the port not be dramatically worse than Python is with itself.
const MEDIAN_GATE: f64 = PYTHON_NULL_MEDIAN - 0.15; // ~0.3747

/// Gate 2: floor on the panel minimum, as a smoke alarm for pathological
/// seeds (e.g. total misclustering, empty labels, a crash-adjacent
/// degenerate case) rather than "ordinary" seed-sensitivity. Set loosely
/// below Python's own observed floor (0.0746) since with only 20 samples
/// per side we should not expect our floor to exceed Python's.
const MIN_GATE: f64 = 0.15;

#[test]
fn evoc_clusters_e2e_ari_panel_against_python_evoc() {
    let fixture = load("clustering", "evoc_clusters_e2e_ari_check");
    let n_samples = fixture["n_samples"].as_u64().unwrap() as usize;
    let n_features = fixture["n_features"].as_u64().unwrap() as usize;
    let data = json_f32_mat(&fixture["data"]);
    let python_labels = json_i64_vec(&fixture["python_labels"]);

    assert_eq!(data.len(), n_samples * n_features);
    assert_eq!(python_labels.len(), n_samples);

    // evoc.EVoC()'s own defaults, matching the fixture generator script.
    fn opts() -> EvocClustersOptions {
        EvocClustersOptions {
            noise_level: 0.5,
            base_min_cluster_size: 5,
            n_neighbors: 15,
            min_samples: 5,
            n_epochs: 50,
            neighbor_scale: 1.0,
            min_similarity_threshold: 0.2,
            max_layers: 10,
            n_label_prop_iter: 20,
            node_embedding_dim: None,
        }
    }

    let mut aris: Vec<f64> = Vec::with_capacity(SEED_PANEL.len());
    for &seed in SEED_PANEL.iter() {
        let mut evoc = EVoC::new(seed, opts());
        let rust_labels = evoc.fit_predict(&data, n_features).to_vec();
        let ari = adjusted_rand_index(&python_labels, &rust_labels);
        eprintln!("  seed {seed:>2}: ARI vs Python(seed=42) = {ari:.4}");
        aris.push(ari);
    }

    let dist = summarize(&aris);
    eprintln!(
        "evoc_clusters e2e ARI panel (Rust, seeds 0..19) vs Python evoc.EVoC(seed=42): \
         min={:.4} p25={:.4} median={:.4} p75={:.4} max={:.4}",
        dist.min, dist.p25, dist.median, dist.p75, dist.max
    );
    eprintln!(
        "  (Python-vs-Python self-agreement null: median={PYTHON_NULL_MEDIAN:.4}; \
         median gate = null_median - 0.15 = {MEDIAN_GATE:.4}; min gate = {MIN_GATE:.4})"
    );

    assert!(
        dist.median >= MEDIAN_GATE,
        "panel median ARI {:.4} < gate {:.4} (Python self-agreement null median {:.4} - 0.15 \
         margin). See PCA_VALIDATION.md's panel methodology section.",
        dist.median,
        MEDIAN_GATE,
        PYTHON_NULL_MEDIAN
    );
    assert!(
        dist.min >= MIN_GATE,
        "panel min ARI {:.4} < floor {:.4}; a pathological seed suggests a real bug, not just \
         ordinary seed-sensitivity. See PCA_VALIDATION.md.",
        dist.min,
        MIN_GATE
    );
}
