//! Dataset-panel extension of `evoc_clusters_ari.rs` (VALIDATION_PLAN.md
//! section 2C). The original ARI panel only ever exercised ONE synthetic
//! dataset shape (240x24, 6 balanced clusters --
//! `evoc_clusters_e2e_ari_check.json`). This file repeats the same
//! panel-vs-null methodology (see that file's and `PCA_VALIDATION.md`'s
//! doc comments for the full rationale) across 4 additional datasets
//! varying n, dimensionality, cluster count, class balance, and noise scale
//! (`scripts/gen_evoc_clusters_panel_fixtures.py`,
//! `fixtures/clustering/panel_*.json` / `panel_*_null.json`).
//!
//! Per dataset: a SMALLER seed panel (0..9, not 0..19, since this is 4x the
//! datasets) is run through `EVoC::fit_predict` and compared via Adjusted
//! Rand Index against one fixed Python reference run
//! (`evoc.EVoC(random_state=42)`), gated against that dataset's OWN measured
//! Python self-agreement null (not the original single-dataset null -- each
//! dataset shape has different inherent stochasticity, so each gets its own
//! null). Also records simple Python-free internal metrics (n_clusters
//! found, noise fraction) per run for an eyeball sanity check against
//! Python's own recorded metrics, per PART 1 item 3 of the task.

mod common;

use common::{as_f32, load};
use uvoc_core::clustering::{EVoC, EvocClustersOptions};

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

fn n_choose_2(n: i64) -> f64 {
    (n as f64) * ((n - 1) as f64) / 2.0
}

/// Same ARI implementation as `evoc_clusters_ari.rs` (duplicated per-file,
/// as each `tests/*.rs` file is its own crate -- see `common/mod.rs`'s doc
/// comment on that convention).
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
        return 1.0;
    }

    (sum_comb_c - expected_index) / (max_index - expected_index)
}

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

/// Smaller seed panel than the original 20-seed panel (0..19), since this
/// covers 4 additional dataset shapes; matches
/// `scripts/gen_evoc_clusters_panel_fixtures.py`'s `NULL_SEEDS`.
const SEED_PANEL: [i64; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

/// One dataset panel case: fixture case name, that dataset's OWN measured
/// Python self-agreement null median/min (from its `panel_<name>_null.json`
/// fixture's `summary`), used to derive per-dataset gates exactly as
/// `evoc_clusters_ari.rs` derives its single gate from the original null.
struct PanelCase {
    name: &'static str,
    null_median: f64,
    null_min: f64,
}

/// Values below are copied from the `summary` block of each
/// `panel_<name>_null.json` fixture generated by
/// `scripts/gen_evoc_clusters_panel_fixtures.py` (seeds 0..9, 45 pairs
/// each). Re-run the generator and update these (plus the fixtures) if the
/// generator's dataset parameters ever change.
const PANEL_CASES: [PanelCase; 4] = [
    PanelCase {
        name: "small_balanced",
        null_median: 0.8821553583424672,
        null_min: 0.5675081126411357,
    },
    PanelCase {
        name: "large_many_clusters",
        null_median: 0.40498395387834163,
        null_min: 0.02973372261369582,
    },
    PanelCase {
        name: "imbalanced",
        null_median: 0.2739452749407144,
        null_min: 0.02964012012244541,
    },
    PanelCase {
        name: "high_dim_noisy",
        null_median: 0.6349687720666672,
        null_min: 0.20729166600989643,
    },
];

/// Same margin as `evoc_clusters_ari.rs`'s `MEDIAN_GATE` derivation.
const MEDIAN_MARGIN: f64 = 0.15;
/// Absolute floor gate, set loosely below the smallest observed null-min
/// across all 4 new datasets (0.0296) as a smoke alarm for pathological
/// seeds/crashes rather than ordinary seed-sensitivity -- deliberately not
/// derived per-dataset from null_min the way the median gate is, since with
/// only 10 self-null seeds (45 pairs) some datasets' null_min is itself
/// close to the degenerate floor and a `null_min - margin` gate would go
/// negative.
const MIN_FLOOR: f64 = 0.01;

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

#[test]
fn evoc_clusters_dataset_panel_ari_against_python() {
    let mut any_failure = String::new();

    for case in PANEL_CASES.iter() {
        let fixture = load("clustering", &format!("panel_{}", case.name));
        let n_samples = fixture["n_samples"].as_u64().unwrap() as usize;
        let n_features = fixture["n_features"].as_u64().unwrap() as usize;
        let data = json_f32_mat(&fixture["data"]);
        let python_labels = json_i64_vec(&fixture["python_labels"]);
        let python_metrics = &fixture["python_internal_metrics"];
        let python_n_clusters = python_metrics["n_clusters"].as_i64().unwrap();
        let python_noise_frac = python_metrics["noise_fraction"].as_f64().unwrap();

        assert_eq!(data.len(), n_samples * n_features);
        assert_eq!(python_labels.len(), n_samples);

        let mut aris: Vec<f64> = Vec::with_capacity(SEED_PANEL.len());
        let mut n_clusters_found: Vec<i64> = Vec::with_capacity(SEED_PANEL.len());
        let mut noise_fracs: Vec<f64> = Vec::with_capacity(SEED_PANEL.len());

        for &seed in SEED_PANEL.iter() {
            let mut evoc = EVoC::new(seed, opts());
            let rust_labels = evoc.fit_predict(&data, n_features).to_vec();
            let ari = adjusted_rand_index(&python_labels, &rust_labels);
            aris.push(ari);

            let n_clusters = rust_labels
                .iter()
                .filter(|&&l| l != -1)
                .collect::<std::collections::HashSet<_>>()
                .len() as i64;
            let n_noise = rust_labels.iter().filter(|&&l| l == -1).count();
            n_clusters_found.push(n_clusters);
            noise_fracs.push(n_noise as f64 / n_samples as f64);
        }

        let dist = summarize(&aris);
        let median_gate = case.null_median - MEDIAN_MARGIN;
        eprintln!(
            "[{}] Rust ARI panel (seeds 0..9) vs Python(seed=42): min={:.4} p25={:.4} \
             median={:.4} p75={:.4} max={:.4} (null_median={:.4} gate={:.4}, \
             null_min={:.4} floor={:.4})",
            case.name,
            dist.min,
            dist.p25,
            dist.median,
            dist.p75,
            dist.max,
            case.null_median,
            median_gate,
            case.null_min,
            MIN_FLOOR
        );
        eprintln!(
            "[{}] internal metrics: Python n_clusters={} noise_frac={:.4}; Rust \
             n_clusters found min/median/max={}/{}/{} noise_frac min/max={:.4}/{:.4}",
            case.name,
            python_n_clusters,
            python_noise_frac,
            n_clusters_found.iter().min().unwrap(),
            {
                let mut s = n_clusters_found.clone();
                s.sort_unstable();
                s[s.len() / 2]
            },
            n_clusters_found.iter().max().unwrap(),
            noise_fracs.iter().cloned().fold(f64::INFINITY, f64::min),
            noise_fracs
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max),
        );

        if dist.median < median_gate {
            any_failure += &format!(
                "[{}] median ARI {:.4} < gate {:.4}\n",
                case.name, dist.median, median_gate
            );
        }
        if dist.min < MIN_FLOOR {
            any_failure += &format!(
                "[{}] min ARI {:.4} < floor {:.4}\n",
                case.name, dist.min, MIN_FLOOR
            );
        }
    }

    assert!(
        any_failure.is_empty(),
        "dataset panel ARI gate failures:\n{any_failure}"
    );
}
