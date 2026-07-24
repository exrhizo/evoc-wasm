//! End-to-end ARI-based validation for `evoc_clusters`/`EVoC::fit_predict`
//! (phase 9). NOT a bit-exact fixture test like the rest of this port --
//! `label_propagation_init`'s recursive path reaches randomized PCA, which is
//! intentionally NOT bit-parity with sklearn's internal randomized-SVD draws
//! (see PORTING.md / PORT_NOTES.md / `rust/core/PCA_VALIDATION.md`). Instead
//! this compares the Rust port's cluster labels against a real
//! `evoc.EVoC(random_state=42).fit_predict(X)` run (recorded in
//! `fixtures/clustering/evoc_clusters_e2e_ari_check.json` by
//! `scripts/gen_evoc_clusters_ari_fixture.py`) via Adjusted Rand Index,
//! with a pragmatic bar of ARI > 0.8.

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

#[test]
fn evoc_clusters_e2e_ari_against_python_evoc() {
    let fixture = load("clustering", "evoc_clusters_e2e_ari_check");
    let n_samples = fixture["n_samples"].as_u64().unwrap() as usize;
    let n_features = fixture["n_features"].as_u64().unwrap() as usize;
    let data = json_f32_mat(&fixture["data"]);
    let python_labels = json_i64_vec(&fixture["python_labels"]);

    assert_eq!(data.len(), n_samples * n_features);
    assert_eq!(python_labels.len(), n_samples);

    // evoc.EVoC()'s own defaults, matching the fixture generator script.
    let opts = EvocClustersOptions {
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
    };

    let mut evoc = EVoC::new(42, opts);
    let rust_labels = evoc.fit_predict(&data, n_features).to_vec();

    let ari = adjusted_rand_index(&python_labels, &rust_labels);
    eprintln!("evoc_clusters e2e ARI against Python evoc.EVoC: {ari:.4}");

    assert!(
        ari > 0.8,
        "expected ARI > 0.8 against Python evoc.EVoC reference labels, got {ari:.4}. \
         See PCA_VALIDATION.md: this is a pragmatic quality bar, not bit-exact parity \
         (impossible once randomized PCA diverges from sklearn's internal SVD draws)."
    );
}
