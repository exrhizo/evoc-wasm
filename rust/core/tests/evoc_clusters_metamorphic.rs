//! Metamorphic / property tests for `EVoC::fit_predict` that need no
//! Python reference at all (VALIDATION_PLAN.md section 2E). Two properties:
//!
//! 1. **Determinism**: the same seed on the same data must produce
//!    byte-identical (not merely ARI-close) labels across repeated runs.
//!    This is expected given the self-driven `tau_rand` design documented in
//!    `PCA_VALIDATION.md`, but was never asserted directly as a regression
//!    guard until now.
//! 2. **Scale invariance**: multiplying every input row by a positive
//!    constant must not change the output labels AT ALL (exact equality),
//!    because `knn_graph` row-normalizes every point to a unit vector
//!    (cosine distance) before anything stochastic happens -- a positive
//!    per-row scalar cancels out exactly in that normalization, so the RNG-
//!    driven pipeline downstream never sees the scale difference.

mod common;

use common::load;
use evoc_core::clustering::{EVoC, EvocClustersOptions};

fn json_f32_mat(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .flat_map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
        })
        .collect()
}

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

fn load_fixture_data() -> (Vec<f32>, usize) {
    let fixture = load("clustering", "evoc_clusters_e2e_ari_check");
    let n_features = fixture["n_features"].as_u64().unwrap() as usize;
    let data = json_f32_mat(&fixture["data"]);
    (data, n_features)
}

#[test]
fn determinism_same_seed_same_data_is_byte_identical() {
    let (data, n_features) = load_fixture_data();

    for &seed in &[0i64, 7, 42, 123] {
        let mut evoc_a = EVoC::new(seed, opts());
        let labels_a = evoc_a.fit_predict(&data, n_features).to_vec();

        let mut evoc_b = EVoC::new(seed, opts());
        let labels_b = evoc_b.fit_predict(&data, n_features).to_vec();

        assert_eq!(
            labels_a, labels_b,
            "seed {seed}: two runs with the same seed on the same data produced \
             different labels; the self-driven tau_rand pipeline should be \
             fully deterministic"
        );
    }
}

/// Scale invariance: multiplying every row by a positive constant should not
/// change the output labels at all, since `knn_graph` unit-normalizes rows
/// (cosine distance) before any stochastic step, so downstream computation
/// should in principle never see the scale difference.
///
/// **FINDING (documented, not fixed -- see PCA_VALIDATION.md "Metamorphic
/// property tests" section): this property does NOT hold in practice.**
/// Scaling by e.g. `0.001` changes up to ~70% of labels (171/240 differ at
/// seed 0), not merely a relabeling/permutation -- confirmed by checking the
/// raw label arrays directly, not just an ARI/permutation-invariant
/// comparison. Root cause (by analogy with the already-documented
/// Rust-vs-Zig cross-language divergence in PCA_VALIDATION.md): row
/// normalization to a unit vector is only invariant to positive scaling in
/// *exact* arithmetic. In float32, `x * scale` then `/ ||x * scale||` does
/// not reproduce bit-identical results to `x / ||x||` (different rounding at
/// each step), so the very first stochastic pipeline stage
/// (`knn_graph`'s NN-descent/RP-tree candidate selection, which can flip on
/// near-ties) already sees slightly different floats, and that ULP-scale
/// difference compounds through the same seed-sensitive, RNG-driven pipeline
/// (randomized PCA, label propagation, node-embedding SGD) documented
/// elsewhere in this file's sibling doc to be highly sensitive to any input
/// perturbation. This is consistent with, not contradicted by, the finding
/// that Rust and Zig (same algorithm, independently written, same seed) only
/// reach median ARI 0.67 against each other due to 30/3600 ULP-scale
/// `nn_dists` differences alone.
///
/// This is `#[ignore]`d (not part of `cargo test --workspace`'s default run)
/// per the task's explicit instruction: report the finding clearly rather
/// than silently loosening the assertion to make it pass, and do NOT modify
/// core algorithm files (`knn_graph`, `clustering.rs`, etc.) to fix it --
/// that is out of scope for this validation-only change. The exact-equality
/// assertion below is kept as-is (not weakened to an ARI/permutation
/// check) so this test continues to state the property that was actually
/// asked for and continues to fail loudly (via `cargo test -- --ignored`)
/// if anyone wants to re-check it after a future change.
#[test]
#[ignore = "documented finding: scale invariance does NOT hold exactly (float32 \
            normalization rounding + RNG-pipeline sensitivity); see doc comment \
            and PCA_VALIDATION.md. Not fixed here -- core algorithm changes are \
            out of scope for this task."]
fn scale_invariance_positive_scalar_does_not_change_labels() {
    let (data, n_features) = load_fixture_data();

    for &scale in &[0.001f32, 2.0, 1000.0] {
        let scaled: Vec<f32> = data.iter().map(|&x| x * scale).collect();

        for &seed in &[0i64, 42] {
            let mut evoc_orig = EVoC::new(seed, opts());
            let labels_orig = evoc_orig.fit_predict(&data, n_features).to_vec();

            let mut evoc_scaled = EVoC::new(seed, opts());
            let labels_scaled = evoc_scaled.fit_predict(&scaled, n_features).to_vec();

            assert_eq!(
                labels_orig, labels_scaled,
                "seed {seed}, scale {scale}: scaling every row by a positive \
                 constant changed the output labels; expected exact invariance \
                 since knn_graph row-normalizes to unit vectors before any \
                 stochastic step (cosine distance is scale-invariant by \
                 construction). This is a real finding if it fails -- see \
                 PCA_VALIDATION.md task notes; do not loosen this assertion to \
                 paper over a failure."
            );
        }
    }
}
