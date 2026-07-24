//! Fixture parity test for `uvoc_core::clustering::build_cluster_layers`.

mod common;

use common::{load, Tolerance};
use uvoc_core::clustering::{build_cluster_layers, BuildClusterLayersOptions};

fn json_i64_vec(v: &serde_json::Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

fn json_f32_vec(v: &serde_json::Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(common::as_f32).collect()
}

#[test]
fn build_cluster_layers_n126() {
    let fixture = load("clustering", "build_cluster_layers_n126");
    let strength_tol = Tolerance::parse_for_field(&fixture["tolerance"], "membership_strengths");
    let persistence_tol = Tolerance::parse_for_field(&fixture["tolerance"], "persistence_scores");

    let rows = fixture["inputs"]["embedding"].as_array().unwrap();
    let n_features = rows[0].as_array().unwrap().len();
    let mut data = Vec::with_capacity(rows.len() * n_features);
    for row in rows {
        for x in row.as_array().unwrap() {
            data.push(common::as_f32(x));
        }
    }

    let opts = BuildClusterLayersOptions {
        min_samples: fixture["inputs"]["min_samples"].as_u64().unwrap() as usize,
        base_min_cluster_size: fixture["inputs"]["base_min_cluster_size"].as_i64().unwrap(),
        base_n_clusters: fixture["inputs"]["base_n_clusters"].as_i64(),
        reproducible_flag: fixture["inputs"]["reproducible_flag"].as_bool().unwrap(),
        min_similarity_threshold: fixture["inputs"]["min_similarity_threshold"]
            .as_f64()
            .unwrap(),
        max_layers: fixture["inputs"]["max_layers"].as_u64().unwrap() as usize,
    };

    let result = build_cluster_layers(&data, n_features, &opts);

    let expected_layers: Vec<Vec<i64>> = fixture["expected"]["cluster_layers"]
        .as_array()
        .unwrap()
        .iter()
        .map(json_i64_vec)
        .collect();
    let expected_strengths: Vec<Vec<f32>> = fixture["expected"]["membership_strengths"]
        .as_array()
        .unwrap()
        .iter()
        .map(json_f32_vec)
        .collect();
    let expected_persistence = json_f32_vec(&fixture["expected"]["persistence_scores"]);

    assert_eq!(
        result.cluster_layers.len(),
        expected_layers.len(),
        "n layers"
    );
    for (l, (got, exp)) in result
        .cluster_layers
        .iter()
        .zip(expected_layers.iter())
        .enumerate()
    {
        assert_eq!(got, exp, "cluster_layers[{l}]");
    }

    for (l, (got, exp)) in result
        .membership_strength_layers
        .iter()
        .zip(expected_strengths.iter())
        .enumerate()
    {
        assert_eq!(got.len(), exp.len(), "membership_strengths[{l}] len");
        for (i, (&a, &e)) in got.iter().zip(exp.iter()).enumerate() {
            strength_tol.assert_close(a, e, &format!("membership_strengths[{l}][{i}]"));
        }
    }

    assert_eq!(result.persistence_scores.len(), expected_persistence.len());
    for (i, (&a, &e)) in result
        .persistence_scores
        .iter()
        .zip(expected_persistence.iter())
        .enumerate()
    {
        persistence_tol.assert_close(a, e, &format!("persistence_scores[{i}]"));
    }
}
