//! Fixture parity tests for `evoc_core::label_propagation`.

mod common;

use common::{as_f32, load};
use evoc_core::label_propagation::{
    label_outliers, label_prop_iteration, label_prop_loop, label_propagation_init, remap_labels,
};
use evoc_core::rng::RngState;

fn json_i64_vec(v: &serde_json::Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

fn json_f32_vec(v: &serde_json::Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(as_f32).collect()
}

fn json_f32_mat(v: &serde_json::Value) -> Vec<Vec<f32>> {
    v.as_array().unwrap().iter().map(json_f32_vec).collect()
}

fn flatten(m: &[Vec<f32>]) -> Vec<f32> {
    m.iter().flatten().cloned().collect()
}

fn json_u32_vec(v: &serde_json::Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

fn rng_state_from(v: &serde_json::Value) -> RngState {
    let a = json_i64_vec(v);
    [a[0], a[1], a[2]]
}

#[test]
fn label_prop_iteration_n120() {
    let fixture = load("label_propagation", "label_prop_iteration_n120");
    let inputs = &fixture["inputs"];

    let indptr = json_i64_vec(&inputs["indptr"]);
    let indices = json_i64_vec(&inputs["indices"]);
    let data = json_f32_vec(&inputs["data"]);
    let labels = json_i64_vec(&inputs["labels"]);
    let rng_state = rng_state_from(&inputs["rng_state"]);

    let result = label_prop_iteration(&indptr, &indices, &data, &labels, &rng_state);
    let expected = json_i64_vec(&fixture["expected"]["labels"]);
    assert_eq!(result, expected);
}

#[test]
fn label_outliers_n120() {
    let fixture = load("label_propagation", "label_outliers_n120");
    let inputs = &fixture["inputs"];

    let indptr = json_i64_vec(&inputs["indptr"]);
    let indices = json_i64_vec(&inputs["indices"]);
    let mut labels = json_i64_vec(&inputs["labels"]);
    let rng_state = rng_state_from(&inputs["rng_state"]);

    label_outliers(&indptr, &indices, &mut labels, &rng_state);
    let expected = json_i64_vec(&fixture["expected"]["labels"]);
    assert_eq!(labels, expected);
}

#[test]
fn label_outliers_disconnected() {
    let fixture = load("label_propagation", "label_outliers_disconnected");
    let inputs = &fixture["inputs"];

    let indptr = json_i64_vec(&inputs["indptr"]);
    let indices = json_i64_vec(&inputs["indices"]);
    let mut labels = json_i64_vec(&inputs["labels"]);
    let rng_state = rng_state_from(&inputs["rng_state"]);

    label_outliers(&indptr, &indices, &mut labels, &rng_state);
    let expected = json_i64_vec(&fixture["expected"]["labels"]);
    assert_eq!(labels, expected);
}

#[test]
fn remap_labels_fixture() {
    let fixture = load("label_propagation", "remap_labels");
    let mut labels = json_i64_vec(&fixture["inputs"]["labels"]);
    remap_labels(&mut labels);
    let expected = json_i64_vec(&fixture["expected"]["labels"]);
    assert_eq!(labels, expected);
}

#[test]
fn label_prop_loop_n120() {
    let fixture = load("label_propagation", "label_prop_loop_n120");
    let inputs = &fixture["inputs"];

    let indptr = json_i64_vec(&inputs["indptr"]);
    let indices = json_i64_vec(&inputs["indices"]);
    let data = json_f32_vec(&inputs["data"]);
    let n_points = inputs["n_points"].as_u64().unwrap() as usize;
    let n_iter = inputs["n_iter"].as_u64().unwrap() as usize;
    let rng_state = rng_state_from(&inputs["rng_state"]);
    let seed_position_draws: Vec<usize> = inputs["seed_position_draws"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();

    let labels = vec![-1i64; n_points];
    let result = label_prop_loop(
        &indptr,
        &indices,
        &data,
        labels,
        &rng_state,
        &seed_position_draws,
        n_iter,
    );

    let expected = json_i64_vec(&fixture["expected"]["partition"]);
    assert_eq!(result, expected);
}

#[test]
fn label_propagation_init_n120() {
    let fixture = load("label_propagation", "label_propagation_init_n120");
    let tol_rel = fixture["tolerance"]["rel"].as_f64().unwrap();
    let tol_abs = fixture["tolerance"]["abs"].as_f64().unwrap();
    let inputs = &fixture["inputs"];

    assert!(!inputs["recursive_init"].as_bool().unwrap());
    assert_eq!(inputs["upscaling"].as_str().unwrap(), "partition_expander");

    let indptr = json_i64_vec(&inputs["indptr"]);
    let indices = json_i64_vec(&inputs["indices"]);
    let data = json_f32_vec(&inputs["data"]);
    let n_points = inputs["n_points"].as_u64().unwrap() as usize;
    let n_label_prop_iter = inputs["n_label_prop_iter"].as_u64().unwrap() as usize;
    let n_embedding_epochs = inputs["n_embedding_epochs"].as_u64().unwrap() as usize;
    let n_components = inputs["n_components"].as_u64().unwrap() as usize;
    let scaling = as_f32(&inputs["scaling"]);
    let noise_level = as_f32(&inputs["noise_level"]);
    let lp_rng_state = rng_state_from(&inputs["lp_rng_state"]);
    let lp_seed_position_draws: Vec<usize> = inputs["lp_seed_position_draws"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    let node_embedding_init_normal = flatten(&json_f32_mat(&inputs["node_embedding_init_normal"]));
    let node_embedding_rng_val: Vec<u32> = inputs["node_embedding_rng_val"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    let node_embedding_shuffles: Vec<Vec<u32>> = inputs["node_embedding_shuffles"]
        .as_array()
        .unwrap()
        .iter()
        .map(json_u32_vec)
        .collect();

    let got = label_propagation_init(
        &indptr,
        &indices,
        &data,
        n_points,
        n_label_prop_iter,
        n_embedding_epochs,
        n_components,
        scaling,
        noise_level,
        &lp_rng_state,
        &lp_seed_position_draws,
        &node_embedding_init_normal,
        &node_embedding_rng_val,
        &node_embedding_shuffles,
    );

    let expected = flatten(&json_f32_mat(&fixture["expected"]["embedding"]));
    assert_eq!(got.len(), expected.len());
    for (i, (&a, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (a as f64 - e as f64).abs();
        let allowed = tol_abs + tol_rel * (e as f64).abs();
        assert!(
            diff <= allowed,
            "embedding[{i}]: actual={a} expected={e} diff={diff} allowed={allowed}"
        );
    }
}
