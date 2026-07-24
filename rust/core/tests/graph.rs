//! Fixture parity tests for `uvoc_core::graph_construction` and
//! `uvoc_core::knn_graph`.

mod common;

use common::{load, Tolerance};
use uvoc_core::graph_construction::{
    compute_membership_strengths, neighbor_graph_matrix, smooth_knn_dist,
};
use uvoc_core::knn_graph::knn_graph;
use uvoc_core::rng::RngState;

fn json_i32_vec(v: &serde_json::Value) -> Vec<i32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as i32)
        .collect()
}

fn json_f32_vec(v: &serde_json::Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(common::as_f32).collect()
}

fn json_i32_matrix(v: &serde_json::Value) -> (Vec<i32>, usize) {
    let rows = v.as_array().unwrap();
    let width = rows[0].as_array().unwrap().len();
    let mut flat = Vec::with_capacity(rows.len() * width);
    for row in rows {
        let row = row.as_array().unwrap();
        assert_eq!(row.len(), width);
        for x in row {
            flat.push(x.as_i64().unwrap() as i32);
        }
    }
    (flat, width)
}

fn json_f32_matrix(v: &serde_json::Value) -> (Vec<f32>, usize) {
    let rows = v.as_array().unwrap();
    let width = rows[0].as_array().unwrap().len();
    let mut flat = Vec::with_capacity(rows.len() * width);
    for row in rows {
        let row = row.as_array().unwrap();
        assert_eq!(row.len(), width);
        for x in row {
            flat.push(common::as_f32(x));
        }
    }
    (flat, width)
}

fn json_rng_state(v: &serde_json::Value) -> RngState {
    let l: Vec<i64> = v
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect();
    [l[0], l[1], l[2]]
}

#[test]
fn smooth_knn_dist_n40_k15() {
    let fixture = load("graph", "smooth_knn_dist_n40_k15");
    let tol = Tolerance::parse(&fixture["tolerance"]);

    let (distances, n_neighbors) = json_f32_matrix(&fixture["inputs"]["distances"]);
    let k = common::as_f32(&fixture["inputs"]["k"]);
    let n_iter = fixture["inputs"]["n_iter"].as_u64().unwrap() as usize;
    let bandwidth = common::as_f32(&fixture["inputs"]["bandwidth"]);

    let (sigmas, rhos) = smooth_knn_dist(&distances, n_neighbors, k, n_iter, bandwidth);

    let expected_sigmas = json_f32_vec(&fixture["expected"]["sigmas"]);
    let expected_rhos = json_f32_vec(&fixture["expected"]["rhos"]);
    assert_eq!(sigmas.len(), expected_sigmas.len());
    for (i, (&a, &e)) in sigmas.iter().zip(expected_sigmas.iter()).enumerate() {
        tol.assert_close(a, e, &format!("sigmas[{i}]"));
    }
    for (i, (&a, &e)) in rhos.iter().zip(expected_rhos.iter()).enumerate() {
        tol.assert_close(a, e, &format!("rhos[{i}]"));
    }
}

#[test]
fn compute_membership_strengths_n40_k15() {
    let fixture = load("graph", "compute_membership_strengths_n40_k15");

    let (knn_indices, n_neighbors) = json_i32_matrix(&fixture["inputs"]["knn_indices"]);
    let (knn_dists, _) = json_f32_matrix(&fixture["inputs"]["knn_dists"]);
    let sigmas = json_f32_vec(&fixture["inputs"]["sigmas"]);
    let rhos = json_f32_vec(&fixture["inputs"]["rhos"]);

    let (rows, cols, vals) =
        compute_membership_strengths(&knn_indices, &knn_dists, &sigmas, &rhos, n_neighbors);

    assert_eq!(rows, json_i32_vec(&fixture["expected"]["rows"]), "rows");
    assert_eq!(cols, json_i32_vec(&fixture["expected"]["cols"]), "cols");

    let vals_tol = Tolerance::parse_for_field(&fixture["tolerance"], "vals");
    let expected_vals = json_f32_vec(&fixture["expected"]["vals"]);
    assert_eq!(vals.len(), expected_vals.len());
    for (i, (&a, &e)) in vals.iter().zip(expected_vals.iter()).enumerate() {
        vals_tol.assert_close(a, e, &format!("vals[{i}]"));
    }
}

fn run_neighbor_graph_case(case: &str, symmetrize: bool) {
    let fixture = load("graph", case);

    let (knn_indices, n_neighbors) = json_i32_matrix(&fixture["inputs"]["knn_indices"]);
    let (knn_dists, _) = json_f32_matrix(&fixture["inputs"]["knn_dists_transformed"]);
    let n_neighbors_f = common::as_f32(&fixture["inputs"]["n_neighbors"]);

    let graph = neighbor_graph_matrix(
        n_neighbors_f,
        &knn_indices,
        &knn_dists,
        n_neighbors,
        symmetrize,
    );

    assert_eq!(
        graph.indptr,
        json_i32_vec(&fixture["expected"]["indptr"]),
        "{case}: indptr"
    );
    assert_eq!(
        graph.indices,
        json_i32_vec(&fixture["expected"]["indices"]),
        "{case}: indices"
    );

    let data_tol = Tolerance::parse_for_field(&fixture["tolerance"], "data");
    let expected_data = json_f32_vec(&fixture["expected"]["data"]);
    assert_eq!(graph.data.len(), expected_data.len(), "{case}: nnz");
    for (i, (&a, &e)) in graph.data.iter().zip(expected_data.iter()).enumerate() {
        data_tol.assert_close(a, e, &format!("{case}: data[{i}]"));
    }
}

#[test]
fn neighbor_graph_matrix_symmetrize_true() {
    run_neighbor_graph_case("neighbor_graph_matrix_symmetrize_true", true);
}

#[test]
fn neighbor_graph_matrix_symmetrize_false() {
    run_neighbor_graph_case("neighbor_graph_matrix_symmetrize_false", false);
}

#[test]
fn knn_graph_n120_d8_k10_seed42() {
    let fixture = load("graph", "knn_graph_n120_d8_k10_seed42");

    let (data, n_features) = json_f32_matrix(&fixture["inputs"]["data"]);
    let n_neighbors = fixture["inputs"]["n_neighbors"].as_u64().unwrap() as usize;
    let rng_state = json_rng_state(&fixture["inputs"]["rng_state"]);
    let mut rng_states: Vec<RngState> = fixture["inputs"]["rng_states"]
        .as_array()
        .unwrap()
        .iter()
        .map(json_rng_state)
        .collect();
    let leaf_size = fixture["inputs"]["leaf_size"].as_u64().unwrap() as usize;
    let n_iters = fixture["inputs"]["n_iters"].as_u64().unwrap() as usize;
    let effective_max_candidates = fixture["inputs"]["effective_max_candidates"]
        .as_u64()
        .unwrap() as usize;
    let delta = fixture["inputs"]["delta"].as_f64().unwrap();
    let delta_improv = fixture["inputs"]["delta_improv"].as_f64();
    let use_sorted_updates = fixture["inputs"]["use_sorted_updates"].as_bool().unwrap();

    let (nn_inds, nn_dists) = knn_graph(
        &data,
        n_features,
        n_neighbors,
        &rng_state,
        &mut rng_states,
        leaf_size,
        200,
        effective_max_candidates,
        n_iters,
        delta,
        delta_improv,
        use_sorted_updates,
        1,
        true,
    );

    let (expected_inds, k) = json_i32_matrix(&fixture["expected"]["nn_inds"]);
    assert_eq!(k, n_neighbors);
    assert_eq!(nn_inds, expected_inds, "nn_inds");

    let dist_tol = Tolerance::parse_for_field(&fixture["tolerance"], "nn_dists");
    let (expected_dists, _) = json_f32_matrix(&fixture["expected"]["nn_dists"]);
    assert_eq!(nn_dists.len(), expected_dists.len());
    for (i, (&a, &e)) in nn_dists.iter().zip(expected_dists.iter()).enumerate() {
        dist_tol.assert_close(a, e, &format!("nn_dists[{}][{}]", i / k, i % k));
    }
}
