//! Fixture parity tests for `evoc_core::boruvka`.

mod common;

use common::{load, Tolerance};
use evoc_core::boruvka::{initialize_boruvka_from_knn, parallel_boruvka};
use evoc_core::disjoint_set::ds_rank_create;
use evoc_core::kdtree::build_kdtree;

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

/// Assert edge endpoints (exact) and distances (tolerance) against a fixture
/// storing separate point_a / point_b / distance arrays.
fn assert_edges(edges: &[[f32; 3]], fixture: &serde_json::Value, dist_tol: &Tolerance, ctx: &str) {
    let point_a: Vec<i32> = fixture["expected"]["point_a"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as i32)
        .collect();
    let point_b: Vec<i32> = fixture["expected"]["point_b"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as i32)
        .collect();
    let distance = json_f32_vec(&fixture["expected"]["distance"]);

    assert_eq!(edges.len(), point_a.len(), "{ctx}: edge count");
    for (i, e) in edges.iter().enumerate() {
        assert_eq!(e[0] as i32, point_a[i], "{ctx}: edge[{i}].point_a");
        assert_eq!(e[1] as i32, point_b[i], "{ctx}: edge[{i}].point_b");
        dist_tol.assert_close(e[2], distance[i], &format!("{ctx}: edge[{i}].distance"));
    }
}

#[test]
fn initialize_boruvka_from_knn_n80_d4() {
    let fixture = load("boruvka", "initialize_boruvka_from_knn_n80_d4");
    let dist_tol = Tolerance::parse_for_field(&fixture["tolerance"], "distance");

    let (knn_indices, k) = json_i32_matrix(&fixture["inputs"]["knn_indices"]);
    let (knn_distances, kd) = json_f32_matrix(&fixture["inputs"]["knn_distances_rdist"]);
    assert_eq!(k, kd);
    let core_distances = json_f32_vec(&fixture["inputs"]["core_distances"]);
    let n_points = fixture["inputs"]["n_points"].as_u64().unwrap() as usize;

    let mut ds = ds_rank_create(n_points);
    let edges =
        initialize_boruvka_from_knn(&knn_indices, &knn_distances, &core_distances, k, &mut ds);

    assert_edges(&edges, &fixture, &dist_tol, "initialize_boruvka_from_knn");
}

fn run_parallel_boruvka_case(case: &str, reproducible: bool) {
    let fixture = load("boruvka", case);
    let dist_tol = Tolerance::parse_for_field(&fixture["tolerance"], "distance");

    let (data, n_features) = json_f32_matrix(&fixture["inputs"]["data"]);
    let leaf_size = fixture["inputs"]["leaf_size"].as_u64().unwrap() as usize;
    let n_threads = fixture["inputs"]["n_threads"].as_u64().unwrap() as usize;
    let min_samples = fixture["inputs"]["min_samples"].as_u64().unwrap() as usize;

    let tree = build_kdtree(&data, n_features, leaf_size);
    let edges = parallel_boruvka(&tree, n_threads, min_samples, reproducible);

    assert_edges(&edges, &fixture, &dist_tol, case);
}

#[test]
fn parallel_boruvka_n80_d4_reproducible_false() {
    run_parallel_boruvka_case("parallel_boruvka_n80_d4_reproducible_false", false);
}

#[test]
fn parallel_boruvka_n80_d4_reproducible_true() {
    run_parallel_boruvka_case("parallel_boruvka_n80_d4_reproducible_true", true);
}
