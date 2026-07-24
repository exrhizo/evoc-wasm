//! Fixture parity tests for `uvoc_core::nnd_heap` and
//! `uvoc_core::nndescent_float`.

mod common;

use common::{load, Tolerance};
use uvoc_core::nnd_heap::{build_candidates, flagged_heap_push, NNDHeap};
use uvoc_core::nndescent_float::{make_float_forest, nn_descent_float, nn_descent_float_sorted};
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

fn json_u8_vec(v: &serde_json::Value) -> Vec<u8> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u8)
        .collect()
}

/// Flatten a JSON list-of-rows of numbers into (flat, row_width).
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
    let lanes: Vec<i64> = v
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect();
    [lanes[0], lanes[1], lanes[2]]
}

#[test]
fn flagged_heap_push_fixture() {
    let fixture = load("nndescent", "flagged_heap_push");
    assert_eq!(fixture["tolerance"], "exact");

    let k = fixture["inputs"]["k"].as_u64().unwrap() as usize;
    let mut priorities = vec![f32::INFINITY; k];
    let mut indices = vec![-1i32; k];
    let mut flags = vec![0u8; k];

    let pushes = fixture["inputs"]["pushes"].as_array().unwrap();
    let expected_returns = json_i32_vec(&fixture["expected"]["push_returns"]);
    assert_eq!(pushes.len(), expected_returns.len());

    for (i, (push, &exp_ret)) in pushes.iter().zip(expected_returns.iter()).enumerate() {
        let p = push["p"].as_f64().unwrap() as f32;
        let n = push["n"].as_i64().unwrap() as i32;
        let ret = flagged_heap_push(&mut priorities, &mut indices, &mut flags, p, n);
        assert_eq!(ret, exp_ret, "push_returns[{i}]");
    }

    let expected_priorities = json_f32_vec(&fixture["expected"]["heap_priorities"]);
    for (i, (&actual, &exp)) in priorities
        .iter()
        .zip(expected_priorities.iter())
        .enumerate()
    {
        common::assert_f32_exact(actual, exp, &format!("heap_priorities[{i}]"));
    }
    assert_eq!(
        indices,
        json_i32_vec(&fixture["expected"]["heap_indices"]),
        "heap_indices"
    );
    assert_eq!(
        flags,
        json_u8_vec(&fixture["expected"]["heap_flags"]),
        "heap_flags"
    );
}

#[test]
fn build_candidates_fixture() {
    let fixture = load("nndescent", "build_candidates");
    assert_eq!(fixture["tolerance"], "exact");

    let cg = &fixture["inputs"]["current_graph"];
    let (indices, n_neighbors) = json_i32_matrix(&cg["indices"]);
    let (priorities, pw) = json_f32_matrix(&cg["priorities"]);
    let flags_rows = cg["flags"].as_array().unwrap();
    let flags: Vec<u8> = flags_rows
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter())
        .map(|x| x.as_u64().unwrap() as u8)
        .collect();
    assert_eq!(pw, n_neighbors);
    let n_vertices = indices.len() / n_neighbors;

    let mut current_graph = NNDHeap {
        n_points: n_vertices,
        size: n_neighbors,
        indices,
        distances: priorities,
        flags,
    };

    let max_candidates = fixture["inputs"]["max_candidates"].as_u64().unwrap() as usize;
    let rng_state = json_rng_state(&fixture["inputs"]["rng_state"]);
    let n_threads = fixture["inputs"]["n_threads"].as_u64().unwrap() as usize;

    let (new_c, old_c) =
        build_candidates(&mut current_graph, max_candidates, &rng_state, n_threads);

    let (expected_new, exp_mc) = json_i32_matrix(&fixture["expected"]["new_candidate_indices"]);
    assert_eq!(exp_mc, max_candidates);
    assert_eq!(new_c, expected_new, "new_candidate_indices");

    let (expected_old, _) = json_i32_matrix(&fixture["expected"]["old_candidate_indices"]);
    assert_eq!(old_c, expected_old, "old_candidate_indices");

    let expected_flags: Vec<u8> = fixture["expected"]["graph_flags_after"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter())
        .map(|x| x.as_u64().unwrap() as u8)
        .collect();
    assert_eq!(current_graph.flags, expected_flags, "graph_flags_after");

    // build_candidates draws only from per-thread copies of the rng state;
    // the caller's state must be untouched (fixture asserts this too).
    let rng_after = json_rng_state(&fixture["expected"]["rng_state_after"]);
    assert_eq!(rng_state, rng_after, "rng_state_after");
}

#[test]
fn make_float_forest_fixture() {
    let fixture = load("nndescent", "make_float_forest_n64_d6_leaf10");
    assert_eq!(fixture["tolerance"], "exact");

    let (data, n_features) = json_f32_matrix(&fixture["inputs"]["data"]);
    let mut rng_states: Vec<RngState> = fixture["inputs"]["rng_states"]
        .as_array()
        .unwrap()
        .iter()
        .map(json_rng_state)
        .collect();
    let leaf_size = fixture["inputs"]["leaf_size"].as_u64().unwrap() as usize;
    let max_depth = fixture["inputs"]["max_depth"].as_i64().unwrap();

    let forest = make_float_forest(
        &data,
        n_features,
        &mut rng_states,
        leaf_size,
        max_depth,
        true,
    );

    let expected_arrays = fixture["expected"]["leaf_arrays"].as_array().unwrap();
    assert_eq!(forest.len(), expected_arrays.len(), "n_trees");

    for (t, ((leaf_flat, width), expected)) in forest.iter().zip(expected_arrays.iter()).enumerate()
    {
        let (expected_flat, expected_width) = json_i32_matrix(expected);
        assert_eq!(*width, expected_width, "tree {t}: leaf width");
        assert_eq!(*leaf_flat, expected_flat, "tree {t}: leaf_array");
    }
}

fn run_nn_descent_case(case: &str, sorted: bool) {
    let fixture = load("nndescent", case);

    let (data, n_features) = json_f32_matrix(&fixture["inputs"]["data"]);
    let n_neighbors = fixture["inputs"]["n_neighbors"].as_u64().unwrap() as usize;
    let rng_state = json_rng_state(&fixture["inputs"]["rng_state"]);
    let max_candidates = fixture["inputs"]["max_candidates"].as_u64().unwrap() as usize;
    let n_iters = fixture["inputs"]["n_iters"].as_u64().unwrap() as usize;
    let delta = fixture["inputs"]["delta"].as_f64().unwrap();
    let delta_improv = fixture["inputs"]["delta_improv"].as_f64();
    let (leaf_array, leaf_width) = json_i32_matrix(&fixture["inputs"]["leaf_array"]);

    let (indices, distances) = if sorted {
        nn_descent_float_sorted(
            &data,
            n_features,
            n_neighbors,
            &rng_state,
            max_candidates,
            n_iters,
            delta,
            delta_improv,
            &leaf_array,
            leaf_width,
            1,
        )
    } else {
        nn_descent_float(
            &data,
            n_features,
            n_neighbors,
            &rng_state,
            max_candidates,
            n_iters,
            delta,
            delta_improv,
            &leaf_array,
            leaf_width,
            1,
        )
    };

    let (expected_indices, exp_k) = json_i32_matrix(&fixture["expected"]["indices"]);
    assert_eq!(exp_k, n_neighbors);
    assert_eq!(indices, expected_indices, "{case}: indices");

    let dist_tol = Tolerance::parse_for_field(&fixture["tolerance"], "distances");
    let (expected_distances, _) = json_f32_matrix(&fixture["expected"]["distances"]);
    assert_eq!(distances.len(), expected_distances.len());
    for (i, (&actual, &exp)) in distances.iter().zip(expected_distances.iter()).enumerate() {
        dist_tol.assert_close(
            actual,
            exp,
            &format!(
                "{case}: distances[{}][{}]",
                i / n_neighbors,
                i % n_neighbors
            ),
        );
    }
}

#[test]
fn nn_descent_float_n150_d8_k10() {
    run_nn_descent_case("nn_descent_float_n150_d8_k10", false);
}

#[test]
fn nn_descent_float_sorted_n150_d8_k10() {
    run_nn_descent_case("nn_descent_float_sorted_n150_d8_k10", true);
}
