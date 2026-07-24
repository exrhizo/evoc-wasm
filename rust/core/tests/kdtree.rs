//! Fixture parity tests for `uvoc_core::kdtree`.

mod common;

use common::{load, Tolerance};
use uvoc_core::kdtree::{
    build_kdtree, deheap_sort, parallel_tree_query, point_to_node_lower_bound_rdist, rdist,
    simple_heap_push, NumbaKDTree,
};

fn json_f32_vec(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn json_usize_vec(v: &serde_json::Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as usize)
        .collect()
}

/// Flatten a JSON list-of-rows into (flat f32 data, n_features).
fn json_matrix(v: &serde_json::Value) -> (Vec<f32>, usize) {
    let rows = v.as_array().unwrap();
    let n_features = rows[0].as_array().unwrap().len();
    let mut flat = Vec::with_capacity(rows.len() * n_features);
    for row in rows {
        let row = row.as_array().unwrap();
        assert_eq!(row.len(), n_features);
        for x in row {
            flat.push(x.as_f64().unwrap() as f32);
        }
    }
    (flat, n_features)
}

fn build_from_fixture(fixture: &serde_json::Value) -> NumbaKDTree {
    let (data, n_features) = json_matrix(&fixture["inputs"]["data"]);
    let leaf_size = fixture["inputs"]["leaf_size"].as_u64().unwrap() as usize;
    build_kdtree(&data, n_features, leaf_size)
}

fn run_build_case(case: &str) {
    let fixture = load("kdtree", case);
    let tree = build_from_fixture(&fixture);
    let expected = &fixture["expected"];

    assert_eq!(
        tree.idx_array,
        json_usize_vec(&expected["idx_array"]),
        "{case}: idx_array"
    );
    assert_eq!(
        tree.idx_start,
        json_usize_vec(&expected["idx_start"]),
        "{case}: idx_start"
    );
    assert_eq!(
        tree.idx_end,
        json_usize_vec(&expected["idx_end"]),
        "{case}: idx_end"
    );

    let expected_is_leaf: Vec<bool> = expected["is_leaf"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect();
    assert_eq!(tree.is_leaf, expected_is_leaf, "{case}: is_leaf");

    let radius_tol = Tolerance::parse_for_field(&fixture["tolerance"], "radius");
    let expected_radius = json_f32_vec(&expected["radius"]);
    assert_eq!(tree.radius.len(), expected_radius.len());
    for (i, (&actual, &exp)) in tree.radius.iter().zip(expected_radius.iter()).enumerate() {
        radius_tol.assert_close(actual, exp, &format!("{case}: radius[{i}]"));
    }

    let bounds_lower_tol = Tolerance::parse_for_field(&fixture["tolerance"], "node_bounds_lower");
    let (expected_lower, nf_lower) = json_matrix(&expected["node_bounds_lower"]);
    assert_eq!(nf_lower, tree.n_features);
    assert_eq!(tree.node_bounds_lower.len(), expected_lower.len());
    for (i, (&actual, &exp)) in tree
        .node_bounds_lower
        .iter()
        .zip(expected_lower.iter())
        .enumerate()
    {
        bounds_lower_tol.assert_close(actual, exp, &format!("{case}: node_bounds_lower[{i}]"));
    }

    let bounds_upper_tol = Tolerance::parse_for_field(&fixture["tolerance"], "node_bounds_upper");
    let (expected_upper, nf_upper) = json_matrix(&expected["node_bounds_upper"]);
    assert_eq!(nf_upper, tree.n_features);
    assert_eq!(tree.node_bounds_upper.len(), expected_upper.len());
    for (i, (&actual, &exp)) in tree
        .node_bounds_upper
        .iter()
        .zip(expected_upper.iter())
        .enumerate()
    {
        bounds_upper_tol.assert_close(actual, exp, &format!("{case}: node_bounds_upper[{i}]"));
    }
}

#[test]
fn build_n100_d4_leaf10() {
    run_build_case("build_n100_d4_leaf10");
}

#[test]
fn build_n37_d3_leaf5() {
    run_build_case("build_n37_d3_leaf5");
}

#[test]
fn build_n20_d2_leaf40_single_node() {
    run_build_case("build_n20_d2_leaf40_single_node");
}

#[test]
fn rdist_pairs() {
    let fixture = load("kdtree", "rdist");
    let tolerance = Tolerance::parse(&fixture["tolerance"]);
    let pairs = fixture["inputs"]["pairs"].as_array().unwrap();
    let expected = json_f32_vec(&fixture["expected"]["values"]);
    assert_eq!(pairs.len(), expected.len());

    for (i, (pair, &exp)) in pairs.iter().zip(expected.iter()).enumerate() {
        let x = json_f32_vec(&pair["x"]);
        let y = json_f32_vec(&pair["y"]);
        tolerance.assert_close(rdist(&x, &y), exp, &format!("rdist pair {i}"));
    }
}

#[test]
fn point_to_node_lower_bound_rdist_cases() {
    let fixture = load("kdtree", "point_to_node_lower_bound_rdist");
    let tolerance = Tolerance::parse(&fixture["tolerance"]);
    let cases = fixture["inputs"]["cases"].as_array().unwrap();
    let expected = json_f32_vec(&fixture["expected"]["values"]);
    assert_eq!(cases.len(), expected.len());

    for (i, (case, &exp)) in cases.iter().zip(expected.iter()).enumerate() {
        let bounds_lower = json_f32_vec(&case["bounds_lower"]);
        let bounds_upper = json_f32_vec(&case["bounds_upper"]);
        let pt = json_f32_vec(&case["pt"]);
        // Reference call convention (see inputs.note and the PORT NOTE on
        // the function): lower bounds go in the parameter named `upper`.
        let actual = point_to_node_lower_bound_rdist(&bounds_lower, &bounds_upper, &pt);
        tolerance.assert_close(actual, exp, &format!("point_to_node_lower_bound case {i}"));
    }
}

#[test]
fn heap_push_deheap_sort() {
    let fixture = load("kdtree", "heap_push_deheap_sort");
    assert_eq!(fixture["tolerance"], "exact");

    let k = fixture["inputs"]["k"].as_u64().unwrap() as usize;
    let mut priorities = vec![f32::INFINITY; k];
    let mut indices = vec![-1i32; k];

    let pushes = fixture["inputs"]["pushes"].as_array().unwrap();
    let expected_returns: Vec<i32> = fixture["expected"]["push_returns"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    assert_eq!(pushes.len(), expected_returns.len());

    for (i, (push, &exp_ret)) in pushes.iter().zip(expected_returns.iter()).enumerate() {
        let p = push["p"].as_f64().unwrap() as f32;
        let n = push["n"].as_i64().unwrap() as i32;
        let ret = simple_heap_push(&mut priorities, &mut indices, p, n);
        assert_eq!(ret, exp_ret, "push_returns[{i}]");
    }

    let expected_heap_p = json_f32_vec(&fixture["expected"]["heap_priorities"]);
    for (i, (&actual, &exp)) in priorities.iter().zip(expected_heap_p.iter()).enumerate() {
        common::assert_f32_exact(actual, exp, &format!("heap_priorities[{i}]"));
    }
    let expected_heap_i: Vec<i32> = fixture["expected"]["heap_indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    assert_eq!(indices, expected_heap_i, "heap_indices");

    // deheap_sort the heap as a 1-row matrix
    deheap_sort(&mut priorities, &mut indices, k);

    let expected_sorted_p = json_f32_vec(&fixture["expected"]["sorted_priorities"]);
    for (i, (&actual, &exp)) in priorities.iter().zip(expected_sorted_p.iter()).enumerate() {
        common::assert_f32_exact(actual, exp, &format!("sorted_priorities[{i}]"));
    }
    let expected_sorted_i: Vec<i32> = fixture["expected"]["sorted_indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    assert_eq!(indices, expected_sorted_i, "sorted_indices");
}

fn run_query_case(case: &str) {
    let fixture = load("kdtree", case);
    let tree = build_from_fixture(&fixture);

    let (queries, nf) = json_matrix(&fixture["inputs"]["queries"]);
    assert_eq!(nf, tree.n_features);
    let k = fixture["inputs"]["k"].as_u64().unwrap() as usize;
    let output_rdist = fixture["inputs"]["output_rdist"].as_bool().unwrap();

    let (distances, indices) = parallel_tree_query(&tree, &queries, k, output_rdist);

    let (expected_distances, exp_k) = json_matrix(&fixture["expected"]["distances"]);
    assert_eq!(exp_k, k);
    let expected_indices: Vec<i32> = fixture["expected"]["indices"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter())
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();

    assert_eq!(indices, expected_indices, "{case}: indices");

    let dist_tol = Tolerance::parse_for_field(&fixture["tolerance"], "distances");
    assert_eq!(distances.len(), expected_distances.len());
    for (i, (&actual, &exp)) in distances.iter().zip(expected_distances.iter()).enumerate() {
        dist_tol.assert_close(
            actual,
            exp,
            &format!("{case}: distances[{}][{}]", i / k, i % k),
        );
    }
}

#[test]
fn parallel_tree_query_k5_rdist_true() {
    run_query_case("parallel_tree_query_k5_rdist_true");
}

#[test]
fn parallel_tree_query_k5_rdist_false() {
    run_query_case("parallel_tree_query_k5_rdist_false");
}
