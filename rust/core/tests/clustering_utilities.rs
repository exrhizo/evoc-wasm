//! Fixture parity tests for `evoc_core::clustering_utilities`.

mod common;

use common::{load, Tolerance};
use evoc_core::cluster_trees::CondensedTree;
use evoc_core::cluster_trees::LinkageTree;
use evoc_core::clustering_utilities::{
    _binary_search_for_n_clusters, build_cluster_tree, compute_total_persistence,
    estimate_cluster_similarity, extract_clusters_by_id, find_duplicates, find_peaks,
    jaccard_similarity, min_cluster_size_barcode, select_diverse_peaks,
};

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

fn condensed_from(v: &serde_json::Value) -> CondensedTree {
    CondensedTree {
        parent: json_i64_vec(&v["parent"]),
        child: json_i64_vec(&v["child"]),
        lambda_val: json_f32_vec(&v["lambda_val"]),
        child_size: json_i64_vec(&v["child_size"]),
    }
}

#[test]
fn find_peaks_synthetic() {
    let fixture = load("clustering_utilities", "find_peaks_synthetic");
    let x = json_f32_vec(&fixture["inputs"]["x"]);
    let expected: Vec<usize> = json_i64_vec(&fixture["expected"]["peak_indices"])
        .into_iter()
        .map(|x| x as usize)
        .collect();
    assert_eq!(find_peaks(&x), expected);
}

#[test]
fn find_peaks_monotone_empty() {
    let fixture = load("clustering_utilities", "find_peaks_monotone_empty");
    let x = json_f32_vec(&fixture["inputs"]["x"]);
    assert_eq!(find_peaks(&x), Vec::<usize>::new());
}

#[test]
fn jaccard_similarity_cases() {
    let fixture = load("clustering_utilities", "jaccard_similarity");
    let cases = fixture["inputs"]["cases"].as_array().unwrap();
    let expected = fixture["expected"]["values"].as_array().unwrap();
    for (case, exp) in cases.iter().zip(expected.iter()) {
        let a = json_i64_vec(&case["a"]);
        let b = json_i64_vec(&case["b"]);
        let got = jaccard_similarity(&a, &b);
        assert_eq!(got, exp.as_f64().unwrap(), "case {:?}", case);
    }
}

#[test]
fn estimate_cluster_similarity_cases() {
    let fixture = load("clustering_utilities", "estimate_cluster_similarity_seed45");
    let births = json_f32_vec(&fixture["inputs"]["births"]);
    let deaths = json_f32_vec(&fixture["inputs"]["deaths"]);
    let cases = fixture["inputs"]["cases"].as_array().unwrap();
    let expected = fixture["expected"]["values"].as_array().unwrap();
    for (case, exp) in cases.iter().zip(expected.iter()) {
        let birth_a = common::as_f32(&case["birth_a"]);
        let birth_b = common::as_f32(&case["birth_b"]);
        let got = estimate_cluster_similarity(&births, &deaths, birth_a, birth_b);
        assert_eq!(got, exp.as_f64().unwrap());
    }
}

#[test]
fn min_cluster_size_barcode_seed45() {
    let fixture = load("clustering_utilities", "min_cluster_size_barcode_seed45");
    let lambda_tol = Tolerance::parse(&fixture["tolerance"]);

    let ct = condensed_from(&fixture["inputs"]["cluster_tree"]);
    let n_points = fixture["inputs"]["n_points"].as_i64().unwrap();
    let min_size = fixture["inputs"]["min_size"].as_f64().unwrap() as f32;

    let (size_births, size_deaths, parents, lambda_deaths) =
        min_cluster_size_barcode(&ct, n_points, min_size);

    let exp_births = json_f32_vec(&fixture["expected"]["size_births"]);
    let exp_deaths = json_f32_vec(&fixture["expected"]["size_deaths"]);
    let exp_parents = json_i64_vec(&fixture["expected"]["parents"]);
    let exp_lambda = json_f32_vec(&fixture["expected"]["lambda_deaths"]);

    assert_eq!(parents, exp_parents, "parents");
    for (i, (&a, &e)) in size_births.iter().zip(exp_births.iter()).enumerate() {
        lambda_tol.assert_close(a, e, &format!("size_births[{i}]"));
    }
    for (i, (&a, &e)) in size_deaths.iter().zip(exp_deaths.iter()).enumerate() {
        lambda_tol.assert_close(a, e, &format!("size_deaths[{i}]"));
    }
    for (i, (&a, &e)) in lambda_deaths.iter().zip(exp_lambda.iter()).enumerate() {
        lambda_tol.assert_close(a, e, &format!("lambda_deaths[{i}]"));
    }
}

#[test]
fn compute_total_persistence_seed45() {
    let fixture = load("clustering_utilities", "compute_total_persistence_seed45");
    let tol = Tolerance::parse(&fixture["tolerance"]);

    let births = json_f32_vec(&fixture["inputs"]["births"]);
    let deaths = json_f32_vec(&fixture["inputs"]["deaths"]);
    let lambda_deaths = json_f32_vec(&fixture["inputs"]["lambda_deaths"]);

    let (sizes, total_persistence) = compute_total_persistence(&births, &deaths, &lambda_deaths);

    let exp_sizes = json_f32_vec(&fixture["expected"]["sizes"]);
    let exp_tp = json_f32_vec(&fixture["expected"]["total_persistence"]);

    assert_eq!(sizes, exp_sizes, "sizes");
    assert_eq!(total_persistence.len(), exp_tp.len());
    for (i, (&a, &e)) in total_persistence.iter().zip(exp_tp.iter()).enumerate() {
        tol.assert_close(a, e, &format!("total_persistence[{i}]"));
    }
}

#[test]
fn select_diverse_peaks_seed45() {
    let fixture = load("clustering_utilities", "select_diverse_peaks_seed45");
    let peaks = json_i64_vec(&fixture["inputs"]["peaks"]);
    let total_persistence = json_f32_vec(&fixture["inputs"]["total_persistence"]);
    let sizes = json_f32_vec(&fixture["inputs"]["sizes"]);
    let births = json_f32_vec(&fixture["inputs"]["births"]);
    let deaths = json_f32_vec(&fixture["inputs"]["deaths"]);
    let min_similarity_threshold = fixture["inputs"]["min_similarity_threshold"]
        .as_f64()
        .unwrap();
    let max_layers = fixture["inputs"]["max_layers"].as_i64().unwrap() as usize;

    let selected = select_diverse_peaks(
        &peaks,
        &total_persistence,
        &sizes,
        &births,
        &deaths,
        min_similarity_threshold,
        max_layers,
    );

    let expected = json_i64_vec(&fixture["expected"]["selected_peak_indices"]);
    assert_eq!(selected, expected);
}

#[test]
fn extract_clusters_by_id_seed45_peak1() {
    let fixture = load(
        "clustering_utilities",
        "extract_clusters_by_id_seed45_peak1",
    );
    let strengths_tol = Tolerance::parse_for_field(&fixture["tolerance"], "strengths");

    let ct = condensed_from(&fixture["inputs"]["condensed_tree"]);
    let selected_ids = json_i64_vec(&fixture["inputs"]["selected_ids"]);

    let (labels, strengths) = extract_clusters_by_id(&ct, &selected_ids);

    let exp_labels = json_i64_vec(&fixture["expected"]["labels"]);
    let exp_strengths = json_f32_vec(&fixture["expected"]["strengths"]);

    assert_eq!(labels, exp_labels, "labels");
    for (i, (&a, &e)) in strengths.iter().zip(exp_strengths.iter()).enumerate() {
        strengths_tol.assert_close(a, e, &format!("strengths[{i}]"));
    }
}

#[test]
fn binary_search_for_n_clusters_n60() {
    let fixture = load("clustering_utilities", "binary_search_for_n_clusters_n60");
    let strengths_tol = Tolerance::parse_for_field(&fixture["tolerance"], "strengths");

    let left = json_f32_vec(&fixture["inputs"]["uncondensed_tree"]["left"]);
    let right = json_f32_vec(&fixture["inputs"]["uncondensed_tree"]["right"]);
    let delta: Vec<f64> = fixture["inputs"]["uncondensed_tree"]["delta"]
        .as_array()
        .unwrap()
        .iter()
        .map(common::num_f64)
        .collect();
    let size = json_f32_vec(&fixture["inputs"]["uncondensed_tree"]["size"]);
    let rows = (0..left.len())
        .map(|i| [left[i] as f64, right[i] as f64, delta[i], size[i] as f64])
        .collect();
    let uncondensed_tree = LinkageTree { rows };

    let approx_n_clusters = fixture["inputs"]["approx_n_clusters"].as_i64().unwrap();
    let n_samples = fixture["inputs"]["data"].as_array().unwrap().len();

    let (leaves, clusters, strengths) =
        _binary_search_for_n_clusters(&uncondensed_tree, approx_n_clusters, n_samples);

    let exp_leaves = json_i64_vec(&fixture["expected"]["leaves"]);
    let exp_labels = json_i64_vec(&fixture["expected"]["labels"]);
    let exp_strengths = json_f32_vec(&fixture["expected"]["strengths"]);

    assert_eq!(leaves, exp_leaves, "leaves");
    assert_eq!(clusters, exp_labels, "labels");
    for (i, (&a, &e)) in strengths.iter().zip(exp_strengths.iter()).enumerate() {
        strengths_tol.assert_close(a, e, &format!("strengths[{i}]"));
    }
}

#[test]
fn find_duplicates_synthetic() {
    let fixture = load("clustering_utilities", "find_duplicates_synthetic");
    let inds = fixture["inputs"]["knn_indices"].as_array().unwrap();
    let dists = fixture["inputs"]["knn_dists"].as_array().unwrap();
    let n = inds.len();
    let k = inds[0].as_array().unwrap().len();
    let mut knn_inds = Vec::with_capacity(n * k);
    let mut knn_dists = Vec::with_capacity(n * k);
    for row in inds {
        for x in row.as_array().unwrap() {
            knn_inds.push(x.as_i64().unwrap());
        }
    }
    for row in dists {
        for x in row.as_array().unwrap() {
            knn_dists.push(common::as_f32(x));
        }
    }

    let got = find_duplicates(&knn_inds, &knn_dists, n, k);
    let expected: Vec<(i64, i64)> = fixture["expected"]["duplicate_pairs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let p = p.as_array().unwrap();
            (p[0].as_i64().unwrap(), p[1].as_i64().unwrap())
        })
        .collect();

    let mut got_sorted: Vec<(i64, i64)> = got.into_iter().collect();
    got_sorted.sort_unstable();
    let mut expected_sorted = expected;
    expected_sorted.sort_unstable();
    assert_eq!(got_sorted, expected_sorted);
}

#[test]
fn find_duplicates_nndescent_n150_empty() {
    let fixture = load("clustering_utilities", "find_duplicates_nndescent_n150");
    let inds = fixture["inputs"]["knn_indices"].as_array().unwrap();
    let dists = fixture["inputs"]["knn_dists"].as_array().unwrap();
    let n = inds.len();
    let k = inds[0].as_array().unwrap().len();
    let mut knn_inds = Vec::with_capacity(n * k);
    let mut knn_dists = Vec::with_capacity(n * k);
    for row in inds {
        for x in row.as_array().unwrap() {
            knn_inds.push(x.as_i64().unwrap());
        }
    }
    for row in dists {
        for x in row.as_array().unwrap() {
            knn_dists.push(common::as_f32(x));
        }
    }
    let got = find_duplicates(&knn_inds, &knn_dists, n, k);
    let expected = fixture["expected"]["duplicate_pairs"].as_array().unwrap();
    assert_eq!(got.len(), expected.len());
}

#[test]
fn build_cluster_tree_3layers() {
    let fixture = load("clustering_utilities", "build_cluster_tree_3layers");
    let labels: Vec<Vec<i64>> = fixture["inputs"]["labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(json_i64_vec)
        .collect();

    let tree = build_cluster_tree(&labels);

    // Canonicalize as the fixture does: sorted list of
    // {parent_layer, parent_cluster, children: [sorted pairs]}.
    let mut got: Vec<(usize, i64, Vec<(usize, i64)>)> = tree
        .into_iter()
        .map(|((pl, pc), mut children)| {
            children.sort_unstable();
            (pl, pc, children)
        })
        .collect();
    got.sort_unstable();

    let mut expected: Vec<(usize, i64, Vec<(usize, i64)>)> = fixture["expected"]["tree"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| {
            let pl = entry["parent_layer"].as_u64().unwrap() as usize;
            let pc = entry["parent_cluster"].as_i64().unwrap();
            let mut children: Vec<(usize, i64)> = entry["children"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| {
                    let c = c.as_array().unwrap();
                    (c[0].as_u64().unwrap() as usize, c[1].as_i64().unwrap())
                })
                .collect();
            children.sort_unstable();
            (pl, pc, children)
        })
        .collect();
    expected.sort_unstable();

    assert_eq!(got, expected);
}
