//! Fixture parity tests for `evoc_core::node_embedding`.

mod common;

use common::{as_f32, load, Tolerance};
use evoc_core::node_embedding::{
    clip, make_epochs_per_sample, node_embedding_epoch, node_embedding_epoch_repr,
    node_embedding_repr, rdist,
};

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

#[test]
fn clip_cases() {
    let fixture = load("node_embedding", "clip");
    let tol = Tolerance::parse(&fixture["tolerance"]);
    let cases = fixture["inputs"]["cases"].as_array().unwrap();
    let expected = json_f32_vec(&fixture["expected"]["values"]);
    for (i, case) in cases.iter().enumerate() {
        let val = as_f32(&case["val"]);
        let lo = as_f32(&case["lo"]);
        let hi = as_f32(&case["hi"]);
        let got = clip(val, lo, hi);
        tol.assert_close(got, expected[i], &format!("clip case {i}"));
    }
}

#[test]
fn rdist_cases() {
    let fixture = load("node_embedding", "rdist");
    let tol = Tolerance::parse(&fixture["tolerance"]);
    let pairs = fixture["inputs"]["pairs"].as_array().unwrap();
    let expected = json_f32_vec(&fixture["expected"]["values"]);
    for (i, pair) in pairs.iter().enumerate() {
        let x = json_f32_vec(&pair["x"]);
        let y = json_f32_vec(&pair["y"]);
        let got = rdist(&x, &y);
        tol.assert_close(got, expected[i], &format!("rdist case {i}"));
    }
}

#[test]
fn make_epochs_per_sample_n_epochs10() {
    let fixture = load("node_embedding", "make_epochs_per_sample_n_epochs10");
    let tol = Tolerance::parse(&fixture["tolerance"]);
    let weights = json_f32_vec(&fixture["inputs"]["weights"]);
    let n_epochs = fixture["inputs"]["n_epochs"].as_u64().unwrap() as usize;
    let expected = json_f32_vec(&fixture["expected"]["epochs_per_sample"]);
    let got = make_epochs_per_sample(&weights, n_epochs);
    assert_eq!(got.len(), expected.len());
    for (i, (&a, &e)) in got.iter().zip(expected.iter()).enumerate() {
        tol.assert_close(a, e, &format!("eps[{i}]"));
    }
}

#[test]
fn make_epochs_per_sample_n_epochs50() {
    let fixture = load("node_embedding", "make_epochs_per_sample_n_epochs50");
    let tol = Tolerance::parse(&fixture["tolerance"]);
    let weights = json_f32_vec(&fixture["inputs"]["weights"]);
    let n_epochs = fixture["inputs"]["n_epochs"].as_u64().unwrap() as usize;
    let expected = json_f32_vec(&fixture["expected"]["epochs_per_sample"]);
    let got = make_epochs_per_sample(&weights, n_epochs);
    assert_eq!(got.len(), expected.len());
    for (i, (&a, &e)) in got.iter().zip(expected.iter()).enumerate() {
        tol.assert_close(a, e, &format!("eps[{i}]"));
    }
}

#[test]
fn node_embedding_epoch_single_epoch_n120() {
    let fixture = load("node_embedding", "node_embedding_epoch_single_epoch_n120");
    let tol = Tolerance::parse(&fixture["tolerance"]);
    let inputs = &fixture["inputs"];

    let embedding_before = flatten(&json_f32_mat(&inputs["embedding_before"]));
    let head = json_u32_vec(&inputs["head"]);
    let tail = json_u32_vec(&inputs["tail"]);
    let n_vertices = inputs["n_vertices"].as_u64().unwrap() as u32;
    let epochs_per_sample = json_f32_vec(&inputs["epochs_per_sample"]);
    let rng_state = inputs["rng_state"].as_u64().unwrap() as u32;
    let dim = inputs["dim"].as_u64().unwrap() as usize;
    let alpha = as_f32(&inputs["alpha"]);
    let epochs_per_negative_sample = json_f32_vec(&inputs["epochs_per_negative_sample"]);
    // NOTE: the fixture's `epoch_of_next_sample` / `epoch_of_next_negative_sample`
    // input fields actually capture the POST-call state (the arrays are mutated
    // in place by the kernel and the fixture generator serialized them after the
    // call), not the true pre-call gate values. Per the fixture note and PORT_NOTES,
    // the true pre-call state for this isolation fixture is all-zero (forcing
    // every sample to fire on this single call) -- verified by replaying the
    // reference algorithm in Python: starting from zero reproduces
    // `embedding_after` exactly, while starting from the given (post-state)
    // values reproduces the *unmodified* `embedding_before` instead.
    let mut epoch_of_next_negative_sample = vec![0.0f32; epochs_per_negative_sample.len()];
    let mut epoch_of_next_sample = vec![0.0f32; epochs_per_sample.len()];
    let n = inputs["n"].as_u64().unwrap() as u32;
    let noise_level = as_f32(&inputs["noise_level"]);

    let mut embedding = embedding_before;
    node_embedding_epoch(
        &mut embedding,
        &head,
        &tail,
        n_vertices,
        &epochs_per_sample,
        rng_state,
        dim,
        alpha,
        &epochs_per_negative_sample,
        &mut epoch_of_next_negative_sample,
        &mut epoch_of_next_sample,
        n,
        noise_level,
    );

    let expected = flatten(&json_f32_mat(&fixture["expected"]["embedding_after"]));
    assert_eq!(embedding.len(), expected.len());
    for (i, (&a, &e)) in embedding.iter().zip(expected.iter()).enumerate() {
        tol.assert_close(a, e, &format!("embedding[{i}]"));
    }
}

#[test]
fn node_embedding_epoch_repr_single_epoch_n120() {
    let fixture = load(
        "node_embedding",
        "node_embedding_epoch_repr_single_epoch_n120",
    );
    let tol = Tolerance::parse(&fixture["tolerance"]);
    let inputs = &fixture["inputs"];

    let embedding_before = flatten(&json_f32_mat(&inputs["embedding_before"]));
    let csr_indptr = json_u32_vec(&inputs["csr_indptr"]);
    let csr_indices = json_u32_vec(&inputs["csr_indices"]);
    let n_vertices = inputs["n_vertices"].as_u64().unwrap() as u32;
    let epochs_per_sample = json_f32_vec(&inputs["epochs_per_sample"]);
    let rng_state = inputs["rng_state"].as_u64().unwrap() as u32;
    let dim = inputs["dim"].as_u64().unwrap() as usize;
    let alpha = as_f32(&inputs["alpha"]);
    let epochs_per_negative_sample = json_f32_vec(&inputs["epochs_per_negative_sample"]);
    // See the analogous comment in `node_embedding_epoch_single_epoch_n120`: the
    // fixture's stored `epoch_of_next_sample`/`epoch_of_next_negative_sample`
    // input arrays are the POST-call state, not the true pre-call gate (which
    // is all-zero per the fixture note, forcing every edge to fire).
    let mut epoch_of_next_negative_sample = vec![0.0f32; epochs_per_negative_sample.len()];
    let mut epoch_of_next_sample = vec![0.0f32; epochs_per_sample.len()];
    let n = inputs["n"].as_u64().unwrap() as u32;
    let noise_level = as_f32(&inputs["noise_level"]);
    let gamma = as_f32(&inputs["gamma"]);
    let node_order = json_u32_vec(&inputs["node_order"]);
    let block_size = inputs["block_size"].as_u64().unwrap() as usize;

    let mut embedding = embedding_before;
    let mut updates = vec![0.0f32; n_vertices as usize * dim];

    node_embedding_epoch_repr(
        &mut embedding,
        &csr_indptr,
        &csr_indices,
        n_vertices,
        &epochs_per_sample,
        rng_state,
        dim,
        alpha,
        &epochs_per_negative_sample,
        &mut epoch_of_next_negative_sample,
        &mut epoch_of_next_sample,
        n,
        noise_level,
        gamma,
        &mut updates,
        &node_order,
        block_size,
    );

    let expected = flatten(&json_f32_mat(&fixture["expected"]["embedding_after"]));
    assert_eq!(embedding.len(), expected.len());
    for (i, (&a, &e)) in embedding.iter().zip(expected.iter()).enumerate() {
        tol.assert_close(a, e, &format!("embedding[{i}]"));
    }

    let expected_eons = json_f32_vec(&fixture["expected"]["epoch_of_next_sample_after"]);
    for (i, (&a, &e)) in epoch_of_next_sample
        .iter()
        .zip(expected_eons.iter())
        .enumerate()
    {
        tol.assert_close(a, e, &format!("epoch_of_next_sample[{i}]"));
    }

    let expected_eonns = json_f32_vec(&fixture["expected"]["epoch_of_next_negative_sample_after"]);
    for (i, (&a, &e)) in epoch_of_next_negative_sample
        .iter()
        .zip(expected_eonns.iter())
        .enumerate()
    {
        tol.assert_close(a, e, &format!("epoch_of_next_negative_sample[{i}]"));
    }

    let expected_updates = flatten(&json_f32_mat(&fixture["expected"]["updates_after"]));
    for (i, (&a, &e)) in updates.iter().zip(expected_updates.iter()).enumerate() {
        tol.assert_close(a, e, &format!("updates[{i}]"));
    }
}

#[test]
fn node_embedding_e2e_n120() {
    let fixture = load("node_embedding", "node_embedding_e2e_n120");
    let tol = Tolerance::parse(&fixture["tolerance"]);
    let inputs = &fixture["inputs"];

    let graph_indptr = json_u32_vec(&inputs["indptr"]);
    let graph_indices = json_u32_vec(&inputs["indices"]);
    let graph_data = json_f32_vec(&inputs["data"]);
    let n_vertices = inputs["n_vertices"].as_u64().unwrap() as usize;
    let n_components = inputs["n_components"].as_u64().unwrap() as usize;
    let n_epochs = inputs["n_epochs"].as_u64().unwrap() as usize;
    let initial_embedding = flatten(&json_f32_mat(&inputs["initial_embedding"]));
    let initial_alpha = as_f32(&inputs["initial_alpha"]);
    let negative_sample_rate = as_f32(&inputs["negative_sample_rate"]);
    let noise_level = as_f32(&inputs["noise_level"]);
    let rng_val: Vec<u32> = inputs["rng_val"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    let node_order_shuffles: Vec<Vec<u32>> = inputs["node_order_shuffles"]
        .as_array()
        .unwrap()
        .iter()
        .map(json_u32_vec)
        .collect();

    assert!(inputs["reproducible_flag"].as_bool().unwrap());

    let got = node_embedding_repr(
        &graph_indptr,
        &graph_indices,
        &graph_data,
        n_vertices,
        n_components,
        n_epochs,
        &initial_embedding,
        initial_alpha,
        negative_sample_rate,
        noise_level,
        &rng_val,
        &node_order_shuffles,
    );

    let expected = flatten(&json_f32_mat(&fixture["expected"]["final_embedding"]));
    assert_eq!(got.len(), expected.len());
    for (i, (&a, &e)) in got.iter().zip(expected.iter()).enumerate() {
        tol.assert_close(a, e, &format!("final_embedding[{i}]"));
    }
}
