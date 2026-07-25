//! NOT a validation test -- a one-shot harness invoked manually via
//! `cargo test --test dump_evoc_clusters_labels_for_cross_lang -- --ignored`
//! to dump Rust's `EVoC::fit_predict` labels for seeds 0..19 on the
//! `evoc_clusters_e2e_ari_check` fixture, as JSON, for the cross-language
//! (Rust vs Zig) agreement check described in
//! `rust/core/PCA_VALIDATION.md`. The companion Zig side is
//! `zig/src/dump_evoc_clusters_labels_for_cross_lang.zig`, and
//! `scripts/compare_rust_zig_evoc_clusters.py` diffs the two dumps and
//! computes ARI. Output path defaults to
//! `/tmp/uvoc_cross_lang/rust_labels.json` but can be overridden with
//! `UVOC_CROSS_LANG_OUT`.

mod common;

use common::as_f32;
use std::io::Write;
use uvoc_core::clustering::{EVoC, EvocClustersOptions};

fn json_f32_mat(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter().map(as_f32))
        .collect()
}

#[test]
#[ignore]
fn dump_rust_evoc_clusters_labels_for_seed_panel() {
    let fixture = common::load("clustering", "evoc_clusters_e2e_ari_check");
    let n_samples = fixture["n_samples"].as_u64().unwrap() as usize;
    let n_features = fixture["n_features"].as_u64().unwrap() as usize;
    let data = json_f32_mat(&fixture["data"]);
    assert_eq!(data.len(), n_samples * n_features);

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

    let mut out = String::from("{\n  \"labels\": {\n");
    let mut seed0_nn_inds: Vec<i32> = Vec::new();
    let mut seed0_nn_dists: Vec<f32> = Vec::new();
    for seed in 0i64..20 {
        let mut evoc = EVoC::new(seed, opts());
        let labels = evoc.fit_predict(&data, n_features).to_vec();
        if seed == 0 {
            seed0_nn_inds = evoc.nn_inds.clone();
            seed0_nn_dists = evoc.nn_dists.clone();
        }
        let labels_str: Vec<String> = labels.iter().map(|l| l.to_string()).collect();
        out.push_str(&format!(
            "    \"{}\": [{}]{}\n",
            seed,
            labels_str.join(","),
            if seed < 19 { "," } else { "" }
        ));
    }
    out.push_str("  },\n");
    // First-stage cross-language check for seed 0: knn_graph output (before
    // label propagation / PCA / node embedding even run).
    let inds_str: Vec<String> = seed0_nn_inds.iter().map(|v| v.to_string()).collect();
    let dists_str: Vec<String> = seed0_nn_dists
        .iter()
        .map(|v| format!("{}", v.to_bits()))
        .collect();
    out.push_str(&format!("  \"seed0_nn_inds\": [{}],\n", inds_str.join(",")));
    out.push_str(&format!(
        "  \"seed0_nn_dists_bits\": [{}]\n",
        dists_str.join(",")
    ));
    out.push_str("}\n");

    let out_dir = std::env::var("UVOC_CROSS_LANG_OUT_DIR")
        .unwrap_or_else(|_| "/tmp/uvoc_cross_lang".to_string());
    std::fs::create_dir_all(&out_dir).unwrap();
    let out_path = format!("{out_dir}/rust_labels.json");
    let mut f = std::fs::File::create(&out_path).unwrap();
    f.write_all(out.as_bytes()).unwrap();
    eprintln!("wrote {out_path}");
}
