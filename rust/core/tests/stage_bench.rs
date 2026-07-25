//! THROWAWAY: per-stage timing breakdown of `evoc_clusters`'s pipeline,
//! for comparison/results.md Part-1 hot-path analysis. Not wired into the
//! default test run. Duplicates `evoc_clusters`'s body (rust/core/src/
//! clustering.rs) inline so each stage can be wrapped in `Instant`; this
//! file makes NO changes to non-test source. Safe to delete once the
//! numbers are recorded.
//!   cargo test --release -p uvoc-core --test stage_bench -- --ignored --nocapture

use std::fs;
use std::time::Instant;
use uvoc_core::clustering::{build_cluster_layers, BuildClusterLayersOptions};
use uvoc_core::graph_construction::neighbor_graph_matrix;
use uvoc_core::knn_graph::knn_graph as knn_graph_impl;
use uvoc_core::label_propagation::label_propagation_init_recursive;
use uvoc_core::node_embedding::node_embedding_repr_self_driven;
use uvoc_core::rng::{seed as rng_seed, tau_rand_int, RngState};

fn load_scale_data(n: usize, d: usize) -> Vec<f32> {
    let path = format!("../../scripts/scale_data/scale_n{n}_d{d}.f32");
    let bytes = fs::read(&path).unwrap_or_else(|e| {
        panic!("missing {path}: {e} (run: python3 scripts/gen_scale_test_data.py)")
    });
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn stage_bench(n: usize) {
    let n_features = 32usize;
    let data = load_scale_data(n, n_features);
    let n_samples = data.len() / n_features;
    assert_eq!(n_samples, n);

    let n_neighbors = 15usize;
    let min_samples = 5usize;
    let n_epochs = 50usize;
    let base_min_cluster_size = 5i64;
    let seed = 0i64;

    let t_total = Instant::now();

    let t0 = Instant::now();
    let mut normalized = data.to_vec();
    for i in 0..n_samples {
        let row = &mut normalized[i * n_features..i * n_features + n_features];
        let mut norm_sq = 0.0f32;
        for &v in row.iter() {
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();
        if norm != 0.0 {
            for v in row.iter_mut() {
                *v /= norm;
            }
        }
    }
    let t_normalize = t0.elapsed();

    let mut state: RngState = [0; 3];
    rng_seed(&mut state, seed);
    let draw_rng_state = |state: &mut RngState| -> RngState {
        [
            tau_rand_int(state) as i64,
            tau_rand_int(state) as i64,
            tau_rand_int(state) as i64,
        ]
    };
    let n_trees = 4usize;
    let leaf_size = 10usize.max(n_neighbors);
    let n_iters = 5usize.max((n_samples as f64).log2().round() as usize);
    let effective_max_candidates = 60usize.min(((n_neighbors as f64) * 1.5) as usize);
    let rng_state = draw_rng_state(&mut state);
    let mut rng_states: Vec<RngState> = (0..n_trees).map(|_| draw_rng_state(&mut state)).collect();

    let t0 = Instant::now();
    let (nn_inds, nn_dists) = knn_graph_impl(
        &normalized,
        n_features,
        n_neighbors,
        &rng_state,
        &mut rng_states,
        leaf_size,
        200,
        effective_max_candidates,
        n_iters,
        0.001,
        Some(0.001),
        true,
        1,
        false,
    );
    let t_knn = t0.elapsed();

    let effective_n_neighbors = 1.0f32 * n_neighbors as f32;
    let t0 = Instant::now();
    let csr = neighbor_graph_matrix(
        effective_n_neighbors,
        &nn_inds,
        &nn_dists,
        n_neighbors,
        true,
    );
    let t_graph_matrix = t0.elapsed();

    let n_components = (n_neighbors / 4).clamp(4, 15);
    let graph_indptr: Vec<i64> = csr.indptr.iter().map(|&x| x as i64).collect();
    let graph_indices: Vec<i64> = csr.indices.iter().map(|&x| x as i64).collect();
    let approx_n_parts = ((8.0 * (n_samples as f64).sqrt()) as i64).clamp(256, 16384) as usize;

    let t0 = Instant::now();
    let init_embedding = label_propagation_init_recursive(
        &graph_indptr,
        &graph_indices,
        &csr.data,
        n_samples,
        Some(&normalized),
        n_features,
        2, // n_label_prop_iter default
        n_epochs,
        approx_n_parts,
        n_components,
        0.5,
        0.0, // noise_level default
        true,
        64,
        &mut state,
    );
    let t_label_prop = t0.elapsed();

    let indptr_u32: Vec<u32> = csr.indptr.iter().map(|&x| x as u32).collect();
    let indices_u32: Vec<u32> = csr.indices.iter().map(|&x| x as u32).collect();

    let t0 = Instant::now();
    let embedding = node_embedding_repr_self_driven(
        &indptr_u32,
        &indices_u32,
        &csr.data,
        n_samples,
        n_components,
        n_epochs,
        Some(&init_embedding),
        0.1,
        1.0,
        0.0,
        &mut state,
    );
    let t_node_embed = t0.elapsed();

    let build_opts = BuildClusterLayersOptions {
        min_samples,
        base_min_cluster_size,
        base_n_clusters: None,
        reproducible_flag: false,
        min_similarity_threshold: 0.0,
        max_layers: 10,
    };
    let t0 = Instant::now();
    let _layers = build_cluster_layers(&embedding, n_components, &build_opts);
    let t_cluster_layers = t0.elapsed();

    let total = t_total.elapsed();

    let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
    println!("\n--- n={n} ---");
    println!("  normalize          {:>10.2} ms", ms(t_normalize));
    println!("  knn_graph          {:>10.2} ms", ms(t_knn));
    println!("  neighbor_graph_mtx {:>10.2} ms", ms(t_graph_matrix));
    println!("  label_prop_init    {:>10.2} ms", ms(t_label_prop));
    println!("  node_embedding     {:>10.2} ms", ms(t_node_embed));
    println!("  build_cluster_lyrs {:>10.2} ms", ms(t_cluster_layers));
    println!(
        "  TOTAL (sum stages) {:>10.2} ms  (wall total incl. overhead: {:.2} ms)",
        ms(t_normalize)
            + ms(t_knn)
            + ms(t_graph_matrix)
            + ms(t_label_prop)
            + ms(t_node_embed)
            + ms(t_cluster_layers),
        ms(total)
    );
}

#[test]
#[ignore]
fn stage_bench_scale() {
    for &n in &[2000usize, 10000, 30000] {
        stage_bench(n);
    }
}
