//! Port of `evoc/clustering.py`: `build_cluster_layers` (phase 8) and, as of
//! phase 9, the full `evoc_clusters` / `EVoC.fit_predict` pipeline.
//!
//! PORT NOTE (scope, phase 9): `evoc_clusters` composes `knn_graph` ->
//! `neighbor_graph_matrix` -> `label_propagation_init` (recursive) ->
//! `node_embedding` -> `build_cluster_layers`. The recursive
//! `label_propagation_init` path eventually reaches sklearn's
//! `PCA(random_state=...)` for small sub-graphs, whose internal randomized-SVD
//! draws are not portably replayable (see PORT_NOTES.md). Rather than leaving
//! this out of scope, this port implements the actual algorithm behind
//! sklearn's randomized PCA (Halko/Martinsson/Tropp, see `pca.rs`) driven by
//! this project's own `tau_rand` RNG throughout the whole pipeline --
//! deterministic and reproducible (same seed -> same output, and
//! cross-language reproducible against the Zig port), but NOT bit-exact with
//! the Python reference (impossible once PCA is involved). Validated instead
//! via Adjusted Rand Index against a real `evoc.EVoC` run -- see
//! `PCA_VALIDATION.md`.

use crate::boruvka::parallel_boruvka;
use crate::cluster_trees::{
    condense_tree, extract_leaves, get_cluster_label_vector, get_point_membership_strength_vector,
    mask_condensed_tree, mst_to_linkage_tree, CondensedTree,
};
use crate::clustering_utilities::{
    _binary_search_for_n_clusters, compute_total_persistence, extract_clusters_by_id, find_peaks,
    min_cluster_size_barcode, select_diverse_peaks,
};
use crate::graph_construction::neighbor_graph_matrix;
use crate::kdtree::build_kdtree;
use crate::knn_graph::knn_graph as knn_graph_impl;
use crate::label_propagation::label_propagation_init_recursive;
use crate::node_embedding::node_embedding_repr_self_driven;
use crate::rng::{tau_rand_int, RngState};

/// Result of [`build_cluster_layers`]: parallel lists, one entry per layer.
pub struct ClusterLayers {
    pub cluster_layers: Vec<Vec<i64>>,
    pub membership_strength_layers: Vec<Vec<f32>>,
    pub persistence_scores: Vec<f32>,
}

/// Options for [`build_cluster_layers`], mirroring the reference's keyword
/// arguments (all have the reference's defaults).
pub struct BuildClusterLayersOptions {
    pub min_samples: usize,
    pub base_min_cluster_size: i64,
    /// PORT NOTE: the `base_n_clusters is not None` branch has no fixture
    /// coverage (see PORT_NOTES.md); ported by composing already-fixtured
    /// pieces (`_binary_search_for_n_clusters`) but not independently
    /// verified end-to-end.
    pub base_n_clusters: Option<i64>,
    pub reproducible_flag: bool,
    pub min_similarity_threshold: f64,
    pub max_layers: usize,
}

impl Default for BuildClusterLayersOptions {
    fn default() -> Self {
        BuildClusterLayersOptions {
            min_samples: 5,
            base_min_cluster_size: 10,
            base_n_clusters: None,
            reproducible_flag: false,
            min_similarity_threshold: 0.2,
            max_layers: 10,
        }
    }
}

/// Port of `build_cluster_layers`. `data` is a flat row-major `n_samples x
/// n_features` embedding (already computed -- see the module PORT NOTE for
/// why this, not `evoc_clusters`, is the phase-8 target).
pub fn build_cluster_layers(
    data: &[f32],
    n_features: usize,
    opts: &BuildClusterLayersOptions,
) -> ClusterLayers {
    let n_samples = data.len() / n_features;
    let mut min_cluster_size = opts.base_min_cluster_size;

    // leaf_size=40 is the reference's build_kdtree default.
    let tree = build_kdtree(data, n_features, 40);
    let mut edges = parallel_boruvka(&tree, 1, opts.min_samples, opts.reproducible_flag);
    edges.sort_by(|a, b| a[2].partial_cmp(&b[2]).unwrap());
    let uncondensed_tree = mst_to_linkage_tree(&edges);

    let condensed_tree: CondensedTree;
    let leaves: Vec<i64>;
    let clusters: Vec<i64>;
    let strengths: Vec<f32>;

    if let Some(base_n_clusters) = opts.base_n_clusters {
        let (bs_leaves, bs_clusters, _bs_strengths) =
            _binary_search_for_n_clusters(&uncondensed_tree, base_n_clusters, n_samples);
        // cluster_sizes = np.bincount(clusters[clusters >= 0])
        let max_label = bs_clusters.iter().filter(|&&c| c >= 0).copied().max();
        min_cluster_size = match max_label {
            Some(max_label) => {
                let mut counts = vec![0i64; (max_label + 1) as usize];
                for &c in bs_clusters.iter() {
                    if c >= 0 {
                        counts[c as usize] += 1;
                    }
                }
                let min_count = counts.into_iter().filter(|&c| c > 0).min().unwrap_or(1);
                1.max(min_count)
            }
            None => opts.base_min_cluster_size,
        };
        let _ = (bs_leaves, bs_clusters); // leaves/clusters recomputed below via condensed_tree
        condensed_tree = condense_tree(&uncondensed_tree, min_cluster_size);
        leaves = extract_leaves(&condensed_tree);
        clusters = get_cluster_label_vector(&condensed_tree, &leaves, 0.0, n_samples);
        strengths = get_point_membership_strength_vector(&condensed_tree, &leaves, &clusters);
    } else {
        condensed_tree = condense_tree(&uncondensed_tree, opts.base_min_cluster_size);
        leaves = extract_leaves(&condensed_tree);
        clusters = get_cluster_label_vector(&condensed_tree, &leaves, 0.0, n_samples);
        strengths = get_point_membership_strength_vector(&condensed_tree, &leaves, &clusters);
    }

    let mask: Vec<bool> = condensed_tree
        .child
        .iter()
        .map(|&c| c >= n_samples as i64)
        .collect();
    let cluster_tree = mask_condensed_tree(&condensed_tree, &mask);

    type BarcodeAndPeaks = (
        Vec<f32>,
        Vec<f32>,
        Vec<i64>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<usize>,
    );
    let (births, deaths, _parents, _lambda_deaths, sizes, total_persistence, peaks): BarcodeAndPeaks =
        if !cluster_tree.child.is_empty()
        && *cluster_tree.child.last().unwrap() >= n_samples as i64
    {
        let (b, d, p, ld) =
            min_cluster_size_barcode(&cluster_tree, n_samples as i64, min_cluster_size as f32);
        let (s, tp) = compute_total_persistence(&b, &d, &ld);
        let pk = find_peaks(&tp);
        (b, d, p, ld, s, tp, pk)
    } else {
        (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
    };

    let mut cluster_layers: Vec<Vec<i64>> = vec![clusters];
    let mut membership_strength_layers: Vec<Vec<f32>> = vec![strengths];
    let mut persistence_scores: Vec<f32> = vec![0.0];

    let peaks_i64: Vec<i64> = peaks.iter().map(|&p| p as i64).collect();
    let selected_peaks = select_diverse_peaks(
        &peaks_i64,
        &total_persistence,
        &sizes,
        &births,
        &deaths,
        opts.min_similarity_threshold,
        opts.max_layers.saturating_sub(1),
    );

    for peak in selected_peaks {
        let peak = peak as usize;
        let best_birth = sizes[peak];
        let persistence = total_persistence[peak];
        let selected_clusters: Vec<i64> = (0..births.len() as i64)
            .filter(|&i| births[i as usize] <= best_birth && deaths[i as usize] > best_birth)
            .map(|i| i + n_samples as i64)
            .collect();
        let (labels, strengths) = extract_clusters_by_id(&condensed_tree, &selected_clusters);
        cluster_layers.push(labels);
        membership_strength_layers.push(strengths);
        persistence_scores.push(persistence);
    }

    // Sort layers by n_clusters_per_layer = layer.max() + 1, descending.
    let n_clusters_per_layer: Vec<i64> = cluster_layers
        .iter()
        .map(|layer| layer.iter().copied().max().unwrap_or(-1) + 1)
        .collect();
    let mut order: Vec<usize> = (0..cluster_layers.len()).collect();
    order.sort_by(|&a, &b| {
        n_clusters_per_layer[a]
            .partial_cmp(&n_clusters_per_layer[b])
            .unwrap()
    });
    order.reverse();

    let cluster_layers = order.iter().map(|&i| cluster_layers[i].clone()).collect();
    let membership_strength_layers = order
        .iter()
        .map(|&i| membership_strength_layers[i].clone())
        .collect();
    let persistence_scores = order.iter().map(|&i| persistence_scores[i]).collect();

    ClusterLayers {
        cluster_layers,
        membership_strength_layers,
        persistence_scores,
    }
}

/// Options for [`evoc_clusters`], mirroring the reference's keyword
/// arguments (all have the reference's defaults except `reproducible_flag`,
/// which is a no-op here -- this port always uses the reproducible/`_repr`
/// node-embedding kernel, per the module PORT NOTE).
pub struct EvocClustersOptions {
    pub noise_level: f32,
    pub base_min_cluster_size: i64,
    pub n_neighbors: usize,
    pub min_samples: usize,
    pub n_epochs: usize,
    pub neighbor_scale: f32,
    pub min_similarity_threshold: f64,
    pub max_layers: usize,
    pub n_label_prop_iter: usize,
    /// `min(max(n_neighbors/4, 4), 15)` if `None`, matching the reference's
    /// `node_embedding_dim` default derivation.
    pub node_embedding_dim: Option<usize>,
}

impl Default for EvocClustersOptions {
    fn default() -> Self {
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
}

/// Port of `evoc_clusters` (module-level function). `data` is a flat
/// row-major `n_samples x n_features` array of float embeddings (the int8/
/// uint8 quantized paths are out of scope, matching the rest of this port --
/// see PORTING.md's module map: `nndescent_int8`/`nndescent_uint8` are listed
/// "later"). `seed` drives every random draw in the pipeline (RP-tree
/// construction, NN-descent candidate sampling, label-propagation partition
/// seeding + tie-breaks, randomized-PCA base cases, node-embedding SGD) via
/// this project's own `tau_rand` RNG -- same seed -> same output,
/// cross-language reproducible against the Zig port, but not bit-exact with
/// the Python reference (see the module PORT NOTE on `evoc_clusters` scope).
///
/// Returns [`ClusterLayers`] plus the raw knn indices/distances (flat
/// row-major `n_samples x n_neighbors`), matching the reference's return
/// shape when `approx_n_clusters` is not requested (the
/// `approx_n_clusters`/`binary_search_for_n_clusters` single-layer branch and
/// `return_duplicates` are not ported here -- out of scope for this pass, see
/// STATUS.md).
pub fn evoc_clusters(
    data: &[f32],
    n_features: usize,
    seed: i64,
    opts: &EvocClustersOptions,
) -> (ClusterLayers, Vec<i32>, Vec<f32>) {
    let n_samples = data.len() / n_features;

    // Unit-normalize rows (cosine distance), matching knn_graph's float path.
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

    let mut state: RngState = [0; 3];
    crate::rng::seed(&mut state, seed);

    let draw_rng_state = |state: &mut RngState| -> RngState {
        [
            tau_rand_int(state) as i64,
            tau_rand_int(state) as i64,
            tau_rand_int(state) as i64,
        ]
    };

    // knn_graph parameters, single-threaded defaults (see PORT_NOTES.md).
    let n_trees = 4usize;
    let leaf_size = 10usize.max(opts.n_neighbors);
    let n_iters = 5usize.max((n_samples as f64).log2().round() as usize);
    let effective_max_candidates = 60usize.min(((opts.n_neighbors as f64) * 1.5) as usize);

    let rng_state = draw_rng_state(&mut state);
    let mut rng_states: Vec<RngState> = (0..n_trees).map(|_| draw_rng_state(&mut state)).collect();

    let (nn_inds, nn_dists) = knn_graph_impl(
        &normalized,
        n_features,
        opts.n_neighbors,
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

    let effective_n_neighbors = opts.neighbor_scale * opts.n_neighbors as f32;
    let csr = neighbor_graph_matrix(
        effective_n_neighbors,
        &nn_inds,
        &nn_dists,
        opts.n_neighbors,
        true,
    );

    let n_components = opts
        .node_embedding_dim
        .unwrap_or_else(|| (opts.n_neighbors / 4).max(4).min(15));

    let graph_indptr: Vec<i64> = csr.indptr.iter().map(|&x| x as i64).collect();
    let graph_indices: Vec<i64> = csr.indices.iter().map(|&x| x as i64).collect();

    let approx_n_parts = ((8.0 * (n_samples as f64).sqrt()) as i64).clamp(256, 16384) as usize;

    let init_embedding = label_propagation_init_recursive(
        &graph_indptr,
        &graph_indices,
        &csr.data,
        n_samples,
        Some(&normalized),
        n_features,
        opts.n_label_prop_iter,
        opts.n_epochs,
        approx_n_parts,
        n_components,
        0.5, // scaling
        opts.noise_level,
        true, // recursive_init
        64,   // base_init_threshold
        &mut state,
    );

    let indptr_u32: Vec<u32> = csr.indptr.iter().map(|&x| x as u32).collect();
    let indices_u32: Vec<u32> = csr.indices.iter().map(|&x| x as u32).collect();

    let embedding = node_embedding_repr_self_driven(
        &indptr_u32,
        &indices_u32,
        &csr.data,
        n_samples,
        n_components,
        opts.n_epochs,
        Some(&init_embedding),
        0.1, // initial_alpha
        1.0, // negative_sample_rate
        opts.noise_level,
        &mut state,
    );

    let build_opts = BuildClusterLayersOptions {
        min_samples: opts.min_samples,
        base_min_cluster_size: opts.base_min_cluster_size,
        base_n_clusters: None,
        reproducible_flag: false, // bit-identical to true at n_threads=1, see boruvka PORT_NOTES
        min_similarity_threshold: opts.min_similarity_threshold,
        max_layers: opts.max_layers,
    };
    let layers = build_cluster_layers(&embedding, n_components, &build_opts);

    (layers, nn_inds, nn_dists)
}

/// Faithful port of `EVoC.fit_predict`'s single-best-layer selection: when
/// [`evoc_clusters`] (with `approx_n_clusters=None`, the only mode ported
/// here) returns multiple layers, `fit_predict` picks the layer with the
/// highest persistence score (`np.argmax`, first-max-wins on ties).
pub struct EVoC {
    pub opts: EvocClustersOptions,
    pub seed: i64,
    pub labels: Vec<i64>,
    pub membership_strengths: Vec<f32>,
    pub cluster_layers: Vec<Vec<i64>>,
    pub membership_strength_layers: Vec<Vec<f32>>,
    pub persistence_scores: Vec<f32>,
    pub nn_inds: Vec<i32>,
    pub nn_dists: Vec<f32>,
}

impl EVoC {
    pub fn new(seed: i64, opts: EvocClustersOptions) -> Self {
        EVoC {
            opts,
            seed,
            labels: Vec::new(),
            membership_strengths: Vec::new(),
            cluster_layers: Vec::new(),
            membership_strength_layers: Vec::new(),
            persistence_scores: Vec::new(),
            nn_inds: Vec::new(),
            nn_dists: Vec::new(),
        }
    }

    /// Port of `EVoC.fit_predict`. Returns `labels_` and stores every
    /// fitted attribute the reference exposes (minus `duplicates_`, out of
    /// scope for this pass -- `find_duplicates` is ported and available in
    /// `clustering_utilities.rs` for callers who want it directly).
    pub fn fit_predict(&mut self, data: &[f32], n_features: usize) -> &[i64] {
        let (layers, nn_inds, nn_dists) = evoc_clusters(data, n_features, self.seed, &self.opts);

        self.nn_inds = nn_inds;
        self.nn_dists = nn_dists;
        self.cluster_layers = layers.cluster_layers;
        self.membership_strength_layers = layers.membership_strength_layers;
        self.persistence_scores = layers.persistence_scores;

        let best_layer = if self.cluster_layers.len() == 1 {
            0
        } else {
            // np.argmax: first index attaining the max.
            let mut best = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in self.persistence_scores.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best = i;
                }
            }
            best
        };

        self.labels = self.cluster_layers[best_layer].clone();
        self.membership_strengths = self.membership_strength_layers[best_layer].clone();
        &self.labels
    }
}

#[cfg(test)]
mod evoc_clusters_smoke_tests {
    use super::*;

    fn make_blobs(n_per: usize, dim: usize, centers: &[Vec<f32>], seed: i64) -> Vec<f32> {
        let mut state: RngState = [0; 3];
        crate::rng::seed(&mut state, seed);
        let mut data = Vec::with_capacity(n_per * centers.len() * dim);
        for c in centers {
            for _ in 0..n_per {
                for d in 0..dim {
                    let noise = crate::pca::next_standard_normal(&mut state) * 0.3;
                    data.push(c[d] + noise);
                }
            }
        }
        data
    }

    #[test]
    fn evoc_clusters_runs_on_small_blobs() {
        let dim = 10;
        let centers: Vec<Vec<f32>> = vec![
            (0..dim).map(|i| if i == 0 { 5.0 } else { 0.0 }).collect(),
            (0..dim).map(|i| if i == 1 { 5.0 } else { 0.0 }).collect(),
            (0..dim).map(|i| if i == 2 { 5.0 } else { 0.0 }).collect(),
        ];
        let data = make_blobs(30, dim, &centers, 11);
        let n_samples = 90;

        let opts = EvocClustersOptions {
            n_neighbors: 10,
            n_epochs: 15,
            ..Default::default()
        };
        let (layers, nn_inds, nn_dists) = evoc_clusters(&data, dim, 42, &opts);
        assert_eq!(nn_inds.len(), n_samples * opts.n_neighbors);
        assert_eq!(nn_dists.len(), n_samples * opts.n_neighbors);
        assert!(!layers.cluster_layers.is_empty());
        assert_eq!(layers.cluster_layers[0].len(), n_samples);
    }

    #[test]
    fn evoc_fit_predict_runs() {
        let dim = 8;
        let centers: Vec<Vec<f32>> = vec![
            (0..dim).map(|i| if i == 0 { 4.0 } else { 0.0 }).collect(),
            (0..dim).map(|i| if i == 1 { 4.0 } else { 0.0 }).collect(),
        ];
        let data = make_blobs(25, dim, &centers, 5);
        let opts = EvocClustersOptions {
            n_neighbors: 8,
            n_epochs: 10,
            ..Default::default()
        };
        let mut evoc = EVoC::new(7, opts);
        let labels = evoc.fit_predict(&data, dim);
        assert_eq!(labels.len(), 50);
    }
}
