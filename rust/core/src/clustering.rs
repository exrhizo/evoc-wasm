//! Port of `evoc/clustering.py`'s `build_cluster_layers` -- the phase-8
//! capstone orchestration function.
//!
//! PORT NOTE (scope): `evoc_clusters` / `EVoC.fit_predict` are explicitly
//! OUT of scope (see PORT_NOTES.md's "clustering.py (phase 8)" section) --
//! they depend on unportable sklearn PCA internals reached through
//! `label_propagation_init`'s recursive default. `build_cluster_layers`
//! sidesteps all of that by taking a pre-computed embedding as input, which
//! is why it -- not `evoc_clusters` -- is the portable phase-8 target.

use crate::boruvka::parallel_boruvka;
use crate::cluster_trees::{
    condense_tree, extract_leaves, get_cluster_label_vector, get_point_membership_strength_vector,
    mask_condensed_tree, mst_to_linkage_tree, CondensedTree,
};
use crate::clustering_utilities::{
    _binary_search_for_n_clusters, compute_total_persistence, extract_clusters_by_id, find_peaks,
    min_cluster_size_barcode, select_diverse_peaks,
};
use crate::kdtree::build_kdtree;

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

    type BarcodeAndPeaks = (Vec<f32>, Vec<f32>, Vec<i64>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<usize>);
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
