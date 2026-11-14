use evoc_core::clustering::{evoc_clusters, EvocClustersOptions};
use evoc_core::label_propagation::label_propagation_init_recursive;
use evoc_core::rng::{seed, RngState};

#[test]
fn pca_base_case_pads_to_the_requested_embedding_width() {
    let samples = [1.0_f32, 2.0, 2.0, 4.0, 3.0, 6.0];
    let mut state: RngState = [0; 3];
    seed(&mut state, 42);
    let result = label_propagation_init_recursive(
        &[0; 4],
        &[],
        &[],
        3,
        Some(&samples),
        2,
        2,
        10,
        4,
        4,
        0.5,
        0.0,
        true,
        64,
        &mut state,
    );
    assert_eq!(result.len(), 3 * 4);
    assert!(result.iter().all(|v| v.is_finite()));
    for row in result.as_chunks::<4>().0 {
        assert_eq!(&row[2..], &[0.0, 0.0]);
    }
}

#[test]
fn end_to_end_clustering_accepts_two_feature_vectors() {
    let data: Vec<f32> = (0..75)
        .flat_map(|i| [i as f32 / 74.0, 1.0 - i as f32 / 74.0])
        .collect();
    let (layers, _, _) = evoc_clusters(&data, 2, 42, &EvocClustersOptions::default());
    assert!(!layers.cluster_layers.is_empty());
    assert!(layers.cluster_layers.iter().all(|layer| layer.len() == 75));
}
