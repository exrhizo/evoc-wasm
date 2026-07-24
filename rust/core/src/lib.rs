//! Core EVoC clustering algorithm (faithful port of TutteInstitute/evoc).
//!
//! Freestanding-friendly: no threads, no filesystem, no clock. See PORTING.md
//! at the repository root for the porting contract and module map.

// Modules are added phase by phase; keep this list in PORTING.md order.
pub mod boruvka;
pub mod cluster_trees;
pub mod clustering;
pub mod clustering_utilities;
pub mod disjoint_set;
pub mod distance;
pub mod graph_construction;
pub mod kdtree;
pub mod knn_graph;
pub mod label_propagation;
pub mod nnd_heap;
pub mod nndescent_float;
pub mod node_embedding;
pub mod rng;
