//! Core EVoC clustering algorithm (faithful port of TutteInstitute/evoc).
//! Freestanding-friendly: core code takes allocators as parameters and uses
//! no OS facilities. See PORTING.md at the repository root.

// Modules are added phase by phase; keep this list in PORTING.md order.

pub const disjoint_set = @import("disjoint_set.zig");
pub const rng = @import("rng.zig");
pub const distance = @import("distance.zig");
pub const kdtree = @import("kdtree.zig");
pub const nnd_heap = @import("nnd_heap.zig");
pub const nndescent_float = @import("nndescent_float.zig");
pub const graph_construction = @import("graph_construction.zig");
pub const knn_graph = @import("knn_graph.zig");
pub const boruvka = @import("boruvka.zig");
pub const cluster_trees = @import("cluster_trees.zig");
pub const label_propagation = @import("label_propagation.zig");
pub const node_embedding = @import("node_embedding.zig");
pub const clustering_utilities = @import("clustering_utilities.zig");
pub const clustering = @import("clustering.zig");

// Test-only fixture-loading helper (not part of the freestanding surface).
const fixture = @import("fixture.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
