//! Core EVoC clustering algorithm (faithful port of TutteInstitute/evoc).
//!
//! Freestanding-friendly: no threads, no filesystem, no clock. See PORTING.md
//! at the repository root for the porting contract and module map.

// Modules are added phase by phase; keep this list in PORTING.md order.
pub mod disjoint_set;
pub mod rng;
pub mod distance;
pub mod kdtree;
