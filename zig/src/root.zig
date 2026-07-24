//! Core EVoC clustering algorithm (faithful port of TutteInstitute/evoc).
//! Freestanding-friendly: core code takes allocators as parameters and uses
//! no OS facilities. See PORTING.md at the repository root.

// Modules are added phase by phase; keep this list in PORTING.md order.

pub const disjoint_set = @import("disjoint_set.zig");
pub const rng = @import("rng.zig");
pub const distance = @import("distance.zig");

// Test-only fixture-loading helper (not part of the freestanding surface).
const fixture = @import("fixture.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
