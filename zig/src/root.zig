//! Core EVoC clustering algorithm (faithful port of TutteInstitute/evoc).
//! Freestanding-friendly: core code takes allocators as parameters and uses
//! no OS facilities. See PORTING.md at the repository root.

// Modules are added phase by phase; keep this list in PORTING.md order.

test {
    @import("std").testing.refAllDecls(@This());
}
