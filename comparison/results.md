# Rust vs Zig comparison

Updated per phase by `scripts/compare.sh` (phase 10) and by hand where noted.
All measurements on the shared fixtures; wasm builds are release/optimized,
run single-threaded under Node v20.

## Binary size (wasm32, optimized)

| Phase | Components included | Rust .wasm | Zig .wasm |
|-------|---------------------|------------|-----------|
| —     | (none yet)          | —          | —         |

## Execution time

| Benchmark | Rust | Zig | Python (numba, native) |
|-----------|------|-----|------------------------|
| —         | —    | —   | —                      |

## Cross-implementation parity

Rust and Zig must agree bit-for-bit on all components (see PORTING.md numeric
policy). Verified: (pending phase 10 harness).

## Toolchain ergonomics notes

Free-form observations comparing the two ports, collected as we go:

- (none yet)
