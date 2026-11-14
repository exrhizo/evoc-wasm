#!/usr/bin/env bash
# Run every quality gate for both language ports. Exits non-zero on first failure.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> cargo fmt --check"
(cd rust && cargo fmt --all -- --check)

echo "==> cargo clippy (deny warnings)"
(cd rust && cargo clippy --workspace --all-targets -- -D warnings)

echo "==> cargo test"
(cd rust && cargo test --workspace)

echo "==> zig fmt --check"
(cd zig && zig fmt --check src/)

echo "==> zig build test"
(cd zig && zig build test)

echo "==> wasm fixture parity (rust + zig .wasm binaries, if built)"
RUST_WASM="rust/target/wasm32-unknown-unknown/release/evoc_core.wasm"
ZIG_WASM="zig/zig-out/bin/evoc_core.wasm"
if [[ -f "$RUST_WASM" || -f "$ZIG_WASM" ]]; then
  node scripts/wasm_fixture_runner.mjs
else
  echo "skipping: neither wasm binary is built."
  echo "  rust: cd rust && cargo build --target wasm32-unknown-unknown --release -p evoc-core"
  echo "  zig:  cd zig && zig build wasm"
fi

echo "All checks passed."
