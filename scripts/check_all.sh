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

echo "All checks passed."
