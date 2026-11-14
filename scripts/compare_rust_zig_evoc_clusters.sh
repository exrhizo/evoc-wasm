#!/usr/bin/env bash
# Cross-language (Rust vs Zig) agreement regression check for
# `evoc_clusters`/`EVoC` (phase 9's self-driven pipeline). Runs both
# languages' `EVoC::fit_predict` on the seed panel 0..19 against the shared
# `fixtures/clustering/evoc_clusters_e2e_ari_check.json` fixture, then
# compares via `scripts/compare_rust_zig_evoc_clusters.py`.
#
# See `rust/core/PCA_VALIDATION.md`, "Cross-language (Rust vs Zig)
# agreement", for the full writeup: output is NOT bit-identical (median ARI
# ~0.67 across the panel), so this is a pragmatic drift-detector, not a
# parity gate -- it fails loudly if a future change makes cross-language
# agreement meaningfully worse than what was measured when this check was
# added, not because 0.67 is a "good" number in isolation.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="${EVOC_CROSS_LANG_OUT_DIR:-/tmp/evoc_cross_lang}"
mkdir -p "$OUT_DIR"

echo "==> Rust: dumping evoc_clusters labels for seed panel"
(cd rust/core && EVOC_CROSS_LANG_OUT_DIR="$OUT_DIR" cargo test --test dump_evoc_clusters_labels_for_cross_lang -- --ignored --nocapture)

echo "==> Zig: dumping evoc_clusters labels for seed panel"
if ! command -v zig >/dev/null 2>&1; then
    echo "zig not found on PATH -- skipping (see PORTING.md for zig install notes)" >&2
    exit 0
fi
(cd zig && EVOC_FIXTURES="$(pwd)/../fixtures" EVOC_CROSS_LANG_OUT_DIR="$OUT_DIR" zig test src/dump_evoc_clusters_labels_for_cross_lang.zig)

echo "==> Comparing Rust vs Zig labels (partition equality + ARI)"
python3 scripts/compare_rust_zig_evoc_clusters.py \
    --rust "$OUT_DIR/rust_labels.json" \
    --zig "$OUT_DIR/zig_labels.json" \
    --min-median-ari 0.5
