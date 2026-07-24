//! Shared fixture-loading helpers for the parity test suite.
//!
//! Fixtures live under `fixtures/<component>/*.json` (see PORTING.md for the
//! schema). Location is resolved via `UVOC_FIXTURES`, falling back to the
//! repo-relative path from this crate.
//!
//! Each `tests/*.rs` file is compiled as its own crate with its own copy of
//! this module, so not every helper is used by every test file; that's
//! expected and silenced here rather than per-file.
#![allow(dead_code)]

use std::path::PathBuf;

/// Root directory containing `disjoint_set/`, `rng/`, `distance/`, etc.
pub fn fixtures_dir() -> PathBuf {
    let base = std::env::var("UVOC_FIXTURES")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures").to_string());
    PathBuf::from(base)
}

/// Load and parse a fixture JSON file, e.g. `load("rng", "tau_rand_seed_42")`.
pub fn load(component: &str, case: &str) -> serde_json::Value {
    let path = fixtures_dir().join(component).join(format!("{case}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {}", path.display(), e));
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("failed to parse fixture {}: {}", path.display(), e))
}

/// Parse a JSON number as f64 and narrow to f32, matching PORTING.md's
/// "exact" f32 comparison rule (JSON numbers are doubles; f32 values
/// round-trip losslessly through double).
pub fn as_f32(v: &serde_json::Value) -> f32 {
    v.as_f64().expect("expected JSON number") as f32
}

/// Assert two f32 values are bit-identical (PORTING.md "exact" tolerance).
pub fn assert_f32_exact(actual: f32, expected: f32, context: &str) {
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "{context}: bit mismatch actual={actual} ({:#010x}) expected={expected} ({:#010x})",
        actual.to_bits(),
        expected.to_bits(),
    );
}

/// Assert two f32 values match within relative/absolute tolerance.
pub fn assert_f32_close(actual: f32, expected: f32, rel: f64, abs: f64, context: &str) {
    let diff = (actual as f64 - expected as f64).abs();
    let allowed = abs + rel * (expected as f64).abs();
    assert!(
        diff <= allowed,
        "{context}: actual={actual} expected={expected} diff={diff} allowed={allowed}"
    );
}

/// Tolerance parsed from a fixture's `tolerance` field.
pub enum Tolerance {
    Exact,
    Rel { rel: f64, abs: f64 },
}

impl Tolerance {
    pub fn parse(v: &serde_json::Value) -> Self {
        if let Some(s) = v.as_str() {
            assert_eq!(s, "exact", "unknown tolerance string {s}");
            return Tolerance::Exact;
        }
        let rel = v["rel"].as_f64().expect("tolerance.rel");
        let abs = v.get("abs").and_then(|a| a.as_f64()).unwrap_or(0.0);
        Tolerance::Rel { rel, abs }
    }

    /// Parse a fixture's `tolerance` field that may be per-field, e.g.
    /// `{"default": "exact", "fields": {"radius": {"rel": 1e-5}}}`, and
    /// return the tolerance applying to `field`. The plain `"exact"` /
    /// `{"rel": ..}` forms apply uniformly to every field.
    pub fn parse_for_field(v: &serde_json::Value, field: &str) -> Self {
        if v.get("default").is_some() {
            if let Some(t) = v["fields"].get(field) {
                return Tolerance::parse(t);
            }
            return Tolerance::parse(&v["default"]);
        }
        Tolerance::parse(v)
    }

    pub fn assert_close(&self, actual: f32, expected: f32, context: &str) {
        match self {
            Tolerance::Exact => assert_f32_exact(actual, expected, context),
            Tolerance::Rel { rel, abs } => assert_f32_close(actual, expected, *rel, *abs, context),
        }
    }
}
