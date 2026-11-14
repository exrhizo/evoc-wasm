//! Fixture parity tests for `evoc_core::distance`.

mod common;

use common::{load, Tolerance};
use evoc_core::distance::{
    fast_bit_jaccard, fast_bit_jaccard_u64, fast_cosine, fast_int_inner_product_dissimilarity,
};

#[test]
fn fast_cosine_pairs() {
    let fixture = load("distance", "fast_cosine");
    let tolerance = Tolerance::parse(&fixture["tolerance"]);

    let pairs = fixture["inputs"]["pairs"].as_array().unwrap();
    let expected: Vec<f32> = fixture["expected"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    assert_eq!(pairs.len(), expected.len());

    for (i, (pair, &exp)) in pairs.iter().zip(expected.iter()).enumerate() {
        let x: Vec<f32> = pair["x"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let y: Vec<f32> = pair["y"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let actual = fast_cosine(&x, &y);
        tolerance.assert_close(actual, exp, &format!("fast_cosine pair {i}"));
    }
}

#[test]
fn fast_int_inner_product_dissimilarity_pairs() {
    let fixture = load("distance", "fast_int_inner_product_dissimilarity");
    let tolerance = Tolerance::parse(&fixture["tolerance"]);

    let pairs = fixture["inputs"]["pairs"].as_array().unwrap();
    let expected: Vec<f32> = fixture["expected"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    assert_eq!(pairs.len(), expected.len());

    for (i, (pair, &exp)) in pairs.iter().zip(expected.iter()).enumerate() {
        let x: Vec<i8> = pair["x"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i8)
            .collect();
        let y: Vec<i8> = pair["y"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i8)
            .collect();
        let actual = fast_int_inner_product_dissimilarity(&x, &y);
        tolerance.assert_close(
            actual,
            exp,
            &format!("fast_int_inner_product_dissimilarity pair {i}"),
        );
    }
}

#[test]
fn fast_bit_jaccard_pairs() {
    let fixture = load("distance", "fast_bit_jaccard");
    let tolerance = Tolerance::parse(&fixture["tolerance"]);

    let pairs = fixture["inputs"]["pairs"].as_array().unwrap();
    let expected: Vec<f32> = fixture["expected"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let expected_u64: Vec<f32> = fixture["expected"]["values_u64"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    assert_eq!(pairs.len(), expected.len());
    assert_eq!(pairs.len(), expected_u64.len());

    for (i, ((pair, &exp), &exp_u64)) in pairs
        .iter()
        .zip(expected.iter())
        .zip(expected_u64.iter())
        .enumerate()
    {
        let x: Vec<u8> = pair["x"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();
        let y: Vec<u8> = pair["y"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();

        let actual = fast_bit_jaccard(&x, &y);
        tolerance.assert_close(actual, exp, &format!("fast_bit_jaccard pair {i}"));

        assert_eq!(
            x.len() % 8,
            0,
            "fast_bit_jaccard_u64 pair {i}: len must be divisible by 8"
        );
        let actual_u64 = fast_bit_jaccard_u64(&x, &y);
        tolerance.assert_close(
            actual_u64,
            exp_u64,
            &format!("fast_bit_jaccard_u64 pair {i}"),
        );
    }
}
