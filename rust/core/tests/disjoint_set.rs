//! Fixture parity tests for `uvoc_core::disjoint_set`.

mod common;

use common::load;
use uvoc_core::disjoint_set::{
    ds_find_rank, ds_find_size, ds_rank_create, ds_size_create, ds_union_by_rank,
    ds_union_by_size,
};

#[test]
fn union_rank_n24() {
    let fixture = load("disjoint_set", "union_rank_n24");
    assert_eq!(fixture["tolerance"], "exact");

    let n = fixture["inputs"]["n"].as_u64().unwrap() as usize;
    let union_ops: Vec<(i32, i32)> = fixture["inputs"]["union_ops"]
        .as_array()
        .unwrap()
        .iter()
        .map(|pair| {
            let pair = pair.as_array().unwrap();
            (pair[0].as_i64().unwrap() as i32, pair[1].as_i64().unwrap() as i32)
        })
        .collect();

    let mut ds = ds_rank_create(n);
    let mut find_after_each_union = Vec::with_capacity(union_ops.len());
    for &(x, y) in &union_ops {
        ds_union_by_rank(&mut ds, x, y);
        find_after_each_union.push(ds_find_rank(&mut ds, x));
    }

    let expected_find: Vec<i32> = fixture["expected"]["find_after_each_union"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    assert_eq!(find_after_each_union, expected_find, "find_after_each_union");

    let expected_parent: Vec<i32> = fixture["expected"]["final_parent"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    assert_eq!(ds.parent, expected_parent, "final_parent");

    let expected_rank: Vec<i32> = fixture["expected"]["final_rank"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    assert_eq!(ds.rank, expected_rank, "final_rank");
}

#[test]
fn union_size_n24() {
    let fixture = load("disjoint_set", "union_size_n24");
    assert_eq!(fixture["tolerance"], "exact");

    let n = fixture["inputs"]["n"].as_u64().unwrap() as usize;
    let union_ops: Vec<(i32, i32)> = fixture["inputs"]["union_ops"]
        .as_array()
        .unwrap()
        .iter()
        .map(|pair| {
            let pair = pair.as_array().unwrap();
            (pair[0].as_i64().unwrap() as i32, pair[1].as_i64().unwrap() as i32)
        })
        .collect();

    let mut ds = ds_size_create(n);
    let mut find_after_each_union = Vec::with_capacity(union_ops.len());
    for &(x, y) in &union_ops {
        ds_union_by_size(&mut ds, x, y);
        find_after_each_union.push(ds_find_size(&mut ds, x));
    }

    let expected_find: Vec<i32> = fixture["expected"]["find_after_each_union"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    assert_eq!(find_after_each_union, expected_find, "find_after_each_union");

    let expected_parent: Vec<i32> = fixture["expected"]["final_parent"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    assert_eq!(ds.parent, expected_parent, "final_parent");

    let expected_size: Vec<i32> = fixture["expected"]["final_size"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    assert_eq!(ds.size, expected_size, "final_size");
}
