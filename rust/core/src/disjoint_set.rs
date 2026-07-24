//! Port of `evoc/disjoint_set.py`.
//!
//! Union-find with either union-by-rank or union-by-size, index-parallel
//! arrays of i32 exactly as the Python/numba reference uses (no widening to
//! usize, per PORTING.md).

/// Port of `RankDisjointSet` (namedtuple of `parent`, `rank`).
pub struct RankDisjointSet {
    pub parent: Vec<i32>,
    pub rank: Vec<i32>,
}

/// Port of `SizeDisjointSet` (namedtuple of `parent`, `size`).
pub struct SizeDisjointSet {
    pub parent: Vec<i32>,
    pub size: Vec<i32>,
}

/// Port of `ds_rank_create`.
pub fn ds_rank_create(n_elements: usize) -> RankDisjointSet {
    RankDisjointSet {
        parent: (0..n_elements as i32).collect(),
        rank: vec![0; n_elements],
    }
}

/// Port of `ds_size_create`.
pub fn ds_size_create(n_elements: usize) -> SizeDisjointSet {
    SizeDisjointSet {
        parent: (0..n_elements as i32).collect(),
        size: vec![1; n_elements],
    }
}

// PORT NOTE: the Python body is a simultaneous assignment:
//   x, parent[x] = parent[x], parent[parent[x]]
// Python evaluates the whole RHS tuple first using the OLD x:
//   rhs0 = parent[x]                  # == new x
//   rhs1 = parent[parent[x]]          # == parent[new x]  (parent[old x] == new x)
// then performs the assignment targets left to right: `x` is rebound to
// rhs0 FIRST, and only then is `parent[x]` evaluated as a store target --
// using the NEW x. So the store is `parent[new_x] = parent[new_x]`: since
// rhs1 == parent[rhs0] == parent[new_x], the write always stores back the
// value that is already there. This holds for ANY tree-shaped parent array
// (proven by substitution: store index == store value's source index), so
// despite appearances this loop does no path compression at all -- it is
// observably equivalent to `while parent[x] != x: x = parent[x]`. Verified
// both algebraically and by fuzzing against the reference semantics, and
// the `final_parent` / `final_rank` / `final_size` fixtures (which would
// differ from a plain "no path compression" union-find if any compression
// were actually happening) are consistent with this reading.
fn ds_find(parent: &mut [i32], x: i32) -> i32 {
    let mut x = x;
    while parent[x as usize] != x {
        let rhs0 = parent[x as usize];
        let rhs1 = parent[parent[x as usize] as usize];
        x = rhs0;
        parent[x as usize] = rhs1;
    }
    x
}

/// Port of `ds_find` for a `RankDisjointSet`.
pub fn ds_find_rank(disjoint_set: &mut RankDisjointSet, x: i32) -> i32 {
    ds_find(&mut disjoint_set.parent, x)
}

/// Port of `ds_find` for a `SizeDisjointSet`.
pub fn ds_find_size(disjoint_set: &mut SizeDisjointSet, x: i32) -> i32 {
    ds_find(&mut disjoint_set.parent, x)
}

/// Port of `ds_union_by_rank`.
pub fn ds_union_by_rank(disjoint_set: &mut RankDisjointSet, x: i32, y: i32) {
    let mut x = ds_find_rank(disjoint_set, x);
    let mut y = ds_find_rank(disjoint_set, y);

    if x == y {
        return;
    }

    if disjoint_set.rank[x as usize] < disjoint_set.rank[y as usize] {
        std::mem::swap(&mut x, &mut y);
    }

    disjoint_set.parent[y as usize] = x;
    if disjoint_set.rank[x as usize] == disjoint_set.rank[y as usize] {
        disjoint_set.rank[x as usize] += 1;
    }
}

/// Port of `ds_union_by_size`.
pub fn ds_union_by_size(disjoint_set: &mut SizeDisjointSet, x: i32, y: i32) {
    let mut x = ds_find_size(disjoint_set, x);
    let mut y = ds_find_size(disjoint_set, y);

    if x == y {
        return;
    }

    if disjoint_set.size[x as usize] < disjoint_set.size[y as usize] {
        std::mem::swap(&mut x, &mut y);
    }

    disjoint_set.parent[y as usize] = x;
    disjoint_set.size[x as usize] += disjoint_set.size[y as usize];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_create_single_element() {
        let ds = ds_rank_create(1);
        assert_eq!(ds.parent, vec![0]);
        assert_eq!(ds.rank, vec![0]);
    }

    #[test]
    fn size_create_single_element() {
        let ds = ds_size_create(1);
        assert_eq!(ds.parent, vec![0]);
        assert_eq!(ds.size, vec![1]);
    }

    #[test]
    fn find_on_singleton_is_self() {
        let mut ds = ds_rank_create(1);
        assert_eq!(ds_find_rank(&mut ds, 0), 0);
    }

    #[test]
    fn union_self_is_noop_by_rank() {
        let mut ds = ds_rank_create(4);
        ds_union_by_rank(&mut ds, 2, 2);
        assert_eq!(ds.parent, vec![0, 1, 2, 3]);
        assert_eq!(ds.rank, vec![0, 0, 0, 0]);
    }

    #[test]
    fn union_self_is_noop_by_size() {
        let mut ds = ds_size_create(4);
        ds_union_by_size(&mut ds, 2, 2);
        assert_eq!(ds.parent, vec![0, 1, 2, 3]);
        assert_eq!(ds.size, vec![1, 1, 1, 1]);
    }

    #[test]
    fn union_by_rank_tie_increments_rank() {
        let mut ds = ds_rank_create(2);
        ds_union_by_rank(&mut ds, 0, 1);
        // tie: rank[0] == rank[1] == 0, so 0 becomes root (x==y is false;
        // no swap since rank[x] < rank[y] is false), rank[x] increments.
        assert_eq!(ds.parent, vec![0, 0]);
        assert_eq!(ds.rank, vec![1, 0]);
    }

    #[test]
    fn union_by_size_accumulates_size() {
        let mut ds = ds_size_create(3);
        ds_union_by_size(&mut ds, 0, 1);
        ds_union_by_size(&mut ds, 0, 2);
        assert_eq!(ds_find_size(&mut ds, 2), 0);
        assert_eq!(ds.size[0], 3);
    }

    #[test]
    fn find_on_chain_does_not_compress_path() {
        // parent chain 3 -> 2 -> 1 -> 0 (root). Despite the simultaneous
        // assignment in the reference looking like path halving, it is a
        // provable no-op (see PORT NOTE above ds_find) -- parent is
        // unchanged after find.
        let mut parent = vec![0, 0, 1, 2];
        let root = ds_find(&mut parent, 3);
        assert_eq!(root, 0);
        assert_eq!(parent, vec![0, 0, 1, 2]);
    }
}
