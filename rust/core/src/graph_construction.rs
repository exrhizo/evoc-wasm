//! Port of `evoc/graph_construction.py`: `smooth_knn_dist`,
//! `compute_membership_strengths`, `neighbor_graph_matrix`.
//!
//! The reference builds the graph with scipy sparse (`coo_array` →
//! `.tocsr()`); we have no scipy, so the sparse algebra
//! (COO→CSR canonicalization with duplicate summation, `eliminate_zeros`,
//! transpose, elementwise multiply, add/sub) is implemented here. The CSR
//! result matches scipy's `.tocsr().sort_indices()` canonical form
//! (indices sorted ascending within each row, duplicates summed).
//!
//! PORT NOTE: every `numba.prange` is a plain sequential loop
//! (single-threaded wasm32 target); the reductions here are per-row
//! independent so order is irrelevant.

const SMOOTH_K_TOLERANCE: f32 = 1e-5;
const MIN_K_DIST_SCALE: f32 = 1e-3;

/// Compressed sparse row matrix (square, n x n).
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix {
    pub n: usize,
    pub indptr: Vec<i32>,
    pub indices: Vec<i32>,
    pub data: Vec<f32>,
}

/// Port of `smooth_knn_dist`. `distances` is flat row-major
/// n_samples x n_neighbors. Returns `(sigmas, rhos)`.
///
/// PORT NOTE: numba `locals` pin `psum/lo/mid/hi` to f32 despite
/// `fastmath=True` -- accumulate in f32. While `hi == inf` the miss branch
/// DOUBLES `mid` rather than bisecting. `mean_distances` is a single mean
/// over the whole matrix (used only in the `rho == 0` fallback);
/// `mean_ith_distances` is per row. `rho[i]` is set only if the row has a
/// nonzero distance. The psum loop starts at j = 1 (index 0 is self/rho).
pub fn smooth_knn_dist(
    distances: &[f32],
    n_neighbors: usize,
    k: f32,
    n_iter: usize,
    bandwidth: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n_samples = distances.len() / n_neighbors;
    let target = k.log2() * bandwidth;
    let mut rho = vec![0.0f32; n_samples];
    let mut sigma = vec![0.0f32; n_samples];

    // np.mean over the whole matrix (f64 reduction then used as f32).
    let mut sum_all = 0.0f64;
    for &d in distances.iter() {
        sum_all += d as f64;
    }
    let mean_distances = (sum_all / distances.len() as f64) as f32;

    for i in 0..n_samples {
        let mut lo: f32 = 0.0;
        let mut hi: f32 = f32::INFINITY;
        let mut mid: f32 = 1.0;

        let ith = &distances[i * n_neighbors..(i + 1) * n_neighbors];

        // non_zero_dists = ith[ith > 0]; rho = first nonzero if any.
        // (rows are sorted ascending, but the reference does not rely on
        //  that -- it takes the first positive entry in row order.)
        for &d in ith.iter() {
            if d > 0.0 {
                rho[i] = d;
                break;
            }
        }

        for _n in 0..n_iter {
            let mut psum: f32 = 0.0;
            for j in 1..n_neighbors {
                let d = distances[i * n_neighbors + j] - rho[i];
                if d > 0.0 {
                    psum += (-(d / mid)).exp();
                } else {
                    psum += 1.0;
                }
            }

            if (psum - target).abs() < SMOOTH_K_TOLERANCE {
                break;
            }

            if psum > target {
                hi = mid;
                mid = (lo + hi) / 2.0;
            } else {
                lo = mid;
                if hi == f32::INFINITY {
                    mid *= 2.0;
                } else {
                    mid = (lo + hi) / 2.0;
                }
            }
        }

        sigma[i] = mid;

        if rho[i] > 0.0 {
            // np.mean over the whole row
            let mut s = 0.0f64;
            for &d in ith.iter() {
                s += d as f64;
            }
            let mean_ith_distances = (s / n_neighbors as f64) as f32;
            if sigma[i] < MIN_K_DIST_SCALE * mean_ith_distances {
                sigma[i] = MIN_K_DIST_SCALE * mean_ith_distances;
            }
        } else if sigma[i] < MIN_K_DIST_SCALE * mean_distances {
            sigma[i] = MIN_K_DIST_SCALE * mean_distances;
        }
    }

    (sigma, rho)
}

/// Port of `compute_membership_strengths`. Returns flat `(rows, cols, vals)`
/// each of length n_samples * n_neighbors; `idx == -1` slots stay zeroed
/// (row/col/val 0), matching the reference's pre-zeroed COO arrays.
///
/// PORT NOTE: the branch order is load-bearing (per PORT_NOTES): `idx == -1`
/// skips the slot entirely (no write); then `idx == i` => val 0; then
/// `(dist - rho) <= 0 or sigma == 0` => val 1; else the exp formula.
pub fn compute_membership_strengths(
    knn_indices: &[i32],
    knn_dists: &[f32],
    sigmas: &[f32],
    rhos: &[f32],
    n_neighbors: usize,
) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let n_samples = knn_indices.len() / n_neighbors;

    let mut rows = vec![0i32; knn_indices.len()];
    let mut cols = vec![0i32; knn_indices.len()];
    let mut vals = vec![0.0f32; knn_indices.len()];

    for i in 0..n_samples {
        let rho = rhos[i];
        let sigma = sigmas[i];
        for j in 0..n_neighbors {
            let idx = knn_indices[i * n_neighbors + j];
            let val: f32;
            if idx == -1 {
                continue; // We didn't get the full knn for i
            } else if idx == i as i32 {
                val = 0.0;
            } else if (knn_dists[i * n_neighbors + j] - rho) <= 0.0 || sigma == 0.0 {
                val = 1.0;
            } else {
                val = (-((knn_dists[i * n_neighbors + j] - rhos[i]) / sigma)).exp();
            }

            rows[i * n_neighbors + j] = i as i32;
            cols[i * n_neighbors + j] = idx;
            vals[i * n_neighbors + j] = val;
        }
    }

    (rows, cols, vals)
}

/// Build a canonical CSR from COO triplets: duplicates within a row are
/// summed and column indices sorted ascending (scipy `.tocsr()` semantics).
/// Explicit zeros in the input are carried through here (they are dropped by
/// [`eliminate_zeros`]).
fn coo_to_csr(n: usize, rows: &[i32], cols: &[i32], vals: &[f32]) -> CsrMatrix {
    let mut indptr = vec![0i32; n + 1];
    for &r in rows.iter() {
        indptr[r as usize + 1] += 1;
    }
    for i in 0..n {
        indptr[i + 1] += indptr[i];
    }

    let nnz = rows.len();
    let mut indices = vec![0i32; nnz];
    let mut data = vec![0.0f32; nnz];
    // fill in COO order using a moving per-row cursor
    let mut next: Vec<i32> = indptr[..n].to_vec();
    for k in 0..nnz {
        let r = rows[k] as usize;
        let dest = next[r] as usize;
        indices[dest] = cols[k];
        data[dest] = vals[k];
        next[r] += 1;
    }

    // sort each row by column index and sum duplicates
    canonicalize(n, indptr, indices, data)
}

/// Sort each row's entries by column and sum duplicate columns
/// (scipy `sum_duplicates` + `sort_indices`).
fn canonicalize(n: usize, indptr: Vec<i32>, indices: Vec<i32>, data: Vec<f32>) -> CsrMatrix {
    let mut out_indptr = vec![0i32; n + 1];
    let mut out_indices: Vec<i32> = Vec::with_capacity(indices.len());
    let mut out_data: Vec<f32> = Vec::with_capacity(data.len());

    for i in 0..n {
        let start = indptr[i] as usize;
        let end = indptr[i + 1] as usize;
        let mut row: Vec<(i32, f32)> = (start..end).map(|k| (indices[k], data[k])).collect();
        row.sort_by_key(|&(c, _)| c);

        let mut j = 0;
        while j < row.len() {
            let col = row[j].0;
            let mut acc = row[j].1;
            let mut k = j + 1;
            while k < row.len() && row[k].0 == col {
                acc += row[k].1;
                k += 1;
            }
            out_indices.push(col);
            out_data.push(acc);
            j = k;
        }
        out_indptr[i + 1] = out_indices.len() as i32;
    }

    CsrMatrix {
        n,
        indptr: out_indptr,
        indices: out_indices,
        data: out_data,
    }
}

/// Port of scipy `eliminate_zeros`: drop entries whose value is exactly 0.
pub fn eliminate_zeros(m: &CsrMatrix) -> CsrMatrix {
    let mut indptr = vec![0i32; m.n + 1];
    let mut indices: Vec<i32> = Vec::with_capacity(m.indices.len());
    let mut data: Vec<f32> = Vec::with_capacity(m.data.len());

    for i in 0..m.n {
        let start = m.indptr[i] as usize;
        let end = m.indptr[i + 1] as usize;
        for k in start..end {
            if m.data[k] != 0.0 {
                indices.push(m.indices[k]);
                data.push(m.data[k]);
            }
        }
        indptr[i + 1] = indices.len() as i32;
    }

    CsrMatrix {
        n: m.n,
        indptr,
        indices,
        data,
    }
}

/// Transpose a canonical CSR (yielding a canonical CSR of the transpose,
/// i.e. columns become rows, entries sorted ascending per new row).
fn transpose(m: &CsrMatrix) -> CsrMatrix {
    let n = m.n;
    let mut indptr = vec![0i32; n + 1];
    for &c in m.indices.iter() {
        indptr[c as usize + 1] += 1;
    }
    for i in 0..n {
        indptr[i + 1] += indptr[i];
    }
    let nnz = m.indices.len();
    let mut indices = vec![0i32; nnz];
    let mut data = vec![0.0f32; nnz];
    let mut next: Vec<i32> = indptr[..n].to_vec();
    // iterate rows in order so that within each new row (old column) the
    // entries are appended in ascending old-row order -> already sorted.
    for i in 0..n {
        for k in m.indptr[i] as usize..m.indptr[i + 1] as usize {
            let c = m.indices[k] as usize;
            let dest = next[c] as usize;
            indices[dest] = i as i32;
            data[dest] = m.data[k];
            next[c] += 1;
        }
    }
    CsrMatrix {
        n,
        indptr,
        indices,
        data,
    }
}

/// Combine two canonical CSR matrices row-by-row with a binary op applied
/// over the union of column positions (missing entries treated as 0). Used
/// for `A + Aᵀ` and, with the intersection-only product, is handled
/// separately.
fn combine<F: Fn(f32, f32) -> f32>(a: &CsrMatrix, b: &CsrMatrix, op: F) -> CsrMatrix {
    assert_eq!(a.n, b.n);
    let n = a.n;
    let mut indptr = vec![0i32; n + 1];
    let mut indices: Vec<i32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();

    for i in 0..n {
        let (mut ka, ea) = (a.indptr[i] as usize, a.indptr[i + 1] as usize);
        let (mut kb, eb) = (b.indptr[i] as usize, b.indptr[i + 1] as usize);
        while ka < ea || kb < eb {
            let ca = if ka < ea { a.indices[ka] } else { i32::MAX };
            let cb = if kb < eb { b.indices[kb] } else { i32::MAX };
            let (col, av, bv) = if ca < cb {
                let v = (ca, a.data[ka], 0.0);
                ka += 1;
                v
            } else if cb < ca {
                let v = (cb, 0.0, b.data[kb]);
                kb += 1;
                v
            } else {
                let v = (ca, a.data[ka], b.data[kb]);
                ka += 1;
                kb += 1;
                v
            };
            indices.push(col);
            data.push(op(av, bv));
        }
        indptr[i + 1] = indices.len() as i32;
    }

    CsrMatrix {
        n,
        indptr,
        indices,
        data,
    }
}

/// Elementwise product over the INTERSECTION of nonzero positions (scipy
/// `A.multiply(B)`): entries present in only one operand yield no output.
fn multiply(a: &CsrMatrix, b: &CsrMatrix) -> CsrMatrix {
    assert_eq!(a.n, b.n);
    let n = a.n;
    let mut indptr = vec![0i32; n + 1];
    let mut indices: Vec<i32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();

    for i in 0..n {
        let (mut ka, ea) = (a.indptr[i] as usize, a.indptr[i + 1] as usize);
        let (mut kb, eb) = (b.indptr[i] as usize, b.indptr[i + 1] as usize);
        while ka < ea && kb < eb {
            let ca = a.indices[ka];
            let cb = b.indices[kb];
            match ca.cmp(&cb) {
                std::cmp::Ordering::Less => ka += 1,
                std::cmp::Ordering::Greater => kb += 1,
                std::cmp::Ordering::Equal => {
                    indices.push(ca);
                    data.push(a.data[ka] * b.data[kb]);
                    ka += 1;
                    kb += 1;
                }
            }
        }
        indptr[i + 1] = indices.len() as i32;
    }

    CsrMatrix {
        n,
        indptr,
        indices,
        data,
    }
}

/// Port of `neighbor_graph_matrix`. `knn_indices`/`knn_dists` are flat
/// row-major n_samples x n_neighbors. `n_neighbors_f` is the FLOAT effective
/// neighbor count used for the kernel width (per the reference signature).
/// Returns a canonical CSR (scipy `.tocsr().sort_indices()` form).
pub fn neighbor_graph_matrix(
    n_neighbors_f: f32,
    knn_indices: &[i32],
    knn_dists: &[f32],
    n_neighbors: usize,
    symmetrize: bool,
) -> CsrMatrix {
    let (sigmas, rhos) = smooth_knn_dist(knn_dists, n_neighbors, n_neighbors_f, 64, 1.0);

    let (rows, cols, vals) =
        compute_membership_strengths(knn_indices, knn_dists, &sigmas, &rhos, n_neighbors);

    let n = knn_indices.len() / n_neighbors;
    let result = coo_to_csr(n, &rows, &cols, &vals);
    let result = eliminate_zeros(&result);

    let result = if symmetrize {
        let t = transpose(&result);
        let prod = multiply(&result, &t);
        // A + Aᵀ − A∘Aᵀ  (fuzzy-set union)
        let sum = combine(&result, &t, |x, y| x + y);
        combine(&sum, &prod, |x, y| x - y)
    } else {
        result
    };

    eliminate_zeros(&result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smooth_knn_empty_distances_row() {
        // single sample, single neighbor slot that is the self (distance 0):
        // no nonzero dists => rho stays 0, sigma falls back on mean branch.
        let dist = [0.0f32];
        let (sigmas, rhos) = smooth_knn_dist(&dist, 1, 1.0, 64, 1.0);
        assert_eq!(rhos, vec![0.0]);
        assert_eq!(sigmas.len(), 1);
    }

    #[test]
    fn membership_skips_minus_one_and_self() {
        // row 0: [-1, 0, 2] with i=0 -> slot0 skipped (zeroed), slot1 self,
        // slot2 normal.
        let knn_indices = vec![-1, 0, 2, /*row1*/ 0, 1, 2];
        let knn_dists = vec![0.0, 0.0, 0.5, 0.1, 0.0, 0.4];
        let sigmas = vec![0.3f32, 0.3];
        let rhos = vec![0.1f32, 0.1];
        let (rows, cols, vals) =
            compute_membership_strengths(&knn_indices, &knn_dists, &sigmas, &rhos, 3);
        // slot 0 stays fully zeroed
        assert_eq!((rows[0], cols[0], vals[0]), (0, 0, 0.0));
        // slot 1: idx == i => val 0
        assert_eq!((rows[1], cols[1], vals[1]), (0, 0, 0.0));
        // slot 2: (0.5 - 0.1) > 0, sigma != 0 => exp formula, in (0,1)
        assert_eq!((rows[2], cols[2]), (0, 2));
        assert!(vals[2] > 0.0 && vals[2] < 1.0);
    }

    #[test]
    fn eliminate_zeros_drops_explicit_zeros() {
        let m = CsrMatrix {
            n: 2,
            indptr: vec![0, 2, 3],
            indices: vec![0, 1, 1],
            data: vec![0.0, 0.5, 0.0],
        };
        let e = eliminate_zeros(&m);
        assert_eq!(e.indptr, vec![0, 1, 1]);
        assert_eq!(e.indices, vec![1]);
        assert_eq!(e.data, vec![0.5]);
    }

    #[test]
    fn symmetrize_fuzzy_union_two_nodes() {
        // A: 0->1 weight 0.5 only (directed). Aᵀ: 1->0 weight 0.5.
        // union = A + Aᵀ − A∘Aᵀ; no shared position so product empty:
        // result has 0->1 = 0.5 and 1->0 = 0.5.
        let knn_indices = vec![1, /*self pad*/ 0, /*row1*/ 1, 0];
        // row0 neighbor0 -> col1 dist small; slot1 self -> val 0 (dropped)
        // row1 neighbor0 -> self (dropped); neighbor1 -> col0
        let knn_dists = vec![0.2, 0.0, 0.0, 0.2];
        let sigmas = vec![0.5f32, 0.5];
        let rhos = vec![0.1f32, 0.1];
        let n_neighbors = 2;
        let (rows, cols, vals) =
            compute_membership_strengths(&knn_indices, &knn_dists, &sigmas, &rhos, n_neighbors);
        let m = eliminate_zeros(&coo_to_csr(2, &rows, &cols, &vals));
        let t = transpose(&m);
        let prod = multiply(&m, &t);
        let sum = combine(&m, &t, |x, y| x + y);
        let sym = eliminate_zeros(&combine(&sum, &prod, |x, y| x - y));
        // symmetric: entry (0,1) == entry (1,0)
        assert_eq!(sym.indptr, vec![0, 1, 2]);
        assert_eq!(sym.indices, vec![1, 0]);
        assert!((sym.data[0] - sym.data[1]).abs() < 1e-6);
    }

    #[test]
    fn transpose_is_involution_on_canonical() {
        let m = CsrMatrix {
            n: 3,
            indptr: vec![0, 1, 2, 3],
            indices: vec![1, 2, 0],
            data: vec![1.0, 2.0, 3.0],
        };
        let tt = transpose(&transpose(&m));
        assert_eq!(tt, m);
    }
}
