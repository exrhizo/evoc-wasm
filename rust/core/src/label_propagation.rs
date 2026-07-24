//! Port of `evoc/label_propagation.py`: graph-based label propagation used
//! both as evoc's core outlier-relabeling utility and (via
//! `label_propagation_init`) as a high-quality node-embedding initializer.
//!
//! PORT NOTE: every `numba.prange` here becomes a plain sequential loop
//! (single-threaded wasm32), matching the convention used throughout this
//! port (see node_embedding.rs). This matters semantically for
//! `label_outliers`, whose loop body mutates the shared `labels` array in
//! place -- later iterations can observe earlier iterations' writes, and a
//! literal sequential-loop translation reproduces that (numba's `prange`
//! would race on this in a real parallel build, but the reference
//! implementation is treated as the ground truth regardless).
//!
//! PORT NOTE: `local_rng_state = rng_state + i` in the Python source is a
//! numpy elementwise add producing a brand new length-3 array per row `i`
//! (not a shared/advancing state) -- ported here as building a fresh
//! `RngState` per row from the base state offset by `i as i64`.
//!
//! PORT NOTE: the `votes` dict in `label_prop_iteration`/
//! `original_label_prop_iteration` relies on Python dict insertion-order
//! iteration for its tie-break semantics. Ported here as a `Vec<(i64, f32)>`
//! scanned linearly (first-seen order preserved, matching dict semantics for
//! small vote counts).
//!
//! PORT NOTE: `label_outliers`'s `node_queue.pop()` pops from the end of the
//! Python list (LIFO), ported here as `Vec::pop`.
//!
//! PORT NOTE: `tau_rand_int(...) % (max_label + 1)` and `... % n_vertices`
//! use Python's floor-mod semantics (always non-negative for a positive
//! divisor), ported via `rem_euclid`.

use crate::node_embedding::node_embedding_repr;
use crate::rng::{tau_rand, tau_rand_int, RngState};

/// Build the fresh per-row RNG state `rng_state + i` (elementwise), matching
/// `local_rng_state = rng_state + i` in the Python source.
fn local_rng_state(rng_state: &RngState, i: i64) -> RngState {
    [rng_state[0] + i, rng_state[1] + i, rng_state[2] + i]
}

/// Port of `label_prop_iteration`. `indptr`/`indices` describe the CSR graph
/// (`indptr.len() == n_rows + 1`); `labels[i] < 0` marks an unlabeled node.
pub fn label_prop_iteration(
    indptr: &[i64],
    indices: &[i64],
    data: &[f32],
    labels: &[i64],
    rng_state: &RngState,
) -> Vec<i64> {
    let n_rows = indptr.len() - 1;
    let mut result = labels.to_vec();

    for i in 0..n_rows {
        let current_l = labels[i];
        if current_l >= 0 {
            continue;
        }
        let mut local_state = local_rng_state(rng_state, i as i64);

        // votes: Vec<(label, accumulated_weight)>, first-seen order preserved
        // (matches Python dict insertion order).
        let mut votes: Vec<(i64, f32)> = Vec::new();
        for k in indptr[i] as usize..indptr[i + 1] as usize {
            let j = indices[k] as usize;
            let l = labels[j];
            if let Some(entry) = votes.iter_mut().find(|(vl, _)| *vl == l) {
                entry.1 += data[k];
            } else {
                votes.push((l, data[k]));
            }
        }

        let mut max_vote: f32 = 1.0;
        let mut tie_count: i64 = 1;
        for &(l, v) in votes.iter() {
            if l == -1 {
                continue;
            } else if v > max_vote {
                max_vote = v;
                result[i] = l;
                tie_count = 1;
            } else if v == max_vote {
                tie_count += 1;
                if current_l == -1 {
                    result[i] = l;
                } else if tau_rand(&mut local_state) < 1.0 / (tie_count as f32) {
                    result[i] = l;
                }
            }
        }
    }

    result
}

/// Port of `original_label_prop_iteration`: identical to
/// [`label_prop_iteration`] except it does NOT skip already-labeled rows
/// (every row is re-voted on every iteration).
pub fn original_label_prop_iteration(
    indptr: &[i64],
    indices: &[i64],
    data: &[f32],
    labels: &[i64],
    rng_state: &RngState,
) -> Vec<i64> {
    let n_rows = indptr.len() - 1;
    let mut result = labels.to_vec();

    for i in 0..n_rows {
        let current_l = labels[i];
        let mut local_state = local_rng_state(rng_state, i as i64);

        let mut votes: Vec<(i64, f32)> = Vec::new();
        for k in indptr[i] as usize..indptr[i + 1] as usize {
            let j = indices[k] as usize;
            let l = labels[j];
            if let Some(entry) = votes.iter_mut().find(|(vl, _)| *vl == l) {
                entry.1 += data[k];
            } else {
                votes.push((l, data[k]));
            }
        }

        let mut max_vote: f32 = 1.0;
        let mut tie_count: i64 = 1;
        for &(l, v) in votes.iter() {
            if l == -1 {
                continue;
            } else if v > max_vote {
                max_vote = v;
                result[i] = l;
                tie_count = 1;
            } else if v == max_vote {
                tie_count += 1;
                if current_l == -1 {
                    result[i] = l;
                } else if tau_rand(&mut local_state) < 1.0 / (tie_count as f32) {
                    result[i] = l;
                }
            }
        }
    }

    result
}

/// Port of `label_outliers`. Mutates `labels` in place (matching the Python
/// in-place mutation semantics, including later rows observing earlier
/// rows' resolutions) and also returns it for convenience.
pub fn label_outliers<'a>(
    indptr: &[i64],
    indices: &[i64],
    labels: &'a mut [i64],
    rng_state: &RngState,
) -> &'a mut [i64] {
    let n_rows = indptr.len() - 1;
    let max_label = *labels.iter().max().unwrap();

    for i in 0..n_rows {
        let mut local_state = local_rng_state(rng_state, i as i64);
        if labels[i] < 0 {
            let mut node_queue: Vec<i64> = vec![i as i64];
            let mut unlabelled = true;
            let mut n_iter = 0;

            while unlabelled && n_iter < 64 && !node_queue.is_empty() {
                n_iter += 1;
                let current_node = node_queue.pop().unwrap() as usize;
                for k in indptr[current_node] as usize..indptr[current_node + 1] as usize {
                    let j = indices[k] as usize;
                    if labels[j] >= 0 {
                        labels[i] = labels[j];
                        unlabelled = false;
                        break;
                    } else {
                        node_queue.push(j as i64);
                    }
                }
            }

            if n_iter >= 64 || unlabelled {
                let r = tau_rand_int(&mut local_state) as i64;
                labels[i] = r.rem_euclid(max_label + 1);
            }
        }
    }

    labels
}

/// Port of `remap_labels`. Mutates `labels` in place and also returns it.
///
/// PORT NOTE: unlabeled (`< 0`) entries do NOT collapse to a single label --
/// each becomes its own fresh singleton cluster, assigned in array-index
/// order starting from `next_label = number of distinct non-negative labels`.
pub fn remap_labels(labels: &mut [i64]) -> &mut [i64] {
    let mut unique_labels: Vec<i64> = labels.to_vec();
    unique_labels.sort_unstable();
    unique_labels.dedup();
    if unique_labels.first() == Some(&-1) {
        unique_labels.remove(0);
    }

    let mapping: std::collections::HashMap<i64, i64> = unique_labels
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, i as i64))
        .collect();
    let mut next_label = unique_labels.len() as i64;

    for l in labels.iter_mut() {
        if *l < 0 {
            *l = next_label;
            next_label += 1;
        } else {
            let mapped = mapping[&*l];
            *l = mapped;
        }
    }

    labels
}

/// Port of `label_prop_loop`. `seed_position_draws` and `rng_state` replace
/// the Python `random_state`-driven draws (recorded verbatim by the
/// fixtures, per PORTING.md); `labels` is the full-length initial labels
/// array (typically all `-1`).
pub fn label_prop_loop(
    indptr: &[i64],
    indices: &[i64],
    data: &[f32],
    mut labels: Vec<i64>,
    rng_state: &RngState,
    seed_position_draws: &[usize],
    n_iter: usize,
) -> Vec<i64> {
    for (i, &pos) in seed_position_draws.iter().enumerate() {
        labels[pos] = i as i64;
    }

    for _ in 0..n_iter {
        labels = label_prop_iteration(indptr, indices, data, &labels, rng_state);
    }

    label_outliers(indptr, indices, &mut labels, rng_state);
    remap_labels(&mut labels);
    labels
}

/// Port of `original_label_prop_loop`.
pub fn original_label_prop_loop(
    indptr: &[i64],
    indices: &[i64],
    data: &[f32],
    mut labels: Vec<i64>,
    rng_state: &RngState,
    seed_position_draws: &[usize],
    n_iter: usize,
) -> Vec<i64> {
    for (i, &pos) in seed_position_draws.iter().enumerate() {
        labels[pos] = i as i64;
    }

    for _ in 0..n_iter {
        labels = original_label_prop_iteration(indptr, indices, data, &labels, rng_state);
    }

    label_outliers(indptr, indices, &mut labels, rng_state);
    remap_labels(&mut labels);
    labels
}

/// Sparse-sparse CSR matrix multiply replicating scipy's SMMP (Gustavson)
/// algorithm (`scipy/sparse/sparsetools/csr.h`'s `csr_matmat_pass2`)
/// exactly, INCLUDING its column emission order: new columns are prepended
/// to a per-row singly-linked list (`next[k] = head; head = k`) as they are
/// first touched, then emitted head-first -- i.e. REVERSE of first-touch
/// order, not sorted. This order is load-bearing: it becomes the CSR
/// `raw_index` fed into `node_embedding_repr`'s negative-sample index
/// arithmetic, which is bit-for-bit sensitive to it. Verified against a
/// direct scipy replay of `label_propagation_init`'s `reduced_graph`
/// computation (see PR history / porting notes) to match scipy's actual
/// output index-for-index.
fn smmp(
    a_indptr: &[usize],
    a_indices: &[usize],
    a_data: &[f64],
    b_indptr: &[usize],
    b_indices: &[usize],
    b_data: &[f64],
    n_row: usize,
    n_col: usize,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut next: Vec<i64> = vec![-1; n_col];
    let mut sums: Vec<f64> = vec![0.0; n_col];
    let mut c_indptr: Vec<usize> = Vec::with_capacity(n_row + 1);
    let mut c_indices: Vec<usize> = Vec::new();
    let mut c_data: Vec<f64> = Vec::new();
    c_indptr.push(0);

    for i in 0..n_row {
        let mut head: i64 = -2;
        let mut length = 0usize;

        for jj in a_indptr[i]..a_indptr[i + 1] {
            let j = a_indices[jj];
            let v = a_data[jj];
            for kk in b_indptr[j]..b_indptr[j + 1] {
                let k = b_indices[kk];
                sums[k] += v * b_data[kk];
                if next[k] == -1 {
                    next[k] = head;
                    head = k as i64;
                    length += 1;
                }
            }
        }

        for _ in 0..length {
            let h = head as usize;
            if sums[h] != 0.0 {
                c_indices.push(h);
                c_data.push(sums[h]);
            }
            let temp = head;
            head = next[h];
            next[temp as usize] = -1;
            sums[temp as usize] = 0.0;
        }

        c_indptr.push(c_indices.len());
    }

    (c_indptr, c_indices, c_data)
}

/// Port of the non-recursive, `upscaling="partition_expander"` path of
/// `label_propagation_init`, for graphs at/above `base_init_threshold` (the
/// small-graph PCA/random/spectral/mds base case and the recursive-init /
/// `jitter_expander` upscaling path are out of scope -- see PORT_NOTES.md).
///
/// All randomness is supplied explicitly (matching the fixture-recording
/// convention used throughout this port): `lp_rng_state` +
/// `lp_seed_position_draws` drive the inner `label_prop_loop`;
/// `node_embedding_init_normal` (flat, `n_parts * n_components`, row-major)
/// is the reduced graph's node-embedding initial embedding;
/// `node_embedding_rng_val` / `node_embedding_shuffles` drive that node
/// embedding's per-epoch SGD (see `node_embedding.rs`).
#[allow(clippy::too_many_arguments)]
pub fn label_propagation_init(
    indptr: &[i64],
    indices: &[i64],
    data: &[f32],
    n_points: usize,
    n_label_prop_iter: usize,
    n_embedding_epochs: usize,
    n_components: usize,
    scaling: f32,
    noise_level: f32,
    lp_rng_state: &RngState,
    lp_seed_position_draws: &[usize],
    node_embedding_init_normal: &[f32],
    node_embedding_rng_val: &[u32],
    node_embedding_shuffles: &[Vec<u32>],
) -> Vec<f32> {
    // Dense adjacency of the input graph (n_points is small in every
    // fixtured/expected use of this path).
    let mut graph = vec![0.0f64; n_points * n_points];
    for i in 0..n_points {
        for k in indptr[i] as usize..indptr[i + 1] as usize {
            let j = indices[k] as usize;
            graph[i * n_points + j] = data[k] as f64;
        }
    }

    // 1. label-propagation partition.
    let initial_labels = vec![-1i64; n_points];
    let partition = label_prop_loop(
        indptr,
        indices,
        data,
        initial_labels,
        lp_rng_state,
        lp_seed_position_draws,
        n_label_prop_iter,
    );
    let n_parts = (*partition.iter().max().unwrap() + 1) as usize;

    let mut counts = vec![0usize; n_parts];
    for &p in &partition {
        counts[p as usize] += 1;
    }
    let inv_sqrt_count: Vec<f64> = counts.iter().map(|&c| 1.0 / (c as f64).sqrt()).collect();

    // 2. reduced_graph = normalized_reduction_map.T @ graph @ base_reduction_map,
    // clipped to [0, 1]. This is a genuine two-stage sparse-sparse matrix
    // product: the RESULT'S CSR COLUMN ORDER PER ROW IS NOT SORTED -- it
    // depends on scipy's SMMP (Gustavson) multiplication algorithm, which
    // this replicates exactly (see `smmp` below), because that column order
    // becomes `raw_index` in the CSR consumed by `node_embedding_repr`, and
    // `raw_index` feeds directly into the negative-sample index arithmetic
    // there (bit-for-bit sensitive, not just a numeric-value concern).
    let graph_indptr_usize: Vec<usize> = indptr.iter().map(|&x| x as usize).collect();
    let graph_indices_usize: Vec<usize> = indices.iter().map(|&x| x as usize).collect();
    let graph_data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();

    // normalized_reduction_map.T (n_parts x n_points), CSR: row `a` lists
    // points `i` with partition[i]==a in ascending `i` order (matches
    // scipy's CSC->CSR conversion of a transposed one-hot column-normalized
    // matrix), each with value `1/sqrt(count[a])`.
    let mut nrm_t_indptr = vec![0usize; n_parts + 1];
    for &p in &partition {
        nrm_t_indptr[p as usize + 1] += 1;
    }
    for a in 0..n_parts {
        nrm_t_indptr[a + 1] += nrm_t_indptr[a];
    }
    let mut nrm_t_indices = vec![0usize; n_points];
    let mut nrm_t_data = vec![0.0f64; n_points];
    {
        let mut cursor = nrm_t_indptr.clone();
        for i in 0..n_points {
            let p = partition[i] as usize;
            let pos = cursor[p];
            nrm_t_indices[pos] = i;
            nrm_t_data[pos] = inv_sqrt_count[p];
            cursor[p] += 1;
        }
    }

    // base_reduction_map (n_points x n_parts), CSR: row `i` has a single
    // nonzero at column partition[i], value 1.0.
    let br_indptr: Vec<usize> = (0..=n_points).collect();
    let br_indices: Vec<usize> = partition.iter().map(|&p| p as usize).collect();
    let br_data: Vec<f64> = vec![1.0; n_points];

    let (stage1_indptr, stage1_indices, stage1_data) = smmp(
        &nrm_t_indptr,
        &nrm_t_indices,
        &nrm_t_data,
        &graph_indptr_usize,
        &graph_indices_usize,
        &graph_data_f64,
        n_parts,
        n_points,
    );
    let (reduced_indptr_usize, reduced_indices_usize, mut reduced_data_f64) = smmp(
        &stage1_indptr,
        &stage1_indices,
        &stage1_data,
        &br_indptr,
        &br_indices,
        &br_data,
        n_parts,
        n_parts,
    );
    for v in reduced_data_f64.iter_mut() {
        *v = v.clamp(0.0, 1.0);
    }

    let reduced_indptr: Vec<u32> = reduced_indptr_usize.iter().map(|&x| x as u32).collect();
    let reduced_indices: Vec<u32> = reduced_indices_usize.iter().map(|&x| x as u32).collect();
    let reduced_data: Vec<f32> = reduced_data_f64.iter().map(|&x| x as f32).collect();

    // 3. node embedding on the reduced graph.
    let initial_alpha = 0.001 * n_embedding_epochs as f32;
    let reduced_layout = node_embedding_repr(
        &reduced_indptr,
        &reduced_indices,
        &reduced_data,
        n_parts,
        n_components,
        n_embedding_epochs,
        node_embedding_init_normal,
        initial_alpha,
        1.0, // negative_sample_rate default
        noise_level,
        node_embedding_rng_val,
        node_embedding_shuffles,
    );

    // 4. partition_expander upscaling.
    // M[i][j] = graph[i,j] * graph[j,i] (elementwise Hadamard with the
    // transpose, i.e. `graph.multiply(graph.T)`).
    let mut data_expander = vec![0.0f64; n_points * n_parts]; // unnormalized, then l1-row-normalized
    for i in 0..n_points {
        for j in 0..n_points {
            let m_ij = graph[i * n_points + j] * graph[j * n_points + i];
            if m_ij != 0.0 {
                let b = partition[j] as usize;
                data_expander[i * n_parts + b] += m_ij * inv_sqrt_count[b];
            }
        }
        let row_sum: f64 = data_expander[i * n_parts..i * n_parts + n_parts]
            .iter()
            .sum();
        if row_sum != 0.0 {
            for v in data_expander[i * n_parts..i * n_parts + n_parts].iter_mut() {
                *v /= row_sum;
            }
        }
    }

    // result = (data_expander @ reduced_layout + reduced_layout[partition]) / 2
    let mut result = vec![0.0f64; n_points * n_components];
    for i in 0..n_points {
        for c in 0..n_components {
            let mut acc = 0.0f64;
            for b in 0..n_parts {
                acc += data_expander[i * n_parts + b] * reduced_layout[b * n_components + c] as f64;
            }
            let p = partition[i] as usize;
            let direct = reduced_layout[p * n_components + c] as f64;
            result[i * n_components + c] = (acc + direct) / 2.0;
        }
    }

    // result = scaling * (result - result.mean(axis=0))
    let mut mean = vec![0.0f64; n_components];
    for i in 0..n_points {
        for c in 0..n_components {
            mean[c] += result[i * n_components + c];
        }
    }
    for m in mean.iter_mut() {
        *m /= n_points as f64;
    }

    let mut out = vec![0.0f32; n_points * n_components];
    for i in 0..n_points {
        for c in 0..n_components {
            out[i * n_components + c] =
                (scaling as f64 * (result[i * n_components + c] - mean[c])) as f32;
        }
    }

    out
}

/// Self-driven, recursive port of `label_propagation_init` for the
/// `evoc_clusters` pipeline (phase 9). Unlike
/// [`label_propagation_init`] (the fixture-parity, `recursive_init=False`
/// port whose randomness is supplied explicitly and verbatim), this drives
/// every random draw from this project's own `tau_rand` RNG, advancing
/// `state` in place, and recurses per the reference's `recursive_init=True`
/// default: `label_prop_loop` -> build a reduced graph -> recurse (or hit the
/// PCA base case once the reduced graph drops below `base_init_threshold`)
/// -> `node_embedding` on the reduced graph -> `partition_expander`
/// upscaling back to the caller's graph size.
///
/// `approx_n_parts` follows the reference's exact formula
/// (`clip(int(8*sqrt(n)), 256, 16384)` at the top level, shrinking `// 4`
/// each recursion level) -- callers at the top of the pipeline should pass
/// that value in; recursive calls compute it from the reduced graph's own
/// size via the same clip (`clip(int(8*sqrt(n_reduced)), 256, 16384)`,
/// matching `evoc_clusters`' top-level derivation, since the Python
/// reference does not re-derive `approx_n_parts` from graph size at each
/// recursion level -- it shrinks the CALLER's value by `// 4`).
///
/// PORT NOTE (scope): base case below `base_init_threshold` uses randomized
/// PCA (see `pca.rs`) instead of sklearn's `PCA`/`SpectralEmbedding`/`MDS` --
/// bit-exact parity there is impossible (see PORT_NOTES.md / PCA_VALIDATION.md).
/// This whole function is validated by ARI against a real Python
/// `evoc.EVoC` run, not bit-exact fixtures.
#[allow(clippy::too_many_arguments)]
pub fn label_propagation_init_recursive(
    indptr: &[i64],
    indices: &[i64],
    data: &[f32],
    n_points: usize,
    data_full: Option<&[f32]>, // original-space rows for this graph's points, row-major, for PCA base case
    n_features: usize,
    n_label_prop_iter: usize,
    n_embedding_epochs: usize,
    approx_n_parts: usize,
    n_components: usize,
    scaling: f32,
    noise_level: f32,
    recursive_init: bool,
    base_init_threshold: usize,
    state: &mut RngState,
) -> Vec<f32> {
    if n_points < base_init_threshold {
        // PCA base case (randomized SVD, see pca.rs).
        let source: Vec<f32> = match data_full {
            Some(d) => d.to_vec(),
            None => {
                // No original-space data available (should not happen in the
                // evoc_clusters call path, which always threads `data`
                // through) -- fall back to standard-normal random init so the
                // function still returns a usable embedding.
                (0..n_points * n_components)
                    .map(|_| crate::pca::next_standard_normal(state))
                    .collect()
            }
        };
        let n_feat = if data_full.is_some() {
            n_features
        } else {
            n_components
        };
        let mut result = crate::pca::randomized_pca(&source, n_points, n_feat, n_components, state);
        // result -= result.mean(); result /= (result.max() - result.min()) / 2.0
        let mean: f64 = result.iter().map(|&x| x as f64).sum::<f64>() / result.len() as f64;
        let mut max_v = f32::NEG_INFINITY;
        let mut min_v = f32::INFINITY;
        for &v in result.iter() {
            if v > max_v {
                max_v = v;
            }
            if v < min_v {
                min_v = v;
            }
        }
        let range = ((max_v - min_v) as f64 / 2.0).max(1e-12);
        for v in result.iter_mut() {
            *v = ((*v as f64 - mean) / range) as f32;
        }
        return result;
    }

    // 1. label-propagation partition, driven by our own RNG.
    let lp_rng_state: RngState = [
        (tau_rand_int(state) as i64).wrapping_mul(2),
        (tau_rand_int(state) as i64).wrapping_mul(2),
        (tau_rand_int(state) as i64).wrapping_mul(2),
    ];
    let seed_position_draws: Vec<usize> = (0..approx_n_parts)
        .map(|_| (tau_rand_int(state).unsigned_abs() as usize) % n_points)
        .collect();

    let initial_labels = vec![-1i64; n_points];
    let partition = label_prop_loop(
        indptr,
        indices,
        data,
        initial_labels,
        &lp_rng_state,
        &seed_position_draws,
        n_label_prop_iter,
    );
    let n_parts = (*partition.iter().max().unwrap() + 1) as usize;

    let mut counts = vec![0usize; n_parts];
    for &p in &partition {
        counts[p as usize] += 1;
    }
    let inv_sqrt_count: Vec<f64> = counts.iter().map(|&c| 1.0 / (c as f64).sqrt()).collect();
    let inv_count_l1: Vec<f64> = counts.iter().map(|&c| 1.0 / c as f64).collect();

    // reduced_graph = normalized_reduction_map.T @ graph @ base_reduction_map, clipped [0,1]
    let graph_indptr_usize: Vec<usize> = indptr.iter().map(|&x| x as usize).collect();
    let graph_indices_usize: Vec<usize> = indices.iter().map(|&x| x as usize).collect();
    let graph_data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();

    let mut nrm_t_indptr = vec![0usize; n_parts + 1];
    for &p in &partition {
        nrm_t_indptr[p as usize + 1] += 1;
    }
    for a in 0..n_parts {
        nrm_t_indptr[a + 1] += nrm_t_indptr[a];
    }
    let mut nrm_t_indices = vec![0usize; n_points];
    let mut nrm_t_data = vec![0.0f64; n_points];
    {
        let mut cursor = nrm_t_indptr.clone();
        for i in 0..n_points {
            let p = partition[i] as usize;
            let pos = cursor[p];
            nrm_t_indices[pos] = i;
            nrm_t_data[pos] = inv_sqrt_count[p];
            cursor[p] += 1;
        }
    }

    let br_indptr: Vec<usize> = (0..=n_points).collect();
    let br_indices: Vec<usize> = partition.iter().map(|&p| p as usize).collect();
    let br_data: Vec<f64> = vec![1.0; n_points];

    let (stage1_indptr, stage1_indices, stage1_data) = smmp(
        &nrm_t_indptr,
        &nrm_t_indices,
        &nrm_t_data,
        &graph_indptr_usize,
        &graph_indices_usize,
        &graph_data_f64,
        n_parts,
        n_points,
    );
    let (reduced_indptr_usize, reduced_indices_usize, mut reduced_data_f64) = smmp(
        &stage1_indptr,
        &stage1_indices,
        &stage1_data,
        &br_indptr,
        &br_indices,
        &br_data,
        n_parts,
        n_parts,
    );
    for v in reduced_data_f64.iter_mut() {
        *v = v.clamp(0.0, 1.0);
    }

    let reduced_indptr: Vec<i64> = reduced_indptr_usize.iter().map(|&x| x as i64).collect();
    let reduced_indices: Vec<i64> = reduced_indices_usize.iter().map(|&x| x as i64).collect();
    let reduced_data: Vec<f32> = reduced_data_f64.iter().map(|&x| x as f32).collect();
    let reduced_indptr_u32: Vec<u32> = reduced_indptr_usize.iter().map(|&x| x as u32).collect();
    let reduced_indices_u32: Vec<u32> = reduced_indices_usize.iter().map(|&x| x as u32).collect();

    // reduced_data (original space) for the recursive PCA base case: mean of
    // each partition's points, matching `data_reducer @ data` where
    // `data_reducer = normalize(normalized_reduction_map.T, norm="l1")`
    // (row-normalized partition-indicator matrix, i.e. per-partition mean).
    let reduced_data_full: Option<Vec<f32>> = data_full.map(|full| {
        let mut acc = vec![0.0f64; n_parts * n_features];
        for i in 0..n_points {
            let p = partition[i] as usize;
            for f in 0..n_features {
                acc[p * n_features + f] += full[i * n_features + f] as f64 * inv_count_l1[p];
            }
        }
        acc.iter().map(|&x| x as f32).collect()
    });

    let reduced_init = if recursive_init {
        let next_approx_n_parts = (approx_n_parts / 4).max(1);
        Some(label_propagation_init_recursive(
            &reduced_indptr,
            &reduced_indices,
            &reduced_data,
            n_parts,
            reduced_data_full.as_deref(),
            n_features,
            n_label_prop_iter,
            n_embedding_epochs.min(255),
            next_approx_n_parts,
            n_components,
            scaling,
            noise_level,
            true,
            base_init_threshold,
            state,
        ))
    } else {
        None
    };

    let initial_alpha = 0.001 * n_embedding_epochs as f32;
    let reduced_layout = crate::node_embedding::node_embedding_repr_self_driven(
        &reduced_indptr_u32,
        &reduced_indices_u32,
        &reduced_data,
        n_parts,
        n_components,
        n_embedding_epochs,
        reduced_init.as_deref(),
        initial_alpha,
        1.0,
        noise_level,
        state,
    );

    // partition_expander upscaling (identical algebra to label_propagation_init).
    let mut graph_dense = vec![0.0f64; n_points * n_points];
    for i in 0..n_points {
        for k in indptr[i] as usize..indptr[i + 1] as usize {
            let j = indices[k] as usize;
            graph_dense[i * n_points + j] = data[k] as f64;
        }
    }

    let mut data_expander = vec![0.0f64; n_points * n_parts];
    for i in 0..n_points {
        for j in 0..n_points {
            let m_ij = graph_dense[i * n_points + j] * graph_dense[j * n_points + i];
            if m_ij != 0.0 {
                let b = partition[j] as usize;
                data_expander[i * n_parts + b] += m_ij * inv_sqrt_count[b];
            }
        }
        let row_sum: f64 = data_expander[i * n_parts..i * n_parts + n_parts]
            .iter()
            .sum();
        if row_sum != 0.0 {
            for v in data_expander[i * n_parts..i * n_parts + n_parts].iter_mut() {
                *v /= row_sum;
            }
        }
    }

    let mut result = vec![0.0f64; n_points * n_components];
    for i in 0..n_points {
        for c in 0..n_components {
            let mut acc = 0.0f64;
            for b in 0..n_parts {
                acc += data_expander[i * n_parts + b] * reduced_layout[b * n_components + c] as f64;
            }
            let p = partition[i] as usize;
            let direct = reduced_layout[p * n_components + c] as f64;
            result[i * n_components + c] = (acc + direct) / 2.0;
        }
    }

    let mut mean = vec![0.0f64; n_components];
    for i in 0..n_points {
        for c in 0..n_components {
            mean[c] += result[i * n_components + c];
        }
    }
    for m in mean.iter_mut() {
        *m /= n_points as f64;
    }

    let mut out = vec![0.0f32; n_points * n_components];
    for i in 0..n_points {
        for c in 0..n_components {
            out[i * n_components + c] =
                (scaling as f64 * (result[i * n_components + c] - mean[c])) as f32;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remap_labels_basic() {
        let mut labels = vec![5, -1, 5, 2, -1, 2, 9, -1];
        remap_labels(&mut labels);
        assert_eq!(labels, vec![1, 3, 1, 0, 4, 0, 2, 5]);
    }

    #[test]
    fn label_outliers_isolated_node_falls_back_to_rng() {
        let indptr = vec![0i64, 0, 2, 4];
        let indices = vec![2i64, 1, 1, 2];
        let mut labels = vec![-1i64, 0, 1];
        let rng_state: RngState = [42, 84, 126];
        label_outliers(&indptr, &indices, &mut labels, &rng_state);
        assert_eq!(labels, vec![0, 0, 1]);
    }
}
