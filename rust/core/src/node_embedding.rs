//! Port of `evoc/node_embedding.py`: UMAP-style SGD graph layout.
//!
//! PORT NOTE: negative sampling here does NOT use `tau_rand`. Targets are
//! pure integer arithmetic on a SCALAR per-epoch `rng_state`
//! (`((n + p) * i * rng_state) % n_vertices` in the plain kernel;
//! `node_order[(raw_index * (n + p + 1) * rng_state) % n_vertices]` in the
//! repr kernel). That scalar is drawn once per epoch from numpy MT and is
//! not portably reproducible, so callers supply the recorded `rng_val`
//! array verbatim. Every `numba.prange` is a plain sequential loop
//! (single-threaded wasm32); the repr kernel's block/two-pass update
//! structure is kept intact for parity.
//!
//! PORT NOTE: the integer products above use i64 arithmetic here. numba's
//! intermediate width is at least 64-bit for these (dominated by the intp
//! `raw_index`); all fixture-realistic operands keep the product well below
//! 2^63 so no truncation occurs. If an input ever pushed the product past
//! i64 range, numba would wrap differently -- not a concern for any
//! fixtured or realistic case.

/// Port of `make_epochs_per_sample`.
///
/// PORT NOTE (cast order): `n_samples = maximum(n_epochs * (w / w.max()),
/// 1.0)` is computed in f64, cast to f32 as a whole array, and only then
/// divided into `float(n_epochs)` (an f64). Replicate the f32 cast on
/// `n_samples` before the division.
pub fn make_epochs_per_sample(weights: &[f32], n_epochs: usize) -> Vec<f32> {
    let w_max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut result = vec![0.0f32; weights.len()];
    for (i, &w) in weights.iter().enumerate() {
        // n_samples in f64, floored at 1.0, then cast to f32
        let n_samples_f64 = (n_epochs as f64 * (w as f64 / w_max as f64)).max(1.0);
        let n_samples_f32 = n_samples_f64 as f32;
        // float(n_epochs) / n_samples_f32  (numpy broadcasts the f64 scalar
        // against the f32 array, yielding f32)
        result[i] = (n_epochs as f64 / n_samples_f32 as f64) as f32;
    }
    result
}

/// Port of `rdist`: squared euclidean, f32 accumulation.
pub fn rdist(x: &[f32], y: &[f32]) -> f32 {
    let mut result: f32 = 0.0;
    let dim = x.len();
    for i in 0..dim {
        let diff: f32 = x[i] - y[i];
        result += diff * diff;
    }
    result
}

/// Port of `clip`: branch order is `> hi` first, then `< lo`, else `val`.
#[inline]
pub fn clip(val: f32, lo: f32, hi: f32) -> f32 {
    if val > hi {
        hi
    } else if val < lo {
        lo
    } else {
        val
    }
}

/// Port of `node_embedding_epoch` (non-reproducible kernel): COO edge lists,
/// updates `embedding` directly in place per edge.
#[allow(clippy::too_many_arguments)]
pub fn node_embedding_epoch(
    embedding: &mut [f32],
    head: &[u32],
    tail: &[u32],
    n_vertices: u32,
    epochs_per_sample: &[f32],
    rng_state: u32,
    dim: usize,
    alpha: f32,
    epochs_per_negative_sample: &[f32],
    epoch_of_next_negative_sample: &mut [f32],
    epoch_of_next_sample: &mut [f32],
    n: u32,
    noise_level: f32,
) {
    for i in 0..epochs_per_sample.len() {
        if epoch_of_next_sample[i] <= n as f32 {
            let j = head[i] as usize;
            let k = tail[i] as usize;

            let dist_squared = rdist(
                &embedding[j * dim..j * dim + dim],
                &embedding[k * dim..k * dim + dim],
            );

            if dist_squared > 0.0 {
                let dist = dist_squared.sqrt();
                let grad_coeff =
                    (-2.0 * noise_level * dist - 2.0) / (2.0 * dist_squared - 0.5 * dist + 1.0);

                for di in 0..dim {
                    let grad_d = grad_coeff * (embedding[j * dim + di] - embedding[k * dim + di]);
                    embedding[j * dim + di] += grad_d * alpha;
                    embedding[k * dim + di] += -grad_d * alpha;
                }
            }

            epoch_of_next_sample[i] += epochs_per_sample[i];
            let n_neg_samples = ((n as f32 - epoch_of_next_negative_sample[i])
                / epochs_per_negative_sample[i]) as i64;

            for p in 0..n_neg_samples {
                // ((n + p) * i * rng_state) % n_vertices  (see module PORT NOTE)
                let kk = (((n as i64 + p) * i as i64 * rng_state as i64)
                    .rem_euclid(n_vertices as i64)) as usize;
                let dist_squared = rdist(
                    &embedding[j * dim..j * dim + dim],
                    &embedding[kk * dim..kk * dim + dim],
                );

                if dist_squared > 1e-2 {
                    let grad_coeff = 4.0 / ((1.0 + 0.25 * dist_squared) * dist_squared);

                    for di in 0..dim {
                        let grad_d = clip(
                            grad_coeff * (embedding[j * dim + di] - embedding[kk * dim + di]),
                            -4.0,
                            4.0,
                        );
                        embedding[j * dim + di] += grad_d * alpha;
                    }
                }
            }

            epoch_of_next_negative_sample[i] +=
                n_neg_samples as f32 * epochs_per_negative_sample[i];
        }
    }
}

/// Port of `node_embedding_epoch_repr` (reproducible kernel): CSR + per-node
/// blocks, deferred `updates` accumulator applied in a second pass per block.
#[allow(clippy::too_many_arguments)]
pub fn node_embedding_epoch_repr(
    embedding: &mut [f32],
    csr_indptr: &[u32],
    csr_indices: &[u32],
    n_vertices: u32,
    epochs_per_sample: &[f32],
    rng_state: u32,
    dim: usize,
    alpha: f32,
    epochs_per_negative_sample: &[f32],
    epoch_of_next_negative_sample: &mut [f32],
    epoch_of_next_sample: &mut [f32],
    n: u32,
    noise_level: f32,
    gamma: f32,
    updates: &mut [f32],
    node_order: &[u32],
    block_size: usize,
) {
    let mut block_start = 0usize;
    while block_start < n_vertices as usize {
        let block_end = std::cmp::min(block_start + block_size, n_vertices as usize);

        for node_idx in block_start..block_end {
            let from_node = node_order[node_idx] as usize;

            for raw_index in csr_indptr[from_node] as usize..csr_indptr[from_node + 1] as usize {
                if epoch_of_next_sample[raw_index] <= n as f32 {
                    let to_node = csr_indices[raw_index] as usize;

                    let dist_squared = rdist(
                        &embedding[from_node * dim..from_node * dim + dim],
                        &embedding[to_node * dim..to_node * dim + dim],
                    );

                    if dist_squared > 0.0 {
                        let dist = dist_squared.sqrt();
                        let grad_coeff = (-2.0 * noise_level * dist - 2.0)
                            / (2.0 * dist_squared - 0.5 * dist + 1.0);
                        for d in 0..dim {
                            let grad_d = grad_coeff
                                * (embedding[from_node * dim + d] - embedding[to_node * dim + d]);
                            updates[from_node * dim + d] += grad_d * alpha;
                        }
                    }

                    epoch_of_next_sample[raw_index] += epochs_per_sample[raw_index];

                    let n_neg_samples = ((n as f32 - epoch_of_next_negative_sample[raw_index])
                        / epochs_per_negative_sample[raw_index])
                        as i64;

                    for p in 0..n_neg_samples {
                        // node_order[(raw_index * (n + p + 1) * rng_state) % n_vertices]
                        let order_idx = ((raw_index as i64 * (n as i64 + p + 1) * rng_state as i64)
                            .rem_euclid(n_vertices as i64))
                            as usize;
                        let to_node = node_order[order_idx] as usize;

                        let dist_squared = rdist(
                            &embedding[from_node * dim..from_node * dim + dim],
                            &embedding[to_node * dim..to_node * dim + dim],
                        );

                        if dist_squared > 1e-2 {
                            let grad_coeff =
                                gamma * 4.0 / ((1.0 + 0.25 * dist_squared) * dist_squared);

                            if grad_coeff > 0.0 {
                                for d in 0..dim {
                                    let grad_d = clip(
                                        grad_coeff
                                            * (embedding[from_node * dim + d]
                                                - embedding[to_node * dim + d]),
                                        -4.0,
                                        4.0,
                                    );
                                    updates[from_node * dim + d] += grad_d * alpha;
                                }
                            }
                        }
                    }

                    epoch_of_next_negative_sample[raw_index] +=
                        n_neg_samples as f32 * epochs_per_negative_sample[raw_index];
                }
            }
        }

        // second pass over the same block: apply accumulated updates
        for node_idx in block_start..block_end {
            let from_node = node_order[node_idx] as usize;
            for d in 0..dim {
                embedding[from_node * dim + d] += updates[from_node * dim + d];
            }
        }

        block_start += block_size;
    }
}

/// Port of `node_embedding` (reproducible path). Consumes the pre-drawn
/// per-epoch `rng_val` and `node_order_shuffles` (each an already-permuted
/// `node_order` for that epoch, applied AFTER the epoch's kernel call), plus
/// an explicit `initial_embedding` (flat row-major n_vertices x
/// n_components). Returns the final flat embedding.
///
/// PORT NOTE: `updates` is NOT re-zeroed between epochs -- it is decayed by
/// `(1 - alpha)^2 * 0.5` after each epoch (gradient momentum). `alpha` for
/// epoch n>=1 is set at the BOTTOM of the loop body as
/// `initial_alpha * (1 - n/n_epochs)`, so epoch 0 uses the raw
/// `initial_alpha`.
#[allow(clippy::too_many_arguments)]
pub fn node_embedding_repr(
    graph_indptr: &[u32],
    graph_indices: &[u32],
    graph_data: &[f32],
    n_vertices: usize,
    n_components: usize,
    n_epochs: usize,
    initial_embedding: &[f32],
    initial_alpha: f32,
    negative_sample_rate: f32,
    noise_level: f32,
    rng_val: &[u32],
    node_order_shuffles: &[Vec<u32>],
) -> Vec<f32> {
    let mut embedding = initial_embedding.to_vec();

    let epochs_per_sample = make_epochs_per_sample(graph_data, n_epochs);
    let mut epochs_per_negative_sample: Vec<f32> = epochs_per_sample
        .iter()
        .map(|&e| e / negative_sample_rate)
        .collect();
    // reproducible path inflates by 1.5
    for e in epochs_per_negative_sample.iter_mut() {
        *e *= 1.5;
    }
    let mut epoch_of_next_negative_sample = epochs_per_negative_sample.clone();
    let mut epoch_of_next_sample = epochs_per_sample.clone();

    let mut updates = vec![0.0f32; n_vertices * n_components];
    let mut node_order: Vec<u32> = (0..n_vertices as u32).collect();
    // gamma_schedule = linspace(0.5, 1.5, n_epochs)
    let gamma_schedule = linspace(0.5, 1.5, n_epochs);
    let block_size = std::cmp::max(1024, n_vertices / 8);
    let dim = n_components;
    let mut alpha = initial_alpha;

    for n in 0..n_epochs {
        node_embedding_epoch_repr(
            &mut embedding,
            graph_indptr,
            graph_indices,
            n_vertices as u32,
            &epochs_per_sample,
            rng_val[n],
            dim,
            alpha,
            &epochs_per_negative_sample,
            &mut epoch_of_next_negative_sample,
            &mut epoch_of_next_sample,
            n as u32,
            noise_level,
            gamma_schedule[n] as f32,
            &mut updates,
            &node_order,
            block_size,
        );
        // updates decay (NOT reset)
        let decay = (1.0 - alpha).powi(2) * 0.5;
        for u in updates.iter_mut() {
            *u *= decay;
        }
        // per-epoch shuffle (recorded), applied after the kernel call
        node_order = node_order_shuffles[n].clone();

        alpha = initial_alpha * (1.0 - (n as f32 / n_epochs as f32));
    }

    embedding
}

/// `np.linspace(start, stop, num)` in f64.
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }
    let step = (stop - start) / (num - 1) as f64;
    (0..num).map(|i| start + step * i as f64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_branch_order() {
        assert_eq!(clip(5.0, -4.0, 4.0), 4.0);
        assert_eq!(clip(-5.0, -4.0, 4.0), -4.0);
        assert_eq!(clip(1.0, -4.0, 4.0), 1.0);
        assert_eq!(clip(4.0, -4.0, 4.0), 4.0);
        assert_eq!(clip(-4.0, -4.0, 4.0), -4.0);
    }

    #[test]
    fn rdist_zero_for_identical() {
        let x = [1.0f32, -2.0, 3.5];
        assert_eq!(rdist(&x, &x), 0.0);
    }

    #[test]
    fn epochs_per_sample_floor_at_one() {
        // max weight edge -> n_samples = n_epochs -> eps = 1.0 exactly.
        // tiny weight edge -> n_samples floored at 1.0 -> eps = n_epochs.
        let weights = [1.0f32, 0.0001, 0.5];
        let eps = make_epochs_per_sample(&weights, 10);
        assert_eq!(eps[0], 1.0);
        assert_eq!(eps[1], 10.0);
    }

    #[test]
    fn linspace_endpoints() {
        let v = linspace(0.5, 1.5, 5);
        assert_eq!(v[0], 0.5);
        assert_eq!(v[4], 1.5);
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn repr_epoch_no_firing_leaves_embedding_unchanged() {
        // epoch_of_next_sample all huge -> nothing fires -> embedding stays.
        let mut embedding = vec![0.1f32, 0.2, 0.3, 0.4];
        let before = embedding.clone();
        let indptr = vec![0u32, 1, 2];
        let indices = vec![1u32, 0];
        let eps = vec![1.0f32, 1.0];
        let mut eons = vec![1000.0f32, 1000.0];
        let mut eonss = vec![1000.0f32, 1000.0];
        let epns = vec![1.0f32, 1.0];
        let mut updates = vec![0.0f32; 4];
        let node_order = vec![0u32, 1];
        node_embedding_epoch_repr(
            &mut embedding,
            &indptr,
            &indices,
            2,
            &eps,
            7,
            2,
            0.1,
            &epns,
            &mut eons,
            &mut eonss,
            0,
            0.5,
            1.0,
            &mut updates,
            &node_order,
            4096,
        );
        assert_eq!(embedding, before);
    }
}
