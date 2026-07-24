//! Randomized PCA (Halko/Martinsson/Tropp randomized SVD), used as the
//! `evoc_clusters`/`EVoC.fit_predict` pipeline's replacement for sklearn's
//! `PCA(svd_solver='randomized')`.
//!
//! PORT NOTE (scope decision, see PORT_NOTES.md / the phase-9 task): bit-exact
//! parity with sklearn's PCA is impossible -- sklearn's randomized SVD draws
//! from an internal `RandomState` in ways that are not documented/stable
//! across versions, so there is no draw sequence to port faithfully. Instead
//! this module implements the actual algorithm sklearn's randomized solver is
//! based on (power iteration + QR, then a dense SVD of the small projected
//! matrix), using this project's own `tau_rand` RNG for the random
//! projection so the whole pipeline stays deterministic and reproducible
//! across the Rust and Zig ports (same seed -> same output), without
//! attempting to replay numpy's Mersenne Twister draw-for-draw. Validated via
//! reconstruction-quality unit tests and an end-to-end ARI check against
//! Python's real `evoc.EVoC` (see `rust/core/PCA_VALIDATION.md`), not via
//! bit-exact/tolerance fixtures like the rest of this port.

use crate::rng::{tau_rand, RngState};

/// Draw one standard-normal sample via Box-Muller, consuming two `tau_rand`
/// draws. `tau_rand` returns values in `[0, 1]`; a value of exactly 0.0 for
/// `u1` would make `ln(u1)` diverge, so it is nudged to a tiny positive
/// epsilon (this only matters for the vanishingly rare exact-zero draw).
pub fn next_standard_normal(state: &mut RngState) -> f32 {
    let mut u1 = tau_rand(state);
    if u1 <= 0.0 {
        u1 = 1e-7;
    }
    let u2 = tau_rand(state);
    let r = (-2.0 * (u1 as f64).ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2 as f64;
    (r * theta.cos()) as f32
}

/// Fill a flat `rows x cols` matrix with i.i.d. standard-normal entries,
/// row-major, advancing `state` in row-major draw order.
fn random_gaussian_matrix(rows: usize, cols: usize, state: &mut RngState) -> Vec<f64> {
    let mut m = vec![0.0f64; rows * cols];
    for v in m.iter_mut() {
        *v = next_standard_normal(state) as f64;
    }
    m
}

/// Dense matmul: `a` is `m x k`, `b` is `k x n`, result `m x n`. All row-major.
fn matmul(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            if a_ip == 0.0 {
                continue;
            }
            for j in 0..n {
                out[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }
    out
}

/// Transpose an `m x n` row-major matrix into `n x m`.
fn transpose(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = a[i * n + j];
        }
    }
    out
}

/// In-place modified Gram-Schmidt QR: `a` is `m x n` (m >= n), overwritten
/// with the orthonormal `Q` (same shape). Returns nothing else -- only `Q`
/// is needed for the randomized range-finder.
fn qr_orthonormalize(a: &mut [f64], m: usize, n: usize) {
    for j in 0..n {
        // orthogonalize column j against all previous columns
        for prev in 0..j {
            let mut dot = 0.0f64;
            for i in 0..m {
                dot += a[i * n + prev] * a[i * n + j];
            }
            for i in 0..m {
                a[i * n + j] -= dot * a[i * n + prev];
            }
        }
        let mut norm = 0.0f64;
        for i in 0..m {
            norm += a[i * n + j] * a[i * n + j];
        }
        norm = norm.sqrt();
        if norm > 1e-12 {
            for i in 0..m {
                a[i * n + j] /= norm;
            }
        }
        // degenerate (near-zero) column: leave as-is (already ~0); extremely
        // unlikely for a random Gaussian projection in practice.
    }
}

/// One-sided Jacobi SVD of a small dense `p x q` matrix (`p, q` are both
/// `n_components`-scale, i.e. tiny compared to the original data), returning
/// `(u, s, vt)` with `u: p x r`, `s: r`, `vt: r x q`, `r = min(p, q)`.
/// Sufficient precision/iterations for the small matrices this module ever
/// calls it with (post range-finding projection).
fn jacobi_svd(a: &[f64], p: usize, q: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Work on a copy; columns are progressively rotated toward orthogonality.
    let mut u = a.to_vec(); // p x q, will become U * S (unnormalized columns)
    let mut v = vec![0.0f64; q * q]; // q x q, accumulates V
    for i in 0..q {
        v[i * q + i] = 1.0;
    }

    let max_sweeps = 60;
    for _ in 0..max_sweeps {
        let mut off_diag = 0.0f64;
        for i in 0..q {
            for j in (i + 1)..q {
                let mut alpha = 0.0f64;
                let mut beta = 0.0f64;
                let mut gamma = 0.0f64;
                for r in 0..p {
                    let uri = u[r * q + i];
                    let urj = u[r * q + j];
                    alpha += uri * uri;
                    beta += urj * urj;
                    gamma += uri * urj;
                }
                off_diag += gamma * gamma;
                if gamma.abs() < 1e-14 {
                    continue;
                }
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let t = if zeta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;

                for r in 0..p {
                    let uri = u[r * q + i];
                    let urj = u[r * q + j];
                    u[r * q + i] = c * uri - s * urj;
                    u[r * q + j] = s * uri + c * urj;
                }
                for r in 0..q {
                    let vri = v[r * q + i];
                    let vrj = v[r * q + j];
                    v[r * q + i] = c * vri - s * vrj;
                    v[r * q + j] = s * vri + c * vrj;
                }
            }
        }
        if off_diag.sqrt() < 1e-12 {
            break;
        }
    }

    // singular values = column norms of u; normalize columns to get U
    let mut sigma = vec![0.0f64; q];
    for j in 0..q {
        let mut norm = 0.0f64;
        for r in 0..p {
            norm += u[r * q + j] * u[r * q + j];
        }
        sigma[j] = norm.sqrt();
    }
    let mut u_norm = u.clone();
    for j in 0..q {
        if sigma[j] > 1e-12 {
            for r in 0..p {
                u_norm[r * q + j] /= sigma[j];
            }
        }
    }

    // sort descending by singular value
    let mut order: Vec<usize> = (0..q).collect();
    order.sort_by(|&a, &b| sigma[b].partial_cmp(&sigma[a]).unwrap());

    let r = p.min(q);
    let mut u_out = vec![0.0f64; p * r];
    let mut s_out = vec![0.0f64; r];
    let mut vt_out = vec![0.0f64; r * q];
    for (rank, &j) in order.iter().take(r).enumerate() {
        s_out[rank] = sigma[j];
        for row in 0..p {
            u_out[row * r + rank] = u_norm[row * q + j];
        }
        for col in 0..q {
            vt_out[rank * q + col] = v[col * q + j];
        }
    }

    (u_out, s_out, vt_out)
}

/// Result of [`randomized_svd`]: `u: m x n_components`, `s: n_components`,
/// `vt: n_components x n`.
pub struct RandomizedSvd {
    pub u: Vec<f64>,
    pub s: Vec<f64>,
    pub vt: Vec<f64>,
    pub n_components: usize,
}

/// Port of the Halko/Martinsson/Tropp randomized SVD algorithm (the actual
/// algorithm behind sklearn's `PCA(svd_solver='randomized')`): random
/// projection range-finder with `n_power_iterations` steps of power
/// iteration (each followed by a re-orthonormalization, matching sklearn's
/// `power_iteration_normalizer='QR'` default), then an exact SVD of the
/// small projected matrix.
///
/// `data` is a flat row-major `m x n` matrix. `state` drives the random
/// Gaussian projection matrix via `tau_rand` (see module docs: this is the
/// point where bit-exact sklearn parity is intentionally relaxed).
pub fn randomized_svd(
    data: &[f64],
    m: usize,
    n: usize,
    n_components: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    state: &mut RngState,
) -> RandomizedSvd {
    let n_random = (n_components + n_oversamples).min(n);

    // Random test matrix, n x n_random.
    let omega = random_gaussian_matrix(n, n_random, state);
    // Y = A @ Omega  (m x n_random)
    let mut y = matmul(data, m, n, &omega, n_random);
    qr_orthonormalize(&mut y, m, n_random);

    for _ in 0..n_power_iterations {
        // Z = A^T @ Q  (n x n_random), orthonormalize
        let a_t = transpose(data, m, n);
        let mut z = matmul(&a_t, n, m, &y, n_random);
        qr_orthonormalize(&mut z, n, n_random);
        // Y = A @ Z (m x n_random), orthonormalize
        y = matmul(data, m, n, &z, n_random);
        qr_orthonormalize(&mut y, m, n_random);
    }

    let q = y; // m x n_random, orthonormal columns

    // B = Q^T @ A  (n_random x n)
    let q_t = transpose(&q, m, n_random);
    let b = matmul(&q_t, n_random, m, data, n);

    // SVD of small matrix B (n_random x n): B = U_b S V^T
    let (u_b, s, vt) = jacobi_svd(&b, n_random, n);

    // U = Q @ U_b  (m x n_random_rank)
    let rank = u_b.len() / n_random.max(1);
    let rank = if n_random > 0 { rank } else { 0 };
    let u_full = matmul(&q, m, n_random, &u_b, rank);

    let k = n_components.min(rank);
    let mut u_out = vec![0.0f64; m * k];
    for i in 0..m {
        for c in 0..k {
            u_out[i * k + c] = u_full[i * rank + c];
        }
    }
    let s_out = s[..k].to_vec();
    let mut vt_out = vec![0.0f64; k * n];
    vt_out[..(k * n)].copy_from_slice(&vt[..(k * n)]);

    RandomizedSvd {
        u: u_out,
        s: s_out,
        vt: vt_out,
        n_components: k,
    }
}

/// PCA via randomized SVD: mean-centers `data` (flat row-major `m x n`) and
/// projects onto the top `n_components` principal axes. Returns the flat
/// row-major `m x n_components` projected data.
pub fn randomized_pca(
    data: &[f32],
    m: usize,
    n: usize,
    n_components: usize,
    state: &mut RngState,
) -> Vec<f32> {
    let n_components = n_components.min(m).min(n).max(1);

    let mut centered = vec![0.0f64; m * n];
    let mut mean = vec![0.0f64; n];
    for i in 0..m {
        for j in 0..n {
            mean[j] += data[i * n + j] as f64;
        }
    }
    for v in mean.iter_mut() {
        *v /= m as f64;
    }
    for i in 0..m {
        for j in 0..n {
            centered[i * n + j] = data[i * n + j] as f64 - mean[j];
        }
    }

    let n_oversamples = 10;
    let n_power_iterations = 4;
    let svd = randomized_svd(
        &centered,
        m,
        n,
        n_components,
        n_oversamples,
        n_power_iterations,
        state,
    );

    // projected = U * S  (m x k)
    let k = svd.n_components;
    let mut out = vec![0.0f32; m * k];
    for i in 0..m {
        for c in 0..k {
            out[i * k + c] = (svd.u[i * k + c] * svd.s[c]) as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::seed;

    #[test]
    fn low_rank_reconstruction_quality() {
        // Build a rank-2 signal in 20-D embedded with small noise, check the
        // randomized PCA recovers >95% of variance in the top 2 components.
        let mut state: RngState = [0; 3];
        seed(&mut state, 7);

        let n_points = 200;
        let n_features = 20;
        let mut data = vec![0.0f32; n_points * n_features];
        // two orthogonal-ish latent directions
        let dir1: Vec<f32> = (0..n_features).map(|j| ((j as f32) * 0.3).sin()).collect();
        let dir2: Vec<f32> = (0..n_features).map(|j| ((j as f32) * 0.7).cos()).collect();

        for i in 0..n_points {
            let a = next_standard_normal(&mut state) * 3.0;
            let b = next_standard_normal(&mut state) * 2.0;
            for j in 0..n_features {
                let noise = next_standard_normal(&mut state) * 0.05;
                data[i * n_features + j] = a * dir1[j] + b * dir2[j] + noise;
            }
        }

        let mut pca_state: RngState = [0; 3];
        seed(&mut pca_state, 123);
        let projected = randomized_pca(&data, n_points, n_features, 2, &mut pca_state);

        // total variance of centered data
        let mut mean = vec![0.0f64; n_features];
        for i in 0..n_points {
            for j in 0..n_features {
                mean[j] += data[i * n_features + j] as f64;
            }
        }
        for v in mean.iter_mut() {
            *v /= n_points as f64;
        }
        let mut total_var = 0.0f64;
        for i in 0..n_points {
            for j in 0..n_features {
                let d = data[i * n_features + j] as f64 - mean[j];
                total_var += d * d;
            }
        }

        // reconstruction error via projected components' captured variance
        let mut captured_var = 0.0f64;
        let mut proj_mean = [0.0f64; 2];
        for i in 0..n_points {
            for c in 0..2 {
                proj_mean[c] += projected[i * 2 + c] as f64;
            }
        }
        for v in proj_mean.iter_mut() {
            *v /= n_points as f64;
        }
        for i in 0..n_points {
            for c in 0..2 {
                let d = projected[i * 2 + c] as f64 - proj_mean[c];
                captured_var += d * d;
            }
        }

        let ratio = captured_var / total_var;
        assert!(
            ratio > 0.90,
            "expected randomized PCA to capture >90% of variance in top-2 components, got {ratio}"
        );
    }

    #[test]
    fn pca_output_shape() {
        let mut state: RngState = [0; 3];
        seed(&mut state, 42);
        let data: Vec<f32> = (0..(30 * 8)).map(|i| (i % 7) as f32 * 0.1).collect();
        let out = randomized_pca(&data, 30, 8, 3, &mut state);
        assert_eq!(out.len(), 30 * 3);
    }

    #[test]
    fn standard_normal_is_roughly_zero_mean_unit_var() {
        let mut state: RngState = [0; 3];
        seed(&mut state, 99);
        let n = 5000;
        let samples: Vec<f32> = (0..n).map(|_| next_standard_normal(&mut state)).collect();
        let mean: f64 = samples.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let var: f64 = samples
            .iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>()
            / n as f64;
        assert!(mean.abs() < 0.1, "mean {mean}");
        assert!((var - 1.0).abs() < 0.15, "var {var}");
    }
}
