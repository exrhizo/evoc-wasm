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
//! bit-exact/tolerance fixtures like the rest of this port. This is a
//! from-scratch Zig port of the same algorithm structure as
//! `rust/core/src/pca.rs`, not a transliteration of it (per PORTING.md's
//! parity requirement -- the two ports are independent implementations).

const std = @import("std");
const Allocator = std.mem.Allocator;
const rng = @import("rng.zig");

/// Draw one standard-normal sample via Box-Muller, consuming two `tau_rand`
/// draws. `tau_rand` returns values in `[0, 1]`; a value of exactly 0.0 for
/// `u1` would make `ln(u1)` diverge, so it is nudged to a tiny positive
/// epsilon (this only matters for the vanishingly rare exact-zero draw).
pub fn next_standard_normal(state: *rng.State) f32 {
    var draw1 = rng.tau_rand(state);
    if (draw1 <= 0.0) draw1 = 1e-7;
    const draw2 = rng.tau_rand(state);
    const r = @sqrt(-2.0 * @log(@as(f64, draw1)));
    const theta = 2.0 * std.math.pi * @as(f64, draw2);
    return @floatCast(r * @cos(theta));
}

/// Fill a flat `rows x cols` matrix with i.i.d. standard-normal entries,
/// row-major, advancing `state` in row-major draw order.
fn random_gaussian_matrix(allocator: Allocator, rows: usize, cols: usize, state: *rng.State) ![]f64 {
    const m = try allocator.alloc(f64, rows * cols);
    for (m) |*v| v.* = next_standard_normal(state);
    return m;
}

/// Dense matmul: `a` is `m x k`, `b` is `k x n`, result `m x n`. All row-major.
fn matmul(allocator: Allocator, a: []const f64, m: usize, k: usize, b: []const f64, n: usize) ![]f64 {
    const out = try allocator.alloc(f64, m * n);
    @memset(out, 0);
    for (0..m) |i| {
        for (0..k) |p| {
            const a_ip = a[i * k + p];
            if (a_ip == 0.0) continue;
            for (0..n) |j| {
                out[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }
    return out;
}

/// Transpose an `m x n` row-major matrix into `n x m`.
fn transpose(allocator: Allocator, a: []const f64, m: usize, n: usize) ![]f64 {
    const out = try allocator.alloc(f64, m * n);
    for (0..m) |i| {
        for (0..n) |j| {
            out[j * m + i] = a[i * n + j];
        }
    }
    return out;
}

/// In-place modified Gram-Schmidt QR: `a` is `m x n` (m >= n), overwritten
/// with the orthonormal `Q` (same shape). Only `Q` is needed for the
/// randomized range-finder.
fn qr_orthonormalize(a: []f64, m: usize, n: usize) void {
    for (0..n) |j| {
        for (0..j) |prev| {
            var dot: f64 = 0.0;
            for (0..m) |i| dot += a[i * n + prev] * a[i * n + j];
            for (0..m) |i| a[i * n + j] -= dot * a[i * n + prev];
        }
        var norm: f64 = 0.0;
        for (0..m) |i| norm += a[i * n + j] * a[i * n + j];
        norm = @sqrt(norm);
        if (norm > 1e-12) {
            for (0..m) |i| a[i * n + j] /= norm;
        }
        // degenerate (near-zero) column: leave as-is.
    }
}

pub const JacobiSvd = struct {
    u: []f64, // p x r
    s: []f64, // r
    vt: []f64, // r x q
    allocator: Allocator,

    pub fn deinit(self: *JacobiSvd) void {
        self.allocator.free(self.u);
        self.allocator.free(self.s);
        self.allocator.free(self.vt);
        self.* = undefined;
    }
};

/// One-sided Jacobi SVD of a small dense `p x q` matrix (`p, q` are both
/// `n_components`-scale, i.e. tiny compared to the original data), returning
/// `(u, s, vt)` with `u: p x r`, `s: r`, `vt: r x q`, `r = min(p, q)`.
/// Sufficient precision/iterations for the small matrices this module ever
/// calls it with (post range-finding projection).
fn jacobi_svd(allocator: Allocator, a: []const f64, p: usize, q: usize) !JacobiSvd {
    var u = try allocator.dupe(f64, a); // p x q, becomes U*S (unnormalized columns)
    defer allocator.free(u);
    const v = try allocator.alloc(f64, q * q); // q x q, accumulates V
    defer allocator.free(v);
    @memset(v, 0);
    for (0..q) |i| v[i * q + i] = 1.0;

    const max_sweeps = 60;
    var sweep: usize = 0;
    while (sweep < max_sweeps) : (sweep += 1) {
        var off_diag: f64 = 0.0;
        for (0..q) |i| {
            for (i + 1..q) |j| {
                var alpha: f64 = 0.0;
                var beta: f64 = 0.0;
                var gamma: f64 = 0.0;
                for (0..p) |r| {
                    const uri = u[r * q + i];
                    const urj = u[r * q + j];
                    alpha += uri * uri;
                    beta += urj * urj;
                    gamma += uri * urj;
                }
                off_diag += gamma * gamma;
                if (@abs(gamma) < 1e-14) continue;
                const zeta = (beta - alpha) / (2.0 * gamma);
                var t = std.math.sign(zeta) / (@abs(zeta) + @sqrt(1.0 + zeta * zeta));
                if (zeta == 0.0) t = 1.0;
                const c = 1.0 / @sqrt(1.0 + t * t);
                const s = c * t;

                for (0..p) |r| {
                    const uri = u[r * q + i];
                    const urj = u[r * q + j];
                    u[r * q + i] = c * uri - s * urj;
                    u[r * q + j] = s * uri + c * urj;
                }
                for (0..q) |r| {
                    const vri = v[r * q + i];
                    const vrj = v[r * q + j];
                    v[r * q + i] = c * vri - s * vrj;
                    v[r * q + j] = s * vri + c * vrj;
                }
            }
        }
        if (@sqrt(off_diag) < 1e-12) break;
    }

    const sigma = try allocator.alloc(f64, q);
    defer allocator.free(sigma);
    for (0..q) |j| {
        var norm: f64 = 0.0;
        for (0..p) |r| norm += u[r * q + j] * u[r * q + j];
        sigma[j] = @sqrt(norm);
    }
    const u_norm = try allocator.dupe(f64, u);
    defer allocator.free(u_norm);
    for (0..q) |j| {
        if (sigma[j] > 1e-12) {
            for (0..p) |r| u_norm[r * q + j] /= sigma[j];
        }
    }

    const order = try allocator.alloc(usize, q);
    defer allocator.free(order);
    for (0..q) |i| order[i] = i;
    const SortCtx = struct {
        sigma: []const f64,
        fn lessThan(ctx: @This(), a_idx: usize, b_idx: usize) bool {
            return ctx.sigma[a_idx] > ctx.sigma[b_idx]; // descending
        }
    };
    std.mem.sort(usize, order, SortCtx{ .sigma = sigma }, SortCtx.lessThan);

    const r = @min(p, q);
    const u_out = try allocator.alloc(f64, p * r);
    const s_out = try allocator.alloc(f64, r);
    const vt_out = try allocator.alloc(f64, r * q);
    for (0..r) |rank| {
        const j = order[rank];
        s_out[rank] = sigma[j];
        for (0..p) |row| u_out[row * r + rank] = u_norm[row * q + j];
        for (0..q) |col| vt_out[rank * q + col] = v[col * q + j];
    }

    return JacobiSvd{ .u = u_out, .s = s_out, .vt = vt_out, .allocator = allocator };
}

/// Result of `randomized_svd`: `u: m x n_components`, `s: n_components`,
/// `vt: n_components x n`.
pub const RandomizedSvd = struct {
    u: []f64,
    s: []f64,
    vt: []f64,
    n_components: usize,
    allocator: Allocator,

    pub fn deinit(self: *RandomizedSvd) void {
        self.allocator.free(self.u);
        self.allocator.free(self.s);
        self.allocator.free(self.vt);
        self.* = undefined;
    }
};

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
    allocator: Allocator,
    data: []const f64,
    m: usize,
    n: usize,
    n_components: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    state: *rng.State,
) !RandomizedSvd {
    const n_random = @min(n_components + n_oversamples, n);

    const omega = try random_gaussian_matrix(allocator, n, n_random, state);
    defer allocator.free(omega);
    var y = try matmul(allocator, data, m, n, omega, n_random);
    defer allocator.free(y);
    qr_orthonormalize(y, m, n_random);

    var iter: usize = 0;
    while (iter < n_power_iterations) : (iter += 1) {
        const a_t = try transpose(allocator, data, m, n);
        defer allocator.free(a_t);
        const z = try matmul(allocator, a_t, n, m, y, n_random);
        defer allocator.free(z);
        qr_orthonormalize(z, n, n_random);
        const y_new = try matmul(allocator, data, m, n, z, n_random);
        allocator.free(y);
        y = y_new;
        qr_orthonormalize(y, m, n_random);
    }

    const q = y; // m x n_random, orthonormal columns; freed via `defer allocator.free(y)` above

    const q_t = try transpose(allocator, q, m, n_random);
    defer allocator.free(q_t);
    const b = try matmul(allocator, q_t, n_random, m, data, n);
    defer allocator.free(b);

    var svd_small = try jacobi_svd(allocator, b, n_random, n);
    defer svd_small.deinit();

    const rank = if (n_random > 0) svd_small.u.len / n_random else 0;
    const u_full = try matmul(allocator, q, m, n_random, svd_small.u, rank);
    defer allocator.free(u_full);

    const k = @min(n_components, rank);
    const u_out = try allocator.alloc(f64, m * k);
    for (0..m) |i| {
        for (0..k) |c| u_out[i * k + c] = u_full[i * rank + c];
    }
    const s_out = try allocator.dupe(f64, svd_small.s[0..k]);
    const vt_out = try allocator.alloc(f64, k * n);
    @memcpy(vt_out, svd_small.vt[0 .. k * n]);

    return RandomizedSvd{ .u = u_out, .s = s_out, .vt = vt_out, .n_components = k, .allocator = allocator };
}

/// PCA via randomized SVD: mean-centers `data` (flat row-major `m x n`) and
/// projects onto the top `n_components` principal axes. Returns the flat
/// row-major `m x n_components` projected data (caller owns, must free).
pub fn randomized_pca(
    allocator: Allocator,
    data: []const f32,
    m: usize,
    n: usize,
    n_components_in: usize,
    state: *rng.State,
) ![]f32 {
    const n_components = @max(1, @min(n_components_in, @min(m, n)));

    const centered = try allocator.alloc(f64, m * n);
    defer allocator.free(centered);
    const mean = try allocator.alloc(f64, n);
    defer allocator.free(mean);
    @memset(mean, 0);
    for (0..m) |i| {
        for (0..n) |j| mean[j] += data[i * n + j];
    }
    for (mean) |*v| v.* /= @floatFromInt(m);
    for (0..m) |i| {
        for (0..n) |j| centered[i * n + j] = @as(f64, data[i * n + j]) - mean[j];
    }

    const n_oversamples = 10;
    const n_power_iterations = 4;
    var svd = try randomized_svd(allocator, centered, m, n, n_components, n_oversamples, n_power_iterations, state);
    defer svd.deinit();

    const k = svd.n_components;
    const out = try allocator.alloc(f32, m * k);
    for (0..m) |i| {
        for (0..k) |c| out[i * k + c] = @floatCast(svd.u[i * k + c] * svd.s[c]);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "standard_normal is roughly zero-mean unit-variance" {
    var state: rng.State = undefined;
    rng.seed(&state, 99);
    const n = 5000;
    var sum: f64 = 0.0;
    var samples: [5000]f32 = undefined;
    for (0..n) |i| {
        samples[i] = next_standard_normal(&state);
        sum += samples[i];
    }
    const mean = sum / @as(f64, n);
    var var_acc: f64 = 0.0;
    for (samples) |s| {
        const d = @as(f64, s) - mean;
        var_acc += d * d;
    }
    const variance = var_acc / @as(f64, n);
    try std.testing.expect(@abs(mean) < 0.1);
    try std.testing.expect(@abs(variance - 1.0) < 0.15);
}

test "pca output shape" {
    const alloc = std.testing.allocator;
    var state: rng.State = undefined;
    rng.seed(&state, 42);
    var data: [30 * 8]f32 = undefined;
    for (0..data.len) |i| data[i] = @as(f32, @floatFromInt(i % 7)) * 0.1;
    const out = try randomized_pca(alloc, &data, 30, 8, 3, &state);
    defer alloc.free(out);
    try std.testing.expectEqual(@as(usize, 30 * 3), out.len);
}

test "low rank reconstruction quality" {
    const alloc = std.testing.allocator;
    var state: rng.State = undefined;
    rng.seed(&state, 7);

    const n_points = 200;
    const n_features = 20;
    const data = try alloc.alloc(f32, n_points * n_features);
    defer alloc.free(data);

    var dir1: [20]f32 = undefined;
    var dir2: [20]f32 = undefined;
    for (0..n_features) |j| {
        dir1[j] = @sin(@as(f32, @floatFromInt(j)) * 0.3);
        dir2[j] = @cos(@as(f32, @floatFromInt(j)) * 0.7);
    }

    for (0..n_points) |i| {
        const a = next_standard_normal(&state) * 3.0;
        const b = next_standard_normal(&state) * 2.0;
        for (0..n_features) |j| {
            const noise = next_standard_normal(&state) * 0.05;
            data[i * n_features + j] = a * dir1[j] + b * dir2[j] + noise;
        }
    }

    var pca_state: rng.State = undefined;
    rng.seed(&pca_state, 123);
    const projected = try randomized_pca(alloc, data, n_points, n_features, 2, &pca_state);
    defer alloc.free(projected);

    var mean: [20]f64 = [_]f64{0.0} ** 20;
    for (0..n_points) |i| {
        for (0..n_features) |j| mean[j] += data[i * n_features + j];
    }
    for (&mean) |*v| v.* /= @as(f64, n_points);
    var total_var: f64 = 0.0;
    for (0..n_points) |i| {
        for (0..n_features) |j| {
            const d = @as(f64, data[i * n_features + j]) - mean[j];
            total_var += d * d;
        }
    }

    var proj_mean: [2]f64 = .{ 0.0, 0.0 };
    for (0..n_points) |i| {
        for (0..2) |c| proj_mean[c] += projected[i * 2 + c];
    }
    for (&proj_mean) |*v| v.* /= @as(f64, n_points);
    var captured_var: f64 = 0.0;
    for (0..n_points) |i| {
        for (0..2) |c| {
            const d = @as(f64, projected[i * 2 + c]) - proj_mean[c];
            captured_var += d * d;
        }
    }

    const ratio = captured_var / total_var;
    try std.testing.expect(ratio > 0.90);
}
