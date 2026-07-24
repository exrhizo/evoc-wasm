//! Port of evoc/graph_construction.py: smooth_knn_dist,
//! compute_membership_strengths, neighbor_graph_matrix. The reference builds
//! the graph with scipy sparse; we have no scipy, so the sparse ops (COO ->
//! CSR with duplicate summation, transpose, elementwise multiply, add/sub,
//! eliminate_zeros) are implemented here. Output CSR is canonical: indices
//! sorted ascending within each row (matches the fixtures'
//! .tocsr().sort_indices()). See PORTING.md / PORT_NOTES.md.

const std = @import("std");
const Allocator = std.mem.Allocator;

const SMOOTH_K_TOLERANCE: f32 = 1e-5;
const MIN_K_DIST_SCALE: f32 = 1e-3;
const NPY_INFINITY: f32 = std.math.inf(f32);

pub const SmoothKnn = struct {
    sigma: []f32,
    rho: []f32,
    allocator: Allocator,

    pub fn deinit(self: *SmoothKnn) void {
        self.allocator.free(self.sigma);
        self.allocator.free(self.rho);
        self.* = undefined;
    }
};

/// Port of smooth_knn_dist. `distances` is n_samples x n_neighbors row-major.
///
/// PORT NOTE: psum/lo/mid/hi are pinned to f32 by the numba locals (fastmath
/// notwithstanding); accumulate in f32. Binary search: hi starts inf, mid
/// starts 1.0; while hi==inf the "too small" branch DOUBLES mid instead of
/// bisecting. mean_distances is a single mean over the whole matrix (used
/// only in the rho==0 fallback); mean_ith_distances is per-row. The psum
/// loop starts at j=1 (index 0 is the self/rho slot).
pub fn smooth_knn_dist(
    allocator: Allocator,
    distances: []const f32,
    n_samples: usize,
    n_neighbors: usize,
    k: f32,
    n_iter: usize,
    bandwidth: f32,
) !SmoothKnn {
    const target: f32 = @as(f32, @floatCast(std.math.log2(@as(f64, k)))) * bandwidth;
    const rho = try allocator.alloc(f32, n_samples);
    errdefer allocator.free(rho);
    @memset(rho, 0);
    const sigma = try allocator.alloc(f32, n_samples);
    errdefer allocator.free(sigma);
    @memset(sigma, 0);

    // np.mean over the whole matrix (f64 numpy accumulation; the value only
    // feeds a rho==0 fallback and a tolerance-scaled comparison).
    var total: f64 = 0.0;
    for (distances) |d| total += d;
    const mean_distances: f32 = @floatCast(total / @as(f64, @floatFromInt(distances.len)));

    for (0..n_samples) |i| { // prange, sequential here
        var lo: f32 = 0.0;
        var hi: f32 = NPY_INFINITY;
        var mid: f32 = 1.0;

        const ith_distances = distances[i * n_neighbors ..][0..n_neighbors];
        // non_zero_dists = ith_distances[ith_distances > 0.0]; rho = first one.
        for (ith_distances) |d| {
            if (d > 0.0) {
                rho[i] = d;
                break;
            }
        }

        for (0..n_iter) |_| {
            var psum: f32 = 0.0;
            for (1..n_neighbors) |j| {
                const d = ith_distances[j] - rho[i];
                if (d > 0) {
                    psum += @exp(-(d / mid));
                } else {
                    psum += 1.0;
                }
            }

            if (@abs(psum - target) < SMOOTH_K_TOLERANCE) {
                break;
            }

            if (psum > target) {
                hi = mid;
                mid = (lo + hi) / 2.0;
            } else {
                lo = mid;
                if (hi == NPY_INFINITY) {
                    mid *= 2;
                } else {
                    mid = (lo + hi) / 2.0;
                }
            }
        }

        sigma[i] = mid;

        if (rho[i] > 0.0) {
            var sum_ith: f32 = 0.0;
            for (ith_distances) |d| sum_ith += d;
            const mean_ith_distances = sum_ith / @as(f32, @floatFromInt(n_neighbors));
            if (sigma[i] < MIN_K_DIST_SCALE * mean_ith_distances) {
                sigma[i] = MIN_K_DIST_SCALE * mean_ith_distances;
            }
        } else {
            if (sigma[i] < MIN_K_DIST_SCALE * mean_distances) {
                sigma[i] = MIN_K_DIST_SCALE * mean_distances;
            }
        }
    }

    return SmoothKnn{ .sigma = sigma, .rho = rho, .allocator = allocator };
}

pub const MembershipStrengths = struct {
    rows: []i32,
    cols: []i32,
    vals: []f32,
    allocator: Allocator,

    pub fn deinit(self: *MembershipStrengths) void {
        self.allocator.free(self.rows);
        self.allocator.free(self.cols);
        self.allocator.free(self.vals);
        self.* = undefined;
    }
};

/// Port of compute_membership_strengths. Returns COO triplets of length
/// n_samples * n_neighbors (rows/cols pre-zeroed; the idx==-1 branch leaves a
/// slot untouched, so it stays (0, 0, 0.0) and is dropped later by
/// eliminate_zeros).
///
/// PORT NOTE: branch order is load-bearing — idx==-1 skips the slot entirely,
/// then idx==i => val=0, then (dist-rho)<=0 or sigma==0 => val=1, else exp.
/// The exp uses rhos[i] (== rho) and sigma; identical to the reference.
pub fn compute_membership_strengths(
    allocator: Allocator,
    knn_indices: []const i32,
    knn_dists: []const f32,
    sigmas: []const f32,
    rhos: []const f32,
    n_samples: usize,
    n_neighbors: usize,
) !MembershipStrengths {
    const size = n_samples * n_neighbors;
    const rows = try allocator.alloc(i32, size);
    errdefer allocator.free(rows);
    @memset(rows, 0);
    const cols = try allocator.alloc(i32, size);
    errdefer allocator.free(cols);
    @memset(cols, 0);
    const vals = try allocator.alloc(f32, size);
    errdefer allocator.free(vals);
    @memset(vals, 0);

    for (0..n_samples) |i| {
        const rho = rhos[i];
        const sigma = sigmas[i];
        for (0..n_neighbors) |j| {
            const idx = knn_indices[i * n_neighbors + j];
            var val: f32 = undefined;
            if (idx == -1) {
                continue; // We didn't get the full knn for i
            } else if (idx == @as(i32, @intCast(i))) {
                val = 0.0;
            } else if ((knn_dists[i * n_neighbors + j] - rho) <= 0.0 or sigma == 0.0) {
                val = 1.0;
            } else {
                val = @exp(-((knn_dists[i * n_neighbors + j] - rhos[i]) / sigma));
            }

            rows[i * n_neighbors + j] = @intCast(i);
            cols[i * n_neighbors + j] = idx;
            vals[i * n_neighbors + j] = val;
        }
    }

    return MembershipStrengths{ .rows = rows, .cols = cols, .vals = vals, .allocator = allocator };
}

/// CSR sparse matrix (square, n x n). Indices are sorted ascending within
/// each row (canonical form). Allocator-owned.
pub const CsrMatrix = struct {
    indptr: []i32,
    indices: []i32,
    data: []f32,
    n: usize,
    allocator: Allocator,

    pub fn deinit(self: *CsrMatrix) void {
        self.allocator.free(self.indptr);
        self.allocator.free(self.indices);
        self.allocator.free(self.data);
        self.* = undefined;
    }
};

/// Build a canonical CSR from COO triplets, summing duplicate (r, c) entries
/// (scipy's tocsr semantics) and sorting indices ascending within each row.
/// Explicit zeros are retained here — eliminate_zeros drops them separately.
fn csrFromCoo(
    allocator: Allocator,
    rows: []const i32,
    cols: []const i32,
    vals: []const f32,
    n: usize,
) !CsrMatrix {
    // Intermediate counting/bucketing indptr (distinct from the output CSR's
    // out_indptr); freed unconditionally once buckets are built.
    const indptr = try allocator.alloc(i32, n + 1);
    defer allocator.free(indptr);
    @memset(indptr, 0);

    // Count entries per row.
    for (rows) |r| indptr[@as(usize, @intCast(r)) + 1] += 1;
    for (1..n + 1) |i| indptr[i] += indptr[i - 1];

    const nnz_raw: usize = @intCast(indptr[n]);
    const tmp_cols = try allocator.alloc(i32, nnz_raw);
    defer allocator.free(tmp_cols);
    const tmp_vals = try allocator.alloc(f32, nnz_raw);
    defer allocator.free(tmp_vals);

    // Scatter into row buckets (unsorted within a row).
    const next = try allocator.alloc(i32, n);
    defer allocator.free(next);
    for (0..n) |i| next[i] = indptr[i];
    for (rows, cols, vals) |r, c, v| {
        const ur: usize = @intCast(r);
        const pos: usize = @intCast(next[ur]);
        tmp_cols[pos] = c;
        tmp_vals[pos] = v;
        next[ur] += 1;
    }

    // Sort each row by column, summing duplicates. Produce compacted CSR.
    var out_indptr = try allocator.alloc(i32, n + 1);
    errdefer allocator.free(out_indptr);
    out_indptr[0] = 0;
    // Worst case: no duplicates -> nnz_raw entries.
    var out_indices = try allocator.alloc(i32, nnz_raw);
    errdefer allocator.free(out_indices);
    var out_data = try allocator.alloc(f32, nnz_raw);
    errdefer allocator.free(out_data);

    var w: usize = 0;
    for (0..n) |i| {
        const start: usize = @intCast(indptr[i]);
        const end: usize = @intCast(indptr[i + 1]);
        // Insertion sort the row slice (rows are short: ~k entries).
        var a = start;
        while (a < end) : (a += 1) {
            const kc = tmp_cols[a];
            const kv = tmp_vals[a];
            var b: isize = @as(isize, @intCast(a)) - 1;
            while (b >= @as(isize, @intCast(start)) and tmp_cols[@intCast(b)] > kc) {
                tmp_cols[@intCast(b + 1)] = tmp_cols[@intCast(b)];
                tmp_vals[@intCast(b + 1)] = tmp_vals[@intCast(b)];
                b -= 1;
            }
            tmp_cols[@intCast(b + 1)] = kc;
            tmp_vals[@intCast(b + 1)] = kv;
        }
        // Merge duplicates (summing values, matching scipy's sum_duplicates).
        var j = start;
        while (j < end) {
            const c = tmp_cols[j];
            var acc = tmp_vals[j];
            var jj = j + 1;
            while (jj < end and tmp_cols[jj] == c) : (jj += 1) {
                acc += tmp_vals[jj];
            }
            out_indices[w] = c;
            out_data[w] = acc;
            w += 1;
            j = jj;
        }
        out_indptr[i + 1] = @intCast(w);
    }

    out_indices = try allocator.realloc(out_indices, w);
    out_data = try allocator.realloc(out_data, w);

    return CsrMatrix{ .indptr = out_indptr, .indices = out_indices, .data = out_data, .n = n, .allocator = allocator };
}

/// scipy eliminate_zeros: drop entries whose value is exactly 0.0, in place,
/// then shrink indices/data to the surviving nnz so slice lengths stay
/// canonical (== indptr[n]).
fn eliminateZeros(m: *CsrMatrix) !void {
    var w: usize = 0;
    // `start` tracks the ORIGINAL row boundary; m.indptr[i] gets overwritten
    // with the compacted offset each iteration, so we cannot re-read it.
    var start: usize = @intCast(m.indptr[0]);
    for (0..m.n) |i| {
        const end: usize = @intCast(m.indptr[i + 1]);
        for (start..end) |j| {
            if (m.data[j] != 0.0) {
                m.indices[w] = m.indices[j];
                m.data[w] = m.data[j];
                w += 1;
            }
        }
        start = end;
        m.indptr[i + 1] = @intCast(w);
    }
    m.indices = try m.allocator.realloc(m.indices, w);
    m.data = try m.allocator.realloc(m.data, w);
}

/// CSR transpose of a square matrix, returning canonical CSR (sorted rows).
fn csrTranspose(allocator: Allocator, m: *const CsrMatrix) !CsrMatrix {
    const n = m.n;
    const nnz: usize = @intCast(m.indptr[n]);
    const rows = try allocator.alloc(i32, nnz);
    defer allocator.free(rows);
    const cols = try allocator.alloc(i32, nnz);
    defer allocator.free(cols);
    const vals = try allocator.alloc(f32, nnz);
    defer allocator.free(vals);

    var e: usize = 0;
    for (0..n) |i| {
        const start: usize = @intCast(m.indptr[i]);
        const end: usize = @intCast(m.indptr[i + 1]);
        for (start..end) |j| {
            rows[e] = m.indices[j]; // transposed: old col becomes new row
            cols[e] = @intCast(i);
            vals[e] = m.data[j];
            e += 1;
        }
    }
    return csrFromCoo(allocator, rows, cols, vals, n);
}

/// Combine two canonical CSR matrices row-by-row: result = alpha*A op ...
/// Implements A + B - (A elementwise* B) in a single merge (the fuzzy-set
/// union). Both inputs must be canonical (sorted rows). Result is canonical.
fn fuzzyUnion(allocator: Allocator, a: *const CsrMatrix, b: *const CsrMatrix) !CsrMatrix {
    const n = a.n;
    const max_nnz: usize = @as(usize, @intCast(a.indptr[n])) + @as(usize, @intCast(b.indptr[n]));

    const out_indptr = try allocator.alloc(i32, n + 1);
    errdefer allocator.free(out_indptr);
    out_indptr[0] = 0;
    var out_indices = try allocator.alloc(i32, max_nnz);
    errdefer allocator.free(out_indices);
    var out_data = try allocator.alloc(f32, max_nnz);
    errdefer allocator.free(out_data);

    var w: usize = 0;
    for (0..n) |i| {
        var ja: usize = @intCast(a.indptr[i]);
        const ea: usize = @intCast(a.indptr[i + 1]);
        var jb: usize = @intCast(b.indptr[i]);
        const eb: usize = @intCast(b.indptr[i + 1]);

        while (ja < ea or jb < eb) {
            const ca: i32 = if (ja < ea) a.indices[ja] else std.math.maxInt(i32);
            const cb: i32 = if (jb < eb) b.indices[jb] else std.math.maxInt(i32);

            if (ca == cb) {
                // both present: a + b - a*b
                const av = a.data[ja];
                const bv = b.data[jb];
                out_indices[w] = ca;
                out_data[w] = av + bv - av * bv;
                w += 1;
                ja += 1;
                jb += 1;
            } else if (ca < cb) {
                out_indices[w] = ca;
                out_data[w] = a.data[ja]; // b term is 0 -> a + 0 - 0
                w += 1;
                ja += 1;
            } else {
                out_indices[w] = cb;
                out_data[w] = b.data[jb];
                w += 1;
                jb += 1;
            }
        }
        out_indptr[i + 1] = @intCast(w);
    }

    out_indices = try allocator.realloc(out_indices, w);
    out_data = try allocator.realloc(out_data, w);
    return CsrMatrix{ .indptr = out_indptr, .indices = out_indices, .data = out_data, .n = n, .allocator = allocator };
}

/// Port of neighbor_graph_matrix. Builds the weighted kNN graph and (when
/// symmetrize) applies the fuzzy-set union A + Aᵀ − A·Aᵀ. eliminate_zeros is
/// applied before AND after symmetrization (matching the reference). Returns
/// a canonical CSR (indices sorted ascending within each row).
///
/// PORT NOTE: n_neighbors is a FLOAT here (feeds smooth_knn_dist's target).
pub fn neighbor_graph_matrix(
    allocator: Allocator,
    n_neighbors: f32,
    knn_indices: []const i32,
    knn_dists_in: []const f32,
    n_samples: usize,
    n_k: usize,
    symmetrize: bool,
) !CsrMatrix {
    // knn_dists.astype(np.float32) — already f32 here.
    const knn_dists = knn_dists_in;

    var smooth = try smooth_knn_dist(allocator, knn_dists, n_samples, n_k, n_neighbors, 64, 1.0);
    defer smooth.deinit();

    var ms = try compute_membership_strengths(
        allocator,
        knn_indices,
        knn_dists,
        smooth.sigma,
        smooth.rho,
        n_samples,
        n_k,
    );
    defer ms.deinit();

    var result = try csrFromCoo(allocator, ms.rows, ms.cols, ms.vals, n_samples);
    try eliminateZeros(&result);

    if (symmetrize) {
        var transpose = try csrTranspose(allocator, &result);
        defer transpose.deinit();
        const unioned = try fuzzyUnion(allocator, &result, &transpose);
        result.deinit();
        result = unioned;
    }

    try eliminateZeros(&result);
    return result;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const fixture = @import("fixture.zig");

fn readMatrixF32(alloc: Allocator, arr: std.json.Array) !struct { data: []f32, rows: usize, cols: usize } {
    const rows = arr.items.len;
    const cols = if (rows > 0) arr.items[0].array.items.len else 0;
    const out = try alloc.alloc(f32, rows * cols);
    errdefer alloc.free(out);
    for (arr.items, 0..) |row, i| {
        for (row.array.items, 0..) |v, j| out[i * cols + j] = @floatCast(fixture.numberAsF64(v));
    }
    return .{ .data = out, .rows = rows, .cols = cols };
}

fn readMatrixI32(alloc: Allocator, arr: std.json.Array) !struct { data: []i32, rows: usize, cols: usize } {
    const rows = arr.items.len;
    const cols = if (rows > 0) arr.items[0].array.items.len else 0;
    const out = try alloc.alloc(i32, rows * cols);
    errdefer alloc.free(out);
    for (arr.items, 0..) |row, i| {
        for (row.array.items, 0..) |v, j| out[i * cols + j] = @intCast(v.integer);
    }
    return .{ .data = out, .rows = rows, .cols = cols };
}

fn readVecF32(alloc: Allocator, arr: std.json.Array) ![]f32 {
    const out = try alloc.alloc(f32, arr.items.len);
    for (arr.items, 0..) |v, i| out[i] = @floatCast(fixture.numberAsF64(v));
    return out;
}

test "smooth_knn_dist: single-row hi-doubling reaches convergence" {
    const alloc = std.testing.allocator;
    // one row, 4 neighbors, first is self (0), rest increasing.
    const dists = [_]f32{ 0.0, 1.0, 2.0, 3.0 };
    var s = try smooth_knn_dist(alloc, &dists, 1, 4, 4.0, 64, 1.0);
    defer s.deinit();
    try std.testing.expect(s.sigma[0] > 0.0);
    try std.testing.expectEqual(@as(f32, 1.0), s.rho[0]); // first nonzero
}

test "smooth_knn_dist: all-zero row -> rho 0" {
    const alloc = std.testing.allocator;
    const dists = [_]f32{ 0.0, 0.0, 0.0 };
    var s = try smooth_knn_dist(alloc, &dists, 1, 3, 3.0, 64, 1.0);
    defer s.deinit();
    try std.testing.expectEqual(@as(f32, 0.0), s.rho[0]);
}

test "compute_membership_strengths: -1 slot skipped, self-index 0, near gives 1" {
    const alloc = std.testing.allocator;
    // 2 samples, 3 neighbors
    const inds = [_]i32{ 0, 1, -1, 1, 0, 0 };
    const dists = [_]f32{ 0.0, 1.0, 2.0, 0.0, 1.0, 1.0 };
    const sig = [_]f32{ 1.0, 1.0 };
    const rho = [_]f32{ 0.0, 0.0 };
    var ms = try compute_membership_strengths(alloc, &inds, &dists, &sig, &rho, 2, 3);
    defer ms.deinit();
    // slot (0,0): idx==0==i -> val 0
    try std.testing.expectEqual(@as(f32, 0.0), ms.vals[0]);
    // slot (0,2): idx -1 -> untouched (0,0,0)
    try std.testing.expectEqual(@as(i32, 0), ms.rows[2]);
    try std.testing.expectEqual(@as(i32, 0), ms.cols[2]);
    try std.testing.expectEqual(@as(f32, 0.0), ms.vals[2]);
    // slot (1,0): idx 1==i -> val 0
    try std.testing.expectEqual(@as(f32, 0.0), ms.vals[3]);
}

test "fixture parity: smooth_knn_dist_n40_k15" {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "graph", "smooth_knn_dist_n40_k15");
    defer fx.deinit();
    const root = fx.value.object;
    const tol = fixture.toleranceForField(root.get("tolerance").?, "sigmas");
    const inputs = root.get("inputs").?.object;
    const m = try readMatrixF32(alloc, inputs.get("distances").?.array);
    defer alloc.free(m.data);
    const k: f32 = @floatCast(fixture.numberAsF64(inputs.get("k").?));
    const n_iter: usize = @intCast(inputs.get("n_iter").?.integer);
    const bandwidth: f32 = @floatCast(fixture.numberAsF64(inputs.get("bandwidth").?));

    var s = try smooth_knn_dist(alloc, m.data, m.rows, m.cols, k, n_iter, bandwidth);
    defer s.deinit();

    const expected = root.get("expected").?.object;
    const exp_sigmas = expected.get("sigmas").?.array;
    const exp_rhos = expected.get("rhos").?.array;
    for (0..m.rows) |i| {
        try fixture.expectF32(tol, s.sigma[i], fixture.numberAsF64(exp_sigmas.items[i]));
        try fixture.expectF32(tol, s.rho[i], fixture.numberAsF64(exp_rhos.items[i]));
    }
}

test "fixture parity: compute_membership_strengths_n40_k15" {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "graph", "compute_membership_strengths_n40_k15");
    defer fx.deinit();
    const root = fx.value.object;
    const tol_value = root.get("tolerance").?;
    const inputs = root.get("inputs").?.object;
    const inds = try readMatrixI32(alloc, inputs.get("knn_indices").?.array);
    defer alloc.free(inds.data);
    const dists = try readMatrixF32(alloc, inputs.get("knn_dists").?.array);
    defer alloc.free(dists.data);
    const sigmas = try readVecF32(alloc, inputs.get("sigmas").?.array);
    defer alloc.free(sigmas);
    const rhos = try readVecF32(alloc, inputs.get("rhos").?.array);
    defer alloc.free(rhos);

    var ms = try compute_membership_strengths(alloc, inds.data, dists.data, sigmas, rhos, inds.rows, inds.cols);
    defer ms.deinit();

    const expected = root.get("expected").?.object;
    const exp_rows = expected.get("rows").?.array;
    const exp_cols = expected.get("cols").?.array;
    const exp_vals = expected.get("vals").?.array;
    const val_tol = fixture.toleranceForField(tol_value, "vals");
    for (0..ms.rows.len) |i| {
        try std.testing.expectEqual(@as(i32, @intCast(exp_rows.items[i].integer)), ms.rows[i]);
        try std.testing.expectEqual(@as(i32, @intCast(exp_cols.items[i].integer)), ms.cols[i]);
        try fixture.expectF32(val_tol, ms.vals[i], fixture.numberAsF64(exp_vals.items[i]));
    }
}

fn checkGraphMatrixFixture(case: []const u8, symmetrize: bool) !void {
    const alloc = std.testing.allocator;
    const fx = try fixture.readFixture(alloc, "graph", case);
    defer fx.deinit();
    const root = fx.value.object;
    const tol_value = root.get("tolerance").?;
    const inputs = root.get("inputs").?.object;
    const inds = try readMatrixI32(alloc, inputs.get("knn_indices").?.array);
    defer alloc.free(inds.data);
    const dists = try readMatrixF32(alloc, inputs.get("knn_dists_transformed").?.array);
    defer alloc.free(dists.data);
    const n_neighbors: f32 = @floatCast(fixture.numberAsF64(inputs.get("n_neighbors").?));

    var g = try neighbor_graph_matrix(alloc, n_neighbors, inds.data, dists.data, inds.rows, inds.cols, symmetrize);
    defer g.deinit();

    const expected = root.get("expected").?.object;
    const exp_indptr = expected.get("indptr").?.array;
    const exp_indices = expected.get("indices").?.array;
    const exp_data = expected.get("data").?.array;
    const data_tol = fixture.toleranceForField(tol_value, "data");

    try std.testing.expectEqual(exp_indptr.items.len, g.indptr.len);
    for (exp_indptr.items, 0..) |v, i| {
        try std.testing.expectEqual(@as(i32, @intCast(v.integer)), g.indptr[i]);
    }
    try std.testing.expectEqual(exp_indices.items.len, g.indices.len);
    for (exp_indices.items, 0..) |v, i| {
        try std.testing.expectEqual(@as(i32, @intCast(v.integer)), g.indices[i]);
    }
    for (exp_data.items, 0..) |v, i| {
        try fixture.expectF32(data_tol, g.data[i], fixture.numberAsF64(v));
    }
}

test "fixture parity: neighbor_graph_matrix_symmetrize_true" {
    try checkGraphMatrixFixture("neighbor_graph_matrix_symmetrize_true", true);
}

test "fixture parity: neighbor_graph_matrix_symmetrize_false" {
    try checkGraphMatrixFixture("neighbor_graph_matrix_symmetrize_false", false);
}
