// Shared helpers for scripts/wasm_fixture_runner.mjs: fixture loading,
// tolerance-class comparison (per PORTING.md), and small stats/ARI helpers.
// No dependencies; pure ESM.

import { readFileSync } from "node:fs";
import path from "node:path";

export const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
export const FIXTURES_DIR = path.join(ROOT, "fixtures");

export function loadFixture(component, caseName) {
  const p = path.join(FIXTURES_DIR, component, `${caseName}.json`);
  return JSON.parse(readFileSync(p, "utf8"));
}

// -- Tolerance handling (PORTING.md "Numeric parity policy") ---------------

// tolSpec is either "exact", {rel, abs?}, or {default, fields: {field: tol}}
export function tolFor(tolSpec, fieldName) {
  if (tolSpec === "exact") return { exact: true };
  if (typeof tolSpec === "object" && tolSpec !== null) {
    if ("fields" in tolSpec) {
      const f = fieldName != null ? tolSpec.fields[fieldName] : undefined;
      return tolFor(f ?? tolSpec.default, fieldName);
    }
    if ("rel" in tolSpec || "abs" in tolSpec) {
      return { exact: false, rel: tolSpec.rel ?? 0, abs: tolSpec.abs ?? 0 };
    }
  }
  throw new Error(`unrecognized tolerance spec: ${JSON.stringify(tolSpec)}`);
}

// f32 round-trip: JSON doubles -> f32 bits, per PORTING.md.
const f32buf = new Float32Array(1);
export function toF32(x) {
  f32buf[0] = x;
  return f32buf[0];
}

export function closeF32(a, b, tol) {
  const af = toF32(a);
  const bf = toF32(b);
  if (tol.exact) return af === bf || (Number.isNaN(af) && Number.isNaN(bf));
  const diff = Math.abs(af - bf);
  return diff <= tol.abs + tol.rel * Math.abs(bf);
}

export function closeExactInt(a, b) {
  return Number(a) === Number(b);
}

// Compare two same-length numeric arrays as f32 under tol; returns
// {ok, firstBadIndex, firstBadDetail} for reporting.
export function compareF32Array(actual, expected, tol, label) {
  if (actual.length !== expected.length) {
    return { ok: false, detail: `${label}: length mismatch (${actual.length} vs ${expected.length})` };
  }
  for (let i = 0; i < actual.length; i++) {
    if (!closeF32(actual[i], expected[i], tol)) {
      return {
        ok: false,
        detail: `${label}[${i}]: got ${actual[i]}, expected ${expected[i]} (tol=${JSON.stringify(tol)})`,
      };
    }
  }
  return { ok: true };
}

export function compareIntArray(actual, expected, label) {
  if (actual.length !== expected.length) {
    return { ok: false, detail: `${label}: length mismatch (${actual.length} vs ${expected.length})` };
  }
  for (let i = 0; i < actual.length; i++) {
    if (Number(actual[i]) !== Number(expected[i])) {
      return { ok: false, detail: `${label}[${i}]: got ${actual[i]}, expected ${expected[i]}` };
    }
  }
  return { ok: true };
}

// -- Stats / ARI -------------------------------------------------------------

export function median(sorted) {
  const n = sorted.length;
  if (n === 0) return NaN;
  const mid = Math.floor(n / 2);
  return n % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function nChoose2(n) {
  return (n * (n - 1)) / 2;
}

// sklearn-compatible ARI (matches rust/core/tests/evoc_clusters_ari.rs's
// adjusted_rand_index): treats -1 (noise) as an ordinary label.
export function adjustedRandIndex(a, b) {
  if (a.length !== b.length) throw new Error("ARI: length mismatch");
  const n = a.length;
  const aLabels = [...new Set(a)].sort((x, y) => x - y);
  const bLabels = [...new Set(b)].sort((x, y) => x - y);
  const aIdx = new Map(aLabels.map((l, i) => [l, i]));
  const bIdx = new Map(bLabels.map((l, i) => [l, i]));
  const contingency = new Array(aLabels.length * bLabels.length).fill(0);
  for (let i = 0; i < n; i++) {
    const ai = aIdx.get(a[i]);
    const bi = bIdx.get(b[i]);
    contingency[ai * bLabels.length + bi] += 1;
  }
  let sumCombC = 0;
  for (const c of contingency) sumCombC += nChoose2(c);
  const rowSums = new Array(aLabels.length).fill(0);
  const colSums = new Array(bLabels.length).fill(0);
  for (let i = 0; i < aLabels.length; i++) {
    for (let j = 0; j < bLabels.length; j++) {
      rowSums[i] += contingency[i * bLabels.length + j];
      colSums[j] += contingency[i * bLabels.length + j];
    }
  }
  const sumCombRow = rowSums.reduce((s, r) => s + nChoose2(r), 0);
  const sumCombCol = colSums.reduce((s, c) => s + nChoose2(c), 0);
  const totalComb = nChoose2(n);
  const expectedIndex = (sumCombRow * sumCombCol) / totalComb;
  const maxIndex = 0.5 * (sumCombRow + sumCombCol);
  if (Math.abs(maxIndex - expectedIndex) < 1e-12) return 1.0;
  return (sumCombC - expectedIndex) / (maxIndex - expectedIndex);
}

// -- Minimal test runner ------------------------------------------------------

export class Reporter {
  constructor(lang) {
    this.lang = lang;
    this.results = [];
  }
  record(name, status, detail) {
    this.results.push({ name, status, detail });
    const tag = status === "pass" ? "ok" : status === "skip" ? "skip" : "FAIL";
    const suffix = detail ? ` - ${detail}` : "";
    console.log(`[${this.lang}] ${tag} - ${name}${suffix}`);
  }
  pass(name, detail) {
    this.record(name, "pass", detail);
  }
  fail(name, detail) {
    this.record(name, "fail", detail);
  }
  skip(name, detail) {
    this.record(name, "skip", detail);
  }
  async run(name, fn) {
    try {
      await fn();
      this.pass(name);
    } catch (e) {
      this.fail(name, e && e.stack ? e.stack.split("\n")[0] : String(e));
    }
  }
  summary() {
    const pass = this.results.filter((r) => r.status === "pass").length;
    const fail = this.results.filter((r) => r.status === "fail").length;
    const skip = this.results.filter((r) => r.status === "skip").length;
    return { pass, fail, skip, total: this.results.length };
  }
}

export function assert(cond, msg) {
  if (!cond) throw new Error(msg || "assertion failed");
}
