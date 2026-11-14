<p align="center">
  <img src="https://raw.githubusercontent.com/TutteInstitute/evoc/main/doc/evoc_logo_horizontal.png" height="72" alt="EVoC">
  &nbsp;&nbsp;<b>×</b>&nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/WebAssembly/web-assembly-logo/main/dist/icon/web-assembly-icon-128px.png" height="72" alt="WebAssembly">
</p>

<h1 align="center">EVoC WASM</h1>

<p align="center">
  <a href="https://github.com/TutteInstitute/evoc">EVoC</a> embedding clustering, ported from scratch to <b>Zig</b> and <b>Rust</b>, running in the browser.
</p>

---

Both ports implement the full EVoC pipeline: k-NN graph (RP-trees + NN-descent),
node embedding, Boruvka MST, condensed tree, and multi-granularity cluster layers.
They match the Python reference against shared fixtures (bit-exact per stage,
ARI-checked end to end).

**Status:** both ports work and are tested. No npm package yet.

## Zig vs Rust

Same algorithm, same fixtures, two languages. Single-threaded, Node 20:

| | Zig | Rust |
|---|---|---|
| `.wasm` size (gzip) | **73 KB** | 89 KB |
| wasm, n=800 | **33 ms** | 65 ms |
| wasm, n=100 | **4.4 ms** | 37 ms |
| native, n=800 | 27 ms | **25 ms** |
| wasm, n=30,000 | **2.0 s** | 2.1–2.8 s |
| wasm memory, n=30,000 | 1.1 GB | **0.56 GB** |

- **Natively they're about the same speed. In wasm, Zig wins by 2–8x.** Rust's
  `wasm32-unknown-unknown` build is up to 11x slower than its own native build.
  Zig's stays close to 1x. Changing `panic=abort`, target features, the
  allocator or the opt level didn't close the gap, so the cause is still open.
- **Zig makes the smaller binary** (`ReleaseSmall`). `ReleaseFast` was only ~5%
  faster but 12x larger.
- **Rust uses about half the memory.** Zig's `WasmAllocator` takes most of its
  memory on the first call (wasm memory never shrinks). It isn't a leak.
- **Rust exposes more functions to wasm** (k-NN graph, node embedding,
  cluster trees). Zig only exposes `evoc_fit_predict` and a few others.
- For reference, single-threaded Python/numba takes 34 ms at n=800 and 1.8 s at
  n=30,000.

Full numbers and methods: [`comparison/results.md`](comparison/results.md).

## Where it could be faster

Profile of the full pipeline: `knn_graph` ~45%, `node_embedding` ~22%,
`label_propagation` ~21%.

1. **Run NN-descent on several threads.** It's ~40% of total time, and it's the
   main reason multi-threaded Python (24 threads) still wins at large n. Plan:
   one shared `WebAssembly.Memory` split into fixed regions, with one region
   per worker. Each worker writes candidates to its own buffer, and the
   buffers are merged between rounds. Same approach as numba. See
   [`RESUME.md`](RESUME.md).
2. **Thread the SGD in `node_embedding`** (Hogwild-style). The results won't
   be bit-exact, so it would be validated by ARI instead.
3. **Use SIMD.** The distance kernels and SGD loops are scalar. Turning on
   `simd128` alone does nothing; the hot loops need rewriting with `@Vector` /
   `core::arch`.
4. **Run `wasm-opt -O3`.** It hasn't been applied yet.
5. **Find out why Rust's wasm build is slow**: profile each stage, then
   compare the hot functions between the two binaries.
6. **Reduce peak memory**: reuse scratch buffers between stages.

Threading RP-tree construction is already prototyped in both languages (2–2.4x
faster for that stage), but it's only ~1–2% of the total.

## Build

```sh
# Zig 0.13
cd zig && zig build test && zig build wasm        # zig-out/bin/evoc_core.wasm

# Rust
cd rust && cargo test --release
cargo build --release --target wasm32-unknown-unknown -p evoc-core
```

Wasm APIs: [`zig/WASM_API.md`](zig/WASM_API.md), [`rust/core/WASM_API.md`](rust/core/WASM_API.md).
Porting notes: [`PORTING.md`](PORTING.md).

## License

BSD-2-Clause, the same license as [EVoC](https://github.com/TutteInstitute/evoc),
© Tutte Institute for Mathematics and Computing. See [LICENSE](LICENSE).
