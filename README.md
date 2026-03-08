# hyper-scatter

<!-- badges -->
[![npm version](https://img.shields.io/npm/v/hyper-scatter.svg?style=flat-square)](https://www.npmjs.com/package/hyper-scatter)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

**Hyperbolic (Poincaré) embeddings at 60 FPS. 20,000,000 points. Pure WebGL2 (no `regl`, no `three.js`).**

<p align="center">
  <a href="https://hackerroomai.github.io/hyper-scatter/"><strong>🚀 Try the Interactive Demo & Benchmark Playground →</strong></a>
</p>

<p align="center">
  <img src="docs/poincare_demo.gif" alt="Poincaré Disk Demo" width="480">
  <br>
  <em>Hyperbolic pan & zoom in the Poincaré disk — points follow geodesics</em>
</p>

A specialized scatterplot engine for [HyperView](https://github.com/HackerRoomAI/HyperView).

- Geometries: **Poincaré (hyperbolic)** + **Euclidean** in 2D, plus WebGL candidates for **Euclidean 3D** and **Sphere**.
- Correctness: a slow CPU **Reference** defines 2D semantics; the fast GPU **Candidate** must match.
- Implementation: **pure WebGL2** (no `regl`, no `three.js`, no runtime deps).

---

## Poincaré (Hyperbolic) semantics

This is the part most scatterplot libs don’t have.

- **View state:** Möbius isometry parameter $a=(a_x,a_y)$ with $|a|<1$, plus a separate `displayZoom` scalar.
- **Pan:** anchor-invariant (the point under the cursor stays under the cursor).
- **Zoom:** anchored zoom; `displayZoom` scales the visual disk without changing the underlying isometry.
- **Project / unproject:** stable round-trips, shared between Reference + Candidate math.
- **Hit-test:** hyperbolic-aware, disk-culls correctly, deterministic tie-breaking.
- **Lasso:** selection polygon is unprojected to data space; membership is verified against the Reference.

For the full invariants + how the harness selects candidate code paths, see [AGENTS.md](AGENTS.md).

## Usage (copy/paste agent prompt)

```text
You are a coding agent; integrate `hyper-scatter` in my repo: install `npm install hyper-scatter`; create `mountHyperScatter(canvas, params)` (or React hook) using `EuclideanWebGLCandidate`, `HyperbolicWebGLCandidate`, `Euclidean3DWebGLCandidate`, `Spherical3DWebGLCandidate`, `createDataset`, `createDataset3D`, and `createInteractionController`; choose renderer by `params.geometry` (`euclidean`, `poincare`, `euclidean3d`, `sphere`); ensure non-zero CSS size, then `const rect = canvas.getBoundingClientRect()` and `renderer.init(canvas, { width: Math.max(1, Math.floor(rect.width)), height: Math.max(1, Math.floor(rect.height)), devicePixelRatio: window.devicePixelRatio })`; set dataset with `createDataset` for 2D or `createDataset3D` for 3D, then `renderer.render()`; wire `createInteractionController(canvas, renderer, { onHover, onLassoComplete })` only for 2D and optionally call `await renderer.countSelection(result, { yieldEveryMs: 0 })` for exact lasso counts; cleanup with `controller?.destroy(); renderer.destroy();`; return concrete code changes + file paths + a minimal example passing `Float32Array` positions (`[x,y,...]` or `[x,y,z,...]`) and optional `Uint16Array` labels.
```

## API Highlights

Runtime updates (no renderer re-creation): `setPalette`, `setCategoryVisibility`, `setCategoryAlpha`, `setInteractionStyle`.

| Dimension | Geometry token | Geometry | Candidate class | Dataset helper |
|---|---|---|---|---|
| 2D | `euclidean` | Euclidean | `EuclideanWebGLCandidate` | `createDataset` |
| 2D | `poincare` | Poincare (hyperbolic disk) | `HyperbolicWebGLCandidate` | `createDataset` |
| 3D | `euclidean3d` | Euclidean 3D | `Euclidean3DWebGLCandidate` | `createDataset3D` |
| 3D | `sphere` | Hypersphere (unit sphere) | `Spherical3DWebGLCandidate` | `createDataset3D` |

Notes:
- Hidden categories are excluded from `render`, `hitTest`, and `lassoSelect`.
- `createInteractionController()` targets the 2D `Renderer` interface.
- 2D `SelectionResult` supports `kind: 'indices' | 'geometry'`; `SelectionResult3D` is index-based.
- 3D helper exports include `packPositionsXYZ`.

## Benchmarks

Main claim, measured via the browser harness (headed):

Config note: canvas `1125x400 @ 1x DPR` (Puppeteer).

| Geometry | Points | FPS (avg) |
|---|---:|---:|
| Euclidean | 20,000,000 | 59.9 |
| Poincaré | 20,000,000 | 59.9 |

Run the stress benchmark that reproduces the rows above:

```bash
npm run bench -- --points=20000000
```

Default sweep (smaller point counts): `npm run bench`

Additional benchmark options:

- `npm run bench -- --geometries=euclidean,poincare,euclidean3d,sphere` runs the WebGL candidate benchmark across 2D and 3D geometries.
- `npm run bench -- --renderer=reference` and `npm run bench:accuracy` remain 2D-only.

Note: for performance numbers, run headed (default). Headless runs can skew GPU timing.

## How we built it

I (Matin) only knew Python. So we built this as a lab with a clear loop.

Roles:

- Matin: architect/product (Python-first).
- Claude: harness/environment engineer (benchmarks + correctness tests + reference semantics).
- Codex: implementation engineer (WebGL2 candidate).

### 1) Reference first

- Write non-performant, readable Canvas2D renderers (`src/impl_reference/`).
- Treat them as semantics: projection, pan/zoom, hit-test, lasso.

### 2) Harness as the reward function

- Accuracy compares Reference vs Candidate for: project/unproject, pan/zoom invariance, hit-test, lasso.
- Performance tracks: FPS, pan/hover FPS, hit-test time, lasso time.

### 3) Candidate optimization

- Implement the WebGL2 candidate (`src/impl_candidate/`).
- Speed comes from GPU rendering + spatial indexing + adaptive quality.

### 4) Reward hacking notes

If you give an agent a benchmark, it will try to win.

- Editing the harness/tolerances instead of fixing precision.
- Making lasso “async” so the timer looks better.

The harness tries to reduce these paths (example: lasso timing is end-to-end and includes the work required to get an exact selected-count).

## Status & Roadmap

- [x] Euclidean Geometry
- [x] Poincaré Disk (Hyperbolic) Geometry
- [x] 3D WebGL candidates (`euclidean3d`, `sphere`)
- [ ] 3D reference renderer + accuracy harness

## License

MIT © [Matin Mahmood](https://www.linkedin.com/in/matin-mahmood/) (X: [@MatinMnM](https://twitter.com/MatinMnM))
