# hyper-scatter

[![npm version](https://img.shields.io/npm/v/hyper-scatter.svg?style=flat-square)](https://www.npmjs.com/package/hyper-scatter)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

High-performance WebGL2 scatterplots for embedding exploration across Euclidean, Poincare, and spherical spaces.

<p align="center">
  <a href="https://hackerroomai.github.io/hyper-scatter/"><strong>Try the interactive demo and benchmark playground</strong></a>
</p>

<p align="center">
  <img src="docs/poincare_demo.gif" alt="Poincaré disk demo" width="480">
  <br>
  <em>Poincaré pan and zoom with geometry-aware interaction</em>
</p>

`hyper-scatter` is a low-level canvas renderer for large 2D and 3D embedding datasets.

- 2D geometries: `euclidean`, `poincare`
- 3D geometries: `euclidean3d`, `sphere`
- Built-in hit testing, lasso selection, hover, selection, and secondary highlight states
- Runtime styling updates without recreating the renderer
- Pure WebGL2 with no runtime dependencies

It is used inside [HyperView](https://github.com/HackerRoomAI/HyperView), but the package is designed to be used directly in your own app.

## Install

```bash
npm install hyper-scatter
```

## Quickstart

Make sure the canvas has a real CSS size before initialization.

```html
<canvas id="plot" style="width: 100%; height: 480px; display: block;"></canvas>
```

```ts
import {
  createDatasetFromColumns,
  createInteractionController,
  createScatterPlot,
} from "hyper-scatter";

const canvas = document.querySelector("#plot");
if (!(canvas instanceof HTMLCanvasElement)) {
  throw new Error("Missing canvas element");
}

const x = new Float32Array([0.15, -0.3, 0.4, -0.55, 0.18, -0.08]);
const y = new Float32Array([0.2, 0.1, -0.25, 0.42, 0.28, -0.4]);
const labels = new Uint16Array([0, 1, 1, 2, 0, 2]);

const rect = canvas.getBoundingClientRect();

const plot = createScatterPlot(canvas, {
  geometry: "poincare",
  width: Math.max(1, Math.floor(rect.width)),
  height: Math.max(1, Math.floor(rect.height)),
  devicePixelRatio: window.devicePixelRatio,
  pointRadius: 3,
  backgroundColor: "#0b1020",
  colors: ["#a7f3d0", "#60a5fa", "#f59e0b"],
  dataset: createDatasetFromColumns("poincare", x, y, labels),
});

plot.setInteractionStyle({
  selectionColor: "#f59e0b",
  highlightColor: "#94a3b8",
  hoverColor: "#ffffff",
});

plot.render();

const controller = createInteractionController(canvas, plot, {
  lassoPredicate: (event) => event.shiftKey,
  onHover: (hit) => {
    console.log("hovered point", hit?.index ?? null);
  },
  onLassoUpdate: (_dataPolygon, screenPolygon) => {
    plot.setLassoPolygon(screenPolygon, {
      strokeColor: "#4f46e5",
      fillColor: "rgba(79, 70, 229, 0.15)",
    });
    plot.render();
  },
  onLassoComplete: async (result, _dataPolygon, screenPolygon) => {
    plot.setLassoPolygon(screenPolygon);

    if (result.kind === "indices" && result.indices) {
      plot.setSelection(result.indices);
      plot.setInactiveOpacity(result.indices.size > 0 ? 0.35 : 1);
      plot.render();
      return;
    }

    const count = await plot.countSelection(result, { yieldEveryMs: 0 });
    console.log("lasso selected", count, "points");
    plot.render();
  },
});

window.addEventListener("resize", () => {
  const next = canvas.getBoundingClientRect();
  plot.resize(
    Math.max(1, Math.floor(next.width)),
    Math.max(1, Math.floor(next.height)),
  );
  plot.render();
});

// Later:
// controller.destroy();
// plot.destroy();
```

The same factory works for 3D renderers. Use `createDataset3D()` or `createDataset3DFromColumns()` with `geometry: "euclidean3d"` or `geometry: "sphere"`.

## Geometry Modes

| Geometry token | Dimension | Helper | Notes |
|---|---|---|---|
| `euclidean` | 2D | `createDataset`, `createDatasetFromColumns` | Standard planar scatterplot |
| `poincare` | 2D | `createDataset`, `createDatasetFromColumns` | Hyperbolic embeddings in the Poincaré disk |
| `euclidean3d` | 3D | `createDataset3D`, `createDataset3DFromColumns` | Orthographic orbit camera |
| `sphere` | 3D | `createDataset3D`, `createDataset3DFromColumns` | Unit-sphere layouts with optional guide rendering |

See [docs/geometries.md](docs/geometries.md) for view-state and styling details.

## Emphasis and Display State

You can update renderer state without rebuilding the dataset:

```ts
plot.setSelection(new Set([1, 4]));
plot.setHighlight(new Set([0, 2, 3]));
plot.setInactiveOpacity(0.3);
plot.setPalette(["#d1fae5", "#93c5fd", "#fdba74"]);
plot.setCategoryVisibility([1, 1, 0]);
plot.render();
```

Important details:

- `setSelection()` is the primary emphasis channel.
- `setHighlight()` is a secondary emphasis channel for neighbors, search hits, or related points.
- `setInactiveOpacity()` dims non-emphasized visible points while keeping the emphasized states readable.
- `setCategoryAlpha()` remains available as a legacy alias for `setInactiveOpacity()`.

## Lasso and Interaction

`createInteractionController()` is the packaged input controller for 2D renderers.

- Default lasso gesture: `Shift` + `Meta` or `Ctrl` drag
- Override `lassoPredicate` if you want `Shift`-drag or another gesture
- Use `onLassoUpdate()` to draw a renderer-owned polygon overlay with `setLassoPolygon()`
- Use `onLassoComplete()` to apply `setSelection()` or to inspect the returned `SelectionResult`

2D selections may be returned as explicit indices or as a geometry-backed predicate. If you only need an exact count, call `countSelection()` instead of scanning the full dataset yourself.

3D renderers expose the same selection and hover methods, but input handling is currently host-driven rather than shipped through `createInteractionController()`.

See [docs/interaction-and-lasso.md](docs/interaction-and-lasso.md) for the full interaction model.

## Guides

- [docs/getting-started.md](docs/getting-started.md)
- [docs/geometries.md](docs/geometries.md)
- [docs/interaction-and-lasso.md](docs/interaction-and-lasso.md)

## Benchmarks

Measured through the browser harness in headed mode on a `1125x400` canvas at `1x` DPR.

| Geometry | Points | FPS (avg) |
|---|---:|---:|
| Euclidean | 20,000,000 | 59.9 |
| Poincaré | 20,000,000 | 59.9 |

Reproduce the stress run:

```bash
npm run bench -- --points=20000000
```

Useful commands:

- `npm run bench` for the default WebGL benchmark sweep
- `npm run bench -- --geometries=euclidean,poincare,euclidean3d,sphere` for all geometry modes
- `npm run bench:accuracy` for the 2D reference-vs-candidate accuracy harness

Headed runs are the source of truth for performance numbers. Headless runs are fine for smoke checks, but not for serious benchmarking.

## Status

- [x] Euclidean 2D WebGL renderer
- [x] Poincaré 2D WebGL renderer
- [x] Euclidean 3D WebGL renderer
- [x] Spherical 3D WebGL renderer
- [x] 2D interaction controller with lasso callbacks
- [ ] 3D packaged interaction controller
- [ ] 3D reference accuracy harness

## License

MIT © [Matin Mahmood](https://www.linkedin.com/in/matin-mahmood/) (X: [@MatinMnM](https://twitter.com/MatinMnM))
