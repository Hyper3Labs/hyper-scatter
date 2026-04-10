# Getting Started

`hyper-scatter` is a low-level canvas renderer. You provide the canvas, data, and app state, and the renderer handles projection, hit testing, and drawing.

## Installation

```bash
npm install hyper-scatter
```

## 2D Example

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
  geometry: "euclidean",
  width: Math.max(1, Math.floor(rect.width)),
  height: Math.max(1, Math.floor(rect.height)),
  devicePixelRatio: window.devicePixelRatio,
  pointRadius: 3,
  backgroundColor: "#0b1020",
  colors: ["#a7f3d0", "#60a5fa", "#f59e0b"],
  dataset: createDatasetFromColumns("euclidean", x, y, labels),
});

plot.render();

const controller = createInteractionController(canvas, plot, {
  lassoPredicate: (event) => event.shiftKey,
  onLassoUpdate: (_dataPolygon, screenPolygon) => {
    plot.setLassoPolygon(screenPolygon);
    plot.render();
  },
  onLassoComplete: (result, _dataPolygon, screenPolygon) => {
    plot.setLassoPolygon(screenPolygon);

    if (result.kind === "indices" && result.indices) {
      plot.setSelection(result.indices);
      plot.setInactiveOpacity(result.indices.size > 0 ? 0.35 : 1);
    }

    plot.render();
  },
});
```

## 3D Example

```ts
import {
  createDataset3DFromColumns,
  createScatterPlot,
} from "hyper-scatter";

const canvas = document.querySelector("#plot3d");
if (!(canvas instanceof HTMLCanvasElement)) {
  throw new Error("Missing canvas element");
}

const x = new Float32Array([1, 0, -1, 0]);
const y = new Float32Array([0, 1, 0, -1]);
const z = new Float32Array([0, 0, 0.25, -0.25]);
const labels = new Uint16Array([0, 0, 1, 1]);

const rect = canvas.getBoundingClientRect();

const plot3d = createScatterPlot(canvas, {
  geometry: "sphere",
  width: Math.max(1, Math.floor(rect.width)),
  height: Math.max(1, Math.floor(rect.height)),
  devicePixelRatio: window.devicePixelRatio,
  pointRadius: 3,
  backgroundColor: "#0b1020",
  colors: ["#a7f3d0", "#60a5fa"],
  sphereGuideColor: "#94a3b8",
  sphereGuideOpacity: 0.2,
  dataset: createDataset3DFromColumns("sphere", x, y, z, labels),
});

plot3d.render();
```

3D interaction is host-managed today. Use the renderer methods directly if you want to wire pointer gestures:

- `pan(deltaX, deltaY, modifiers)`
- `zoom(anchorX, anchorY, delta, modifiers)`
- `hitTest(screenX, screenY)`
- `lassoSelect(polyline)`

## Resize and Cleanup

The renderer does not own layout. Resize it when the canvas size changes:

```ts
function resizePlot(plot: { resize: (width: number, height: number) => void; render: () => void }) {
  const rect = canvas.getBoundingClientRect();
  plot.resize(
    Math.max(1, Math.floor(rect.width)),
    Math.max(1, Math.floor(rect.height)),
  );
  plot.render();
}
```

Cleanup:

```ts
controller?.destroy();
plot.destroy();
plot3d.destroy();
```

## Common Gotchas

- The canvas must have non-zero CSS size before you call `createScatterPlot()`.
- The geometry token passed to `createScatterPlot()` must match the dataset geometry.
- State changes such as `setSelection()` or `setHighlight()` do not schedule rendering for you. Call `render()` after mutating renderer state.
- `createInteractionController()` is for 2D renderers only.