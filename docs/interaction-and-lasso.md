# Interaction and Lasso

`hyper-scatter` separates renderer state from input handling.

- The renderer owns view math, hit testing, lasso selection, and emphasis rendering.
- The 2D `createInteractionController()` helper wires common DOM input patterns to a `Renderer`.
- 3D input handling is currently host-managed.

## 2D Interaction Controller

```ts
import { createInteractionController } from "hyper-scatter";

const controller = createInteractionController(canvas, plot, {
  lassoPredicate: (event) => event.shiftKey,
  onHover: (hit) => {
    console.log(hit?.index ?? null);
  },
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

Supported controller options:

- `observeResize`: keep renderer size in sync with the canvas using `ResizeObserver`
- `wheelZoomScale`: customize wheel-to-zoom sensitivity
- `lassoPredicate`: choose the gesture that enters lasso mode
- `lassoMinSampleDistPx`: screen-space point sampling threshold while dragging
- `lassoMaxVertsInteraction`: polygon simplification budget during drag
- `lassoMaxVertsFinal`: polygon simplification budget on completion
- `onHover`: hover callback after hit testing
- `onLassoUpdate`: callback during drag with both data-space and screen-space polygons
- `onLassoComplete`: callback with the final `SelectionResult`

Default lasso gesture:

- `Shift` + `Meta` drag
- `Shift` + `Ctrl` drag

If you want `Shift`-drag instead, override `lassoPredicate`.

## Emphasis States

The renderer has three emphasis channels:

- `setSelection(indices)` for the primary active subset
- `setHighlight(indices)` for secondary emphasis such as neighbors or search hits
- `setHovered(index)` for transient pointer feedback

You can dim everything else with `setInactiveOpacity(alpha)`.

```ts
plot.setSelection(new Set([12, 18]));
plot.setHighlight(new Set([4, 5, 6, 7]));
plot.setInactiveOpacity(0.3);
plot.render();
```

Customize ring treatment with `setInteractionStyle()`:

```ts
plot.setInteractionStyle({
  selectionColor: "#f59e0b",
  selectionRadiusOffset: 2,
  selectionRingWidth: 2,
  highlightColor: "#94a3b8",
  highlightRadiusOffset: 1,
  highlightRingWidth: 1.5,
  hoverColor: "#ffffff",
  hoverFillColor: null,
  hoverRadiusOffset: 3,
});
plot.render();
```

## Lasso Overlay vs Selection Result

These are separate concerns.

### Drawing the active polygon

Use `setLassoPolygon()` while the user is dragging:

```ts
plot.setLassoPolygon(screenPolygon, {
  strokeColor: "#4f46e5",
  strokeWidth: 2,
  fillColor: "rgba(79, 70, 229, 0.15)",
});
plot.render();
```

Clear it with:

```ts
plot.setLassoPolygon(null);
plot.render();
```

### Computing the selected points

Call `lassoSelect(screenPolygon)` directly, or use the controller callback.

2D renderers may return:

- `kind: "indices"` with an explicit `Set<number>`
- `kind: "geometry"` with a data-space polygon-backed predicate

If you only need the final count, use `countSelection()`:

```ts
const count = await plot.countSelection(result, { yieldEveryMs: 0 });
```

3D renderers always return index-based selections.

## 3D Input Handling

For 3D renderers, wire your own pointer events and call the renderer directly:

- `pan(deltaX, deltaY, modifiers)`
- `zoom(anchorX, anchorY, delta, modifiers)`
- `hitTest(screenX, screenY)`
- `lassoSelect(polyline)`

That keeps the public renderer surface aligned across 2D and 3D, even though only 2D ships with a built-in controller today.