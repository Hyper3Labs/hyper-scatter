# Selection Rendering Plan

Goal: make emphasis states first-class in hyper-scatter so host apps can express focus inside the renderer instead of layering a second canvas on top.

This is intentionally narrow. It covers renderer-side selection, highlight, hover, and inactive-state rendering. It does not try to redesign the full interaction system.

## Problems To Solve

- The renderer only exposes one selection set, so apps have to merge "selected anchor" and "related points" into the same visual state.
- Global dimming also affects the emphasized points, which weakens the main focus state.
- Ring radius and ring weight are partly hardcoded.
- Stronger emphasis currently requires host-side overlay drawing.
- There is no per-point size channel for lightweight local emphasis.

## Model

Keep these concerns separate:

- `selection`: the primary active subset.
- `highlight`: secondary context, for example neighbors or search matches.
- `hover`: transient pointer feedback.
- `inactive opacity`: how much everything else is dimmed.

Filtering and visibility are separate concerns and should stay out of this document.

## Proposed API

```ts
interface InteractionStyle {
  selectionColor?: string;
  selectionRadiusOffset?: number;
  selectionRingWidth?: number;

  highlightColor?: string;
  highlightRadiusOffset?: number;
  highlightRingWidth?: number;

  hoverColor?: string;
  hoverFillColor?: string | null;
  hoverRadiusOffset?: number;
}

interface Renderer {
  setSelection(indices: Set<number> | null): void;
  setHighlight(indices: Set<number> | null): void;
  setInactiveOpacity(alpha: number): void;
  setInteractionStyle(style: InteractionStyle): void;

  // Optional phase 2 channel for local emphasis.
  setPointSizes(sizes: Float32Array | null): void;
}
```

Notes:

- `setInactiveOpacity()` is a better long-term API than overloading `setCategoryAlpha()` with selection-specific behavior.
- `setPointSizes()` should stay optional. The main fix is the emphasis model, not a broad style-channel redesign.

## Rendering Rules

- Selection is visually stronger than highlight.
- Hover is transient and should render above both selection and highlight.
- Inactive opacity should never dim the hovered point.
- Inactive opacity should not dim the selected point, and should usually skip highlighted points as well.
- Rings are additive. The underlying point color should remain visible.
- The default styling should work well without host-side overlays.

## Phasing

### Phase 1: Fix The Current Pain

Add `setHighlight()`, `setInactiveOpacity()`, and configurable selection/highlight ring styling.

This is the minimum change set that lets HyperView stop merging anchor selection with neighbor highlighting.

### Phase 2: Add Per-Point Size

Add `setPointSizes()` for cases where a host wants a stronger emphasis treatment without a separate overlay pass.

This is useful, but it should follow the simpler renderer-side fixes above.

## Related Follow-Up

If the goal is to remove the overlay canvas entirely, the renderer should also support drawing the active lasso polygon itself.

That can live as a separate API:

```ts
interface LassoStyle {
  strokeColor?: string;
  strokeWidth?: number;
  fillColor?: string;
}

interface Renderer {
  setLassoPolygon(polygon: Float32Array | null, style?: LassoStyle): void;
}
```

That is adjacent to selection rendering, but it is a separate concern and should not block Phase 1.

## Non-Goals

- Rectangle or brush selection modes.
- Point-level filtering and visibility masks.
- Theme presets.
- A full event system redesign.
- Data-format convenience helpers.

## Success Criteria

- A host app can distinguish primary selection from secondary highlight without merging sets.
- Dimming preserves the readability of the emphasized points.
- Hosts can tune ring size and weight without custom overlay drawing.
- HyperView can remove the point-emphasis overlay path.
- The API still reads like a general scatter library API, not a HyperView-specific patch.