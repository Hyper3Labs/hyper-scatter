# Geometries

`hyper-scatter` supports four geometry tokens. The geometry determines which renderer is created, which dataset helper you use, and how the camera behaves.

## Overview

| Geometry token | Dimension | Dataset helper | View model | Notes |
|---|---|---|---|---|
| `euclidean` | 2D | `createDataset`, `createDatasetFromColumns` | `EuclideanViewState` | Standard planar scatterplot |
| `poincare` | 2D | `createDataset`, `createDatasetFromColumns` | `HyperbolicViewState` | Hyperbolic embeddings in the Poincaré disk |
| `euclidean3d` | 3D | `createDataset3D`, `createDataset3DFromColumns` | `OrbitViewState3D` | Orthographic camera and free orbit |
| `sphere` | 3D | `createDataset3D`, `createDataset3DFromColumns` | `OrbitViewState3D` | Unit-sphere rendering with optional guide lines |

## Euclidean 2D

Use `euclidean` for standard scatterplots in planar coordinates.

Relevant init options:

- `backgroundColor`
- `pointRadius`
- `colors`

View state:

- `centerX`
- `centerY`
- `zoom`

## Poincaré 2D

Use `poincare` when your embeddings already live in the Poincaré disk or when you want geometry-aware hyperbolic interaction.

Relevant init options:

- `poincareDiskFillColor`
- `poincareDiskBorderColor`
- `poincareGridColor`
- `poincareDiskBorderWidthPx`
- `poincareGridWidthPx`

View state:

- `ax`
- `ay`
- `displayZoom`

Interaction notes:

- Pan is anchor-invariant.
- Zoom is anchored at the cursor.
- Projection and unprojection remain disk-aware.

## Euclidean 3D

Use `euclidean3d` for generic 3D point clouds.

View state uses `OrbitViewState3D`:

- `yaw`
- `pitch`
- `distance`
- `targetX`
- `targetY`
- `targetZ`
- `orthoScale`

## Sphere 3D

Use `sphere` for point clouds constrained to a unit sphere.

Additional init options:

- `sphereGuideColor`
- `sphereGuideOpacity`

The spherical renderer shares the same orbit camera shape as `euclidean3d`, but it also draws optional guide geometry for orientation.

## Matching Geometry and Dataset Helpers

Examples:

```ts
createScatterPlot(canvas, {
  geometry: "euclidean",
  dataset: createDatasetFromColumns("euclidean", x, y, labels),
  width,
  height,
});
```

```ts
createScatterPlot(canvas, {
  geometry: "sphere",
  dataset: createDataset3DFromColumns("sphere", x, y, z, labels),
  width,
  height,
});
```

Do not mix geometry tokens between the renderer and the dataset helper.