import type { Dataset, GeometryMode, InitOptions, Renderer } from './core/types.js';
import type { Dataset3D, GeometryMode3D, InitOptions3D, Renderer3D } from './core/types3d.js';
import {
  EuclideanWebGLCandidate,
  HyperbolicWebGLCandidate,
} from './impl_candidate/webgl_candidate.js';
import {
  Euclidean3DWebGLCandidate,
  Spherical3DWebGLCandidate,
} from './impl_candidate/webgl_candidate_3d.js';

export type {
  Dataset,
  GeometryMode,
  Renderer,
  InitOptions,
  LassoStyle,
  ViewState,
  EuclideanViewState,
  HyperbolicViewState,
  Modifiers,
  HitResult,
  Bounds2D,
  SelectionGeometry,
  SelectionResult,
  CountSelectionOptions,
  CategoryVisibilityMask,
  InteractionStyle,
  DisplayStateRenderer,
} from './core/types.js';

export type {
  Dataset3D,
  GeometryMode3D,
  Renderer3D,
  InitOptions3D,
  OrbitViewState3D,
  Modifiers3D,
  HitResult3D,
  ProjectedPoint3D,
  SelectionResult3D,
} from './core/types3d.js';

export {
  DEFAULT_COLORS,
  SELECTION_COLOR,
  HOVER_COLOR,
} from './core/types.js';

export {
  EuclideanWebGLCandidate,
  HyperbolicWebGLCandidate,
} from './impl_candidate/webgl_candidate.js';

export {
  Euclidean3DWebGLCandidate,
  Spherical3DWebGLCandidate,
} from './impl_candidate/webgl_candidate_3d.js';

export {
  createInteractionController,
  type InteractionController,
  type InteractionControllerOptions,
} from './controller/interaction_controller.js';

export {
  boundsOfPolygon,
  pointInPolygon,
} from './core/selection/point_in_polygon.js';

export interface CreateScatterPlot2DOptions extends InitOptions {
  geometry: GeometryMode;
  dataset?: Dataset;
}

export interface CreateScatterPlot3DOptions extends InitOptions3D {
  geometry: GeometryMode3D;
  dataset?: Dataset3D;
}

export type ScatterPlot = Renderer | Renderer3D;

export function createScatterPlot(
  canvas: HTMLCanvasElement,
  options: CreateScatterPlot2DOptions,
): Renderer;
export function createScatterPlot(
  canvas: HTMLCanvasElement,
  options: CreateScatterPlot3DOptions,
): Renderer3D;
export function createScatterPlot(
  canvas: HTMLCanvasElement,
  options: CreateScatterPlot2DOptions | CreateScatterPlot3DOptions,
): ScatterPlot {
  const { geometry, dataset, ...initOptions } = options;

  let renderer: ScatterPlot;
  if (geometry === 'sphere') {
    renderer = new Spherical3DWebGLCandidate();
  } else if (geometry === 'euclidean3d') {
    renderer = new Euclidean3DWebGLCandidate();
  } else if (geometry === 'poincare') {
    renderer = new HyperbolicWebGLCandidate();
  } else {
    renderer = new EuclideanWebGLCandidate();
  }

  (renderer as Renderer | Renderer3D).init(canvas, initOptions as InitOptions & InitOptions3D);
  if (dataset) {
    (renderer as Renderer | Renderer3D).setDataset(dataset as Dataset & Dataset3D);
  }
  return renderer;
}

export function createDataset(
  geometry: GeometryMode,
  positions: Float32Array,
  labels?: Uint16Array,
): Dataset {
  if (positions.length % 2 !== 0) {
    throw new Error(`positions length must be even (got ${positions.length})`);
  }

  const n = positions.length / 2;
  const labelArray = labels ?? new Uint16Array(n);

  if (labelArray.length !== n) {
    throw new Error(`labels length must equal number of points (${n}), got ${labelArray.length}`);
  }

  return {
    n,
    positions,
    labels: labelArray,
    geometry,
  };
}

export function createDataset3D(
  geometry: GeometryMode3D,
  positions: Float32Array,
  labels?: Uint16Array,
): Dataset3D {
  if (positions.length % 3 !== 0) {
    throw new Error(`positions length must be divisible by 3 (got ${positions.length})`);
  }

  const n = positions.length / 3;
  const labelArray = labels ?? new Uint16Array(n);

  if (labelArray.length !== n) {
    throw new Error(`labels length must equal number of points (${n}), got ${labelArray.length}`);
  }

  return {
    n,
    positions,
    labels: labelArray,
    geometry,
  };
}

export function createDatasetFromColumns(
  geometry: GeometryMode,
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  labels?: Uint16Array,
): Dataset {
  return createDataset(geometry, packPositionsXY(x, y), labels);
}

export function createDataset3DFromColumns(
  geometry: GeometryMode3D,
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  z: ArrayLike<number>,
  labels?: Uint16Array,
): Dataset3D {
  return createDataset3D(geometry, packPositionsXYZ(x, y, z), labels);
}

export function packPositions(points: ReadonlyArray<readonly [number, number]>): Float32Array {
  const out = new Float32Array(points.length * 2);
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    out[i * 2] = p[0];
    out[i * 2 + 1] = p[1];
  }
  return out;
}

export function packPositionsXY(x: ArrayLike<number>, y: ArrayLike<number>): Float32Array {
  if (x.length !== y.length) {
    throw new Error(`x/y length mismatch: ${x.length} vs ${y.length}`);
  }

  const out = new Float32Array(x.length * 2);
  for (let i = 0; i < x.length; i++) {
    out[i * 2] = x[i];
    out[i * 2 + 1] = y[i];
  }
  return out;
}

export function packPositionsXYZ(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  z: ArrayLike<number>,
): Float32Array {
  if (x.length !== y.length || x.length !== z.length) {
    throw new Error(`x/y/z length mismatch: ${x.length} vs ${y.length} vs ${z.length}`);
  }

  const out = new Float32Array(x.length * 3);
  for (let i = 0; i < x.length; i++) {
    out[i * 3] = x[i];
    out[i * 3 + 1] = y[i];
    out[i * 3 + 2] = z[i];
  }
  return out;
}

export function packUint16Labels(categories: ArrayLike<number>): Uint16Array {
  const out = new Uint16Array(categories.length);
  for (let i = 0; i < categories.length; i++) {
    const v = categories[i];
    if (!Number.isFinite(v) || v < 0 || v > 0xffff) {
      throw new Error(`label index out of range at ${i}: ${v}`);
    }
    out[i] = v;
  }
  return out;
}
