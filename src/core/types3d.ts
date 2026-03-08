import type {
  CountSelectionOptions,
  DisplayStateRenderer,
} from "./types.js";

export interface Dataset3D {
  /** Number of points */
  n: number;
  /** Interleaved x,y,z coordinates: [x0, y0, z0, x1, y1, z1, ...] length = 3*n */
  positions: Float32Array;
  /** Label for each point (for coloring) */
  labels: Uint16Array;
  /** Geometry mode */
  geometry: GeometryMode3D;
}

export type GeometryMode3D = "euclidean3d" | "sphere";

export interface OrbitViewState3D {
  type: "orbit3d";
  /** Yaw in radians around world-up (Y axis). */
  yaw: number;
  /** Pitch in radians. */
  pitch: number;
  /** Camera distance from target. */
  distance: number;
  /** Camera target point in data coordinates. */
  targetX: number;
  targetY: number;
  targetZ: number;
  /** Orthographic half-height in data coordinates. */
  orthoScale: number;
}

export interface Modifiers3D {
  shift: boolean;
  ctrl: boolean;
  alt: boolean;
  meta: boolean;
}

export interface HitResult3D {
  index: number;
  screenX: number;
  screenY: number;
  distance: number;
  /** Smaller depth means closer to the camera. */
  depth: number;
}

export interface SelectionResult3D {
  kind: "indices";
  indices?: Set<number>;
  computeTimeMs: number;
  has(index: number): boolean;
}

export interface ProjectedPoint3D {
  x: number;
  y: number;
  depth: number;
  visible: boolean;
}

export interface InitOptions3D {
  width: number;
  height: number;
  devicePixelRatio?: number;
  backgroundColor?: string;
  pointRadius?: number;
  colors?: string[];
  sphereGuideColor?: string;
  sphereGuideOpacity?: number;
}

export interface Renderer3D extends DisplayStateRenderer {
  init(canvas: HTMLCanvasElement, opts: InitOptions3D): void;
  setDataset(dataset: Dataset3D): void;
  setView(view: OrbitViewState3D): void;
  getView(): OrbitViewState3D;
  render(): void;
  resize(width: number, height: number): void;
  setSelection(indices: Set<number>): void;
  getSelection(): Set<number>;
  setHovered(index: number): void;
  destroy(): void;
  pan(deltaX: number, deltaY: number, modifiers: Modifiers3D): void;
  zoom(anchorX: number, anchorY: number, delta: number, modifiers: Modifiers3D): void;
  hitTest(screenX: number, screenY: number): HitResult3D | null;
  lassoSelect(polyline: Float32Array): SelectionResult3D;
  countSelection(result: SelectionResult3D, opts?: CountSelectionOptions): Promise<number>;
  projectToScreen(dataX: number, dataY: number, dataZ: number): ProjectedPoint3D;
}

export function createIndicesSelectionResult3D(
  indices: Set<number>,
  computeTimeMs: number,
): SelectionResult3D {
  return {
    kind: "indices",
    indices,
    computeTimeMs,
    has: (index: number) => indices.has(index),
  };
}

export function createDefaultOrbitView3D(): OrbitViewState3D {
  return {
    type: "orbit3d",
    yaw: 0,
    pitch: 0,
    distance: 3,
    targetX: 0,
    targetY: 0,
    targetZ: 0,
    orthoScale: 1.25,
  };
}
