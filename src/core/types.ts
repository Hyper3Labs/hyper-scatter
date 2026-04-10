/**
 * Core types and contracts for the viz-lab.
 * Both reference and candidate implementations must satisfy these interfaces.
 */

// ============================================================================
// Data Types
// ============================================================================

export interface Dataset {
  /** Number of points */
  n: number;
  /** Interleaved x,y coordinates: [x0, y0, x1, y1, ...] length = 2*n */
  positions: Float32Array;
  /** Label for each point (for coloring) */
  labels: Uint16Array;
  /** Geometry mode */
  geometry: GeometryMode;
}

export type GeometryMode = 'euclidean' | 'poincare';

// ============================================================================
// View State
// ============================================================================

/** Euclidean view state: simple pan + zoom */
export interface EuclideanViewState {
  type: 'euclidean';
  /** Center of view in data coordinates */
  centerX: number;
  centerY: number;
  /** Zoom level (1 = no zoom, >1 = zoomed in) */
  zoom: number;
}

/**
 * Hyperbolic view state using Mobius transformation.
 * The isometry is represented as translation by a point in the disk.
 * Camera transform: z -> (z - a) / (1 - conj(a) * z) where a = (ax, ay)
 */
export interface HyperbolicViewState {
  type: 'poincare';
  /** Translation point x-coordinate (in Poincare disk, |a| < 1) */
  ax: number;
  /** Translation point y-coordinate */
  ay: number;
  /** Display zoom scalar (purely visual, applied after Mobius) */
  displayZoom: number;
}

export type ViewState = EuclideanViewState | HyperbolicViewState;

// ============================================================================
// Interaction Types
// ============================================================================

export interface Modifiers {
  shift: boolean;
  ctrl: boolean;
  alt: boolean;
  meta: boolean;
}

export interface HitResult {
  index: number;
  screenX: number;
  screenY: number;
  distance: number;
}

/** Axis-aligned bounds in 2D data coordinates. */
export interface Bounds2D {
  xMin: number;
  yMin: number;
  xMax: number;
  yMax: number;
}

/**
 * Geometry-based selection shape in data coordinates.
 * Allows implementations to defer index enumeration.
 */
export interface SelectionGeometry {
  type: 'polygon';
  /** Polygon vertices in data coordinates: [x0, y0, x1, y1, ...] */
  coords: Float32Array;
  /** Optional AABB bounds of coords for fast prefiltering (same coordinate space as coords). */
  bounds?: Bounds2D;
}

/**
 * Result of a lasso/selection operation.
 *
 * Implementations may return either:
 * - kind: 'indices' with a Set<number> of selected point indices
 * - kind: 'geometry' with the selection shape in data coordinates
 *
 * Both kinds must provide a has() method for membership testing.
 */
export interface SelectionResult {
  /** How the selection is represented */
  kind: 'indices' | 'geometry';

  /** For kind === 'indices': the selected point indices */
  indices?: Set<number>;

  /** For kind === 'geometry': selection shape in data coordinates */
  geometry?: SelectionGeometry;

  /** Time to compute the selection */
  computeTimeMs: number;

  /**
   * Test if a point index is in the selection.
   * Required for all kinds - enables correctness verification.
   */
  has(index: number): boolean;
}

export type InteractionMode = 'pan' | 'lasso';

export type CategoryVisibilityMask = ArrayLike<number | boolean>;

export interface InteractionStyle {
  /** Color for the selected ring/halo. */
  selectionColor?: string;
  /** Additional radius in CSS px for the selection ring. */
  selectionRadiusOffset?: number;
  /** Ring thickness in CSS px for the selection ring. */
  selectionRingWidth?: number;
  /** Color for the secondary highlight ring/halo. */
  highlightColor?: string;
  /** Additional radius in CSS px for the highlight ring. */
  highlightRadiusOffset?: number;
  /** Ring thickness in CSS px for the highlight ring. */
  highlightRingWidth?: number;
  /** Color for the hovered ring/outline. */
  hoverColor?: string;
  /** Optional fill override for hovered, non-selected points. */
  hoverFillColor?: string | null;
  /** Additional radius in CSS px for the hover ring. */
  hoverRadiusOffset?: number;
}

export interface LassoStyle {
  /** Polygon stroke color in CSS color format. */
  strokeColor?: string;
  /** Polygon stroke width in CSS px. */
  strokeWidth?: number;
  /** Polygon fill color in CSS color format. */
  fillColor?: string;
}

export interface DisplayStateRenderer {
  /** Update the categorical palette without replacing the dataset. */
  setPalette(colors: string[]): void;
  /** Set a per-category visibility mask (truthy = visible). */
  setCategoryVisibility(mask: CategoryVisibilityMask | null): void;
  /**
   * Apply an inactive opacity multiplier to visible categories.
   *
   * Legacy alias for setInactiveOpacity(). Emphasized points may still be
   * redrawn at full opacity via renderer overlays.
   */
  setCategoryAlpha(alpha: number): void;
  /** Apply an inactive opacity multiplier to non-emphasized visible points. */
  setInactiveOpacity(alpha: number): void;
  /** Update hover/selection interaction colors at runtime. */
  setInteractionStyle(style: InteractionStyle): void;
}

// ============================================================================
// Renderer Contract
// ============================================================================

export interface InitOptions {
  width: number;
  height: number;
  devicePixelRatio?: number;
  backgroundColor?: string;
  pointRadius?: number;
  colors?: string[];

  /** Optional styling overrides for the hyperbolic (Poincaré) disk backdrop. */
  poincareDiskFillColor?: string;
  poincareDiskBorderColor?: string;
  poincareGridColor?: string;
  poincareDiskBorderWidthPx?: number;
  poincareGridWidthPx?: number;

}

export interface Renderer extends DisplayStateRenderer {
  /** Initialize the renderer with a canvas */
  init(canvas: HTMLCanvasElement, opts: InitOptions): void;

  /** Set the dataset to render */
  setDataset(dataset: Dataset): void;

  /** Set the current view state */
  setView(view: ViewState): void;

  /** Get the current view state */
  getView(): ViewState;

  /** Render the current frame */
  render(): void;

  /** Handle canvas resize */
  resize(width: number, height: number): void;

  /** Set selected point indices */
  setSelection(indices: Set<number> | null): void;

  /** Get current selection */
  getSelection(): Set<number>;

  /** Set highlighted point indices */
  setHighlight(indices: Set<number> | null): void;

  /** Get current highlight */
  getHighlight(): Set<number>;

  /** Set hovered point index (-1 for none) */
  setHovered(index: number): void;

  /** Draw or clear a renderer-owned lasso polygon overlay in screen space. */
  setLassoPolygon(polygon: Float32Array | null, style?: LassoStyle): void;

  /** Clean up resources */
  destroy(): void;

  // Interaction methods

  /** Pan the view (in screen pixels) */
  pan(deltaX: number, deltaY: number, modifiers: Modifiers): void;

  /** Zoom at anchor point (wheel delta, positive = zoom in) */
  zoom(anchorX: number, anchorY: number, delta: number, modifiers: Modifiers): void;

  /** Hit test at screen coordinates */
  hitTest(screenX: number, screenY: number): HitResult | null;

  /** Lasso selection with screen-space polyline */
  lassoSelect(polyline: Float32Array): SelectionResult;

  /**
   * Count the number of points selected by a SelectionResult.
   *
   * Motivation: geometry-based (predicate) selections are fast to create, but
   * naively counting them by scanning all N points in the UI is too slow.
   * Implementations should use any available spatial index / backend to count
   * efficiently.
   */
  countSelection(result: SelectionResult, opts?: CountSelectionOptions): Promise<number>;

  /** Project a data point to screen coordinates */
  projectToScreen(dataX: number, dataY: number): { x: number; y: number };

  /** Unproject screen coordinates to data space */
  unprojectFromScreen(screenX: number, screenY: number): { x: number; y: number };
}

export interface CountSelectionOptions {
  /** Optional cancellation predicate (return true to abort early). */
  shouldCancel?: () => boolean;

  /** Optional progress callback for long-running counts. */
  onProgress?: (selectedCount: number, processedCandidates: number) => void;

  /**
   * Time slice budget (ms). If > 0, counting yields back to the browser after
   * roughly this amount of work to keep the UI responsive.
   *
   * Set to 0 to run synchronously without yielding (useful for benchmarks).
   * Default: 8ms.
   */
  yieldEveryMs?: number;
}

// ============================================================================
// Default Colors (categorical palette)
// ============================================================================

export const DEFAULT_COLORS = [
  '#4e79a7', // blue
  '#f28e2c', // orange
  '#e15759', // red
  '#76b7b2', // teal
  '#59a14f', // green
  '#edc949', // yellow
  '#af7aa1', // purple
  '#ff9da7', // pink
  '#9c755f', // brown
  '#bab0ab', // gray
];

export const SELECTION_COLOR = '#ff0000';
export const HOVER_COLOR = '#ffffff';

// ============================================================================
// Selection Result Helpers
// ============================================================================

/**
 * Create a SelectionResult from a Set of indices.
 * Convenience helper for implementations that compute indices directly.
 */
export function createIndicesSelectionResult(
  indices: Set<number>,
  computeTimeMs: number
): SelectionResult {
  return {
    kind: 'indices',
    indices,
    computeTimeMs,
    has: (index: number) => indices.has(index),
  };
}

/**
 * Create a SelectionResult from geometry and a positions array.
 * The has() method tests point membership against the polygon.
 */
export function createGeometrySelectionResult(
  geometry: SelectionGeometry,
  positions: Float32Array,
  computeTimeMs: number,
  pointInPolygonFn: (px: number, py: number, polygon: Float32Array) => boolean
): SelectionResult {
  return {
    kind: 'geometry',
    geometry,
    computeTimeMs,
    has: (index: number) => {
      const x = positions[index * 2];
      const y = positions[index * 2 + 1];
      return pointInPolygonFn(x, y, geometry.coords);
    },
  };
}
