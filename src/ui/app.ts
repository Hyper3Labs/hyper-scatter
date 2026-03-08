/**
 * Demo application for the viz-lab.
 * Provides interactive UI for testing reference implementations.
 */

import { Dataset, GeometryMode, Renderer, InteractionMode, Modifiers, SelectionResult } from '../core/types.js';
import { Dataset3D, GeometryMode3D, Renderer3D, SelectionResult3D } from '../core/types3d.js';
import { generateDataset, type DatasetDistribution } from '../core/dataset.js';
import { EuclideanReference } from '../impl_reference/euclidean_reference.js';
import { HyperbolicReference } from '../impl_reference/hyperbolic_reference.js';
import { EuclideanWebGLCandidate, HyperbolicWebGLCandidate } from '../impl_candidate/webgl_candidate.js';
import { Euclidean3DWebGLCandidate, Spherical3DWebGLCandidate } from '../impl_candidate/webgl_candidate_3d.js';
import { simplifyPolygonData } from '../core/lasso_simplify.js';

type RendererType = 'webgl' | 'reference';
type GeometryChoice = GeometryMode | GeometryMode3D;
type AnyRenderer = Renderer | Renderer3D;
type AnyDataset = Dataset | Dataset3D;

// DOM elements
let canvas = document.getElementById('canvas') as HTMLCanvasElement;
const overlayCanvas = document.getElementById('overlayCanvas') as HTMLCanvasElement;
const canvasBody = document.getElementById('canvasBody') as HTMLDivElement;
const canvasHeader = document.getElementById('canvasHeader') as HTMLSpanElement;
const rendererInputs = Array.from(document.querySelectorAll<HTMLInputElement>('input[name="renderer"]'));
const geometryInputs = Array.from(document.querySelectorAll<HTMLInputElement>('input[name="geometry"]'));
const numPointsInput = document.getElementById('numPoints') as HTMLInputElement;
const numPointsLabel = document.getElementById('numPointsLabel') as HTMLSpanElement;
const datasetModeSelect = document.getElementById('datasetMode') as HTMLSelectElement;
const labelCountInput = document.getElementById('labelCount') as HTMLInputElement;
const labelCountLabel = document.getElementById('labelCountLabel') as HTMLSpanElement | null;
const seedInput = document.getElementById('seed') as HTMLInputElement;
const modeIndicator = document.getElementById('modeIndicator') as HTMLSpanElement;
const rendererWebglInput = document.getElementById('rendWebgl') as HTMLInputElement;
const rendererReferenceInput = document.getElementById('rendRef') as HTMLInputElement;
const rendererReferenceLabel = document.querySelector('label[for="rendRef"]') as HTMLLabelElement | null;

// Stats elements
const statPoints = document.getElementById('statPoints') as HTMLSpanElement;
const statSelected = document.getElementById('statSelected') as HTMLSpanElement;
const statHovered = document.getElementById('statHovered') as HTMLSpanElement;
const statFrameTime = document.getElementById('statFrameTime') as HTMLSpanElement;
const statLassoTime = document.getElementById('statLassoTime') as HTMLSpanElement;

// State
let renderer: AnyRenderer | null = null;
let dataset: AnyDataset | null = null;
let lastDatasetKey = '';
let currentGeometry: GeometryChoice = 'euclidean';
let currentRendererType: RendererType = 'webgl';
let mode: InteractionMode = 'pan';

const POINT_PRESETS = [
  1_000,
  10_000,
  50_000,
  100_000,
  250_000,
  500_000,
  1_000_000,
  2_000_000,
  5_000_000,
  10_000_000,
  20_000_000,
];

// Interaction state
let isDragging = false;
let isLassoing = false;
let lastMouseX = 0;
let lastMouseY = 0;

// Lasso UX notes (Embedding Atlas style):
// - Gesture: Shift + Meta/Ctrl drag (momentary) starts lasso.
// - Lasso vertices are simplified continuously while dragging.
// - The lasso overlay persists after mouse-up until explicitly cleared.
// - Any camera movement (pan/zoom) clears persisted lasso because the old
//   polygon is no longer guaranteed to represent an accurate range in view.
const LASSO_MAX_VERTS_INTERACTION = 24;
const LASSO_MAX_VERTS_FINAL = 24;
const LASSO_MIN_SAMPLE_DIST_PX = 2.0;

// Raw sampled lasso points in DATA SPACE (interleaved x,y).
// We sample in data space (Embedding Atlas style) so simplification is
// view-invariant and avoids screen-space artifacts.
let lassoRawDataPoints: number[] = [];
let lassoRawScreenPoints: number[] = [];

// Current simplified lasso polygon in DATA SPACE (interleaved x,y).
let lassoActiveDataPolygon: Float32Array | null = null;
let lassoActiveScreenPolygon: Float32Array | null = null;

// For sampling thresholding (screen space).
let lassoLastScreenX = 0;
let lassoLastScreenY = 0;
let lassoDirty = false;

// Persisted range selection (data-space polygon: interleaved x,y).
let rangeSelectionDataPolygon: Float32Array | null = null;
let rangeSelectionScreenPolygon: Float32Array | null = null;

function is3DGeometry(geometry: GeometryChoice): geometry is GeometryMode3D {
  return geometry === 'euclidean3d' || geometry === 'sphere';
}

function syncRendererAvailabilityForGeometry(geometry: GeometryChoice): void {
  const needsWebGL = is3DGeometry(geometry);
  rendererReferenceInput.disabled = needsWebGL;

  if (needsWebGL && rendererReferenceInput.checked) {
    rendererWebglInput.checked = true;
  }

  if (rendererReferenceLabel) {
    rendererReferenceLabel.title = needsWebGL
      ? 'Reference renderer is available only for 2D geometries'
      : '';
  }
}

function createRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function randomUnitVector(rand: () => number): [number, number, number] {
  const u = rand() * 2 - 1;
  const theta = rand() * Math.PI * 2;
  const s = Math.sqrt(Math.max(0, 1 - u * u));
  return [s * Math.cos(theta), u, s * Math.sin(theta)];
}

function generateDataset3D(options: {
  seed: number;
  n: number;
  labelCount: number;
  geometry: GeometryMode3D;
  distribution: DatasetDistribution;
}): Dataset3D {
  const { seed, n, labelCount, geometry, distribution } = options;
  const positions = new Float32Array(n * 3);
  const labels = new Uint16Array(n);
  const rand = createRng(seed);

  const clusterCount = Math.max(2, Math.min(16, labelCount));
  const centers: Array<[number, number, number]> = [];
  if (distribution === 'clustered') {
    for (let i = 0; i < clusterCount; i++) {
      if (geometry === 'sphere') {
        centers.push(randomUnitVector(rand));
      } else {
        centers.push([
          (rand() * 2 - 1) * 1.1,
          (rand() * 2 - 1) * 1.1,
          (rand() * 2 - 1) * 1.1,
        ]);
      }
    }
  }

  for (let i = 0; i < n; i++) {
    const label = labelCount > 0 ? Math.floor(rand() * labelCount) : 0;
    labels[i] = label;

    let x: number;
    let y: number;
    let z: number;

    if (distribution === 'clustered') {
      const c = centers[label % centers.length];

      if (geometry === 'sphere') {
        const jx = c[0] + (rand() * 2 - 1) * 0.22;
        const jy = c[1] + (rand() * 2 - 1) * 0.22;
        const jz = c[2] + (rand() * 2 - 1) * 0.22;
        const invLen = 1 / Math.max(1e-9, Math.hypot(jx, jy, jz));
        x = jx * invLen;
        y = jy * invLen;
        z = jz * invLen;
      } else {
        x = c[0] + (rand() * 2 - 1) * 0.35;
        y = c[1] + (rand() * 2 - 1) * 0.35;
        z = c[2] + (rand() * 2 - 1) * 0.35;
      }
    } else if (geometry === 'sphere') {
      [x, y, z] = randomUnitVector(rand);
    } else {
      x = (rand() * 2 - 1) * 1.5;
      y = (rand() * 2 - 1) * 1.5;
      z = (rand() * 2 - 1) * 1.5;
    }

    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;
  }

  return {
    n,
    positions,
    labels,
    geometry,
  };
}

function projectDataPolygonToScreen(polyData: Float32Array): Float32Array {
  if (!renderer || is3DGeometry(currentGeometry)) return polyData;
  const n = polyData.length / 2;
  const out = new Float32Array(polyData.length);
  const renderer2D = renderer as Renderer;
  for (let i = 0; i < n; i++) {
    const x = polyData[i * 2];
    const y = polyData[i * 2 + 1];
    const s = renderer2D.projectToScreen(x, y);
    out[i * 2] = s.x;
    out[i * 2 + 1] = s.y;
  }
  return out;
}

// Cancel token for async selection materialization.
let selectionJobId = 0;

async function countSelectionAsync(
  jobId: number,
  result: SelectionResult | SelectionResult3D,
): Promise<void> {
  if (!renderer) return;

  const total = is3DGeometry(currentGeometry)
    ? await (renderer as Renderer3D).countSelection(result as SelectionResult3D)
    : await (renderer as Renderer).countSelection(result as SelectionResult, {
        shouldCancel: () => jobId !== selectionJobId,
        onProgress: (selectedCount) => {
          if (jobId !== selectionJobId) return;
          statSelected.textContent = `${selectedCount.toLocaleString()} (computing…)`;
        },
        yieldEveryMs: 8,
      });

  if (jobId !== selectionJobId) return;
  statSelected.textContent = total.toLocaleString();
}

function clearPersistentRangeSelection(opts: { cancelSelectionJob?: boolean; resetStats?: boolean } = {}): void {
  const hadPersistent = rangeSelectionDataPolygon !== null || rangeSelectionScreenPolygon !== null;
  if (!hadPersistent) return;

  rangeSelectionDataPolygon = null;
  rangeSelectionScreenPolygon = null;

  if (opts.cancelSelectionJob ?? true) {
    // Cancel in-flight counting because the persisted lasso predicate changed.
    selectionJobId++;
  }

  if (opts.resetStats) {
    statSelected.textContent = '0';
    statLassoTime.textContent = '—';
  }

  overlayCanvas.style.display = 'none';
  const ctx = overlayCanvas.getContext('2d');
  if (ctx) ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

// Frame scheduling + throttling
let rafPending = false;
let needsRender = false;
let lastRafTs = 0;

// Coalesced interaction work (apply at most once per frame)
let pendingPanDx = 0;
let pendingPanDy = 0;
let pendingModifiers: Modifiers = { shift: false, ctrl: false, alt: false, meta: false };

let pendingHoverX = 0;
let pendingHoverY = 0;
let hoverDirty = false;

// Performance tracking
let frameCount = 0;
const frameTimes: number[] = [];
const frameIntervals: number[] = [];

// Debug/automation hook (used by demo interaction benchmark)
declare global {
  interface Window {
    __vizDemo?: {
      getRenderer: () => AnyRenderer | null;
      getView: () => any;
      getCanvasSize: () => { cssWidth: number; cssHeight: number; bufferWidth: number; bufferHeight: number };
    };
  }
}

window.__vizDemo = {
  getRenderer: () => renderer,
  getView: () => renderer ? renderer.getView() : null,
  getCanvasSize: () => ({
    cssWidth: canvas.clientWidth,
    cssHeight: canvas.clientHeight,
    bufferWidth: canvas.width,
    bufferHeight: canvas.height,
  }),
};

/**
 * Replace the canvas element to reset context.
 * A canvas can only have one context type (WebGL or 2D), so we need a fresh canvas when switching.
 */
function replaceCanvas(): HTMLCanvasElement {
  const newCanvas = document.createElement('canvas');
  newCanvas.id = 'canvas';
  canvas.replaceWith(newCanvas);
  canvas = newCanvas;

  // Re-attach event listeners to new canvas
  canvas.addEventListener('mousedown', handleMouseDown);
  canvas.addEventListener('mousemove', handleMouseMove);
  canvas.addEventListener('mouseup', handleMouseUp);
  canvas.addEventListener('mouseleave', handleMouseUp);
  canvas.addEventListener('wheel', handleWheel, { passive: false });
  canvas.addEventListener('dblclick', handleDoubleClick);

  return newCanvas;
}

/**
 * Read theme-aware visualization colors from CSS custom properties.
 */
function getVizThemeColors(): {
  backgroundColor: string;
  diskFillColor: string;
  diskBorderColor: string;
  gridColor: string;
} {
  const styles = getComputedStyle(document.documentElement);
  const pick = (name: string, fallback: string) => {
    const value = styles.getPropertyValue(name).trim();
    return value || fallback;
  };

  return {
    backgroundColor: pick('--viz-bg', '#13171f'),
    diskFillColor: pick('--viz-disk', '#1b2230'),
    diskBorderColor: pick('--viz-border', '#6b7280'),
    gridColor: pick('--viz-grid', '#2b334266'),
  };
}

/**
 * Initialize the renderer based on geometry and renderer type.
 */
function initRenderer(geometry: GeometryChoice, rendererType: RendererType): void {
  const effectiveRendererType: RendererType = is3DGeometry(geometry) ? 'webgl' : rendererType;

  if (renderer) {
    renderer.destroy();
  }

  // Replace canvas when switching between WebGL and 2D contexts
  // A canvas can only have one context type
  const needsNewCanvas = currentRendererType !== effectiveRendererType ||
    (effectiveRendererType === 'webgl' && !canvas.getContext('webgl2')) ||
    (effectiveRendererType === 'reference' && !canvas.getContext('2d'));

  if (needsNewCanvas) {
    replaceCanvas();
  }

  const rect = canvasBody.getBoundingClientRect();
  const width = Math.floor(rect.width);
  const height = Math.floor(rect.height);

  // Keep overlay canvas in sync regardless of renderer type.
  resizeOverlay(width, height);

  if (effectiveRendererType === 'webgl') {
    if (geometry === 'euclidean') {
      renderer = new EuclideanWebGLCandidate();
    } else if (geometry === 'poincare') {
      renderer = new HyperbolicWebGLCandidate();
    } else if (geometry === 'euclidean3d') {
      renderer = new Euclidean3DWebGLCandidate();
    } else {
      renderer = new Spherical3DWebGLCandidate();
    }
    canvasHeader.textContent = 'WebGL';
  } else {
    if (geometry === 'euclidean') {
      renderer = new EuclideanReference();
    } else {
      renderer = new HyperbolicReference();
    }
    canvasHeader.textContent = 'Reference';
  }

  const theme = getVizThemeColors();
  if (is3DGeometry(geometry)) {
    (renderer as Renderer3D).init(canvas, {
      width,
      height,
      devicePixelRatio: window.devicePixelRatio,
      backgroundColor: theme.backgroundColor,
      sphereGuideColor: theme.diskBorderColor,
      sphereGuideOpacity: 0.25,
    });
  } else {
    (renderer as Renderer).init(canvas, {
      width,
      height,
      devicePixelRatio: window.devicePixelRatio,
      backgroundColor: theme.backgroundColor,
      poincareDiskFillColor: theme.diskFillColor,
      poincareDiskBorderColor: theme.diskBorderColor,
      poincareGridColor: theme.gridColor,
    });
  }

  currentGeometry = geometry;
  currentRendererType = effectiveRendererType;
}

function resizeOverlay(width: number, height: number): void {
  // Overlay is UI-only (lasso). Keep it at DPR=1 for performance.
  // Clearing/drawing a DPR=2 overlay every frame can noticeably hurt FPS.
  const dpr = 1;
  overlayCanvas.width = Math.max(1, Math.floor(width * dpr));
  overlayCanvas.height = Math.max(1, Math.floor(height * dpr));
  overlayCanvas.style.width = `${width}px`;
  overlayCanvas.style.height = `${height}px`;

  const ctx = overlayCanvas.getContext('2d');
  if (!ctx) return;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, width, height);
}

function getSelectedGeometry(): GeometryChoice {
  const el = document.querySelector<HTMLInputElement>('input[name="geometry"]:checked');
  return (el?.value as GeometryChoice) ?? 'euclidean';
}

function getSelectedRendererType(): RendererType {
  const el = document.querySelector<HTMLInputElement>('input[name="renderer"]:checked');
  return (el?.value as RendererType) ?? 'webgl';
}

function getSelectedDatasetDistribution(): DatasetDistribution {
  return (datasetModeSelect?.value as DatasetDistribution) ?? 'default';
}

function formatCount(n: number): string {
  if (n >= 1_000_000) return `${n / 1_000_000}M`;
  if (n >= 1_000) return `${n / 1_000}K`;
  return `${n}`;
}

function getPointCount(): number {
  const i = Math.max(0, Math.min(POINT_PRESETS.length - 1, parseInt(numPointsInput.value, 10) || 0));
  return POINT_PRESETS[i];
}

function syncPointLabel(): void {
  numPointsLabel.textContent = formatCount(getPointCount());
}

let generateTimer: number | null = null;
function scheduleGenerateAndLoad(): void {
  if (generateTimer !== null) window.clearTimeout(generateTimer);
  generateTimer = window.setTimeout(() => {
    generateTimer = null;
    generateAndLoad();
  }, 150);
}

/**
 * Generate and load a new dataset.
 */
function generateAndLoad(): void {
  const geometry = getSelectedGeometry();
  syncRendererAvailabilityForGeometry(geometry);

  const rendererType = getSelectedRendererType();
  const n = getPointCount();
  const labelCount = parseInt(labelCountInput.value, 10);
  const seed = parseInt(seedInput.value, 10);
  const distribution = getSelectedDatasetDistribution();

  const datasetKey = `${geometry}/${distribution}/${n}/${labelCount}/${seed}`;
  const needsNewDataset = !dataset || datasetKey !== lastDatasetKey;

  // Cancel any in-flight selection job.
  selectionJobId++;

  // Initialize renderer if needed (geometry or renderer type changed)
  if (currentGeometry !== geometry || currentRendererType !== rendererType || !renderer) {
    initRenderer(geometry, rendererType);
  }

  // Generate dataset only if inputs changed.
  if (needsNewDataset) {
    if (is3DGeometry(geometry)) {
      dataset = generateDataset3D({
        seed,
        n,
        labelCount,
        geometry,
        distribution,
      });
    } else {
      dataset = generateDataset({
        seed,
        n,
        labelCount,
        geometry,
        distribution,
      });
    }
    lastDatasetKey = datasetKey;
  }

  // Load dataset
  if (is3DGeometry(geometry)) {
    (renderer as Renderer3D).setDataset(dataset as Dataset3D);
  } else {
    (renderer as Renderer).setDataset(dataset as Dataset);
  }

  // Update stats
  statPoints.textContent = n.toLocaleString();
  statSelected.textContent = '0';
  statHovered.textContent = '-';
  statFrameTime.textContent = '—';
  statFrameTime.style.color = '';
  statLassoTime.textContent = '—';
  frameTimes.length = 0; // Reset frame time tracking
  frameIntervals.length = 0;
  lastRafTs = 0;
  rangeSelectionDataPolygon = null;
  rangeSelectionScreenPolygon = null;

  // Clear overlay UI.
  {
    overlayCanvas.style.display = 'none';
    const ctx = overlayCanvas.getContext('2d');
    if (ctx) {
      const rect = canvasBody.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
    }
  }

  // Render
  requestRender();
}

/**
 * Request a render frame.
 */
function requestRender(): void {
  needsRender = true;
  if (rafPending) return;
  rafPending = true;
  requestAnimationFrame(tick);
}

function tick(ts: number): void {
  rafPending = false;

  // Track actual frame interval (what people perceive as FPS).
  if (lastRafTs !== 0) {
    frameIntervals.push(ts - lastRafTs);
    if (frameIntervals.length > 60) frameIntervals.shift();
  }
  lastRafTs = ts;

  if (needsRender) {
    needsRender = false;
    render();
  }

  // Keep animating during interaction so motion stays smooth even if input
  // events arrive irregularly.
  if (isDragging || isLassoing || hoverDirty) {
    requestRender();
  }
}

/**
 * Render the current frame.
 */
function render(): void {
  if (!renderer) return;

  // Apply coalesced pan at most once per frame.
  // Important: apply even if the drag ended before this frame executed.
  if (renderer && (pendingPanDx !== 0 || pendingPanDy !== 0)) {
    clearPersistentRangeSelection({ resetStats: true });
    renderer.pan(pendingPanDx, pendingPanDy, pendingModifiers);
    pendingPanDx = 0;
    pendingPanDy = 0;
  }

  // Throttle hover hit-test to once per frame.
  if (!isDragging && !isLassoing && hoverDirty && renderer) {
    const hit = renderer.hitTest(pendingHoverX, pendingHoverY);
    if (hit) {
      renderer.setHovered(hit.index);
      statHovered.textContent = `#${hit.index}`;
    } else {
      renderer.setHovered(-1);
      statHovered.textContent = '-';
    }
    hoverDirty = false;
  }

  const startTime = performance.now();
  renderer.render();
  const endTime = performance.now();

  // Track frame time
  const frameTime = endTime - startTime;
  frameTimes.push(frameTime);
  if (frameTimes.length > 60) frameTimes.shift();

  // Update frame time display (every 10 frames)
  frameCount++;
  if (frameCount % 10 === 0) {
    const avgCpuMs = frameTimes.reduce((a, b) => a + b, 0) / Math.max(1, frameTimes.length);
    const avgIntervalMs = frameIntervals.reduce((a, b) => a + b, 0) / Math.max(1, frameIntervals.length);
    const fps = avgIntervalMs > 1e-6 ? (1000 / avgIntervalMs) : 0;
    statFrameTime.textContent = `fps ${fps.toFixed(1)} · cpu ${avgCpuMs.toFixed(2)}ms`;
    statFrameTime.style.color = fps >= 50 ? '#8b8' : '#b66';
  }

  // Draw lasso only while actively drawing.
  if (isLassoing) {
    // Continuously simplify while dragging to keep overlay + selection snappy.
    // (Embedding Atlas uses simplifyPolygon(points, 24).)
    if (lassoDirty) {
      lassoDirty = false;
      if (is3DGeometry(currentGeometry)) {
        if (lassoRawScreenPoints.length >= 6) {
          lassoActiveScreenPolygon = simplifyPolygonData(lassoRawScreenPoints, LASSO_MAX_VERTS_INTERACTION);
        } else {
          lassoActiveScreenPolygon = null;
        }
      } else {
        if (lassoRawDataPoints.length >= 6) {
          lassoActiveDataPolygon = simplifyPolygonData(lassoRawDataPoints, LASSO_MAX_VERTS_INTERACTION);
        } else {
          lassoActiveDataPolygon = null;
        }
      }
    }
  }

  // Overlay rendering is UI-only; keep it decoupled from WebGL.
  // - While dragging, draw the (simplified) in-progress lasso in screen space.
  // - When not dragging, draw the persisted lasso projected from data space.
  if (is3DGeometry(currentGeometry)) {
    if (isLassoing && lassoActiveScreenPolygon && lassoActiveScreenPolygon.length >= 6) {
      drawLassoScreen(lassoActiveScreenPolygon);
    } else if (rangeSelectionScreenPolygon && rangeSelectionScreenPolygon.length >= 6) {
      drawLassoScreen(rangeSelectionScreenPolygon);
    } else {
      overlayCanvas.style.display = 'none';
      const ctx = overlayCanvas.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }
  } else {
    if (isLassoing && lassoActiveDataPolygon && lassoActiveDataPolygon.length >= 6) {
      drawLassoData(lassoActiveDataPolygon);
    } else if (rangeSelectionDataPolygon && rangeSelectionDataPolygon.length >= 6) {
      drawLassoData(rangeSelectionDataPolygon);
    } else {
      // Nothing to show.
      overlayCanvas.style.display = 'none';
      const ctx = overlayCanvas.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }
  }
}

/**
 * Draw a lasso polygon given in SCREEN SPACE (interleaved x,y).
 */
function drawLassoScreen(polyline: Float32Array): void {
  overlayCanvas.style.display = 'block';
  const ctx = overlayCanvas.getContext('2d');
  if (!ctx) return;

  const width = overlayCanvas.width;
  const height = overlayCanvas.height;
  ctx.clearRect(0, 0, width, height);

  if (polyline.length < 6) return;

  ctx.strokeStyle = '#cccccc';
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 5]);
  ctx.beginPath();
  ctx.moveTo(polyline[0], polyline[1]);
  for (let i = 2; i < polyline.length; i += 2) {
    ctx.lineTo(polyline[i], polyline[i + 1]);
  }
  ctx.closePath();
  ctx.stroke();

  ctx.fillStyle = 'rgba(255, 255, 255, 0.06)';
  ctx.fill();
}

/**
 * Draw a lasso polygon given in DATA SPACE (interleaved x,y).
 * The overlay is projected each frame so it tracks pan/zoom correctly.
 */
function drawLassoData(polygonData: Float32Array): void {
  if (!renderer) return;

  const n = polygonData.length / 2;
  if (n < 3) return;

  drawLassoScreen(projectDataPolygonToScreen(polygonData));
}

/**
 * Update the mode indicator.
 */
function updateModeIndicator(): void {
  modeIndicator.textContent = mode.toUpperCase();
  modeIndicator.className = `mode ${mode}`;
}

// === Event Handlers ===

function handleMouseDown(e: MouseEvent): void {
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  lastMouseX = x;
  lastMouseY = y;

  // Embedding Atlas gesture:
  // - momentary lasso: Shift + Meta (Cmd on macOS) or Shift + Ctrl
  const wantsLasso = e.shiftKey && (e.metaKey || e.ctrlKey);

  // Plain click clears the persistent range selection (lasso overlay).
  // NOTE: We only clear range selection here; point selection is preserved.
  if (!wantsLasso && !e.shiftKey && !e.ctrlKey && !e.altKey && !e.metaKey && (rangeSelectionDataPolygon || rangeSelectionScreenPolygon)) {
    clearPersistentRangeSelection({ resetStats: true });
    requestRender();
  }

  if (wantsLasso) {
    if (!renderer) return;
    mode = 'lasso';
    isLassoing = true;
    overlayCanvas.style.display = 'block';
    if (is3DGeometry(currentGeometry)) {
      lassoRawScreenPoints = [x, y];
      lassoRawDataPoints = [];
      lassoActiveScreenPolygon = null;
      lassoActiveDataPolygon = null;
    } else {
      const d0 = (renderer as Renderer).unprojectFromScreen(x, y);
      lassoRawDataPoints = [d0.x, d0.y];
      lassoRawScreenPoints = [];
      lassoActiveDataPolygon = null;
      lassoActiveScreenPolygon = null;
    }
    lassoLastScreenX = x;
    lassoLastScreenY = y;
    lassoDirty = true;
    statSelected.textContent = '0';
    updateModeIndicator();
    requestRender();
  } else {
    // Start pan
    mode = 'pan';
    isDragging = true;
    updateModeIndicator();

    pendingPanDx = 0;
    pendingPanDy = 0;
    pendingModifiers = {
      shift: e.shiftKey,
      ctrl: e.ctrlKey,
      alt: e.altKey,
      meta: e.metaKey,
    };

    // For hyperbolic, notify start of pan
    if (renderer && 'startPan' in renderer) {
      (renderer as any).startPan(x, y);
    }

    requestRender();
  }
}

function handleMouseMove(e: MouseEvent): void {
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  if (isDragging && renderer) {
    const deltaX = x - lastMouseX;
    const deltaY = y - lastMouseY;
    pendingPanDx += deltaX;
    pendingPanDy += deltaY;
    pendingModifiers = {
      shift: e.shiftKey,
      ctrl: e.ctrlKey,
      alt: e.altKey,
      meta: e.metaKey,
    };
    lastMouseX = x;
    lastMouseY = y;
    requestRender();
  } else if (isLassoing) {
    // Avoid capturing every single mousemove event (can be thousands of vertices).
    // Sample only when we moved enough in screen space.
    const dx = x - lassoLastScreenX;
    const dy = y - lassoLastScreenY;
    if (dx * dx + dy * dy >= LASSO_MIN_SAMPLE_DIST_PX * LASSO_MIN_SAMPLE_DIST_PX) {
      if (renderer) {
        if (is3DGeometry(currentGeometry)) {
          lassoRawScreenPoints.push(x, y);
        } else {
          const d = (renderer as Renderer).unprojectFromScreen(x, y);
          lassoRawDataPoints.push(d.x, d.y);
        }
        lassoDirty = true;
      }
      lassoLastScreenX = x;
      lassoLastScreenY = y;
    }
    requestRender();
  } else if (renderer) {
    // Hover test (throttled to rAF)
    pendingHoverX = x;
    pendingHoverY = y;
    hoverDirty = true;
    requestRender();
  }
}

function handleMouseUp(_e: MouseEvent): void {
  // Flush any pending pan deltas *before* clearing state.
  // Otherwise, releasing the mouse before the scheduled rAF frame runs can
  // drop the final (or entire) pan and appear as if the view snaps back.
  if (renderer && (pendingPanDx !== 0 || pendingPanDy !== 0)) {
    clearPersistentRangeSelection({ resetStats: true });
    renderer.pan(pendingPanDx, pendingPanDy, pendingModifiers);
    pendingPanDx = 0;
    pendingPanDy = 0;
  }

  if (isLassoing && renderer) {
    if (is3DGeometry(currentGeometry)) {
      if (lassoRawScreenPoints.length >= 6) {
        const screenPolyline = lassoActiveScreenPolygon ?? simplifyPolygonData(lassoRawScreenPoints, LASSO_MAX_VERTS_FINAL);
        const result = (renderer as Renderer3D).lassoSelect(screenPolyline);
        rangeSelectionScreenPolygon = screenPolyline;
        rangeSelectionDataPolygon = null;

        renderer.setSelection(new Set());
        statSelected.textContent = '…';
        const jobId = ++selectionJobId;
        void countSelectionAsync(jobId, result);
        statLassoTime.textContent = `${result.computeTimeMs.toFixed(2)}ms`;
      }
    } else if (lassoRawDataPoints.length >= 6) {
      const dataPoly = lassoActiveDataPolygon ?? simplifyPolygonData(lassoRawDataPoints, LASSO_MAX_VERTS_FINAL);
      const screenPolyline = projectDataPolygonToScreen(dataPoly);
      const result = (renderer as Renderer).lassoSelect(screenPolyline);

      // Persist the range selection overlay in DATA SPACE (so it tracks pan/zoom).
      rangeSelectionDataPolygon = dataPoly;
      rangeSelectionScreenPolygon = null;

      // Apply selection (Embedding Atlas style): keep only the range-selection
      // overlay (no point highlighting) and compute an exact count asynchronously.
      renderer.setSelection(new Set());
      statSelected.textContent = '…';
      const jobId = ++selectionJobId;
      void countSelectionAsync(jobId, result);
      statLassoTime.textContent = `${result.computeTimeMs.toFixed(2)}ms`;
    }
  }

  isDragging = false;
  isLassoing = false;
  lassoRawDataPoints = [];
  lassoRawScreenPoints = [];
  lassoActiveDataPolygon = null;
  lassoActiveScreenPolygon = null;
  lassoDirty = false;
  pendingPanDx = 0;
  pendingPanDy = 0;
  hoverDirty = false;

  // If the renderer supports it, tell it the interaction is over so the next
  // render uses the stable (non-interaction) policy immediately.
  // This avoids a visible LOD "pop" when the next hover-triggered render would
  // otherwise switch from interaction subsampling back to full detail.
  if (renderer && 'endInteraction' in (renderer as any)) {
    (renderer as any).endInteraction();
  }

  // Keep overlay after lasso completion; it is cleared by plain click,
  // double-click, or any camera movement (pan/zoom).
  mode = 'pan';
  updateModeIndicator();
  requestRender();
}

function handleWheel(e: WheelEvent): void {
  e.preventDefault();
  if (!renderer) return;

  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  // Normalize wheel delta
  const delta = -e.deltaY / 100;

  clearPersistentRangeSelection({ resetStats: true });

  renderer.zoom(x, y, delta, {
    shift: e.shiftKey,
    ctrl: e.ctrlKey,
    alt: e.altKey,
    meta: e.metaKey,
  });

  requestRender();
}

function handleDoubleClick(): void {
  if (!renderer) return;

  // Clear selection
  selectionJobId++;
  renderer.setSelection(new Set());
  statSelected.textContent = '0';

  // Hide overlay since range selection is cleared.
  clearPersistentRangeSelection({ cancelSelectionJob: false, resetStats: true });
  requestRender();
}

function handleResize(): void {
  if (!renderer) return;

  const rect = canvasBody.getBoundingClientRect();
  const width = Math.floor(rect.width);
  const height = Math.floor(rect.height);

  renderer.resize(width, height);
  resizeOverlay(width, height);
  requestRender();
}

// === Initialization ===

// Set up event listeners
canvas.addEventListener('mousedown', handleMouseDown);
canvas.addEventListener('mousemove', handleMouseMove);
canvas.addEventListener('mouseup', handleMouseUp);
canvas.addEventListener('mouseleave', handleMouseUp);
canvas.addEventListener('wheel', handleWheel, { passive: false });
canvas.addEventListener('dblclick', handleDoubleClick);
window.addEventListener('resize', handleResize);

for (const el of geometryInputs) {
  el.addEventListener('change', () => {
    syncRendererAvailabilityForGeometry(getSelectedGeometry());
    scheduleGenerateAndLoad();
  });
}
for (const el of rendererInputs) el.addEventListener('change', scheduleGenerateAndLoad);
datasetModeSelect.addEventListener('change', scheduleGenerateAndLoad);
numPointsInput.addEventListener('input', () => {
  syncPointLabel();
  scheduleGenerateAndLoad();
});
numPointsInput.addEventListener('change', () => {
  syncPointLabel();
  scheduleGenerateAndLoad();
});
labelCountInput.addEventListener('input', () => {
  if (labelCountLabel) labelCountLabel.textContent = labelCountInput.value;
  scheduleGenerateAndLoad();
});
labelCountInput.addEventListener('change', () => {
  if (labelCountLabel) labelCountLabel.textContent = labelCountInput.value;
  scheduleGenerateAndLoad();
});
seedInput.addEventListener('change', scheduleGenerateAndLoad);

// Re-init renderer when color scheme changes
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  if (renderer) {
    initRenderer(currentGeometry, currentRendererType);
    if (dataset) {
      if (is3DGeometry(currentGeometry)) {
        (renderer as Renderer3D).setDataset(dataset as Dataset3D);
      } else {
        (renderer as Renderer).setDataset(dataset as Dataset);
      }
    }
    requestRender();
  }
});

// Initial generation
syncRendererAvailabilityForGeometry(getSelectedGeometry());
syncPointLabel();
if (labelCountLabel) labelCountLabel.textContent = labelCountInput.value;
generateAndLoad();

console.log('Viz Lab initialized');
