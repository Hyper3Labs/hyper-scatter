/**
 * Browser Benchmark Script
 *
 * Benchmarks rendering + interaction performance in the browser.
 *
 * This harness supports two renderer modes:
 * - `reference`: Canvas2D reference implementations (correctness baseline)
 * - `webgl`: WebGL2 candidate implementations (performance path)
 *
 * IMPORTANT: a single <canvas> cannot hold both a 2D and WebGL context.
 * Therefore we standardize on:
 * - main canvas (#canvas): reserved for Canvas2D (reference)
 * - hidden canvas (#candidateCanvas): reserved for WebGL (candidate)
 *
 * Used by benchmark.html and the CLI-driven puppeteer runners.
 *
 * ============================================================================
 * CANDIDATE INTEGRATION GUIDE (updated)
 * ============================================================================
 *
 * To test your optimized renderer candidate against the reference:
 *
 * 1. Create your candidate renderer implementing the `Renderer` interface
 *    from `src/core/types.ts`. See `impl_reference/` for examples.
 *
 * 2. Import your candidate in this file:
 *    ```typescript
 *    import { MyOptimizedRenderer } from '../impl_candidate/my_renderer.js';
 *    ```
 *
 * 3. In `runAccuracyBenchmarks()`, replace the candidateRenderer lines:
 *    ```typescript
 *    const candidateRenderer: Renderer = geometry === 'euclidean'
 *      ? new EuclideanReference()  // or your Euclidean candidate
 *      : new MyOptimizedRenderer();  // <-- Your hyperbolic candidate
 *    ```
 *
 * 4. Run `npm run bench:browser` and click "Run Accuracy Tests"
 *
 * 5. For performance comparison, run the benchmark with `renderer: 'webgl'`
 *    (default). To benchmark the reference baseline, use `renderer: 'reference'`.
 *
 * Expected tolerances:
 * - Projection: < 1e-6 pixels
 * - Pan/Zoom view state: < 1e-10 (machine precision)
 * - Hit test: exact match (same point index)
 * - Lasso: exact match (same set of indices)
 *
 * ============================================================================
 */

import { GeometryMode, Renderer, SelectionResult } from '../core/types.js';
import {
  Dataset3D,
  GeometryMode3D,
  Renderer3D,
  SelectionResult3D,
} from '../core/types3d.js';
import { generateDataset } from '../core/dataset.js';
import { EuclideanReference } from '../impl_reference/euclidean_reference.js';
import { HyperbolicReference } from '../impl_reference/hyperbolic_reference.js';
import {
  EuclideanWebGLCandidate,
  HyperbolicWebGLCandidate,
} from '../impl_candidate/webgl_candidate.js';
import {
  Euclidean3DWebGLCandidate,
  Spherical3DWebGLCandidate,
} from '../impl_candidate/webgl_candidate_3d.js';
import {
  TimingStats,
  calculateStats,
  nextFrame,
  sleep,
  generateTestPolygon,
} from './utils.js';
import {
  runAccuracyTests,
  AccuracyReport,
} from './accuracy.js';

// ============================================================================
// Types
// ============================================================================

export interface BenchmarkConfig {
  pointCounts: number[];
  geometries: BenchmarkGeometry[];
  warmupFrames: number;
  measuredFrames: number;
  hitTestSamples: number;

  /**
   * Optional: override the lasso polygon radius as a fraction of min(width,height).
   *
   * This is benchmark-only (not used by accuracy tests). It exists so we can
   * stress dense lasso selections at huge N without editing code each time.
   * Example: 0.4 roughly matches the original large-lasso setting.
   */
  lassoRadiusScale?: number;

  /**
   * Which renderer implementation to benchmark.
   * NOTE: One run should benchmark exactly one renderer mode.
   */
  renderer?: 'webgl' | 'reference';
}

export interface BenchmarkResult {
  geometry: BenchmarkGeometry;
  points: number;
  datasetGenMs: number;
  renderMs: TimingStats;
  frameIntervalMs: TimingStats;  // Actual frame-to-frame time
  hitTestMs: TimingStats;
  panMs?: TimingStats;
  hoverMs?: TimingStats;
  panFrameIntervalMs?: TimingStats;     // Frame interval while panning
  hoverFrameIntervalMs?: TimingStats;   // Frame interval while hovering
  lassoMs: number;
  lassoSelectedCount: number;
  memoryMB?: number;

  /** Optional: candidate renderer policy snapshot(s) for diagnostics. */
  candidatePolicy?: {
    steady?: any;
    pan?: any;
    hover?: any;
  };
}

export interface BenchmarkReport {
  timestamp: string;
  system: {
    userAgent: string;
    devicePixelRatio: number;
    canvasWidth: number;
    canvasHeight: number;
  };
  config: BenchmarkConfig;
  results: BenchmarkResult[];
  accuracyReports?: AccuracyReport[];
}

export type ProgressCallback = (message: string, progress: number) => void;
export type BenchmarkGeometry = GeometryMode | GeometryMode3D;

// ============================================================================
// Default Configuration
// ============================================================================

export const DEFAULT_CONFIG: BenchmarkConfig = {
  pointCounts: [1000, 10000, 50000, 100000, 250000, 500000, 1000000],
  geometries: ['euclidean', 'poincare'],
  warmupFrames: 5,
  measuredFrames: 20,
  hitTestSamples: 100,
  renderer: 'webgl',
};

// Optional: large-N stress config for the updated 20M@60FPS scope.
// Not used by default because it can be very memory- and time-intensive.
export const STRESS_CONFIG: BenchmarkConfig = {
  pointCounts: [1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000],
  geometries: ['euclidean', 'poincare'],
  warmupFrames: 5,
  measuredFrames: 20,
  hitTestSamples: 100,
  renderer: 'webgl',
};

function is2DGeometry(geometry: BenchmarkGeometry): geometry is GeometryMode {
  return geometry === 'euclidean' || geometry === 'poincare';
}

function is3DGeometry(geometry: BenchmarkGeometry): geometry is GeometryMode3D {
  return geometry === 'euclidean3d' || geometry === 'sphere';
}

function createRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function generateDataset3DForBenchmark(geometry: GeometryMode3D, n: number): Dataset3D {
  const positions = new Float32Array(n * 3);
  const labels = new Uint16Array(n);
  const rand = createRng(42 + n + (geometry === 'sphere' ? 1337 : 0));

  for (let i = 0; i < n; i++) {
    labels[i] = i % 10;

    if (geometry === 'sphere') {
      const u = rand() * 2 - 1;
      const theta = rand() * Math.PI * 2;
      const s = Math.sqrt(Math.max(0, 1 - u * u));
      positions[i * 3] = s * Math.cos(theta);
      positions[i * 3 + 1] = u;
      positions[i * 3 + 2] = s * Math.sin(theta);
    } else {
      positions[i * 3] = (rand() * 2 - 1) * 1.5;
      positions[i * 3 + 1] = (rand() * 2 - 1) * 1.5;
      positions[i * 3 + 2] = (rand() * 2 - 1) * 1.5;
    }
  }

  return {
    n,
    positions,
    labels,
    geometry,
  };
}

function getRendererMode(config: BenchmarkConfig): 'webgl' | 'reference' {
  return config.renderer ?? 'webgl';
}

function getOrCreateCandidateCanvas(referenceCanvas: HTMLCanvasElement): HTMLCanvasElement {
  let candidateCanvas = document.getElementById('candidateCanvas') as HTMLCanvasElement | null;
  if (!candidateCanvas) {
    candidateCanvas = document.createElement('canvas');
    candidateCanvas.id = 'candidateCanvas';
    candidateCanvas.style.display = 'none';
    document.body.appendChild(candidateCanvas);
  }

  // Keep CSS sizing in sync with the reference canvas.
  // (Even if the canvas is hidden, some code paths inspect style sizes.)
  const rect = referenceCanvas.getBoundingClientRect();

  // Prefer explicit styles if they exist (e.g. benchmark.html sets canvas.style.width).
  if (referenceCanvas.style.width) candidateCanvas.style.width = referenceCanvas.style.width;
  if (referenceCanvas.style.height) candidateCanvas.style.height = referenceCanvas.style.height;

  // If the reference canvas is visible, trust its layout rect.
  if (rect.width > 0 && rect.height > 0) {
    candidateCanvas.style.width = `${rect.width}px`;
    candidateCanvas.style.height = `${rect.height}px`;
  } else {
    // If the reference canvas is hidden (display:none), its rect is 0x0.
    // Fall back to parent container sizing if available.
    const parent = referenceCanvas.parentElement as HTMLElement | null;
    if (parent) {
      const pr = parent.getBoundingClientRect();
      if (pr.width > 0 && pr.height > 0) {
        // Note: this may include padding. In benchmark.html the JS sets an
        // explicit width on the canvas, so this is primarily a safety net.
        if (!candidateCanvas.style.width) candidateCanvas.style.width = `${pr.width}px`;
        if (!candidateCanvas.style.height) candidateCanvas.style.height = `${pr.height}px`;
      }
    }
  }

  return candidateCanvas;
}

function setBenchmarkCanvasVisibility(
  referenceCanvas: HTMLCanvasElement,
  candidateCanvas: HTMLCanvasElement,
  rendererMode: 'webgl' | 'reference'
): void {
  // For manual runs (benchmark.html), it is much nicer to *see* what you're
  // benchmarking. Also, some browsers may treat a display:none WebGL canvas as
  // “not presented”, which can skew rAF/frame pacing.
  if (rendererMode === 'webgl') {
    referenceCanvas.style.display = 'none';
    candidateCanvas.style.display = 'block';
  } else {
    referenceCanvas.style.display = 'block';
    candidateCanvas.style.display = 'none';
  }
}

// ============================================================================
// Benchmark Runner
// ============================================================================

/**
 * Run a single benchmark for a specific geometry and point count.
 */
async function runSingleBenchmark(
  canvas: HTMLCanvasElement,
  geometry: BenchmarkGeometry,
  pointCount: number,
  config: BenchmarkConfig,
  onProgress?: ProgressCallback
): Promise<BenchmarkResult> {
  const rendererMode = getRendererMode(config);

  const candidateCanvas = getOrCreateCandidateCanvas(canvas);
  setBenchmarkCanvasVisibility(canvas, candidateCanvas, rendererMode);

  // Measure from the visible canvas (clientWidth/clientHeight are 0 for display:none).
  const measureCanvas = rendererMode === 'webgl' ? candidateCanvas : canvas;
  const width = measureCanvas.clientWidth;
  const height = measureCanvas.clientHeight;

  const renderCanvas = rendererMode === 'webgl' ? candidateCanvas : canvas;

  onProgress?.(`Generating ${pointCount.toLocaleString()} ${geometry} points...`, 0);

  let renderer: Renderer | Renderer3D;
  let datasetGenMs: number;

  if (is2DGeometry(geometry)) {
    const datasetStart = performance.now();
    const dataset = generateDataset({
      seed: 42,
      n: pointCount,
      labelCount: 10,
      geometry,
    });
    datasetGenMs = performance.now() - datasetStart;

    renderer = rendererMode === 'reference'
      ? (geometry === 'euclidean' ? new EuclideanReference() : new HyperbolicReference())
      : (geometry === 'euclidean' ? new EuclideanWebGLCandidate() : new HyperbolicWebGLCandidate());

    renderer.init(renderCanvas, {
      width,
      height,
      devicePixelRatio: window.devicePixelRatio,
    });
    (renderer as Renderer).setDataset(dataset);
  } else {
    if (rendererMode === 'reference') {
      throw new Error(`Geometry ${geometry} requires renderer='webgl' (reference mode is 2D-only).`);
    }

    const datasetStart = performance.now();
    const dataset = generateDataset3DForBenchmark(geometry, pointCount);
    datasetGenMs = performance.now() - datasetStart;

    renderer = geometry === 'sphere'
      ? new Spherical3DWebGLCandidate()
      : new Euclidean3DWebGLCandidate();

    renderer.init(renderCanvas, {
      width,
      height,
      devicePixelRatio: window.devicePixelRatio,
      pointRadius: 3,
    });
    (renderer as Renderer3D).setDataset(dataset);
  }

  onProgress?.(`Warming up render...`, 0.1);

  // Warmup renders
  for (let i = 0; i < config.warmupFrames; i++) {
    renderer.render();
    await nextFrame();
  }

  onProgress?.(`Measuring render performance...`, 0.3);

  // Measured render frames - track both CPU submit time and actual frame intervals
  const renderTimes: number[] = [];
  const frameIntervals: number[] = [];
  let lastFrameTime = performance.now();

  for (let i = 0; i < config.measuredFrames; i++) {
    const start = performance.now();
    renderer.render();
    renderTimes.push(performance.now() - start);

    await nextFrame();

    const now = performance.now();
    if (i > 0) {  // Skip first interval (includes warmup latency)
      frameIntervals.push(now - lastFrameTime);
    }
    lastFrameTime = now;
  }

  const policySteady = rendererMode === 'webgl' ? (renderer as any).__debugPolicy : undefined;

  // ------------------------------------------------------------------------
  // Interactive benchmarks (demo-like)
  // ------------------------------------------------------------------------
  // These are the paths that often dominate perceived performance:
  // - Panning: updates view state every frame.
  // - Hovering: hitTest + setHovered + render every frame.
  // The earlier benchmark measured render() only, which can look great even if
  // interaction stutters.

  onProgress?.(`Measuring pan performance...`, 0.45);

  const panTimes: number[] = [];
  const panIntervals: number[] = [];
  let lastPanFrame = performance.now();

  // Use more frames for interaction to better approximate real drag behavior
  // and to ensure we actually reach the edges even when measuredFrames is small.
  const panFrames = Math.max(config.measuredFrames, 60);

  // Start pan gesture at canvas center if supported.
  const startX = width / 2;
  const startY = height / 2;
  if ('startPan' in (renderer as any)) {
    try {
      (renderer as any).startPan(startX, startY);
    } catch {
      // ignore
    }
  }

  // Pan aggressively towards the edges (like a real user dragging around).
  // We construct absolute cursor targets and convert to deltas so the
  // hyperbolic renderer's anchor-invariant pan behaves correctly.
  const centerX = width / 2;
  const centerY = height / 2;

  // For poincare, target near the disk boundary; for euclidean, target near
  // the canvas edge.
  const diskRadius = Math.min(width, height) * 0.45;
  const amp = geometry === 'poincare'
    ? diskRadius * 0.92
    : Math.min(width, height) * 0.45;

  const keypoints: Array<{ x: number; y: number }> = [
    { x: centerX, y: centerY },
    { x: centerX + amp, y: centerY },
    { x: centerX - amp, y: centerY },
    { x: centerX, y: centerY - amp },
    { x: centerX, y: centerY + amp },
    { x: centerX, y: centerY },
  ];

  let curX = startX;
  let curY = startY;

  for (let i = 0; i < panFrames; i++) {
    const t = i / Math.max(1, panFrames - 1);
    const segs = keypoints.length - 1;
    const sFloat = t * segs;
    const s = Math.min(segs - 1, Math.floor(sFloat));
    const u = sFloat - s;

    const a = keypoints[s];
    const b = keypoints[s + 1];
    const targetX = a.x + (b.x - a.x) * u;
    const targetY = a.y + (b.y - a.y) * u;

    const dx = targetX - curX;
    const dy = targetY - curY;
    curX = targetX;
    curY = targetY;

    const p0 = performance.now();
    renderer.pan(dx, dy, { shift: false, ctrl: false, alt: false, meta: false });
    panTimes.push(performance.now() - p0);

    renderer.render();
    await nextFrame();

    const now = performance.now();
    if (i > 0) panIntervals.push(now - lastPanFrame);
    lastPanFrame = now;
  }

  const policyPan = rendererMode === 'webgl' ? (renderer as any).__debugPolicy : undefined;

  onProgress?.(`Measuring hover performance...`, 0.55);

  const hoverTimes: number[] = [];
  const hoverIntervals: number[] = [];
  let lastHoverFrame = performance.now();

  // Move cursor over a circular path and run hitTest + setHovered each frame.
  for (let i = 0; i < config.measuredFrames; i++) {
    const t = i / Math.max(1, config.measuredFrames - 1);
    const x = width / 2 + Math.cos(t * Math.PI * 2) * (Math.min(width, height) * 0.25);
    const y = height / 2 + Math.sin(t * Math.PI * 2) * (Math.min(width, height) * 0.25);

    const h0 = performance.now();
    const hit = renderer.hitTest(x, y);
    renderer.setHovered(hit ? hit.index : -1);
    hoverTimes.push(performance.now() - h0);

    renderer.render();
    await nextFrame();

    const now = performance.now();
    if (i > 0) hoverIntervals.push(now - lastHoverFrame);
    lastHoverFrame = now;
  }

  const policyHover = rendererMode === 'webgl' ? (renderer as any).__debugPolicy : undefined;

  onProgress?.(`Measuring hit test performance...`, 0.6);

  // Hit test benchmark (random positions)
  const hitTestTimes: number[] = [];
  for (let i = 0; i < config.hitTestSamples; i++) {
    const x = Math.random() * width;
    const y = Math.random() * height;
    const start = performance.now();
    renderer.hitTest(x, y);
    hitTestTimes.push(performance.now() - start);
  }

  onProgress?.(`Measuring lasso selection...`, 0.8);

  // Lasso selection benchmark
  // Use consistent lasso size regardless of point count. Implementations must
  // handle large selections efficiently - the harness does not compensate.
  const overrideScale = config.lassoRadiusScale;
  const lassoRadius = (typeof overrideScale === 'number' && Number.isFinite(overrideScale) && overrideScale > 0)
    ? Math.min(width, height) * overrideScale
    : Math.min(width, height) * 0.4;
  const lassoPolygon = generateTestPolygon(width / 2, height / 2, lassoRadius);
  const lassoStart = performance.now();
  const lassoResult = renderer.lassoSelect(lassoPolygon);
  const lassoSelectedCount = is3DGeometry(geometry)
    ? await (renderer as Renderer3D).countSelection(lassoResult as SelectionResult3D, { yieldEveryMs: 0 })
    : await (renderer as Renderer).countSelection(lassoResult as SelectionResult, { yieldEveryMs: 0 });
  const lassoMs = performance.now() - lassoStart;

  // Memory usage (Chrome only)
  let memoryMB: number | undefined;
  if ('memory' in performance) {
    const mem = (performance as any).memory;
    memoryMB = mem.usedJSHeapSize / (1024 * 1024);
  }

  // Cleanup
  renderer.destroy();

  onProgress?.(`Completed ${geometry} ${pointCount.toLocaleString()}`, 1);

  return {
    geometry,
    points: pointCount,
    datasetGenMs,
    renderMs: calculateStats(renderTimes),
    frameIntervalMs: calculateStats(frameIntervals.length > 0 ? frameIntervals : [16.67]),
    hitTestMs: calculateStats(hitTestTimes),
    panMs: calculateStats(panTimes),
    hoverMs: calculateStats(hoverTimes),
    panFrameIntervalMs: calculateStats(panIntervals.length > 0 ? panIntervals : [16.67]),
    hoverFrameIntervalMs: calculateStats(hoverIntervals.length > 0 ? hoverIntervals : [16.67]),
    lassoMs,
    lassoSelectedCount,
    memoryMB,

    candidatePolicy: rendererMode === 'webgl'
      ? {
          steady: policySteady,
          pan: policyPan,
          hover: policyHover,
        }
      : undefined,
  };
}

/**
 * Run the full benchmark suite.
 */
export async function runBenchmarks(
  canvas: HTMLCanvasElement,
  config: BenchmarkConfig = DEFAULT_CONFIG,
  onProgress?: ProgressCallback
): Promise<BenchmarkReport> {
  const results: BenchmarkResult[] = [];
  const totalTests = config.pointCounts.length * config.geometries.length;
  let completedTests = 0;

  // Ensure the correct canvas is visible for the duration of this run.
  // (If we hide the passed `canvas` in WebGL mode, `canvas.clientWidth` becomes
  // 0, so any size reporting must use the visible canvas instead.)
  const rendererMode = getRendererMode(config);
  const candidateCanvas = getOrCreateCandidateCanvas(canvas);
  setBenchmarkCanvasVisibility(canvas, candidateCanvas, rendererMode);

  if (rendererMode === 'reference') {
    const unsupported = config.geometries.filter(is3DGeometry);
    if (unsupported.length > 0) {
      throw new Error(
        `Reference mode only supports 2D geometries (euclidean, poincare). Unsupported: ${unsupported.join(', ')}`
      );
    }
  }

  for (const geometry of config.geometries) {
    for (const pointCount of config.pointCounts) {
      onProgress?.(
        `Running ${geometry} ${pointCount.toLocaleString()}...`,
        completedTests / totalTests
      );

      try {
        const result = await runSingleBenchmark(
          canvas,
          geometry,
          pointCount,
          config,
          (msg, progress) => {
            const overallProgress = (completedTests + progress) / totalTests;
            onProgress?.(msg, overallProgress);
          }
        );
        results.push(result);
      } catch (error) {
        console.error(`Benchmark failed for ${geometry} ${pointCount}:`, error);
        // Continue with other tests
      }

      completedTests++;

      // Small delay between tests to let GC run
      await sleep(100);
    }
  }

  return {
    timestamp: new Date().toISOString(),
    system: {
      userAgent: navigator.userAgent,
      devicePixelRatio: window.devicePixelRatio,
      canvasWidth: (rendererMode === 'webgl' ? candidateCanvas : canvas).clientWidth,
      canvasHeight: (rendererMode === 'webgl' ? candidateCanvas : canvas).clientHeight,
    },
    config,
    results,
  };
}

/**
 * Run accuracy tests comparing reference implementations.
 *
 * IMPORTANT: Uses separate canvases for reference (Canvas2D) and candidate (WebGL)
 * because a single canvas cannot have both 2D and WebGL contexts simultaneously.
 *
 * CANDIDATE INTEGRATION: To test your optimized renderer, replace
 * `candidateRenderer` below with your implementation:
 *
 * ```typescript
 * import { MyOptimizedRenderer } from '../impl_candidate/my_renderer.js';
 *
 * const candidateRenderer: Renderer = geometry === 'euclidean'
 *   ? new EuclideanReference()    // or your Euclidean candidate
 *   : new MyOptimizedRenderer();  // <-- Your hyperbolic candidate
 * ```
 */
export async function runAccuracyBenchmarks(
  canvas: HTMLCanvasElement,
  onProgress?: ProgressCallback
): Promise<AccuracyReport[]> {
  const reports: AccuracyReport[] = [];
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;

  // Get or create separate canvas for candidate (WebGL needs its own canvas)
  const candidateCanvas = getOrCreateCandidateCanvas(canvas);

  for (const geometry of ['euclidean', 'poincare'] as const) {
    onProgress?.(`Running ${geometry} accuracy tests...`, geometry === 'euclidean' ? 0 : 0.5);

    const dataset = generateDataset({
      seed: 42,
      n: 10000,
      labelCount: 10,
      geometry,
    });

    // Reference implementation (ground truth) - uses Canvas2D on main canvas
    const refRenderer: Renderer = geometry === 'euclidean'
      ? new EuclideanReference()
      : new HyperbolicReference();

    // Candidate implementation - uses WebGL on separate canvas
    const candidateRenderer: Renderer = geometry === 'euclidean'
      ? new EuclideanWebGLCandidate()
      : new HyperbolicWebGLCandidate();

    // Initialize on SEPARATE canvases to avoid context conflicts
    refRenderer.init(canvas, { width, height, devicePixelRatio: window.devicePixelRatio });
    candidateRenderer.init(candidateCanvas, { width, height, devicePixelRatio: window.devicePixelRatio });

    const report = runAccuracyTests(
      refRenderer,
      candidateRenderer,
      dataset,
      width,
      height
    );

    reports.push(report);

    refRenderer.destroy();
    candidateRenderer.destroy();
  }

  return reports;
}

// ============================================================================
// Formatting
// ============================================================================

/**
 * Format benchmark results as a text table.
 */
export function formatResultsTable(report: BenchmarkReport): string {
  const lines: string[] = [
    '',
    '═'.repeat(110),
    'BENCHMARK RESULTS',
    '═'.repeat(110),
    `Timestamp: ${report.timestamp}`,
    `Canvas: ${report.system.canvasWidth}x${report.system.canvasHeight} @ ${report.system.devicePixelRatio}x DPR`,
    '─'.repeat(110),
    'Geometry   | Points     | Dataset  | CPU Submit   | Frame Int.   | Actual FPS | Pan FPS    | Hover FPS  | Hit Test   | Lasso      | Memory',
    '─'.repeat(110),
  ];

  for (const r of report.results) {
    const geo = r.geometry.padEnd(10);
    const pts = r.points.toLocaleString().padStart(10);
    const dgen = `${r.datasetGenMs.toFixed(1)}ms`.padStart(8);
    const cpuAvg = `${r.renderMs.avg.toFixed(2)}ms`.padStart(12);
    const frameInt = `${r.frameIntervalMs.avg.toFixed(2)}ms`.padStart(12);
    const fps = `${(1000 / r.frameIntervalMs.avg).toFixed(1)}`.padStart(10);

    const panFps = r.panFrameIntervalMs
      ? `${(1000 / r.panFrameIntervalMs.avg).toFixed(1)}`.padStart(10)
      : 'N/A'.padStart(10);
    const hoverFps = r.hoverFrameIntervalMs
      ? `${(1000 / r.hoverFrameIntervalMs.avg).toFixed(1)}`.padStart(10)
      : 'N/A'.padStart(10);

    const ht = `${r.hitTestMs.avg.toFixed(3)}ms`.padStart(10);
    const lasso = `${r.lassoMs.toFixed(2)}ms`.padStart(10);
    const mem = r.memoryMB ? `${r.memoryMB.toFixed(0)}MB`.padStart(7) : 'N/A'.padStart(7);

    lines.push(`${geo} | ${pts} | ${dgen} | ${cpuAvg} | ${frameInt} | ${fps} | ${panFps} | ${hoverFps} | ${ht} | ${lasso} | ${mem}`);
  }

  lines.push('═'.repeat(110));
  lines.push('');

  return lines.join('\n');
}

/**
 * Format benchmark results as HTML table.
 */
export function formatResultsHTML(report: BenchmarkReport): string {
  let html = `
    <table class="benchmark-table">
      <thead>
        <tr>
          <th>Geometry</th>
          <th>Points</th>
          <th>Dataset Gen</th>
          <th>CPU Submit (avg)</th>
          <th>Frame Interval</th>
          <th>Actual FPS</th>
          <th>Pan FPS</th>
          <th>Hover FPS</th>
          <th>Hit Test</th>
          <th>Lasso</th>
          <th>Memory</th>
        </tr>
      </thead>
      <tbody>
  `;

  for (const r of report.results) {
    // Use frame interval for actual FPS (more accurate for GPU rendering)
    const actualFps = (1000 / r.frameIntervalMs.avg).toFixed(1);
    const fpsClass = parseFloat(actualFps) >= 55 ? 'good' : parseFloat(actualFps) >= 30 ? 'warning' : 'bad';

    const panFps = r.panFrameIntervalMs ? (1000 / r.panFrameIntervalMs.avg).toFixed(1) : 'N/A';
    const hoverFps = r.hoverFrameIntervalMs ? (1000 / r.hoverFrameIntervalMs.avg).toFixed(1) : 'N/A';

    html += `
      <tr>
        <td>${r.geometry}</td>
        <td>${r.points.toLocaleString()}</td>
        <td>${r.datasetGenMs.toFixed(1)}ms</td>
        <td>${r.renderMs.avg.toFixed(2)}ms</td>
        <td>${r.frameIntervalMs.avg.toFixed(2)}ms</td>
        <td class="${fpsClass}">${actualFps}</td>
        <td>${panFps}</td>
        <td>${hoverFps}</td>
        <td>${r.hitTestMs.avg.toFixed(3)}ms</td>
        <td>${r.lassoMs.toFixed(2)}ms</td>
        <td>${r.memoryMB ? r.memoryMB.toFixed(0) + 'MB' : 'N/A'}</td>
      </tr>
    `;
  }

  html += `
      </tbody>
    </table>
  `;

  return html;
}

// ============================================================================
// Export for global access in browser
// ============================================================================

// Attach to window for use in benchmark.html
if (typeof window !== 'undefined') {
  (window as any).VizBenchmark = {
    runBenchmarks,
    runAccuracyBenchmarks,
    formatResultsTable,
    formatResultsHTML,
    DEFAULT_CONFIG,
    STRESS_CONFIG,
  };
}
