/**
 * Mathematical Accuracy Comparison
 *
 * Validates optimized implementations against reference by comparing:
 * - View states before/after operations
 * - Projection outputs
 * - Hit test results
 * - Selection results
 */

import {
  Dataset,
  Renderer,
  ViewState,
  EuclideanViewState,
  HyperbolicViewState,
  Modifiers,
} from '../core/types.js';
import { arraysApproxEqual } from './utils.js';

// ============================================================================
// Types
// ============================================================================

export interface Snapshot {
  /** View state at capture time */
  view: ViewState;
  /** All points projected to screen coordinates [x0, y0, x1, y1, ...] */
  projectedPositions: Float32Array;
  /** Capture timestamp */
  timestamp: number;
}

export interface ComparisonResult {
  passed: boolean;
  viewMatch: boolean;
  projectionMatch: boolean;
  maxViewError: number;
  maxProjectionError: number;
  errorDetails?: string;
}

export interface OperationResult {
  operation: string;
  params: Record<string, unknown>;
  reference: ViewState;
  candidate: ViewState;
  maxError: number;
  passed: boolean;
  details?: string;
}

export interface AccuracyReport {
  timestamp: string;
  geometry: 'euclidean' | 'poincare';
  pointCount: number;
  tests: OperationResult[];
  allPassed: boolean;
  summary: string;
}

// ============================================================================
// Snapshot Capture
// ============================================================================

/**
 * Capture a snapshot of the renderer's current state.
 */
export function captureSnapshot(
  renderer: Renderer,
  dataset: Dataset,
  _width: number,
  _height: number
): Snapshot {
  const view = renderer.getView();
  const projectedPositions = new Float32Array(dataset.n * 2);

  for (let i = 0; i < dataset.n; i++) {
    const dataX = dataset.positions[i * 2];
    const dataY = dataset.positions[i * 2 + 1];
    const screen = renderer.projectToScreen(dataX, dataY);
    projectedPositions[i * 2] = screen.x;
    projectedPositions[i * 2 + 1] = screen.y;
  }

  return {
    view,
    projectedPositions,
    timestamp: performance.now(),
  };
}

// ============================================================================
// View State Comparison
// ============================================================================

/**
 * Compare two Euclidean view states.
 */
export function compareEuclideanViews(
  a: EuclideanViewState,
  b: EuclideanViewState,
  epsilon = 1e-10
): { match: boolean; maxError: number; details?: string } {
  const errors = [
    Math.abs(a.centerX - b.centerX),
    Math.abs(a.centerY - b.centerY),
    Math.abs(a.zoom - b.zoom),
  ];
  const maxError = Math.max(...errors);

  if (maxError > epsilon) {
    return {
      match: false,
      maxError,
      details: `centerX: ${a.centerX} vs ${b.centerX}, centerY: ${a.centerY} vs ${b.centerY}, zoom: ${a.zoom} vs ${b.zoom}`,
    };
  }

  return { match: true, maxError };
}

/**
 * Compare two Hyperbolic view states.
 */
export function compareHyperbolicViews(
  a: HyperbolicViewState,
  b: HyperbolicViewState,
  epsilon = 1e-10
): { match: boolean; maxError: number; details?: string } {
  const errors = [
    Math.abs(a.ax - b.ax),
    Math.abs(a.ay - b.ay),
    Math.abs(a.displayZoom - b.displayZoom),
  ];
  const maxError = Math.max(...errors);

  if (maxError > epsilon) {
    return {
      match: false,
      maxError,
      details: `ax: ${a.ax} vs ${b.ax}, ay: ${a.ay} vs ${b.ay}, displayZoom: ${a.displayZoom} vs ${b.displayZoom}`,
    };
  }

  return { match: true, maxError };
}

/**
 * Compare two view states (auto-detect type).
 */
export function compareViewStates(
  a: ViewState,
  b: ViewState,
  epsilon = 1e-10
): { match: boolean; maxError: number; details?: string } {
  if (a.type !== b.type) {
    return {
      match: false,
      maxError: Infinity,
      details: `Type mismatch: ${a.type} vs ${b.type}`,
    };
  }

  if (a.type === 'euclidean') {
    return compareEuclideanViews(a, b as EuclideanViewState, epsilon);
  } else {
    return compareHyperbolicViews(a, b as HyperbolicViewState, epsilon);
  }
}

// ============================================================================
// Snapshot Comparison
// ============================================================================

/**
 * Compare two snapshots for equality within tolerance.
 */
export function compareSnapshots(
  a: Snapshot,
  b: Snapshot,
  epsilon = 1e-6
): ComparisonResult {
  const viewComparison = compareViewStates(a.view, b.view, epsilon);
  const projectionComparison = arraysApproxEqual(
    a.projectedPositions,
    b.projectedPositions,
    epsilon
  );

  const passed = viewComparison.match && projectionComparison.equal;

  return {
    passed,
    viewMatch: viewComparison.match,
    projectionMatch: projectionComparison.equal,
    maxViewError: viewComparison.maxError,
    maxProjectionError: projectionComparison.maxError,
    errorDetails: passed
      ? undefined
      : [
          viewComparison.details,
          projectionComparison.equal
            ? null
            : `Projection error at index ${projectionComparison.errorIndex}: ${projectionComparison.maxError}`,
        ]
          .filter(Boolean)
          .join('; '),
  };
}

// ============================================================================
// Operation Comparison
// ============================================================================

const DEFAULT_MODIFIERS: Modifiers = {
  shift: false,
  ctrl: false,
  alt: false,
  meta: false,
};

/**
 * Compare pan operation between two renderers.
 */
export function comparePan(
  refRenderer: Renderer,
  candidateRenderer: Renderer,
  startX: number,
  startY: number,
  deltaX: number,
  deltaY: number,
  epsilon = 1e-10
): OperationResult {
  refRenderer.startPan(startX, startY);
  candidateRenderer.startPan(startX, startY);

  // Apply pan
  refRenderer.pan(deltaX, deltaY, DEFAULT_MODIFIERS);
  candidateRenderer.pan(deltaX, deltaY, DEFAULT_MODIFIERS);

  // Compare results
  const refView = refRenderer.getView();
  const candView = candidateRenderer.getView();
  const comparison = compareViewStates(refView, candView, epsilon);

  return {
    operation: 'pan',
    params: { startX, startY, deltaX, deltaY },
    reference: refView,
    candidate: candView,
    maxError: comparison.maxError,
    passed: comparison.match,
    details: comparison.details,
  };
}

/**
 * Compare zoom operation between two renderers.
 */
export function compareZoom(
  refRenderer: Renderer,
  candidateRenderer: Renderer,
  anchorX: number,
  anchorY: number,
  delta: number,
  epsilon = 1e-10
): OperationResult {
  // Apply zoom
  refRenderer.zoom(anchorX, anchorY, delta, DEFAULT_MODIFIERS);
  candidateRenderer.zoom(anchorX, anchorY, delta, DEFAULT_MODIFIERS);

  // Compare results
  const refView = refRenderer.getView();
  const candView = candidateRenderer.getView();
  const comparison = compareViewStates(refView, candView, epsilon);

  return {
    operation: 'zoom',
    params: { anchorX, anchorY, delta },
    reference: refView,
    candidate: candView,
    maxError: comparison.maxError,
    passed: comparison.match,
    details: comparison.details,
  };
}

/**
 * Compare projection operation for a single point.
 */
export function compareProjection(
  refRenderer: Renderer,
  candidateRenderer: Renderer,
  dataX: number,
  dataY: number,
  epsilon = 1e-6
): OperationResult {
  const refScreen = refRenderer.projectToScreen(dataX, dataY);
  const candScreen = candidateRenderer.projectToScreen(dataX, dataY);

  const errorX = Math.abs(refScreen.x - candScreen.x);
  const errorY = Math.abs(refScreen.y - candScreen.y);
  const maxError = Math.max(errorX, errorY);
  const passed = maxError <= epsilon;

  return {
    operation: 'projectToScreen',
    params: { dataX, dataY },
    reference: { x: refScreen.x, y: refScreen.y } as any,
    candidate: { x: candScreen.x, y: candScreen.y } as any,
    maxError,
    passed,
    details: passed
      ? undefined
      : `ref=(${refScreen.x}, ${refScreen.y}) vs cand=(${candScreen.x}, ${candScreen.y})`,
  };
}

/**
 * Compare unprojection operation for a single point.
 */
export function compareUnprojection(
  refRenderer: Renderer,
  candidateRenderer: Renderer,
  screenX: number,
  screenY: number,
  epsilon = 1e-6
): OperationResult {
  const refData = refRenderer.unprojectFromScreen(screenX, screenY);
  const candData = candidateRenderer.unprojectFromScreen(screenX, screenY);

  const errorX = Math.abs(refData.x - candData.x);
  const errorY = Math.abs(refData.y - candData.y);
  const maxError = Math.max(errorX, errorY);
  const passed = maxError <= epsilon;

  return {
    operation: 'unprojectFromScreen',
    params: { screenX, screenY },
    reference: { x: refData.x, y: refData.y } as any,
    candidate: { x: candData.x, y: candData.y } as any,
    maxError,
    passed,
    details: passed
      ? undefined
      : `ref=(${refData.x}, ${refData.y}) vs cand=(${candData.x}, ${candData.y})`,
  };
}

/**
 * Compare hit test results.
 */
export function compareHitTest(
  refRenderer: Renderer,
  candidateRenderer: Renderer,
  screenX: number,
  screenY: number
): OperationResult {
  const refHit = refRenderer.hitTest(screenX, screenY);
  const candHit = candidateRenderer.hitTest(screenX, screenY);

  const refIndex = refHit?.index ?? -1;
  const candIndex = candHit?.index ?? -1;
  const passed = refIndex === candIndex;

  return {
    operation: 'hitTest',
    params: { screenX, screenY },
    reference: { index: refIndex } as any,
    candidate: { index: candIndex } as any,
    maxError: passed ? 0 : 1,
    passed,
    details: passed ? undefined : `ref=${refIndex} vs cand=${candIndex}`,
  };
}

/**
 * Compare lasso selection results.
 * Uses the has() method for membership verification, allowing implementations
 * to return either indices or geometry-based selections.
 */
export function compareLassoSelect(
  refRenderer: Renderer,
  candidateRenderer: Renderer,
  polyline: Float32Array,
  n: number,
  positions?: Float32Array,
  testName?: string
): OperationResult {
  const refResult = refRenderer.lassoSelect(polyline);
  const candResult = candidateRenderer.lassoSelect(polyline);

  // Verify membership for all points using has()
  // Collect up to MAX_SAMPLE mismatches for detailed reporting
  const MAX_SAMPLE = 5;
  const mismatches: Array<{ index: number; inRef: boolean; inCand: boolean }> = [];
  let mismatchCount = 0;
  let refCount = 0;
  let candCount = 0;

  for (let i = 0; i < n; i++) {
    const inRef = refResult.has(i);
    const inCand = candResult.has(i);
    if (inRef) refCount++;
    if (inCand) candCount++;
    if (inRef !== inCand) {
      mismatchCount++;
      if (mismatches.length < MAX_SAMPLE) {
        mismatches.push({ index: i, inRef, inCand });
      }
    }
  }

  const passed = mismatchCount === 0;

  // Build detailed error message
  let details: string | undefined;
  if (!passed) {
    const lines: string[] = [
      `ref=${refCount} points, cand=${candCount} points (${mismatchCount} mismatches)`,
    ];
    for (const m of mismatches) {
      let coordStr = '';
      if (positions) {
        const x = positions[m.index * 2];
        const y = positions[m.index * 2 + 1];
        coordStr = ` at (${x.toFixed(6)}, ${y.toFixed(6)})`;
      }
      lines.push(`  #${m.index}${coordStr}: ref=${m.inRef}, cand=${m.inCand}`);
    }
    if (mismatchCount > MAX_SAMPLE) {
      lines.push(`  ... and ${mismatchCount - MAX_SAMPLE} more`);
    }
    details = lines.join('\n');
  }

  return {
    operation: testName ? `lassoSelect[${testName}]` : 'lassoSelect',
    params: { polylineLength: polyline.length / 2, testName },
    reference: { count: refCount } as any,
    candidate: { count: candCount } as any,
    maxError: passed ? 0 : mismatchCount,
    passed,
    details,
  };
}

// ============================================================================
// Roundtrip Tests
// ============================================================================

/**
 * Test projection/unprojection roundtrip accuracy.
 */
export function testProjectionRoundtrip(
  renderer: Renderer,
  dataX: number,
  dataY: number,
  epsilon = 1e-6
): { passed: boolean; maxError: number; details?: string } {
  const screen = renderer.projectToScreen(dataX, dataY);
  const roundtrip = renderer.unprojectFromScreen(screen.x, screen.y);

  const errorX = Math.abs(roundtrip.x - dataX);
  const errorY = Math.abs(roundtrip.y - dataY);
  const maxError = Math.max(errorX, errorY);
  const passed = maxError <= epsilon;

  return {
    passed,
    maxError,
    details: passed
      ? undefined
      : `(${dataX}, ${dataY}) -> (${screen.x}, ${screen.y}) -> (${roundtrip.x}, ${roundtrip.y})`,
  };
}

// ============================================================================
// Full Accuracy Test Suite
// ============================================================================

/**
 * Test projection roundtrip for both renderers.
 * This ensures project -> unproject returns the original point.
 */
export function compareProjectionRoundtrip(
  refRenderer: Renderer,
  candidateRenderer: Renderer,
  dataX: number,
  dataY: number,
  epsilon = 1e-6
): OperationResult {
  const refResult = testProjectionRoundtrip(refRenderer, dataX, dataY, epsilon);
  const candResult = testProjectionRoundtrip(candidateRenderer, dataX, dataY, epsilon);

  const maxError = Math.max(refResult.maxError, candResult.maxError);
  const passed = refResult.passed && candResult.passed;

  return {
    operation: 'projectionRoundtrip',
    params: { dataX, dataY },
    reference: { maxError: refResult.maxError } as any,
    candidate: { maxError: candResult.maxError } as any,
    maxError,
    passed,
    details: passed
      ? undefined
      : `ref=${refResult.maxError.toExponential(2)}, cand=${candResult.maxError.toExponential(2)}`,
  };
}

/**
 * Run a comprehensive accuracy test suite comparing two renderers.
 */
export function runAccuracyTests(
  refRenderer: Renderer,
  candidateRenderer: Renderer,
  dataset: Dataset,
  width: number,
  height: number,
  epsilon = 1e-6
): AccuracyReport {
  const tests: OperationResult[] = [];
  const geometry = dataset.geometry === 'euclidean' ? 'euclidean' : 'poincare';

  // Reset both renderers to same initial state
  refRenderer.setDataset(dataset);
  candidateRenderer.setDataset(dataset);

  // Sync views
  const initialView = refRenderer.getView();
  candidateRenderer.setView(initialView);

  // Test 1: Initial projection comparison (sample points)
  const sampleIndices = [0, Math.floor(dataset.n / 4), Math.floor(dataset.n / 2), dataset.n - 1];
  for (const i of sampleIndices) {
    const dataX = dataset.positions[i * 2];
    const dataY = dataset.positions[i * 2 + 1];
    tests.push(compareProjection(refRenderer, candidateRenderer, dataX, dataY, epsilon));
  }

  // Test 2: Projection roundtrip for sample points
  for (const i of sampleIndices) {
    const dataX = dataset.positions[i * 2];
    const dataY = dataset.positions[i * 2 + 1];
    tests.push(compareProjectionRoundtrip(refRenderer, candidateRenderer, dataX, dataY, epsilon));
  }

  // Test 3: Edge case tests for hyperbolic (points near boundary)
  if (geometry === 'poincare') {
    const boundaryPoints = [
      { x: 0.95, y: 0 },     // Near right boundary
      { x: -0.9, y: 0.3 },   // Near left boundary
      { x: 0.1, y: 0.92 },   // Near top boundary
      { x: 0, y: 0 },        // Origin (easy case)
    ];
    for (const pt of boundaryPoints) {
      tests.push(compareProjection(refRenderer, candidateRenderer, pt.x, pt.y, epsilon * 10));
      tests.push(compareProjectionRoundtrip(refRenderer, candidateRenderer, pt.x, pt.y, epsilon * 10));
    }
  }

  // Test 4: Pan operations
  const panTests = [
    { startX: width / 2, startY: height / 2, deltaX: 50, deltaY: 0 },
    { startX: width / 2, startY: height / 2, deltaX: 0, deltaY: -30 },
    { startX: width / 4, startY: height / 4, deltaX: -20, deltaY: 40 },
  ];

  for (const pan of panTests) {
    // Reset views before each pan test
    refRenderer.setView(initialView);
    candidateRenderer.setView(initialView);
    tests.push(
      comparePan(refRenderer, candidateRenderer, pan.startX, pan.startY, pan.deltaX, pan.deltaY, epsilon)
    );
  }

  // Test 5: Zoom operations
  refRenderer.setView(initialView);
  candidateRenderer.setView(initialView);

  const zoomTests = [
    { anchorX: width / 2, anchorY: height / 2, delta: 0.5 },
    { anchorX: width / 2, anchorY: height / 2, delta: -0.3 },
    { anchorX: width / 4, anchorY: height / 4, delta: 0.2 },
  ];

  for (const zoom of zoomTests) {
    // Reset views before each zoom test
    refRenderer.setView(initialView);
    candidateRenderer.setView(initialView);
    tests.push(
      compareZoom(refRenderer, candidateRenderer, zoom.anchorX, zoom.anchorY, zoom.delta, epsilon)
    );
  }

  // Test 6: Extreme zoom (edge case)
  refRenderer.setView(initialView);
  candidateRenderer.setView(initialView);
  tests.push(compareZoom(refRenderer, candidateRenderer, width / 2, height / 2, 3.0, epsilon * 10));
  refRenderer.setView(initialView);
  candidateRenderer.setView(initialView);
  tests.push(compareZoom(refRenderer, candidateRenderer, width / 2, height / 2, -2.0, epsilon * 10));

  // Test 7: Hit tests (various screen positions)
  refRenderer.setView(initialView);
  candidateRenderer.setView(initialView);

  const hitTestPositions = [
    { x: width / 2, y: height / 2 },
    { x: width / 4, y: height / 4 },
    { x: (width * 3) / 4, y: (height * 3) / 4 },
  ];

  // Hyperbolic edge case: cursor slightly outside the disk boundary.
  // Reference hitTest does NOT require the cursor to be inside the disk; it
  // only culls points whose projected positions are outside the disk.
  //
  // To make this deterministic (and likely to produce a hit), we:
  // 1) Find the dataset point whose *projected* position is closest to the disk boundary.
  // 2) Test hitTest at that point's screen position (should hit itself).
  // 3) Move the cursor just outside the disk along the radial direction while
  //    still within typical hit radius.
  if (geometry === 'poincare') {
    const v = initialView as HyperbolicViewState;
    const centerX = width / 2;
    const centerY = height / 2;
    const diskRadius = Math.min(width, height) * 0.45 * v.displayZoom;

    let bestI = 0;
    let bestR = -Infinity;
    for (let i = 0; i < dataset.n; i++) {
      const dataX = dataset.positions[i * 2];
      const dataY = dataset.positions[i * 2 + 1];
      const s = refRenderer.projectToScreen(dataX, dataY);
      const dx = s.x - centerX;
      const dy = s.y - centerY;
      const r = Math.hypot(dx, dy);
      if (r > bestR) {
        bestR = r;
        bestI = i;
      }
    }

    const bx = dataset.positions[bestI * 2];
    const by = dataset.positions[bestI * 2 + 1];
    const bScreen = refRenderer.projectToScreen(bx, by);

    // Inside hit: should reliably pick bestI.
    hitTestPositions.push({ x: bScreen.x, y: bScreen.y });

    // Outside hit: move to just outside the disk boundary along the radial direction.
    const dx = bScreen.x - centerX;
    const dy = bScreen.y - centerY;
    const r = Math.max(1e-9, Math.hypot(dx, dy));
    // Put cursor 1 CSS px outside disk.
    const needed = (diskRadius + 1) - r;
    // If the best point isn't near the boundary for some reason, this may move
    // too far and result in a null hit for both implementations (still fine).
    const outX = bScreen.x + (dx / r) * needed;
    const outY = bScreen.y + (dy / r) * needed;
    hitTestPositions.push({ x: outX, y: outY });
  }

  for (const pos of hitTestPositions) {
    tests.push(compareHitTest(refRenderer, candidateRenderer, pos.x, pos.y));
  }

  // Test 8: Lasso selection with multiple polygons
  // Each polygon tests different aspects of the selection algorithm
  const lassoTests: Array<{ name: string; coords: number[] }> = [
    // Centered large square (40%) - original test, covers many points
    {
      name: 'center-large',
      coords: [0.3, 0.3, 0.7, 0.3, 0.7, 0.7, 0.3, 0.7],
    },
    // Small centered square (10%) - tests precision with fewer points
    {
      name: 'center-small',
      coords: [0.45, 0.45, 0.55, 0.45, 0.55, 0.55, 0.45, 0.55],
    },
    // Off-center: top-left corner
    {
      name: 'top-left',
      coords: [0.05, 0.05, 0.25, 0.05, 0.25, 0.25, 0.05, 0.25],
    },
    // Off-center: bottom-right corner
    {
      name: 'bottom-right',
      coords: [0.75, 0.75, 0.95, 0.75, 0.95, 0.95, 0.75, 0.95],
    },
    // Triangle - tests non-rectangular shapes
    {
      name: 'triangle',
      coords: [0.5, 0.2, 0.8, 0.7, 0.2, 0.7],
    },
    // Concave polygon (arrow/chevron) - tests winding/ray-casting edge cases
    {
      name: 'concave',
      coords: [0.3, 0.3, 0.5, 0.45, 0.7, 0.3, 0.7, 0.7, 0.3, 0.7],
    },
    // Thin horizontal strip - tests elongated shapes
    {
      name: 'thin-horizontal',
      coords: [0.1, 0.48, 0.9, 0.48, 0.9, 0.52, 0.1, 0.52],
    },
  ];

  // Add hyperbolic-specific tests
  if (geometry === 'poincare') {
    // For hyperbolic, also test near the disk boundary
    // The disk is centered at (0.5, 0.5) in normalized coords with radius ~0.45
    lassoTests.push({
      name: 'near-boundary',
      coords: [0.7, 0.4, 0.9, 0.4, 0.9, 0.6, 0.7, 0.6],
    });
    // Polygon crossing through the disk center
    lassoTests.push({
      name: 'through-center',
      coords: [0.2, 0.4, 0.8, 0.4, 0.8, 0.6, 0.2, 0.6],
    });
  }

  for (const test of lassoTests) {
    const polygon = new Float32Array(
      test.coords.map((v, i) => (i % 2 === 0 ? v * width : v * height))
    );
    tests.push(compareLassoSelect(
      refRenderer,
      candidateRenderer,
      polygon,
      dataset.n,
      dataset.positions,
      test.name
    ));
  }

  // Generate report
  const allPassed = tests.every(t => t.passed);
  const failedCount = tests.filter(t => !t.passed).length;

  return {
    timestamp: new Date().toISOString(),
    geometry,
    pointCount: dataset.n,
    tests,
    allPassed,
    summary: allPassed
      ? `All ${tests.length} tests passed`
      : `${failedCount}/${tests.length} tests failed`,
  };
}

/**
 * Format an accuracy report for console output.
 */
export function formatAccuracyReport(report: AccuracyReport): string {
  const lines: string[] = [
    `\n${'='.repeat(60)}`,
    `ACCURACY REPORT - ${report.geometry.toUpperCase()}`,
    `${'='.repeat(60)}`,
    `Timestamp: ${report.timestamp}`,
    `Points: ${report.pointCount.toLocaleString()}`,
    `Result: ${report.allPassed ? 'PASSED' : 'FAILED'}`,
    `Summary: ${report.summary}`,
    `${'-'.repeat(60)}`,
  ];

  for (const test of report.tests) {
    const status = test.passed ? '[PASS]' : '[FAIL]';
    lines.push(`${status} ${test.operation}: maxError=${test.maxError.toExponential(2)}`);
    if (!test.passed && test.details) {
      lines.push(`       ${test.details}`);
    }
  }

  lines.push(`${'='.repeat(60)}\n`);
  return lines.join('\n');
}
