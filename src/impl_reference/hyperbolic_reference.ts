/**
 * Hyperbolic (Poincare Disk) Reference Implementation
 *
 * This is the naive but accurate Canvas2D implementation for hyperbolic space.
 * It serves as ground truth for comparison with optimized candidates.
 *
 * Key features:
 * - Correct Mobius transformation for camera navigation
 * - Anchor-invariant pan (point under cursor stays under cursor)
 * - Proper hyperbolic distance semantics
 * - Lasso selection in data space (unprojected)
 */

import {
  Dataset,
  Renderer,
  InitOptions,
  ViewState,
  HyperbolicViewState,
  Modifiers,
  HitResult,
  SelectionResult,
  CategoryVisibilityMask,
  InteractionStyle,
  CountSelectionOptions,
  DEFAULT_COLORS,
  SELECTION_COLOR,
  HOVER_COLOR,
  createIndicesSelectionResult,
} from '../core/types.js';
import {
  createHyperbolicView,
  projectPoincare,
  unprojectPoincare,
  panPoincare,
  zoomPoincare,
  mobiusTransform,
} from '../core/math/poincare.js';
import { pointInPolygon } from '../core/selection/point_in_polygon.js';

export class HyperbolicReference implements Renderer {
  private canvas!: HTMLCanvasElement;
  private ctx!: CanvasRenderingContext2D;
  private width = 0;
  private height = 0;
  private dpr = 1;

  private dataset: Dataset | null = null;
  private view: HyperbolicViewState = createHyperbolicView();
  private selection = new Set<number>();
  private highlight = new Set<number>();
  private hoveredIndex = -1;

  private pointRadius = 3;
  private colors = DEFAULT_COLORS;
  private backgroundColor = '#0a0a0a';
  private poincareDiskFillColor = '#141414';
  private poincareDiskBorderColor = '#666666';
  private poincareGridColor = '#2a2a2a';
  private poincareDiskBorderWidthPx = 2;
  private poincareGridWidthPx = 0.5;
  private categoryVisibilityMask = new Uint8Array(0);
  private hasCategoryVisibilityMask = false;
  private categoryAlpha = 1;
  private interactionStyle: Required<InteractionStyle> = {
    selectionColor: SELECTION_COLOR,
    selectionRadiusOffset: 0,
    selectionRingWidth: 1,
    highlightColor: '#94a3b8',
    highlightRadiusOffset: 0,
    highlightRingWidth: 1,
    hoverColor: HOVER_COLOR,
    hoverFillColor: null,
    hoverRadiusOffset: 0,
  };

  private isCategoryVisible(category: number): boolean {
    if (!this.hasCategoryVisibilityMask) return true;
    if (category < 0 || category >= this.categoryVisibilityMask.length) return true;
    return this.categoryVisibilityMask[category] !== 0;
  }

  private isPointVisibleByCategory(index: number): boolean {
    const ds = this.dataset;
    if (!ds || index < 0 || index >= ds.n) return false;
    return this.isCategoryVisible(ds.labels[index]);
  }

  init(canvas: HTMLCanvasElement, opts: InitOptions): void {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Failed to get 2D context');
    this.ctx = ctx;

    this.width = opts.width;
    this.height = opts.height;
    this.dpr = opts.devicePixelRatio ?? 1;

    if (opts.backgroundColor) this.backgroundColor = opts.backgroundColor;
    if (opts.pointRadius) this.pointRadius = opts.pointRadius;
    if (opts.colors) this.colors = opts.colors;
    if (opts.poincareDiskFillColor) this.poincareDiskFillColor = opts.poincareDiskFillColor;
    if (opts.poincareDiskBorderColor) this.poincareDiskBorderColor = opts.poincareDiskBorderColor;
    if (opts.poincareGridColor) this.poincareGridColor = opts.poincareGridColor;
    if (typeof opts.poincareDiskBorderWidthPx === 'number' && Number.isFinite(opts.poincareDiskBorderWidthPx)) {
      this.poincareDiskBorderWidthPx = Math.max(0, opts.poincareDiskBorderWidthPx);
    }
    if (typeof opts.poincareGridWidthPx === 'number' && Number.isFinite(opts.poincareGridWidthPx)) {
      this.poincareGridWidthPx = Math.max(0, opts.poincareGridWidthPx);
    }

    // Set canvas size
    canvas.width = this.width * this.dpr;
    canvas.height = this.height * this.dpr;
    canvas.style.width = `${this.width}px`;
    canvas.style.height = `${this.height}px`;

    // IMPORTANT: reset transform before applying DPR scaling.
    // Canvas contexts persist their transform across getContext() calls and
    // renderer swaps; repeated scale() would otherwise accumulate.
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.scale(this.dpr, this.dpr);
  }

  setDataset(dataset: Dataset): void {
    if (dataset.geometry !== 'poincare') {
      throw new Error('HyperbolicReference only supports poincare geometry');
    }
    this.dataset = dataset;
    this.selection.clear();
    this.hoveredIndex = -1;

    // Reset view to center
    this.view = createHyperbolicView();
  }

  setView(view: ViewState): void {
    if (view.type !== 'poincare') {
      throw new Error('HyperbolicReference only supports poincare view state');
    }
    this.view = view;
  }

  getView(): ViewState {
    return { ...this.view };
  }

  setSelection(indices: Set<number>): void {
    this.selection = new Set(indices);
  }

  setHighlight(indices: Set<number> | null): void {
    this.highlight = indices ? new Set(indices) : new Set<number>();
  }

  getHighlight(): Set<number> {
    return new Set(this.highlight);
  }

  setPalette(colors: string[]): void {
    this.colors = colors;
  }

  setCategoryVisibility(mask: CategoryVisibilityMask | null): void {
    if (mask == null) {
      this.categoryVisibilityMask = new Uint8Array(0);
      this.hasCategoryVisibilityMask = false;
    } else {
      const n = mask.length >>> 0;
      const next = new Uint8Array(n);
      for (let i = 0; i < n; i++) {
        next[i] = mask[i] ? 1 : 0;
      }
      this.categoryVisibilityMask = next;
      this.hasCategoryVisibilityMask = true;
    }

    if (this.hoveredIndex >= 0 && !this.isPointVisibleByCategory(this.hoveredIndex)) {
      this.hoveredIndex = -1;
    }
  }

  setCategoryAlpha(alpha: number): void {
    this.categoryAlpha = Number.isFinite(alpha) ? Math.max(0, Math.min(1, alpha)) : 1;
  }

  setInactiveOpacity(alpha: number): void {
    this.setCategoryAlpha(alpha);
  }

  setInteractionStyle(style: InteractionStyle): void {
    if (typeof style.selectionColor === 'string' && style.selectionColor.length > 0) {
      this.interactionStyle.selectionColor = style.selectionColor;
    }
    if (typeof style.hoverColor === 'string' && style.hoverColor.length > 0) {
      this.interactionStyle.hoverColor = style.hoverColor;
    }
    if (Object.prototype.hasOwnProperty.call(style, 'hoverFillColor')) {
      this.interactionStyle.hoverFillColor = style.hoverFillColor ?? null;
    }
  }

  getSelection(): Set<number> {
    return new Set(this.selection);
  }

  setLassoPolygon(_polygon: Float32Array | null): void {
    // Reference renderer keeps lasso overlay outside the core draw path.
  }

  setHovered(index: number): void {
    if (index >= 0 && !this.isPointVisibleByCategory(index)) {
      this.hoveredIndex = -1;
      return;
    }
    this.hoveredIndex = index;
  }

  render(): void {
    const { ctx, width, height, dataset, view } = this;
    if (!dataset) return;

    // Clear background
    ctx.fillStyle = this.backgroundColor;
    ctx.fillRect(0, 0, width, height);

    // Draw the Poincare disk boundary
    const diskRadius = Math.min(width, height) * 0.45 * view.displayZoom;
    const centerX = width / 2;
    const centerY = height / 2;

    // Disk background (slightly lighter)
    ctx.fillStyle = this.poincareDiskFillColor;
    ctx.beginPath();
    ctx.arc(centerX, centerY, diskRadius, 0, Math.PI * 2);
    ctx.fill();

    // Disk border
    ctx.strokeStyle = this.poincareDiskBorderColor;
    ctx.lineWidth = this.poincareDiskBorderWidthPx;
    ctx.beginPath();
    ctx.arc(centerX, centerY, diskRadius, 0, Math.PI * 2);
    ctx.stroke();

    // Draw hyperbolic grid lines (geodesics) - optional but helpful for visualization
    this.drawHyperbolicGrid(ctx, centerX, centerY, diskRadius);

    // Draw all points
    const { positions, labels, n } = dataset;
    const radius = this.pointRadius;

    // First pass: all non-hovered points
    ctx.globalAlpha = this.categoryAlpha;
    for (let i = 0; i < n; i++) {
      if (i === this.hoveredIndex) continue;
      if (!this.isCategoryVisible(labels[i])) continue;

      const dataX = positions[i * 2];
      const dataY = positions[i * 2 + 1];
      const screen = projectPoincare(dataX, dataY, view, width, height);

      // Check if inside disk (after transform)
      const dx = screen.x - centerX;
      const dy = screen.y - centerY;
      if (dx * dx + dy * dy > diskRadius * diskRadius) continue;

      ctx.fillStyle = this.colors[labels[i] % this.colors.length];
      ctx.beginPath();
      ctx.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    // Second pass: selected points
    ctx.strokeStyle = this.interactionStyle.selectionColor;
    ctx.lineWidth = 2;
    for (const i of this.selection) {
      if (i === this.hoveredIndex) continue;
      if (!this.isPointVisibleByCategory(i)) continue;

      const dataX = positions[i * 2];
      const dataY = positions[i * 2 + 1];
      const screen = projectPoincare(dataX, dataY, view, width, height);

      ctx.beginPath();
      ctx.arc(screen.x, screen.y, radius + 1, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Third pass: hovered point
    if (this.hoveredIndex >= 0 && this.hoveredIndex < n && this.isPointVisibleByCategory(this.hoveredIndex)) {
      const i = this.hoveredIndex;
      const dataX = positions[i * 2];
      const dataY = positions[i * 2 + 1];
      const screen = projectPoincare(dataX, dataY, view, width, height);

      ctx.strokeStyle = this.interactionStyle.hoverColor;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(screen.x, screen.y, radius + 3, 0, Math.PI * 2);
      ctx.stroke();

      if (this.selection.has(i)) {
        ctx.strokeStyle = this.interactionStyle.selectionColor;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(screen.x, screen.y, radius + 1, 0, Math.PI * 2);
        ctx.stroke();

        ctx.fillStyle = this.colors[labels[i] % this.colors.length];
        ctx.beginPath();
        ctx.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.fillStyle = this.interactionStyle.hoverFillColor ?? this.colors[labels[i] % this.colors.length];
        ctx.beginPath();
        ctx.arc(screen.x, screen.y, radius + 1, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  /**
   * Draw hyperbolic geodesics through the origin.
   * In the Poincare disk, geodesics through the origin are straight lines.
   * Other geodesics are circular arcs perpendicular to the boundary.
   */
  private drawHyperbolicGrid(
    ctx: CanvasRenderingContext2D,
    centerX: number,
    centerY: number,
    diskRadius: number
  ): void {
    ctx.strokeStyle = this.poincareGridColor;
    ctx.lineWidth = this.poincareGridWidthPx;

    // Draw radial lines (geodesics through origin after camera transform)
    const numRadial = 8;
    for (let i = 0; i < numRadial; i++) {
      const angle = (i / numRadial) * Math.PI;
      ctx.beginPath();
      ctx.moveTo(
        centerX - diskRadius * Math.cos(angle),
        centerY - diskRadius * Math.sin(angle)
      );
      ctx.lineTo(
        centerX + diskRadius * Math.cos(angle),
        centerY + diskRadius * Math.sin(angle)
      );
      ctx.stroke();
    }

    // Draw concentric circles (horocycles)
    const numCircles = 5;
    for (let i = 1; i <= numCircles; i++) {
      const r = (i / (numCircles + 1)) * diskRadius;
      ctx.beginPath();
      ctx.arc(centerX, centerY, r, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
    this.canvas.width = width * this.dpr;
    this.canvas.height = height * this.dpr;
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.scale(this.dpr, this.dpr);
  }

  destroy(): void {
    // Nothing to clean up for Canvas2D
  }

  // === Interaction Methods ===

  private lastPanScreenX = 0;
  private lastPanScreenY = 0;
  private hasPanAnchor = false;

  pan(deltaX: number, deltaY: number, _modifiers: Modifiers): void {
    // Hyperbolic pan must be anchor-invariant w.r.t. the cursor.
    // The `Renderer.pan()` contract only gives us incremental deltas, so we
    // track the last known cursor position (set via startPan()).
    //
    // If startPan() was not called, fall back to using the canvas center as the
    // anchor to avoid undefined behavior.
    if (!this.hasPanAnchor) {
      this.lastPanScreenX = this.width / 2;
      this.lastPanScreenY = this.height / 2;
      this.hasPanAnchor = true;
    }

    const startX = this.lastPanScreenX;
    const startY = this.lastPanScreenY;
    const endX = startX + deltaX;
    const endY = startY + deltaY;

    this.view = panPoincare(this.view, startX, startY, endX, endY, this.width, this.height);

    this.lastPanScreenX = endX;
    this.lastPanScreenY = endY;
  }

  /** Call this at the start of a pan gesture with the cursor position (in screen px). */
  startPan(screenX: number, screenY: number): void {
    this.lastPanScreenX = screenX;
    this.lastPanScreenY = screenY;
    this.hasPanAnchor = true;
  }

  zoom(anchorX: number, anchorY: number, delta: number, _modifiers: Modifiers): void {
    this.view = zoomPoincare(this.view, anchorX, anchorY, delta, this.width, this.height);
  }

  hitTest(screenX: number, screenY: number): HitResult | null {
    if (!this.dataset) return null;

    const { positions, n } = this.dataset;
    const { view, width, height } = this;

    // Match render() culling: ignore points outside the displayed disk.
    const centerX = width / 2;
    const centerY = height / 2;
    const diskRadius = Math.min(width, height) * 0.45 * view.displayZoom;

    let bestIndex = -1;
    let bestDistSq = Infinity;
    const maxDistSq = (this.pointRadius + 5) ** 2;

    // Brute force: check all points
    for (let i = 0; i < n; i++) {
      if (!this.isCategoryVisible(this.dataset.labels[i])) continue;
      const dataX = positions[i * 2];
      const dataY = positions[i * 2 + 1];
      const screen = projectPoincare(dataX, dataY, view, width, height);

      const dxDisk = screen.x - centerX;
      const dyDisk = screen.y - centerY;
      if (dxDisk * dxDisk + dyDisk * dyDisk > diskRadius * diskRadius) continue;

      const dx = screen.x - screenX;
      const dy = screen.y - screenY;
      const distSq = dx * dx + dy * dy;

      if (distSq < bestDistSq && distSq <= maxDistSq) {
        bestDistSq = distSq;
        bestIndex = i;
      }
    }

    if (bestIndex < 0) return null;

    const dataX = positions[bestIndex * 2];
    const dataY = positions[bestIndex * 2 + 1];
    const screen = projectPoincare(dataX, dataY, view, width, height);

    return {
      index: bestIndex,
      screenX: screen.x,
      screenY: screen.y,
      distance: Math.sqrt(bestDistSq),
    };
  }

  lassoSelect(polyline: Float32Array): SelectionResult {
    if (!this.dataset) {
      return createIndicesSelectionResult(new Set(), 0);
    }

    const startTime = performance.now();

    // Transform polyline from screen to data space (Poincare coordinates)
    const dataPolyline = new Float32Array(polyline.length);
    for (let i = 0; i < polyline.length / 2; i++) {
      const screenX = polyline[i * 2];
      const screenY = polyline[i * 2 + 1];
      const data = unprojectPoincare(screenX, screenY, this.view, this.width, this.height);
      dataPolyline[i * 2] = data.x;
      dataPolyline[i * 2 + 1] = data.y;
    }

    const indices = new Set<number>();
    for (let i = 0; i < this.dataset.n; i++) {
      if (!this.isCategoryVisible(this.dataset.labels[i])) continue;
      const x = this.dataset.positions[i * 2];
      const y = this.dataset.positions[i * 2 + 1];
      if (pointInPolygon(x, y, dataPolyline)) {
        indices.add(i);
      }
    }

    const computeTimeMs = performance.now() - startTime;
    return createIndicesSelectionResult(indices, computeTimeMs);
  }

  async countSelection(result: SelectionResult, _opts: CountSelectionOptions = {}): Promise<number> {
    if (!result.indices) {
      throw new Error('HyperbolicReference.countSelection expects indices-based selections');
    }
    let visibleCount = 0;
    for (const i of result.indices) {
      if (this.isPointVisibleByCategory(i)) visibleCount++;
    }
    return visibleCount;
  }

  projectToScreen(dataX: number, dataY: number): { x: number; y: number } {
    return projectPoincare(dataX, dataY, this.view, this.width, this.height);
  }

  unprojectFromScreen(screenX: number, screenY: number): { x: number; y: number } {
    return unprojectPoincare(screenX, screenY, this.view, this.width, this.height);
  }
}
