/**
 * Euclidean Reference Implementation
 *
 * This is the naive but accurate Canvas2D implementation.
 * It serves as ground truth for comparison with optimized candidates.
 */

import {
  Dataset,
  Renderer,
  InitOptions,
  ViewState,
  EuclideanViewState,
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
  createEuclideanView,
  projectEuclidean,
  unprojectEuclidean,
  panEuclidean,
  zoomEuclidean,
} from '../core/math/euclidean.js';
import { pointInPolygon } from '../core/selection/point_in_polygon.js';

export class EuclideanReference implements Renderer {
  private canvas!: HTMLCanvasElement;
  private ctx!: CanvasRenderingContext2D;
  private width = 0;
  private height = 0;
  private dpr = 1;

  private dataset: Dataset | null = null;
  private view: EuclideanViewState = createEuclideanView();
  private selection = new Set<number>();
  private hoveredIndex = -1;

  private pointRadius = 3;
  private colors = DEFAULT_COLORS;
  private backgroundColor = '#0a0a0a';
  private categoryVisibilityMask = new Uint8Array(0);
  private hasCategoryVisibilityMask = false;
  private categoryAlpha = 1;
  private interactionStyle: Required<InteractionStyle> = {
    selectionColor: SELECTION_COLOR,
    hoverColor: HOVER_COLOR,
    hoverFillColor: null,
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

    // Set canvas size
    canvas.width = this.width * this.dpr;
    canvas.height = this.height * this.dpr;
    canvas.style.width = `${this.width}px`;
    canvas.style.height = `${this.height}px`;

    // IMPORTANT: reset transform before applying DPR scaling.
    // Canvas contexts persist their transform across calls to getContext() and
    // across renderer swaps, so calling scale() repeatedly would accumulate.
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.scale(this.dpr, this.dpr);
  }

  setDataset(dataset: Dataset): void {
    if (dataset.geometry !== 'euclidean') {
      throw new Error('EuclideanReference only supports euclidean geometry');
    }
    this.dataset = dataset;
    this.selection.clear();
    this.hoveredIndex = -1;

    // Auto-fit view to data
    this.fitToData();
  }

  private fitToData(): void {
    if (!this.dataset || this.dataset.n === 0) return;

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    for (let i = 0; i < this.dataset.n; i++) {
      const x = this.dataset.positions[i * 2];
      const y = this.dataset.positions[i * 2 + 1];
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }

    // Calculate zoom to fit data with 10% padding
    const dataWidth = maxX - minX || 1;
    const dataHeight = maxY - minY || 1;
    const dataSize = Math.max(dataWidth, dataHeight);

    // The projection formula: screenPos = canvasCenter + (dataPos - viewCenter) * baseScale * zoom
    // where baseScale = min(width, height) * 0.4
    // To fit dataSize into canvas with padding: dataSize * baseScale * zoom = canvasSize * 0.8
    // zoom = (canvasSize * 0.8) / (dataSize * baseScale)
    // zoom = 0.8 / (dataSize * 0.4) = 2 / dataSize
    const fitZoom = 2 / dataSize;

    this.view = {
      type: 'euclidean',
      centerX: (minX + maxX) / 2,
      centerY: (minY + maxY) / 2,
      zoom: Math.max(0.1, Math.min(100, fitZoom)),
    };
  }

  setView(view: ViewState): void {
    if (view.type !== 'euclidean') {
      throw new Error('EuclideanReference only supports euclidean view state');
    }
    this.view = view;
  }

  getView(): ViewState {
    return { ...this.view };
  }

  setSelection(indices: Set<number>): void {
    this.selection = new Set(indices);
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

    // Draw all points (naive: no culling, no batching)
    const { positions, labels, n } = dataset;
    const radius = this.pointRadius;

    // First pass: all non-hovered points
    ctx.globalAlpha = this.categoryAlpha;
    for (let i = 0; i < n; i++) {
      if (i === this.hoveredIndex) continue;
      if (!this.isCategoryVisible(labels[i])) continue;

      const dataX = positions[i * 2];
      const dataY = positions[i * 2 + 1];
      const screen = projectEuclidean(dataX, dataY, view, width, height);

      // Simple frustum culling
      if (screen.x < -radius || screen.x > width + radius ||
          screen.y < -radius || screen.y > height + radius) {
        continue;
      }

      ctx.fillStyle = this.colors[labels[i] % this.colors.length];
      ctx.beginPath();
      ctx.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    // Second pass: selected points (ring drawn on top)
    ctx.strokeStyle = this.interactionStyle.selectionColor;
    ctx.lineWidth = 2;
    for (const i of this.selection) {
      if (i === this.hoveredIndex) continue;
      if (!this.isPointVisibleByCategory(i)) continue;

      const dataX = positions[i * 2];
      const dataY = positions[i * 2 + 1];
      const screen = projectEuclidean(dataX, dataY, view, width, height);

      ctx.beginPath();
      ctx.arc(screen.x, screen.y, radius + 1, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Third pass: hovered point (topmost)
    if (this.hoveredIndex >= 0 && this.hoveredIndex < n && this.isPointVisibleByCategory(this.hoveredIndex)) {
      const i = this.hoveredIndex;
      const dataX = positions[i * 2];
      const dataY = positions[i * 2 + 1];
      const screen = projectEuclidean(dataX, dataY, view, width, height);

      // White ring
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

  pan(deltaX: number, deltaY: number, _modifiers: Modifiers): void {
    this.view = panEuclidean(this.view, deltaX, deltaY, this.width, this.height);
  }

  zoom(anchorX: number, anchorY: number, delta: number, _modifiers: Modifiers): void {
    this.view = zoomEuclidean(this.view, anchorX, anchorY, delta, this.width, this.height);
  }

  hitTest(screenX: number, screenY: number): HitResult | null {
    if (!this.dataset) return null;

    const { positions, n } = this.dataset;
    let bestIndex = -1;
    let bestDistSq = Infinity;
    const maxDistSq = (this.pointRadius + 5) ** 2; // Hit radius with padding

    // Brute force: check all points
    for (let i = 0; i < n; i++) {
      if (!this.isCategoryVisible(this.dataset.labels[i])) continue;
      const dataX = positions[i * 2];
      const dataY = positions[i * 2 + 1];
      const screen = projectEuclidean(dataX, dataY, this.view, this.width, this.height);

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
    const screen = projectEuclidean(dataX, dataY, this.view, this.width, this.height);

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

    // Transform polyline from screen to data space
    const dataPolyline = new Float32Array(polyline.length);
    for (let i = 0; i < polyline.length / 2; i++) {
      const screenX = polyline[i * 2];
      const screenY = polyline[i * 2 + 1];
      const data = unprojectEuclidean(screenX, screenY, this.view, this.width, this.height);
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
      throw new Error('EuclideanReference.countSelection expects indices-based selections');
    }
    let visibleCount = 0;
    for (const i of result.indices) {
      if (this.isPointVisibleByCategory(i)) visibleCount++;
    }
    return visibleCount;
  }

  projectToScreen(dataX: number, dataY: number): { x: number; y: number } {
    return projectEuclidean(dataX, dataY, this.view, this.width, this.height);
  }

  unprojectFromScreen(screenX: number, screenY: number): { x: number; y: number } {
    return unprojectEuclidean(screenX, screenY, this.view, this.width, this.height);
  }
}
