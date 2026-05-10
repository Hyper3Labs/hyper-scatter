import type { HitResult, Modifiers, Renderer, SelectionResult } from '../core/types.js';
import { simplifyPolygonData } from '../core/lasso_simplify.js';

export interface InteractionController {
  /** Detach all event listeners / observers. */
  destroy(): void;

  /** Force a size re-measure + renderer.resize() on next frame. */
  requestResize(): void;
}

export interface InteractionControllerOptions {
  /** If true, uses ResizeObserver (if available) to keep renderer size in sync. Default: true. */
  observeResize?: boolean;

  /** Controls how wheel deltas map to renderer.zoom(). Default matches the demo: -deltaY/100. */
  wheelZoomScale?: number;

  /** Trigger momentary lasso selection. Default: Shift + (Meta or Ctrl) (Embedding Atlas style). */
  lassoPredicate?: (e: PointerEvent) => boolean;

  /** Minimum screen-space sampling distance while lassoing (CSS px). Default: 2. */
  lassoMinSampleDistPx?: number;

  /** Vertex budget while dragging (for onLassoUpdate). Default: 24. */
  lassoMaxVertsInteraction?: number;

  /** Vertex budget on completion (for onLassoComplete). Default: 48. */
  lassoMaxVertsFinal?: number;

  /** Optional hook: hover updates (after hitTest). */
  onHover?: (hit: HitResult | null) => void;

  /** Optional hook: lasso polygon updates during drag. */
  onLassoUpdate?: (dataPolygon: Float32Array, screenPolygon: Float32Array) => void;

  /** Optional hook: lasso completion. */
  onLassoComplete?: (result: SelectionResult, dataPolygon: Float32Array, screenPolygon: Float32Array) => void;
}

function modifiersFromEvent(e: { shiftKey: boolean; ctrlKey: boolean; altKey: boolean; metaKey: boolean }): Modifiers {
  return {
    shift: e.shiftKey,
    ctrl: e.ctrlKey,
    alt: e.altKey,
    meta: e.metaKey,
  };
}

function clampSizePx(v: number): number {
  if (!Number.isFinite(v)) return 1;
  return Math.max(1, Math.floor(v));
}

export function createInteractionController(
  canvas: HTMLCanvasElement,
  renderer: Renderer,
  opts: InteractionControllerOptions = {}
): InteractionController {
  const wheelZoomScale = Number.isFinite(opts.wheelZoomScale) ? (opts.wheelZoomScale as number) : 1 / 100;

  const lassoPredicate = opts.lassoPredicate ?? ((e: PointerEvent) => e.shiftKey && (e.metaKey || e.ctrlKey));
  const lassoMinSampleDistPx = Number.isFinite(opts.lassoMinSampleDistPx) ? (opts.lassoMinSampleDistPx as number) : 2;
  const lassoMaxVertsInteraction = Number.isFinite(opts.lassoMaxVertsInteraction) ? (opts.lassoMaxVertsInteraction as number) : 24;
  const lassoMaxVertsFinal = Number.isFinite(opts.lassoMaxVertsFinal) ? (opts.lassoMaxVertsFinal as number) : 48;

  const observeResize = opts.observeResize ?? true;

  let raf = 0;

  let lastWidth = 0;
  let lastHeight = 0;
  let resizeDirty = true;

  let mode: 'idle' | 'pan' | 'lasso' = 'idle';
  let activePointerId: number | null = null;

  // Pan state (accumulate in CSS px; apply once per frame)
  let lastX = 0;
  let lastY = 0;
  let pendingPanDx = 0;
  let pendingPanDy = 0;
  let pendingPanMods: Modifiers = { shift: false, ctrl: false, alt: false, meta: false };

  // Wheel zoom (accumulate delta; apply once per frame)
  let pendingZoom = 0;
  let pendingZoomX = 0;
  let pendingZoomY = 0;
  let pendingZoomMods: Modifiers = { shift: false, ctrl: false, alt: false, meta: false };
  let zoomDirty = false;

  // Hover state (throttle to rAF)
  let hoverDirty = false;
  let pendingHoverX = 0;
  let pendingHoverY = 0;
  let lastHoverIndex = -2; // sentinel to ensure first update passes

  // Lasso state (sample in DATA space; project back to screen when needed)
  let lassoRawData: number[] = [];
  let lassoActiveDataPolygon: Float32Array | null = null;
  let lassoLastScreenX = 0;
  let lassoLastScreenY = 0;
  let lassoDirty = false;

  const rectToLocal = (clientX: number, clientY: number): { x: number; y: number } => {
    const rect = canvas.getBoundingClientRect();
    return { x: clientX - rect.left, y: clientY - rect.top };
  };

  const projectDataPolygonToScreen = (dataPoly: Float32Array): Float32Array => {
    const out = new Float32Array(dataPoly.length);
    for (let i = 0; i < dataPoly.length / 2; i++) {
      const dx = dataPoly[i * 2];
      const dy = dataPoly[i * 2 + 1];
      const s = renderer.projectToScreen(dx, dy);
      out[i * 2] = s.x;
      out[i * 2 + 1] = s.y;
    }
    return out;
  };

  const ensureSize = (): boolean => {
    const rect = canvas.getBoundingClientRect();
    const w = clampSizePx(rect.width);
    const h = clampSizePx(rect.height);
    if (w !== lastWidth || h !== lastHeight) {
      lastWidth = w;
      lastHeight = h;
      renderer.resize(w, h);
      return true;
    }
    return false;
  };

  const requestFrame = (): void => {
    if (raf) return;
    raf = requestAnimationFrame(() => {
      raf = 0;

      let didChange = false;

      if (resizeDirty) {
        resizeDirty = false;
        didChange = ensureSize() || didChange;
      }

      if (pendingPanDx !== 0 || pendingPanDy !== 0) {
        renderer.pan(pendingPanDx, pendingPanDy, pendingPanMods);
        pendingPanDx = 0;
        pendingPanDy = 0;
        didChange = true;
      }

      if (zoomDirty && pendingZoom !== 0) {
        renderer.zoom(pendingZoomX, pendingZoomY, pendingZoom, pendingZoomMods);
        pendingZoom = 0;
        zoomDirty = false;
        didChange = true;
      }

      if (mode === 'idle' && hoverDirty) {
        hoverDirty = false;
        const hit = renderer.hitTest(pendingHoverX, pendingHoverY);
        const idx = hit ? hit.index : -1;

        // Only update renderer + re-render if hover actually changed.
        if (idx !== lastHoverIndex) {
          lastHoverIndex = idx;
          renderer.setHovered(idx);
          didChange = true;
        }

        opts.onHover?.(hit);
      }

      if (mode === 'lasso' && lassoDirty && opts.onLassoUpdate && lassoRawData.length >= 6) {
        lassoDirty = false;
        const dataPoly = simplifyPolygonData(lassoRawData, lassoMaxVertsInteraction);
        lassoActiveDataPolygon = dataPoly;
        const screenPoly = projectDataPolygonToScreen(dataPoly);
        opts.onLassoUpdate(dataPoly, screenPoly);
      }

      if (didChange) {
        renderer.render();
      }
    });
  };

  const handlePointerDown = (e: PointerEvent): void => {
    if (e.button !== 0) return;

    const { x, y } = rectToLocal(e.clientX, e.clientY);

    activePointerId = e.pointerId;
    lastX = x;
    lastY = y;

    // Clear hover while interacting.
    if (lastHoverIndex !== -1) {
      lastHoverIndex = -1;
      renderer.setHovered(-1);
    }

    const wantsLasso = lassoPredicate(e);
    if (wantsLasso) {
      mode = 'lasso';
      lassoRawData = [];
      lassoActiveDataPolygon = null;
      lassoLastScreenX = x;
      lassoLastScreenY = y;

      const d0 = renderer.unprojectFromScreen(x, y);
      lassoRawData.push(d0.x, d0.y);
      lassoDirty = true;
      requestFrame();
    } else {
      mode = 'pan';
      pendingPanDx = 0;
      pendingPanDy = 0;
      pendingPanMods = modifiersFromEvent(e);

      renderer.startPan(x, y);

      requestFrame();
    }

    canvas.setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: PointerEvent): void => {
    if (activePointerId !== e.pointerId) {
      // Still allow hover updates from other pointers.
      if (mode === 'idle') {
        const { x, y } = rectToLocal(e.clientX, e.clientY);
        pendingHoverX = x;
        pendingHoverY = y;
        hoverDirty = true;
        requestFrame();
      }
      return;
    }

    const { x, y } = rectToLocal(e.clientX, e.clientY);

    if (mode === 'pan') {
      pendingPanDx += x - lastX;
      pendingPanDy += y - lastY;
      pendingPanMods = modifiersFromEvent(e);
      lastX = x;
      lastY = y;
      requestFrame();
      return;
    }

    if (mode === 'lasso') {
      const dx = x - lassoLastScreenX;
      const dy = y - lassoLastScreenY;
      const minD2 = lassoMinSampleDistPx * lassoMinSampleDistPx;
      if (dx * dx + dy * dy >= minD2) {
        const d = renderer.unprojectFromScreen(x, y);
        lassoRawData.push(d.x, d.y);
        lassoDirty = true;
        lassoLastScreenX = x;
        lassoLastScreenY = y;
        requestFrame();
      }
      return;
    }

    // idle
    pendingHoverX = x;
    pendingHoverY = y;
    hoverDirty = true;
    requestFrame();
  };

  const handlePointerUp = (e: PointerEvent): void => {
    if (activePointerId !== e.pointerId) return;

    // Flush any remaining pan before ending interaction.
    if (mode === 'pan' && (pendingPanDx !== 0 || pendingPanDy !== 0)) {
      renderer.pan(pendingPanDx, pendingPanDy, pendingPanMods);
      pendingPanDx = 0;
      pendingPanDy = 0;
      renderer.render();
    }

    if (mode === 'lasso' && lassoRawData.length >= 6) {
      const dataPoly = lassoActiveDataPolygon ?? simplifyPolygonData(lassoRawData, lassoMaxVertsFinal);
      const screenPoly = projectDataPolygonToScreen(dataPoly);
      const result = renderer.lassoSelect(screenPoly);
      opts.onLassoComplete?.(result, dataPoly, screenPoly);
    }

    // End interaction policy (avoid LOD pop if implementation supports it).
    if ('endInteraction' in (renderer as any)) {
      (renderer as any).endInteraction();
    }

    mode = 'idle';
    activePointerId = null;
    lassoRawData = [];
    lassoActiveDataPolygon = null;
    lassoDirty = false;
    hoverDirty = false;

    try {
      canvas.releasePointerCapture(e.pointerId);
    } catch {
      // ignore
    }
  };

  const handleWheel = (e: WheelEvent): void => {
    e.preventDefault();

    const { x, y } = rectToLocal(e.clientX, e.clientY);

    // Normalize wheel delta (matches demo)
    const delta = -e.deltaY * wheelZoomScale;

    pendingZoom += delta;
    pendingZoomX = x;
    pendingZoomY = y;
    pendingZoomMods = modifiersFromEvent(e);
    zoomDirty = true;

    requestFrame();
  };

  // --- Attach listeners ---

  canvas.addEventListener('pointerdown', handlePointerDown);
  canvas.addEventListener('pointermove', handlePointerMove);
  canvas.addEventListener('pointerup', handlePointerUp);
  canvas.addEventListener('pointercancel', handlePointerUp);
  canvas.addEventListener('wheel', handleWheel, { passive: false });

  // Optional resize observation
  let ro: ResizeObserver | null = null;
  if (observeResize && typeof ResizeObserver !== 'undefined') {
    ro = new ResizeObserver(() => {
      resizeDirty = true;
      requestFrame();
    });
    ro.observe(canvas);
  }

  // Initial size sync
  resizeDirty = true;
  requestFrame();

  return {
    destroy(): void {
      if (raf) {
        cancelAnimationFrame(raf);
        raf = 0;
      }

      canvas.removeEventListener('pointerdown', handlePointerDown);
      canvas.removeEventListener('pointermove', handlePointerMove);
      canvas.removeEventListener('pointerup', handlePointerUp);
      canvas.removeEventListener('pointercancel', handlePointerUp);
      canvas.removeEventListener('wheel', handleWheel);

      ro?.disconnect();
      ro = null;
    },

    requestResize(): void {
      resizeDirty = true;
      requestFrame();
    },
  };
}
