/**
 * WebGL2 candidate renderers.
 *
 * Key idea:
 * - Keep all math (view state, project/unproject, pan/zoom) identical to reference
 *   by delegating to src/core/math/*.
 * - Move rendering to GPU (WebGL2 point sprites) for high throughput.
 * - Speed up hit-testing and lasso selection via spatial indexes.
 *
 * ---------------------------------------------------------------------------
 * Adaptive Quality / Performance Policy (budget-based)
 * ---------------------------------------------------------------------------
 * This renderer intentionally adapts quality to keep interaction smooth on
 * typical developer hardware.
 *
 * Instead of relying on point-count-only thresholds, we choose settings from
 * simple *work budgets*:
 *
 * 1) Fragment budget (fill-rate proxy)
 *    estFragments ≈ drawCount * π * r^2 * dpr^2
 *    where r is point radius in CSS pixels and dpr is the offscreen points-FBO
 *    pixel ratio (NOT necessarily window.devicePixelRatio).
 *
 * 2) Points FBO pixel budget (memory/bandwidth proxy)
 *    pointsPixels = (width * height) * dpr^2
 *
 * The policy chooses an offscreen points DPR that respects BOTH budgets. When
 * fragment pressure is high, we may switch from anti-aliased circles to faster
 * squares (with hysteresis) rather than doing it at a fixed N threshold.
 *
 * NOTE: These adaptations affect *rendering only*; hit-testing and lasso
 * selection remain exact (CPU-side) and must match the reference semantics.
 *
 * IMPORTANT:
 * - The browser benchmark/accuracy harness uses a *separate* hidden canvas for
 *   the WebGL candidate (a single <canvas> cannot hold both a 2D and WebGL
 *   context at the same time).
 * - This renderer still lazily creates its WebGL2 context only when `render()`
 *   is called, because the demo/harness may re-initialize renderers and we
 *   want `init()` to remain side-effect-light.
 */

import {
  Dataset,
  Renderer,
  InitOptions,
  ViewState,
  EuclideanViewState,
  HyperbolicViewState,
  Modifiers,
  HitResult,
  SelectionResult,
  SelectionGeometry,
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

import {
  createHyperbolicView,
  projectPoincare,
  unprojectPoincare,
  panPoincare,
  zoomPoincare,
} from '../core/math/poincare.js';

import {
  UniformGridIndex,
} from './spatial_index.js';

import { pointInPolygon } from '../core/selection/point_in_polygon.js';

// ============================================================================
// Small helpers
// ============================================================================

function parseHexColor(color: string): [number, number, number, number] {
  // Accept: #rgb, #rrggbb, #rrggbbaa
  const s = color.trim();
  if (!s.startsWith('#')) return [1, 1, 1, 1];

  const hex = s.slice(1);
  if (hex.length === 3) {
    const r = parseInt(hex[0] + hex[0], 16) / 255;
    const g = parseInt(hex[1] + hex[1], 16) / 255;
    const b = parseInt(hex[2] + hex[2], 16) / 255;
    return [r, g, b, 1];
  }
  if (hex.length === 6 || hex.length === 8) {
    const r = parseInt(hex.slice(0, 2), 16) / 255;
    const g = parseInt(hex.slice(2, 4), 16) / 255;
    const b = parseInt(hex.slice(4, 6), 16) / 255;
    const a = hex.length === 8 ? parseInt(hex.slice(6, 8), 16) / 255 : 1;
    return [r, g, b, a];
  }

  return [1, 1, 1, 1];
}

function parseHexColorBytes(color: string): [number, number, number, number] {
  // Accept: #rgb, #rrggbb, #rrggbbaa
  const s = color.trim();
  if (!s.startsWith('#')) return [255, 255, 255, 255];

  const hex = s.slice(1);
  if (hex.length === 3) {
    const r = parseInt(hex[0] + hex[0], 16);
    const g = parseInt(hex[1] + hex[1], 16);
    const b = parseInt(hex[2] + hex[2], 16);
    return [r, g, b, 255];
  }
  if (hex.length === 6 || hex.length === 8) {
    const r = parseInt(hex.slice(0, 2), 16);
    const g = parseInt(hex.slice(2, 4), 16);
    const b = parseInt(hex.slice(4, 6), 16);
    const a = hex.length === 8 ? parseInt(hex.slice(6, 8), 16) : 255;
    return [r, g, b, a];
  }

  return [255, 255, 255, 255];
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error('Failed to create shader');
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader) ?? 'unknown';
    gl.deleteShader(shader);
    throw new Error(`Shader compile failed: ${info}`);
  }
  return shader;
}

function linkProgram(gl: WebGL2RenderingContext, vsSource: string, fsSource: string): WebGLProgram {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSource);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSource);
  const program = gl.createProgram();
  if (!program) throw new Error('Failed to create program');

  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);

  gl.deleteShader(vs);
  gl.deleteShader(fs);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program) ?? 'unknown';
    gl.deleteProgram(program);
    throw new Error(`Program link failed: ${info}`);
  }

  return program;
}

function setCanvasSize(canvas: HTMLCanvasElement, width: number, height: number, dpr: number): void {
  canvas.width = Math.max(1, Math.floor(width * dpr));
  canvas.height = Math.max(1, Math.floor(height * dpr));
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
}

// Note: Palette upload is handled by WebGLRendererBase via a small palette
// texture (supports arbitrary label counts) and cached upload buffers.

// ============================================================================
// Shader sources
// ============================================================================

const FS_POINTS = `#version 300 es
precision highp float;
precision highp int;

flat in uint v_label;

// Present in the vertex stage too; redeclare here so we can compute AA width.
uniform float u_dpr;
uniform float u_pointRadiusCss;

uniform sampler2D u_paletteTex;
uniform int u_paletteSize;
uniform int u_paletteWidth;

out vec4 outColor;

void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  // Anti-aliased circle: avoid harsh discard edges that can look like
  // "weird polygons" at small sizes or without MSAA.
  float r = length(p);
  // Ensure at least ~1px transition (in point-local coordinates) so small
  // points remain visually circular.
  float radiusPx = max(u_pointRadiusCss * u_dpr, 1.0);
  // Slightly wider than 1px helps circles stay round-looking when zoomed out
  // (where points are perceptually tiny and aliasing is more obvious).
  float aa = max(fwidth(r), 1.5 / radiusPx);
  float alpha = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, r);
  if (alpha <= 0.0) discard;

  int size = max(u_paletteSize, 1);
  int w = max(u_paletteWidth, 1);
  int idx = int(v_label) % size;
  int x = idx % w;
  int y = idx / w;
  vec4 c = texelFetch(u_paletteTex, ivec2(x, y), 0);
  outColor = vec4(c.rgb, c.a * alpha);
}
`;

// Performance mode: square points (no discard).
// This is often faster on some GPUs for very large point counts.
const FS_POINTS_SQUARE = `#version 300 es
precision highp float;
precision highp int;

flat in uint v_label;

uniform sampler2D u_paletteTex;
uniform int u_paletteSize;
uniform int u_paletteWidth;

out vec4 outColor;

void main() {
  int size = max(u_paletteSize, 1);
  int w = max(u_paletteWidth, 1);
  int idx = int(v_label) % size;
  int x = idx % w;
  int y = idx / w;
  outColor = texelFetch(u_paletteTex, ivec2(x, y), 0);
}
`;

const FS_SOLID = `#version 300 es
precision highp float;
precision highp int;

uniform float u_dpr;
uniform float u_pointRadiusCss;

uniform vec4 u_color;
uniform float u_pointSizePx;
uniform float u_ringThicknessPx;
uniform int u_ringMode; // 0 = solid, 1 = ring

out vec4 outColor;

void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float r = length(p);
  float radiusPx = max(u_pointRadiusCss * u_dpr, 1.0);
  float aa = max(fwidth(r), 1.5 / radiusPx);
  float outer = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, r);
  if (outer <= 0.0) discard;

  float alpha = outer;

  if (u_ringMode == 1) {
    float radiusPx = u_pointSizePx * 0.5;
    float t = clamp(u_ringThicknessPx / max(radiusPx, 1e-6), 0.0, 1.0);
    float inner = 1.0 - t;
    // Keep only the outer ring with an anti-aliased inner boundary.
    float innerMask = smoothstep(inner - aa, inner + aa, r);
    alpha *= innerMask;
    if (alpha <= 0.0) discard;
  }

  outColor = vec4(u_color.rgb, u_color.a * alpha);
}
`;

// Fullscreen triangle (no vertex attributes)
const VS_FULLSCREEN = `#version 300 es
precision highp float;

out vec2 v_uv;

void main() {
  // Fullscreen triangle
  // (-1,-1), (3,-1), (-1,3)
  if (gl_VertexID == 0) {
    gl_Position = vec4(-1.0, -1.0, 0.0, 1.0);
    v_uv = vec2(0.0, 0.0);
  } else if (gl_VertexID == 1) {
    gl_Position = vec4(3.0, -1.0, 0.0, 1.0);
    v_uv = vec2(2.0, 0.0);
  } else {
    gl_Position = vec4(-1.0, 3.0, 0.0, 1.0);
    v_uv = vec2(0.0, 2.0);
  }
}
`;

// Composite pass: draw points texture over the background using alpha.
const FS_COMPOSITE = `#version 300 es
precision highp float;

in vec2 v_uv;

uniform sampler2D u_tex;

out vec4 outColor;

void main() {
  vec2 uv = clamp(v_uv, 0.0, 1.0);
  outColor = texture(u_tex, uv);
}
`;

// Poincaré disk background + border (matches reference styling closely)
// Drawn as a single fullscreen pass. Outside the disk it discards, leaving the
// cleared background color.
const FS_POINCARE_DISK = `#version 300 es
precision highp float;
precision highp int;

uniform vec2 u_cssSize;
uniform float u_dpr;
uniform float u_displayZoom;

uniform vec4 u_diskFillColor;
uniform vec4 u_diskBorderColor;
uniform vec4 u_gridColor;
uniform float u_diskBorderWidthPx;
uniform float u_gridWidthPx;

out vec4 outColor;

void main() {
  // Convert framebuffer pixels to CSS pixels
  vec2 fragCss = gl_FragCoord.xy / max(u_dpr, 1.0);
  vec2 center = u_cssSize * 0.5;

  float diskRadius = min(u_cssSize.x, u_cssSize.y) * 0.45 * u_displayZoom;
  vec2 p = fragCss - center;
  float dist = length(p);

  // Reference-like styling
  vec3 diskFill = u_diskFillColor.rgb;
  vec3 diskBorder = u_diskBorderColor.rgb;

  float borderWidth = max(u_diskBorderWidthPx, 0.0);
  float halfW = 0.5 * borderWidth;

  // Anti-aliasing width (CSS px). Keep at least 1px for crisp edges.
  float aa = max(1.0, fwidth(dist));

  // Discard outside disk+border region so the clear color remains intact.
  if (dist > diskRadius + halfW + aa) discard;

  // Outer fade for anti-aliased boundary
  float outerAlpha = 1.0 - smoothstep(diskRadius + halfW - aa, diskRadius + halfW + aa, dist);

  // Border mask
  float borderInner = smoothstep(diskRadius - halfW - aa, diskRadius - halfW + aa, dist);
  float borderOuter = 1.0 - smoothstep(diskRadius + halfW - aa, diskRadius + halfW + aa, dist);
  float borderMask = clamp(borderInner * borderOuter, 0.0, 1.0);

  vec3 col = mix(diskFill, diskBorder, borderMask);

  // ------------------------------------------------------------------------
  // Reference-like hyperbolic grid overlay
  // ------------------------------------------------------------------------
  // Matches HyperbolicReference.drawHyperbolicGrid():
  // - 8 radial lines (geodesics through origin)
  // - 5 concentric circles
  vec3 gridCol = u_gridColor.rgb;
  float gridWidth = max(u_gridWidthPx, 0.0);
  float halfGrid = 0.5 * gridWidth;

  // AA width for thin lines in CSS pixel space.
  float aaLine = max(1.0, fwidth(dist));

  float gridMask = 0.0;

  // Concentric circles (5)
  for (int i = 1; i <= 5; i++) {
    float r = (float(i) / 6.0) * diskRadius;
    float d = abs(dist - r);
    float m = 1.0 - smoothstep(halfGrid - aaLine, halfGrid + aaLine, d);
    gridMask = max(gridMask, m);
  }

  // Radial lines (8), angle = (i/8)*pi
  // Distance to line through origin with direction (cos a, sin a): |cross(p, dir)|
  for (int i = 0; i < 8; i++) {
    float a = (float(i) / 8.0) * 3.141592653589793;
    vec2 dir = vec2(cos(a), sin(a));
    float d = abs(p.x * dir.y - p.y * dir.x);
    float m = 1.0 - smoothstep(halfGrid - aaLine, halfGrid + aaLine, d);
    gridMask = max(gridMask, m);
  }

  // Apply grid on top of disk fill/border. Use u_gridColor alpha as intensity.
  col = mix(col, gridCol, clamp(gridMask, 0.0, 1.0) * clamp(u_gridColor.a, 0.0, 1.0));
  outColor = vec4(col, outerAlpha);
}
`;

const VS_EUCLIDEAN = `#version 300 es
precision highp float;
precision highp int;

layout(location = 0) in vec2 a_pos;
layout(location = 1) in uint a_label;

uniform vec2 u_center;
uniform vec2 u_cssSize;
uniform float u_zoom;
uniform float u_dpr;
uniform float u_pointRadiusCss;

flat out uint v_label;

void main() {
  float baseScale = min(u_cssSize.x, u_cssSize.y) * 0.4 * u_zoom;
  float sx = u_cssSize.x * 0.5 + (a_pos.x - u_center.x) * baseScale;
  float sy = u_cssSize.y * 0.5 - (a_pos.y - u_center.y) * baseScale;

  vec2 dbufSize = u_cssSize * u_dpr;
  vec2 dbuf = vec2(sx, sy) * u_dpr;

  float cx = (dbuf.x / dbufSize.x) * 2.0 - 1.0;
  float cy = 1.0 - (dbuf.y / dbufSize.y) * 2.0;

  gl_Position = vec4(cx, cy, 0.0, 1.0);
  gl_PointSize = (u_pointRadiusCss * 2.0) * u_dpr;
  v_label = a_label;
}
`;

const VS_POINCARE = `#version 300 es
precision highp float;
precision highp int;

layout(location = 0) in vec2 a_pos;
layout(location = 1) in uint a_label;

uniform vec2 u_cssSize;
uniform float u_dpr;
uniform float u_pointRadiusCss;

uniform vec2 u_a;            // camera translation (ax, ay)
uniform float u_displayZoom; // visual zoom

flat out uint v_label;

vec2 mobiusTransform(vec2 z, vec2 a) {
  // (z - a) / (1 - conj(a) * z)
  vec2 num = z - a;

  // denom = 1 - (ax*zx + ay*zy)  + i * (-(ax*zy - ay*zx))
  float denomX = 1.0 - (a.x * z.x + a.y * z.y);
  float denomY = -(a.x * z.y - a.y * z.x);
  float denomNormSq = denomX * denomX + denomY * denomY;
  if (denomNormSq < 1e-12) {
    // Push outside clip
    return vec2(2.0, 2.0);
  }

  // complex division
  float rx = (num.x * denomX + num.y * denomY) / denomNormSq;
  float ry = (num.y * denomX - num.x * denomY) / denomNormSq;
  return vec2(rx, ry);
}

void main() {
  vec2 w = mobiusTransform(a_pos, u_a);
  float r2 = dot(w, w);
  if (r2 >= 1.0) {
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
    gl_PointSize = 0.0;
    v_label = a_label;
    return;
  }

  float diskRadius = min(u_cssSize.x, u_cssSize.y) * 0.45 * u_displayZoom;
  float sx = u_cssSize.x * 0.5 + w.x * diskRadius;
  float sy = u_cssSize.y * 0.5 - w.y * diskRadius;

  vec2 dbufSize = u_cssSize * u_dpr;
  vec2 dbuf = vec2(sx, sy) * u_dpr;

  float cx = (dbuf.x / dbufSize.x) * 2.0 - 1.0;
  float cy = 1.0 - (dbuf.y / dbufSize.y) * 2.0;

  gl_Position = vec4(cx, cy, 0.0, 1.0);
  gl_PointSize = (u_pointRadiusCss * 2.0) * u_dpr;
  v_label = a_label;
}
`;

// ============================================================================
// Base WebGL renderer (lazy context init)
// ============================================================================

type GeometryKind = 'euclidean' | 'poincare';

abstract class WebGLRendererBase implements Renderer {
  protected canvas: HTMLCanvasElement | null = null;
  protected width = 0;
  protected height = 0;
  protected deviceDpr = 1;
  // Canvas DPR (drawing buffer for the final composite). Keep at device DPR so
  // thin background/grid lines match the reference.
  protected canvasDpr = 1;
  // Points DPR (adaptive). We render points into a low-res offscreen buffer and
  // composite it onto the full-res canvas.
  protected dpr = 1;

  protected dataset: Dataset | null = null;
  protected selection = new Set<number>();
  protected hoveredIndex = -1;

  protected pointRadiusCss = 3;
  protected colors: string[] = DEFAULT_COLORS;
  protected backgroundColor = '#0a0a0a';
  protected categoryVisibilityMask = new Uint8Array(0);
  protected hasCategoryVisibilityMask = false;
  protected categoryAlpha = 1;
  protected interactionStyle: Required<InteractionStyle> = {
    selectionColor: SELECTION_COLOR,
    hoverColor: HOVER_COLOR,
    hoverFillColor: null,
  };

  // Hyperbolic backdrop styling (Poincaré disk). Neutral grayscale defaults.
  // Override per app via InitOptions as needed.
  protected poincareDiskFillColor = '#141414';
  protected poincareDiskBorderColor = '#666666';
  protected poincareGridColor = '#66666633';
  protected poincareDiskBorderWidthPx = 2;
  protected poincareGridWidthPx = 0.5;

  // Palette (label -> RGBA). Implemented as a small 2D texture so we can
  // support arbitrary label counts (not limited to 16 uniforms).
  protected paletteSize = 0;
  protected paletteDirty = true;
  protected paletteTex: WebGLTexture | null = null;
  protected paletteTexW = 0;
  protected paletteTexH = 0;
  protected paletteBytes = new Uint8Array(0);
  protected readonly paletteTexUnit = 1;

  // Scratch arrays (avoid per-call allocations in hit testing / selection)
  protected scratchIds: number[] = [];

  // Scratch typed arrays for hover uploads (avoid per-frame allocations)
  protected hoverPosScratch = new Float32Array(2);
  protected hoverLabScratch = new Uint16Array(1);
  protected hoverIndexScratch = new Uint32Array(1);

  // Interaction-adaptive rendering (used to keep panning smooth at very large N)
  protected lastViewChangeTs = 0;

  protected markViewChanged(): void {
    // performance.now() is available in browsers; in non-DOM contexts this class
    // isn't used.
    this.lastViewChangeTs = performance.now();
  }

  /**
   * Optional UI hook: call when the user ends an interaction (mouse up / gesture end).
   *
   * The renderer uses `lastViewChangeTs` to decide whether to enable interaction
   * LOD (subsampling) for smooth panning/zooming. In the demo app we only render
   * on demand; if the final frame after mouseup is still considered "interacting",
   * we can end up showing a subsample until the next hover-triggered render,
   * which looks like a visual snap/pop.
   *
   * By resetting the interaction timer, the next render will use the stable
   * (non-interaction) policy immediately.
   */
  endInteraction(): void {
    this.lastViewChangeTs = 0;
  }

  protected markBackdropDirty(): void {
    this.backdropDirty = true;
  }

  protected uploadPoincareDiskStyleUniforms(): void {
    const gl = this.gl;
    const disk = this.poincareDisk;
    if (!gl || !disk) return;

    const fill = parseHexColor(this.poincareDiskFillColor);
    const border = parseHexColor(this.poincareDiskBorderColor);
    const grid = parseHexColor(this.poincareGridColor);

    if (disk.uDiskFillColor) gl.uniform4f(disk.uDiskFillColor, fill[0], fill[1], fill[2], fill[3]);
    if (disk.uDiskBorderColor) gl.uniform4f(disk.uDiskBorderColor, border[0], border[1], border[2], border[3]);
    if (disk.uGridColor) gl.uniform4f(disk.uGridColor, grid[0], grid[1], grid[2], grid[3]);
    if (disk.uDiskBorderWidthPx) gl.uniform1f(disk.uDiskBorderWidthPx, this.poincareDiskBorderWidthPx);
    if (disk.uGridWidthPx) gl.uniform1f(disk.uGridWidthPx, this.poincareGridWidthPx);
  }

  // Overridden by the hyperbolic renderer.
  protected getBackdropZoom(): number {
    return 1;
  }

  // CPU spatial index (data space)
  protected dataIndex: UniformGridIndex | null = null;

  // WebGL state (created lazily in render())
  protected gl: WebGL2RenderingContext | null = null;
  protected vao: WebGLVertexArrayObject | null = null;
  protected posBuffer: WebGLBuffer | null = null;
  protected labelBuffer: WebGLBuffer | null = null;

  // Overlay buffers (used when main GPU buffers are a subsample).
  protected hoverVao: WebGLVertexArrayObject | null = null;
  protected hoverPosBuffer: WebGLBuffer | null = null;
  protected hoverLabelBuffer: WebGLBuffer | null = null;

  protected selectionVao: WebGLVertexArrayObject | null = null;
  protected selectionPosBuffer: WebGLBuffer | null = null;
  protected selectionLabelBuffer: WebGLBuffer | null = null;
  protected selectionOverlayCount = 0;

  protected selectionEbo: WebGLBuffer | null = null;
  protected hoverEbo: WebGLBuffer | null = null;
  protected interactionEbo: WebGLBuffer | null = null;
  protected interactionCount = 0;

  // Practical cap for vertex work in the base pass at very large N.
  // For 20M datasets, drawing all points every frame is often vertex-bound;
  // we instead draw a deterministic subsample and keep interaction semantics
  // exact via CPU hit-testing + exact lasso.
  //
  // This does *not* affect the accuracy harness, which compares math + hit/lasso
  // results rather than pixel output.
  protected maxBaseDrawPoints = 4_000_000;

  // Above this, upload only a deterministic subsample to GPU to avoid huge
  // GPU allocations / upload stalls at 10M-20M points.
  protected maxGpuUploadPoints = 10_000_000;
  protected gpuUsesFullDataset = true;
  protected gpuPointCount = 0;

  // -------------------------------------------------------------------------
  // Policy knobs (tuned via benchmarks; intended to generalize across hardware)
  // -------------------------------------------------------------------------

  protected policy = {
    // Rough target for 60 FPS. This is a proxy budget (fragment invocations).
    // If you change point radius defaults, re-evaluate this budget.
    fragmentBudget: 100_000_000,

    // Circles (AA + discard) are noticeably more expensive per-fragment than
    // squares. Use a separate (lower) threshold for when we allow circles.
    // Above this, prefer squares even if we're still under fragmentBudget.
    circleBudget: 60_000_000,

    // Hysteresis for circle<->square switching.
    // Switch ON squares when estimated fragment load is high;
    // switch back OFF when comfortably below the threshold.
    squareOnRatio: 1.0,
    squareOffRatio: 0.75,

    // Minimum acceptable offscreen DPR for points (quality floor).
    // Keeping this too high can cause perf cliffs at huge N; too low can make
    // points overly blurry.
    minPointsDpr: 0.35,
  } as const;

  // Sticky render mode (to avoid per-frame flip-flops)
  protected renderAsSquares = false;

  // Exposed for benchmarks (read via reflection)
  public __debugPolicy: any = null;

  // Cached hyperbolic backdrop (disk + grid) rendered to an offscreen texture.
  // Rendering the backdrop shader every frame is expensive; we render it only
  // when size/DPR or displayZoom changes, then blit the cached image.
  protected backdropTex: WebGLTexture | null = null;
  protected backdropFbo: WebGLFramebuffer | null = null;
  protected backdropW = 0;
  protected backdropH = 0;
  protected backdropDpr = 1;
  protected backdropZoom = NaN;
  protected backdropDirty = true;

  // Low-res points render target (adaptive DPR)
  protected pointsTex: WebGLTexture | null = null;
  protected pointsFbo: WebGLFramebuffer | null = null;
  protected pointsW = 0;
  protected pointsH = 0;

  // Program for compositing the points texture to the main framebuffer
  protected programComposite: WebGLProgram | null = null;
  protected uCompositeTex: WebGLUniformLocation | null = null;

  protected poincareDisk: {
    program: WebGLProgram;
    uCssSize: WebGLUniformLocation | null;
    uDpr: WebGLUniformLocation | null;
    uDiskFillColor: WebGLUniformLocation | null;
    uDiskBorderColor: WebGLUniformLocation | null;
    uGridColor: WebGLUniformLocation | null;
    uDiskBorderWidthPx: WebGLUniformLocation | null;
    uGridWidthPx: WebGLUniformLocation | null;
    // u_displayZoom is set via bindViewUniformsForProgram() (Hyperbolic)
  } | null = null;

  protected pointsCircle: {
    program: WebGLProgram;
    uPaletteTex: WebGLUniformLocation | null;
    uPaletteSize: WebGLUniformLocation | null;
    uPaletteWidth: WebGLUniformLocation | null;
    uCssSize: WebGLUniformLocation | null;
    uDpr: WebGLUniformLocation | null;
    uPointRadius: WebGLUniformLocation | null;
  } | null = null;

  protected pointsSquare: {
    program: WebGLProgram;
    uPaletteTex: WebGLUniformLocation | null;
    uPaletteSize: WebGLUniformLocation | null;
    uPaletteWidth: WebGLUniformLocation | null;
    uCssSize: WebGLUniformLocation | null;
    uDpr: WebGLUniformLocation | null;
    uPointRadius: WebGLUniformLocation | null;
  } | null = null;

  protected programSolid: WebGLProgram | null = null;

  // Uniform locations (solid)
  protected uSolidColor: WebGLUniformLocation | null = null;
  protected uSolidPointSizePx: WebGLUniformLocation | null = null;
  protected uSolidRingThicknessPx: WebGLUniformLocation | null = null;
  protected uSolidRingMode: WebGLUniformLocation | null = null;

  protected uCssSizeSolid: WebGLUniformLocation | null = null;
  protected uDprSolid: WebGLUniformLocation | null = null;
  protected uPointRadiusSolid: WebGLUniformLocation | null = null;

  protected selectionDirty = true;
  protected hoverDirty = true;

  init(canvas: HTMLCanvasElement, opts: InitOptions): void {
    this.canvas = canvas;
    this.width = opts.width;
    this.height = opts.height;
    this.deviceDpr = opts.devicePixelRatio ?? window.devicePixelRatio ?? 1;
    this.canvasDpr = this.deviceDpr;
    this.dpr = this.deviceDpr;

    const hasDiskFillOverride = typeof opts.poincareDiskFillColor === 'string';
    if (opts.backgroundColor) this.backgroundColor = opts.backgroundColor;
    if (opts.pointRadius) this.pointRadiusCss = opts.pointRadius;
    if (opts.colors) this.colors = opts.colors;

    // Optional per-app styling for hyperbolic disk/grid.
    // If the app did not specify a disk fill, keep the neutral default.
    this.poincareDiskFillColor = hasDiskFillOverride
      ? opts.poincareDiskFillColor!
      : this.poincareDiskFillColor;
    if (opts.poincareDiskBorderColor) this.poincareDiskBorderColor = opts.poincareDiskBorderColor;
    if (opts.poincareGridColor) this.poincareGridColor = opts.poincareGridColor;
    if (typeof opts.poincareDiskBorderWidthPx === 'number' && Number.isFinite(opts.poincareDiskBorderWidthPx)) {
      this.poincareDiskBorderWidthPx = Math.max(0, opts.poincareDiskBorderWidthPx);
    }
    if (typeof opts.poincareGridWidthPx === 'number' && Number.isFinite(opts.poincareGridWidthPx)) {
      this.poincareGridWidthPx = Math.max(0, opts.poincareGridWidthPx);
    }

    this.paletteDirty = true;

    // IMPORTANT:
    // Do NOT touch `canvas.width/height` here.
    // The accuracy harness initializes reference and candidate on the same
    // canvas; resizing would reset the reference's 2D context.
    // We size the canvas only when we actually acquire a WebGL context.
  }

  protected chooseRenderDpr(pointCount: number): number {
    const d = this.deviceDpr;
    const cssPixels = Math.max(1, this.width) * Math.max(1, this.height);

    // Expected draw count for rendering. For very large datasets we expect to
    // draw a deterministic subsample (LOD) rather than all points.
    const expectedDrawCount = pointCount > this.maxBaseDrawPoints
      ? this.estimateSubsampleCount(pointCount)
      : pointCount;

    // Budget 1: points-FBO pixel budget (memory / bandwidth proxy).
    // These numbers were originally tuned empirically; we keep them but treat
    // them explicitly as a *budget* rather than a point-count threshold.
    const pointsFboPixelBudget =
      pointCount >= 1_000_000
        ? (cssPixels > 1_000_000 ? 200_000 : 500_000)
        : pointCount >= 500_000
          ? 1_400_000
          : pointCount >= 250_000
            ? 2_100_000
            : 8_000_000; // cap allocations for very large canvases even at small N

    const dprFromPointsPixels = Math.sqrt(pointsFboPixelBudget / cssPixels);

    // Budget 2: fragment budget (fill-rate proxy).
    // estFragments ≈ N * π * r^2 * dpr^2  =>  dpr <= sqrt(budget / (N * π * r^2))
    const r = Math.max(0.5, this.pointRadiusCss);
    const denom = Math.max(1, expectedDrawCount) * Math.PI * r * r;
    const dprFromFragments = Math.sqrt(this.policy.fragmentBudget / denom);

    // Cap for stability (avoid huge offscreen buffers) and quality.
    const cap = pointCount >= 1_000_000 ? 1.0 : pointCount >= 500_000 ? 1.25 : 1.5;
    const floor = pointCount >= 1_000_000 ? this.policy.minPointsDpr : pointCount >= 500_000 ? 0.75 : 1.0;

    const chosen = Math.min(d, cap, dprFromPointsPixels, dprFromFragments);
    return Math.max(floor, chosen);
  }

  protected estimateSubsampleCount(n: number): number {
    // Mirrors uploadDatasetToGPU() subsample logic, but used for budgeting.
    if (n < 500_000) return n;
    const target = Math.min(n, Math.max(250_000, Math.min(this.maxBaseDrawPoints, Math.floor(n * 0.25))));
    const step = Math.max(1, Math.floor(n / target));
    const count = Math.min(target, Math.ceil(n / step));
    return count;
  }

  protected estimatePointFragments(drawCount: number, pointsDpr: number): number {
    // Proxy for total fragment shader invocations spent on point sprites.
    // This is intentionally simple and stable.
    const r = Math.max(0.5, this.pointRadiusCss);
    const n = Math.max(0, drawCount);
    const dpr = Math.max(0, pointsDpr);
    return n * Math.PI * r * r * dpr * dpr;
  }

  protected updateSquarePointPolicy(estimatedFragments: number): void {
    // Switch based on the circle budget, not the overall fragment budget.
    const on = this.policy.circleBudget * this.policy.squareOnRatio;
    const off = this.policy.circleBudget * this.policy.squareOffRatio;

    // If we're rendering points at a reduced offscreen DPR, the AA circle shader
    // tends to be a poor trade (extra ALU + discard) while visual fidelity is
    // already limited by resolution. Prefer squares in that regime.
    const forceSquaresForLowDpr = this.dpr <= 0.75;
    if (forceSquaresForLowDpr) {
      this.renderAsSquares = true;
      return;
    }

    if (!this.renderAsSquares) {
      if (estimatedFragments >= on) this.renderAsSquares = true;
    } else {
      if (estimatedFragments <= off) this.renderAsSquares = false;
    }
  }

  setDataset(dataset: Dataset): void {
    this.dataset = dataset;
    // Reset selection without mutating any external object passed via setSelection().
    this.selection = new Set<number>();
    this.hoveredIndex = -1;
    this.selectionDirty = true;
    this.hoverDirty = true;

    // Potentially clamp points-FBO DPR for large datasets (performance priority).
    const nextDpr = this.chooseRenderDpr(dataset.n);
    if (nextDpr !== this.dpr) {
      this.dpr = nextDpr;
    }

    // Bounds are computed internally by UniformGridIndex if omitted.
    this.dataIndex = new UniformGridIndex(dataset.positions, undefined, 64);

    // If WebGL is already active (performance benchmarks), upload immediately.
    if (this.gl) {
      this.uploadDatasetToGPU();
    }

    // Dataset changes don't affect the backdrop, but point DPR might have changed.
    this.markBackdropDirty();
  }

  setPalette(colors: string[]): void {
    this.colors = colors;
    this.paletteDirty = true;
    if (this.gl) {
      this.uploadPaletteUniforms();
    }
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

    this.paletteDirty = true;
    this.selectionDirty = true;
    this.hoverDirty = true;

    if (this.gl) {
      this.uploadPaletteUniforms();
      this.uploadSelectionToGPU();
      this.uploadHoverToGPU();
    }
  }

  setCategoryAlpha(alpha: number): void {
    const next = Number.isFinite(alpha) ? Math.max(0, Math.min(1, alpha)) : 1;
    if (Math.abs(next - this.categoryAlpha) <= 1e-12) return;
    this.categoryAlpha = next;
    this.paletteDirty = true;
    if (this.gl) {
      this.uploadPaletteUniforms();
    }
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

  protected isCategoryVisible(category: number): boolean {
    if (!this.hasCategoryVisibilityMask) return true;
    const mask = this.categoryVisibilityMask;
    if (category < 0 || category >= mask.length) return true;
    return mask[category] !== 0;
  }

  protected isPointVisibleByCategory(index: number): boolean {
    const ds = this.dataset;
    if (!ds || index < 0 || index >= ds.n) return false;
    return this.isCategoryVisible(ds.labels[index]);
  }

  abstract setView(view: ViewState): void;
  abstract getView(): ViewState;

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;

    // Only resize the drawing buffer when we own a WebGL context.
    if (this.gl && this.canvas) {
      setCanvasSize(this.canvas, width, height, this.canvasDpr);
      this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      this.markBackdropDirty();
    }
  }

  setSelection(indices: Set<number>): void {
    // IMPORTANT: Do not eagerly clone huge selections into a JS Set.
    // For large-N lasso this can OOM. We keep the provided Set-like object.
    // For small sets, clone to keep reference semantics similar to reference impls.
    const n = indices.size;
    this.selection = n <= 200_000 ? new Set(indices) : indices;
    this.selectionDirty = true;

    if (this.gl) {
      this.uploadSelectionToGPU();
    }
  }

  getSelection(): Set<number> {
    // Returning a cloned Set is nice for encapsulation, but can OOM for huge
    // selections. For large selections, return the internal Set-like object.
    return this.selection.size <= 200_000 ? new Set(this.selection) : this.selection;
  }

  setHovered(index: number): void {
    if (index >= 0 && !this.isPointVisibleByCategory(index)) {
      this.hoveredIndex = -1;
    } else {
      this.hoveredIndex = index;
    }
    this.hoverDirty = true;

    if (this.gl) {
      this.uploadHoverToGPU();
    }
  }

  destroy(): void {
    const gl = this.gl;

    if (gl) {
      if (this.pointsCircle) gl.deleteProgram(this.pointsCircle.program);
      if (this.pointsSquare) gl.deleteProgram(this.pointsSquare.program);
      if (this.programSolid) gl.deleteProgram(this.programSolid);
      if (this.poincareDisk) gl.deleteProgram(this.poincareDisk.program);
      if (this.vao) gl.deleteVertexArray(this.vao);
      if (this.hoverVao) gl.deleteVertexArray(this.hoverVao);
      if (this.selectionVao) gl.deleteVertexArray(this.selectionVao);
      if (this.posBuffer) gl.deleteBuffer(this.posBuffer);
      if (this.labelBuffer) gl.deleteBuffer(this.labelBuffer);
      if (this.hoverPosBuffer) gl.deleteBuffer(this.hoverPosBuffer);
      if (this.hoverLabelBuffer) gl.deleteBuffer(this.hoverLabelBuffer);
      if (this.selectionPosBuffer) gl.deleteBuffer(this.selectionPosBuffer);
      if (this.selectionLabelBuffer) gl.deleteBuffer(this.selectionLabelBuffer);
      if (this.selectionEbo) gl.deleteBuffer(this.selectionEbo);
      if (this.hoverEbo) gl.deleteBuffer(this.hoverEbo);
      if (this.interactionEbo) gl.deleteBuffer(this.interactionEbo);
      if (this.backdropFbo) gl.deleteFramebuffer(this.backdropFbo);
      if (this.backdropTex) gl.deleteTexture(this.backdropTex);
      if (this.pointsFbo) gl.deleteFramebuffer(this.pointsFbo);
      if (this.pointsTex) gl.deleteTexture(this.pointsTex);
      if (this.paletteTex) gl.deleteTexture(this.paletteTex);
      if (this.programComposite) gl.deleteProgram(this.programComposite);
    }

    this.gl = null;
    this.vao = null;
    this.hoverVao = null;
    this.selectionVao = null;
    this.posBuffer = null;
    this.labelBuffer = null;
    this.hoverPosBuffer = null;
    this.hoverLabelBuffer = null;
    this.selectionPosBuffer = null;
    this.selectionLabelBuffer = null;
    this.selectionOverlayCount = 0;
    this.selectionEbo = null;
    this.hoverEbo = null;
    this.interactionEbo = null;
    this.interactionCount = 0;
    this.gpuUsesFullDataset = true;
    this.gpuPointCount = 0;
    this.backdropFbo = null;
    this.backdropTex = null;
    this.backdropW = 0;
    this.backdropH = 0;
    this.backdropDpr = 1;
    this.backdropZoom = NaN;
    this.backdropDirty = true;

    this.pointsFbo = null;
    this.pointsTex = null;
    this.pointsW = 0;
    this.pointsH = 0;

    this.programComposite = null;
    this.uCompositeTex = null;
    this.pointsCircle = null;
    this.pointsSquare = null;
    this.programSolid = null;
    this.poincareDisk = null;

    this.paletteTex = null;
    this.paletteTexW = 0;
    this.paletteTexH = 0;
    this.paletteSize = 0;
    this.paletteDirty = true;
  }

  protected uploadPaletteUniforms(): void {
    const gl = this.gl;
    if (!gl) return;

    const rawSize = this.colors.length;
    // Labels are Uint16 in the dataset, so the max addressable palette size is 65536.
    const size = Math.max(1, Math.min(rawSize, 0xffff + 1));

    const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    const texW = Math.min(maxTex, size);
    const texH = Math.ceil(size / texW);
    if (texH > maxTex) {
      throw new Error(`Palette too large for WebGL texture: size=${size}, maxTex=${maxTex}`);
    }

    // Allocate upload buffer (RGBA8)
    const capacity = texW * texH;
    if (this.paletteBytes.length !== capacity * 4) {
      this.paletteBytes = new Uint8Array(capacity * 4);
    } else {
      this.paletteBytes.fill(0);
    }

    if (rawSize === 0) {
      // Fallback: opaque white
      this.paletteBytes[0] = 255;
      this.paletteBytes[1] = 255;
      this.paletteBytes[2] = 255;
      const visible = this.isCategoryVisible(0);
      this.paletteBytes[3] = visible ? Math.round(255 * this.categoryAlpha) : 0;
    } else {
      for (let i = 0; i < size; i++) {
        const [r, g, b, a] = parseHexColorBytes(this.colors[i]);
        const visible = this.isCategoryVisible(i);
        const alpha = visible ? Math.round(a * this.categoryAlpha) : 0;
        const o = i * 4;
        this.paletteBytes[o + 0] = r;
        this.paletteBytes[o + 1] = g;
        this.paletteBytes[o + 2] = b;
        this.paletteBytes[o + 3] = alpha;
      }
    }

    if (!this.paletteTex) {
      this.paletteTex = gl.createTexture();
      if (!this.paletteTex) throw new Error('Failed to create palette texture');
      gl.activeTexture(gl.TEXTURE0 + this.paletteTexUnit);
      gl.bindTexture(gl.TEXTURE_2D, this.paletteTex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.bindTexture(gl.TEXTURE_2D, null);
      gl.activeTexture(gl.TEXTURE0);
    }

    this.paletteSize = size;
    this.paletteTexW = texW;
    this.paletteTexH = texH;

    // Upload texture
    gl.activeTexture(gl.TEXTURE0 + this.paletteTexUnit);
    gl.bindTexture(gl.TEXTURE_2D, this.paletteTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, texW, texH, 0, gl.RGBA, gl.UNSIGNED_BYTE, this.paletteBytes);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE0);

    // Upload uniforms to both palette programs; uniforms persist per-program.
    const upload = (p: typeof this.pointsCircle | typeof this.pointsSquare) => {
      if (!p) return;
      gl.useProgram(p.program);
      if (p.uPaletteTex) gl.uniform1i(p.uPaletteTex, this.paletteTexUnit);
      if (p.uPaletteSize) gl.uniform1i(p.uPaletteSize, this.paletteSize);
      if (p.uPaletteWidth) gl.uniform1i(p.uPaletteWidth, this.paletteTexW);
    };

    upload(this.pointsCircle);
    upload(this.pointsSquare);

    this.paletteDirty = false;
  }

  protected bindPaletteTexture(): void {
    const gl = this.gl;
    if (!gl || !this.paletteTex) return;
    gl.activeTexture(gl.TEXTURE0 + this.paletteTexUnit);
    gl.bindTexture(gl.TEXTURE_2D, this.paletteTex);
    // Restore predictable state for resource alloc helpers.
    gl.activeTexture(gl.TEXTURE0);
  }

  async countSelection(result: SelectionResult, opts: CountSelectionOptions = {}): Promise<number> {
    const ds = this.dataset;
    const idx = this.dataIndex;
    if (!ds || !idx) return 0;

    if (result.indices) {
      let visibleCount = 0;
      for (const i of result.indices) {
        if (this.isPointVisibleByCategory(i)) visibleCount++;
      }
      return visibleCount;
    }
    if (result.kind !== 'geometry' || !result.geometry) return 0;

    const polygon = result.geometry.coords;
    const nPoly = polygon.length / 2;
    if (nPoly < 3) return 0;

    // Bounds are usually attached by the candidate lassoSelect(). Compute as
    // a fallback to keep this method robust.
    let bounds = result.geometry.bounds;
    if (!bounds) {
      let xMin = Infinity;
      let yMin = Infinity;
      let xMax = -Infinity;
      let yMax = -Infinity;
      for (let i = 0; i < polygon.length; i += 2) {
        const x = polygon[i];
        const y = polygon[i + 1];
        if (x < xMin) xMin = x;
        if (x > xMax) xMax = x;
        if (y < yMin) yMin = y;
        if (y > yMax) yMax = y;
      }
      bounds = { xMin, yMin, xMax, yMax };
    }

    const shouldCancel = opts.shouldCancel;
    const onProgress = opts.onProgress;
    const yieldEveryMs = (typeof opts.yieldEveryMs === 'number' && Number.isFinite(opts.yieldEveryMs))
      ? Math.max(0, opts.yieldEveryMs)
      : 8;

    const eps = 1e-12;
    const minX = bounds.xMin - eps;
    const minY = bounds.yMin - eps;
    const maxX = bounds.xMax + eps;
    const maxY = bounds.yMax + eps;

    const clampInt = (v: number, lo: number, hi: number): number => {
      if (v < lo) return lo;
      if (v > hi) return hi;
      return v | 0;
    };

    const cx0 = clampInt(Math.floor((minX - idx.bounds.minX) / idx.cellSizeX), 0, idx.cellsX - 1);
    const cy0 = clampInt(Math.floor((minY - idx.bounds.minY) / idx.cellSizeY), 0, idx.cellsY - 1);
    const cx1 = clampInt(Math.floor((maxX - idx.bounds.minX) / idx.cellSizeX), 0, idx.cellsX - 1);
    const cy1 = clampInt(Math.floor((maxY - idx.bounds.minY) / idx.cellSizeY), 0, idx.cellsY - 1);

    const positions = ds.positions;
    const ids = idx.ids;
    const offsets = idx.offsets;

    let selected = 0;
    let processed = 0;

    const CHECK_STRIDE = 16_384;
    let nextCheck = CHECK_STRIDE;
    let lastYieldTs = yieldEveryMs > 0 ? performance.now() : 0;

    for (let cy = cy0; cy <= cy1; cy++) {
      const rowBase = cy * idx.cellsX;
      for (let cx = cx0; cx <= cx1; cx++) {
        const cell = rowBase + cx;
        const start = offsets[cell];
        const end = offsets[cell + 1];
        for (let k = start; k < end; k++) {
          const i = ids[k];
          if (!this.isCategoryVisible(ds.labels[i])) continue;
          const x = positions[i * 2];
          const y = positions[i * 2 + 1];

          // Tight AABB prefilter (cells overlap bounds).
          if (x < bounds.xMin || x > bounds.xMax || y < bounds.yMin || y > bounds.yMax) continue;

          if (pointInPolygon(x, y, polygon)) selected++;
          processed++;

          if (yieldEveryMs > 0 && processed >= nextCheck) {
            nextCheck = processed + CHECK_STRIDE;

            if (shouldCancel?.()) return selected;

            const now = performance.now();
            if (now - lastYieldTs >= yieldEveryMs) {
              onProgress?.(selected, processed);
              await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
              lastYieldTs = performance.now();
            }
          }
        }
      }
    }

    onProgress?.(selected, processed);
    return selected;
  }

  // ==== Required interaction methods (abstracts for math) ==== 

  abstract pan(deltaX: number, deltaY: number, modifiers: Modifiers): void;
  abstract zoom(anchorX: number, anchorY: number, delta: number, modifiers: Modifiers): void;
  abstract hitTest(screenX: number, screenY: number): HitResult | null;
  abstract lassoSelect(polyline: Float32Array): SelectionResult;
  abstract projectToScreen(dataX: number, dataY: number): { x: number; y: number };
  abstract unprojectFromScreen(screenX: number, screenY: number): { x: number; y: number };

  // ==== GPU ==== 

  protected abstract geometryKind(): GeometryKind;

  protected ensureGL(): void {
    if (this.gl) return;
    if (!this.canvas) throw new Error('Renderer not initialized');

    // Now that we actually intend to render with WebGL, we can safely size the
    // canvas drawing buffer.
    // Keep final canvas at device DPR for reference-like background quality.
    setCanvasSize(this.canvas, this.width, this.height, this.canvasDpr);

    const gl = this.canvas.getContext('webgl2', {
      // MSAA improves point sprite edge quality (less "squarish" at small sizes).
      // This is especially noticeable for Euclidean where points are drawn all over
      // the canvas and your eye picks up aliasing more easily.
      // But it can cost noticeable fill-rate at very large N, so keep it off
      // and rely on shader-based AA instead.
      antialias: false,
      // Keep opaque for performance; we render the hyperbolic backdrop in WebGL.
      alpha: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: false,
      premultipliedAlpha: false,
      desynchronized: true,
    } as WebGLContextAttributes);

    if (!gl) {
      throw new Error('Failed to get WebGL2 context (is the canvas already using 2D context?)');
    }

    this.gl = gl;
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    // Enable blending so anti-aliased point edges can blend with the background.
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    const [br, bg, bb, ba] = parseHexColor(this.backgroundColor);
    gl.clearColor(br, bg, bb, ba);

    gl.viewport(0, 0, this.canvas.width, this.canvas.height);

    this.createProgramsAndBuffers();

    // Upload palette uniforms once after program creation.
    this.uploadPaletteUniforms();

    if (this.dataset) {
      this.uploadDatasetToGPU();
    }
    this.uploadSelectionToGPU();
    this.uploadHoverToGPU();

    // Backdrop depends on size/zoom.
    this.markBackdropDirty();
  }

  protected ensurePointsResources(): void {
    if (!this.gl || !this.canvas) return;
    const gl = this.gl;

    // Points buffer size at adaptive DPR.
    let w = Math.max(1, Math.floor(this.width * this.dpr));
    let h = Math.max(1, Math.floor(this.height * this.dpr));

    const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    if (w > maxTex || h > maxTex) {
      const s = Math.min(1, maxTex / w, maxTex / h);
      w = Math.max(1, Math.floor(w * s));
      h = Math.max(1, Math.floor(h * s));
    }

    if (!this.pointsTex) {
      this.pointsTex = gl.createTexture();
      if (!this.pointsTex) throw new Error('Failed to create points texture');
      gl.bindTexture(gl.TEXTURE_2D, this.pointsTex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.bindTexture(gl.TEXTURE_2D, null);
    }

    if (!this.pointsFbo) {
      this.pointsFbo = gl.createFramebuffer();
      if (!this.pointsFbo) throw new Error('Failed to create points framebuffer');
    }

    if (w !== this.pointsW || h !== this.pointsH) {
      this.pointsW = w;
      this.pointsH = h;

      gl.bindTexture(gl.TEXTURE_2D, this.pointsTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.bindTexture(gl.TEXTURE_2D, null);

      gl.bindFramebuffer(gl.FRAMEBUFFER, this.pointsFbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.pointsTex, 0);
      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      if (status !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error(`Points framebuffer incomplete: ${status}`);
      }
    }
  }

  protected ensureBackdropResources(): void {
    if (!this.gl || !this.canvas) return;
    if (this.geometryKind() !== 'poincare') return;
    if (!this.poincareDisk || !this.vao) return;

    const gl = this.gl;

    // Render backdrop at full canvas DPR so thin grid lines survive.
    const desiredDpr = Math.max(1, this.canvasDpr);
    let w = Math.max(1, Math.floor(this.width * desiredDpr));
    let h = Math.max(1, Math.floor(this.height * desiredDpr));

    const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    if (w > maxTex || h > maxTex) {
      const s = Math.min(1, maxTex / w, maxTex / h);
      w = Math.max(1, Math.floor(w * s));
      h = Math.max(1, Math.floor(h * s));
      this.backdropDpr = desiredDpr * s;
    } else {
      this.backdropDpr = desiredDpr;
    }

    if (!this.backdropTex) {
      this.backdropTex = gl.createTexture();
      if (!this.backdropTex) throw new Error('Failed to create backdrop texture');
      gl.bindTexture(gl.TEXTURE_2D, this.backdropTex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.bindTexture(gl.TEXTURE_2D, null);
    }

    if (!this.backdropFbo) {
      this.backdropFbo = gl.createFramebuffer();
      if (!this.backdropFbo) throw new Error('Failed to create backdrop framebuffer');
    }

    if (w !== this.backdropW || h !== this.backdropH) {
      this.backdropW = w;
      this.backdropH = h;

      gl.bindTexture(gl.TEXTURE_2D, this.backdropTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.bindTexture(gl.TEXTURE_2D, null);

      gl.bindFramebuffer(gl.FRAMEBUFFER, this.backdropFbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.backdropTex, 0);
      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      if (status !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error(`Backdrop framebuffer incomplete: ${status}`);
      }

      this.markBackdropDirty();
    }
  }

  protected renderBackdropIfNeeded(): void {
    if (!this.gl || !this.canvas) return;
    if (this.geometryKind() !== 'poincare') return;
    if (!this.poincareDisk || !this.vao) return;

    this.ensureBackdropResources();
    if (!this.backdropFbo) return;

    const zoom = this.getBackdropZoom();
    const zoomSame = Number.isFinite(this.backdropZoom) && Math.abs(this.backdropZoom - zoom) <= 1e-12;
    if (!this.backdropDirty && zoomSame) return;

    const gl = this.gl;

    // Render disk+grid into texture.
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.backdropFbo);
    gl.viewport(0, 0, this.backdropW, this.backdropH);

    const [br, bg, bb, ba] = parseHexColor(this.backgroundColor);
    gl.clearColor(br, bg, bb, ba);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.poincareDisk.program);
    this.bindViewUniformsForProgram(this.poincareDisk.program);
    this.uploadPoincareDiskStyleUniforms();
    if (this.poincareDisk.uCssSize) gl.uniform2f(this.poincareDisk.uCssSize, this.width, this.height);
    if (this.poincareDisk.uDpr) gl.uniform1f(this.poincareDisk.uDpr, this.backdropDpr);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Restore default framebuffer + viewport.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);

    this.backdropZoom = zoom;
    this.backdropDirty = false;
  }

  protected createProgramsAndBuffers(): void {
    const gl = this.gl!;

    const vs = this.geometryKind() === 'euclidean' ? VS_EUCLIDEAN : VS_POINCARE;

    // Programs
    const circleProgram = linkProgram(gl, vs, FS_POINTS);
    const squareProgram = linkProgram(gl, vs, FS_POINTS_SQUARE);
    this.programSolid = linkProgram(gl, vs, FS_SOLID);

    // Composite program (fullscreen textured triangle)
    this.programComposite = linkProgram(gl, VS_FULLSCREEN, FS_COMPOSITE);
    gl.useProgram(this.programComposite);
    this.uCompositeTex = gl.getUniformLocation(this.programComposite, 'u_tex');

    // Poincaré disk background (only for hyperbolic)
    if (this.geometryKind() === 'poincare') {
      const diskProgram = linkProgram(gl, VS_FULLSCREEN, FS_POINCARE_DISK);
      this.poincareDisk = {
        program: diskProgram,
        uCssSize: gl.getUniformLocation(diskProgram, 'u_cssSize'),
        uDpr: gl.getUniformLocation(diskProgram, 'u_dpr'),
        uDiskFillColor: gl.getUniformLocation(diskProgram, 'u_diskFillColor'),
        uDiskBorderColor: gl.getUniformLocation(diskProgram, 'u_diskBorderColor'),
        uGridColor: gl.getUniformLocation(diskProgram, 'u_gridColor'),
        uDiskBorderWidthPx: gl.getUniformLocation(diskProgram, 'u_diskBorderWidthPx'),
        uGridWidthPx: gl.getUniformLocation(diskProgram, 'u_gridWidthPx'),
      };

      // Upload style uniforms once; they persist for this program.
      gl.useProgram(diskProgram);
      this.uploadPoincareDiskStyleUniforms();
    }

    // Points pipeline (circle)
    gl.useProgram(circleProgram);
    this.pointsCircle = {
      program: circleProgram,
      uPaletteTex: gl.getUniformLocation(circleProgram, 'u_paletteTex'),
      uPaletteSize: gl.getUniformLocation(circleProgram, 'u_paletteSize'),
      uPaletteWidth: gl.getUniformLocation(circleProgram, 'u_paletteWidth'),
      uCssSize: gl.getUniformLocation(circleProgram, 'u_cssSize'),
      uDpr: gl.getUniformLocation(circleProgram, 'u_dpr'),
      uPointRadius: gl.getUniformLocation(circleProgram, 'u_pointRadiusCss'),
    };

    // Points pipeline (square)
    gl.useProgram(squareProgram);
    this.pointsSquare = {
      program: squareProgram,
      uPaletteTex: gl.getUniformLocation(squareProgram, 'u_paletteTex'),
      uPaletteSize: gl.getUniformLocation(squareProgram, 'u_paletteSize'),
      uPaletteWidth: gl.getUniformLocation(squareProgram, 'u_paletteWidth'),
      uCssSize: gl.getUniformLocation(squareProgram, 'u_cssSize'),
      uDpr: gl.getUniformLocation(squareProgram, 'u_dpr'),
      uPointRadius: gl.getUniformLocation(squareProgram, 'u_pointRadiusCss'),
    };

    // Uniform locations (solid)
    gl.useProgram(this.programSolid);
    this.uSolidColor = gl.getUniformLocation(this.programSolid, 'u_color');
    this.uSolidPointSizePx = gl.getUniformLocation(this.programSolid, 'u_pointSizePx');
    this.uSolidRingThicknessPx = gl.getUniformLocation(this.programSolid, 'u_ringThicknessPx');
    this.uSolidRingMode = gl.getUniformLocation(this.programSolid, 'u_ringMode');
    this.uCssSizeSolid = gl.getUniformLocation(this.programSolid, 'u_cssSize');
    this.uDprSolid = gl.getUniformLocation(this.programSolid, 'u_dpr');
    this.uPointRadiusSolid = gl.getUniformLocation(this.programSolid, 'u_pointRadiusCss');

    // Buffers + VAO
    this.vao = gl.createVertexArray();
    this.posBuffer = gl.createBuffer();
    this.labelBuffer = gl.createBuffer();

    // Overlay VAOs/buffers
    this.hoverVao = gl.createVertexArray();
    this.hoverPosBuffer = gl.createBuffer();
    this.hoverLabelBuffer = gl.createBuffer();

    this.selectionVao = gl.createVertexArray();
    this.selectionPosBuffer = gl.createBuffer();
    this.selectionLabelBuffer = gl.createBuffer();

    this.selectionEbo = gl.createBuffer();
    this.hoverEbo = gl.createBuffer();
    this.interactionEbo = gl.createBuffer();

    if (!this.vao || !this.posBuffer || !this.labelBuffer ||
        !this.hoverVao || !this.hoverPosBuffer || !this.hoverLabelBuffer ||
        !this.selectionVao || !this.selectionPosBuffer || !this.selectionLabelBuffer ||
        !this.selectionEbo || !this.hoverEbo || !this.interactionEbo) {
      throw new Error('Failed to allocate WebGL resources');
    }

    gl.bindVertexArray(this.vao);

    // Positions (vec2)
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    // Labels (uint)
    gl.bindBuffer(gl.ARRAY_BUFFER, this.labelBuffer);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_SHORT, 0, 0);

    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // Hover VAO (single point)
    gl.bindVertexArray(this.hoverVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.hoverPosBuffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.hoverLabelBuffer);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_SHORT, 0, 0);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // Selection VAO (N points)
    gl.bindVertexArray(this.selectionVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.selectionPosBuffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.selectionLabelBuffer);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_SHORT, 0, 0);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  protected uploadDatasetToGPU(): void {
    const gl = this.gl!;
    const ds = this.dataset;
    if (!ds) return;

    gl.bindVertexArray(this.vao);

    // Decide whether to upload the full dataset or only a deterministic subsample.
    // NOTE: CPU-side interaction (hitTest/lasso) always uses the full dataset.
    const useFullUpload = ds.n <= this.maxGpuUploadPoints;
    this.gpuUsesFullDataset = useFullUpload;

    if (useFullUpload) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, ds.positions, gl.STATIC_DRAW);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.labelBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, ds.labels, gl.STATIC_DRAW);

      this.gpuPointCount = ds.n;
    } else {
      const n = ds.n;
      const target = Math.min(n, Math.max(250_000, Math.min(this.maxBaseDrawPoints, Math.floor(n * 0.25))));
      const step = Math.max(1, Math.floor(n / target));
      const count = Math.min(target, Math.ceil(n / step));

      const subPos = new Float32Array(count * 2);
      const subLab = new Uint16Array(count);
      let k = 0;
      for (let i = 0; i < n && k < count; i += step) {
        subPos[k * 2] = ds.positions[i * 2];
        subPos[k * 2 + 1] = ds.positions[i * 2 + 1];
        subLab[k] = ds.labels[i];
        k++;
      }

      gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, subPos, gl.STATIC_DRAW);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.labelBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, subLab, gl.STATIC_DRAW);

      this.gpuPointCount = k;
    }

    // Precompute a deterministic, globally distributed subsample for LOD.
    // This is used during interaction (to keep panning smooth), and also as an
    // always-on cap for very large N where drawing every point every frame is
    // not realistic.
    this.interactionCount = 0;
    if (this.interactionEbo && this.gpuUsesFullDataset) {
      const n = ds.n;
      if (n >= 500_000) {
        // Keep enough points for a faithful density impression, but avoid
        // unbounded vertex cost.
        const target = Math.min(n, Math.max(250_000, Math.min(this.maxBaseDrawPoints, Math.floor(n * 0.25))));
        const step = Math.max(1, Math.floor(n / target));
        const count = Math.min(target, Math.ceil(n / step));
        const indices = new Uint32Array(count);
        let k = 0;
        for (let i = 0; i < n && k < count; i += step) {
          indices[k++] = i;
        }
        this.interactionCount = k;

        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.interactionEbo);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
      }
    }

    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  protected uploadSelectionToGPU(): void {
    if (!this.gl || !this.selectionEbo) return;
    const gl = this.gl;

    // Rendering selection overlays for millions of points is both expensive and
    // memory-heavy (index buffers / position buffers). We keep the selection
    // semantics exact, but cap the *rendered* overlay for practicality.
    const MAX_RENDER_SELECTION = 250_000;

    if (!this.gpuUsesFullDataset) {
      const ds = this.dataset;
      if (!ds || !this.selectionVao || !this.selectionPosBuffer || !this.selectionLabelBuffer) return;

      const count = this.selection.size;
      this.selectionOverlayCount = Math.min(count, MAX_RENDER_SELECTION);
      if (count === 0) {
        this.selectionDirty = false;
        return;
      }

      // For huge selections, render only a deterministic prefix (iteration
      // order of Set is deterministic for a fixed construction).
      const renderCount = Math.min(count, MAX_RENDER_SELECTION);
      const pos = new Float32Array(renderCount * 2);
      const lab = new Uint16Array(renderCount);
      let k = 0;
      for (const i of this.selection) {
        if (!this.isPointVisibleByCategory(i)) continue;
        pos[k * 2] = ds.positions[i * 2];
        pos[k * 2 + 1] = ds.positions[i * 2 + 1];
        lab[k] = ds.labels[i];
        k++;
        if (k >= renderCount) break;
      }

      this.selectionOverlayCount = k;

      gl.bindVertexArray(this.selectionVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.selectionPosBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.selectionLabelBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, lab, gl.DYNAMIC_DRAW);
      gl.bindVertexArray(null);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);

      this.selectionDirty = false;
      return;
    }

    // Pack selection indices to Uint32 element buffer.
    const count = this.selection.size;
    const renderCount = Math.min(count, MAX_RENDER_SELECTION);
    this.selectionOverlayCount = renderCount;
    if (renderCount === 0) {
      this.selectionDirty = false;
      return;
    }

    const indices = new Uint32Array(renderCount);
    let k = 0;
    for (const i of this.selection) {
      if (!this.isPointVisibleByCategory(i)) continue;
      indices[k++] = i;
      if (k >= renderCount) break;
    }

    this.selectionOverlayCount = k;

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.selectionEbo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

    this.selectionDirty = false;
  }

  protected uploadHoverToGPU(): void {
    if (!this.gl || !this.hoverEbo) return;
    const gl = this.gl;

    if (!this.gpuUsesFullDataset) {
      const ds = this.dataset;
      if (!ds || !this.hoverVao || !this.hoverPosBuffer || !this.hoverLabelBuffer) return;

      const i = (
        this.hoveredIndex >= 0 &&
        this.hoveredIndex < ds.n &&
        this.isPointVisibleByCategory(this.hoveredIndex)
      )
        ? this.hoveredIndex
        : -1;
      const pos = this.hoverPosScratch;
      const lab = this.hoverLabScratch;
      if (i >= 0) {
        pos[0] = ds.positions[i * 2];
        pos[1] = ds.positions[i * 2 + 1];
        lab[0] = ds.labels[i];
      } else {
        pos[0] = 2;
        pos[1] = 2;
        lab[0] = 0;
      }

      gl.bindVertexArray(this.hoverVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.hoverPosBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.hoverLabelBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, lab, gl.DYNAMIC_DRAW);
      gl.bindVertexArray(null);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);

      this.hoverDirty = false;
      return;
    }

    const idx = this.hoveredIndex >= 0 && this.isPointVisibleByCategory(this.hoveredIndex)
      ? this.hoveredIndex
      : 0;
    const indices = this.hoverIndexScratch;
    indices[0] = idx;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.hoverEbo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

    this.hoverDirty = false;
  }

  // Implemented per-geometry (uniform sets)
  protected abstract bindViewUniformsForProgram(program: WebGLProgram): void;

  render(): void {
    this.ensureGL();

    const gl = this.gl!;
    const ds = this.dataset;
    if (!ds) return;

    // During active interaction (pan/zoom), hyperbolic rendering can become
    // vertex-transform bound at 1M points. Temporarily draw fewer points to
    // keep the interaction at (or near) 60 FPS, then restore full detail when
    // the view stabilizes.
    const now = performance.now();
    const isInteracting = (now - this.lastViewChangeTs) < 80;
    const hasLod = !!this.interactionEbo && this.interactionCount > 0;

    // 1) Interaction LOD (primarily hyperbolic; vertex math is heavier)
    // NOTE: For ~1M points the subsample->full switch can be perceptually jarring
    // (a density "pop") even if the view state is correct. We therefore only
    // enable interaction LOD above a higher threshold.
    const interactionLodMinPoints = 2_000_000;
    const useInteractionLod =
      isInteracting &&
      this.geometryKind() === 'poincare' &&
      ds.n >= interactionLodMinPoints &&
      hasLod;

    // 2) Always-on LOD cap for very large datasets (both geometries)
    // Apply regardless of interaction to avoid catastrophic pan/hover stalls
    // on huge datasets (especially Euclidean).
    const useLargeNLod =
      ds.n > this.maxBaseDrawPoints &&
      hasLod;

    // If the main GPU buffers already contain a subsample, do not additionally
    // apply EBO LOD.
    const useLod = this.gpuUsesFullDataset && (useInteractionLod || useLargeNLod);

    const baseDrawCount = this.gpuUsesFullDataset
      ? (useLod ? this.interactionCount : ds.n)
      : this.gpuPointCount;
    const estimatedFragments = this.estimatePointFragments(baseDrawCount, this.dpr);
    this.updateSquarePointPolicy(estimatedFragments);

    if (this.selectionDirty) this.uploadSelectionToGPU();
    if (this.hoverDirty) this.uploadHoverToGPU();

    // Background (full-res)
    // NOTE: We intentionally avoid gl.blitFramebuffer() here because it can
    // fail (INVALID_OPERATION) depending on driver/default framebuffer
    // constraints, leading to a missing disk/grid. Sampling the cached
    // backdrop texture via a fullscreen draw is robust.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas!.width, this.canvas!.height);
    gl.disable(gl.BLEND);

    if (this.geometryKind() === 'poincare') {
      this.renderBackdropIfNeeded();
      if (this.backdropTex && this.programComposite) {
        gl.useProgram(this.programComposite);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.backdropTex);
        if (this.uCompositeTex) gl.uniform1i(this.uCompositeTex, 0);
        gl.bindVertexArray(this.vao);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.bindTexture(gl.TEXTURE_2D, null);
      } else {
        const [br, bg, bb, ba] = parseHexColor(this.backgroundColor);
        gl.clearColor(br, bg, bb, ba);
        gl.clear(gl.COLOR_BUFFER_BIT);
      }
    } else {
      // Restore clearColor (may have been changed to transparent for offscreen FBO)
      const [br, bg, bb, ba] = parseHexColor(this.backgroundColor);
      gl.clearColor(br, bg, bb, ba);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }

    // Render points into low-res offscreen buffer.
    this.ensurePointsResources();
    if (!this.pointsFbo || !this.pointsTex || !this.programComposite) return;

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.pointsFbo);
    gl.viewport(0, 0, this.pointsW, this.pointsH);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Base points
    // NOTE: WebGL points are square by default. We implement circles by discarding
    // fragments outside the unit disk in the fragment shader (FS_POINTS).
    // For extremely large point counts, using square points can be faster on
    // some GPUs (no discard). However, it noticeably changes appearance.
    //
    // Heuristic: prefer circles up to a few million points.
    const basePoints = this.renderAsSquares ? this.pointsSquare : this.pointsCircle;
    if (!basePoints) return;

    gl.useProgram(basePoints.program);
    this.bindViewUniformsForProgram(basePoints.program);

    // Palette uniforms are uploaded once per program.
    if (this.paletteDirty) this.uploadPaletteUniforms();
    this.bindPaletteTexture();

    if (basePoints.uCssSize) gl.uniform2f(basePoints.uCssSize, this.width, this.height);
    if (basePoints.uDpr) gl.uniform1f(basePoints.uDpr, this.dpr);
    if (basePoints.uPointRadius) gl.uniform1f(basePoints.uPointRadius, this.pointRadiusCss);

    // Optimization: Disable blending when rendering squares (performance mode).
    // This avoids expensive read-modify-write operations for every pixel.
    if (this.renderAsSquares) {
      gl.disable(gl.BLEND);
    } else {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    }

    gl.bindVertexArray(this.vao);
    if (useLod) {
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.interactionEbo);
      gl.drawElements(gl.POINTS, this.interactionCount, gl.UNSIGNED_INT, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    } else {
      const count = this.gpuUsesFullDataset ? ds.n : this.gpuPointCount;
      gl.drawArrays(gl.POINTS, 0, count);
    }

    // Expose policy snapshot (for benchmarks / diagnostics).
    (this as any).__debugPolicy = {
      pointsDpr: this.dpr,
      deviceDpr: this.deviceDpr,
      canvasDpr: this.canvasDpr,
      renderAsSquares: this.renderAsSquares,
      useLod,
      baseDrawCount,
      interactionCount: this.interactionCount,
      gpuUsesFullDataset: this.gpuUsesFullDataset,
      gpuPointCount: this.gpuPointCount,
      estimatedPointFragments: estimatedFragments,
      fragmentBudget: this.policy.fragmentBudget,
      isInteracting,
    };

    // Selection overlay (ring + label-colored fill, still into points buffer)
    if (!isInteracting && this.selection.size > 0) {
      gl.useProgram(this.programSolid);
      this.bindViewUniformsForProgram(this.programSolid!);

      if (this.uCssSizeSolid) gl.uniform2f(this.uCssSizeSolid, this.width, this.height);
      if (this.uDprSolid) gl.uniform1f(this.uDprSolid, this.dpr);
      if (this.uPointRadiusSolid) gl.uniform1f(this.uPointRadiusSolid, this.pointRadiusCss + 1);

      if (this.uSolidColor) {
        const [r, g, b, a] = parseHexColor(this.interactionStyle.selectionColor);
        gl.uniform4f(this.uSolidColor, r, g, b, a);
      }
      if (this.uSolidRingMode) gl.uniform1i(this.uSolidRingMode, 1);
      if (this.uSolidRingThicknessPx) gl.uniform1f(this.uSolidRingThicknessPx, 2);
      if (this.uSolidPointSizePx) gl.uniform1f(this.uSolidPointSizePx, (this.pointRadiusCss + 1) * 2 * this.dpr);

      if (this.gpuUsesFullDataset) {
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.selectionEbo);
        gl.drawElements(gl.POINTS, this.selectionOverlayCount, gl.UNSIGNED_INT, 0);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
      } else if (this.selectionVao && this.selectionOverlayCount > 0) {
        gl.bindVertexArray(this.selectionVao);
        gl.drawArrays(gl.POINTS, 0, this.selectionOverlayCount);
        gl.bindVertexArray(this.vao);
      }

      const circlePoints = this.pointsCircle;
      if (circlePoints) {
        gl.useProgram(circlePoints.program);
        this.bindViewUniformsForProgram(circlePoints.program);
        if (this.paletteDirty) this.uploadPaletteUniforms();
        this.bindPaletteTexture();
        if (circlePoints.uCssSize) gl.uniform2f(circlePoints.uCssSize, this.width, this.height);
        if (circlePoints.uDpr) gl.uniform1f(circlePoints.uDpr, this.dpr);
        if (circlePoints.uPointRadius) gl.uniform1f(circlePoints.uPointRadius, this.pointRadiusCss);

        if (this.gpuUsesFullDataset) {
          gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.selectionEbo);
          gl.drawElements(gl.POINTS, this.selectionOverlayCount, gl.UNSIGNED_INT, 0);
          gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
        } else if (this.selectionVao && this.selectionOverlayCount > 0) {
          gl.bindVertexArray(this.selectionVao);
          gl.drawArrays(gl.POINTS, 0, this.selectionOverlayCount);
          gl.bindVertexArray(this.vao);
        }
      }
    }

    // Hover overlay (still into points buffer)
    if (!isInteracting && this.hoveredIndex >= 0 && this.hoveredIndex < ds.n && this.isPointVisibleByCategory(this.hoveredIndex)) {
      // Ring
      gl.useProgram(this.programSolid);
      this.bindViewUniformsForProgram(this.programSolid!);

      if (this.uCssSizeSolid) gl.uniform2f(this.uCssSizeSolid, this.width, this.height);
      if (this.uDprSolid) gl.uniform1f(this.uDprSolid, this.dpr);

      // Ring pass
      const ringRadius = this.pointRadiusCss + 3;
      if (this.uPointRadiusSolid) gl.uniform1f(this.uPointRadiusSolid, ringRadius);
      if (this.uSolidColor) {
        const [r, g, b, a] = parseHexColor(this.interactionStyle.hoverColor);
        gl.uniform4f(this.uSolidColor, r, g, b, a);
      }
      if (this.uSolidRingMode) gl.uniform1i(this.uSolidRingMode, 1);
      if (this.uSolidRingThicknessPx) gl.uniform1f(this.uSolidRingThicknessPx, 2);
      if (this.uSolidPointSizePx) gl.uniform1f(this.uSolidPointSizePx, ringRadius * 2 * this.dpr);

      if (this.gpuUsesFullDataset) {
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.hoverEbo);
        gl.drawElements(gl.POINTS, 1, gl.UNSIGNED_INT, 0);
      } else if (this.hoverVao) {
        gl.bindVertexArray(this.hoverVao);
        gl.drawArrays(gl.POINTS, 0, 1);
        gl.bindVertexArray(this.vao);
      }

      // Fill pass (selection ring if selected else fill)
      const fillRadius = this.pointRadiusCss + 1;
      if (this.selection.has(this.hoveredIndex)) {
        if (this.uPointRadiusSolid) gl.uniform1f(this.uPointRadiusSolid, fillRadius);
        if (this.uSolidColor) {
          const [r, g, b, a] = parseHexColor(this.interactionStyle.selectionColor);
          gl.uniform4f(this.uSolidColor, r, g, b, a);
        }
        if (this.uSolidRingMode) gl.uniform1i(this.uSolidRingMode, 1);
        if (this.uSolidRingThicknessPx) gl.uniform1f(this.uSolidRingThicknessPx, 2);
        if (this.uSolidPointSizePx) gl.uniform1f(this.uSolidPointSizePx, fillRadius * 2 * this.dpr);
        if (this.gpuUsesFullDataset) {
          gl.drawElements(gl.POINTS, 1, gl.UNSIGNED_INT, 0);
        } else if (this.hoverVao) {
          gl.bindVertexArray(this.hoverVao);
          gl.drawArrays(gl.POINTS, 0, 1);
          gl.bindVertexArray(this.vao);
        }

        const circlePoints = this.pointsCircle;
        if (circlePoints) {
          gl.useProgram(circlePoints.program);
          this.bindViewUniformsForProgram(circlePoints.program);
          if (this.paletteDirty) this.uploadPaletteUniforms();
          this.bindPaletteTexture();
          if (circlePoints.uCssSize) gl.uniform2f(circlePoints.uCssSize, this.width, this.height);
          if (circlePoints.uDpr) gl.uniform1f(circlePoints.uDpr, this.dpr);
          if (circlePoints.uPointRadius) gl.uniform1f(circlePoints.uPointRadius, this.pointRadiusCss);

          if (this.gpuUsesFullDataset) {
            gl.drawElements(gl.POINTS, 1, gl.UNSIGNED_INT, 0);
          } else if (this.hoverVao) {
            gl.bindVertexArray(this.hoverVao);
            gl.drawArrays(gl.POINTS, 0, 1);
            gl.bindVertexArray(this.vao);
          }
        }
      } else {
        const hoverFillColor = this.interactionStyle.hoverFillColor;
        if (hoverFillColor) {
          gl.useProgram(this.programSolid);
          this.bindViewUniformsForProgram(this.programSolid!);
          if (this.uPointRadiusSolid) gl.uniform1f(this.uPointRadiusSolid, fillRadius);
          if (this.uSolidColor) {
            const [r, g, b, a] = parseHexColor(hoverFillColor);
            gl.uniform4f(this.uSolidColor, r, g, b, a);
          }
          if (this.uSolidRingMode) gl.uniform1i(this.uSolidRingMode, 0);
          if (this.uSolidRingThicknessPx) gl.uniform1f(this.uSolidRingThicknessPx, 0);
          if (this.uSolidPointSizePx) gl.uniform1f(this.uSolidPointSizePx, fillRadius * 2 * this.dpr);
          if (this.gpuUsesFullDataset) {
            gl.drawElements(gl.POINTS, 1, gl.UNSIGNED_INT, 0);
          } else if (this.hoverVao) {
            gl.bindVertexArray(this.hoverVao);
            gl.drawArrays(gl.POINTS, 0, 1);
            gl.bindVertexArray(this.vao);
          }
        } else {
          // Use palette program for category color when no hover fill override is set.
          const circlePoints = this.pointsCircle;
          if (!circlePoints) return;
          gl.useProgram(circlePoints.program);
          this.bindViewUniformsForProgram(circlePoints.program);
          if (this.paletteDirty) this.uploadPaletteUniforms();
          this.bindPaletteTexture();
          if (circlePoints.uCssSize) gl.uniform2f(circlePoints.uCssSize, this.width, this.height);
          if (circlePoints.uDpr) gl.uniform1f(circlePoints.uDpr, this.dpr);
          if (circlePoints.uPointRadius) gl.uniform1f(circlePoints.uPointRadius, fillRadius);

          if (this.gpuUsesFullDataset) {
            gl.drawElements(gl.POINTS, 1, gl.UNSIGNED_INT, 0);
          } else if (this.hoverVao) {
            gl.bindVertexArray(this.hoverVao);
            gl.drawArrays(gl.POINTS, 0, 1);
            gl.bindVertexArray(this.vao);
          }
        }
      }

      if (this.gpuUsesFullDataset) {
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
      }
    }

    // Composite points buffer onto full-res default framebuffer.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas!.width, this.canvas!.height);

    gl.useProgram(this.programComposite);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.pointsTex);
    if (this.uCompositeTex) gl.uniform1i(this.uCompositeTex, 0);

    // Blend points over background.
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Restore point VAO for next frame.
    gl.bindVertexArray(this.vao);
    gl.bindTexture(gl.TEXTURE_2D, null);

    gl.bindVertexArray(null);
  }
}

// ============================================================================
// Euclidean candidate
// ============================================================================

export class EuclideanWebGLCandidate extends WebGLRendererBase {
  private view: EuclideanViewState = createEuclideanView();

  private uniformCache = new Map<WebGLProgram, { uCenter: WebGLUniformLocation | null; uZoom: WebGLUniformLocation | null }>();

  protected geometryKind(): GeometryKind {
    return 'euclidean';
  }

  setDataset(dataset: Dataset): void {
    if (dataset.geometry !== 'euclidean') {
      throw new Error('EuclideanWebGLCandidate only supports euclidean geometry');
    }
    super.setDataset(dataset);
    this.fitToData();
  }

  private fitToData(): void {
    const ds = this.dataset;
    if (!ds || ds.n === 0) return;

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < ds.n; i++) {
      const x = ds.positions[i * 2];
      const y = ds.positions[i * 2 + 1];
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }

    const dataWidth = maxX - minX || 1;
    const dataHeight = maxY - minY || 1;
    const dataSize = Math.max(dataWidth, dataHeight);
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
      throw new Error('EuclideanWebGLCandidate only supports euclidean view state');
    }
    this.view = view;
  }

  getView(): ViewState {
    return { ...this.view };
  }

  protected bindViewUniformsForProgram(program: WebGLProgram): void {
    if (!this.gl) return;
    const gl = this.gl;

    let cached = this.uniformCache.get(program);
    if (!cached) {
      cached = {
        uCenter: gl.getUniformLocation(program, 'u_center'),
        uZoom: gl.getUniformLocation(program, 'u_zoom'),
      };
      this.uniformCache.set(program, cached);
    }

    if (cached.uCenter) gl.uniform2f(cached.uCenter, this.view.centerX, this.view.centerY);
    if (cached.uZoom) gl.uniform1f(cached.uZoom, this.view.zoom);
  }

  pan(deltaX: number, deltaY: number, _modifiers: Modifiers): void {
    this.view = panEuclidean(this.view, deltaX, deltaY, this.width, this.height);
    this.markViewChanged();
  }

  zoom(anchorX: number, anchorY: number, delta: number, _modifiers: Modifiers): void {
    this.view = zoomEuclidean(this.view, anchorX, anchorY, delta, this.width, this.height);
    this.markViewChanged();
  }

  hitTest(screenX: number, screenY: number): HitResult | null {
    const ds = this.dataset;
    const idx = this.dataIndex;
    if (!ds || !idx) return null;

    // Reference hit radius rule
    const maxDistPx = this.pointRadiusCss + 5;
    const maxDistSq = maxDistPx * maxDistPx;

    const scale = Math.min(this.width, this.height) * 0.4 * this.view.zoom;
    if (!(scale > 0)) return null;

    // Convert hit radius to data space.
    // Add a tiny epsilon to avoid rare edge-case misses due to floating-point rounding.
    const dataRadius = (maxDistPx / scale) * (1 + 1e-12);
    const maxDataDistSq = dataRadius * dataRadius;

    const dataPt = unprojectEuclidean(screenX, screenY, this.view, this.width, this.height);

    // Inline projection math in the loop to avoid allocating {x,y} objects.
    // We still compute exact screen-space distance for correctness.
    const cx = this.width * 0.5;
    const cy = this.height * 0.5;
    const cX = this.view.centerX;
    const cY = this.view.centerY;

    let bestIndex = -1;
    let bestDistSq = Infinity;

    // Avoid building a potentially large candidates array (push-heavy).
    // Instead iterate overlapping grid cells directly.
    idx.forEachInAABB(
      dataPt.x - dataRadius,
      dataPt.y - dataRadius,
      dataPt.x + dataRadius,
      dataPt.y + dataRadius,
      (i) => {
        if (!this.isCategoryVisible(ds.labels[i])) return;
        const dataX = ds.positions[i * 2];
        const dataY = ds.positions[i * 2 + 1];

        // Fast reject in data space (equivalent up to scale).
        const dxData = dataX - dataPt.x;
        const dyData = dataY - dataPt.y;
        const dataDistSq = dxData * dxData + dyData * dyData;
        if (dataDistSq > maxDataDistSq) return;

        const sx = cx + (dataX - cX) * scale;
        const sy = cy - (dataY - cY) * scale;
        const dx = sx - screenX;
        const dy = sy - screenY;
        const distSq = dx * dx + dy * dy;

        if (distSq <= maxDistSq) {
          if (distSq < bestDistSq || (distSq === bestDistSq && i < bestIndex)) {
            bestDistSq = distSq;
            bestIndex = i;
          }
        }
      },
    );

    if (bestIndex < 0) return null;

    const bx = ds.positions[bestIndex * 2];
    const by = ds.positions[bestIndex * 2 + 1];
    const screen = projectEuclidean(bx, by, this.view, this.width, this.height);

    return {
      index: bestIndex,
      screenX: screen.x,
      screenY: screen.y,
      distance: Math.sqrt(bestDistSq),
    };
  }

  lassoSelect(polyline: Float32Array): SelectionResult {
    const ds = this.dataset;
    const idx = this.dataIndex;
    if (!ds || !idx) return createIndicesSelectionResult(new Set(), 0);

    const startTime = performance.now();

    // Always return geometry (Embedding Atlas style). The UI/benchmarks should
    // use Renderer.countSelection(...) to obtain counts efficiently.

    const dataPolyline = new Float32Array(polyline.length);
    for (let i = 0; i < polyline.length / 2; i++) {
      const sx = polyline[i * 2];
      const sy = polyline[i * 2 + 1];
      const data = unprojectEuclidean(sx, sy, this.view, this.width, this.height);
      dataPolyline[i * 2] = data.x;
      dataPolyline[i * 2 + 1] = data.y;
    }

    // Tight AABB for fast reject in has() and efficient indexed counting.
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (let i = 0; i < dataPolyline.length; i += 2) {
      const x = dataPolyline[i];
      const y = dataPolyline[i + 1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }

    const bounds = { xMin: minX, yMin: minY, xMax: maxX, yMax: maxY };
    const geometry: SelectionGeometry = { type: 'polygon', coords: dataPolyline, bounds };
    const computeTimeMs = performance.now() - startTime;
    return {
      kind: 'geometry',
      geometry,
      computeTimeMs,
      has: (index: number) => {
        if (index < 0 || index >= ds.n) return false;
        if (!this.isCategoryVisible(ds.labels[index])) return false;
        const px = ds.positions[index * 2];
        const py = ds.positions[index * 2 + 1];
        if (px < bounds.xMin || px > bounds.xMax || py < bounds.yMin || py > bounds.yMax) return false;
        return pointInPolygon(px, py, dataPolyline);
      },
    };
  }

  projectToScreen(dataX: number, dataY: number): { x: number; y: number } {
    return projectEuclidean(dataX, dataY, this.view, this.width, this.height);
  }

  unprojectFromScreen(screenX: number, screenY: number): { x: number; y: number } {
    return unprojectEuclidean(screenX, screenY, this.view, this.width, this.height);
  }
}

// ============================================================================
// Hyperbolic candidate
// ============================================================================

export class HyperbolicWebGLCandidate extends WebGLRendererBase {
  private view: HyperbolicViewState = createHyperbolicView();

  private uniformCache = new Map<WebGLProgram, { uA: WebGLUniformLocation | null; uDisplayZoom: WebGLUniformLocation | null }>();

  // Pan tracking (same as reference)
  private lastPanScreenX = 0;
  private lastPanScreenY = 0;
  private hasPanAnchor = false;

  protected geometryKind(): GeometryKind {
    return 'poincare';
  }

  protected override getBackdropZoom(): number {
    return this.view.displayZoom;
  }

  setDataset(dataset: Dataset): void {
    if (dataset.geometry !== 'poincare') {
      throw new Error('HyperbolicWebGLCandidate only supports poincare geometry');
    }
    super.setDataset(dataset);
    this.view = createHyperbolicView();

    this.hasPanAnchor = false;
  }

  setView(view: ViewState): void {
    if (view.type !== 'poincare') {
      throw new Error('HyperbolicWebGLCandidate only supports poincare view state');
    }
    this.view = view;
    this.markBackdropDirty();
  }

  getView(): ViewState {
    return { ...this.view };
  }

  protected bindViewUniformsForProgram(program: WebGLProgram): void {
    if (!this.gl) return;
    const gl = this.gl;

    let cached = this.uniformCache.get(program);
    if (!cached) {
      cached = {
        uA: gl.getUniformLocation(program, 'u_a'),
        uDisplayZoom: gl.getUniformLocation(program, 'u_displayZoom'),
      };
      this.uniformCache.set(program, cached);
    }

    if (cached.uA) gl.uniform2f(cached.uA, this.view.ax, this.view.ay);
    if (cached.uDisplayZoom) gl.uniform1f(cached.uDisplayZoom, this.view.displayZoom);
  }

  // For accuracy harness: called via reflection if present.
  startPan(screenX: number, screenY: number): void {
    this.lastPanScreenX = screenX;
    this.lastPanScreenY = screenY;
    this.hasPanAnchor = true;
  }

  pan(deltaX: number, deltaY: number, _modifiers: Modifiers): void {
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

    this.markViewChanged();

    this.lastPanScreenX = endX;
    this.lastPanScreenY = endY;
  }

  zoom(anchorX: number, anchorY: number, delta: number, _modifiers: Modifiers): void {
    this.view = zoomPoincare(this.view, anchorX, anchorY, delta, this.width, this.height);
    this.markViewChanged();
    this.markBackdropDirty();
  }

  private mobiusDerivativeScaleAt(zx: number, zy: number): number {
    // For T_a(z) = (z - a) / (1 - conj(a) z), the conformal scale factor is:
    // |T'_a(z)| = (1 - |a|^2) / |1 - conj(a) z|^2
    const ax = this.view.ax;
    const ay = this.view.ay;
    const a2 = ax * ax + ay * ay;
    const denomX = 1.0 - (ax * zx + ay * zy);
    const denomY = -(ax * zy - ay * zx);
    const denomNormSq = denomX * denomX + denomY * denomY;
    if (denomNormSq < 1e-12) return 0;
    const num = Math.max(0, 1.0 - a2);
    return num / denomNormSq;
  }

  /**
   * Compute a conservative Euclidean data-space radius that guarantees we won't
   * miss any point within `screenRadiusPx` of the cursor.
   *
   * Derivation:
   * - Screen-space displacement is (locally) scaled by:
   *     localScale(z) = diskRadius * |T'_a(z)|
   *   where T_a is the Möbius transform used by the camera.
   * - |T'_a(z)| = (1 - |a|^2) / |1 - conj(a) z|^2.
   * - Over a Euclidean ball |z - z0| <= r, the denominator norm is Lipschitz:
   *     | |1 - conj(a)z| - |1 - conj(a)z0| | <= |a| * r
   *   hence for any z in the ball:
   *     |1 - conj(a)z| <= D0 + |a| r
   *   which yields a lower bound on |T'_a(z)| (worst-case smallest scale).
   *
   * We solve the fixed point:
   *   r = screenRadiusPx / (diskRadius * min|T'_a|)
   *     = screenRadiusPx * (D0 + |a| r)^2 / (diskRadius * (1 - |a|^2))
   * by a few iterations (converges quickly for |a|<1).
   */
  private conservativeDataRadiusForScreenRadius(
    zx: number,
    zy: number,
    screenRadiusPx: number,
    diskRadius: number
  ): number {
    const ax = this.view.ax;
    const ay = this.view.ay;
    const a2 = ax * ax + ay * ay;
    const aMag = Math.sqrt(a2);

    const C = Math.max(1e-12, 1.0 - a2);
    if (!(diskRadius > 1e-9) || !(screenRadiusPx > 0)) return 0;

    // D0 = |1 - conj(a) z0|
    const denomX0 = 1.0 - (ax * zx + ay * zy);
    const denomY0 = -(ax * zy - ay * zx);
    const D0 = Math.sqrt(denomX0 * denomX0 + denomY0 * denomY0);
    if (!Number.isFinite(D0) || D0 < 1e-12) return 2.0;

    const K = screenRadiusPx / (diskRadius * C);
    let r = K * D0 * D0;

    // Fixed-point iterations. 4-5 is plenty.
    for (let it = 0; it < 5; it++) {
      const D = D0 + aMag * r;
      r = K * D * D;
    }

    if (!Number.isFinite(r)) return 2.0;
    // Tiny slack for floating point noise.
    r *= 1.001;
    return Math.min(1.999, Math.max(0, r));
  }

  hitTest(screenX: number, screenY: number): HitResult | null {
    const ds = this.dataset;
    const idx = this.dataIndex;
    if (!ds || !idx) return null;

    const { width, height, view } = this;
    const centerX = width / 2;
    const centerY = height / 2;
    const diskRadius = Math.min(width, height) * 0.45 * view.displayZoom;
    const diskR2 = diskRadius * diskRadius;

    const maxDistPx = this.pointRadiusCss + 5;
    const maxDistSq = maxDistPx * maxDistPx;

    // Reference semantics: cursor may be outside the disk. We only cull points
    // based on their *projected* position being outside the disk.
    //
    // However, if the cursor is far enough outside the disk that no point
    // inside the disk could be within the hit radius, we can return null.
    const dxCur = screenX - centerX;
    const dyCur = screenY - centerY;
    const maxCursorR = diskRadius + maxDistPx;
    if (dxCur * dxCur + dyCur * dyCur > maxCursorR * maxCursorR) return null;

    // Convert cursor to data space.
    const dataPt = unprojectPoincare(screenX, screenY, view, width, height);
    const queryRadius = this.conservativeDataRadiusForScreenRadius(
      dataPt.x,
      dataPt.y,
      maxDistPx,
      diskRadius
    );

    let bestIndex = -1;
    let bestDistSq = Infinity;

    // Inline the Poincaré projection math to avoid per-candidate object allocations.
    // This mirrors `projectPoincare()` + `mobiusTransform()` clamping behavior.
    const ax = view.ax;
    const ay = view.ay;

    // Avoid building a candidates array; iterate overlapping cells directly.
    idx.forEachInAABB(
      dataPt.x - queryRadius,
      dataPt.y - queryRadius,
      dataPt.x + queryRadius,
      dataPt.y + queryRadius,
      (i) => {
        if (!this.isCategoryVisible(ds.labels[i])) return;
        const dataX = ds.positions[i * 2];
        const dataY = ds.positions[i * 2 + 1];

        // mobiusTransform(z) = (z - a) / (1 - conj(a) * z)
        const numX = dataX - ax;
        const numY = dataY - ay;
        const denomX = 1.0 - (ax * dataX + ay * dataY);
        const denomY = -(ax * dataY - ay * dataX);
        const denomNormSq = denomX * denomX + denomY * denomY;

        let wx = 0.0;
        let wy = 0.0;

        if (denomNormSq < 1e-12) {
          const norm = Math.sqrt(numX * numX + numY * numY);
          if (norm < 1e-12) {
            wx = 0.0;
            wy = 0.0;
          } else {
            wx = (numX / norm) * 0.999;
            wy = (numY / norm) * 0.999;
          }
        } else {
          wx = (numX * denomX + numY * denomY) / denomNormSq;
          wy = (numY * denomX - numX * denomY) / denomNormSq;
          const rSq = wx * wx + wy * wy;
          if (rSq >= 1.0) {
            const r = Math.sqrt(rSq);
            wx = (wx / r) * 0.999;
            wy = (wy / r) * 0.999;
          }
        }

        const sx = centerX + wx * diskRadius;
        const sy = centerY - wy * diskRadius;

        const dxDisk = sx - centerX;
        const dyDisk = sy - centerY;
        if (dxDisk * dxDisk + dyDisk * dyDisk > diskR2) return;

        const dx = sx - screenX;
        const dy = sy - screenY;
        const distSq = dx * dx + dy * dy;

        if (distSq <= maxDistSq) {
          if (distSq < bestDistSq || (distSq === bestDistSq && i < bestIndex)) {
            bestDistSq = distSq;
            bestIndex = i;
          }
        }
      },
    );

    if (bestIndex < 0) return null;

    const bx = ds.positions[bestIndex * 2];
    const by = ds.positions[bestIndex * 2 + 1];

    // Recompute best screen position (single point) without allocations.
    const bNumX = bx - ax;
    const bNumY = by - ay;
    const bDenomX = 1.0 - (ax * bx + ay * by);
    const bDenomY = -(ax * by - ay * bx);
    const bDenomNormSq = bDenomX * bDenomX + bDenomY * bDenomY;

    let bwx = 0.0;
    let bwy = 0.0;
    if (bDenomNormSq < 1e-12) {
      const norm = Math.sqrt(bNumX * bNumX + bNumY * bNumY);
      if (norm < 1e-12) {
        bwx = 0.0;
        bwy = 0.0;
      } else {
        bwx = (bNumX / norm) * 0.999;
        bwy = (bNumY / norm) * 0.999;
      }
    } else {
      bwx = (bNumX * bDenomX + bNumY * bDenomY) / bDenomNormSq;
      bwy = (bNumY * bDenomX - bNumX * bDenomY) / bDenomNormSq;
      const rSq = bwx * bwx + bwy * bwy;
      if (rSq >= 1.0) {
        const r = Math.sqrt(rSq);
        bwx = (bwx / r) * 0.999;
        bwy = (bwy / r) * 0.999;
      }
    }

    const bestScreenX = centerX + bwx * diskRadius;
    const bestScreenY = centerY - bwy * diskRadius;

    return {
      index: bestIndex,
      screenX: bestScreenX,
      screenY: bestScreenY,
      distance: Math.sqrt(bestDistSq),
    };
  }

  lassoSelect(polyline: Float32Array): SelectionResult {
    const ds = this.dataset;
    const idx = this.dataIndex;
    if (!ds || !idx) return createIndicesSelectionResult(new Set(), 0);

    const startTime = performance.now();

    // Always return geometry (Embedding Atlas style). The UI/benchmarks should
    // use Renderer.countSelection(...) to obtain counts efficiently.

    const dataPolyline = new Float32Array(polyline.length);
    for (let i = 0; i < polyline.length / 2; i++) {
      const sx = polyline[i * 2];
      const sy = polyline[i * 2 + 1];
      const data = unprojectPoincare(sx, sy, this.view, this.width, this.height);
      dataPolyline[i * 2] = data.x;
      dataPolyline[i * 2 + 1] = data.y;
    }

    // Tight AABB for fast reject in has() and efficient indexed counting.
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (let i = 0; i < dataPolyline.length; i += 2) {
      const x = dataPolyline[i];
      const y = dataPolyline[i + 1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }

    const bounds = { xMin: minX, yMin: minY, xMax: maxX, yMax: maxY };
    const geometry: SelectionGeometry = { type: 'polygon', coords: dataPolyline, bounds };
    const computeTimeMs = performance.now() - startTime;
    return {
      kind: 'geometry',
      geometry,
      computeTimeMs,
      has: (index: number) => {
        if (index < 0 || index >= ds.n) return false;
        if (!this.isCategoryVisible(ds.labels[index])) return false;
        const px = ds.positions[index * 2];
        const py = ds.positions[index * 2 + 1];
        if (px < bounds.xMin || px > bounds.xMax || py < bounds.yMin || py > bounds.yMax) return false;
        return pointInPolygon(px, py, dataPolyline);
      },
    };
  }

  projectToScreen(dataX: number, dataY: number): { x: number; y: number } {
    return projectPoincare(dataX, dataY, this.view, this.width, this.height);
  }

  unprojectFromScreen(screenX: number, screenY: number): { x: number; y: number } {
    return unprojectPoincare(screenX, screenY, this.view, this.width, this.height);
  }
}
