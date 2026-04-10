import {
  DEFAULT_COLORS,
  HOVER_COLOR,
  type LassoStyle,
  SELECTION_COLOR,
  type CategoryVisibilityMask,
  type CountSelectionOptions,
  type InteractionStyle,
} from "../core/types.js";
import { pointInPolygon } from "../core/selection/point_in_polygon.js";
import {
  createDefaultOrbitView3D,
  createIndicesSelectionResult3D,
  type Dataset3D,
  type GeometryMode3D,
  type HitResult3D,
  type InitOptions3D,
  type Modifiers3D,
  type OrbitViewState3D,
  type ProjectedPoint3D,
  type Renderer3D,
  type SelectionResult3D,
} from "../core/types3d.js";

const WORLD_UP: Vec3 = [0, 1, 0];
const MAX_SELECTION_RENDER_POINTS = 250_000;

type Vec3 = [number, number, number];

type Mat4 = Float32Array;

function clamp(v: number, lo: number, hi: number): number {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

function createMat4Identity(): Mat4 {
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);
}

function mat4Multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] =
        a[0 * 4 + r] * b[c * 4 + 0] +
        a[1 * 4 + r] * b[c * 4 + 1] +
        a[2 * 4 + r] * b[c * 4 + 2] +
        a[3 * 4 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

function vec3Length(v: Vec3): number {
  return Math.hypot(v[0], v[1], v[2]);
}

function vec3Normalize(v: Vec3): Vec3 {
  const len = vec3Length(v);
  if (len < 1e-12) return [0, 0, 0];
  return [v[0] / len, v[1] / len, v[2] / len];
}

function vec3Sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function vec3Add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function vec3Scale(v: Vec3, s: number): Vec3 {
  return [v[0] * s, v[1] * s, v[2] * s];
}

function vec3Cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function mat4LookAt(eye: Vec3, target: Vec3, up: Vec3): Mat4 {
  const zAxis = vec3Normalize(vec3Sub(eye, target));
  let xAxis = vec3Normalize(vec3Cross(up, zAxis));
  if (vec3Length(xAxis) < 1e-9) {
    xAxis = [1, 0, 0];
  }
  const yAxis = vec3Cross(zAxis, xAxis);

  const out = createMat4Identity();
  out[0] = xAxis[0];
  out[1] = yAxis[0];
  out[2] = zAxis[0];
  out[3] = 0;

  out[4] = xAxis[1];
  out[5] = yAxis[1];
  out[6] = zAxis[1];
  out[7] = 0;

  out[8] = xAxis[2];
  out[9] = yAxis[2];
  out[10] = zAxis[2];
  out[11] = 0;

  out[12] = -(xAxis[0] * eye[0] + xAxis[1] * eye[1] + xAxis[2] * eye[2]);
  out[13] = -(yAxis[0] * eye[0] + yAxis[1] * eye[1] + yAxis[2] * eye[2]);
  out[14] = -(zAxis[0] * eye[0] + zAxis[1] * eye[1] + zAxis[2] * eye[2]);
  out[15] = 1;
  return out;
}

function mat4Ortho(
  left: number,
  right: number,
  bottom: number,
  top: number,
  near: number,
  far: number,
): Mat4 {
  const out = new Float32Array(16);
  out[0] = 2 / (right - left);
  out[5] = 2 / (top - bottom);
  out[10] = -2 / (far - near);
  out[12] = -(right + left) / (right - left);
  out[13] = -(top + bottom) / (top - bottom);
  out[14] = -(far + near) / (far - near);
  out[15] = 1;
  return out;
}

function transformClip(m: Mat4, x: number, y: number, z: number): [number, number, number, number] {
  return [
    m[0] * x + m[4] * y + m[8] * z + m[12],
    m[1] * x + m[5] * y + m[9] * z + m[13],
    m[2] * x + m[6] * y + m[10] * z + m[14],
    m[3] * x + m[7] * y + m[11] * z + m[15],
  ];
}

function parseHexColor(color: string): [number, number, number, number] {
  const s = color.trim();
  if (!s.startsWith("#")) return [1, 1, 1, 1];

  const hex = s.slice(1);
  if (hex.length === 3) {
    const r = Number.parseInt(hex[0] + hex[0], 16) / 255;
    const g = Number.parseInt(hex[1] + hex[1], 16) / 255;
    const b = Number.parseInt(hex[2] + hex[2], 16) / 255;
    return [r, g, b, 1];
  }

  if (hex.length === 6 || hex.length === 8) {
    const r = Number.parseInt(hex.slice(0, 2), 16) / 255;
    const g = Number.parseInt(hex.slice(2, 4), 16) / 255;
    const b = Number.parseInt(hex.slice(4, 6), 16) / 255;
    const a = hex.length === 8 ? Number.parseInt(hex.slice(6, 8), 16) / 255 : 1;
    return [r, g, b, a];
  }

  return [1, 1, 1, 1];
}

function parseHexColorBytes(color: string): [number, number, number, number] {
  const s = color.trim();
  if (!s.startsWith("#")) return [255, 255, 255, 255];

  const hex = s.slice(1);
  if (hex.length === 3) {
    return [
      Number.parseInt(hex[0] + hex[0], 16),
      Number.parseInt(hex[1] + hex[1], 16),
      Number.parseInt(hex[2] + hex[2], 16),
      255,
    ];
  }

  if (hex.length === 6 || hex.length === 8) {
    return [
      Number.parseInt(hex.slice(0, 2), 16),
      Number.parseInt(hex.slice(2, 4), 16),
      Number.parseInt(hex.slice(4, 6), 16),
      hex.length === 8 ? Number.parseInt(hex.slice(6, 8), 16) : 255,
    ];
  }

  return [255, 255, 255, 255];
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("Failed to create shader");
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader) ?? "unknown";
    gl.deleteShader(shader);
    throw new Error(`Shader compile failed: ${info}`);
  }
  return shader;
}

function linkProgram(gl: WebGL2RenderingContext, vsSource: string, fsSource: string): WebGLProgram {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSource);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSource);
  const program = gl.createProgram();
  if (!program) throw new Error("Failed to create program");

  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);

  gl.deleteShader(vs);
  gl.deleteShader(fs);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program) ?? "unknown";
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

function drawScreenSpaceLassoOverlay(
  ctx: CanvasRenderingContext2D,
  dpr: number,
  width: number,
  height: number,
  polygon: Float32Array | null,
  style: Required<LassoStyle>,
): void {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, Math.max(1, Math.floor(width * dpr)), Math.max(1, Math.floor(height * dpr)));

  if (!polygon || polygon.length < 6) return;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.beginPath();
  ctx.moveTo(polygon[0], polygon[1]);
  for (let i = 2; i < polygon.length; i += 2) {
    ctx.lineTo(polygon[i], polygon[i + 1]);
  }
  ctx.closePath();
  ctx.fillStyle = style.fillColor;
  ctx.fill();
  ctx.strokeStyle = style.strokeColor;
  ctx.lineWidth = Math.max(1, style.strokeWidth);
  ctx.stroke();
}

function createSelectionResult(indices: Set<number>, computeTimeMs: number): SelectionResult3D {
  return createIndicesSelectionResult3D(indices, computeTimeMs);
}

const VS_POINTS_3D = `#version 300 es
precision highp float;
precision highp int;

layout(location = 0) in vec3 a_pos;
layout(location = 1) in uint a_label;

uniform mat4 u_mvp;
uniform float u_pointSizePx;

flat out uint v_label;

void main() {
  gl_Position = u_mvp * vec4(a_pos, 1.0);
  gl_PointSize = u_pointSizePx;
  v_label = a_label;
}
`;

const FS_POINTS_3D = `#version 300 es
precision highp float;
precision highp int;

flat in uint v_label;

uniform sampler2D u_paletteTex;
uniform int u_paletteSize;
uniform int u_paletteWidth;

out vec4 outColor;

void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float r = length(p);
  float aa = max(fwidth(r), 0.02);
  float alpha = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, r);
  if (alpha <= 0.0) discard;

  int size = max(u_paletteSize, 1);
  int w = max(u_paletteWidth, 1);
  int idx = int(v_label) % size;
  int x = idx % w;
  int y = idx / w;
  vec4 c = texelFetch(u_paletteTex, ivec2(x, y), 0);
  float outAlpha = c.a * alpha;
  if (outAlpha <= 0.0) discard;
  outColor = vec4(c.rgb, outAlpha);
}
`;

const VS_SOLID_3D = `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_pos;

uniform mat4 u_mvp;
uniform float u_pointSizePx;

void main() {
  gl_Position = u_mvp * vec4(a_pos, 1.0);
  gl_PointSize = u_pointSizePx;
}
`;

const FS_SOLID_3D = `#version 300 es
precision highp float;

uniform vec4 u_color;
uniform float u_pointSizePx;
uniform int u_ringMode;
uniform float u_ringThicknessPx;

out vec4 outColor;

void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float r = length(p);
  float radiusPx = max(u_pointSizePx * 0.5, 1.0);
  float aa = max(fwidth(r), 1.2 / radiusPx);
  float outer = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, r);
  if (outer <= 0.0) discard;

  float alpha = outer;

  if (u_ringMode == 1) {
    float t = clamp(u_ringThicknessPx / max(radiusPx, 1e-6), 0.0, 1.0);
    float inner = 1.0 - t;
    float innerMask = smoothstep(inner - aa, inner + aa, r);
    alpha *= innerMask;
    if (alpha <= 0.0) discard;
  }

  outColor = vec4(u_color.rgb, u_color.a * alpha);
}
`;

const VS_GUIDE_3D = `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_pos;
uniform mat4 u_mvp;

void main() {
  gl_Position = u_mvp * vec4(a_pos, 1.0);
}
`;

const FS_GUIDE_3D = `#version 300 es
precision highp float;

uniform vec4 u_color;
out vec4 outColor;

void main() {
  outColor = u_color;
}
`;

interface PointsProgram {
  program: WebGLProgram;
  uMvp: WebGLUniformLocation | null;
  uPointSizePx: WebGLUniformLocation | null;
  uPaletteTex: WebGLUniformLocation | null;
  uPaletteSize: WebGLUniformLocation | null;
  uPaletteWidth: WebGLUniformLocation | null;
}

interface SolidProgram {
  program: WebGLProgram;
  uMvp: WebGLUniformLocation | null;
  uPointSizePx: WebGLUniformLocation | null;
  uColor: WebGLUniformLocation | null;
  uRingMode: WebGLUniformLocation | null;
  uRingThicknessPx: WebGLUniformLocation | null;
}

interface GuideProgram {
  program: WebGLProgram;
  uMvp: WebGLUniformLocation | null;
  uColor: WebGLUniformLocation | null;
}

abstract class PointCloud3DWebGLBase implements Renderer3D {
  protected canvas: HTMLCanvasElement | null = null;
  protected overlayCanvas: HTMLCanvasElement | null = null;
  protected overlayCtx: CanvasRenderingContext2D | null = null;
  protected gl: WebGL2RenderingContext | null = null;

  protected width = 0;
  protected height = 0;
  protected dpr = 1;

  protected dataset: Dataset3D | null = null;

  protected view: OrbitViewState3D = createDefaultOrbitView3D();
  protected sceneRadius = 1;

  protected backgroundColor = "#0a0a0a";
  protected pointRadiusCss = 4;
  protected colors: string[] = DEFAULT_COLORS;
  protected sphereGuideColor = "#94a3b8";
  protected sphereGuideOpacity = 0.2;
  protected categoryVisibilityMask = new Uint8Array(0);
  protected hasCategoryVisibilityMask = false;
  protected categoryAlpha = 1;
  protected interactionStyle: Required<InteractionStyle> = {
    selectionColor: SELECTION_COLOR,
    selectionRadiusOffset: 2,
    selectionRingWidth: 2,
    highlightColor: "#94a3b8",
    highlightRadiusOffset: 1,
    highlightRingWidth: 1.5,
    hoverColor: HOVER_COLOR,
    hoverFillColor: null,
    hoverRadiusOffset: 3,
  };

  protected selection = new Set<number>();
  protected highlight = new Set<number>();
  protected hoveredIndex = -1;
  protected lassoPolygon: Float32Array | null = null;
  protected lassoStyle: Required<LassoStyle> = {
    strokeColor: "#6366f1",
    strokeWidth: 2,
    fillColor: "rgba(99, 102, 241, 0.12)",
  };

  protected pointsProgram: PointsProgram | null = null;
  protected solidProgram: SolidProgram | null = null;
  protected guideProgram: GuideProgram | null = null;

  protected pointsVao: WebGLVertexArrayObject | null = null;
  protected pointsPosBuffer: WebGLBuffer | null = null;
  protected pointsLabelBuffer: WebGLBuffer | null = null;

  protected selectionVao: WebGLVertexArrayObject | null = null;
  protected selectionPosBuffer: WebGLBuffer | null = null;
  protected selectionLabelBuffer: WebGLBuffer | null = null;
  protected selectionVertexCount = 0;

  protected highlightVao: WebGLVertexArrayObject | null = null;
  protected highlightPosBuffer: WebGLBuffer | null = null;
  protected highlightLabelBuffer: WebGLBuffer | null = null;
  protected highlightVertexCount = 0;

  protected hoverVao: WebGLVertexArrayObject | null = null;
  protected hoverPosBuffer: WebGLBuffer | null = null;
  protected hoverLabelBuffer: WebGLBuffer | null = null;
  protected hoverVertexCount = 0;

  protected guideVao: WebGLVertexArrayObject | null = null;
  protected guideBuffer: WebGLBuffer | null = null;
  protected guideVertexCount = 0;
  protected guideSegmentVerts = 0;
  protected guideAxisVertexOffset = 0;
  protected guideAxisVertexCount = 0;

  protected paletteTex: WebGLTexture | null = null;
  protected paletteSize = 0;
  protected paletteWidth = 0;
  protected paletteHeight = 0;
  protected paletteBytes = new Uint8Array(0);
  protected readonly paletteTexUnit = 1;

  protected mvpMatrix: Mat4 = createMat4Identity();
  protected projectedDirty = true;
  protected selectionDirty = true;
  protected highlightDirty = true;
  protected hoverDirty = true;

  protected projectedScreenX = new Float32Array(0);
  protected projectedScreenY = new Float32Array(0);
  protected projectedDepth = new Float32Array(0);
  protected projectedPixelIndex = new Int32Array(0);
  protected projectedVisible = new Uint8Array(0);
  protected projectedVisibleIndices = new Uint32Array(0);
  protected projectedVisibleCount = 0;
  protected depthBuffer = new Float32Array(0);

  protected selectionPositionsScratch = new Float32Array(0);
  protected selectionLabelsScratch = new Uint16Array(0);
  protected highlightPositionsScratch = new Float32Array(0);
  protected highlightLabelsScratch = new Uint16Array(0);
  protected hoverPositionScratch = new Float32Array(3);
  protected hoverLabelScratch = new Uint16Array(1);

  protected ensureOverlayCanvas(): void {
    if (!this.canvas || typeof document === "undefined") return;

    if (!this.overlayCanvas || !this.overlayCtx || !this.overlayCanvas.isConnected) {
      const parent = this.canvas.parentElement;
      if (!parent) return;

      const parentStyle = window.getComputedStyle(parent);
      if (parentStyle.position === "static") {
        parent.style.position = "relative";
      }

      const overlay = document.createElement("canvas");
      overlay.setAttribute("aria-hidden", "true");
      overlay.dataset.hyperScatterOverlay = "lasso";
      overlay.style.position = "absolute";
      overlay.style.inset = "0";
      overlay.style.pointerEvents = "none";

      const baseZ = Number.parseInt(window.getComputedStyle(this.canvas).zIndex ?? "0", 10);
      overlay.style.zIndex = Number.isFinite(baseZ) ? String(baseZ + 1) : "1";

      parent.appendChild(overlay);
      this.overlayCanvas = overlay;
      this.overlayCtx = overlay.getContext("2d");
    }

    this.syncOverlayCanvas();
  }

  protected syncOverlayCanvas(): void {
    if (!this.overlayCanvas || !this.overlayCtx) return;
    setCanvasSize(this.overlayCanvas, this.width, this.height, this.dpr);
    drawScreenSpaceLassoOverlay(
      this.overlayCtx,
      this.dpr,
      this.width,
      this.height,
      this.lassoPolygon,
      this.lassoStyle,
    );
  }

  protected removeOverlayCanvas(): void {
    if (this.overlayCanvas?.parentElement) {
      this.overlayCanvas.parentElement.removeChild(this.overlayCanvas);
    }
    this.overlayCanvas = null;
    this.overlayCtx = null;
  }

  protected abstract expectedGeometry(): GeometryMode3D;

  protected supportsSphereGuide(): boolean {
    return false;
  }

  protected supportsEuclideanGuide(): boolean {
    return false;
  }

  protected preprocessPositions(input: Float32Array): Float32Array {
    return input;
  }

  init(canvas: HTMLCanvasElement, opts: InitOptions3D): void {
    this.canvas = canvas;
    this.width = opts.width;
    this.height = opts.height;
    this.dpr = opts.devicePixelRatio ?? window.devicePixelRatio ?? 1;

    if (opts.backgroundColor) this.backgroundColor = opts.backgroundColor;
    if (typeof opts.pointRadius === "number" && Number.isFinite(opts.pointRadius)) {
      this.pointRadiusCss = Math.max(1, opts.pointRadius);
    }
    if (opts.colors?.length) this.colors = opts.colors;
    if (opts.sphereGuideColor) this.sphereGuideColor = opts.sphereGuideColor;
    if (typeof opts.sphereGuideOpacity === "number" && Number.isFinite(opts.sphereGuideOpacity)) {
      this.sphereGuideOpacity = clamp(opts.sphereGuideOpacity, 0, 1);
    }

    setCanvasSize(canvas, this.width, this.height, this.dpr);
    this.projectedDirty = true;
    this.selectionDirty = true;
    this.highlightDirty = true;
    this.hoverDirty = true;
    this.ensureOverlayCanvas();
  }

  setDataset(dataset: Dataset3D): void {
    if (dataset.geometry !== this.expectedGeometry()) {
      throw new Error(`Expected geometry '${this.expectedGeometry()}', got '${dataset.geometry}'`);
    }

    const processedPositions = this.preprocessPositions(dataset.positions);
    this.dataset = {
      n: dataset.n,
      positions: processedPositions,
      labels: dataset.labels,
      geometry: dataset.geometry,
    };

    this.selection = new Set<number>();
    this.highlight = new Set<number>();
    this.hoveredIndex = -1;
    this.lassoPolygon = null;

    this.fitViewToDataset();
    this.ensureProjectedCapacity(this.dataset.n);

    this.selectionDirty = true;
    this.highlightDirty = true;
    this.hoverDirty = true;
    this.projectedDirty = true;
    this.syncOverlayCanvas();

    if (this.gl) {
      this.uploadDatasetToGPU();
      this.rebuildGuideGeometry();
      this.uploadSelectionToGPU();
      this.uploadHighlightToGPU();
      this.uploadHoverToGPU();
    }
  }

  setPalette(colors: string[]): void {
    this.colors = colors;
    if (this.gl) {
      this.uploadPalette();
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

    this.selectionDirty = true;
    this.highlightDirty = true;
    this.hoverDirty = true;
    this.projectedDirty = true;
    if (this.gl) {
      this.uploadPalette();
      this.uploadSelectionToGPU();
      this.uploadHighlightToGPU();
      this.uploadHoverToGPU();
    }
  }

  setCategoryAlpha(alpha: number): void {
    this.setInactiveOpacity(alpha);
  }

  setInactiveOpacity(alpha: number): void {
    const next = Number.isFinite(alpha) ? clamp(alpha, 0, 1) : 1;
    if (Math.abs(next - this.categoryAlpha) <= 1e-12) return;
    this.categoryAlpha = next;
    if (this.gl) {
      this.uploadPalette();
    }
  }

  setInteractionStyle(style: InteractionStyle): void {
    if (typeof style.selectionColor === "string" && style.selectionColor.length > 0) {
      this.interactionStyle.selectionColor = style.selectionColor;
    }
    if (typeof style.selectionRadiusOffset === "number" && Number.isFinite(style.selectionRadiusOffset)) {
      this.interactionStyle.selectionRadiusOffset = Math.max(0, style.selectionRadiusOffset);
    }
    if (typeof style.selectionRingWidth === "number" && Number.isFinite(style.selectionRingWidth)) {
      this.interactionStyle.selectionRingWidth = Math.max(0.5, style.selectionRingWidth);
    }
    if (typeof style.highlightColor === "string" && style.highlightColor.length > 0) {
      this.interactionStyle.highlightColor = style.highlightColor;
    }
    if (typeof style.highlightRadiusOffset === "number" && Number.isFinite(style.highlightRadiusOffset)) {
      this.interactionStyle.highlightRadiusOffset = Math.max(0, style.highlightRadiusOffset);
    }
    if (typeof style.highlightRingWidth === "number" && Number.isFinite(style.highlightRingWidth)) {
      this.interactionStyle.highlightRingWidth = Math.max(0.5, style.highlightRingWidth);
    }
    if (typeof style.hoverColor === "string" && style.hoverColor.length > 0) {
      this.interactionStyle.hoverColor = style.hoverColor;
    }
    if (Object.prototype.hasOwnProperty.call(style, "hoverFillColor")) {
      this.interactionStyle.hoverFillColor = style.hoverFillColor ?? null;
    }
    if (typeof style.hoverRadiusOffset === "number" && Number.isFinite(style.hoverRadiusOffset)) {
      this.interactionStyle.hoverRadiusOffset = Math.max(0, style.hoverRadiusOffset);
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

  setView(view: OrbitViewState3D): void {
    this.view = { ...view };
    this.projectedDirty = true;
  }

  getView(): OrbitViewState3D {
    return { ...this.view };
  }

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
    if (this.canvas) {
      setCanvasSize(this.canvas, width, height, this.dpr);
    }
    this.syncOverlayCanvas();
    if (this.gl && this.canvas) {
      this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    }
    this.projectedDirty = true;
  }

  setSelection(indices: Set<number> | null): void {
    // Keep selection state encapsulated: never retain caller-owned Set references.
    this.selection = new Set(indices ?? new Set<number>());
    this.selectionDirty = true;
    this.highlightDirty = true;
  }

  getSelection(): Set<number> {
    return new Set(this.selection);
  }

  setHighlight(indices: Set<number> | null): void {
    this.highlight = new Set(indices ?? new Set<number>());
    this.highlightDirty = true;
  }

  getHighlight(): Set<number> {
    return new Set(this.highlight);
  }

  setHovered(index: number): void {
    if (index >= 0 && !this.isPointVisibleByCategory(index)) {
      this.hoveredIndex = -1;
    } else {
      this.hoveredIndex = index;
    }
    this.hoverDirty = true;
  }

  setLassoPolygon(polygon: Float32Array | null, style?: LassoStyle): void {
    if (style) {
      if (typeof style.strokeColor === "string" && style.strokeColor.length > 0) {
        this.lassoStyle.strokeColor = style.strokeColor;
      }
      if (typeof style.fillColor === "string" && style.fillColor.length > 0) {
        this.lassoStyle.fillColor = style.fillColor;
      }
      if (typeof style.strokeWidth === "number" && Number.isFinite(style.strokeWidth)) {
        this.lassoStyle.strokeWidth = Math.max(1, style.strokeWidth);
      }
    }

    this.lassoPolygon = polygon && polygon.length >= 6 ? new Float32Array(polygon) : null;
    this.ensureOverlayCanvas();
  }

  pan(deltaX: number, deltaY: number, modifiers: Modifiers3D): void {
    if (modifiers.meta || modifiers.ctrl || modifiers.alt) {
      const { right, up } = this.getCameraFrame();
      const scale = (2 * this.view.orthoScale) / Math.max(1, Math.min(this.width, this.height));
      const tx = -deltaX * scale;
      const ty = deltaY * scale;
      this.view.targetX += right[0] * tx + up[0] * ty;
      this.view.targetY += right[1] * tx + up[1] * ty;
      this.view.targetZ += right[2] * tx + up[2] * ty;
    } else {
      this.view.yaw -= deltaX * 0.005;
      this.view.pitch = clamp(this.view.pitch - deltaY * 0.005, -1.45, 1.45);
    }
    this.projectedDirty = true;
  }

  zoom(_anchorX: number, _anchorY: number, delta: number, _modifiers: Modifiers3D): void {
    const factor = Math.pow(1.1, -delta);
    this.view.orthoScale = clamp(this.view.orthoScale * factor, 0.02, 10000);
    this.view.distance = clamp(this.view.distance * factor, 0.05, 10000);
    this.projectedDirty = true;
  }

  hitTest(screenX: number, screenY: number): HitResult3D | null {
    const ds = this.dataset;
    if (!ds) return null;
    this.ensureProjectedCache();

    const maxDist = this.pointRadiusCss + 5;
    const maxDistSq = maxDist * maxDist;

    let bestIndex = -1;
    let bestDistSq = Infinity;
    let bestDepth = Infinity;

    for (let k = 0; k < this.projectedVisibleCount; k++) {
      const i = this.projectedVisibleIndices[k];
      if (!this.isCategoryVisible(ds.labels[i])) continue;

      const dx = this.projectedScreenX[i] - screenX;
      const dy = this.projectedScreenY[i] - screenY;
      const distSq = dx * dx + dy * dy;
      if (distSq > maxDistSq) continue;

      const depth = this.projectedDepth[i];
      if (
        distSq < bestDistSq ||
        (Math.abs(distSq - bestDistSq) <= 1e-12 && depth < bestDepth) ||
        (Math.abs(distSq - bestDistSq) <= 1e-12 && Math.abs(depth - bestDepth) <= 1e-12 && i < bestIndex)
      ) {
        bestIndex = i;
        bestDistSq = distSq;
        bestDepth = depth;
      }
    }

    if (bestIndex < 0) return null;

    return {
      index: bestIndex,
      screenX: this.projectedScreenX[bestIndex],
      screenY: this.projectedScreenY[bestIndex],
      distance: Math.sqrt(bestDistSq),
      depth: bestDepth,
    };
  }

  lassoSelect(polyline: Float32Array): SelectionResult3D {
    const ds = this.dataset;
    if (!ds || polyline.length < 6) {
      return createSelectionResult(new Set<number>(), 0);
    }

    const startTime = performance.now();
    this.ensureProjectedCache();

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (let i = 0; i < polyline.length; i += 2) {
      const x = polyline[i];
      const y = polyline[i + 1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }

    const indices = new Set<number>();
    for (let k = 0; k < this.projectedVisibleCount; k++) {
      const i = this.projectedVisibleIndices[k];
      if (!this.isCategoryVisible(ds.labels[i])) continue;
      const x = this.projectedScreenX[i];
      const y = this.projectedScreenY[i];
      if (x < minX || x > maxX || y < minY || y > maxY) continue;
      if (pointInPolygon(x, y, polyline)) {
        indices.add(i);
      }
    }

    return createSelectionResult(indices, performance.now() - startTime);
  }

  async countSelection(result: SelectionResult3D, opts: CountSelectionOptions = {}): Promise<number> {
    const indices = result.indices;
    if (!indices) return 0;

    const shouldCancel = opts.shouldCancel;
    const onProgress = opts.onProgress;
    const yieldEveryMs =
      typeof opts.yieldEveryMs === "number" && Number.isFinite(opts.yieldEveryMs)
        ? Math.max(0, opts.yieldEveryMs)
        : 8;

    let visibleCount = 0;
    let processed = 0;

    const CHECK_STRIDE = 16_384;
    let nextCheck = CHECK_STRIDE;
    let lastYieldTs = yieldEveryMs > 0 ? performance.now() : 0;

    for (const i of indices) {
      if (this.isPointVisibleByCategory(i)) visibleCount++;
      processed++;

      if (yieldEveryMs > 0 && processed >= nextCheck) {
        nextCheck = processed + CHECK_STRIDE;

        if (shouldCancel?.()) return visibleCount;

        const now = performance.now();
        if (now - lastYieldTs >= yieldEveryMs) {
          onProgress?.(visibleCount, processed);
          await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
          lastYieldTs = performance.now();
        }
      }
    }

    onProgress?.(visibleCount, processed);
    return visibleCount;
  }

  projectToScreen(dataX: number, dataY: number, dataZ: number): ProjectedPoint3D {
    this.updateMvpMatrix();

    const [clipX, clipY, clipZ, clipW] = transformClip(this.mvpMatrix, dataX, dataY, dataZ);
    const invW = Math.abs(clipW) > 1e-12 ? 1 / clipW : 0;
    const ndcX = clipX * invW;
    const ndcY = clipY * invW;
    const ndcZ = clipZ * invW;

    const x = (ndcX * 0.5 + 0.5) * this.width;
    const y = (1 - (ndcY * 0.5 + 0.5)) * this.height;
    const depth = ndcZ * 0.5 + 0.5;

    return {
      x,
      y,
      depth,
      visible: ndcX >= -1 && ndcX <= 1 && ndcY >= -1 && ndcY <= 1 && ndcZ >= -1 && ndcZ <= 1,
    };
  }

  render(): void {
    this.ensureGL();

    const gl = this.gl;
    const ds = this.dataset;
    const canvas = this.canvas;
    if (!gl || !ds || !canvas || !this.pointsProgram || !this.solidProgram) return;

    if (this.selectionDirty) this.uploadSelectionToGPU();
    if (this.highlightDirty) this.uploadHighlightToGPU();
    if (this.hoverDirty) this.uploadHoverToGPU();
    this.updateMvpMatrix();

    const [br, bg, bb, ba] = parseHexColor(this.backgroundColor);
    gl.clearColor(br, bg, bb, ba);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    if (this.supportsSphereGuide()) {
      this.drawSphereGuide();
    } else if (this.supportsEuclideanGuide()) {
      this.drawEuclideanGuide();
    }

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    gl.useProgram(this.pointsProgram.program);
    if (this.pointsProgram.uMvp) gl.uniformMatrix4fv(this.pointsProgram.uMvp, false, this.mvpMatrix);
    if (this.pointsProgram.uPointSizePx) gl.uniform1f(this.pointsProgram.uPointSizePx, this.pointRadiusCss * 2 * this.dpr);

    this.bindPaletteTexture();
    if (this.pointsProgram.uPaletteTex) gl.uniform1i(this.pointsProgram.uPaletteTex, this.paletteTexUnit);
    if (this.pointsProgram.uPaletteSize) gl.uniform1i(this.pointsProgram.uPaletteSize, this.paletteSize);
    if (this.pointsProgram.uPaletteWidth) gl.uniform1i(this.pointsProgram.uPaletteWidth, this.paletteWidth);

    gl.bindVertexArray(this.pointsVao);
    gl.drawArrays(gl.POINTS, 0, ds.n);
    gl.bindVertexArray(null);

    if (this.highlightVertexCount > 0) {
      this.drawEmphasisPoints(
        this.highlightVao,
        this.highlightVertexCount,
        this.interactionStyle.highlightColor,
        this.interactionStyle.highlightRadiusOffset,
        this.interactionStyle.highlightRingWidth,
      );
    }

    if (this.selectionVertexCount > 0) {
      this.drawEmphasisPoints(
        this.selectionVao,
        this.selectionVertexCount,
        this.interactionStyle.selectionColor,
        this.interactionStyle.selectionRadiusOffset,
        this.interactionStyle.selectionRingWidth,
      );
    }

    if (this.hoverVertexCount > 0) {
      this.drawSolidPoints(
        this.hoverVao,
        this.hoverVertexCount,
        this.interactionStyle.hoverColor,
        this.pointRadiusCss + this.interactionStyle.hoverRadiusOffset,
        true,
      );
      if (this.selection.has(this.hoveredIndex)) {
        this.drawEmphasisPoints(
          this.hoverVao,
          this.hoverVertexCount,
          this.interactionStyle.selectionColor,
          this.interactionStyle.selectionRadiusOffset,
          this.interactionStyle.selectionRingWidth,
        );
      } else if (this.highlight.has(this.hoveredIndex)) {
        this.drawEmphasisPoints(
          this.hoverVao,
          this.hoverVertexCount,
          this.interactionStyle.highlightColor,
          this.interactionStyle.highlightRadiusOffset,
          this.interactionStyle.highlightRingWidth,
        );
      } else {
        const hoverColor = this.interactionStyle.hoverFillColor ?? this.resolveLabelColor(this.hoveredIndex);
        if (this.interactionStyle.hoverFillColor) {
          this.drawSolidPoints(this.hoverVao, this.hoverVertexCount, hoverColor, this.pointRadiusCss + 1, false);
        } else {
          this.drawPalettePoints(this.hoverVao, this.hoverVertexCount, this.pointRadiusCss + 1);
        }
      }
    }
  }

  destroy(): void {
    const gl = this.gl;
    if (gl) {
      if (this.pointsProgram) gl.deleteProgram(this.pointsProgram.program);
      if (this.solidProgram) gl.deleteProgram(this.solidProgram.program);
      if (this.guideProgram) gl.deleteProgram(this.guideProgram.program);

      if (this.pointsVao) gl.deleteVertexArray(this.pointsVao);
      if (this.selectionVao) gl.deleteVertexArray(this.selectionVao);
      if (this.highlightVao) gl.deleteVertexArray(this.highlightVao);
      if (this.hoverVao) gl.deleteVertexArray(this.hoverVao);
      if (this.guideVao) gl.deleteVertexArray(this.guideVao);

      if (this.pointsPosBuffer) gl.deleteBuffer(this.pointsPosBuffer);
      if (this.pointsLabelBuffer) gl.deleteBuffer(this.pointsLabelBuffer);
      if (this.selectionPosBuffer) gl.deleteBuffer(this.selectionPosBuffer);
      if (this.selectionLabelBuffer) gl.deleteBuffer(this.selectionLabelBuffer);
      if (this.highlightPosBuffer) gl.deleteBuffer(this.highlightPosBuffer);
      if (this.highlightLabelBuffer) gl.deleteBuffer(this.highlightLabelBuffer);
      if (this.hoverPosBuffer) gl.deleteBuffer(this.hoverPosBuffer);
      if (this.hoverLabelBuffer) gl.deleteBuffer(this.hoverLabelBuffer);
      if (this.guideBuffer) gl.deleteBuffer(this.guideBuffer);

      if (this.paletteTex) gl.deleteTexture(this.paletteTex);
    }

    this.gl = null;
    this.pointsProgram = null;
    this.solidProgram = null;
    this.guideProgram = null;

    this.pointsVao = null;
    this.selectionVao = null;
    this.highlightVao = null;
    this.hoverVao = null;
    this.guideVao = null;

    this.pointsPosBuffer = null;
    this.pointsLabelBuffer = null;
    this.selectionPosBuffer = null;
    this.selectionLabelBuffer = null;
    this.highlightPosBuffer = null;
    this.highlightLabelBuffer = null;
    this.hoverPosBuffer = null;
    this.hoverLabelBuffer = null;
    this.guideBuffer = null;

    this.paletteTex = null;
    this.removeOverlayCanvas();
  }

  protected ensureGL(): void {
    if (this.gl) return;
    if (!this.canvas) throw new Error("Renderer not initialized");

    const gl = this.canvas.getContext("webgl2", {
      antialias: false,
      alpha: false,
      depth: true,
      stencil: false,
      preserveDrawingBuffer: false,
      premultipliedAlpha: false,
      desynchronized: true,
    } as WebGLContextAttributes);

    if (!gl) {
      throw new Error("Failed to get WebGL2 context");
    }

    this.gl = gl;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);

    this.createProgramsAndBuffers();
    this.uploadPalette();

    if (this.dataset) {
      this.uploadDatasetToGPU();
      this.rebuildGuideGeometry();
      this.uploadSelectionToGPU();
      this.uploadHighlightToGPU();
      this.uploadHoverToGPU();
    }

    this.projectedDirty = true;
  }

  protected createProgramsAndBuffers(): void {
    const gl = this.gl;
    if (!gl) return;

    const pointsProgram = linkProgram(gl, VS_POINTS_3D, FS_POINTS_3D);
    const solidProgram = linkProgram(gl, VS_SOLID_3D, FS_SOLID_3D);
    const guideProgram = linkProgram(gl, VS_GUIDE_3D, FS_GUIDE_3D);

    this.pointsProgram = {
      program: pointsProgram,
      uMvp: gl.getUniformLocation(pointsProgram, "u_mvp"),
      uPointSizePx: gl.getUniformLocation(pointsProgram, "u_pointSizePx"),
      uPaletteTex: gl.getUniformLocation(pointsProgram, "u_paletteTex"),
      uPaletteSize: gl.getUniformLocation(pointsProgram, "u_paletteSize"),
      uPaletteWidth: gl.getUniformLocation(pointsProgram, "u_paletteWidth"),
    };

    this.solidProgram = {
      program: solidProgram,
      uMvp: gl.getUniformLocation(solidProgram, "u_mvp"),
      uPointSizePx: gl.getUniformLocation(solidProgram, "u_pointSizePx"),
      uColor: gl.getUniformLocation(solidProgram, "u_color"),
      uRingMode: gl.getUniformLocation(solidProgram, "u_ringMode"),
      uRingThicknessPx: gl.getUniformLocation(solidProgram, "u_ringThicknessPx"),
    };

    this.guideProgram = {
      program: guideProgram,
      uMvp: gl.getUniformLocation(guideProgram, "u_mvp"),
      uColor: gl.getUniformLocation(guideProgram, "u_color"),
    };

    this.pointsVao = gl.createVertexArray();
    this.pointsPosBuffer = gl.createBuffer();
    this.pointsLabelBuffer = gl.createBuffer();

    this.selectionVao = gl.createVertexArray();
    this.selectionPosBuffer = gl.createBuffer();
    this.selectionLabelBuffer = gl.createBuffer();

    this.highlightVao = gl.createVertexArray();
    this.highlightPosBuffer = gl.createBuffer();
    this.highlightLabelBuffer = gl.createBuffer();

    this.hoverVao = gl.createVertexArray();
    this.hoverPosBuffer = gl.createBuffer();
    this.hoverLabelBuffer = gl.createBuffer();

    this.guideVao = gl.createVertexArray();
    this.guideBuffer = gl.createBuffer();

    if (
      !this.pointsVao ||
      !this.pointsPosBuffer ||
      !this.pointsLabelBuffer ||
      !this.selectionVao ||
      !this.selectionPosBuffer ||
      !this.selectionLabelBuffer ||
      !this.highlightVao ||
      !this.highlightPosBuffer ||
      !this.highlightLabelBuffer ||
      !this.hoverVao ||
      !this.hoverPosBuffer ||
      !this.hoverLabelBuffer ||
      !this.guideVao ||
      !this.guideBuffer
    ) {
      throw new Error("Failed to allocate WebGL buffers for 3D renderer");
    }

    gl.bindVertexArray(this.pointsVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.pointsPosBuffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.pointsLabelBuffer);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_SHORT, 0, 0);
    gl.bindVertexArray(null);

    gl.bindVertexArray(this.selectionVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.selectionPosBuffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.selectionLabelBuffer);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_SHORT, 0, 0);
    gl.bindVertexArray(null);

    gl.bindVertexArray(this.highlightVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.highlightPosBuffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.highlightLabelBuffer);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_SHORT, 0, 0);
    gl.bindVertexArray(null);

    gl.bindVertexArray(this.hoverVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.hoverPosBuffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.hoverLabelBuffer);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_SHORT, 0, 0);
    gl.bindVertexArray(null);

    gl.bindVertexArray(this.guideVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.guideBuffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  protected uploadPalette(): void {
    const gl = this.gl;
    if (!gl || !this.pointsProgram) return;

    const rawSize = this.colors.length;
    const size = Math.max(1, Math.min(rawSize, 0xffff + 1));

    const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    const texW = Math.min(maxTex, size);
    const texH = Math.ceil(size / texW);

    const capacity = texW * texH;
    if (this.paletteBytes.length !== capacity * 4) {
      this.paletteBytes = new Uint8Array(capacity * 4);
    } else {
      this.paletteBytes.fill(0);
    }

    if (rawSize === 0) {
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
        this.paletteBytes[o] = r;
        this.paletteBytes[o + 1] = g;
        this.paletteBytes[o + 2] = b;
        this.paletteBytes[o + 3] = alpha;
      }
    }

    if (!this.paletteTex) {
      this.paletteTex = gl.createTexture();
      if (!this.paletteTex) throw new Error("Failed to create palette texture");
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
    this.paletteWidth = texW;
    this.paletteHeight = texH;

    gl.activeTexture(gl.TEXTURE0 + this.paletteTexUnit);
    gl.bindTexture(gl.TEXTURE_2D, this.paletteTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, texW, texH, 0, gl.RGBA, gl.UNSIGNED_BYTE, this.paletteBytes);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE0);
  }

  protected bindPaletteTexture(): void {
    const gl = this.gl;
    if (!gl || !this.paletteTex) return;
    gl.activeTexture(gl.TEXTURE0 + this.paletteTexUnit);
    gl.bindTexture(gl.TEXTURE_2D, this.paletteTex);
    gl.activeTexture(gl.TEXTURE0);
  }

  protected uploadDatasetToGPU(): void {
    const gl = this.gl;
    const ds = this.dataset;
    if (!gl || !ds || !this.pointsVao || !this.pointsPosBuffer || !this.pointsLabelBuffer) return;

    gl.bindVertexArray(this.pointsVao);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.pointsPosBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, ds.positions, gl.STATIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.pointsLabelBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, ds.labels, gl.STATIC_DRAW);

    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  protected drawSolidPoints(
    vao: WebGLVertexArrayObject | null,
    count: number,
    color: string,
    radiusCss: number,
    ring: boolean,
    ringWidthPx = 2,
  ): void {
    const gl = this.gl;
    const program = this.solidProgram;
    if (!gl || !program || !vao || count <= 0) return;

    const [r, g, b, a] = parseHexColor(color);
    gl.useProgram(program.program);
    if (program.uMvp) gl.uniformMatrix4fv(program.uMvp, false, this.mvpMatrix);
    if (program.uPointSizePx) gl.uniform1f(program.uPointSizePx, radiusCss * 2 * this.dpr);
    if (program.uColor) gl.uniform4f(program.uColor, r, g, b, a);
    if (program.uRingMode) gl.uniform1i(program.uRingMode, ring ? 1 : 0);
    if (program.uRingThicknessPx) gl.uniform1f(program.uRingThicknessPx, ring ? Math.max(0.5, ringWidthPx) : 0);

    gl.bindVertexArray(vao);
    gl.drawArrays(gl.POINTS, 0, count);
    gl.bindVertexArray(null);
  }

  protected drawPalettePoints(
    vao: WebGLVertexArrayObject | null,
    count: number,
    radiusCss: number,
  ): void {
    const gl = this.gl;
    const program = this.pointsProgram;
    if (!gl || !program || !vao || count <= 0) return;

    gl.useProgram(program.program);
    if (program.uMvp) gl.uniformMatrix4fv(program.uMvp, false, this.mvpMatrix);
    if (program.uPointSizePx) gl.uniform1f(program.uPointSizePx, radiusCss * 2 * this.dpr);
    this.bindPaletteTexture();
    if (program.uPaletteTex) gl.uniform1i(program.uPaletteTex, this.paletteTexUnit);
    if (program.uPaletteSize) gl.uniform1i(program.uPaletteSize, this.paletteSize);
    if (program.uPaletteWidth) gl.uniform1i(program.uPaletteWidth, this.paletteWidth);

    gl.bindVertexArray(vao);
    gl.drawArrays(gl.POINTS, 0, count);
    gl.bindVertexArray(null);
  }

  protected drawEmphasisPoints(
    vao: WebGLVertexArrayObject | null,
    count: number,
    color: string,
    radiusOffset: number,
    ringWidth: number,
  ): void {
    if (!vao || count <= 0) return;

    this.drawSolidPoints(
      vao,
      count,
      color,
      this.pointRadiusCss + Math.max(0, radiusOffset),
      true,
      ringWidth,
    );

    this.drawPalettePoints(vao, count, this.pointRadiusCss);
  }

  protected drawSphereGuide(): void {
    const gl = this.gl;
    const program = this.guideProgram;
    if (!gl || !program || !this.guideVao || this.guideVertexCount === 0) return;

    const [r, g, b] = parseHexColor(this.sphereGuideColor);

    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    gl.useProgram(program.program);
    if (program.uMvp) gl.uniformMatrix4fv(program.uMvp, false, this.mvpMatrix);
    if (program.uColor) gl.uniform4f(program.uColor, r, g, b, this.sphereGuideOpacity);

    gl.bindVertexArray(this.guideVao);
    const seg = this.guideSegmentVerts;
    gl.drawArrays(gl.LINE_STRIP, 0, seg);
    gl.drawArrays(gl.LINE_STRIP, seg, seg);
    gl.drawArrays(gl.LINE_STRIP, seg * 2, seg);
    gl.bindVertexArray(null);

    gl.enable(gl.DEPTH_TEST);
  }

  protected drawEuclideanGuide(): void {
    const gl = this.gl;
    const program = this.guideProgram;
    if (!gl || !program || !this.guideVao || this.guideAxisVertexCount === 0) return;

    const [r, g, b] = parseHexColor(this.sphereGuideColor);
    const axisOpacity = clamp(this.sphereGuideOpacity * 0.8, 0.06, 0.18);

    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(false);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    gl.useProgram(program.program);
    if (program.uMvp) gl.uniformMatrix4fv(program.uMvp, false, this.mvpMatrix);
    if (program.uColor) gl.uniform4f(program.uColor, r, g, b, axisOpacity);

    gl.bindVertexArray(this.guideVao);
    gl.drawArrays(gl.LINES, this.guideAxisVertexOffset, this.guideAxisVertexCount);
    gl.bindVertexArray(null);
    gl.depthMask(true);
  }

  protected uploadSelectionToGPU(): void {
    const gl = this.gl;
    const ds = this.dataset;
    if (!gl || !ds || !this.selectionPosBuffer || !this.selectionLabelBuffer) return;

    if (this.selection.size === 0) {
      this.selectionVertexCount = 0;
      this.selectionDirty = false;
      return;
    }

    const renderCount = Math.min(this.selection.size, MAX_SELECTION_RENDER_POINTS);
    if (this.selectionPositionsScratch.length < renderCount * 3) {
      this.selectionPositionsScratch = new Float32Array(renderCount * 3);
    }
    if (this.selectionLabelsScratch.length < renderCount) {
      this.selectionLabelsScratch = new Uint16Array(renderCount);
    }

    let k = 0;
    for (const i of this.selection) {
      if (i < 0 || i >= ds.n) continue;
      if (!this.isCategoryVisible(ds.labels[i])) continue;
      this.selectionPositionsScratch[k * 3] = ds.positions[i * 3];
      this.selectionPositionsScratch[k * 3 + 1] = ds.positions[i * 3 + 1];
      this.selectionPositionsScratch[k * 3 + 2] = ds.positions[i * 3 + 2];
      this.selectionLabelsScratch[k] = ds.labels[i];
      k++;
      if (k >= renderCount) break;
    }

    this.selectionVertexCount = k;

    gl.bindBuffer(gl.ARRAY_BUFFER, this.selectionPosBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.selectionPositionsScratch.subarray(0, k * 3), gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.selectionLabelBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.selectionLabelsScratch.subarray(0, k), gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    this.selectionDirty = false;
  }

  protected uploadHighlightToGPU(): void {
    const gl = this.gl;
    const ds = this.dataset;
    if (!gl || !ds || !this.highlightPosBuffer || !this.highlightLabelBuffer) return;

    if (this.highlight.size === 0) {
      this.highlightVertexCount = 0;
      this.highlightDirty = false;
      return;
    }

    const renderCount = Math.min(this.highlight.size, MAX_SELECTION_RENDER_POINTS);
    if (this.highlightPositionsScratch.length < renderCount * 3) {
      this.highlightPositionsScratch = new Float32Array(renderCount * 3);
    }
    if (this.highlightLabelsScratch.length < renderCount) {
      this.highlightLabelsScratch = new Uint16Array(renderCount);
    }

    let k = 0;
    for (const i of this.highlight) {
      if (this.selection.has(i)) continue;
      if (i < 0 || i >= ds.n) continue;
      if (!this.isCategoryVisible(ds.labels[i])) continue;
      this.highlightPositionsScratch[k * 3] = ds.positions[i * 3];
      this.highlightPositionsScratch[k * 3 + 1] = ds.positions[i * 3 + 1];
      this.highlightPositionsScratch[k * 3 + 2] = ds.positions[i * 3 + 2];
      this.highlightLabelsScratch[k] = ds.labels[i];
      k++;
      if (k >= renderCount) break;
    }

    this.highlightVertexCount = k;

    gl.bindBuffer(gl.ARRAY_BUFFER, this.highlightPosBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.highlightPositionsScratch.subarray(0, k * 3), gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.highlightLabelBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.highlightLabelsScratch.subarray(0, k), gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    this.highlightDirty = false;
  }

  protected uploadHoverToGPU(): void {
    const gl = this.gl;
    const ds = this.dataset;
    if (!gl || !ds || !this.hoverPosBuffer || !this.hoverLabelBuffer) return;

    const i = this.hoveredIndex;
    if (i < 0 || i >= ds.n || !this.isCategoryVisible(ds.labels[i])) {
      this.hoverVertexCount = 0;
      this.hoverDirty = false;
      return;
    }

    this.hoverPositionScratch[0] = ds.positions[i * 3];
    this.hoverPositionScratch[1] = ds.positions[i * 3 + 1];
    this.hoverPositionScratch[2] = ds.positions[i * 3 + 2];
    this.hoverLabelScratch[0] = ds.labels[i];

    gl.bindBuffer(gl.ARRAY_BUFFER, this.hoverPosBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.hoverPositionScratch, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.hoverLabelBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.hoverLabelScratch, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    this.hoverVertexCount = 1;
    this.hoverDirty = false;
  }

  protected fitViewToDataset(): void {
    const ds = this.dataset;
    if (!ds || ds.n === 0) return;

    let centerX = 0;
    let centerY = 0;
    let centerZ = 0;

    for (let i = 0; i < ds.n; i++) {
      centerX += ds.positions[i * 3];
      centerY += ds.positions[i * 3 + 1];
      centerZ += ds.positions[i * 3 + 2];
    }

    centerX /= ds.n;
    centerY /= ds.n;
    centerZ /= ds.n;

    let radius = 0;
    for (let i = 0; i < ds.n; i++) {
      const dx = ds.positions[i * 3] - centerX;
      const dy = ds.positions[i * 3 + 1] - centerY;
      const dz = ds.positions[i * 3 + 2] - centerZ;
      radius = Math.max(radius, Math.hypot(dx, dy, dz));
    }

    this.sceneRadius = Math.max(radius, 0.25);

    this.view = {
      type: "orbit3d",
      yaw: 0.7,
      pitch: 0.35,
      distance: Math.max(this.sceneRadius * 3.0, 1.5),
      targetX: centerX,
      targetY: centerY,
      targetZ: centerZ,
      orthoScale: Math.max(this.sceneRadius * 1.4, 0.75),
    };

    this.projectedDirty = true;
  }

  protected getCameraFrame(): { eye: Vec3; target: Vec3; right: Vec3; up: Vec3; forward: Vec3 } {
    const cp = Math.cos(this.view.pitch);
    const sp = Math.sin(this.view.pitch);
    const cy = Math.cos(this.view.yaw);
    const sy = Math.sin(this.view.yaw);

    const dirTargetToEye: Vec3 = [cp * sy, sp, cp * cy];
    const target: Vec3 = [this.view.targetX, this.view.targetY, this.view.targetZ];
    const eye = vec3Add(target, vec3Scale(dirTargetToEye, this.view.distance));

    let forward = vec3Normalize(vec3Sub(target, eye));
    if (vec3Length(forward) < 1e-9) {
      forward = [0, 0, -1];
    }

    let right = vec3Normalize(vec3Cross(forward, WORLD_UP));
    if (vec3Length(right) < 1e-9) {
      right = [1, 0, 0];
    }

    const up = vec3Normalize(vec3Cross(right, forward));

    return { eye, target, right, up, forward };
  }

  protected updateMvpMatrix(): void {
    const { eye, target, up } = this.getCameraFrame();

    const aspect = this.width > 0 && this.height > 0 ? this.width / this.height : 1;
    const halfH = Math.max(0.01, this.view.orthoScale);
    const halfW = halfH * aspect;

    const near = 0.01;
    const far = Math.max(near + 1, this.view.distance + this.sceneRadius * 6 + 10);

    const view = mat4LookAt(eye, target, up);
    const proj = mat4Ortho(-halfW, halfW, -halfH, halfH, near, far);
    this.mvpMatrix = mat4Multiply(proj, view);
  }

  protected ensureProjectedCapacity(n: number): void {
    if (this.projectedScreenX.length !== n) this.projectedScreenX = new Float32Array(n);
    if (this.projectedScreenY.length !== n) this.projectedScreenY = new Float32Array(n);
    if (this.projectedDepth.length !== n) this.projectedDepth = new Float32Array(n);
    if (this.projectedPixelIndex.length !== n) this.projectedPixelIndex = new Int32Array(n);
    if (this.projectedVisible.length !== n) this.projectedVisible = new Uint8Array(n);
    if (this.projectedVisibleIndices.length !== n) this.projectedVisibleIndices = new Uint32Array(n);

    const pixelCount = Math.max(1, this.width * this.height);
    if (this.depthBuffer.length !== pixelCount) {
      this.depthBuffer = new Float32Array(pixelCount);
    }
  }

  protected ensureProjectedCache(): void {
    if (!this.projectedDirty) return;

    const ds = this.dataset;
    if (!ds) return;

    this.updateMvpMatrix();
    this.ensureProjectedCapacity(ds.n);

    this.depthBuffer.fill(Number.POSITIVE_INFINITY);
    this.projectedVisibleCount = 0;

    const w = Math.max(1, this.width);
    const h = Math.max(1, this.height);
    const useVisibilityMask = this.hasCategoryVisibilityMask;
    const visibilityMask = this.categoryVisibilityMask;

    for (let i = 0; i < ds.n; i++) {
      if (useVisibilityMask) {
        const label = ds.labels[i];
        if (label < visibilityMask.length && visibilityMask[label] === 0) {
          this.projectedPixelIndex[i] = -1;
          this.projectedVisible[i] = 0;
          continue;
        }
      }

      const x = ds.positions[i * 3];
      const y = ds.positions[i * 3 + 1];
      const z = ds.positions[i * 3 + 2];

      const [clipX, clipY, clipZ, clipW] = transformClip(this.mvpMatrix, x, y, z);
      const invW = Math.abs(clipW) > 1e-12 ? 1 / clipW : 0;
      const ndcX = clipX * invW;
      const ndcY = clipY * invW;
      const ndcZ = clipZ * invW;

      if (ndcX < -1 || ndcX > 1 || ndcY < -1 || ndcY > 1 || ndcZ < -1 || ndcZ > 1) {
        this.projectedPixelIndex[i] = -1;
        this.projectedVisible[i] = 0;
        continue;
      }

      const sx = (ndcX * 0.5 + 0.5) * w;
      const sy = (1 - (ndcY * 0.5 + 0.5)) * h;
      const depth = ndcZ * 0.5 + 0.5;

      const ix = Math.floor(sx);
      const iy = Math.floor(sy);
      if (ix < 0 || ix >= w || iy < 0 || iy >= h) {
        this.projectedPixelIndex[i] = -1;
        this.projectedVisible[i] = 0;
        continue;
      }

      const p = iy * w + ix;

      this.projectedScreenX[i] = sx;
      this.projectedScreenY[i] = sy;
      this.projectedDepth[i] = depth;
      this.projectedPixelIndex[i] = p;
      this.projectedVisible[i] = 0;

      if (depth < this.depthBuffer[p]) {
        this.depthBuffer[p] = depth;
      }
    }

    for (let i = 0; i < ds.n; i++) {
      const p = this.projectedPixelIndex[i];
      if (p < 0) continue;
      const depth = this.projectedDepth[i];
      if (depth <= this.depthBuffer[p] + 1e-4) {
        this.projectedVisible[i] = 1;
        this.projectedVisibleIndices[this.projectedVisibleCount++] = i;
      }
    }

    this.projectedDirty = false;
  }

  protected rebuildGuideGeometry(): void {
    const gl = this.gl;
    if (!gl || !this.guideBuffer) return;

    this.guideVertexCount = 0;
    this.guideSegmentVerts = 0;
    this.guideAxisVertexOffset = 0;
    this.guideAxisVertexCount = 0;

    if (this.supportsSphereGuide()) {
      const radius = this.estimateGuideRadius();
      const segments = 128;
      const vertsPerCircle = segments + 1;
      const out = new Float32Array(vertsPerCircle * 3 * 3);

      let k = 0;
      for (let i = 0; i <= segments; i++) {
        const a = (i / segments) * Math.PI * 2;
        out[k++] = radius * Math.cos(a);
        out[k++] = radius * Math.sin(a);
        out[k++] = 0;
      }
      for (let i = 0; i <= segments; i++) {
        const a = (i / segments) * Math.PI * 2;
        out[k++] = radius * Math.cos(a);
        out[k++] = 0;
        out[k++] = radius * Math.sin(a);
      }
      for (let i = 0; i <= segments; i++) {
        const a = (i / segments) * Math.PI * 2;
        out[k++] = 0;
        out[k++] = radius * Math.cos(a);
        out[k++] = radius * Math.sin(a);
      }

      gl.bindBuffer(gl.ARRAY_BUFFER, this.guideBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, out, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);

      this.guideVertexCount = out.length / 3;
      this.guideSegmentVerts = vertsPerCircle;
      return;
    }

    if (!this.supportsEuclideanGuide()) return;

    const ds = this.dataset;
    if (!ds || ds.n === 0) return;

    let minX = Number.POSITIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let minZ = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    let maxZ = Number.NEGATIVE_INFINITY;

    for (let i = 0; i < ds.n; i++) {
      const x = ds.positions[i * 3];
      const y = ds.positions[i * 3 + 1];
      const z = ds.positions[i * 3 + 2];
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (z < minZ) minZ = z;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      if (z > maxZ) maxZ = z;
    }

    const spanX = Math.max(0, maxX - minX);
    const spanY = Math.max(0, maxY - minY);
    const spanZ = Math.max(0, maxZ - minZ);
    const longestSpan = Math.max(spanX, spanY, spanZ, this.sceneRadius * 2, 1e-3);
    const pad = longestSpan * 0.04;
    const fallbackHalf = Math.max(longestSpan * 0.3, 0.2);
    const centerX = (minX + maxX) * 0.5;
    const centerY = (minY + maxY) * 0.5;
    const centerZ = (minZ + maxZ) * 0.5;
    const halfX = spanX > 1e-4 ? spanX * 0.5 + pad : fallbackHalf;
    const halfY = spanY > 1e-4 ? spanY * 0.5 + pad : fallbackHalf;
    const halfZ = spanZ > 1e-4 ? spanZ * 0.5 + pad : fallbackHalf;
    const out = new Float32Array(6 * 3);

    let k = 0;

    out[k++] = centerX - halfX;
    out[k++] = centerY;
    out[k++] = centerZ;
    out[k++] = centerX + halfX;
    out[k++] = centerY;
    out[k++] = centerZ;

    out[k++] = centerX;
    out[k++] = centerY - halfY;
    out[k++] = centerZ;
    out[k++] = centerX;
    out[k++] = centerY + halfY;
    out[k++] = centerZ;

    out[k++] = centerX;
    out[k++] = centerY;
    out[k++] = centerZ - halfZ;
    out[k++] = centerX;
    out[k++] = centerY;
    out[k++] = centerZ + halfZ;

    gl.bindBuffer(gl.ARRAY_BUFFER, this.guideBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, out, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    this.guideVertexCount = out.length / 3;
    this.guideAxisVertexOffset = 0;
    this.guideAxisVertexCount = 6;
  }

  protected estimateGuideRadius(): number {
    const ds = this.dataset;
    if (!ds || ds.n === 0) return 1;

    let maxNorm = 0;
    for (let i = 0; i < ds.n; i++) {
      const x = ds.positions[i * 3];
      const y = ds.positions[i * 3 + 1];
      const z = ds.positions[i * 3 + 2];
      maxNorm = Math.max(maxNorm, Math.hypot(x, y, z));
    }

    return Math.max(0.25, maxNorm);
  }

  protected resolveLabelColor(index: number): string {
    const ds = this.dataset;
    if (!ds || index < 0 || index >= ds.n || this.colors.length === 0) {
      return HOVER_COLOR;
    }

    const label = ds.labels[index];
    return this.colors[label % this.colors.length];
  }
}

export class Euclidean3DWebGLCandidate extends PointCloud3DWebGLBase {
  protected expectedGeometry(): GeometryMode3D {
    return "euclidean3d";
  }

  protected override supportsEuclideanGuide(): boolean {
    return true;
  }
}

export class Spherical3DWebGLCandidate extends PointCloud3DWebGLBase {
  protected expectedGeometry(): GeometryMode3D {
    return "sphere";
  }

  protected override supportsSphereGuide(): boolean {
    return true;
  }

  protected override preprocessPositions(input: Float32Array): Float32Array {
    const out = new Float32Array(input.length);
    for (let i = 0; i < input.length; i += 3) {
      const x = input[i];
      const y = input[i + 1];
      const z = input[i + 2];
      const norm = Math.hypot(x, y, z);
      if (norm < 1e-8) {
        out[i] = 0;
        out[i + 1] = 0;
        out[i + 2] = 1;
      } else {
        out[i] = x / norm;
        out[i + 1] = y / norm;
        out[i + 2] = z / norm;
      }
    }
    return out;
  }

  protected override fitViewToDataset(): void {
    const ds = this.dataset;
    if (!ds || ds.n === 0) return;

    let radius = 0;
    for (let i = 0; i < ds.n; i++) {
      const x = ds.positions[i * 3];
      const y = ds.positions[i * 3 + 1];
      const z = ds.positions[i * 3 + 2];
      radius = Math.max(radius, Math.hypot(x, y, z));
    }

    this.sceneRadius = Math.max(radius, 1);

    this.view = {
      type: "orbit3d",
      yaw: 0.9,
      pitch: 0.4,
      distance: Math.max(this.sceneRadius * 3.2, 2.4),
      targetX: 0,
      targetY: 0,
      targetZ: 0,
      orthoScale: Math.max(this.sceneRadius * 1.45, 1.0),
    };

    this.projectedDirty = true;
  }

  protected override estimateGuideRadius(): number {
    return 1;
  }
}
