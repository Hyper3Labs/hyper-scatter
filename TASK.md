# Embedding Visualizer (51-like) with Geometry-Aware Camera + WebGL-first Pipeline

_Last updated: 2025-12-22_

This document is intentionally **spec-like**: it should be unambiguous enough that two different implementers could build compatible systems and get the same pass/fail results.

## 0) What we are building (in plain English)

We are building an **application/component similar to FiftyOne’s embeddings panel (“51”) and Apple’s Embedding Atlas**: an interactive embedding scatterplot that stays smooth at **very large N**.

Key twist: we must support **multiple geometries**, starting with:
- **Euclidean 2D scatter**
- **Hyperbolic embeddings in the Poincaré disk**, with **correct hyperbolic navigation and selection semantics** (no “Euclidean hacks”)

We will implement this in two layers:
1) a **naive but accurate** implementation (ground truth)
2) a **high-performance WebGL implementation** (candidate)

We will iterate the candidate quickly by continuously comparing it to the reference (correctness harness) and tracking performance regressions (benchmark harness).

**Tech direction:** WebGL first, then port the candidate path to WebGPU later (keeping the same API and harness).

## 0.1) Ground rules / defaults (explicit)

These are the intended defaults based on the current direction:

- Language: **TypeScript**
- Candidate renderer: **WebGL2**
- Future: add a **WebGPU candidate** behind the same interface
- UI framework: optional (vanilla or React are both acceptable)

If we later change any of these, the API contracts and harness requirements stay the same.

This document specifies a **separate module / codebase** (a “viz lab”) whose purpose is to build and validate the embedding visualizer across multiple geometries.

The lab must contain:
- A **naive reference implementation** (correctness-first; may be slow)
- A **performance-oriented candidate implementation** (fast; must match reference)
- A **comparison + benchmarking harness** that can prove the candidate is both correct and faster

The lab must support at least two geometry modes:
- **Euclidean 2D scatter**
- **Hyperbolic Poincaré disk** scatter with **mathematically correct camera pan** + hover/picking + lasso selection

The primary requirement is objective verification: the system must be able to say, with evidence, whether the candidate is matching the reference and whether it is improving performance.

---

## 1) Implementation strategy (the order matters)

The most important part is how we’ll converge on correctness + performance.

We will implement in this order:

### Milestone 1 — Euclidean reference (naive but correct)
- Implement Euclidean rendering + interactions in the simplest way (Canvas2D is fine)
- Implement trace record/replay
- Implement correctness diffs (hover, selection set, view checkpoints)

**Exit criteria:** Euclidean reference is deterministic under trace replay.

### Milestone 2 — Euclidean candidate (WebGL, very fast)
- Implement a WebGL2 renderer and interaction pipeline for Euclidean
- Make it match the Euclidean reference under the harness

**Exit criteria:** candidate matches reference and is measurably faster on at least one large dataset.

### Milestone 3 — Hyperbolic reference (naive but correct)
- Implement Poincaré disk transforms + camera navigation semantics
- Implement hyperbolic hover + unprojected lasso selection correctly
- Validate with projection probes + trace replay

**Exit criteria:** hyperbolic reference is deterministic and “feels right” for pan/zoom (matching common embedding tools, but with correct hyperbolic math).

### Milestone 4 — Hyperbolic candidate (WebGL, very fast)
- Implement WebGL2 hyperbolic path (GPU transforms + fast picking/selection as needed)
- Make it match the hyperbolic reference under the harness

**Exit criteria:** hyperbolic candidate matches reference and meets performance targets.

### Milestone 5 — WebGPU port (later)
- Keep the same contracts/harness
- Add a WebGPU candidate path (feature-flagged), validated against the same reference

---

## 0) Background and Purpose

### What we are trying to build
We want an interactive embedding visualization component that can handle large point clouds (commonly 100k–20M+ points) and supports **multiple geometries**.

The intent is not limited to hyperbolic space. In the future, we may support additional geometry/camera models (e.g. spherical projections and other non-Euclidean navigation semantics).

### Why we need this lab
High-performance visualization pipelines (WebGL/WebGPU, GPU shaders, spatial indices, workers) are easy to get **fast** but hard to get **correct**, especially when:
- the camera transform is not Euclidean (e.g. hyperbolic pan)
- selection is defined in a different space than rendering (e.g. unprojected lasso)

This lab provides a controlled environment where:
- the naive implementation is treated as “ground truth”
- the optimized implementation is iterated until it matches the ground truth under objective checks

### Where this will be used
This lab is intended to:
- validate geometry-specific math (hyperbolic now, spherical later)
- validate interaction semantics (hover, lasso, camera)
- measure performance and regressions
- generate replayable interaction traces usable for regression testing

This is not production app code. It is a correctness + performance testbed.

---

## 1) Goals and Non-Goals

### Goals
- Provide a framework where an optimized renderer can be iterated repeatedly until:
  - it is measurably faster than the naive reference
  - it matches the reference outputs within defined tolerances
- Support both:
  - Euclidean interactions (pan, zoom, hover, lasso)
  - Hyperbolic interactions (correct hyperbolic pan, display zoom, hover, unprojected lasso)
- Make comparisons objective via:
  - PNG/screenshot outputs
  - numeric projection checks
  - interaction trace replay and result comparison

### UX goal (important)
The interaction model should feel like common embedding viewers (e.g. FiftyOne embeddings panel / Embedding Atlas):
- drag to pan
- scroll/wheel to zoom
- lasso/box selection modes
- shift to add to selection (where applicable)
- double-click to clear selection (where applicable)

### Non-Goals
- Not a production UI/UX application
- Not a dataset management system
- Not a backend service (unless added for CI automation)

---

## 2) Deliverables

### A) Two implementations per geometry
For each geometry (Euclidean and Hyperbolic), implement:
1. **Reference (naive) implementation**
2. **Candidate (optimized) implementation**

Both must implement the same interface and semantics.

### B) Comparison harness
Must support:
- Side-by-side mode (reference vs candidate)
- A/B toggle mode (swap implementations in the same viewport)
- Visual diff mode (pixel differences)
- Numeric diffs for math and interaction results

### C) Performance harness
Must compute and report:
- FPS / frame times (p50/p95/p99)
- Lasso selection compute time
- Hover/picking compute time
- End-to-end interaction latency under trace replay

### D) Replayable interaction traces
Must record and replay:
- pointer events
- wheel events
- modifier key states
- lasso polylines
- resize events
- tool/mode changes (pan vs lasso)

Also support the common “embedding viewer ergonomics” that appear in practice:
- selection add/remove modifiers (shift/cmd)
- double-click to clear
- optional reset-to-fit control

---

## 3) Suggested Project Structure

You may rename folders as desired, but keep a clean separation between:
- shared core definitions
- reference impl
- candidate impl
- runner/comparison logic

Suggested layout:

```
viz-lab/
  README.md
  src/
    core/
      types.ts
      rng.ts
      dataset.ts
      view_state.ts
      interaction_trace.ts
      perf/
        timers.ts
        stats.ts
      comparison/
        image_capture.ts
        image_diff.ts
        numeric_diff.ts
        set_diff.ts
      math/
        euclidean.ts
        poincare.ts
      selection/
        point_in_polygon.ts
    impl_reference/
      euclidean_reference.ts
      hyperbolic_reference.ts
    impl_candidate/
      euclidean_candidate.ts
      hyperbolic_candidate.ts
    runner/
      scenario_runner.ts
      trace_replay.ts
      report_writer.ts
    ui/ (optional)
      app.tsx
      panels/
```

---

## 4) Shared Contracts (Strict Interfaces)

Define a single interface in `core/types.ts` that both reference and candidate implementations must satisfy.

### 4.1 Renderer Contract
Each implementation must support:
- `init(canvas: HTMLCanvasElement, opts: InitOptions): void`
- `setDataset(dataset: Dataset): void`
- `setView(view: ViewState): void`
- `getView(): ViewState`
- `render(): void`
- `resize(width: number, height: number): void`
- `destroy(): void`

### 4.2 Interaction Contract
Both must support the same semantics:
- `startPan(screenX: number, screenY: number): void`
- `pan(deltaX: number, deltaY: number, modifiers: Modifiers): void`
- `zoom(anchorX: number, anchorY: number, wheelDelta: number, modifiers: Modifiers): void`
- `hitTest(screenX: number, screenY: number): HitResult | null`
- `lassoSelect(screenPolyline: Float32Array): SelectionResult`

**Interaction semantics (explicit, to match common tools):**
- **Pan:** call `startPan()` at pointer-down, then drag-to-pan with delta updates. Pan is **anchor-invariant**: if the user drags with the pointer at screen location $s$, the data point under $s$ should remain under $s$ throughout the drag.
  - Euclidean: this corresponds to a simple translation in data-space.
  - Hyperbolic: this must be implemented via a Poincaré-disk isometry update (not Euclidean translation).
- **Zoom:** scroll/wheel to zoom, also **anchor-invariant** around the cursor location.
- **Selection modes:** support at least lasso (polygon) and optionally box selection.
- **Selection modifiers:** shift adds to selection; cmd/ctrl toggles individual point selection (recommended). Exact bindings can be configurable, but must be recorded in traces.
- **Clear selection:** double-click clears selection (match common embedding tools).

### 4.3 Determinism Requirement
Given:
- the same dataset (generated with fixed seed)
- the same initial view state
- the same interaction trace
- the same canvas size and device pixel ratio settings

Then:
- selection outputs must match (per defined rules)
- hover outputs must match (per defined rules)
- view state checkpoints must match (within tolerance)
- rendered output must match (within tolerance)

**Determinism notes (make these explicit in code):**
- Force `devicePixelRatio` handling to be deterministic: either (a) run all comparisons at DPR=1 in headless, or (b) explicitly record DPR in the trace/session config and scale canvases accordingly.
- Any randomness (dataset gen, jitter, sampling) must be seeded and recorded.
- For performance benchmarking, determinism is less strict, but trace semantics must match.

**Tolerance policy (default):**
- Numeric comparisons are authoritative; image diffs are supportive.
- Express projection/view tolerances in **screen pixels** (e.g. $\le 0.5$ px p99) and/or in disk coordinates (e.g. $\le 1e{-6}$ in model units) where appropriate.

---

## 5) Dataset Requirements

### 5.1 Deterministic Dataset Generation
Implement deterministic dataset generation (seeded PRNG).

Inputs:
- `seed`
- `N`
- `labelCount`
- `geometryMode: "euclidean" | "poincare"`

Outputs:
- `positions: Float32Array` length `2N` (x,y)
- `labels: Uint16Array`
- optional `ids` (or implicit indices)

### 5.2 Dataset Sets
Include at least:
- `N = 10_000`
- `N = 100_000`
- `N = 1_000_000`
- a hyperbolic stress dataset: many points close to disk boundary (e.g. radius in `[0.9, 0.999]`)

---

## 6) Reference Implementation (Naive, Accurate)

The reference implementation defines the “truth” behavior. It may be slow.

### 6.1 Euclidean Reference
Implement the simplest accurate approach (Canvas2D recommended).

Required behaviors:
- render all points
- Euclidean pan and zoom
- hover: brute-force nearest in screen space
- lasso: brute-force point-in-polygon over all points

### 6.2 Hyperbolic Reference (Poincaré Disk)
This must implement **correct hyperbolic navigation**.

#### View State
Separate:
- hyperbolic camera state (an isometry representation)
- display zoom scalar

We will choose a concrete, deterministic isometry representation in code (e.g. Möbius transform parameters) with:
- `applyCamera(p: vec2_poincare) -> vec2_poincare`
- `invertCamera(p: vec2_poincare) -> vec2_poincare`

The critical requirement is that **pan/zoom invariants match the Interaction Contract** (anchor-invariant pan/zoom).

#### Rendering (Reference)
For each Poincaré point:
1. map to an internal model for transforms
2. apply isometry
3. map back to Poincaré
4. apply display zoom and map to screen
5. draw

#### Pan Semantics
Pan must update the hyperbolic isometry (it is not Euclidean translation). Pan is anchor-invariant as defined above.

#### Lasso Semantics (Reference: unprojected lasso)
- lasso is drawn in screen space (polyline)
- transform lasso vertices back to data-space Poincaré using inverse camera mapping
- brute-force point-in-polygon in data space for all points

#### Hover Semantics (Reference)
Define hover as:
- nearest point in screen pixel distance after projection
- brute force over all points is acceptable

**Tie-break rule (make explicit):** if multiple points are within $\varepsilon$ of the minimum distance, choose the smallest point index (recommended) so results are deterministic.

---

## 7) Candidate Implementation (Optimized)

The candidate implementation is optimized for performance but must match reference semantics.

The candidate may use:
- WebGL/WebGPU rendering
- workers
- spatial indices
- shader-based transforms

It must still:
- expose the same interface
- match reference outputs within tolerances
- be benchmarked by the harness

**Important:** we will not duplicate code per-geometry unless needed. Prefer shared core logic:
- shared trace format + runner
- shared selection set diffing
- shared dataset loaders/generators
- geometry-specific camera/math modules
- renderer-specific code (Canvas2D reference vs WebGL candidate)

---

## 8) How to Compare Rendering Outputs (Screenshots / PNG)

### 8.1 In-browser PNG capture
For each implementation:
1. set dataset + view state
2. render
3. export image:
   - `canvas.toBlob("image/png")` or `canvas.toDataURL("image/png")`
4. store artifacts with deterministic naming:
   - `artifacts/{geometry}/{dataset}/{impl}/{scenario}/{frameIndex}.png`

### 8.2 Headless PNG capture (recommended)
Use a headless browser test runner to automate exact comparisons:
- open a test page with a deterministic viewport
- load dataset
- replay a trace
- capture screenshots at checkpoints or every N frames

Artifacts:
- `reference.png`, `candidate.png`, `diff.png`
- `metrics.json`, `comparison.json`

### 8.3 Image diff requirements
Implement pixel-level comparison:
- same resolution required
- compute:
  - max per-channel diff
  - MAE or RMSE
  - number of pixels over threshold
- generate a `diff.png` heatmap

Tolerances must be explicit and configurable.

**Important:** WebGL/WebGPU renderers may differ slightly from Canvas2D due to antialiasing and rasterization. The harness should treat image diffs as:
- a strong signal for gross mismatches
- not the sole correctness gate unless we control rendering settings tightly (e.g. disable MSAA, use identical point sprites, lock blending)

---

## 9) How to Compare Interactions

### 9.1 Interaction traces (record + replay)
Implement:
- trace recording from live UI events
- trace replay with deterministic timing or fixed-step execution

Trace must include:
- pointer events (position, buttons)
- wheel events (delta)
- key modifiers
- mode changes (pan vs lasso)
- resize events

Store traces as JSON:
- `traces/{geometry}/{name}.json`

### 9.2 What to compare during replay
At a minimum compare:

**Selection**
- On “lasso end”:
  - compare selected indices sets
  - save differences (missing/excess indices)

**Boundary rule (make explicit):** decide whether “point on edge” counts as inside. Whatever we pick, encode it in the reference and document it here.

**Hover**
- Sample hover at fixed intervals (e.g. every 50ms during replay):
  - compare hovered index
  - log mismatches

**View state checkpoints**
- Compare view state:
  - start
  - after each pan/zoom completion
  - end of trace

### 9.3 Numeric projection checks (recommended)
To isolate math differences quickly:
- choose a fixed set of “probe points” (e.g. 1,000 indices)
- compare `project(point)` results (screen x,y):
  - max error
  - p99 error

This is often more diagnostic than pixel diffs alone.

---

## 10) Performance Benchmarking

### 10.1 Metrics to record
- frame times: p50/p95/p99
- FPS over windows (optional)
- lasso computation time (start → selection result)
- hover computation time (hitTest cost)

Also record (at least in development runs):
- GPU frame time (if available)
- memory usage snapshots (best-effort; optional)

### 10.2 Procedure
For each dataset size and scenario:
1. warm up for a fixed time
2. replay trace for fixed duration
3. collect metrics
4. repeat multiple times
5. output JSON summaries

Store artifacts:
- `artifacts/{geometry}/{dataset}/{impl}/{scenario}/metrics.json`

---

## 11) Canonical Scenarios (Traces)

Create standard traces per geometry, recorded once and reused.

### Euclidean scenarios
- gentle pan + zoom
- aggressive pan
- lasso small region
- lasso large region
- hover scrub across dense region

### Hyperbolic scenarios
- gentle hyperbolic pan
- pan near boundary (stress)
- display zoom in/out cycles
- lasso near center
- lasso near boundary
- hover scrub near boundary

Each scenario must be saved and replayable:
- `traces/euclidean/*.json`
- `traces/hyperbolic/*.json`

---

## 12) Pass/Fail Criteria

All thresholds must be configurable and documented.

Minimum required checks:
- selection set equality (exact unless a boundary tolerance rule is defined)
- hover equality (exact under sampling rule, with a documented tie-break policy)
- view state numeric parameters within tolerance
- image diff below thresholds (max diff, error pixel count, etc.)

**Recommended default thresholds (edit to taste):**
- Projection probe screen error: p99 $\le 0.5$ px, max $\le 2$ px
- View-state numeric tolerance: $\le 1e{-6}$ in model params _or_ derived screen error $\le 0.5$ px
- Image diff: max channel diff $\le 10$ and error pixels $\le 0.1\%$ (only if rendering settings are controlled)

**Performance targets (need to be made concrete):**
- Define target devices and minimum acceptable interactivity.
- As a starting point, treat these as stretch goals:
  - Euclidean: smooth pan/zoom at N=20,000,000
  - Hyperbolic: smooth pan/zoom near boundary at N=20,000,000

The harness must report performance numerically so we can compare candidate versions over time.

---

## 13) Iteration Workflow (How this achieves “recreate naive but accurate”)

For each geometry:

1. Build and stabilize the reference implementation.
2. Generate fixed datasets and record canonical traces.
3. Build candidate v1.
4. Run verification suite:
   - selection diff
   - hover diff
   - view state diff
   - numeric projection diff
   - image diff
5. If candidate fails:
   - use projection probes and diff images to locate mismatches
   - correct candidate behavior
6. Once candidate passes:
   - run performance benchmark suite
7. Optimize candidate and repeat verification after each change.

Rule: the candidate should only be considered “better” when it both (a) passes correctness checks and (b) improves performance on the benchmark suite.

---
