#!/usr/bin/env npx tsx
/**
 * Demo Interaction Benchmark Runner
 *
 * Drives the real demo page (index.html) with Puppeteer and measures perceived FPS
 * while:
 *  - Panning aggressively towards the edges (Poincaré disk boundary / canvas edges)
 *  - Hovering (mousemove triggering hitTest + render)
 *
 * This is intended to match real user interaction more closely than benchmark.html.
 */

import puppeteer, { Browser, Page } from 'puppeteer';
import { spawn, ChildProcess } from 'child_process';
import { setTimeout as sleep } from 'timers/promises';
import { mkdirSync } from 'fs';
import { dirname } from 'path';

function getNpxCommand(): string {
  return process.platform === 'win32' ? 'npx.cmd' : 'npx';
}

async function stopDevServer(proc: ChildProcess): Promise<void> {
  if (proc.killed) return;
  try {
    if (process.platform !== 'win32' && proc.pid) {
      try {
        process.kill(-proc.pid, 'SIGTERM');
      } catch {
        proc.kill('SIGTERM');
      }
    } else {
      proc.kill('SIGTERM');
    }
  } catch {
    // ignore
  }

  try {
    if (process.platform !== 'win32' && proc.pid) {
      try {
        process.kill(-proc.pid, 'SIGKILL');
      } catch {
        proc.kill('SIGKILL');
      }
    } else {
      proc.kill('SIGKILL');
    }
  } catch {
    // ignore
  }
}

interface Config {
  headless: boolean;
  dpr: number;
  width: number;
  height: number;
  geometry: 'euclidean' | 'poincare';
  points: number;
  // Duration per phase (ms)
  panMs: number;
  hoverMs: number;
  screenshot?: string;
}

const DEFAULTS: Config = {
  headless: false,
  dpr: 2,
  // Keep defaults small enough to fit on typical laptop screens when running headed.
  // You can still override for perf comparisons via --width/--height.
  width: 1400,
  height: 900,
  geometry: 'poincare',
  points: 1_000_000,
  panMs: 5000,
  hoverMs: 5000,
};

function parseArgs(): Partial<Config> {
  const args: Partial<Config> = {};
  for (const arg of process.argv.slice(2)) {
    if (arg === '--headless') args.headless = true;
    else if (arg === '--headed' || arg === '--no-headless') args.headless = false;
    else if (arg.startsWith('--dpr=')) {
      const v = Number(arg.slice('--dpr='.length));
      if (Number.isFinite(v) && v > 0) args.dpr = v;
    } else if (arg.startsWith('--width=')) {
      const v = Number(arg.slice('--width='.length));
      if (Number.isFinite(v) && v > 0) args.width = v;
    } else if (arg.startsWith('--height=')) {
      const v = Number(arg.slice('--height='.length));
      if (Number.isFinite(v) && v > 0) args.height = v;
    } else if (arg.startsWith('--geometry=')) {
      const v = arg.slice('--geometry='.length);
      if (v === 'euclidean' || v === 'poincare') args.geometry = v;
    } else if (arg.startsWith('--points=')) {
      const v = Number(arg.slice('--points='.length));
      if (Number.isFinite(v) && v > 0) args.points = v;
    } else if (arg.startsWith('--panMs=')) {
      const v = Number(arg.slice('--panMs='.length));
      if (Number.isFinite(v) && v > 0) args.panMs = v;
    } else if (arg.startsWith('--hoverMs=')) {
      const v = Number(arg.slice('--hoverMs='.length));
      if (Number.isFinite(v) && v > 0) args.hoverMs = v;
    } else if (arg.startsWith('--screenshot=')) {
      const v = arg.slice('--screenshot='.length).trim();
      if (v) args.screenshot = v;
    }
  }
  return args;
}

async function maybeCaptureScreenshot(page: Page, cfg: Config, label: string): Promise<void> {
  if (!cfg.screenshot) return;

  const base = cfg.screenshot;
  const path = base.includes('.')
    ? base.replace(/\.png$/i, `-${label}.png`)
    : `${base}-${label}.png`;

  mkdirSync(dirname(path), { recursive: true });

  // Try to capture the canvas container; fall back to full page.
  const el = await page.$('#canvasBody');
  if (el) {
    await el.screenshot({ path });
  } else {
    await page.screenshot({ path, fullPage: true });
  }
  console.log(`[Demo Interaction Bench] Saved screenshot: ${path}`);
}

async function startDevServer(): Promise<{ proc: ChildProcess; url: string }> {
  return new Promise((resolve, reject) => {
    const proc = spawn(getNpxCommand(), ['vite', '--port', '5174', '--strictPort'], {
      stdio: ['ignore', 'pipe', 'pipe'],
      shell: false,
      detached: process.platform !== 'win32',
    });

    let resolved = false;
    const timeoutId = global.setTimeout(() => {
      if (!resolved) reject(new Error('Dev server startup timeout'));
    }, 30000);

    const checkForUrl = (text: string) => {
      const match = text.match(/Local:\s+(http:\/\/localhost:\d+)/);
      if (match && !resolved) {
        resolved = true;
        clearTimeout(timeoutId);
        resolve({ proc, url: match[1] });
      }
    };

    proc.stdout?.on('data', (data: Buffer) => checkForUrl(data.toString()));
    proc.stderr?.on('data', (data: Buffer) => {
      const text = data.toString();
      checkForUrl(text);
      if (text.includes('error') || text.includes('Error')) {
        console.error('Dev server error:', text);
      }
    });

    proc.on('error', (err) => {
      if (!resolved) {
        clearTimeout(timeoutId);
        reject(err);
      }
    });
  });
}

async function installFpsProbe(page: Page): Promise<void> {
  await page.evaluate(() => {
    (window as any).__fpsProbe = {
      running: false,
      last: 0,
      intervals: [] as number[],
      handle: 0 as number,
      start() {
        this.running = true;
        this.last = 0;
        this.intervals = [];
        const loop = (ts: number) => {
          if (!this.running) return;
          if (this.last !== 0) {
            this.intervals.push(ts - this.last);
            if (this.intervals.length > 2000) this.intervals.shift();
          }
          this.last = ts;
          this.handle = requestAnimationFrame(loop);
        };
        this.handle = requestAnimationFrame(loop);
      },
      stop() {
        this.running = false;
        if (this.handle) cancelAnimationFrame(this.handle);
      },
      stats() {
        const xs = this.intervals.slice().sort((a: number, b: number) => a - b);
        const n = xs.length;
        const avg = n ? xs.reduce((s: number, v: number) => s + v, 0) / n : 0;
        const median = n ? xs[Math.floor(n / 2)] : 0;
        const p95 = n ? xs[Math.floor(n * 0.95)] : 0;
        return {
          samples: n,
          avgMs: avg,
          medianMs: median,
          p95Ms: p95,
          fpsAvg: avg > 0 ? 1000 / avg : 0,
          fpsMedian: median > 0 ? 1000 / median : 0,
        };
      },
    };
  });
}

async function setDemoControls(page: Page, cfg: Config): Promise<void> {
  // The demo page (index.html / app.ts) uses:
  //   - Radio buttons:  input[name="geometry"]  (values: 'euclidean' | 'poincare')
  //   - Radio buttons:  input[name="renderer"]  (values: 'webgl' | 'reference')
  //   - Range slider:   input#numPoints         (0–10, mapped to POINT_PRESETS)
  // There is no explicit "Generate" button; the app auto-generates on input changes.

  // POINT_PRESETS from app.ts — kept in sync so we can map a point count to a slider index.
  const POINT_PRESETS = [
    1_000, 10_000, 50_000, 100_000, 250_000, 500_000,
    1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000,
  ];

  // Snap to the nearest preset.
  const sliderIndex = POINT_PRESETS.reduce(
    (best, v, i) =>
      Math.abs(v - cfg.points) < Math.abs(POINT_PRESETS[best] - cfg.points) ? i : best,
    0,
  );
  const actualPoints = POINT_PRESETS[sliderIndex];

  // Select geometry via its radio button.
  await page.evaluate((geometry: string) => {
    const el = document.querySelector<HTMLInputElement>(
      `input[name="geometry"][value="${geometry}"]`,
    );
    if (el) {
      el.checked = true;
      el.dispatchEvent(new Event('change', { bubbles: true }));
    }
  }, cfg.geometry);

  // Select the WebGL renderer via its radio button.
  await page.evaluate(() => {
    const el = document.querySelector<HTMLInputElement>('input[name="renderer"][value="webgl"]');
    if (el) {
      el.checked = true;
      el.dispatchEvent(new Event('change', { bubbles: true }));
    }
  });

  // Set the point-count slider and fire both 'input' and 'change' so the app
  // debounce timer is triggered and the label is updated.
  await page.evaluate((idx: number) => {
    const el = document.getElementById('numPoints') as HTMLInputElement | null;
    if (el) {
      el.value = String(idx);
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
    }
  }, sliderIndex);

  // Wait for statPoints to reflect the newly generated dataset.
  const expected = actualPoints.toLocaleString();
  await page.waitForFunction(
    (exp: string) => {
      const el = document.getElementById('statPoints');
      return !!el && (el.textContent ?? '').includes(exp);
    },
    { timeout: 120000 },
    expected,
  );

  // Give the app a beat to finish uploading GPU buffers.
  await sleep(250);
}

async function measurePhase(page: Page, label: string, run: () => Promise<void>): Promise<any> {
  await page.evaluate(() => (window as any).__fpsProbe.start());
  await run();
  await page.evaluate(() => (window as any).__fpsProbe.stop());
  const stats = await page.evaluate(() => (window as any).__fpsProbe.stats());
  return { label, ...stats };
}

async function runPanToEdges(page: Page, cfg: Config): Promise<void> {
  await page.evaluate(async (params: { geometry: 'euclidean' | 'poincare'; durationMs: number }) => {
    const canvas = document.getElementById('canvas') as HTMLCanvasElement | null;
    if (!canvas) throw new Error('Canvas not found');
    const rect = canvas.getBoundingClientRect();

    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;

    const diskRadius = Math.min(rect.width, rect.height) * 0.45;
    const amp = params.geometry === 'poincare'
      ? diskRadius * 0.92
      : Math.min(rect.width, rect.height) * 0.45;

    // One-way drag: center -> near right boundary.
    // This matches the typical user action (drag and release) and avoids
    // returning to the start which can legitimately cancel out the net pan.
    const keypoints = [
      { x: cx, y: cy },
      { x: cx + amp, y: cy },
    ];

    const dispatch = (type: string, x: number, y: number, buttons: number) => {
      const ev = new MouseEvent(type, {
        bubbles: true,
        cancelable: true,
        clientX: x,
        clientY: y,
        buttons,
      });
      canvas.dispatchEvent(ev);
    };

    // Start drag at center.
    dispatch('mousedown', cx, cy, 1);

    const segs = keypoints.length - 1;
    const start = performance.now();

    await new Promise<void>((resolve) => {
      const step = () => {
        const now = performance.now();
        const t = Math.min(1, (now - start) / Math.max(1, params.durationMs));

        const sFloat = t * segs;
        const s = Math.min(segs - 1, Math.floor(sFloat));
        const u = sFloat - s;

        const a = keypoints[s];
        const b = keypoints[s + 1];
        const x = a.x + (b.x - a.x) * u;
        const y = a.y + (b.y - a.y) * u;

        dispatch('mousemove', x, y, 1);

        if (t >= 1) {
          dispatch('mouseup', x, y, 0);
          // Give the app a chance to flush pending pan in the mouseup handler
          // and/or a following rAF.
          requestAnimationFrame(() => resolve());
          return;
        }
        requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
  }, { geometry: cfg.geometry, durationMs: cfg.panMs });
}

async function runHoverPath(page: Page, cfg: Config): Promise<void> {
  await page.evaluate(async (params: { durationMs: number }) => {
    const canvas = document.getElementById('canvas') as HTMLCanvasElement | null;
    if (!canvas) throw new Error('Canvas not found');
    const rect = canvas.getBoundingClientRect();

    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const radius = Math.min(rect.width, rect.height) * 0.25;

    const dispatch = (x: number, y: number) => {
      const ev = new MouseEvent('mousemove', {
        bubbles: true,
        cancelable: true,
        clientX: x,
        clientY: y,
        buttons: 0,
      });
      canvas.dispatchEvent(ev);
    };

    const start = performance.now();

    await new Promise<void>((resolve) => {
      const step = () => {
        const now = performance.now();
        const t = Math.min(1, (now - start) / Math.max(1, params.durationMs));
        const ang = t * Math.PI * 2;
        const x = cx + Math.cos(ang) * radius;
        const y = cy + Math.sin(ang) * radius;
        dispatch(x, y);

        if (t >= 1) {
          resolve();
          return;
        }
        requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
  }, { durationMs: cfg.hoverMs });
}

async function main() {
  const cfg: Config = { ...DEFAULTS, ...parseArgs() };

  console.log('[Demo Interaction Bench] Starting...');
  console.log(`  Geometry: ${cfg.geometry}`);
  console.log(`  Points:   ${cfg.points.toLocaleString()}`);
  console.log(`  DPR:      ${cfg.dpr}`);
  console.log(`  Viewport: ${cfg.width}x${cfg.height}`);
  console.log(`  Pan:      ${cfg.panMs}ms`);
  console.log(`  Hover:    ${cfg.hoverMs}ms`);
  console.log('');

  let dev: { proc: ChildProcess; url: string } | null = null;
  let browser: Browser | null = null;

  try {
    console.log('[1/4] Starting dev server...');
    dev = await startDevServer();
    console.log(`  Dev server running at ${dev.url}`);

    console.log('[2/4] Launching browser...');
    browser = await puppeteer.launch({
      headless: cfg.headless,
      args: [
        // Ensure the window is large enough so the configured viewport isn't clipped in headed mode.
        `--window-size=${cfg.width},${cfg.height}`,
        '--window-position=0,0',
        '--disable-background-timer-throttling',
        '--disable-backgrounding-occluded-windows',
        '--disable-renderer-backgrounding',
        '--disable-features=CalculateNativeWinOcclusion',
      ],
    });

    const page = await browser.newPage();
    await page.setViewport({ width: cfg.width, height: cfg.height, deviceScaleFactor: cfg.dpr });

    // tsx/esbuild may wrap serialized functions passed to page.evaluate() with
    // __name(...) calls. Define a no-op __name helper before any evaluate runs.
    await page.evaluateOnNewDocument('globalThis.__name = (fn) => fn;');

    console.log('[3/4] Loading demo page...');
    await page.goto(dev.url + '/', { waitUntil: 'networkidle2', timeout: 120000 });
    await page.bringToFront();

    // Also define it in the currently loaded document (defensive).
    await page.evaluate('globalThis.__name = (fn) => fn;');

    await installFpsProbe(page);

    console.log('[4/4] Generating dataset + running interaction phases...');
    await setDemoControls(page, cfg);
    await maybeCaptureScreenshot(page, cfg, 'after-generate');

    // The main browser benchmarks already cover broad perf metrics; this runner
    // focuses on perceived FPS for real demo interactions.
    const panStats = await measurePhase(page, 'panToEdges', async () => runPanToEdges(page, cfg));
    await maybeCaptureScreenshot(page, cfg, 'after-pan');
    const hoverStats = await measurePhase(page, 'hover', async () => runHoverPath(page, cfg));
    await maybeCaptureScreenshot(page, cfg, 'after-hover');

    const sys = await page.evaluate(() => ({
      userAgent: navigator.userAgent,
      devicePixelRatio: window.devicePixelRatio,
      inner: { width: window.innerWidth, height: window.innerHeight },
      visualViewport: (window as any).visualViewport
        ? { width: (window as any).visualViewport.width, height: (window as any).visualViewport.height, scale: (window as any).visualViewport.scale }
        : null,
      canvas: (() => {
        const c = document.getElementById('canvas') as HTMLCanvasElement | null;
        if (!c) return null;
        return {
          cssWidth: c.clientWidth,
          cssHeight: c.clientHeight,
          bufWidth: c.width,
          bufHeight: c.height,
        };
      })(),
      statFrameTime: (document.getElementById('statFrameTime')?.textContent ?? ''),
    }));

    const pretty = (s: any) =>
      `${s.label.padEnd(10)} | samples=${String(s.samples).padStart(5)} | avg=${s.avgMs.toFixed(2)}ms (${s.fpsAvg.toFixed(1)} fps) | p95=${s.p95Ms.toFixed(2)}ms`;

    console.log('');
    console.log('════════════════════════════════════════════════════════════════════');
    console.log('DEMO INTERACTION RESULTS');
    console.log('════════════════════════════════════════════════════════════════════');
    console.log(`UserAgent: ${String(sys.userAgent).slice(0, 90)}...`);
    console.log(`Window DPR: ${sys.devicePixelRatio}`);
    console.log(`Inner viewport: ${sys.inner?.width}x${sys.inner?.height}`);
    if (sys.visualViewport) {
      console.log(`VisualViewport: ${sys.visualViewport.width}x${sys.visualViewport.height} (scale ${sys.visualViewport.scale})`);
    }
    console.log(`Canvas: css=${sys.canvas?.cssWidth}x${sys.canvas?.cssHeight}, buffer=${sys.canvas?.bufWidth}x${sys.canvas?.bufHeight}`);
    console.log(`Demo statFrameTime: ${sys.statFrameTime}`);
    console.log('────────────────────────────────────────────────────────────────────');
    console.log(pretty(panStats));
    console.log(pretty(hoverStats));
    console.log('════════════════════════════════════════════════════════════════════');
    console.log('');
  } finally {
    if (browser) await browser.close().catch(() => {});
    if (dev?.proc) await stopDevServer(dev.proc);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
