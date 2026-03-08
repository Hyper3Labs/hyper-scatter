#!/usr/bin/env npx tsx
/**
 * Automated Browser Benchmark Runner
 *
 * Runs the full benchmark suite in a browser and reports results.
 * By default, shows the browser window so you can see progress.
 *
 * Usage:
 *   npm run bench                              # Run with visible browser (default)
 *   npm run bench -- --headless                # Run without browser window
 *   npm run bench -- --points=100000,500000    # Custom point counts
 *   npm run bench -- --geometries=poincare     # Only test poincare
 *   npm run bench -- --geometries=euclidean,poincare,euclidean3d,sphere
 */

import puppeteer, { Browser, Page } from 'puppeteer';
import { spawn, ChildProcess } from 'child_process';

function getNpxCommand(): string {
  // Windows uses npx.cmd; POSIX uses npx.
  return process.platform === 'win32' ? 'npx.cmd' : 'npx';
}

// ============================================================================
// Configuration
// ============================================================================

interface BenchConfig {
  headless: boolean;
  dpr: number;
  viewport: { width: number; height: number };
  canvas: { width?: number; height?: number };
  renderer: 'webgl' | 'reference';
  lassoRadiusScale?: number;
  pointCounts: number[];
  geometries: BenchGeometry[];
  measuredFrames: number;
  hitTestSamples: number;
  timeout: number;  // ms per benchmark
}

type BenchGeometry = 'euclidean' | 'poincare' | 'euclidean3d' | 'sphere';

const DEFAULT_CONFIG: BenchConfig = {
  headless: false,  // Show browser by default so user can see progress
  dpr: 1,
  viewport: { width: 1200, height: 800 },
  canvas: { height: 400 },
  renderer: 'webgl',
  pointCounts: [1000, 10000, 50000, 100000, 500000, 1000000],
  geometries: ['euclidean', 'poincare'],
  measuredFrames: 20,
  hitTestSamples: 100,
  timeout: 120000,
};

// ============================================================================
// Dev Server Management
// ============================================================================

async function startDevServer(): Promise<{ proc: ChildProcess; url: string }> {
  return new Promise((resolve, reject) => {
    const proc = spawn(getNpxCommand(), ['vite', '--port', '5174', '--strictPort'], {
      stdio: ['ignore', 'pipe', 'pipe'],
      // `shell: true` makes it harder to reliably kill the actual Vite process.
      shell: false,
      // On POSIX, start a new process group so we can kill the whole tree.
      detached: process.platform !== 'win32',
    });

    let resolved = false;
    const timeoutId = global.setTimeout(() => {
      if (!resolved) {
        reject(new Error('Dev server startup timeout'));
      }
    }, 30000);

    // Helper to check for URL in output (Vite may print to either stdout or stderr)
    const checkForUrl = (text: string) => {
      const match = text.match(/Local:\s+(http:\/\/localhost:\d+)/);
      if (match && !resolved) {
        resolved = true;
        clearTimeout(timeoutId);
        resolve({ proc, url: match[1] });
      }
    };

    proc.stdout?.on('data', (data: Buffer) => {
      checkForUrl(data.toString());
    });

    proc.stderr?.on('data', (data: Buffer) => {
      const text = data.toString();
      // Vite sometimes outputs the Local URL to stderr
      checkForUrl(text);
      // Only log actual errors
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

async function stopDevServer(proc: ChildProcess): Promise<void> {
  if (proc.killed) return;

  // Best-effort shutdown.
  try {
    if (process.platform !== 'win32' && proc.pid) {
      // If the server was spawned in its own process group, kill the group.
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

  // If the process didn't die on SIGTERM, force kill.
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

// ============================================================================
// Benchmark Runner
// ============================================================================

interface BenchmarkResult {
  geometry: string;
  points: number;
  datasetGenMs: number;
  renderMs: { avg: number; min: number; max: number };
  frameIntervalMs: { avg: number; min: number; max: number };
  hitTestMs: { avg: number; min: number; max: number };
  panMs?: { avg: number; min: number; max: number };
  hoverMs?: { avg: number; min: number; max: number };
  panFrameIntervalMs?: { avg: number; min: number; max: number };
  hoverFrameIntervalMs?: { avg: number; min: number; max: number };
  lassoMs: number;
  lassoSelectedCount: number;
  memoryMB?: number;
}

interface BenchmarkReport {
  timestamp: string;
  system: {
    userAgent: string;
    devicePixelRatio: number;
    canvasWidth: number;
    canvasHeight: number;
  };
  results: BenchmarkResult[];
}

async function runBenchmarks(
  page: Page,
  config: BenchConfig
): Promise<BenchmarkReport> {
  // Optionally override canvas CSS size before starting.
  // Width defaults to responsive container width; height defaults to 400px in benchmark.html.
  await page.evaluate((canvasCfg: { width?: number; height?: number }) => {
    const canvas = document.getElementById('canvas') as HTMLCanvasElement | null;
    const candidateCanvas = document.getElementById('candidateCanvas') as HTMLCanvasElement | null;
    if (!canvas) return;

    if (typeof canvasCfg.width === 'number' && Number.isFinite(canvasCfg.width) && canvasCfg.width > 0) {
      canvas.style.width = `${Math.floor(canvasCfg.width)}px`;
    }
    if (typeof canvasCfg.height === 'number' && Number.isFinite(canvasCfg.height) && canvasCfg.height > 0) {
      canvas.style.height = `${Math.floor(canvasCfg.height)}px`;
    }

    // Keep the hidden WebGL canvas in sync (some renderers read client sizes).
    if (candidateCanvas) {
      if (canvas.style.width) candidateCanvas.style.width = canvas.style.width;
      if (canvas.style.height) candidateCanvas.style.height = canvas.style.height;
    }

    // Trigger layout + any resize listeners.
    window.dispatchEvent(new Event('resize'));
  }, config.canvas);

  // Run benchmarks directly (do not depend on benchmark.html UI state)
  const report = await page.evaluate(async (cfg: {
    renderer: 'webgl' | 'reference';
    lassoRadiusScale?: number;
    pointCounts: number[];
    geometries: BenchGeometry[];
    measuredFrames: number;
    hitTestSamples: number;
  }) => {
    // When this function is serialized and executed in the browser context,
    // it must be self-contained. Some build pipelines (esbuild) may wrap
    // inner functions with a `__name(fn, "name")` helper for debugging.
    // That helper does not exist in the page context, so we provide a shim.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const __name = (fn: any) => fn;

    const canvas = document.getElementById('canvas') as HTMLCanvasElement | null;
    if (!canvas) throw new Error('Canvas element not found');

    const VizBenchmark = (window as any).VizBenchmark;
    if (!VizBenchmark?.runBenchmarks) {
      throw new Error('VizBenchmark.runBenchmarks not found');
    }

    const benchCfg = {
      pointCounts: cfg.pointCounts,
      geometries: cfg.geometries,
      warmupFrames: 5,
      measuredFrames: cfg.measuredFrames,
      hitTestSamples: cfg.hitTestSamples,
      renderer: cfg.renderer,
      lassoRadiusScale: cfg.lassoRadiusScale,
    };

    return await VizBenchmark.runBenchmarks(canvas, benchCfg);
  }, {
    renderer: config.renderer,
    lassoRadiusScale: config.lassoRadiusScale,
    pointCounts: config.pointCounts,
    geometries: config.geometries,
    measuredFrames: config.measuredFrames,
    hitTestSamples: config.hitTestSamples,
  });

  if (!report) throw new Error('Benchmark produced no report');
  return report;
}

// ============================================================================
// Result Formatting
// ============================================================================

function formatResults(report: BenchmarkReport): string {
  const lines: string[] = [
    '',
    '\x1b[1m\x1b[36m' + '═'.repeat(120) + '\x1b[0m',
    '\x1b[1m\x1b[36mBROWSER BENCHMARK RESULTS\x1b[0m',
    '\x1b[1m\x1b[36m' + '═'.repeat(120) + '\x1b[0m',
    `Timestamp: ${report.timestamp}`,
    `Canvas: ${report.system.canvasWidth}x${report.system.canvasHeight} @ ${report.system.devicePixelRatio}x DPR`,
    `User Agent: ${report.system.userAgent.substring(0, 80)}...`,
    '─'.repeat(120),
    '\x1b[1mGeometry   │ Points      │ Dataset  │ CPU Submit │ Frame Int. │ FPS  │ Pan FPS │ Hover │ HitTest  │ Lasso     │ Memory\x1b[0m',
    '─'.repeat(120),
  ];

  for (const r of report.results) {
    const geo = r.geometry.padEnd(10);
    const pts = r.points.toLocaleString().padStart(11);
    const dgen = `${r.datasetGenMs.toFixed(1)}ms`.padStart(8);
    const cpuAvg = `${r.renderMs.avg.toFixed(2)}ms`.padStart(10);
    const frameInt = `${r.frameIntervalMs.avg.toFixed(2)}ms`.padStart(10);
    const fps = (1000 / r.frameIntervalMs.avg).toFixed(1);
    const fpsColored = parseFloat(fps) >= 55
      ? `\x1b[32m${fps.padStart(10)}\x1b[0m`  // Green
      : parseFloat(fps) >= 30
        ? `\x1b[33m${fps.padStart(10)}\x1b[0m`  // Yellow
        : `\x1b[31m${fps.padStart(10)}\x1b[0m`; // Red
    const panFpsRaw = r.panFrameIntervalMs ? (1000 / r.panFrameIntervalMs.avg).toFixed(1) : 'N/A';
    const hoverFpsRaw = r.hoverFrameIntervalMs ? (1000 / r.hoverFrameIntervalMs.avg).toFixed(1) : 'N/A';

    const panFps = panFpsRaw.padStart(7);
    const hoverFps = hoverFpsRaw.padStart(6);

    const ht = `${r.hitTestMs.avg.toFixed(3)}ms`.padStart(8);
    const lasso = `${r.lassoMs.toFixed(2)}ms`.padStart(9);
    const mem = r.memoryMB ? `${r.memoryMB.toFixed(0)}MB`.padStart(6) : 'N/A'.padStart(6);

    lines.push(`${geo} │ ${pts} │ ${dgen} │ ${cpuAvg} │ ${frameInt} │ ${fpsColored} │ ${panFps} │ ${hoverFps} │ ${ht} │ ${lasso} │ ${mem}`);
  }

  lines.push('═'.repeat(120));

  // Summary
  const geometries = Array.from(new Set(report.results.map((r) => r.geometry)));
  for (const geometry of geometries) {
    const rows = report.results.filter((r) => r.geometry === geometry);
    if (rows.length === 0) continue;

    const maxPts = Math.max(...rows.map((r) => r.points));
    const maxResult = rows.find((r) => r.points === maxPts);
    if (!maxResult) continue;

    const fps = (1000 / maxResult.frameIntervalMs.avg).toFixed(1);
    lines.push(`\x1b[1m${geometry} max:\x1b[0m ${maxPts.toLocaleString()} points @ ${fps} FPS (${maxResult.renderMs.avg.toFixed(2)}ms CPU)`);
  }

  lines.push('');
  return lines.join('\n');
}

// ============================================================================
// Main
// ============================================================================

function parseArgs(): Partial<BenchConfig> {
  const args: Partial<BenchConfig> = {};

  for (const arg of process.argv.slice(2)) {
    if (arg === '--headless') {
      args.headless = true;
    } else if (arg === '--headed' || arg === '--no-headless') {
      args.headless = false;
    } else if (arg.startsWith('--renderer=')) {
      const v = arg.slice('--renderer='.length);
      if (v === 'webgl' || v === 'candidate') args.renderer = 'webgl';
      if (v === 'reference' || v === 'ref') args.renderer = 'reference';
    } else if (arg.startsWith('--points=')) {
      const points = arg.slice('--points='.length).split(',').map(Number);
      args.pointCounts = points.filter(n => !isNaN(n) && n > 0);
    } else if (arg.startsWith('--geometries=')) {
      const geos = arg.slice('--geometries='.length).split(',');
      args.geometries = geos.filter(
        (g): g is BenchGeometry =>
          g === 'euclidean' || g === 'poincare' || g === 'euclidean3d' || g === 'sphere'
      );
    } else if (arg.startsWith('--dpr=')) {
      const v = Number(arg.slice('--dpr='.length));
      if (Number.isFinite(v) && v > 0) args.dpr = v;
    } else if (arg.startsWith('--viewport=')) {
      // --viewport=1200x800
      const raw = arg.slice('--viewport='.length);
      const m = raw.match(/^(\d+)x(\d+)$/);
      if (m) {
        const w = Number(m[1]);
        const h = Number(m[2]);
        if (Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0) {
          args.viewport = { width: w, height: h };
        }
      }
    } else if (arg.startsWith('--canvasWidth=')) {
      const v = Number(arg.slice('--canvasWidth='.length));
      if (Number.isFinite(v) && v > 0) {
        args.canvas = { ...(args.canvas ?? {}), width: v };
      }
    } else if (arg.startsWith('--canvasHeight=')) {
      const v = Number(arg.slice('--canvasHeight='.length));
      if (Number.isFinite(v) && v > 0) {
        args.canvas = { ...(args.canvas ?? {}), height: v };
      }
    } else if (arg.startsWith('--measuredFrames=')) {
      const v = Number(arg.slice('--measuredFrames='.length));
      if (Number.isFinite(v) && v >= 5 && v <= 300) args.measuredFrames = Math.floor(v);
    } else if (arg.startsWith('--hitTestSamples=')) {
      const v = Number(arg.slice('--hitTestSamples='.length));
      if (Number.isFinite(v) && v >= 0 && v <= 10000) args.hitTestSamples = Math.floor(v);
    } else if (arg.startsWith('--lassoScale=')) {
      const v = Number(arg.slice('--lassoScale='.length));
      if (Number.isFinite(v) && v > 0 && v <= 2) {
        args.lassoRadiusScale = v;
      }
    } else if (arg.startsWith('--timeout=')) {
      const v = Number(arg.slice('--timeout='.length));
      // Allow large timeouts for multi-million point runs.
      if (Number.isFinite(v) && v >= 10_000 && v <= 3_600_000) args.timeout = Math.floor(v);
    }
  }

  return args;
}

async function main() {
  const config: BenchConfig = { ...DEFAULT_CONFIG, ...parseArgs() };

  if (
    config.renderer === 'reference' &&
    config.geometries.some((g) => g === 'euclidean3d' || g === 'sphere')
  ) {
    throw new Error("Reference renderer supports only euclidean and poincare; use renderer=webgl for 3D geometries.");
  }

  console.log('\x1b[1m\x1b[36m[Browser Benchmark]\x1b[0m Starting...');
  console.log(`  Points: ${config.pointCounts.join(', ')}`);
  console.log(`  Geometries: ${config.geometries.join(', ')}`);
  console.log(`  Renderer: ${config.renderer}`);
  console.log(`  Headless: ${config.headless}`);
  console.log(`  DPR: ${config.dpr}`);
  console.log(`  Viewport: ${config.viewport.width}x${config.viewport.height}`);
  console.log(`  Canvas override: ${config.canvas.width ? `w=${config.canvas.width}` : 'w=auto'}, ${config.canvas.height ? `h=${config.canvas.height}` : 'h=auto'}`);
  console.log(`  Measured frames: ${config.measuredFrames}`);
  console.log(`  Hit test samples: ${config.hitTestSamples}`);
  if (typeof config.lassoRadiusScale === 'number') {
    console.log(`  Lasso scale: ${config.lassoRadiusScale}`);
  }
  console.log(`  Timeout per test: ${config.timeout}ms`);
  console.log('');

  let devServer: { proc: ChildProcess; url: string } | null = null;
  let browser: Browser | null = null;

  // For large-N runs the page can be unresponsive (long JS tasks) while
  // generating datasets / building indexes. Puppeteer will otherwise fail with:
  //   ProtocolError: Runtime.callFunctionOn timed out
  // so we raise the *protocol* timeout in addition to page timeouts.
  const totalTests = Math.max(1, config.pointCounts.length * config.geometries.length);
  const maxWait = config.timeout * totalTests + 30_000;

  try {
    // Start dev server
    console.log('\x1b[33m[1/4]\x1b[0m Starting dev server...');
    devServer = await startDevServer();
    console.log(`  Dev server running at ${devServer.url}`);

    // Launch browser
    console.log('\x1b[33m[2/4]\x1b[0m Launching browser...');
    browser = await puppeteer.launch({
      headless: config.headless,
      // Must be large enough to survive multi-minute main-thread stalls.
      protocolTimeout: Math.max(180_000, maxWait),
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',

        // Try to keep requestAnimationFrame timing representative.
        // Without these, Chromium may throttle to ~30 FPS in automation.
        '--disable-background-timer-throttling',
        '--disable-backgrounding-occluded-windows',
        '--disable-renderer-backgrounding',

        // Prevent Windows occlusion detection from throttling (also helps on macOS)
        '--disable-features=CalculateNativeWinOcclusion',

        // Help ensure we get real GPU (where available) rather than falling back.
        '--ignore-gpu-blocklist',

        // Stabilize viewport size.
        `--window-size=${config.viewport.width},${config.viewport.height}`,
      ],
    });

    const page = await browser.newPage();

    // Some bundlers/transpilers (notably esbuild) may emit a `__name(fn, name)`
    // helper when serializing functions. Puppeteer executes `page.evaluate`
    // callbacks in the *page* context, where that helper is not defined.
    // Define a no-op shim early so any serialized callbacks can run.
    // Use the string form to avoid our own callback being wrapped.
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      await (page as any).evaluateOnNewDocument('globalThis.__name = (fn) => fn;');
    } catch {
      // If the puppeteer version doesn't support string injection here,
      // we'll rely on local shims inside individual evaluate calls.
    }

    await page.setViewport({ width: config.viewport.width, height: config.viewport.height, deviceScaleFactor: config.dpr });

    // Set a generous default timeout to allow large-N runs.
    page.setDefaultTimeout(maxWait);
    page.setDefaultNavigationTimeout(maxWait);

    // Navigate to benchmark page
    console.log('\x1b[33m[3/4]\x1b[0m Loading benchmark page...');
    await page.goto(`${devServer.url}/benchmark.html`, { waitUntil: 'networkidle0' });

    // Bring page to front to ensure it's not throttled as "background"
    await page.bringToFront();

    // Run benchmarks
    console.log('\x1b[33m[4/4]\x1b[0m Running benchmarks...');
    console.log(`  This may take a few minutes for ${config.pointCounts.length * config.geometries.length} tests...`);
    console.log('');

    const report = await runBenchmarks(page, config);

    // Format and print results
    console.log(formatResults(report));

    // Also output JSON for programmatic use
    if (process.env.JSON_OUTPUT) {
      console.log('\n--- JSON OUTPUT ---');
      console.log(JSON.stringify(report, null, 2));
    }

  } catch (error) {
    console.error('\x1b[31mBenchmark failed:\x1b[0m', error);
    process.exitCode = 1;
  } finally {
    // Cleanup
    try {
      if (browser) await browser.close();
    } catch {
      // ignore
    }
    try {
      if (devServer) await stopDevServer(devServer.proc);
    } catch {
      // ignore
    }
  }
}

main();
