import { generateDataset } from '../core/dataset.js';
import {
	buildSemanticLabelModel,
	layoutSemanticLabels,
	type SemanticLabelEngine,
} from '../ui/semantic_labels.js';

interface BenchmarkOptions {
	pointCounts: number[];
	geometries: Array<'euclidean' | 'poincare'>;
	warmupRuns: number;
	measuredRuns: number;
}

interface BenchmarkRow {
	geometry: 'euclidean' | 'poincare';
	points: number;
	baselineBuildMs: number;
	candidateBuildMs: number;
	buildSpeedup: number;
	layoutMs: number;
	baselineLabels: number;
	candidateLabels: number;
}

const DEFAULT_OPTIONS: BenchmarkOptions = {
	pointCounts: [10_000, 100_000, 500_000],
	geometries: ['euclidean', 'poincare'],
	warmupRuns: 1,
	measuredRuns: 5,
};

function parseIntegerList(value: string | undefined, fallback: number[]): number[] {
	if (!value) return fallback;
	const parsed = value
		.split(',')
		.map((part) => Number.parseInt(part.trim(), 10))
		.filter((num) => Number.isFinite(num) && num > 0);
	return parsed.length > 0 ? parsed : fallback;
}

function parseGeometryList(value: string | undefined, fallback: Array<'euclidean' | 'poincare'>): Array<'euclidean' | 'poincare'> {
	if (!value) return fallback;
	const parsed = value
		.split(',')
		.map((part) => part.trim())
		.filter((part): part is 'euclidean' | 'poincare' => part === 'euclidean' || part === 'poincare');
	return parsed.length > 0 ? parsed : fallback;
}

function readOptionsFromArgs(): BenchmarkOptions {
	const args = process.argv.slice(2);
	const optionMap = new Map<string, string>();
	for (const arg of args) {
		if (!arg.startsWith('--')) continue;
		const [key, rawValue] = arg.slice(2).split('=');
		if (!key || rawValue === undefined) continue;
		optionMap.set(key, rawValue);
	}

	return {
		pointCounts: parseIntegerList(optionMap.get('points'), DEFAULT_OPTIONS.pointCounts),
		geometries: parseGeometryList(optionMap.get('geometries'), DEFAULT_OPTIONS.geometries),
		warmupRuns: Number.parseInt(optionMap.get('warmup') ?? '', 10) || DEFAULT_OPTIONS.warmupRuns,
		measuredRuns: Number.parseInt(optionMap.get('runs') ?? '', 10) || DEFAULT_OPTIONS.measuredRuns,
	};
}

function mean(values: number[]): number {
	if (values.length === 0) return 0;
	return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function makeTerms(labels: Uint16Array): string[] {
	const terms = new Array<string>(labels.length);
	for (let i = 0; i < labels.length; i++) {
		terms[i] = `cluster ${labels[i]}`;
	}
	return terms;
}

function buildOnce(
	positions: Float32Array,
	terms: readonly string[],
	geometry: 'euclidean' | 'poincare',
	engine: SemanticLabelEngine,
) {
	return buildSemanticLabelModel({
		positions,
		terms,
		geometry,
		engine,
	});
}

function benchmarkBuild(
	positions: Float32Array,
	terms: readonly string[],
	geometry: 'euclidean' | 'poincare',
	engine: SemanticLabelEngine,
	options: BenchmarkOptions,
) {
	for (let i = 0; i < options.warmupRuns; i++) {
		buildOnce(positions, terms, geometry, engine);
	}

	const runs: number[] = [];
	let lastModel = buildOnce(positions, terms, geometry, engine);
	for (let i = 0; i < options.measuredRuns; i++) {
		const startedAt = performance.now();
		lastModel = buildOnce(positions, terms, geometry, engine);
		runs.push(performance.now() - startedAt);
	}

	return {
		avgMs: mean(runs),
		model: lastModel,
	};
}

function benchmarkLayout(modelNodes: ReturnType<typeof buildSemanticLabelModel>['nodes']): number {
	const viewportWidth = 1120;
	const viewportHeight = 400;
	const startedAt = performance.now();
	for (let iteration = 0; iteration < 24; iteration++) {
		const zoom = iteration < 12 ? 1 : 2.1;
		layoutSemanticLabels({
			nodes: modelNodes,
			zoom,
			viewportWidth,
			viewportHeight,
			displayMode: 'auto',
			project: (x, y) => ({
				x: ((x + 1.25) / 2.5) * viewportWidth,
				y: ((1.25 - y) / 2.5) * viewportHeight,
			}),
		});
	}
	return (performance.now() - startedAt) / 24;
}

async function main() {
	const options = readOptionsFromArgs();
	const rows: BenchmarkRow[] = [];

	for (const geometry of options.geometries) {
		for (const points of options.pointCounts) {
			const dataset = generateDataset({
				seed: 42,
				n: points,
				labelCount: 10,
				geometry,
				distribution: 'clustered',
			});
			const terms = makeTerms(dataset.labels);

			const baseline = benchmarkBuild(dataset.positions, terms, geometry, 'baseline', options);
			const candidate = benchmarkBuild(dataset.positions, terms, geometry, 'candidate', options);
			const layoutMs = benchmarkLayout(candidate.model.nodes);

			rows.push({
				geometry,
				points,
				baselineBuildMs: baseline.avgMs,
				candidateBuildMs: candidate.avgMs,
				buildSpeedup: baseline.avgMs > 0 ? baseline.avgMs / candidate.avgMs : 0,
				layoutMs,
				baselineLabels: baseline.model.nodes.length,
				candidateLabels: candidate.model.nodes.length,
			});
		}
	}

	console.table(
		rows.map((row) => ({
			geometry: row.geometry,
			points: row.points.toLocaleString(),
			'baseline build (ms)': row.baselineBuildMs.toFixed(2),
			'candidate build (ms)': row.candidateBuildMs.toFixed(2),
			'speedup x': row.buildSpeedup.toFixed(2),
			'layout (ms/frame)': row.layoutMs.toFixed(3),
			'baseline labels': row.baselineLabels,
			'candidate labels': row.candidateLabels,
		})),
	);
}

void main();