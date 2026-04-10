import type { GeometryMode } from '../core/types.js';

export type SemanticLabelEngine = 'baseline' | 'candidate';
export type SemanticLabelDisplayMode = 'auto' | 'coarse' | 'fine';

export interface SemanticLabelBounds {
	xMin: number;
	yMin: number;
	xMax: number;
	yMax: number;
	width: number;
	height: number;
}

export interface SemanticLabelNode {
	key: string;
	level: 0 | 1;
	x: number;
	y: number;
	count: number;
	priority: number;
	dominance: number;
	primaryText: string;
	secondaryText: string | null;
	zoomMin: number;
	zoomMax: number;
}

export interface SemanticLabelModel {
	engine: SemanticLabelEngine;
	geometry: GeometryMode;
	pointCount: number;
	buildMs: number;
	bounds: SemanticLabelBounds;
	nodes: SemanticLabelNode[];
}

export interface SemanticLabelLevelConfig {
	level: 0 | 1;
	gridSize: number;
	blurRadius: number;
	searchRadiusCells: number;
	maxNodes: number;
	minPoints: number;
	thresholdRatio: number;
	minSeparationCells: number;
	zoomMin: number;
	zoomMax: number;
}

export interface BuildSemanticLabelModelOptions {
	positions: Float32Array;
	terms: readonly string[];
	geometry: GeometryMode;
	engine?: SemanticLabelEngine;
	bounds?: Partial<SemanticLabelBounds>;
	coarse?: Partial<SemanticLabelLevelConfig>;
	fine?: Partial<SemanticLabelLevelConfig>;
}

export interface VisibleSemanticLabel {
	key: string;
	level: 0 | 1;
	x: number;
	y: number;
	width: number;
	height: number;
	fontSize: number;
	secondaryFontSize: number;
	lineGap: number;
	opacity: number;
	primaryText: string;
	secondaryText: string | null;
	priority: number;
}

export interface LayoutSemanticLabelsOptions {
	nodes: readonly SemanticLabelNode[];
	zoom: number;
	viewportWidth: number;
	viewportHeight: number;
	displayMode?: SemanticLabelDisplayMode;
	project: (x: number, y: number) => { x: number; y: number };
	measureText?: (text: string, font: string) => number;
	fontFamily?: string;
	maxVisible?: number;
	mobile?: boolean;
}

export interface DrawSemanticLabelsOptions {
	fontFamily?: string;
	fillStyle?: string;
	strokeStyle?: string;
	shadowColor?: string;
}

interface GridPeak {
	cellX: number;
	cellY: number;
	cellIndex: number;
	score: number;
}

interface ClusterSummary {
	x: number;
	y: number;
	count: number;
	dominance: number;
	primaryText: string;
	secondaryText: string | null;
	priority: number;
}

interface CellBuckets {
	counts: Float32Array;
	members: number[][];
	cellWidth: number;
	cellHeight: number;
	smoothed: Float32Array;
}

const DEFAULT_COARSE_LEVEL: SemanticLabelLevelConfig = {
	level: 0,
	gridSize: 24,
	blurRadius: 2,
	searchRadiusCells: 2,
	maxNodes: 14,
	minPoints: 6,
	thresholdRatio: 0.1,
	minSeparationCells: 3,
	zoomMin: 0.55,
	zoomMax: 2.6,
};

const DEFAULT_FINE_LEVEL: SemanticLabelLevelConfig = {
	level: 1,
	gridSize: 42,
	blurRadius: 1,
	searchRadiusCells: 1,
	maxNodes: 36,
	minPoints: 4,
	thresholdRatio: 0.04,
	minSeparationCells: 2,
	zoomMin: 1.35,
	zoomMax: 9,
};

const VIEWPORT_MARGIN_PX = 18;
const LABEL_PADDING_PX = 8;
const LABEL_VERTICAL_PADDING_PX = 6;
const MAX_AUTO_DESKTOP_LABELS = 22;
const MAX_AUTO_MOBILE_LABELS = 12;
const COARSE_LEVELS = new Set([0]);
const FINE_LEVELS = new Set([0, 1]);

function clamp(value: number, min: number, max: number): number {
	if (value < min) return min;
	if (value > max) return max;
	return value;
}

function mergeLevelConfig(
	defaults: SemanticLabelLevelConfig,
	overrides: Partial<SemanticLabelLevelConfig> | undefined,
	pointCount: number,
): SemanticLabelLevelConfig {
	const minPointsFloor = defaults.level === 0 ? 6 : 4;
	const adaptiveMinPoints = Math.max(minPointsFloor, Math.round(pointCount * (defaults.level === 0 ? 0.01 : 0.004)));
	return {
		...defaults,
		...overrides,
		minPoints: overrides?.minPoints ?? adaptiveMinPoints,
	};
}

function normalizeTerm(term: string | null | undefined): string {
	if (!term) return '';
	const trimmed = term
		.replace(/[_-]+/g, ' ')
		.replace(/\s+/g, ' ')
		.trim();
	if (!trimmed || trimmed === 'undefined' || trimmed === 'null') return '';
	return trimmed;
}

function resolveBounds(
	positions: Float32Array,
	geometry: GeometryMode,
	overrides: Partial<SemanticLabelBounds> | undefined,
): SemanticLabelBounds {
	if (geometry === 'poincare') {
		return {
			xMin: overrides?.xMin ?? -1,
			yMin: overrides?.yMin ?? -1,
			xMax: overrides?.xMax ?? 1,
			yMax: overrides?.yMax ?? 1,
			width: (overrides?.xMax ?? 1) - (overrides?.xMin ?? -1),
			height: (overrides?.yMax ?? 1) - (overrides?.yMin ?? -1),
		};
	}

	let xMin = Number.POSITIVE_INFINITY;
	let yMin = Number.POSITIVE_INFINITY;
	let xMax = Number.NEGATIVE_INFINITY;
	let yMax = Number.NEGATIVE_INFINITY;

	for (let i = 0; i < positions.length; i += 2) {
		const x = positions[i];
		const y = positions[i + 1];
		if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
		if (x < xMin) xMin = x;
		if (x > xMax) xMax = x;
		if (y < yMin) yMin = y;
		if (y > yMax) yMax = y;
	}

	if (!Number.isFinite(xMin) || !Number.isFinite(yMin) || !Number.isFinite(xMax) || !Number.isFinite(yMax)) {
		xMin = -1;
		yMin = -1;
		xMax = 1;
		yMax = 1;
	}

	const width = Math.max(1e-6, xMax - xMin);
	const height = Math.max(1e-6, yMax - yMin);
	const padX = width * 0.04;
	const padY = height * 0.04;
	const resolvedXMin = overrides?.xMin ?? (xMin - padX);
	const resolvedYMin = overrides?.yMin ?? (yMin - padY);
	const resolvedXMax = overrides?.xMax ?? (xMax + padX);
	const resolvedYMax = overrides?.yMax ?? (yMax + padY);

	return {
		xMin: resolvedXMin,
		yMin: resolvedYMin,
		xMax: resolvedXMax,
		yMax: resolvedYMax,
		width: Math.max(1e-6, resolvedXMax - resolvedXMin),
		height: Math.max(1e-6, resolvedYMax - resolvedYMin),
	};
	}

function buildBuckets(
	positions: Float32Array,
	bounds: SemanticLabelBounds,
	config: SemanticLabelLevelConfig,
): CellBuckets {
	const gridSize = config.gridSize;
	const counts = new Float32Array(gridSize * gridSize);
	const members = Array.from({ length: gridSize * gridSize }, () => [] as number[]);
	const cellWidth = bounds.width / gridSize;
	const cellHeight = bounds.height / gridSize;

	for (let pointIndex = 0; pointIndex < positions.length / 2; pointIndex++) {
		const x = positions[pointIndex * 2];
		const y = positions[pointIndex * 2 + 1];
		if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
		const cellX = clamp(Math.floor(((x - bounds.xMin) / bounds.width) * gridSize), 0, gridSize - 1);
		const cellY = clamp(Math.floor(((y - bounds.yMin) / bounds.height) * gridSize), 0, gridSize - 1);
		const cellIndex = cellY * gridSize + cellX;
		counts[cellIndex] += 1;
		members[cellIndex].push(pointIndex);
	}

	return {
		counts,
		members,
		cellWidth,
		cellHeight,
		smoothed: smoothGrid(counts, gridSize, config.blurRadius),
	};
}

function smoothGrid(values: Float32Array, gridSize: number, radius: number): Float32Array {
	if (radius <= 0) return values;
	const out = new Float32Array(values.length);
	for (let cellY = 0; cellY < gridSize; cellY++) {
		for (let cellX = 0; cellX < gridSize; cellX++) {
			let sum = 0;
			let count = 0;
			for (let dy = -radius; dy <= radius; dy++) {
				const sampleY = cellY + dy;
				if (sampleY < 0 || sampleY >= gridSize) continue;
				for (let dx = -radius; dx <= radius; dx++) {
					const sampleX = cellX + dx;
					if (sampleX < 0 || sampleX >= gridSize) continue;
					sum += values[sampleY * gridSize + sampleX];
					count += 1;
				}
			}
			out[cellY * gridSize + cellX] = count > 0 ? sum / count : 0;
		}
	}
	return out;
}

function findPeaks(
	buckets: CellBuckets,
	config: SemanticLabelLevelConfig,
): GridPeak[] {
	const peaks: GridPeak[] = [];
	const gridSize = config.gridSize;
	let maxScore = 0;
	for (let index = 0; index < buckets.smoothed.length; index++) {
		if (buckets.smoothed[index] > maxScore) maxScore = buckets.smoothed[index];
	}
	if (maxScore <= 0) return peaks;

	const threshold = maxScore * config.thresholdRatio;

	for (let cellY = 0; cellY < gridSize; cellY++) {
		for (let cellX = 0; cellX < gridSize; cellX++) {
			const cellIndex = cellY * gridSize + cellX;
			const score = buckets.smoothed[cellIndex];
			if (score < threshold || buckets.counts[cellIndex] === 0) continue;
			let isPeak = true;
			for (let dy = -1; dy <= 1 && isPeak; dy++) {
				const sampleY = cellY + dy;
				if (sampleY < 0 || sampleY >= gridSize) continue;
				for (let dx = -1; dx <= 1; dx++) {
					const sampleX = cellX + dx;
					if (sampleX < 0 || sampleX >= gridSize) continue;
					if (dx === 0 && dy === 0) continue;
					if (buckets.smoothed[sampleY * gridSize + sampleX] > score) {
						isPeak = false;
						break;
					}
				}
			}
			if (!isPeak) continue;
			peaks.push({ cellX, cellY, cellIndex, score });
		}
	}

	peaks.sort((left, right) => right.score - left.score);
	return peaks;
}

function cellCenterX(cellX: number, buckets: CellBuckets, bounds: SemanticLabelBounds): number {
	return bounds.xMin + (cellX + 0.5) * buckets.cellWidth;
}

function cellCenterY(cellY: number, buckets: CellBuckets, bounds: SemanticLabelBounds): number {
	return bounds.yMin + (cellY + 0.5) * buckets.cellHeight;
}

function summarizeTerms(termCounts: Map<string, number>): {
	primaryText: string;
	secondaryText: string | null;
	dominance: number;
	priorityMultiplier: number;
} | null {
	if (termCounts.size === 0) return null;
	const ranked = Array.from(termCounts.entries())
		.filter((entry) => entry[0].length > 0)
		.sort((left, right) => {
			if (right[1] !== left[1]) return right[1] - left[1];
			return left[0].localeCompare(right[0]);
		});
	if (ranked.length === 0) return null;

	const total = ranked.reduce((sum, entry) => sum + entry[1], 0);
	const [primaryText, primaryCount] = ranked[0];
	const secondary = ranked[1] ?? null;
	const dominance = total > 0 ? primaryCount / total : 1;
	const useSecondary =
		secondary !== null &&
		secondary[0] !== primaryText &&
		dominance < 0.72 &&
		primaryCount < secondary[1] * 2.2;

	return {
		primaryText,
		secondaryText: useSecondary ? secondary![0] : null,
		dominance,
		priorityMultiplier: useSecondary ? 0.92 : 1,
	};
	}

function buildClusterSummaryFromIndices(
	positions: Float32Array,
	terms: readonly string[],
	indices: readonly number[],
	centerX: number,
	centerY: number,
	radiusX: number,
	radiusY: number,
	minPoints: number,
): ClusterSummary | null {
	if (indices.length === 0) return null;
	const invRadiusX = 1 / Math.max(1e-6, radiusX);
	const invRadiusY = 1 / Math.max(1e-6, radiusY);
	let sumX = 0;
	let sumY = 0;
	let count = 0;
	const termCounts = new Map<string, number>();

	for (const pointIndex of indices) {
		const x = positions[pointIndex * 2];
		const y = positions[pointIndex * 2 + 1];
		const dx = (x - centerX) * invRadiusX;
		const dy = (y - centerY) * invRadiusY;
		if ((dx * dx) + (dy * dy) > 1) continue;
		sumX += x;
		sumY += y;
		count += 1;
		const term = normalizeTerm(terms[pointIndex]);
		if (term) {
			termCounts.set(term, (termCounts.get(term) ?? 0) + 1);
		}
	}

	if (count < minPoints) return null;
	const summary = summarizeTerms(termCounts);
	if (!summary) return null;

	return {
		x: sumX / count,
		y: sumY / count,
		count,
		dominance: summary.dominance,
		primaryText: summary.primaryText,
		secondaryText: summary.secondaryText,
		priority: count * (0.6 + summary.dominance) * summary.priorityMultiplier,
	};
	}

function buildClusterSummaryBaseline(
	positions: Float32Array,
	terms: readonly string[],
	buckets: CellBuckets,
	bounds: SemanticLabelBounds,
	peak: GridPeak,
	config: SemanticLabelLevelConfig,
): ClusterSummary | null {
	const centerX = cellCenterX(peak.cellX, buckets, bounds);
	const centerY = cellCenterY(peak.cellY, buckets, bounds);
	const radiusX = buckets.cellWidth * (config.searchRadiusCells + 0.6);
	const radiusY = buckets.cellHeight * (config.searchRadiusCells + 0.6);
	const allIndices: number[] = [];
	for (let pointIndex = 0; pointIndex < positions.length / 2; pointIndex++) {
		allIndices.push(pointIndex);
	}
	return buildClusterSummaryFromIndices(
		positions,
		terms,
		allIndices,
		centerX,
		centerY,
		radiusX,
		radiusY,
		config.minPoints,
	);
}

function buildClusterSummaryCandidate(
	positions: Float32Array,
	terms: readonly string[],
	buckets: CellBuckets,
	bounds: SemanticLabelBounds,
	peak: GridPeak,
	config: SemanticLabelLevelConfig,
): ClusterSummary | null {
	const centerX = cellCenterX(peak.cellX, buckets, bounds);
	const centerY = cellCenterY(peak.cellY, buckets, bounds);
	const radiusX = buckets.cellWidth * (config.searchRadiusCells + 0.6);
	const radiusY = buckets.cellHeight * (config.searchRadiusCells + 0.6);
	const invRadiusX = 1 / Math.max(1e-6, radiusX);
	const invRadiusY = 1 / Math.max(1e-6, radiusY);
	let sumX = 0;
	let sumY = 0;
	let count = 0;
	const termCounts = new Map<string, number>();

	for (let dy = -config.searchRadiusCells; dy <= config.searchRadiusCells; dy++) {
		const cellY = peak.cellY + dy;
		if (cellY < 0 || cellY >= config.gridSize) continue;
		for (let dx = -config.searchRadiusCells; dx <= config.searchRadiusCells; dx++) {
			const cellX = peak.cellX + dx;
			if (cellX < 0 || cellX >= config.gridSize) continue;
			const members = buckets.members[cellY * config.gridSize + cellX];
			for (let memberIndex = 0; memberIndex < members.length; memberIndex++) {
				const pointIndex = members[memberIndex];
				const x = positions[pointIndex * 2];
				const y = positions[pointIndex * 2 + 1];
				const ddx = (x - centerX) * invRadiusX;
				const ddy = (y - centerY) * invRadiusY;
				if ((ddx * ddx) + (ddy * ddy) > 1) continue;
				sumX += x;
				sumY += y;
				count += 1;
				const term = normalizeTerm(terms[pointIndex]);
				if (term) {
					termCounts.set(term, (termCounts.get(term) ?? 0) + 1);
				}
			}
		}
	}

	if (count < config.minPoints) return null;
	const summary = summarizeTerms(termCounts);
	if (!summary) return null;

	return {
		x: sumX / count,
		y: sumY / count,
		count,
		dominance: summary.dominance,
		primaryText: summary.primaryText,
		secondaryText: summary.secondaryText,
		priority: count * (0.6 + summary.dominance) * summary.priorityMultiplier,
	};
}

function buildLevelNodes(
	positions: Float32Array,
	terms: readonly string[],
	bounds: SemanticLabelBounds,
	config: SemanticLabelLevelConfig,
	engine: SemanticLabelEngine,
): SemanticLabelNode[] {
	const buckets = buildBuckets(positions, bounds, config);
	const peaks = findPeaks(buckets, config);
	const nodes: SemanticLabelNode[] = [];
	for (const peak of peaks) {
		let tooClose = false;
		for (const existing of nodes) {
			const existingCellX = ((existing.x - bounds.xMin) / bounds.width) * config.gridSize;
			const existingCellY = ((existing.y - bounds.yMin) / bounds.height) * config.gridSize;
			if (
				Math.abs(existingCellX - peak.cellX) <= config.minSeparationCells &&
				Math.abs(existingCellY - peak.cellY) <= config.minSeparationCells
			) {
				tooClose = true;
				break;
			}
		}
		if (tooClose) continue;

		const summary = engine === 'candidate'
			? buildClusterSummaryCandidate(positions, terms, buckets, bounds, peak, config)
			: buildClusterSummaryBaseline(positions, terms, buckets, bounds, peak, config);
		if (!summary) continue;

		nodes.push({
			key: `${config.level}:${summary.primaryText}:${Math.round(summary.x * 1000)}:${Math.round(summary.y * 1000)}`,
			level: config.level,
			x: summary.x,
			y: summary.y,
			count: summary.count,
			priority: summary.priority,
			dominance: summary.dominance,
			primaryText: summary.primaryText,
			secondaryText: summary.secondaryText,
			zoomMin: config.zoomMin,
			zoomMax: config.zoomMax,
		});

		if (nodes.length >= config.maxNodes) break;
	}

	nodes.sort((left, right) => right.priority - left.priority);
	return nodes;
	}

export function buildSemanticLabelModel(options: BuildSemanticLabelModelOptions): SemanticLabelModel {
	const startedAt = performance.now();
	const engine = options.engine ?? 'candidate';
	const bounds = resolveBounds(options.positions, options.geometry, options.bounds);
	const coarse = mergeLevelConfig(DEFAULT_COARSE_LEVEL, options.coarse, options.positions.length / 2);
	const fine = mergeLevelConfig(DEFAULT_FINE_LEVEL, options.fine, options.positions.length / 2);
	const coarseNodes = buildLevelNodes(options.positions, options.terms, bounds, coarse, engine);
	const fineNodes = buildLevelNodes(options.positions, options.terms, bounds, fine, engine);

	return {
		engine,
		geometry: options.geometry,
		pointCount: options.positions.length / 2,
		buildMs: performance.now() - startedAt,
		bounds,
		nodes: [...coarseNodes, ...fineNodes],
	};
	}

function resolveActiveLevels(displayMode: SemanticLabelDisplayMode, zoom: number): Set<number> {
	if (displayMode === 'coarse') return COARSE_LEVELS;
	if (displayMode === 'fine') return FINE_LEVELS;
	if (zoom < 1.35) return COARSE_LEVELS;
	return FINE_LEVELS;
	}

function approximateTextWidth(text: string, fontSize: number): number {
	return Math.max(fontSize * 0.66, text.length * fontSize * 0.58);
	}

function measureLabel(
	primaryText: string,
	secondaryText: string | null,
	fontSize: number,
	secondaryFontSize: number,
	lineGap: number,
	fontFamily: string,
	measureText: ((text: string, font: string) => number) | undefined,
): { width: number; height: number } {
	const primaryFont = `600 ${fontSize}px ${fontFamily}`;
	const primaryWidth = measureText ? measureText(primaryText, primaryFont) : approximateTextWidth(primaryText, fontSize);
	if (!secondaryText) {
		return {
			width: primaryWidth,
			height: fontSize,
		};
	}
	const secondaryFont = `500 ${secondaryFontSize}px ${fontFamily}`;
	const secondaryWidth = measureText ? measureText(secondaryText, secondaryFont) : approximateTextWidth(secondaryText, secondaryFontSize);
	return {
		width: Math.max(primaryWidth, secondaryWidth),
		height: fontSize + lineGap + secondaryFontSize,
	};
	}

function overlaps(left: VisibleSemanticLabel, right: VisibleSemanticLabel): boolean {
	return !(
		left.x + (left.width / 2) < right.x - (right.width / 2) ||
		left.x - (left.width / 2) > right.x + (right.width / 2) ||
		left.y + (left.height / 2) < right.y - (right.height / 2) ||
		left.y - (left.height / 2) > right.y + (right.height / 2)
	);
	}

export function layoutSemanticLabels(options: LayoutSemanticLabelsOptions): VisibleSemanticLabel[] {
	const {
		nodes,
		zoom,
		viewportWidth,
		viewportHeight,
		displayMode = 'auto',
		project,
		measureText,
		fontFamily = 'ui-sans-serif, "Helvetica Neue", Helvetica, Arial, sans-serif',
		maxVisible,
		mobile = viewportWidth < 720,
	} = options;

	if (nodes.length === 0 || viewportWidth <= 0 || viewportHeight <= 0) return [];

	const activeLevels = resolveActiveLevels(displayMode, zoom);
	const visible: VisibleSemanticLabel[] = [];
	const maxLabels = maxVisible ?? (mobile ? MAX_AUTO_MOBILE_LABELS : MAX_AUTO_DESKTOP_LABELS);

	for (const node of nodes) {
		if (!activeLevels.has(node.level)) continue;
		if (zoom < node.zoomMin || zoom > node.zoomMax) continue;
		const projected = project(node.x, node.y);
		if (
			projected.x < -VIEWPORT_MARGIN_PX ||
			projected.x > viewportWidth + VIEWPORT_MARGIN_PX ||
			projected.y < -VIEWPORT_MARGIN_PX ||
			projected.y > viewportHeight + VIEWPORT_MARGIN_PX
		) {
			continue;
		}

		const zoomT = clamp((zoom - node.zoomMin) / Math.max(0.2, node.zoomMax - node.zoomMin), 0, 1);
		const fontSize = clamp(node.level === 0 ? 18 - (zoomT * 2.5) : 15 + (zoomT * 2), mobile ? 10 : 12, mobile ? 16 : 20);
		const secondaryFontSize = Math.max(10, fontSize * 0.82);
		const lineGap = node.secondaryText ? Math.max(2, fontSize * 0.16) : 0;
		const measured = measureLabel(
			node.primaryText,
			node.secondaryText,
			fontSize,
			secondaryFontSize,
			lineGap,
			fontFamily,
			measureText,
		);
		const label: VisibleSemanticLabel = {
			key: node.key,
			level: node.level,
			x: projected.x,
			y: projected.y,
			width: measured.width + (LABEL_PADDING_PX * 2),
			height: measured.height + (LABEL_VERTICAL_PADDING_PX * 2),
			fontSize,
			secondaryFontSize,
			lineGap,
			opacity: clamp(node.level === 0 ? 0.98 - (zoomT * 0.08) : 0.88 + (zoomT * 0.08), 0.75, 0.98),
			primaryText: node.primaryText,
			secondaryText: node.secondaryText,
			priority: node.priority,
		};

		let collides = false;
		for (const placed of visible) {
			if (overlaps(label, placed)) {
				collides = true;
				break;
			}
		}
		if (collides) continue;
		visible.push(label);
		if (visible.length >= maxLabels) break;
	}

	return visible;
	}

export function drawSemanticLabels(
	ctx: CanvasRenderingContext2D,
	labels: readonly VisibleSemanticLabel[],
	options: DrawSemanticLabelsOptions = {},
): void {
	const fontFamily = options.fontFamily ?? 'ui-sans-serif, "Helvetica Neue", Helvetica, Arial, sans-serif';
	const fillStyle = options.fillStyle ?? 'rgba(244, 247, 251, 0.95)';
	const strokeStyle = options.strokeStyle ?? 'rgba(9, 12, 18, 0.88)';
	const shadowColor = options.shadowColor ?? 'rgba(9, 12, 18, 0.55)';

	ctx.save();
	ctx.textAlign = 'center';
	ctx.textBaseline = 'middle';
	ctx.lineJoin = 'round';
	ctx.shadowColor = shadowColor;
	ctx.shadowBlur = 8;

	for (const label of labels) {
		ctx.globalAlpha = label.opacity;
		ctx.font = `600 ${label.fontSize}px ${fontFamily}`;
		ctx.lineWidth = Math.max(3, label.fontSize * 0.34);
		ctx.strokeStyle = strokeStyle;
		ctx.fillStyle = fillStyle;
		const primaryY = label.secondaryText
			? label.y - ((label.secondaryFontSize + label.lineGap) * 0.38)
			: label.y;
		ctx.strokeText(label.primaryText, label.x, primaryY);
		ctx.fillText(label.primaryText, label.x, primaryY);

		if (label.secondaryText) {
			ctx.font = `500 ${label.secondaryFontSize}px ${fontFamily}`;
			ctx.lineWidth = Math.max(2.4, label.secondaryFontSize * 0.32);
			const secondaryY = primaryY + label.fontSize * 0.65 + label.lineGap;
			ctx.strokeText(label.secondaryText, label.x, secondaryY);
			ctx.fillText(label.secondaryText, label.x, secondaryY);
		}
	}

	ctx.restore();
	}