import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

type Bar = { t: number; c: number };
type Task = { universe: { symbol: string; file: string }[]; places: number; bin_edges: number[]; window: { days: { date: string }[] }; output: string };

const input = process.env.MAKEFUN_INPUT_DIR ?? "/input";
const output = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";
function parseTask(): Task {
  const path = join(input, "task.json");
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch {
    throw new Error(`Missing input file: ${path}`);
  }
  let task: Task;
  try {
    const parsed: unknown = JSON.parse(text);
    if (parsed === null || typeof parsed !== "object") throw new Error("top-level value must be an object");
    task = parsed as Task;
  } catch (error) {
    throw new Error(`Malformed JSON in input file ${path}: ${error instanceof Error ? error.message : String(error)}`);
  }
  if (!Array.isArray(task.universe) || task.universe.length === 0) throw new Error(`Invalid task.json: universe must be a non-empty array`);
  if (!Array.isArray(task.window?.days) || task.window.days.length === 0) throw new Error(`Invalid task.json: window.days must be a non-empty array`);
  if (!Number.isInteger(task.places) || task.places < 1 || task.places > task.universe.length) throw new Error(`Invalid task.json: places must be between 1 and universe size`);
  if (!Array.isArray(task.bin_edges) || task.bin_edges.some((edge) => !Number.isFinite(edge))) throw new Error(`Invalid task.json: bin_edges must contain finite numbers`);
  if (typeof task.output !== "string" || task.output.length === 0) throw new Error(`Invalid task.json: output must be a non-empty filename`);
  return task;
}

const task = parseTask();
const symbols = task.universe.map((u) => u.symbol);
const places = task.places;
const edges = task.bin_edges;
const SQRT_TWO = 1.4142135623730951;
const MIN_SIGMA = 0.005;
const SIZE_MEAN_SHRINK = 0.75;
const VOLATILITY_SCALE = 0.35;

function finiteSizeMean(mu: number): number { const shrunk = mu * SIZE_MEAN_SHRINK; return Number.isFinite(shrunk) ? shrunk : 0; }
const RANK_SEED = 0x9e3779b9;

function readBars(file: string): Bar[] {
  let text: string;
  try {
    text = readFileSync(join(input, file), "utf8");
  } catch (error) {
    throw new Error(`Missing or unreadable input file ${join(input, file)}: ${error instanceof Error ? error.message : String(error)}`);
  }
  const parsed = text.split(/\r?\n/).flatMap((line, index) => {
    if (!line.trim()) return [];
    try {
      const raw = JSON.parse(line) as { t?: unknown; c?: unknown };
      const t = Number(raw.t);
      const c = Number(raw.c);
      if (!Number.isFinite(t) || !Number.isFinite(c) || c <= 0) {
        throw new Error(`Invalid price or timestamp in ${join(input, file)} at line ${index + 1}: t must be finite and c must be positive`);
      }
      return [{ bar: { t, c }, index }];
    } catch (error) {
      if (error instanceof Error && error.message.startsWith("Invalid price")) throw error;
      throw new Error(`Malformed JSON in ${join(input, file)} at line ${index + 1}: ${error instanceof Error ? error.message : String(error)}`);
    }
  });
  return parsed.sort((a, b) => a.bar.t - b.bar.t || a.index - b.index).map(({ bar }) => bar);
}

function finiteReturns(prices: number[]): number[] {
  const returns: number[] = [];
  let previous: number | undefined;
  for (const price of prices) {
    if (!Number.isFinite(price) || price <= 0) {
      previous = undefined;
      continue;
    }
    if (previous !== undefined) {
      const value = Math.log(price) - Math.log(previous);
      if (Number.isFinite(value)) returns.push(value);
    }
    previous = price;
  }
  return returns;
}

function moments(values: number[]): { mean: number; variance: number } {
  let count = 0;
  let mean = 0;
  let m2 = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    count++;
    const delta = value - mean;
    mean += delta / count;
    m2 += delta * (value - mean);
  }
  const variance = count > 1 ? m2 / (count - 1) : 0;
  return {
    mean: Number.isFinite(mean) ? mean : 0,
    variance: Number.isFinite(variance) && variance >= 0 ? variance : 0,
  };
}

function historicalMoments(prices: number[]): { mean: number; variance: number; returns: number[] } {
  const returns = finiteReturns(prices);
  const summary = moments(returns.slice(-480));
  return { ...summary, returns };
}

function trailingMean(returns: number[], horizon: number): number {
  const recent = returns.slice(-Math.min(horizon, returns.length));
  return recent.length === 0 ? 0 : moments(recent).mean;
}

function shrunkDrift(returns: number[], fallback: number): number {
  const short = trailingMean(returns, 3);
  const medium = trailingMean(returns, 8);
  const long = trailingMean(returns, 480);
  const blended = -0.60 * short - 0.25 * medium + 0.15 * long;
  const observations = Math.min(returns.length, 480);
  const shrink = observations / (observations + 60);
  const signal = blended * shrink;
  return Number.isFinite(signal) ? signal : fallback;
}

function trailingVariance(returns: number[], horizon: number, fallback: number): number {
  const recent = returns.slice(-Math.min(horizon, returns.length));
  if (recent.length < 2) return fallback;
  const variance = moments(recent).variance;
  return Number.isFinite(variance) && variance >= 0 ? variance : fallback;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function robustVariance(returns: number[], horizon: number, fallback: number): number {
  const recent = returns.slice(-Math.min(horizon, returns.length));
  if (recent.length < 2) return fallback;
  const center = median(recent);
  const mad = median(recent.map((value) => Math.abs(value - center)));
  const variance = (1.4826 * mad) ** 2;
  return Number.isFinite(variance) && variance >= 0 ? variance : fallback;
}

function volatilityEstimate(returns: number[], fallback: number): number {
  const base = Number.isFinite(fallback) && fallback >= 0 ? fallback : 0;
  const variance = 0.35 * trailingVariance(returns, 3, base)
    + 0.25 * trailingVariance(returns, 8, base)
    + 0.20 * trailingVariance(returns, 480, base)
    + 0.20 * robustVariance(returns, 480, base);
  const annualized = Math.sqrt(Math.max(0, Number.isFinite(variance) ? variance : base) * 24) * VOLATILITY_SCALE;
  return Number.isFinite(annualized) ? Math.max(MIN_SIGMA, annualized) : MIN_SIGMA;
}

function clamp(x: number, lo: number, hi: number): number { return Math.max(lo, Math.min(hi, x)); }
function normalCdf(x: number): number {
  if (Number.isNaN(x)) return 0.5;
  if (x === Infinity) return 1;
  if (x === -Infinity) return 0;
  const sign = x < 0 ? -1 : 1;
  const z = Math.abs(x) / SQRT_TWO;
  const t = 1 / (1 + 0.3275911 * z);
  const poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  return clamp(0.5 * (1 + sign * (1 - poly * Math.exp(-z * z))), 0, 1);
}

function normalizeProbabilities(values: number[]): number[] {
  const clean = values.map((value) => Number.isFinite(value) && value >= 0 ? value : 0);
  const total = clean.reduce((sum, value) => sum + value, 0);
  if (!Number.isFinite(total) || total <= 0) return Array.from({ length: values.length }, () => 1 / values.length);
  return clean.map((value) => value / total);
}

function binProbabilities(mu: number, sigma: number): number[] {
  const cuts = [-Infinity, ...edges, Infinity];
  const safeMu = Number.isFinite(mu) ? mu : 0;
  const safeSigma = Number.isFinite(sigma) && sigma > 0 ? sigma : 0.005;
  const raw = cuts.slice(0, -1).map((lo, i) => {
    const upper = normalCdf((cuts[i + 1] - safeMu) / safeSigma);
    const lower = normalCdf((lo - safeMu) / safeSigma);
    return Math.max(0, upper - lower);
  });
  return normalizeProbabilities(raw);
}

function rankDistribution(scores: number[][], places: number): number[][] {
  const symbolCount = scores.length;
  const counts = Array.from({ length: symbolCount }, () => Array(places).fill(0));
  const pairDraws = 6000;
  let seed = RANK_SEED;
  const rand = () => {
    seed = (Math.imul(seed ^ (seed >>> 16), 2246822519) + 3266489917) >>> 0;
    return seed / 4294967296;
  };
  const recordRanks = (sign: number, z: number[]) => {
    const sample = scores.map(([mu, sd], index) => mu + sd * sign * z[index]);
    const ranked = sample.map((value, index) => [value, index] as const)
      .sort((left, right) => right[0] - left[0] || left[1] - right[1])
      .slice(0, places);
    ranked.forEach(([, index], place) => { counts[index][place] += 1; });
  };
  for (let draw = 0; draw < pairDraws; draw++) {
    const z = scores.map(() => {
      const u = Math.max(rand(), 1e-12);
      const v = rand();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    });
    recordRanks(1, z);
    recordRanks(-1, z);
  }
  return counts.map((row) => {
    const total = row.reduce((sum, value) => sum + value, 0);
    if (total <= 0 || !Number.isFinite(total)) return row.map(() => 1 / places);
    return row.map((value) => {
      const probability = value / total;
      return Number.isFinite(probability) && probability >= 0 ? probability : 0;
    });
  });
}

const forecasts = symbols.map((symbol, index) => {
  const bars = readBars(task.universe[index].file);
  const prices = bars.map((bar) => bar.c);
  const { mean, variance, returns } = historicalMoments(prices);
  const drift = shrunkDrift(returns, mean);
  const sigma = volatilityEstimate(returns, variance);
  return { symbol, mu: clamp(drift * 24, -0.03, 0.03), sigma };
});

const rank = rankDistribution(forecasts.map((forecast) => [forecast.mu, forecast.sigma]), places);
const days = task.window.days.map((windowDay) => {
  const ordered = [...forecasts]
    .sort((left, right) => right.mu - left.mu || left.symbol.localeCompare(right.symbol))
    .slice(0, places);
  const order = ordered.map((forecast) => forecast.symbol);
  const probabilities = forecasts.flatMap((forecast, symbolIndex) =>
    rank[symbolIndex].map((probability, placeIndex) => ({
      symbol: forecast.symbol,
      place: placeIndex + 1,
      p: probability,
    })),
  );
  const sizes = forecasts.flatMap((forecast) =>
    binProbabilities(finiteSizeMean(forecast.mu), forecast.sigma).map((probability, binIndex) => ({
      symbol: forecast.symbol,
      bin: binIndex,
      p: probability,
    })),
  );
  return { date: windowDay.date, probabilities, order, sizes };
});

const outputPath = join(output, task.output);
try {
  mkdirSync(output, { recursive: true });
} catch (error) {
  throw new Error(`Output directory unavailable ${output}: ${error instanceof Error ? error.message : String(error)}`);
}
try {
  writeFileSync(outputPath, JSON.stringify({ days }));
} catch (error) {
  throw new Error(`Failed to write forecast file ${outputPath}: ${error instanceof Error ? error.message : String(error)}`);
}
