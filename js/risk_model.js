import fs from 'node:fs';
import os from 'node:os';

export const FEATURE_NAMES = ['bias', 'own_bumps', 'side_neighbour_bumps', 'corner_neighbour_bumps', 'edge'];
export const PRIOR_WEIGHTS = [-2.0, 1.0, 0.8, 0.4, 0.3];

const _SIDE_STEPS = [[0, 1], [0, -1], [1, 0], [-1, 0]];
const _CORNER_STEPS = [[1, 1], [1, -1], [-1, 1], [-1, -1]];

const sigmoid = (z) => {
  if (z >= 0) {
    return 1 / (1 + Math.exp(-z));
  }
  const e = Math.exp(z);
  return e / (1 + e);
};

export class RiskModel {
  static LEARNING_RATE = 0.15;
  static HIT_SAMPLE_WEIGHT = 4.0;
  static L2 = 0.001;
  static WEIGHT_LIMIT = 8.0;
  static SAVE_VERSION = 1;

  constructor(path = null) {
    this.path = path == null ? null : String(path);
    this.halfCells = 3;
    this.weights = [...PRIOR_WEIGHTS];
    this.samples = 0;
    this.hits = 0;
    this.bumpCounts = {};
    this._load();
  }

  startRun(halfCells) {
    this.halfCells = Number(halfCells);
    this.bumpCounts = {};
  }

  features(cell) {
    const counts = this.bumpCounts;
    const side = _SIDE_STEPS.reduce((sum, [dx, dy]) => sum + (counts[JSON.stringify([cell[0] + dx, cell[1] + dy])] ?? 0), 0);
    const corner = _CORNER_STEPS.reduce((sum, [dx, dy]) => sum + (counts[JSON.stringify([cell[0] + dx, cell[1] + dy])] ?? 0), 0);
    const onEdge = Math.max(Math.abs(cell[0]), Math.abs(cell[1])) >= this.halfCells ? 1.0 : 0.0;
    return [
      1.0,
      Math.min(counts[JSON.stringify(cell)] ?? 0, 3) / 3.0,
      Math.min(side, 4) / 4.0,
      Math.min(corner, 4) / 4.0,
      onEdge,
    ];
  }

  predict(cell) {
    return this._predict(cell);
  }

  _predict(cell) {
    const x = this.features(cell);
    const score = this.weights.reduce((sum, weight, index) => sum + weight * x[index], 0);
    return sigmoid(score);
  }

  _learn(cell, hit) {
    const x = this.features(cell);
    const score = this.weights.reduce((sum, weight, index) => sum + weight * x[index], 0);
    const p = sigmoid(score);
    const error = (hit ? 1.0 : 0.0) - p;
    const sampleWeight = hit ? RiskModel.HIT_SAMPLE_WEIGHT : 1.0;
    const limit = RiskModel.WEIGHT_LIMIT;

    for (let k = 0; k < this.weights.length; k += 1) {
      const value = x[k];
      const step = RiskModel.LEARNING_RATE * sampleWeight * (error * value - RiskModel.L2 * this.weights[k]);
      this.weights[k] = Math.max(-limit, Math.min(limit, this.weights[k] + step));
    }
    this.samples += 1;
  }

  recordSafePass(cell) {
    this._learn(cell, false);
  }

  recordBump(cell) {
    this._learn(cell, true);
    this.hits += 1;
    const key = JSON.stringify(cell);
    this.bumpCounts[key] = (this.bumpCounts[key] ?? 0) + 1;
  }

  travelCost(cell, riskWeight = 10.0) {
    return 1.0 + riskWeight * this.predict(cell);
  }

  riskMap(halfCells) {
    const cells = [];
    for (let i = -halfCells; i <= halfCells; i += 1) {
      for (let j = -halfCells; j <= halfCells; j += 1) {
        cells.push([i, j, Number(Number(this._predict([i, j])).toFixed(4))]);
      }
    }
    return cells;
  }

  save() {
    if (this.path == null) return;
    const data = {
      version: RiskModel.SAVE_VERSION,
      features: [...FEATURE_NAMES],
      weights: [...this.weights],
      samples: this.samples,
      hits: this.hits,
    };

    const dir = this.path.includes('/') ? this.path.slice(0, this.path.lastIndexOf('/')) || '.' : '.';
    try {
      fs.mkdirSync(dir, { recursive: true });
      const tempPath = `${this.path}.tmp-${process.pid}-${Date.now()}`;
      fs.writeFileSync(tempPath, JSON.stringify(data, null, 2), 'utf8');
      fs.renameSync(tempPath, this.path);
    } catch {
      // ignore write failures while preserving in-memory learning
    }
  }

  _load() {
    if (this.path == null || !fs.existsSync(this.path)) return;
    try {
      const data = JSON.parse(fs.readFileSync(this.path, 'utf8'));
      const weights = Array.isArray(data.weights) ? data.weights.map(Number) : [];
      if (JSON.stringify(data.features ?? []) !== JSON.stringify(FEATURE_NAMES) || weights.length !== PRIOR_WEIGHTS.length) {
        return;
      }
      if (!weights.every(Number.isFinite)) {
        return;
      }
      this.weights = weights;
      this.samples = Number(data.samples ?? 0);
      this.hits = Number(data.hits ?? 0);
    } catch {
      this.weights = [...PRIOR_WEIGHTS];
      this.samples = 0;
      this.hits = 0;
    }
  }
}
